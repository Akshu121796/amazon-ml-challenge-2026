"""Candidate generation.
 
Strategy: an IDF-weighted, hashed sparse token index over the S2/S3 reference pool.
 
Every reference record is represented by a bag of blocking tokens drawn from its
business name and address (word tokens, digit runs and name+number compounds).
Tokens that occur in many reference records carry almost no discriminative power,
dominate the candidate lists and cost most of the query time, so they are dropped
(``max_doc_freq`` / ``max_doc_count``).  Candidates for a Source 1 record are the
reference records sharing at least one surviving token, ranked by cosine similarity
in that TF-IDF space and truncated to the ``top_k`` strongest.  Buckets (by country)
keep each index small.
 
Tokens are hashed rather than kept in a vocabulary: the name+number compounds alone
produce tens of millions of distinct tokens over the full pool, and the vocabulary
dictionary would cost more memory than the index itself.  Both the index build and
the querying stream in chunks so that peak memory stays a function of ``chunk_size``
rather than of the number of Source 1 records.
"""
 
from __future__ import annotations
 
import logging
from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence, Tuple
 
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
 
from .utils import LEGAL_SUFFIXES
 
LOGGER = logging.getLogger("ber.blocking")
 
CANDIDATE_COLUMNS = ["source1_entity_id", "candidate_entity_id", "block_score", "block_rank"]
 
 
@dataclass
class BlockingConfig:
    top_k: int = 30
    max_doc_freq: float = 0.002
    max_doc_count: int = 3000
    min_token_len: int = 2
    bucket_by_country: bool = True
    chunk_size: int = 2000
    index_chunk_size: int = 250_000
    use_name_number_tokens: bool = True
    n_features: int = 1 << 22
 
 
def blocking_tokens(
    name_norm: str,
    addr_norm: str,
    numbers: Sequence[str],
    use_name_number_tokens: bool = True,
    min_token_len: int = 2,
) -> List[str]:
    """Bag of blocking tokens for one record."""
    name_tokens = [t for t in name_norm.split() if len(t) >= min_token_len]
    core_tokens = [t for t in name_tokens if t not in LEGAL_SUFFIXES] or name_tokens
    addr_tokens = [t for t in addr_norm.split() if len(t) >= min_token_len]
 
    tokens = [f"n:{t}" for t in core_tokens]
    tokens += [f"a:{t}" for t in addr_tokens]
    tokens += [f"d:{d}" for d in numbers]
    if core_tokens:
        tokens.append("N:" + " ".join(sorted(set(core_tokens))))
    if use_name_number_tokens:
        tokens += [f"x:{t}|{d}" for t in core_tokens for d in numbers]
    return tokens
 
 
def _token_frame(df: pd.DataFrame, config: BlockingConfig) -> List[List[str]]:
    names = df["name_norm"].tolist()
    addresses = df["addr_norm"].tolist()
    numbers = df["addr_numbers"].tolist()
    return [
        blocking_tokens(
            name,
            address,
            number.split() if isinstance(number, str) else number,
            config.use_name_number_tokens,
            config.min_token_len,
        )
        for name, address, number in zip(names, addresses, numbers)
    ]
 
 
def _vectorizer(config: BlockingConfig) -> HashingVectorizer:
    return HashingVectorizer(
        analyzer=lambda tokens: tokens,
        lowercase=False,
        n_features=config.n_features,
        alternate_sign=False,
        binary=True,
        norm=None,
        dtype=np.float32,
    )
 
 
class BucketIndex:
    """IDF-weighted hashed index over one bucket of reference records."""
 
    def __init__(self, targets: pd.DataFrame, config: BlockingConfig) -> None:
        self.config = config
        self.vectorizer = _vectorizer(config)
        self.target_ids = targets["entity_id"].to_numpy()
 
        blocks = []
        for start in range(0, len(targets), config.index_chunk_size):
            chunk = targets.iloc[start : start + config.index_chunk_size]
            blocks.append(self.vectorizer.transform(_token_frame(chunk, config)))
        matrix = sparse.vstack(blocks, format="csr") if blocks else None
        del blocks
        if matrix is None or matrix.nnz == 0:
            self.idf = np.zeros(config.n_features, dtype=np.float32)
            self.matrix_t = sparse.csr_matrix((config.n_features, len(targets)), dtype=np.float32)
            return
 
        n_docs = matrix.shape[0]
        doc_freq = np.bincount(matrix.indices, minlength=config.n_features)
        cutoff = min(config.max_doc_count, max(int(config.max_doc_freq * n_docs), 1))
        self.idf = np.where(
            (doc_freq > 0) & (doc_freq <= cutoff),
            np.log(1.0 + n_docs / np.maximum(doc_freq, 1)),
            0.0,
        ).astype(np.float32)
        LOGGER.info(
            "index: %s docs, %s live token slots (df cutoff %s), %s postings",
            f"{n_docs:,}",
            f"{int((self.idf > 0).sum()):,}",
            f"{cutoff:,}",
            f"{matrix.nnz:,}",
        )
 
        self._weight(matrix)
        self.matrix_t = matrix.T.tocsr()
        del matrix
 
    def _weight(self, matrix: sparse.csr_matrix) -> None:
        matrix.data *= self.idf[matrix.indices]
        matrix.eliminate_zeros()
        normalize(matrix, norm="l2", copy=False)
 
    def query(self, queries: pd.DataFrame) -> sparse.csr_matrix:
        matrix = self.vectorizer.transform(_token_frame(queries, self.config))
        self._weight(matrix)
        return (matrix @ self.matrix_t).tocsr()
 
 
def _top_k_from_rows(
    scores: sparse.csr_matrix, top_k: int
) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    indptr, indices, data = scores.indptr, scores.indices, scores.data
    for local_row in range(scores.shape[0]):
        start, end = indptr[local_row], indptr[local_row + 1]
        if start == end:
            yield local_row, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
            continue
        row_idx = indices[start:end]
        row_val = data[start:end]
        if row_val.size > top_k:
            keep = np.argpartition(row_val, -top_k)[-top_k:]
            row_idx, row_val = row_idx[keep], row_val[keep]
        order = np.argsort(-row_val)
        yield local_row, row_idx[order], row_val[order]
 
 
def _candidate_frame(rows: List[Tuple[str, str, float, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
 
 
@dataclass
class CandidateChunk:
    """Candidate pairs for a batch of whole Source 1 records.
 
    ``s1`` and ``targets`` carry only the records the pairs refer to, so downstream
    feature building never has to index into the full reference pool.
    """
 
    pairs: pd.DataFrame
    s1: pd.DataFrame
    targets: pd.DataFrame
 
 
def iter_bucket_candidates(
    s1_bucket: pd.DataFrame,
    target_bucket: pd.DataFrame,
    config: BlockingConfig,
) -> Iterator[CandidateChunk]:
    """Candidate pairs for one bucket, streamed in ``chunk_size`` Source 1 batches."""
    index = BucketIndex(target_bucket, config)
    target_ids = index.target_ids
    s1_ids = s1_bucket["entity_id"].to_numpy()
 
    for start in range(0, len(s1_bucket), config.chunk_size):
        chunk = s1_bucket.iloc[start : start + config.chunk_size]
        scores = index.query(chunk)
        rows: List[Tuple[str, str, float, int]] = []
        positions: List[int] = []
        for local_row, cand_idx, cand_score in _top_k_from_rows(scores, config.top_k):
            s1_id = s1_ids[start + local_row]
            positions.extend(cand_idx.tolist())
            for rank, (idx, score) in enumerate(zip(cand_idx, cand_score)):
                rows.append((s1_id, target_ids[idx], float(score), rank))
        referenced = np.unique(np.asarray(positions, dtype=np.int64)) if positions else []
        yield CandidateChunk(
            pairs=_candidate_frame(rows),
            s1=chunk,
            targets=target_bucket.iloc[referenced],
        )
 
 
def _buckets(
    s1: pd.DataFrame, targets: pd.DataFrame, config: BlockingConfig
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    if not config.bucket_by_country:
        return [(s1, targets)]
    countries = sorted(set(s1["country_norm"]) & set(targets["country_norm"]))
    groups = [
        (s1[s1["country_norm"] == country], targets[targets["country_norm"] == country])
        for country in countries
    ]
    unmatched = s1[~s1["country_norm"].isin(countries)]
    if len(unmatched):
        groups.append((unmatched, targets))
    return groups
 
 
def iter_candidates(
    s1: pd.DataFrame,
    targets: pd.DataFrame,
    config: BlockingConfig | None = None,
) -> Iterator[CandidateChunk]:
    """Stream candidate pairs bucket by bucket, chunk by chunk.
 
    Each yielded frame holds whole Source 1 records, so downstream per-record
    aggregation (feature normalisation, the decision rule) is safe chunk-wise.
    """
    config = config or BlockingConfig()
    for s1_bucket, target_bucket in _buckets(s1, targets, config):
        if s1_bucket.empty or target_bucket.empty:
            continue
        LOGGER.info(
            "blocking bucket: %s S1 records against %s reference records",
            f"{len(s1_bucket):,}",
            f"{len(target_bucket):,}",
        )
        yield from iter_bucket_candidates(s1_bucket, target_bucket, config)
 
 
def generate_candidates(
    s1: pd.DataFrame,
    targets: pd.DataFrame,
    config: BlockingConfig | None = None,
) -> pd.DataFrame:
    """Materialise all candidate pairs (only for samples that fit in memory)."""
    chunks = [chunk.pairs for chunk in iter_candidates(s1, targets, config) if not chunk.pairs.empty]
    if not chunks:
        return _candidate_frame([])
    candidates = pd.concat(chunks, ignore_index=True)
    return candidates.sort_values(
        ["source1_entity_id", "block_score"], ascending=[True, False]
    ).reset_index(drop=True)
 
 
def candidate_recall(candidates: pd.DataFrame, truth: Dict[str, set]) -> Dict[str, float]:
    """Macro candidate recall plus candidate-volume diagnostics."""
    grouped = candidates.groupby("source1_entity_id")["candidate_entity_id"].apply(set)
    recalls, sizes = [], []
    for s1_id, true_set in truth.items():
        predicted = grouped.get(s1_id, set())
        sizes.append(len(predicted))
        recalls.append(len(true_set & predicted) / len(true_set) if true_set else 1.0)
    return {
        "macro_candidate_recall": float(np.mean(recalls)) if recalls else 0.0,
        "mean_candidates_per_s1": float(np.mean(sizes)) if sizes else 0.0,
        "median_candidates_per_s1": float(np.median(sizes)) if sizes else 0.0,
        "n_evaluated": len(recalls),
    }