"""Pairwise feature engineering for candidate pairs."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from rapidfuzz import distance, fuzz

FEATURE_COLUMNS = [
    "block_score",
    "block_rank",
    "block_score_ratio_to_best",
    "n_candidates",
    "name_ratio",
    "name_token_sort",
    "name_token_set",
    "name_partial",
    "name_jaro_winkler",
    "name_core_ratio",
    "name_core_exact",
    "name_token_jaccard",
    "name_token_containment",
    "addr_ratio",
    "addr_token_sort",
    "addr_token_set",
    "addr_token_jaccard",
    "addr_number_jaccard",
    "addr_first_number_match",
    "same_country",
    "name_len_ratio",
    "addr_len_ratio",
    "addr_missing",
    "is_source3",
]

_EMPTY_RECORD: Dict[str, object] = {
    "name_norm": "",
    "name_core": "",
    "addr_norm": "",
    "country_norm": "",
    "addr_numbers": [],
}


def _jaccard(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _containment(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def _len_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return min(len(left), len(right)) / max(len(left), len(right))


def build_features(
    candidates: pd.DataFrame,
    s1_lookup: Dict[str, Dict[str, object]],
    target_lookup: Dict[str, Dict[str, object]],
) -> pd.DataFrame:
    """Compute the pairwise feature matrix for the candidate pairs.

    ``candidates`` needs the columns produced by :func:`blocking.generate_candidates`.
    The returned frame keeps the pair identifiers alongside ``FEATURE_COLUMNS``.
    """
    if candidates.empty:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", *FEATURE_COLUMNS])

    grouped = candidates.groupby("source1_entity_id")["block_score"]
    best_score = grouped.transform("max").to_numpy()
    n_candidates = grouped.transform("size").to_numpy()

    rows: List[Dict[str, float]] = []
    for position, pair in enumerate(candidates.itertuples()):
        left = s1_lookup.get(pair.source1_entity_id, _EMPTY_RECORD)
        right = target_lookup.get(pair.candidate_entity_id, _EMPTY_RECORD)

        name_left, name_right = left["name_norm"], right["name_norm"]
        core_left, core_right = left["name_core"], right["name_core"]
        addr_left, addr_right = left["addr_norm"], right["addr_norm"]
        name_tokens_left, name_tokens_right = set(core_left.split()), set(core_right.split())
        addr_tokens_left, addr_tokens_right = set(addr_left.split()), set(addr_right.split())
        numbers_left, numbers_right = set(left["addr_numbers"]), set(right["addr_numbers"])

        rows.append(
            {
                "block_score": pair.block_score,
                "block_rank": float(pair.block_rank),
                "block_score_ratio_to_best": (
                    pair.block_score / best_score[position] if best_score[position] else 0.0
                ),
                "n_candidates": float(n_candidates[position]),
                "name_ratio": fuzz.ratio(name_left, name_right) / 100.0,
                "name_token_sort": fuzz.token_sort_ratio(name_left, name_right) / 100.0,
                "name_token_set": fuzz.token_set_ratio(name_left, name_right) / 100.0,
                "name_partial": fuzz.partial_ratio(name_left, name_right) / 100.0,
                "name_jaro_winkler": distance.JaroWinkler.similarity(name_left, name_right),
                "name_core_ratio": fuzz.ratio(core_left, core_right) / 100.0,
                "name_core_exact": float(bool(core_left) and core_left == core_right),
                "name_token_jaccard": _jaccard(name_tokens_left, name_tokens_right),
                "name_token_containment": _containment(name_tokens_left, name_tokens_right),
                "addr_ratio": fuzz.ratio(addr_left, addr_right) / 100.0,
                "addr_token_sort": fuzz.token_sort_ratio(addr_left, addr_right) / 100.0,
                "addr_token_set": fuzz.token_set_ratio(addr_left, addr_right) / 100.0,
                "addr_token_jaccard": _jaccard(addr_tokens_left, addr_tokens_right),
                "addr_number_jaccard": _jaccard(numbers_left, numbers_right),
                "addr_first_number_match": float(
                    bool(left["addr_numbers"])
                    and bool(right["addr_numbers"])
                    and left["addr_numbers"][0] == right["addr_numbers"][0]
                ),
                "same_country": float(left["country_norm"] == right["country_norm"]),
                "name_len_ratio": _len_ratio(name_left, name_right),
                "addr_len_ratio": _len_ratio(addr_left, addr_right),
                "addr_missing": float(not addr_left or not addr_right),
                "is_source3": float(str(pair.candidate_entity_id).startswith("S3-")),
            }
        )

    features = pd.DataFrame(rows, columns=FEATURE_COLUMNS).astype(np.float32)
    identifiers = candidates[["source1_entity_id", "candidate_entity_id"]].reset_index(drop=True)
    return pd.concat([identifiers, features], axis=1)
