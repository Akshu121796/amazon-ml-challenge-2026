"""Shared IO, normalization and text utilities."""
 
from __future__ import annotations
 
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set
 
import pandas as pd
 
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
 
LEGAL_SUFFIXES = {
    "inc", "llc", "llp", "ltd", "limited", "pvt", "private", "co", "corp",
    "corporation", "company", "gmbh", "bv", "nv", "sa", "srl", "spa", "ag",
    "plc", "pte", "sdn", "bhd", "kk", "oy", "ab", "as", "aps", "sl",
}
 
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"\d+")
 
try:  # Arrow strings keep the 10M-record reference pool within memory
    import pyarrow  # noqa: F401
 
    _TEXT_DTYPE = "string[pyarrow]"
except ImportError:  # pragma: no cover - fallback for minimal installs
    _TEXT_DTYPE = "object"
 
 
def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger("ber")
 
 
def normalize_text(value: object) -> str:
    """Unicode-normalize, lowercase, strip punctuation and collapse whitespace."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()
 
 
def strip_legal_suffixes(normalized_name: str) -> str:
    tokens = [t for t in normalized_name.split() if t not in LEGAL_SUFFIXES]
    return " ".join(tokens) if tokens else normalized_name
 
 
def extract_numbers(value: object) -> List[str]:
    """Digit runs in a field, e.g. street numbers, unit numbers, postcodes."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return _NUM_RE.findall(str(value))
 
 
def join_numbers(value: object) -> str:
    return " ".join(extract_numbers(value))
 
 
def parse_match_ids(value: object) -> Set[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return set()
    return {part.strip() for part in text.split(",") if part.strip()}
 
 
def format_match_ids(ids: Iterable[str]) -> str:
    return ",".join(sorted(ids))
 
 
def read_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    return df[SOURCE_COLUMNS]
 
 
def read_normalized_source(
    path: Path, keep_ids: Set[str] | None = None, chunk_rows: int = 500_000
) -> pd.DataFrame:
    """Read a source file straight into its normalized, Arrow-backed representation.
 
    Chunked so the raw Python-object columns of a 5M-row file never exist all at once.
    """
    frames = []
    reader = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        chunksize=chunk_rows,
    )
    for chunk in reader:
        chunk.columns = [c.strip() for c in chunk.columns]
        missing = [c for c in SOURCE_COLUMNS if c not in chunk.columns]
        if missing:
            raise ValueError(f"{path} is missing columns {missing}")
        if keep_ids is not None:
            chunk = chunk[chunk["entity_id"].isin(keep_ids)]
        if len(chunk):
            frames.append(add_normalized_columns(chunk))
    if not frames:
        return add_normalized_columns(pd.DataFrame(columns=SOURCE_COLUMNS))
    return pd.concat(frames, ignore_index=True)
 
 
def read_entity_ids(path: Path) -> List[str]:
    """Just the id column of a source file."""
    return pd.read_csv(
        path, sep="\t", usecols=["entity_id"], dtype=str, keep_default_na=False, na_filter=False
    )["entity_id"].tolist()
 
 
def read_ground_truth(path: Path, keep_ids: Set[str] | None = None) -> Dict[str, Set[str]]:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)
    df.columns = [c.strip() for c in df.columns]
    if keep_ids is not None:
        df = df[df["source1_entity_id"].isin(keep_ids)]
    return {
        str(row.source1_entity_id): parse_match_ids(row.matched_entity_ids)
        for row in df.itertuples()
    }
 
 
NORMALIZED_COLUMNS = ["name_norm", "name_core", "addr_norm", "country_norm", "addr_numbers"]
 
 
def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the normalized text columns, dropping the raw ones.
 
    Columns are stored as Arrow strings: at 10M reference records the Python-object
    representation of the same text costs several GB more.  ``addr_numbers`` holds the
    digit runs space-joined; :func:`build_record_lookup` splits them again for the
    (much smaller) subsets that features are computed over.
    """
    out = pd.DataFrame({"entity_id": df["entity_id"].to_numpy()})
    out["name_norm"] = df["business_name"].map(normalize_text).to_numpy()
    out["name_core"] = out["name_norm"].map(strip_legal_suffixes).to_numpy()
    out["addr_norm"] = df["business_address"].map(normalize_text).to_numpy()
    out["country_norm"] = df["country"].map(normalize_text).to_numpy()
    out["addr_numbers"] = df["business_address"].map(join_numbers).to_numpy()
    for column in NORMALIZED_COLUMNS:
        out[column] = out[column].astype(_TEXT_DTYPE)
    return out
 
 
def build_record_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    lookup: Dict[str, Dict[str, object]] = {}
    columns = [df[column].tolist() for column in NORMALIZED_COLUMNS]
    for position, entity_id in enumerate(df["entity_id"].tolist()):
        name_norm, name_core, addr_norm, country_norm, numbers = (
            column[position] for column in columns
        )
        lookup[entity_id] = {
            "name_norm": name_norm,
            "name_core": name_core,
            "addr_norm": addr_norm,
            "country_norm": country_norm,
            "addr_numbers": numbers.split(),
        }
    return lookup
 
 
def write_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=list)
 
 
def read_json(path: Path) -> object:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
 
 
def chunk_ranges(n_rows: int, chunk_size: int) -> Sequence[range]:
    return [range(start, min(start + chunk_size, n_rows)) for start in range(0, n_rows, chunk_size)]