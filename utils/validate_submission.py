"""Schema and content validation for the submission files.
 
Checks, for both ``matching_results.tsv`` and ``candidate_pairs.tsv``:
  * required columns are present and TSV-parseable;
  * exactly one row per Source 1 entity, covering the test Source 1 file;
  * every referenced entity id exists in test Source 2 or Source 3;
  * no duplicate ids inside a row;
  * predicted matches are a subset of the generated candidates.
 
Usage::
 
    python utils/validate_submission.py --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv --test-dir dataset/test
"""
 
from __future__ import annotations
 
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set
 
import pandas as pd
 
 
def _parse_ids(value: object) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]
 
 
def _read_tsv(path: Path, required_columns: Set[str], problems: List[str]) -> pd.DataFrame | None:
    if not path.exists():
        problems.append(f"missing file: {path}")
        return None
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)
    frame.columns = [c.strip() for c in frame.columns]
    missing = required_columns - set(frame.columns)
    if missing:
        problems.append(f"{path.name}: missing columns {sorted(missing)}")
        return None
    return frame
 
 
def _check_id_column(
    frame: pd.DataFrame, path: Path, expected_s1: Set[str], problems: List[str]
) -> None:
    ids = frame["source1_entity_id"].astype(str)
    duplicates = ids[ids.duplicated()].unique()
    if len(duplicates):
        problems.append(f"{path.name}: {len(duplicates)} duplicate source1_entity_id values")
    missing = expected_s1 - set(ids)
    extra = set(ids) - expected_s1
    if missing:
        problems.append(f"{path.name}: {len(missing)} test Source 1 ids are absent")
    if extra:
        problems.append(f"{path.name}: {len(extra)} ids are not in test_source1.tsv")
 
 
def _check_referenced_ids(
    frame: pd.DataFrame,
    column: str,
    path: Path,
    valid_targets: Set[str],
    problems: List[str],
) -> Dict[str, Set[str]]:
    parsed: Dict[str, Set[str]] = {}
    unknown, duplicated_rows = 0, 0
    for row in frame.itertuples():
        ids = _parse_ids(getattr(row, column))
        if len(ids) != len(set(ids)):
            duplicated_rows += 1
        unknown += len([i for i in ids if i not in valid_targets])
        parsed[str(row.source1_entity_id)] = set(ids)
    if unknown:
        problems.append(f"{path.name}: {unknown} referenced ids are not in test source 2/3")
    if duplicated_rows:
        problems.append(f"{path.name}: {duplicated_rows} rows repeat an id in {column}")
    return parsed
 
 
def validate(matching_path: Path, candidate_path: Path, test_dir: Path) -> List[str]:
    problems: List[str] = []
    s1 = pd.read_csv(test_dir / "test_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s2 = pd.read_csv(test_dir / "test_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    s3 = pd.read_csv(test_dir / "test_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
    expected_s1 = set(s1["entity_id"].astype(str))
    valid_targets = set(s2["entity_id"].astype(str)) | set(s3["entity_id"].astype(str))
 
    matching = _read_tsv(matching_path, {"source1_entity_id", "matched_entity_ids"}, problems)
    candidates = _read_tsv(candidate_path, {"source1_entity_id", "candidate_entity_ids"}, problems)
    if matching is None or candidates is None:
        return problems
 
    _check_id_column(matching, matching_path, expected_s1, problems)
    _check_id_column(candidates, candidate_path, expected_s1, problems)
 
    matched = _check_referenced_ids(
        matching, "matched_entity_ids", matching_path, valid_targets, problems
    )
    generated = _check_referenced_ids(
        candidates, "candidate_entity_ids", candidate_path, valid_targets, problems
    )
 
    not_in_candidates = sum(
        len(ids - generated.get(s1_id, set())) for s1_id, ids in matched.items()
    )
    if not_in_candidates:
        problems.append(
            f"{matching_path.name}: {not_in_candidates} predicted matches are absent from "
            f"{candidate_path.name}"
        )
 
    empty_predictions = sum(1 for ids in matched.values() if not ids)
    mean_matches = sum(map(len, matched.values())) / max(len(matched), 1)
    mean_candidates = sum(map(len, generated.values())) / max(len(generated), 1)
    print(f"Source 1 entities: {len(expected_s1):,}")
    print(f"Predicted matches per entity (mean): {mean_matches:.2f}")
    print(f"Entities predicted as no-match: {empty_predictions:,}")
    print(f"Candidates per entity (mean): {mean_candidates:.2f}")
    return problems
 
 
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--test-dir", required=True, type=Path)
    args = parser.parse_args()
 
    problems = validate(args.matching, args.candidate, args.test_dir)
    if problems:
        print("\nFAIL: submission validation found problems:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nPASS: submission validation successful.")
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())