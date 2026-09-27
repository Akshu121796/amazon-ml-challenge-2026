"""Tests for the submission validator."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "utils"))

from validate_submission import validate  # noqa: E402

COLUMNS = ["entity_id", "business_name", "business_address", "country"]


@pytest.fixture
def test_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "test"
    directory.mkdir()
    pd.DataFrame([("S1-1", "A", "addr", "US"), ("S1-2", "B", "addr", "US")], columns=COLUMNS).to_csv(
        directory / "test_source1.tsv", sep="\t", index=False
    )
    pd.DataFrame([("S2-1", "A", "addr", "US")], columns=COLUMNS).to_csv(
        directory / "test_source2.tsv", sep="\t", index=False
    )
    pd.DataFrame([("S3-1", "B", "addr", "US")], columns=COLUMNS).to_csv(
        directory / "test_source3.tsv", sep="\t", index=False
    )
    return directory


def _write(tmp_path: Path, matching_rows, candidate_rows) -> tuple[Path, Path]:
    matching = tmp_path / "matching_results.tsv"
    candidate = tmp_path / "candidate_pairs.tsv"
    pd.DataFrame(matching_rows, columns=["source1_entity_id", "matched_entity_ids"]).to_csv(
        matching, sep="\t", index=False
    )
    pd.DataFrame(candidate_rows, columns=["source1_entity_id", "candidate_entity_ids"]).to_csv(
        candidate, sep="\t", index=False
    )
    return matching, candidate


def test_valid_submission_passes(tmp_path: Path, test_dir: Path):
    matching, candidate = _write(
        tmp_path,
        [("S1-1", "S2-1"), ("S1-2", "S3-1")],
        [("S1-1", "S2-1,S3-1"), ("S1-2", "S3-1")],
    )
    assert validate(matching, candidate, test_dir) == []


def test_missing_source1_rows_are_reported(tmp_path: Path, test_dir: Path):
    matching, candidate = _write(tmp_path, [("S1-1", "S2-1")], [("S1-1", "S2-1")])
    problems = validate(matching, candidate, test_dir)
    assert any("absent" in problem for problem in problems)


def test_unknown_and_duplicate_ids_are_reported(tmp_path: Path, test_dir: Path):
    matching, candidate = _write(
        tmp_path,
        [("S1-1", "S2-999"), ("S1-2", "S3-1,S3-1")],
        [("S1-1", "S2-999"), ("S1-2", "S3-1")],
    )
    problems = validate(matching, candidate, test_dir)
    assert any("not in test source 2/3" in problem for problem in problems)
    assert any("repeat an id" in problem for problem in problems)


def test_predictions_outside_candidates_are_reported(tmp_path: Path, test_dir: Path):
    matching, candidate = _write(
        tmp_path,
        [("S1-1", "S2-1"), ("S1-2", "S3-1")],
        [("S1-1", ""), ("S1-2", "S3-1")],
    )
    problems = validate(matching, candidate, test_dir)
    assert any("absent from" in problem for problem in problems)
