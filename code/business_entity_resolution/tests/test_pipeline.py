"""Unit tests for the entity resolution pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "utils"))

from src.blocking import BlockingConfig, blocking_tokens, candidate_recall, generate_candidates
from src.evaluation import compute_f05_single, evaluate_f05_macro
from src.features import FEATURE_COLUMNS, build_features
from src.model import DecisionRule, _TuningView, apply_decision_rule, label_pairs
from src.utils import add_normalized_columns, extract_numbers, normalize_text, parse_match_ids


def _frame(rows):
    return add_normalized_columns(
        pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])
    )


@pytest.fixture
def sources():
    s1 = _frame(
        [
            ("S1-1", "Orelee's Barbershop", "1795 Westchester Drive, High Point, NC", "US"),
            ("S1-2", "Aditya Properties LLP", "G-3/571, Gulmohar Colony, Bhopal", "India"),
        ]
    )
    targets = _frame(
        [
            ("S2-1", "ORELEE BARBERSHOP INC", "1795 Westchester Dr, High Point, NC", "US"),
            ("S3-1", "Orelee Barber Shop", "1795 Westchester Drive, High Point", "US"),
            ("S2-2", "Aditya Properties Private Limited", "G-3/571 Gulmohar Colony, Bhopal", "India"),
            ("S2-3", "Unrelated Motors Ltd", "42 Station Road, Pune", "India"),
        ]
    )
    truth = {"S1-1": {"S2-1", "S3-1"}, "S1-2": {"S2-2"}}
    return s1, targets, truth


def test_normalize_text_handles_unicode_and_punctuation():
    assert normalize_text("Orelee's  Barbershop, Inc.") == "orelee s barbershop inc"
    assert normalize_text(None) == ""
    assert normalize_text("ＡＢＣ") == "abc"


def test_extract_numbers_and_parse_match_ids():
    assert extract_numbers("G-3/571, Bhopal 462001") == ["3", "571", "462001"]
    assert parse_match_ids(" S2-1, S3-2 ,") == {"S2-1", "S3-2"}
    assert parse_match_ids("") == set()


def test_blocking_tokens_drop_legal_suffixes_and_build_compounds():
    tokens = blocking_tokens("aditya properties llp", "g 3 571 bhopal", ["3", "571"])
    assert "n:aditya" in tokens and "n:llp" not in tokens
    assert "d:571" in tokens
    assert "x:aditya|571" in tokens


def test_generate_candidates_recovers_true_matches(sources):
    s1, targets, truth = sources
    candidates = generate_candidates(s1, targets, BlockingConfig(top_k=10, max_doc_freq=1.0))
    assert candidate_recall(candidates, truth)["macro_candidate_recall"] == 1.0
    assert set(candidates.columns) == {
        "source1_entity_id",
        "candidate_entity_id",
        "block_score",
        "block_rank",
    }


def test_country_bucketing_excludes_cross_country_pairs(sources):
    s1, targets, _ = sources
    candidates = generate_candidates(s1, targets, BlockingConfig(top_k=10, max_doc_freq=1.0))
    india_candidates = candidates[candidates["source1_entity_id"] == "S1-2"]["candidate_entity_id"]
    assert set(india_candidates) <= {"S2-2", "S2-3"}


def test_features_are_finite_and_discriminative(sources):
    s1, targets, _ = sources
    from src.utils import build_record_lookup

    candidates = pd.DataFrame(
        {
            "source1_entity_id": ["S1-2", "S1-2"],
            "candidate_entity_id": ["S2-2", "S2-3"],
            "block_score": [0.9, 0.1],
            "block_rank": [0, 1],
        }
    )
    features = build_features(candidates, build_record_lookup(s1), build_record_lookup(targets))
    assert list(features.columns) == ["source1_entity_id", "candidate_entity_id", *FEATURE_COLUMNS]
    assert np.isfinite(features[FEATURE_COLUMNS].to_numpy()).all()

    match = features[
        (features.source1_entity_id == "S1-2") & (features.candidate_entity_id == "S2-2")
    ].iloc[0]
    non_match = features[
        (features.source1_entity_id == "S1-2") & (features.candidate_entity_id == "S2-3")
    ].iloc[0]
    assert match.name_token_set > non_match.name_token_set


def test_label_pairs_marks_ground_truth(sources):
    s1, targets, truth = sources
    candidates = generate_candidates(s1, targets, BlockingConfig(top_k=10, max_doc_freq=1.0))
    labels = label_pairs(candidates, truth)
    positives = {
        (row.source1_entity_id, row.candidate_entity_id)
        for row, label in zip(candidates.itertuples(), labels)
        if label
    }
    assert positives == {("S1-1", "S2-1"), ("S1-1", "S3-1"), ("S1-2", "S2-2")}


def test_f05_matches_reference_evaluator():
    sys.path.insert(0, str(REPO_ROOT))
    import evaluator as reference

    cases = [
        ({"A"}, {"A"}),
        (set(), set()),
        ({"A"}, set()),
        (set(), {"A"}),
        ({"A"}, {"A", "B"}),
        ({"A", "C"}, {"A", "B"}),
    ]
    for predicted, true in cases:
        assert compute_f05_single(predicted, true) == pytest.approx(
            reference.compute_f05_single(predicted, true)
        )


def test_f05_prefers_precision_over_recall():
    precise = compute_f05_single({"A"}, {"A", "B", "C"})
    noisy = compute_f05_single({"A", "B", "C", "D", "E", "F"}, {"A", "B", "C"})
    assert precise > noisy


def _scored():
    return pd.DataFrame(
        {
            "source1_entity_id": ["S1-1", "S1-1", "S1-1", "S1-2", "S1-2"],
            "candidate_entity_id": ["S2-1", "S3-1", "S2-9", "S2-2", "S2-3"],
            "probability": [0.95, 0.80, 0.10, 0.15, 0.05],
        }
    )


def test_decision_rule_threshold_margin_and_cap():
    scored = _scored()
    strict = apply_decision_rule(
        scored, DecisionRule(threshold=0.9, margin=1.0, max_k=5, always_keep_best=False)
    )
    assert strict == {"S1-1": {"S2-1"}, "S1-2": set()}

    margin_limited = apply_decision_rule(
        scored, DecisionRule(threshold=0.05, margin=0.2, max_k=5, always_keep_best=False)
    )
    assert margin_limited["S1-1"] == {"S2-1", "S3-1"}

    capped = apply_decision_rule(
        scored, DecisionRule(threshold=0.05, margin=1.0, max_k=1, always_keep_best=False)
    )
    assert capped["S1-1"] == {"S2-1"}

    fallback = apply_decision_rule(
        scored, DecisionRule(threshold=0.9, margin=1.0, max_k=5, always_keep_best=True, best_threshold=0.1)
    )
    assert fallback["S1-2"] == {"S2-2"}


@pytest.mark.parametrize(
    "rule",
    [
        DecisionRule(threshold=0.5, margin=1.0, max_k=5, always_keep_best=False),
        DecisionRule(threshold=0.05, margin=0.2, max_k=3, always_keep_best=False),
        DecisionRule(threshold=0.9, margin=1.0, max_k=2, always_keep_best=True, best_threshold=0.1),
    ],
)
def test_tuning_view_matches_direct_evaluation(rule):
    scored = _scored()
    truth = {"S1-1": {"S2-1", "S3-1"}, "S1-2": {"S2-2"}}
    direct, _ = evaluate_f05_macro(apply_decision_rule(scored, rule, truth.keys()), truth)
    assert _TuningView(scored, truth).macro_f05(rule) == pytest.approx(direct)


def test_tuning_view_penalises_records_without_candidates():
    scored = _scored()
    truth = {"S1-1": {"S2-1", "S3-1"}, "S1-2": {"S2-2"}, "S1-3": {"S2-7"}}
    rule = DecisionRule(threshold=0.5, margin=1.0, max_k=5, always_keep_best=False)
    direct, _ = evaluate_f05_macro(apply_decision_rule(scored, rule, truth.keys()), truth)
    assert _TuningView(scored, truth).macro_f05(rule) == pytest.approx(direct)
