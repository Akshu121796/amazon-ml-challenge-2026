"""Pairwise match classifier and F0.5-optimal decision rule."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS

LOGGER = logging.getLogger("ber.model")

DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": 42,
}


@dataclass
class DecisionRule:
    """Turns pair probabilities into a predicted match set per Source 1 record.

    A candidate is predicted when its probability clears ``threshold`` and is
    within ``margin`` of the record's best candidate; at most ``max_k``
    candidates are kept.  ``always_keep_best`` additionally keeps the single best
    candidate when it clears ``best_threshold`` — useful because the ground truth
    has at least one match for every Source 1 record.
    """

    threshold: float = 0.5
    margin: float = 1.0
    max_k: int = 12
    always_keep_best: bool = True
    best_threshold: float = 0.2

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @staticmethod
    def from_json(path: Path) -> "DecisionRule":
        return DecisionRule(**json.loads(Path(path).read_text(encoding="utf-8")))


def label_pairs(candidates: pd.DataFrame, truth: Dict[str, Set[str]]) -> np.ndarray:
    return np.fromiter(
        (
            1 if pair.candidate_entity_id in truth.get(pair.source1_entity_id, ()) else 0
            for pair in candidates.itertuples()
        ),
        dtype=np.int8,
        count=len(candidates),
    )


def train_classifier(
    features: pd.DataFrame,
    labels: np.ndarray,
    valid_features: pd.DataFrame | None = None,
    valid_labels: np.ndarray | None = None,
    params: Dict[str, object] | None = None,
    num_boost_round: int = 600,
    early_stopping_rounds: int = 50,
) -> lgb.Booster:
    params = {**DEFAULT_PARAMS, **(params or {})}
    train_set = lgb.Dataset(features[FEATURE_COLUMNS], label=labels, free_raw_data=False)
    valid_sets, callbacks = [], [lgb.log_evaluation(period=100)]
    if valid_features is not None and valid_labels is not None and len(valid_features):
        valid_sets.append(
            lgb.Dataset(valid_features[FEATURE_COLUMNS], label=valid_labels, reference=train_set)
        )
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
    return lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets or None,
        callbacks=callbacks,
    )


def predict_proba(booster: lgb.Booster, features: pd.DataFrame) -> np.ndarray:
    if features.empty:
        return np.empty(0, dtype=np.float64)
    return booster.predict(
        features[FEATURE_COLUMNS], num_iteration=booster.best_iteration or None
    )


def apply_decision_rule(
    scored: pd.DataFrame, rule: DecisionRule, all_s1_ids: Iterable[str] | None = None
) -> Dict[str, Set[str]]:
    """Map scored candidate pairs to a predicted match set per Source 1 record."""
    predictions: Dict[str, Set[str]] = {s1_id: set() for s1_id in (all_s1_ids or [])}
    if scored.empty:
        return predictions

    ordered = scored.sort_values(["source1_entity_id", "probability"], ascending=[True, False])
    for s1_id, group in ordered.groupby("source1_entity_id", sort=False):
        probabilities = group["probability"].to_numpy()
        candidate_ids = group["candidate_entity_id"].to_numpy()
        best = probabilities[0]
        keep = (probabilities >= rule.threshold) & (probabilities >= best - rule.margin)
        selected = list(candidate_ids[keep][: rule.max_k])
        if not selected and rule.always_keep_best and best >= rule.best_threshold:
            selected = [candidate_ids[0]]
        predictions[s1_id] = set(selected)
    return predictions


class _TuningView:
    """Candidate scores laid out for fast repeated rule evaluation.

    Pairs are grouped per Source 1 record and sorted by descending probability, so
    any (threshold, margin, max_k) rule selects a prefix of each group: the number
    of selected pairs is a segmented count and the true positives are a lookup in
    the group's cumulative hit counts.  Scoring a rule is therefore a handful of
    vectorised passes over the flattened pair arrays.
    """

    def __init__(self, scored: pd.DataFrame, truth: Dict[str, Set[str]]):
        ordered = scored[scored["source1_entity_id"].isin(truth.keys())].sort_values(
            ["source1_entity_id", "probability"], ascending=[True, False]
        )
        self.n_records = len(truth)
        self.probabilities = ordered["probability"].to_numpy(dtype=np.float64)
        s1_ids = ordered["source1_entity_id"].to_numpy()
        hits = np.fromiter(
            (
                cid in truth[s1]
                for s1, cid in zip(s1_ids, ordered["candidate_entity_id"].to_numpy())
            ),
            dtype=np.float64,
            count=len(ordered),
        )

        if len(ordered):
            boundaries = np.flatnonzero(s1_ids[1:] != s1_ids[:-1]) + 1
            self.group_start = np.concatenate([[0], boundaries])
            self.group_end = np.concatenate([boundaries, [len(ordered)]])
            self.group_ids = s1_ids[self.group_start]
        else:
            self.group_start = np.empty(0, dtype=np.int64)
            self.group_end = np.empty(0, dtype=np.int64)
            self.group_ids = np.empty(0, dtype=object)

        # cumulative_hits[i] = hits strictly before flat position i, per group
        cumulative = np.concatenate([[0.0], np.cumsum(hits)])
        self.cumulative = cumulative
        self.group_base = cumulative[self.group_start]
        self.group_best = (
            self.probabilities[self.group_start] if len(self.group_start) else np.empty(0)
        )
        self.group_sizes = self.group_end - self.group_start
        self.truth_sizes = np.array(
            [len(truth[s1_id]) for s1_id in self.group_ids], dtype=np.float64
        )

    def macro_f05(self, rule: DecisionRule) -> float:
        if not self.n_records or not len(self.group_start):
            return 0.0
        cutoffs = np.maximum(rule.threshold, self.group_best - rule.margin)
        selected_mask = self.probabilities >= np.repeat(cutoffs, self.group_sizes)
        n_selected = np.add.reduceat(selected_mask, self.group_start).astype(np.int64)
        n_selected = np.minimum(n_selected, rule.max_k)
        if rule.always_keep_best:
            n_selected = np.where(
                (n_selected == 0) & (self.group_best >= rule.best_threshold), 1, n_selected
            )

        tp = self.cumulative[self.group_start + n_selected] - self.group_base
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.where(n_selected > 0, tp / np.maximum(n_selected, 1), 0.0)
            recall = tp / self.truth_sizes
            denominator = 0.25 * precision + recall
            f05 = np.where(denominator > 0, (1.25 * precision * recall) / denominator, 0.0)
        return float(f05.sum() / self.n_records)


def tune_decision_rule(
    scored: pd.DataFrame,
    truth: Dict[str, Set[str]],
    thresholds: Sequence[float] = tuple(np.round(np.arange(0.05, 0.96, 0.05), 2)),
    margins: Sequence[float] = (0.2, 0.5, 1.0),
    max_ks: Sequence[int] = (1, 2, 3, 4, 6, 8, 12),
    best_thresholds: Sequence[float] = (0.0, 0.2),
) -> Tuple[DecisionRule, float, pd.DataFrame]:
    """Grid-search the decision rule that maximises macro F0.5 on a labelled split."""
    view = _TuningView(scored, truth)
    results: List[Dict[str, float]] = []
    best_rule, best_score = DecisionRule(), -1.0
    for threshold in thresholds:
        for margin in margins:
            for max_k in max_ks:
                for best_threshold in best_thresholds:
                    rule = DecisionRule(
                        threshold=float(threshold),
                        margin=float(margin),
                        max_k=int(max_k),
                        always_keep_best=best_threshold > 0.0,
                        best_threshold=float(best_threshold),
                    )
                    score = view.macro_f05(rule)
                    results.append({**asdict(rule), "macro_f05": score})
                    if score > best_score:
                        best_rule, best_score = rule, score
    LOGGER.info("best decision rule %s with macro F0.5 %.6f", best_rule, best_score)
    grid = pd.DataFrame(results).sort_values("macro_f05", ascending=False).reset_index(drop=True)
    return best_rule, best_score, grid
