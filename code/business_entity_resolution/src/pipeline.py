"""End-to-end entity resolution pipeline: train, evaluate and predict.
 
Examples
--------
Train the matcher on the 80% training split and tune the decision rule on the
frozen 20% validation split::
 
    python -m src.pipeline train --train-dir dataset/train --model-dir artifacts \
        --val-ids validation_artifacts/val_s1_ids.txt --train-sample 150000
 
Score the held-out validation split with the trained artefacts::
 
    python -m src.pipeline evaluate --train-dir dataset/train --model-dir artifacts \
        --val-ids validation_artifacts/val_s1_ids.txt --eval-sample 50000
 
Generate the submission files for the test data::
 
    python -m src.pipeline predict --test-dir dataset/test --model-dir artifacts \
        --output-dir output
"""
 
from __future__ import annotations
 
import argparse
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Set, Tuple
 
import lightgbm as lgb
import numpy as np
import pandas as pd
 
from .blocking import BlockingConfig, CandidateChunk, candidate_recall, iter_candidates
from .evaluation import aggregate_diagnostics, evaluate_f05_macro
from .features import build_features
from .model import (
    DecisionRule,
    apply_decision_rule,
    label_pairs,
    predict_proba,
    train_classifier,
    tune_decision_rule,
)
from .utils import (
    build_record_lookup,
    format_match_ids,
    read_entity_ids,
    read_ground_truth,
    read_normalized_source,
    setup_logging,
    write_json,
)
 
LOGGER = logging.getLogger("ber.pipeline")
 
MODEL_FILE = "matcher.txt"
RULE_FILE = "decision_rule.json"
METRICS_FILE = "metrics.json"
 
 
def _blocking_config(args: argparse.Namespace) -> BlockingConfig:
    return BlockingConfig(
        top_k=args.top_k,
        max_doc_freq=args.max_doc_freq,
        max_doc_count=args.max_doc_count,
        bucket_by_country=not args.no_country_bucket,
        chunk_size=args.chunk_size,
    )
 
 
def load_sources(
    directory: Path, prefix: str, s1_ids: Set[str] | None = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Source 1 and the concatenated Source 2/3 reference pool, normalized.
 
    Reading and normalizing in chunks (and keeping only ``s1_ids`` when given) is what
    keeps the 10M-record reference pool inside a few GB of RAM.
    """
    s1 = read_normalized_source(directory / f"{prefix}_source1.tsv", keep_ids=s1_ids)
    targets = pd.concat(
        [
            read_normalized_source(directory / f"{prefix}_source2.tsv"),
            read_normalized_source(directory / f"{prefix}_source3.tsv"),
        ],
        ignore_index=True,
    )
    LOGGER.info("loaded %s S1 records and %s reference records", f"{len(s1):,}", f"{len(targets):,}")
    return s1, targets
 
 
def read_id_list(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
 
 
def sample_ids(ids: Sequence[str], limit: int | None, seed: int = 42) -> List[str]:
    if not limit or limit >= len(ids):
        return list(ids)
    rng = np.random.default_rng(seed)
    return list(np.array(ids)[rng.choice(len(ids), size=limit, replace=False)])
 
 
def chunk_features(chunk: CandidateChunk) -> pd.DataFrame:
    return build_features(
        chunk.pairs, build_record_lookup(chunk.s1), build_record_lookup(chunk.targets)
    )
 
 
def iter_features(
    s1: pd.DataFrame, targets: pd.DataFrame, config: BlockingConfig
) -> Iterator[pd.DataFrame]:
    for chunk in iter_candidates(s1, targets, config):
        if chunk.pairs.empty:
            continue
        yield chunk_features(chunk)
 
 
def collect_features(
    s1: pd.DataFrame, targets: pd.DataFrame, config: BlockingConfig
) -> pd.DataFrame:
    """Materialise the feature matrix for a sample small enough to hold in memory."""
    frames = list(iter_features(s1, targets, config))
    if not frames:
        return build_features(pd.DataFrame(), {}, {})
    return pd.concat(frames, ignore_index=True)
 
 
def score_features(booster: lgb.Booster, features: pd.DataFrame) -> pd.DataFrame:
    scored = features[["source1_entity_id", "candidate_entity_id"]].copy()
    scored["probability"] = predict_proba(booster, features)
    return scored
 
 
def score_candidates(
    s1: pd.DataFrame,
    targets: pd.DataFrame,
    booster: lgb.Booster,
    config: BlockingConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Block, featurise and score; returns (candidates, scored pairs)."""
    features = collect_features(s1, targets, config)
    candidates = features[["source1_entity_id", "candidate_entity_id", "block_score", "block_rank"]]
    return candidates, score_features(booster, features)
 
 
def write_column(
    s1_ids: Sequence[str], values: Sequence[str], column: str, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"source1_entity_id": s1_ids, column: values})
    frame.to_csv(path, sep="\t", index=False)
    LOGGER.info("wrote %s rows to %s", f"{len(frame):,}", path)
 
 
def command_train(args: argparse.Namespace) -> None:
    config = _blocking_config(args)
    train_dir = Path(args.train_dir)
 
    all_ids = read_entity_ids(train_dir / "train_source1.tsv")
    val_ids = set(read_id_list(Path(args.val_ids))) if args.val_ids else set()
    train_ids = sample_ids([i for i in all_ids if i not in val_ids], args.train_sample, args.seed)
    tune_ids = sample_ids(sorted(val_ids), args.tune_sample, args.seed) if val_ids else []
    del all_ids
 
    LOGGER.info(
        "training on %s S1 records, tuning on %s", f"{len(train_ids):,}", f"{len(tune_ids):,}"
    )
    keep = set(train_ids) | set(tune_ids)
    s1, targets = load_sources(train_dir, "train", s1_ids=keep)
    truth = read_ground_truth(train_dir / "train_ground_truth.tsv", keep_ids=keep)
 
    fit_s1 = s1[s1["entity_id"].isin(set(train_ids))]
    fit_features = collect_features(fit_s1, targets, config)
    fit_labels = label_pairs(fit_features, truth)
    LOGGER.info(
        "training pairs: %s (%.3f%% positive)",
        f"{len(fit_features):,}",
        100 * float(fit_labels.mean()) if len(fit_labels) else 0.0,
    )
 
    tune_s1 = s1[s1["entity_id"].isin(set(tune_ids))] if tune_ids else fit_s1.head(0)
    tune_features = collect_features(tune_s1, targets, config) if len(tune_s1) else None
    tune_labels = label_pairs(tune_features, truth) if tune_features is not None else None
 
    booster = train_classifier(
        fit_features,
        fit_labels,
        tune_features,
        tune_labels,
        num_boost_round=args.num_boost_round,
    )
 
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_dir / MODEL_FILE))
 
    metrics: Dict[str, object] = {
        "n_train_s1": len(train_ids),
        "n_train_pairs": int(len(fit_features)),
        "train_positive_rate": float(fit_labels.mean()) if len(fit_labels) else 0.0,
        "blocking": {
            "top_k": config.top_k,
            "max_doc_freq": config.max_doc_freq,
            "bucket_by_country": config.bucket_by_country,
        },
        "feature_importance": dict(
            zip(booster.feature_name(), booster.feature_importance("gain").tolist())
        ),
    }
 
    rule = DecisionRule()
    if tune_features is not None and len(tune_features):
        tune_truth = {i: truth.get(i, set()) for i in tune_ids}
        scored = tune_features[["source1_entity_id", "candidate_entity_id"]].copy()
        scored["probability"] = predict_proba(booster, tune_features)
        rule, tuned_f05, grid = tune_decision_rule(scored, tune_truth)
        grid.head(50).to_csv(model_dir / "decision_rule_grid.csv", index=False)
        predictions = apply_decision_rule(scored, rule, tune_truth.keys())
        metrics["validation"] = {
            "macro_f05": tuned_f05,
            "candidate_recall": candidate_recall(tune_features, tune_truth),
            "diagnostics": aggregate_diagnostics(predictions, tune_truth),
        }
    rule.to_json(model_dir / RULE_FILE)
    write_json(metrics, model_dir / METRICS_FILE)
    LOGGER.info("saved model artefacts to %s", model_dir)
 
 
def command_evaluate(args: argparse.Namespace) -> None:
    config = _blocking_config(args)
    model_dir = Path(args.model_dir)
    booster = lgb.Booster(model_file=str(model_dir / MODEL_FILE))
    rule = DecisionRule.from_json(model_dir / RULE_FILE)
 
    train_dir = Path(args.train_dir)
    val_ids = (
        read_id_list(Path(args.val_ids))
        if args.val_ids
        else read_entity_ids(train_dir / "train_source1.tsv")
    )
    val_ids = sample_ids(val_ids, args.eval_sample, args.seed)
 
    eval_s1, targets = load_sources(train_dir, "train", s1_ids=set(val_ids))
    truth = read_ground_truth(train_dir / "train_ground_truth.tsv", keep_ids=set(val_ids))
    candidates, scored = score_candidates(eval_s1, targets, booster, config)
    eval_truth = {i: truth.get(i, set()) for i in val_ids}
    predictions = apply_decision_rule(scored, rule, eval_truth.keys())
 
    macro_f05, per_entity = evaluate_f05_macro(predictions, eval_truth)
    report = {
        "n_evaluated": len(eval_truth),
        "macro_f05": macro_f05,
        "decision_rule": rule.__dict__,
        "candidate_recall": candidate_recall(candidates, eval_truth),
        "diagnostics": aggregate_diagnostics(predictions, eval_truth),
    }
    output_dir = Path(args.output_dir)
    write_json(report, output_dir / "validation_report.json")
    per_entity.to_csv(output_dir / "validation_per_entity.tsv", sep="\t", index=False)
    LOGGER.info("macro F0.5 = %.6f over %s records", macro_f05, f"{len(eval_truth):,}")
    print(f"macro_f05={macro_f05:.6f}")
 
 
def command_predict(args: argparse.Namespace) -> None:
    config = _blocking_config(args)
    model_dir = Path(args.model_dir)
    booster = lgb.Booster(model_file=str(model_dir / MODEL_FILE))
    rule = DecisionRule.from_json(model_dir / RULE_FILE)
 
    s1, targets = load_sources(Path(args.test_dir), "test")
    s1_ids = s1["entity_id"].tolist()
    position = {entity_id: index for index, entity_id in enumerate(s1_ids)}
 
    # Results are filled in by Source 1 row position: the candidate lists of 1.7M test
    # records do not fit in memory as Python collections, only as the output strings.
    candidate_column = np.full(len(s1_ids), "", dtype=object)
    matched_column = np.full(len(s1_ids), "", dtype=object)
    n_pairs = 0
    n_matches = 0
 
    for processed, chunk in enumerate(iter_candidates(s1, targets, config), start=1):
        if chunk.pairs.empty:
            continue
        features = chunk_features(chunk)
        scored = score_features(booster, features)
        predictions = apply_decision_rule(scored, rule, chunk.s1["entity_id"].tolist())
        n_pairs += len(scored)
        for entity_id, group in scored.groupby("source1_entity_id", sort=False):
            candidate_column[position[entity_id]] = format_match_ids(group["candidate_entity_id"])
        for entity_id, matches in predictions.items():
            matched_column[position[entity_id]] = format_match_ids(matches)
            n_matches += len(matches)
        if processed % 50 == 0:
            LOGGER.info("scored %s candidate pairs so far", f"{n_pairs:,}")
 
    output_dir = Path(args.output_dir)
    write_column(s1_ids, candidate_column, "candidate_entity_ids", output_dir / "candidate_pairs.tsv")
    write_column(s1_ids, matched_column, "matched_entity_ids", output_dir / "matching_results.tsv")
    write_json(
        {
            "n_source1": len(s1_ids),
            "n_candidate_pairs": n_pairs,
            "mean_predicted_matches": n_matches / max(len(s1_ids), 1),
            "decision_rule": rule.__dict__,
        },
        output_dir / "prediction_summary.json",
    )
 
 
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
 
    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--model-dir", default="artifacts")
        sub.add_argument("--output-dir", default="output")
        sub.add_argument("--top-k", type=int, default=30)
        sub.add_argument("--max-doc-freq", type=float, default=0.002)
        sub.add_argument("--max-doc-count", type=int, default=3000)
        sub.add_argument("--chunk-size", type=int, default=2000)
        sub.add_argument("--no-country-bucket", action="store_true")
        sub.add_argument("--seed", type=int, default=42)
 
    train = subparsers.add_parser("train", help="fit the matcher and tune the decision rule")
    train.add_argument("--train-dir", required=True)
    train.add_argument("--val-ids", default=None, help="frozen validation S1 ids (one per line)")
    train.add_argument("--train-sample", type=int, default=150_000)
    train.add_argument("--tune-sample", type=int, default=25_000)
    train.add_argument("--num-boost-round", type=int, default=600)
    add_common(train)
    train.set_defaults(func=command_train)
 
    evaluate = subparsers.add_parser("evaluate", help="score the held-out validation split")
    evaluate.add_argument("--train-dir", required=True)
    evaluate.add_argument("--val-ids", default=None)
    evaluate.add_argument("--eval-sample", type=int, default=50_000)
    add_common(evaluate)
    evaluate.set_defaults(func=command_evaluate)
 
    predict = subparsers.add_parser("predict", help="write the submission files for the test data")
    predict.add_argument("--test-dir", required=True)
    add_common(predict)
    predict.set_defaults(func=command_predict)
 
    return parser
 
 
def main(argv: Sequence[str] | None = None) -> None:
    setup_logging()
    args = build_parser().parse_args(argv)
    args.func(args)
 
 
if __name__ == "__main__":
    main()