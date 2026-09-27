#!/usr/bin/env bash
# End-to-end run: train -> evaluate on the frozen validation split -> predict -> validate.
set -euo pipefail
 
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="$ROOT/code/business_entity_resolution"
 
TRAIN_DIR="${TRAIN_DIR:-$ROOT/dataset/train}"
TEST_DIR="${TEST_DIR:-$ROOT/dataset/test}"
MODEL_DIR="${MODEL_DIR:-$ROOT/artifacts}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output}"
VAL_IDS="${VAL_IDS:-$ROOT/validation_artifacts/val_s1_ids.txt}"
TRAIN_SAMPLE="${TRAIN_SAMPLE:-30000}"
TUNE_SAMPLE="${TUNE_SAMPLE:-15000}"
EVAL_SAMPLE="${EVAL_SAMPLE:-30000}"
TOP_K="${TOP_K:-50}"
 
cd "$PKG"
 
py -m src.pipeline train \
    --train-dir "$TRAIN_DIR" \
    --val-ids "$VAL_IDS" \
    --model-dir "$MODEL_DIR" \
    --train-sample "$TRAIN_SAMPLE" \
    --tune-sample "$TUNE_SAMPLE" \
    --top-k "$TOP_K"
 
py -m src.pipeline evaluate \
    --train-dir "$TRAIN_DIR" \
    --val-ids "$VAL_IDS" \
    --model-dir "$MODEL_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --eval-sample "$EVAL_SAMPLE" \
    --top-k "$TOP_K"
 
py -m src.pipeline predict \
    --test-dir "$TEST_DIR" \
    --model-dir "$MODEL_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --top-k "$TOP_K"
 
py "$ROOT/utils/validate_submission.py" \
    --matching "$OUTPUT_DIR/matching_results.tsv" \
    --candidate "$OUTPUT_DIR/candidate_pairs.tsv" \
    --test-dir "$TEST_DIR"