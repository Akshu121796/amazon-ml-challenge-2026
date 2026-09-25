
import numpy as np
import pandas as pd

def compute_f05_single(predicted, true):
    predicted = set(predicted)
    true = set(true)

    tp = len(predicted & true)
    fp = len(predicted - true)
    fn = len(true - predicted)

    if len(true) == 0:
        return 1.0 if len(predicted) == 0 else 0.0

    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(true) if true else 0.0

    denominator = 0.25 * precision + recall
    if denominator == 0:
        return 0.0

    return (1.25 * precision * recall) / denominator


def evaluate_f05_macro(pred_dict, true_dict):
    scores = []
    rows = []

    for s1_id, true_set in true_dict.items():
        pred_set = pred_dict.get(s1_id, set())
        score = compute_f05_single(pred_set, true_set)
        scores.append(score)

        tp = len(set(pred_set) & set(true_set))
        fp = len(set(pred_set) - set(true_set))
        fn = len(set(true_set) - set(pred_set))

        rows.append({
            "source1_entity_id": s1_id,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if (tp + fp) else (1.0 if not true_set and not pred_set else 0.0),
            "recall": tp / len(true_set) if true_set else (1.0 if not pred_set else 0.0),
            "f05": score,
        })

    return (float(np.mean(scores)) if scores else 0.0), pd.DataFrame(rows)


def aggregate_diagnostics(pred_dict, true_dict):
    tp = fp = fn = 0

    for s1_id, true_set in true_dict.items():
        pred_set = set(pred_dict.get(s1_id, set()))
        true_set = set(true_set)
        tp += len(pred_set & true_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "aggregate_precision": precision,
        "aggregate_recall": recall,
    }
