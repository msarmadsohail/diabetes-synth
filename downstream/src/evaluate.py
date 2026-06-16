from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    f1_score, precision_score, recall_score,
    precision_recall_curve,
)


def best_f1_threshold(y_true, probs):
    prec, rec, thresholds = precision_recall_curve(y_true, probs)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = np.argmax(f1s[:-1])
    return float(thresholds[best_idx]), float(f1s[best_idx])


def compute_metrics(y_true, probs) -> dict:
    threshold, f1 = best_f1_threshold(y_true, probs)
    preds = (probs >= threshold).astype(int)
    return {
        "pr_auc":    round(float(average_precision_score(y_true, probs)), 6),
        "roc_auc":   round(float(roc_auc_score(y_true, probs)), 6),
        "f1":        round(float(f1), 6),
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 6),
        "recall":    round(float(recall_score(y_true, preds, zero_division=0)), 6),
        "threshold": round(threshold, 6),
    }


def save_fold_results(results: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))


def save_summary(all_fold_results: list[dict], out_path: Path) -> None:
    classifiers = list(all_fold_results[0].keys())
    summary = {}
    for clf in classifiers:
        metrics = ["pr_auc", "roc_auc", "f1", "precision", "recall"]
        summary[clf] = {}
        for m in metrics:
            vals = [r[clf][m] for r in all_fold_results if clf in r and m in r[clf]]
            summary[clf][m] = {
                "mean": round(float(np.mean(vals)), 6),
                "std":  round(float(np.std(vals)), 6),
                "per_fold": [round(v, 6) for v in vals],
            }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    return summary
