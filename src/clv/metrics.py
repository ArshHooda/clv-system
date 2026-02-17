from __future__ import annotations

from sklearn.metrics import average_precision_score, roc_auc_score


def binary_metrics(y_true, probs) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
    }
