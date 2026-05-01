"""Metric helpers shared by training, tuning and evaluate stages."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None
) -> dict[str, float]:
    """Standard binary-classification scoreboard.

    `y_proba` is optional because not every model exposes it (e.g. SVM
    without probability calibration). When absent, AUC metrics are skipped
    rather than zeroed — a missing key is more honest than a fake 0.
    """
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
    return metrics
