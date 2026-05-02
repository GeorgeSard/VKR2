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


def multiclass_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    labels: list | None = None,
) -> dict[str, float]:
    """Multi-class scoreboard for the delay-cause head.

    macro_f1 is the headline metric: it weights every cause class equally,
    so the dominant 'none' class can't drown out per-cause performance the
    way accuracy or weighted_f1 does. weighted_f1 is also logged for
    contrast — a big gap between the two flags class-imbalance pain.

    roc_auc_ovr_weighted is logged only when probabilities are supplied
    (and only if every label has at least one positive example in y_true).
    """
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }
    if y_proba is not None and labels is not None:
        try:
            metrics["roc_auc_ovr_weighted"] = float(
                roc_auc_score(
                    y_true,
                    y_proba,
                    multi_class="ovr",
                    average="weighted",
                    labels=labels,
                )
            )
        except ValueError:
            # Happens when a class has 0 positive examples in y_true; skip silently.
            pass
    return metrics
