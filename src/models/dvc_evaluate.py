"""DVC stage 5 (evaluate) — scores the held-out test split.

Loads the model produced by `dvc_train`, runs it on the featurized test
parquet, and writes both a JSON metrics file (for `dvc metrics show`) and
a confusion-matrix CSV (for `dvc plots`). This is the final node of the
DVC pipeline — the report screenshot of `dvc dag` ends here.

Usage:
    python -m src.models.dvc_evaluate
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.config import PROJECT_ROOT, load_params
from src.features.build_features import Task
from src.models.evaluate import (
    binary_classification_metrics,
    multiclass_classification_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("dvc_evaluate")

FEATURIZED_DIR = PROJECT_ROOT / "data" / "featurized"
MODEL_IN = PROJECT_ROOT / "models" / "dvc_model.pkl"
LABEL_CLASSES_IN = PROJECT_ROOT / "models" / "dvc_label_classes.json"
METRICS_OUT = PROJECT_ROOT / "reports" / "test_metrics.json"
CONFUSION_OUT = PROJECT_ROOT / "reports" / "confusion_matrix.csv"


def _load_xy(split: str, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    x = pd.read_parquet(FEATURIZED_DIR / f"X_{split}.parquet")
    y = pd.read_parquet(FEATURIZED_DIR / f"y_{split}.parquet")[target_col]
    return x, y


def main() -> int:
    params = load_params()
    task: Task = params["train"]["task"]

    with (FEATURIZED_DIR / "manifest.json").open(encoding="utf-8") as fh:
        manifest = json.load(fh)
    target_col = manifest["target_column"]

    log.info("Loading model: %s", MODEL_IN)
    with MODEL_IN.open("rb") as fh:
        pipeline = pickle.load(fh)

    x_test, y_test = _load_xy("test", target_col)
    log.info("Test split: X=%s y=%s", x_test.shape, y_test.shape)

    est = pipeline.named_steps["estimator"]

    if task == "delay_binary":
        y_pred = pipeline.predict(x_test)
        y_proba: np.ndarray | None = None
        if hasattr(est, "predict_proba"):
            y_proba = pipeline.predict_proba(x_test)[:, 1]
        metrics = binary_classification_metrics(y_test.to_numpy(), y_pred, y_proba)
        cm_labels = [0, 1]
        cm = confusion_matrix(y_test.to_numpy(), y_pred, labels=cm_labels)
        cm_index = ["actual_on_time", "actual_delayed"]
        cm_columns = ["pred_on_time", "pred_delayed"]
    elif task == "delay_cause":
        if not LABEL_CLASSES_IN.exists():
            raise FileNotFoundError(
                f"Expected label-classes file at {LABEL_CLASSES_IN} for delay_cause "
                "task. Re-run dvc_train."
            )
        with LABEL_CLASSES_IN.open(encoding="utf-8") as fh:
            label_classes = json.load(fh)["classes"]
        from sklearn.preprocessing import LabelEncoder

        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(label_classes)
        y_test_int = label_encoder.transform(y_test)

        y_pred = pipeline.predict(x_test)
        y_proba_mc: np.ndarray | None = None
        if hasattr(est, "predict_proba"):
            y_proba_mc = pipeline.predict_proba(x_test)
        labels_int = list(range(len(label_classes)))
        metrics = multiclass_classification_metrics(
            y_test_int, y_pred, y_proba_mc, labels=labels_int
        )
        cm = confusion_matrix(y_test_int, y_pred, labels=labels_int)
        cm_index = [f"actual_{c}" for c in label_classes]
        cm_columns = [f"pred_{c}" for c in label_classes]
    else:
        raise NotImplementedError(f"task={task} not supported by DVC evaluate stage")

    log.info("Test metrics: %s", {k: round(v, 4) for k, v in metrics.items()})

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_OUT.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    log.info("Wrote metrics: %s", METRICS_OUT)

    cm_df = pd.DataFrame(cm, index=cm_index, columns=cm_columns)
    cm_df.to_csv(CONFUSION_OUT, index=True, encoding="utf-8")
    log.info("Wrote confusion matrix: %s", CONFUSION_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
