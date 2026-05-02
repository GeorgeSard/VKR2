"""DVC stage 4 (train) — fits the active model on featurized splits.

Reads the X/y parquets produced by `dvc_featurize` plus the manifest,
then builds the same Pipeline (preprocessor → estimator) used by the
MLflow training script. Outputs the fitted model as a pickled artifact
DVC tracks, and emits a JSON metrics file DVC reads via `dvc metrics show`.

This entry point intentionally does NOT log to MLflow: the DVC pipeline
exists to make the ML lifecycle reproducible from raw → metrics with one
command, and mixing two tracking systems in the same run muddies the
story. The MLflow workflow (`src/models/train.py`) covers that side.

Usage:
    python -m src.models.dvc_train
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.config import PROJECT_ROOT, load_params
from src.features.build_features import Task, build_preprocessor
from src.features.feature_sets import get_feature_set
from src.models.evaluate import (
    binary_classification_metrics,
    multiclass_classification_metrics,
)
from src.models.train import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("dvc_train")

FEATURIZED_DIR = PROJECT_ROOT / "data" / "featurized"
MODEL_OUT = PROJECT_ROOT / "models" / "dvc_model.pkl"
LABEL_CLASSES_OUT = PROJECT_ROOT / "models" / "dvc_label_classes.json"
METRICS_OUT = PROJECT_ROOT / "reports" / "val_metrics.json"


def _load_xy(split: str, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    x = pd.read_parquet(FEATURIZED_DIR / f"X_{split}.parquet")
    y = pd.read_parquet(FEATURIZED_DIR / f"y_{split}.parquet")[target_col]
    return x, y


def main() -> int:
    params = load_params()
    seed: int = params["base"]["random_seed"]
    model_name: str = params["train"]["active_model"]
    task: Task = params["train"]["task"]
    hp: dict[str, Any] = params["train"][model_name]

    with (FEATURIZED_DIR / "manifest.json").open(encoding="utf-8") as fh:
        manifest = json.load(fh)
    set_name = manifest["feature_set"]
    target_col = manifest["target_column"]
    if manifest["task"] != task:
        raise RuntimeError(
            f"Manifest task '{manifest['task']}' does not match params train.task '{task}'. "
            "Re-run dvc_featurize after changing params.yaml → train.task."
        )

    fs = get_feature_set(set_name)
    log.info(
        "feature_set=%s | model=%s | task=%s | seed=%s", set_name, model_name, task, seed
    )

    x_train, y_train = _load_xy("train", target_col)
    x_val, y_val = _load_xy("val", target_col)
    log.info("Loaded train=%d val=%d rows", len(x_train), len(x_val))

    label_encoder: LabelEncoder | None = None
    y_train_fit: np.ndarray | pd.Series = y_train
    y_val_fit: np.ndarray | pd.Series = y_val
    if task == "delay_cause":
        label_encoder = LabelEncoder()
        y_train_fit = label_encoder.fit_transform(y_train)
        y_val_fit = label_encoder.transform(y_val)
        log.info("Label encoder classes: %s", list(label_encoder.classes_))

    preprocessor = build_preprocessor(fs)
    estimator = build_model(model_name, hp, seed)
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("estimator", estimator)]
    )

    fit_params: dict[str, Any] = {}
    if task == "delay_cause" and model_name in {"xgboost", "lightgbm", "catboost"}:
        from sklearn.utils.class_weight import compute_sample_weight

        sw = compute_sample_weight(class_weight="balanced", y=y_train_fit)
        fit_params["estimator__sample_weight"] = sw

    log.info("Fitting %s on %d rows...", model_name, len(x_train))
    pipeline.fit(x_train, y_train_fit, **fit_params)

    est = pipeline.named_steps["estimator"]
    if task == "delay_binary":
        y_pred = pipeline.predict(x_val)
        y_proba: np.ndarray | None = None
        if hasattr(est, "predict_proba"):
            y_proba = pipeline.predict_proba(x_val)[:, 1]
        metrics = binary_classification_metrics(y_val.to_numpy(), y_pred, y_proba)
    elif task == "delay_cause":
        y_pred = pipeline.predict(x_val)
        y_proba_mc: np.ndarray | None = None
        if hasattr(est, "predict_proba"):
            y_proba_mc = pipeline.predict_proba(x_val)
        assert label_encoder is not None
        labels_int = list(range(len(label_encoder.classes_)))
        metrics = multiclass_classification_metrics(
            np.asarray(y_val_fit), y_pred, y_proba_mc, labels=labels_int
        )
    else:
        raise NotImplementedError(f"task={task} not supported by DVC train stage")

    log.info("Validation metrics: %s", {k: round(v, 4) for k, v in metrics.items()})

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)

    with MODEL_OUT.open("wb") as fh:
        pickle.dump(pipeline, fh)
    log.info("Saved model: %s", MODEL_OUT)

    if label_encoder is not None:
        with LABEL_CLASSES_OUT.open("w", encoding="utf-8") as fh:
            json.dump(
                {"classes": [str(c) for c in label_encoder.classes_]},
                fh,
                indent=2,
                ensure_ascii=False,
            )
        log.info("Saved label classes: %s", LABEL_CLASSES_OUT)

    with METRICS_OUT.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    log.info("Wrote metrics: %s", METRICS_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
