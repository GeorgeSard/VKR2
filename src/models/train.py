"""Stage 6 — baseline training entry-point with MLflow tracking.

Everything that defines an experiment is read from params.yaml so changing
one section there and re-running this script gives you a new MLflow run
that differs from the previous one along exactly one axis (feature set,
or model, or hyperparameter). That is the screenshot story for the report.

Usage:
    python -m src.models.train
    python -m src.models.train --feature-set extended --model xgboost
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.config import load_params
from src.features.build_features import Task, make_xy
from src.features.feature_sets import get_feature_set
from src.models.evaluate import (
    binary_classification_metrics,
    multiclass_classification_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("train")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _dvc_data_hash() -> str:
    """Read the md5 DVC recorded for the raw parquet — that uniquely
    identifies the input dataset version regardless of code changes."""
    dvc_file = PROJECT_ROOT / "data/raw/flight_delays_ru.parquet.dvc"
    if not dvc_file.exists():
        return "unknown"
    with dvc_file.open() as fh:
        meta = yaml.safe_load(fh)
    try:
        return meta["outs"][0]["md5"][:12]
    except (KeyError, IndexError, TypeError):
        return "unknown"


def build_model(model_name: str, hp: dict[str, Any], random_seed: int) -> Any:
    """Returns an unfitted estimator. Hyperparameters come from
    params.yaml → train.<model_name>. Switching `train.active_model`
    between runs (with feature_set held fixed) is the model-axis demo.

    For boosted-tree libraries we keep n_jobs/threads modest by default —
    the dataset is small enough (<150k rows) that single-thread fit is
    seconds, and stable timings across runs make MLflow comparisons
    cleaner.
    """
    if model_name == "logreg":
        return LogisticRegression(
            C=hp.get("C", 1.0),
            max_iter=hp.get("max_iter", 1000),
            class_weight=hp.get("class_weight"),
            random_state=random_seed,
            solver="lbfgs",
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=hp.get("n_estimators", 200),
            max_depth=hp.get("max_depth"),
            class_weight=hp.get("class_weight"),
            random_state=random_seed,
            n_jobs=-1,
        )
    if model_name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=hp.get("n_estimators", 400),
            max_depth=hp.get("max_depth", 6),
            learning_rate=hp.get("learning_rate", 0.05),
            subsample=hp.get("subsample", 0.9),
            colsample_bytree=hp.get("colsample_bytree", 0.9),
            scale_pos_weight=hp.get("scale_pos_weight", 1.0),
            random_state=random_seed,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=-1,
        )
    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=hp.get("n_estimators", 400),
            num_leaves=hp.get("num_leaves", 63),
            learning_rate=hp.get("learning_rate", 0.05),
            subsample=hp.get("subsample", 0.9),
            colsample_bytree=hp.get("colsample_bytree", 0.9),
            random_state=random_seed,
            n_jobs=-1,
            verbose=-1,
        )
    if model_name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=hp.get("iterations", 400),
            depth=hp.get("depth", 6),
            learning_rate=hp.get("learning_rate", 0.05),
            random_seed=random_seed,
            verbose=False,
            allow_writing_files=False,
        )
    raise NotImplementedError(
        f"Model '{model_name}' not yet wired up in train.py. "
        "Add it here and to params.yaml → train.<name>."
    )


def _flatten(prefix: str, mapping: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}.{k}": v for k, v in mapping.items() if not isinstance(v, (dict, list))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train baseline + log to MLflow")
    parser.add_argument("--feature-set", default=None, help="Override features.active_set")
    parser.add_argument("--model", default=None, help="Override train.active_model")
    parser.add_argument("--task", default=None, help="Override train.task")
    parser.add_argument("--run-name", default=None, help="Override MLflow run name")
    args = parser.parse_args(argv)

    params = load_params()
    seed = params["base"]["random_seed"]
    set_name = args.feature_set or params["features"]["active_set"]
    model_name = args.model or params["train"]["active_model"]
    task: Task = args.task or params["train"]["task"]
    hp = params["train"][model_name]

    log.info("feature_set=%s | model=%s | task=%s | seed=%s", set_name, model_name, task, seed)

    # MLflow setup. tracking_uri may be a local file: URI for the baseline
    # phase; switch to http://mlflow:5000 once Docker stack is up.
    from src.config import PROJECT_ROOT as _ROOT  # avoid stale import in tests

    raw_uri = params["mlflow"]["tracking_uri"]
    if raw_uri.startswith("file:") and not raw_uri.startswith("file:/"):
        # Resolve relative file:./mlruns to an absolute path so MLflow CLI
        # launched from a different working directory still finds the runs.
        rel = raw_uri.removeprefix("file:")
        tracking_uri = f"file:{(_ROOT / rel).resolve()}"
    else:
        tracking_uri = raw_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    log.info("MLflow tracking URI: %s", tracking_uri)

    fs = get_feature_set(set_name)

    train_df = pd.read_parquet(params["split"]["train_path"])
    val_df = pd.read_parquet(params["split"]["val_path"])
    log.info("Loaded train=%d val=%d rows", len(train_df), len(val_df))

    x_train, y_train = make_xy(train_df, fs, task)
    x_val, y_val = make_xy(val_df, fs, task)
    log.info("After cleanup: train X=%s y=%s | val X=%s y=%s",
             x_train.shape, y_train.shape, x_val.shape, y_val.shape)

    log.info("Class balance train: %s", y_train.value_counts(normalize=True).round(3).to_dict())

    # XGBoost requires integer-encoded labels for multi-class. To keep
    # the train.py main path uniform across estimators we wrap multi-class
    # targets in a LabelEncoder; binary tasks pass through unchanged
    # (LabelEncoder would still work, but is_departure_delayed_15m is
    # already 0/1 so we keep the original dtype for sanity).
    label_encoder: LabelEncoder | None = None
    y_train_fit: np.ndarray | pd.Series = y_train
    y_val_fit: np.ndarray | pd.Series = y_val
    if task == "delay_cause":
        label_encoder = LabelEncoder()
        y_train_fit = label_encoder.fit_transform(y_train)
        y_val_fit = label_encoder.transform(y_val)
        log.info("Label encoder classes: %s", list(label_encoder.classes_))

    from src.features.build_features import build_preprocessor  # local to avoid sklearn at import time
    preprocessor = build_preprocessor(fs)
    estimator = build_model(model_name, hp, seed)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])

    run_name = args.run_name or f"{model_name}-{set_name}-{task}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "git_commit": _git_commit(),
                "dvc_data_hash": _dvc_data_hash(),
                "feature_set": set_name,
                "model": model_name,
                "task": task,
                "params_version": params["base"]["params_version"],
                "split_strategy": params["split"]["strategy"],
            }
        )
        if task == "delay_cause" and label_encoder is not None:
            mlflow.set_tag("n_classes", str(len(label_encoder.classes_)))
        mlflow.log_params(
            {
                "random_seed": seed,
                "feature_set": set_name,
                "model": model_name,
                "task": task,
                "n_train": len(x_train),
                "n_val": len(x_val),
                "n_features_raw": len(fs.all_columns),
                **_flatten(model_name, hp),
            }
        )

        log.info("Fitting %s on %d rows...", model_name, len(x_train))
        # For multi-class we counter cause-imbalance with sample weights
        # derived from y_train frequencies. logreg also has class_weight
        # in its own hp; sample_weight stacks on top harmlessly because
        # logreg's class_weight is applied to the loss term *additively*
        # with sample_weight — we keep sample_weight only for boosters
        # to avoid double-weighting on logreg.
        fit_params: dict[str, Any] = {}
        if task == "delay_cause" and model_name in {"xgboost", "lightgbm", "catboost"}:
            from sklearn.utils.class_weight import compute_sample_weight

            sw = compute_sample_weight(class_weight="balanced", y=y_train_fit)
            fit_params["estimator__sample_weight"] = sw
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
            raise NotImplementedError(f"Eval for task={task} not implemented")

        mlflow.log_metrics(metrics)
        log.info("Metrics: %s", {k: round(v, 4) for k, v in metrics.items()})

        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        if label_encoder is not None:
            # Persist class order so the inference layer can decode integer
            # predictions back into human-readable cause names.
            classes_path = PROJECT_ROOT / "models" / f"label_classes_{task}.yaml"
            classes_path.parent.mkdir(parents=True, exist_ok=True)
            with classes_path.open("w") as fh:
                yaml.safe_dump(
                    {"classes": [str(c) for c in label_encoder.classes_]}, fh
                )
            mlflow.log_artifact(str(classes_path))
        log.info("Logged MLflow run: %s", run.info.run_id)

    return 0


if __name__ == "__main__":
    sys.exit(main())
