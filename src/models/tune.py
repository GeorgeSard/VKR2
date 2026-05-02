"""Stage 7 — Optuna tuning of a boosting model on the binary delay task.

One MLflow run for the whole study (not one per trial — that would
flood the UI). The run logs:
  - n_trials, sampler, seed
  - the best hyperparameters discovered (as params best_<name>)
  - all six classification metrics for the refitted best model on val
  - the trained pipeline as an artifact (sklearn flavor)
  - best_params.yaml as a side-artifact, ready to be pasted into
    params.yaml under train.<model> for the next manual run

Trial-level history is kept inside the Optuna study object; MLflow
sees only the summary so the screenshot story stays one-line-per-run.

Generic over the boosting library: --model xgboost (default) or
--model lightgbm. Search spaces are kept parallel where the libraries
share concepts (n_estimators, learning_rate, subsample, scale_pos_weight)
so cross-library comparison stays fair.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Callable

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import yaml
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from src.config import PROJECT_ROOT, load_params
from src.features.build_features import build_preprocessor, make_xy
from src.features.feature_sets import get_feature_set
from src.models.evaluate import (
    binary_classification_metrics,
    multiclass_classification_metrics,
)
from src.models.train import _dvc_data_hash, _git_commit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("tune")

SUPPORTED_MODELS = ("xgboost", "lightgbm")
SUPPORTED_TASKS = ("delay_binary", "delay_cause")


# ---------------------------------------------------------------------------
# XGBoost — original search space (preserved verbatim from the Run #6 study
# so re-running tune.py for xgboost still reproduces the historical study).
# ---------------------------------------------------------------------------


def _suggest_params_xgboost(trial: optuna.Trial, seed: int) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
        "random_state": seed,
        "tree_method": "hist",
        "eval_metric": "logloss",
        "n_jobs": -1,
    }


def _xgboost_kwargs_from_best(best: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        **best,
        "random_state": seed,
        "tree_method": "hist",
        "eval_metric": "logloss",
        "n_jobs": -1,
    }


def _suggest_params_xgboost_cause(trial: optuna.Trial, seed: int) -> dict[str, Any]:
    """Multi-class variant — drops scale_pos_weight (binary-only in XGBoost;
    silently ignored on multi-class) and switches eval_metric to mlogloss.
    Class imbalance is countered via sample_weight at fit time, not via
    a per-class scaling parameter.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "random_state": seed,
        "tree_method": "hist",
        "eval_metric": "mlogloss",
        "n_jobs": -1,
    }


def _xgboost_cause_kwargs_from_best(best: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        **best,
        "random_state": seed,
        "tree_method": "hist",
        "eval_metric": "mlogloss",
        "n_jobs": -1,
    }


# ---------------------------------------------------------------------------
# LightGBM — parallel search where concepts overlap, library-native otherwise.
#
# Notes on choices:
#   - num_leaves is the primary complexity knob in LightGBM; max_depth left
#     at the library default (-1 = unlimited) so the two complexity controls
#     don't fight each other.
#   - subsample_freq=1 is REQUIRED — without it LightGBM silently ignores
#     the subsample value (bagging only kicks in when freq > 0).
#   - min_child_samples is the LightGBM analogue of XGBoost's
#     min_child_weight — it controls the minimum rows per leaf.
#   - scale_pos_weight range mirrors the XGBoost study so the class-balance
#     dimension is searched on the same grid → fair head-to-head with Run #6.
# ---------------------------------------------------------------------------


def _suggest_params_lightgbm(trial: optuna.Trial, seed: int) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def _lightgbm_kwargs_from_best(best: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        **best,
        "subsample_freq": 1,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def _build_estimator(model_name: str, kwargs: dict[str, Any]):
    if model_name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(**kwargs)
    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**kwargs)
    raise ValueError(f"Unsupported model for tuning: {model_name!r}")


def _registry_for(model_name: str, task: str) -> tuple[
    Callable[[optuna.Trial, int], dict[str, Any]],
    Callable[[dict[str, Any], int], dict[str, Any]],
]:
    if model_name == "xgboost":
        if task == "delay_cause":
            return _suggest_params_xgboost_cause, _xgboost_cause_kwargs_from_best
        return _suggest_params_xgboost, _xgboost_kwargs_from_best
    if model_name == "lightgbm":
        # LightGBM tuning currently only wired up for delay_binary; cause
        # variant is a TODO (would mirror the xgboost_cause pattern above).
        if task != "delay_binary":
            raise NotImplementedError(
                f"LightGBM tuning for task={task} not implemented; use --model xgboost"
            )
        return _suggest_params_lightgbm, _lightgbm_kwargs_from_best
    raise ValueError(f"Unsupported model for tuning: {model_name!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Optuna-tune a boosting model; one MLflow run per study"
    )
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument(
        "--model",
        default="xgboost",
        choices=SUPPORTED_MODELS,
        help="Boosting library to tune (default: xgboost — preserves Run #6 reproducibility)",
    )
    parser.add_argument(
        "--task",
        default="delay_binary",
        choices=SUPPORTED_TASKS,
        help="Which head to tune (default: delay_binary)",
    )
    parser.add_argument("--feature-set", default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(argv)

    params = load_params()
    seed = params["base"]["random_seed"]
    set_name = args.feature_set or params["features"]["active_set"]
    fs = get_feature_set(set_name)
    suggest_fn, kwargs_from_best = _registry_for(args.model, args.task)
    is_multiclass = args.task == "delay_cause"

    log.info(
        "Tuning %s on feature_set=%s task=%s with %d trials (seed=%d)",
        args.model, set_name, args.task, args.n_trials, seed,
    )

    train_df = pd.read_parquet(params["split"]["train_path"])
    val_df = pd.read_parquet(params["split"]["val_path"])
    x_train, y_train_raw = make_xy(train_df, fs, args.task)
    x_val, y_val_raw = make_xy(val_df, fs, args.task)
    log.info("train=%s val=%s", x_train.shape, x_val.shape)

    label_encoder: LabelEncoder | None = None
    if is_multiclass:
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train_raw)
        y_val = label_encoder.transform(y_val_raw)
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        log.info("Encoded %d classes: %s", len(label_encoder.classes_), list(label_encoder.classes_))
    else:
        y_train = y_train_raw
        y_val = y_val_raw
        sample_weight = None

    preprocessor = build_preprocessor(fs)

    def objective(trial: optuna.Trial) -> float:
        hp = suggest_fn(trial, seed)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("estimator", _build_estimator(args.model, hp)),
            ]
        )
        if sample_weight is not None:
            pipeline.fit(x_train, y_train, estimator__sample_weight=sample_weight)
        else:
            pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_val)
        if is_multiclass:
            return float(f1_score(y_val, y_pred, average="macro", zero_division=0))
        return float(f1_score(y_val, y_pred, zero_division=0))

    sampler = TPESampler(seed=seed)
    study_metric = "macro_f1" if is_multiclass else "f1"
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"{args.model}-{args.task}-{study_metric}",
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    best_params_full = kwargs_from_best(study.best_params, seed)

    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("estimator", _build_estimator(args.model, best_params_full)),
        ]
    )
    if sample_weight is not None:
        final_pipeline.fit(x_train, y_train, estimator__sample_weight=sample_weight)
    else:
        final_pipeline.fit(x_train, y_train)
    y_pred = final_pipeline.predict(x_val)
    if is_multiclass:
        assert label_encoder is not None
        y_proba_mc = final_pipeline.predict_proba(x_val)
        labels_int = list(range(len(label_encoder.classes_)))
        metrics = multiclass_classification_metrics(
            np.asarray(y_val), y_pred, y_proba_mc, labels=labels_int
        )
    else:
        y_proba = final_pipeline.predict_proba(x_val)[:, 1]
        metrics = binary_classification_metrics(y_val.to_numpy(), y_pred, y_proba)
    log.info("Best %s (study): %.4f", study_metric, study.best_value)
    log.info("Final metrics on val: %s", {k: round(v, 4) for k, v in metrics.items()})

    raw_uri = params["mlflow"]["tracking_uri"]
    if raw_uri.startswith("file:") and not raw_uri.startswith("file:/"):
        rel = raw_uri.removeprefix("file:")
        tracking_uri = f"file:{(PROJECT_ROOT / rel).resolve()}"
    else:
        tracking_uri = raw_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    run_name = args.run_name or f"optuna-{args.model}-{args.task}-{set_name}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "git_commit": _git_commit(),
                "dvc_data_hash": _dvc_data_hash(),
                "feature_set": set_name,
                "model": args.model,
                "task": args.task,
                "params_version": params["base"]["params_version"],
                "tuning": "optuna_tpe",
            }
        )
        mlflow.log_params(
            {
                "tuning_method": "optuna_tpe",
                "n_trials": args.n_trials,
                "sampler_seed": seed,
                **{f"best_{k}": v for k, v in study.best_params.items()},
            }
        )
        mlflow.log_metrics({**metrics, "best_trial_number": float(study.best_trial.number)})

        artifact_dir = PROJECT_ROOT / "models"
        artifact_dir.mkdir(exist_ok=True)
        # File naming preserves Run #6 reproducibility — binary tuning still
        # writes best_xgboost_params.yaml; cause tuning gets a qualified name.
        suffix = "" if args.task == "delay_binary" else f"_{args.task}"
        best_yaml = artifact_dir / f"best_{args.model}{suffix}_params.yaml"
        with best_yaml.open("w") as fh:
            yaml.safe_dump(study.best_params, fh, sort_keys=False)
        mlflow.log_artifact(str(best_yaml))

        mlflow.sklearn.log_model(final_pipeline, artifact_path="model")
        log.info("MLflow run id: %s", run.info.run_id)

    return 0


if __name__ == "__main__":
    sys.exit(main())
