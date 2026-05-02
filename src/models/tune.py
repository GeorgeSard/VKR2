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
import optuna
import pandas as pd
import yaml
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from src.config import PROJECT_ROOT, load_params
from src.features.build_features import build_preprocessor, make_xy
from src.features.feature_sets import get_feature_set
from src.models.evaluate import binary_classification_metrics
from src.models.train import _dvc_data_hash, _git_commit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("tune")

SUPPORTED_MODELS = ("xgboost", "lightgbm")


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


def _registry_for(model_name: str) -> tuple[
    Callable[[optuna.Trial, int], dict[str, Any]],
    Callable[[dict[str, Any], int], dict[str, Any]],
]:
    if model_name == "xgboost":
        return _suggest_params_xgboost, _xgboost_kwargs_from_best
    if model_name == "lightgbm":
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
    parser.add_argument("--feature-set", default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(argv)

    params = load_params()
    seed = params["base"]["random_seed"]
    set_name = args.feature_set or params["features"]["active_set"]
    fs = get_feature_set(set_name)
    suggest_fn, kwargs_from_best = _registry_for(args.model)

    log.info(
        "Tuning %s on feature_set=%s with %d trials (seed=%d)",
        args.model, set_name, args.n_trials, seed,
    )

    train_df = pd.read_parquet(params["split"]["train_path"])
    val_df = pd.read_parquet(params["split"]["val_path"])
    x_train, y_train = make_xy(train_df, fs, "delay_binary")
    x_val, y_val = make_xy(val_df, fs, "delay_binary")
    log.info("train=%s val=%s", x_train.shape, x_val.shape)

    preprocessor = build_preprocessor(fs)

    def objective(trial: optuna.Trial) -> float:
        hp = suggest_fn(trial, seed)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("estimator", _build_estimator(args.model, hp)),
            ]
        )
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_val)
        score = float(f1_score(y_val, y_pred, zero_division=0))
        return score

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize", sampler=sampler, study_name=f"{args.model}-f1"
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
    final_pipeline.fit(x_train, y_train)
    y_pred = final_pipeline.predict(x_val)
    y_proba = final_pipeline.predict_proba(x_val)[:, 1]
    metrics = binary_classification_metrics(y_val.to_numpy(), y_pred, y_proba)
    log.info("Best F1 (study): %.4f", study.best_value)
    log.info("Final metrics on val: %s", {k: round(v, 4) for k, v in metrics.items()})

    raw_uri = params["mlflow"]["tracking_uri"]
    if raw_uri.startswith("file:") and not raw_uri.startswith("file:/"):
        rel = raw_uri.removeprefix("file:")
        tracking_uri = f"file:{(PROJECT_ROOT / rel).resolve()}"
    else:
        tracking_uri = raw_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    run_name = args.run_name or f"optuna-{args.model}-{set_name}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "git_commit": _git_commit(),
                "dvc_data_hash": _dvc_data_hash(),
                "feature_set": set_name,
                "model": args.model,
                "task": "delay_binary",
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
        best_yaml = artifact_dir / f"best_{args.model}_params.yaml"
        with best_yaml.open("w") as fh:
            yaml.safe_dump(study.best_params, fh, sort_keys=False)
        mlflow.log_artifact(str(best_yaml))

        mlflow.sklearn.log_model(final_pipeline, artifact_path="model")
        log.info("MLflow run id: %s", run.info.run_id)

    return 0


if __name__ == "__main__":
    sys.exit(main())
