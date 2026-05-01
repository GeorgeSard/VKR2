"""Stage 7 — Optuna tuning of XGBoost on the binary delay task.

One MLflow run for the whole study (not one per trial — that would
flood the UI). The run logs:
  - n_trials, sampler, seed
  - the best hyperparameters discovered (as params best_<name>)
  - all six classification metrics for the refitted best model on val
  - the trained pipeline as an artifact (sklearn flavor)
  - best_params.yaml as a side-artifact, ready to be pasted into
    params.yaml under train.xgboost for the next manual run

Trial-level history is kept inside the Optuna study object; MLflow
sees only the summary so the screenshot story stays one-line-per-run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
import yaml
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

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


def _suggest_params(trial: optuna.Trial, seed: int) -> dict[str, Any]:
    """Search space chosen to span both narrow/deep trees and shallow/wide
    forests, plus a wide scale_pos_weight range so the sampler can re-find
    the class-balance fix from Run #5 if it still helps."""
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optuna-tune XGBoost; one MLflow run per study")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--feature-set", default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(argv)

    params = load_params()
    seed = params["base"]["random_seed"]
    set_name = args.feature_set or params["features"]["active_set"]
    fs = get_feature_set(set_name)

    log.info("Tuning XGBoost on feature_set=%s with %d trials (seed=%d)",
             set_name, args.n_trials, seed)

    train_df = pd.read_parquet(params["split"]["train_path"])
    val_df = pd.read_parquet(params["split"]["val_path"])
    x_train, y_train = make_xy(train_df, fs, "delay_binary")
    x_val, y_val = make_xy(val_df, fs, "delay_binary")
    log.info("train=%s val=%s", x_train.shape, x_val.shape)

    preprocessor = build_preprocessor(fs)

    def objective(trial: optuna.Trial) -> float:
        hp = _suggest_params(trial, seed)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("estimator", XGBClassifier(**hp)),
            ]
        )
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_val)
        score = float(f1_score(y_val, y_pred, zero_division=0))
        return score

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="xgb-f1")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    best_params_full = _suggest_params_from_dict(study.best_params, seed)

    # Refit best on train, score on val for the final summary metrics.
    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("estimator", XGBClassifier(**best_params_full)),
        ]
    )
    final_pipeline.fit(x_train, y_train)
    y_pred = final_pipeline.predict(x_val)
    y_proba = final_pipeline.predict_proba(x_val)[:, 1]
    metrics = binary_classification_metrics(y_val.to_numpy(), y_pred, y_proba)
    log.info("Best F1 (study): %.4f", study.best_value)
    log.info("Final metrics on val: %s", {k: round(v, 4) for k, v in metrics.items()})

    # MLflow logging.
    raw_uri = params["mlflow"]["tracking_uri"]
    if raw_uri.startswith("file:") and not raw_uri.startswith("file:/"):
        rel = raw_uri.removeprefix("file:")
        tracking_uri = f"file:{(PROJECT_ROOT / rel).resolve()}"
    else:
        tracking_uri = raw_uri
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    run_name = args.run_name or f"optuna-xgboost-{set_name}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "git_commit": _git_commit(),
                "dvc_data_hash": _dvc_data_hash(),
                "feature_set": set_name,
                "model": "xgboost",
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

        # Save best params as artifact in MLflow + on-disk so the user can
        # promote them into params.yaml for the next deterministic run.
        artifact_dir = PROJECT_ROOT / "models"
        artifact_dir.mkdir(exist_ok=True)
        best_yaml = artifact_dir / "best_xgboost_params.yaml"
        with best_yaml.open("w") as fh:
            yaml.safe_dump(study.best_params, fh, sort_keys=False)
        mlflow.log_artifact(str(best_yaml))

        mlflow.sklearn.log_model(final_pipeline, artifact_path="model")
        log.info("MLflow run id: %s", run.info.run_id)

    return 0


def _suggest_params_from_dict(d: dict[str, Any], seed: int) -> dict[str, Any]:
    """Reconstruct the full XGBoost kwargs dict from Optuna best_params.
    Mirrors _suggest_params() but skips the trial.suggest_* calls."""
    return {
        **d,
        "random_state": seed,
        "tree_method": "hist",
        "eval_metric": "logloss",
        "n_jobs": -1,
    }


if __name__ == "__main__":
    sys.exit(main())
