"""Stage 8 — MLflow Model Registry helper.

Registers the two production-leader runs (Run #6 binary leader; C5 cause
leader) in MLflow's Model Registry under stable names so the FastAPI
inference layer can load them without hard-coding run_ids.

Names registered:
  flight-delay-binary  → Run #6 (xgboost optuna, with_weather, delay_binary)
  flight-delay-cause   → C5      (xgboost optuna, with_weather, delay_cause)

Idempotent: running twice creates a second model version pointing at the
same run; the latest version wins. To overwrite explicitly, delete the
registered model first via `mlflow models delete-registered-model`.

Usage:
    python -m src.models.registry         # register both leaders
    python -m src.models.registry --check # list current registered models
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from src.config import PROJECT_ROOT, load_params

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("registry")

LEADERS: dict[str, dict[str, str]] = {
    "flight-delay-binary": {
        "run_id": "4227cd7be43c4fa0b1ae2a5e01a5b152",
        "description": "Run #6 — xgboost optuna 30 trials on with_weather + delay_binary. F1=0.630, ROC-AUC=0.839.",
    },
    "flight-delay-cause": {
        "run_id": "400b6695d0204f229107d5c127334410",
        "description": "Cause C5 — xgboost optuna 30 trials on with_weather + delay_cause. macro_f1=0.361, accuracy=0.656.",
    },
}


def _resolve_tracking_uri() -> str:
    params = load_params()
    raw_uri = params["mlflow"]["tracking_uri"]
    if raw_uri.startswith("file:") and not raw_uri.startswith("file:/"):
        rel = raw_uri.removeprefix("file:")
        return f"file:{(PROJECT_ROOT / rel).resolve()}"
    return raw_uri


def register(name: str, run_id: str, description: str) -> None:
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"

    try:
        client.create_registered_model(name, description=description)
        log.info("Created registered model: %s", name)
    except mlflow.exceptions.MlflowException as exc:
        if "already exists" not in str(exc).lower():
            raise
        log.info("Registered model %s already exists — adding new version", name)

    mv = client.create_model_version(
        name=name,
        source=model_uri,
        run_id=run_id,
        description=description,
    )
    log.info("Registered %s version=%s from run=%s", name, mv.version, run_id[:8])


def check() -> None:
    client = MlflowClient()
    for name in LEADERS:
        try:
            versions = client.search_model_versions(f"name='{name}'")
            if not versions:
                log.warning("%s — no versions registered", name)
                continue
            for v in sorted(versions, key=lambda x: int(x.version)):
                log.info(
                    "%s v%s | run=%s | status=%s",
                    name,
                    v.version,
                    v.run_id[:8],
                    v.status,
                )
        except mlflow.exceptions.MlflowException as exc:
            log.warning("%s — not found in registry (%s)", name, exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Register production-leader models in MLflow Registry")
    parser.add_argument("--check", action="store_true", help="List existing registered models and exit")
    args = parser.parse_args(argv)

    tracking_uri = _resolve_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    log.info("MLflow tracking URI: %s", tracking_uri)

    if args.check:
        check()
        return 0

    for name, info in LEADERS.items():
        register(name, info["run_id"], info["description"])

    log.info("Done. Run with --check to list registered versions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
