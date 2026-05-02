"""In-memory model loader and predictor.

Models are loaded once at FastAPI startup from the MLflow Model Registry
and cached for the lifetime of the process. Each request only does
`pipeline.predict()` — no disk I/O, no MLflow round-trip.

Why this layer exists separately from main.py:
- Keeps the FastAPI route handlers thin (just I/O glue).
- Makes the loader testable without spinning up the HTTP layer.
- One place to swap loading strategy (registry → run_id → local pkl)
  if/when we move to a remote MLflow server in Docker.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from src.config import PROJECT_ROOT, load_params

log = logging.getLogger("inference")


@dataclass
class LoadedModel:
    """One MLflow registry entry materialised in memory."""

    name: str
    version: str
    run_id: str
    pipeline: Any
    label_classes: list[str] | None = None
    tags: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


class ModelStore:
    """Holds binary + cause models. Singleton-ish, owned by the FastAPI app."""

    def __init__(self) -> None:
        self.binary: LoadedModel | None = None
        self.cause: LoadedModel | None = None

    @property
    def ready(self) -> bool:
        return self.binary is not None and self.cause is not None

    def load_all(self) -> None:
        # Env var wins over params.yaml so docker-compose can point at the
        # mlflow service (http://mlflow:5000) without rewriting config.
        env_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if env_uri:
            tracking_uri = env_uri
        else:
            params = load_params()
            raw_uri = params["mlflow"]["tracking_uri"]
            if raw_uri.startswith("file:") and not raw_uri.startswith("file:/"):
                rel = raw_uri.removeprefix("file:")
                tracking_uri = f"file:{(PROJECT_ROOT / rel).resolve()}"
            else:
                tracking_uri = raw_uri
        mlflow.set_tracking_uri(tracking_uri)
        log.info("MLflow tracking URI: %s", tracking_uri)

        client = MlflowClient()
        self.binary = self._load_one(client, "flight-delay-binary", expects_label_encoder=False)
        self.cause = self._load_one(client, "flight-delay-cause", expects_label_encoder=True)
        log.info("Both models loaded into memory")

    def _load_one(
        self, client: MlflowClient, name: str, expects_label_encoder: bool
    ) -> LoadedModel:
        versions = client.search_model_versions(f"name='{name}'")
        if not versions:
            raise RuntimeError(
                f"No versions registered for {name}. "
                "Run `python -m src.models.registry` first."
            )
        # Pick highest version number (= latest registered)
        latest = max(versions, key=lambda v: int(v.version))
        log.info("Loading %s v%s from run=%s", name, latest.version, latest.run_id[:8])

        model_uri = f"models:/{name}/{latest.version}"
        pipeline = mlflow.sklearn.load_model(model_uri)

        run = client.get_run(latest.run_id)
        tags = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}
        metrics = {k: float(v) for k, v in run.data.metrics.items()}

        label_classes: list[str] | None = None
        if expects_label_encoder:
            label_classes = self._fetch_label_classes(client, latest.run_id)
            log.info("%s label classes: %s", name, label_classes)

        return LoadedModel(
            name=name,
            version=str(latest.version),
            run_id=latest.run_id,
            pipeline=pipeline,
            label_classes=label_classes,
            tags=tags,
            metrics=metrics,
        )

    def _fetch_label_classes(self, client: MlflowClient, run_id: str) -> list[str]:
        """Resolves cause-class names by checking, in order:
            1. MLflow artifact `label_classes_delay_cause.yaml` for the run
               (logged by `src/models/train.py`)
            2. Local fallback `models/label_classes_delay_cause.yaml`
               (always present in this repo; baked into Docker images)

        The fallback exists because Optuna-tuned runs (e.g. C5) are produced
        by `tune.py` which doesn't log the label-classes artifact. Cause-class
        names are constant across all runs of this dataset, so the fallback
        is safe.
        """
        import yaml

        try:
            local_path = client.download_artifacts(
                run_id, "label_classes_delay_cause.yaml"
            )
        except OSError:
            log.warning(
                "Run %s missing label_classes artifact; using local fallback",
                run_id[:8],
            )
            local_path = str(PROJECT_ROOT / "models" / "label_classes_delay_cause.yaml")

        with open(local_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return [str(c) for c in data["classes"]]


_FEATURE_ORDER = [
    "month",
    "day_of_week",
    "scheduled_dep_hour",
    "scheduled_dep_minute",
    "is_weekend",
    "is_holiday_window",
    "quarter",
    "distance_km",
    "planned_block_minutes",
    "airline_fleet_avg_age",
    "origin_hub_tier",
    "destination_hub_tier",
    "inbound_delay_minutes",
    "origin_congestion_index",
    "destination_congestion_index",
    "origin_temperature_c",
    "origin_precip_mm",
    "origin_visibility_km",
    "origin_wind_mps",
    "destination_temperature_c",
    "destination_precip_mm",
    "destination_visibility_km",
    "destination_wind_mps",
    "airline_code",
    "aircraft_family",
    "origin_iata",
    "destination_iata",
    "route_group",
    "origin_weather_severity",
    "destination_weather_severity",
]


def to_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    """Pydantic dict → single-row DataFrame with columns in training order."""
    return pd.DataFrame([{c: payload[c] for c in _FEATURE_ORDER}])


def predict_binary(model: LoadedModel, payload: dict[str, Any]) -> dict[str, Any]:
    df = to_dataframe(payload)
    proba = float(model.pipeline.predict_proba(df)[0, 1])
    pred = bool(proba >= 0.5)
    return {
        "is_delayed": pred,
        "delay_probability": proba,
        "model_name": model.name,
        "model_version": model.version,
    }


def predict_cause(model: LoadedModel, payload: dict[str, Any]) -> dict[str, Any]:
    if model.label_classes is None:
        raise RuntimeError("Cause model loaded without label_classes — check registry")
    df = to_dataframe(payload)
    proba = model.pipeline.predict_proba(df)[0]
    pred_idx = int(np.argmax(proba))
    return {
        "predicted_cause": model.label_classes[pred_idx],
        "class_probabilities": {
            cls: float(p) for cls, p in zip(model.label_classes, proba, strict=True)
        },
        "model_name": model.name,
        "model_version": model.version,
    }
