"""FastAPI application — Stage 8 deployment surface.

Four endpoints, intentionally minimal (CLAUDE.md §5: "программного
обеспечения должно быть минимум"):

    GET  /health        — liveness + model-loaded check
    GET  /model/info    — versions, run_ids, training metrics
    POST /predict/delay — binary head: P(delay > 15 min)
    POST /predict/cause — multi-class head: cause distribution

Both POST endpoints share `FlightFeatures` schema. Model objects live in
process memory (loaded on startup) — no per-request disk I/O.

Run locally:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

In Docker (compose), the same module is the container entrypoint.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from src.api.inference import ModelStore, predict_binary, predict_cause
from src.api.schemas import (
    CausePrediction,
    DelayPrediction,
    FlightFeatures,
    HealthResponse,
    ModelInfo,
    ModelInfoResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("api")

app = FastAPI(
    title="Flight Delay Prediction API",
    description=(
        "Two heads on top of the same `with_weather` feature set:\n"
        "- /predict/delay — будет ли рейс задержан (>15 мин)\n"
        "- /predict/cause — какая из 7 причин задержки наиболее вероятна\n\n"
        "Модели регистрируются в MLflow Model Registry скриптом "
        "`python -m src.models.registry`."
    ),
    version="0.1.0",
)
store = ModelStore()


@app.on_event("startup")
def _load_models() -> None:
    log.info("Loading models from MLflow Registry...")
    store.load_all()


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if store.ready else "degraded",
        binary_loaded=store.binary is not None,
        cause_loaded=store.cause is not None,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["meta"])
def model_info() -> ModelInfoResponse:
    if not store.ready:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    def _to_info(m) -> ModelInfo:
        return ModelInfo(
            name=m.name,
            version=m.version,
            run_id=m.run_id,
            git_commit=m.tags.get("git_commit"),
            dvc_data_hash=m.tags.get("dvc_data_hash"),
            feature_set=m.tags.get("feature_set"),
            metrics=m.metrics,
        )

    return ModelInfoResponse(
        binary=_to_info(store.binary),
        cause=_to_info(store.cause),
    )


@app.post("/predict/delay", response_model=DelayPrediction, tags=["predict"])
def predict_delay(features: FlightFeatures) -> DelayPrediction:
    if store.binary is None:
        raise HTTPException(status_code=503, detail="Binary model not loaded")
    result = predict_binary(store.binary, features.model_dump())
    return DelayPrediction(**result)


@app.post("/predict/cause", response_model=CausePrediction, tags=["predict"])
def predict_cause_endpoint(features: FlightFeatures) -> CausePrediction:
    if store.cause is None:
        raise HTTPException(status_code=503, detail="Cause model not loaded")
    result = predict_cause(store.cause, features.model_dump())
    return CausePrediction(**result)
