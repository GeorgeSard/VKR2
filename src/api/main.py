"""FastAPI application — Stage 8 deployment + Stage 9 observability.

Endpoints:

    GET  /health        — liveness + model-loaded check
    GET  /model/info    — versions, run_ids, training metrics
    POST /predict/delay — binary head: P(delay > 15 min)
    POST /predict/cause — multi-class head: cause distribution
    POST /feedback      — real-world outcome → seed for next DVC iteration
    GET  /metrics       — Prometheus text exposition

Models live in process memory (loaded on startup). Every request is
tagged with an X-Request-ID; that same id is logged on /predict/* and
expected back on /feedback so we can match prediction ↔ ground truth.

Run locally:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from src.api.inference import ModelStore, predict_binary, predict_cause
from src.api.schemas import (
    CausePrediction,
    DelayPrediction,
    FeedbackAck,
    FeedbackRecord,
    FlightFeatures,
    HealthResponse,
    ModelInfo,
    ModelInfoResponse,
)
from src.monitoring import feedback as feedback_store
from src.monitoring import metrics as prom
from src.monitoring.logger import configure as configure_logging
from src.monitoring.logger import log

configure_logging()

app = FastAPI(
    title="Flight Delay Prediction API",
    description=(
        "Two heads on top of the same `with_weather` feature set:\n"
        "- /predict/delay — будет ли рейс задержан (>15 мин)\n"
        "- /predict/cause — какая из 7 причин задержки наиболее вероятна\n\n"
        "Stage 9 adds /metrics (Prometheus) and /feedback (loop closure)."
    ),
    version="0.2.0",
)
store = ModelStore()


@app.on_event("startup")
def _load_models() -> None:
    log.info("Loading models from MLflow Registry...")
    store.load_all()
    log.info("Models loaded", extra={"endpoint": "startup"})


@app.middleware("http")
async def observability(request: Request, call_next):
    """Assign request_id, time the call, emit one structured log + Prometheus."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    request.state.request_id = request_id
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        endpoint = request.url.path
        method = request.method
        prom.REQUESTS.labels(endpoint=endpoint, method=method, status=str(status)).inc()
        prom.LATENCY.labels(endpoint=endpoint, method=method).observe(duration_ms / 1000)
        if endpoint != "/metrics":
            log.info(
                "request",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "path": endpoint,
                    "status": status,
                    "duration_ms": duration_ms,
                },
            )
        # Echo request_id so clients can correlate later /feedback calls.
        if "response" in locals():
            response.headers["x-request-id"] = request_id


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

    return ModelInfoResponse(binary=_to_info(store.binary), cause=_to_info(store.cause))


@app.post("/predict/delay", response_model=DelayPrediction, tags=["predict"])
def predict_delay(features: FlightFeatures, request: Request) -> DelayPrediction:
    if store.binary is None:
        raise HTTPException(status_code=503, detail="Binary model not loaded")
    result = predict_binary(store.binary, features.model_dump())
    outcome = "delayed" if result["is_delayed"] else "on_time"
    prom.PREDICTIONS.labels(model="binary", outcome=outcome).inc()
    log.info(
        "prediction",
        extra={
            "request_id": request.state.request_id,
            "endpoint": "/predict/delay",
            "model": result["model_name"],
            "model_version": result["model_version"],
            "prediction": outcome,
            "probability": round(result["delay_probability"], 4),
        },
    )
    return DelayPrediction(**result)


@app.post("/predict/cause", response_model=CausePrediction, tags=["predict"])
def predict_cause_endpoint(features: FlightFeatures, request: Request) -> CausePrediction:
    if store.cause is None:
        raise HTTPException(status_code=503, detail="Cause model not loaded")
    result = predict_cause(store.cause, features.model_dump())
    cause = result["predicted_cause"]
    prom.PREDICTIONS.labels(model="cause", outcome=cause).inc()
    log.info(
        "prediction",
        extra={
            "request_id": request.state.request_id,
            "endpoint": "/predict/cause",
            "model": result["model_name"],
            "model_version": result["model_version"],
            "prediction": cause,
            "probability": round(result["class_probabilities"][cause], 4),
        },
    )
    return CausePrediction(**result)


@app.post("/feedback", response_model=FeedbackAck, tags=["feedback"])
def submit_feedback(record: FeedbackRecord, request: Request) -> FeedbackAck:
    if all(
        v is None
        for v in (record.actual_is_delayed, record.actual_delay_minutes, record.actual_cause)
    ):
        raise HTTPException(
            status_code=422,
            detail="Provide at least one of actual_is_delayed, actual_delay_minutes, actual_cause",
        )
    feedback_store.append(record.model_dump())
    total = feedback_store.count()
    log.info(
        "feedback",
        extra={
            "request_id": request.state.request_id,
            "endpoint": "/feedback",
            "prediction": record.actual_cause or ("delayed" if record.actual_is_delayed else "on_time"),
        },
    )
    return FeedbackAck(stored=True, total_records=total)


@app.get("/metrics", tags=["meta"])
def metrics() -> Response:
    payload, content_type = prom.render()
    return Response(content=payload, media_type=content_type)
