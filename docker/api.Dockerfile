# Multi-stage Docker image for the FastAPI inference service.
#
# Stage 1 (builder): install Python deps via uv into a self-contained venv.
# Stage 2 (runtime): copy the venv + source. No build toolchain in final
# image → smaller surface, faster cold start.
#
# Build context expected at the repo root:
#     docker build -f docker/api.Dockerfile -t flight-delay-api .
#
# Runtime expects the MLflow tracking URI via env var MLFLOW_TRACKING_URI
# (overrides params.yaml). In docker-compose this points at the mlflow
# service: http://mlflow:5000.

# ---------- builder ----------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libgomp1 is required by xgboost/lightgbm at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv for fast, deterministic dep resolution.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /build
COPY pyproject.toml ./

# Install only the api-relevant subset. We deliberately skip catboost,
# shap, optuna, dvc — they're not needed at inference time. Keeps the
# image ~500 MB lighter.
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install \
        "fastapi>=0.115,<1.0" \
        "uvicorn[standard]>=0.32,<1.0" \
        "pydantic>=2.9,<3.0" \
        "mlflow>=2.17,<3.0" \
        "scikit-learn>=1.5,<2.0" \
        "xgboost>=2.1,<3.0" \
        "pandas>=2.2,<3.0" \
        "numpy>=1.26,<3.0" \
        "pyarrow>=15.0" \
        "pyyaml>=6.0,<7.0"

# ---------- runtime ----------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --shell /bin/bash app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=app:app src/ ./src/
COPY --chown=app:app params.yaml ./
COPY --chown=app:app models/label_classes_delay_cause.yaml ./models/

USER app

EXPOSE 8000

# Healthcheck hits /health — `status: ok` once both models are loaded.
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
