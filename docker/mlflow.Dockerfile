# Minimal MLflow tracking server image.
#
# Backend + artifact store: file:/mlflow/mlruns (bind-mounted from the
# host's mlruns/ directory in docker-compose). This preserves all 14
# existing runs + the two registered models without re-running training.
#
# Intentionally simpler than production (no Postgres, no S3) per
# CLAUDE.md §5 "минимум software". File backend supports Model Registry
# as of MLflow 2.x, which is enough for this demo.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir "mlflow>=2.17,<3.0"

WORKDIR /mlflow

EXPOSE 5000

# Bind-mounted mlruns/ contains both runs and the model registry. Using
# the same path as backend AND artifact root mirrors the local file:./mlruns
# layout exactly, so artifacts logged via mlflow.sklearn.log_model resolve
# to disk paths that work both in the host and the container.
CMD ["mlflow", "server", \
    "--host", "0.0.0.0", \
    "--port", "5000", \
    "--backend-store-uri", "file:/mlflow/mlruns", \
    "--default-artifact-root", "file:/mlflow/mlruns"]
