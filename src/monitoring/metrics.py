"""Prometheus metrics for the inference API.

Three families of metrics, all labelled by endpoint:

  api_requests_total{endpoint,method,status}    counter
  api_request_duration_seconds{endpoint,method} histogram
  api_predictions_total{model,outcome}          counter — distribution of
      predicted classes (binary: delayed/on_time, cause: 7 classes).
      Lets us spot model drift on the dashboard without storing each
      individual prediction.

Exposed at GET /metrics in Prometheus text format.
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

REGISTRY = CollectorRegistry()

REQUESTS = Counter(
    "api_requests_total",
    "Total HTTP requests handled by the API",
    ["endpoint", "method", "status"],
    registry=REGISTRY,
)

LATENCY = Histogram(
    "api_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint", "method"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=REGISTRY,
)

PREDICTIONS = Counter(
    "api_predictions_total",
    "Predictions emitted by each model, broken down by outcome",
    ["model", "outcome"],
    registry=REGISTRY,
)


def render() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
