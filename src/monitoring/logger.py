"""Structured JSON logger for the inference API.

Why JSON: a single record per request, machine-parseable, ships to
Loki/ELK/CloudWatch without extra parsing. Each log line carries the
request_id so a prediction event can be cross-referenced with a later
/feedback call.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in (
            "request_id",
            "method",
            "path",
            "status",
            "duration_ms",
            "endpoint",
            "model",
            "model_version",
            "prediction",
            "probability",
        ):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    for noisy in ("uvicorn.access",):
        logging.getLogger(noisy).disabled = True


log = logging.getLogger("api")
