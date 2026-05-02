"""Feedback sink — closes the ML lifecycle loop.

Real-world outcomes (was the flight actually delayed? what was the real
cause?) are POSTed back to /feedback and appended to a Parquet file.
That file is the seed for the next DVC iteration: bring it into
data/raw/, regenerate splits, retrain — the cycle CLAUDE.md §4 demands.

Storage is a single growing Parquet file under data/feedback/. Path is
overridable via FEEDBACK_STORE env var so tests can use tmp dirs.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import pandas as pd

_LOCK = Lock()


def _store_path() -> Path:
    return Path(os.environ.get("FEEDBACK_STORE", "data/feedback/feedback.parquet"))


def append(record: dict) -> Path:
    """Append a feedback record. Thread-safe; creates parent dir on demand."""
    record = {**record, "received_at": datetime.now(timezone.utc).isoformat()}
    path = _store_path()
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        new_df = pd.DataFrame([record])
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_parquet(path, index=False)
    return path


def count() -> int:
    path = _store_path()
    if not path.exists():
        return 0
    return len(pd.read_parquet(path))
