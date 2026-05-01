"""Stage 3 — ingest.

Read the raw synthetic flight-delays dataset, perform minimal technical
validation (schema, types, duplicates), and write a cleaned interim parquet
that downstream stages (split, features) consume.

Intentionally does NOT do feature engineering or row filtering by domain
logic — that belongs to later stages so we can A/B-test cleaning rules
in MLflow without re-running ingest.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.config import get

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("ingest")

EXPECTED_KEY = "flight_id"
DATETIME_COLS = (
    "flight_date",
    "scheduled_departure_local",
    "scheduled_arrival_local",
    "actual_departure_local",
    "actual_arrival_local",
)


def read_raw(raw_path: Path) -> pd.DataFrame:
    log.info("Reading raw parquet: %s", raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}. "
            "Did you run generate_dataset.py or restore data from DVC remote?"
        )
    df = pd.read_parquet(raw_path)
    log.info("Loaded %d rows × %d cols", len(df), df.shape[1])
    return df


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in DATETIME_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def validate(df: pd.DataFrame) -> None:
    if EXPECTED_KEY not in df.columns:
        raise ValueError(f"Missing primary key column '{EXPECTED_KEY}'")
    n_dupes = df[EXPECTED_KEY].duplicated().sum()
    if n_dupes:
        raise ValueError(f"Duplicate {EXPECTED_KEY}s: {n_dupes}")
    log.info("Validation OK — primary key unique, %d unique flights", df[EXPECTED_KEY].nunique())


def write_interim(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Writing interim parquet: %s", out_path)
    df.to_parquet(out_path, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest raw flight-delay parquet → interim")
    parser.add_argument("--raw", type=Path, default=None, help="Override raw parquet path")
    parser.add_argument("--out", type=Path, default=None, help="Override interim parquet path")
    args = parser.parse_args(argv)

    raw_path = args.raw or Path(get("data", "raw_parquet"))
    out_path = args.out or Path(get("data", "interim_parquet"))

    df = read_raw(raw_path)
    df = coerce_types(df)
    validate(df)
    write_interim(df, out_path)

    log.info("Done. Interim shape: %s", df.shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
