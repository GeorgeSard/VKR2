"""Stage 3 — split.

Time-based train/val/test split. We split by `flight_date` rather than at
random because the dataset has concept drift baked in (security delays
escalate in 2024–2025) and a random split would let the model peek at the
future, inflating metrics in a way that does not survive deployment.

Boundaries are configured in params.yaml → split.train_end_date / val_end_date.
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
log = logging.getLogger("split")


def time_based_split(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    date_col: str = "flight_date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    train = df[df[date_col] <= train_end]
    val = df[(df[date_col] > train_end) & (df[date_col] <= val_end)]
    test = df[df[date_col] > val_end]
    return train, val, test


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Time-based train/val/test split")
    parser.add_argument("--input", type=Path, default=None)
    args = parser.parse_args(argv)

    in_path = args.input or Path(get("data", "interim_parquet"))
    split_cfg = get("split")

    train_end = pd.Timestamp(split_cfg["train_end_date"])
    val_end = pd.Timestamp(split_cfg["val_end_date"])
    train_path = Path(split_cfg["train_path"])
    val_path = Path(split_cfg["val_path"])
    test_path = Path(split_cfg["test_path"])

    log.info("Reading %s", in_path)
    df = pd.read_parquet(in_path)

    train, val, test = time_based_split(df, train_end, val_end)
    log.info(
        "Split (≤%s | ≤%s | >%s): train=%d, val=%d, test=%d",
        train_end.date(),
        val_end.date(),
        val_end.date(),
        len(train),
        len(val),
        len(test),
    )

    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("One of the splits is empty — check date boundaries in params.yaml")

    for path, frame in ((train_path, train), (val_path, val), (test_path, test)):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        log.info("Wrote %s (%d rows)", path, len(frame))

    return 0


if __name__ == "__main__":
    sys.exit(main())
