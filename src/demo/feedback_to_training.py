"""Convert /feedback Parquet into rows ready for the next DVC ingest round.

Closes the lifecycle CLAUDE.md §4 demands: real-world labels collected
via POST /feedback are merged with the original flight features (joined
by request_id against a predictions log) and emitted as a Parquet that
matches the raw dataset schema.

This module is intentionally thin — the full enrichment pipeline (geo
lookups, weather joins, congestion indices) lives in
`src/data/generate.py` for the synthetic source. In a real deployment
the predict middleware would also persist the request payload alongside
the X-Request-ID so this join is local and lossless. For the demo we
take that payload from the original test split using flight_id captured
in the feedback `notes` field.

Usage:
    python -m src.demo.feedback_to_training \
        --feedback data/feedback/feedback.parquet \
        --source data/processed/test.parquet \
        --out data/feedback/next_round.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def merge(feedback_path: Path, source_path: Path) -> pd.DataFrame:
    fb = pd.read_parquet(feedback_path)
    if "notes" not in fb.columns:
        raise ValueError("feedback parquet missing notes column")
    fb["flight_id"] = fb["notes"].str.extract(r"flight_id=(\S+)", expand=False)
    fb = fb.dropna(subset=["flight_id"])
    if fb.empty:
        return fb

    src = pd.read_parquet(source_path)
    joined = src.merge(fb, on="flight_id", how="inner", suffixes=("", "_fb"))

    # Replace the "ground truth" columns with the user-supplied feedback.
    # In a real loop the source rows would NOT have labels yet — this
    # demo overlays feedback onto held-out test rows to show the merge
    # mechanics end-to-end.
    joined["is_departure_delayed_15m"] = joined["actual_is_delayed"].astype(float)
    joined["dep_delay_minutes"] = joined["actual_delay_minutes"]
    joined["probable_delay_cause"] = joined["actual_cause"]
    joined["label_source"] = "feedback"
    joined["label_received_at"] = joined["received_at"]
    return joined.drop(
        columns=[
            "actual_is_delayed",
            "actual_delay_minutes",
            "actual_cause",
            "received_at",
            "request_id",
            "notes",
        ],
        errors="ignore",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feedback", type=Path, default=REPO_ROOT / "data/feedback/feedback.parquet")
    ap.add_argument("--source", type=Path, default=REPO_ROOT / "data/processed/test.parquet")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data/feedback/next_round.parquet")
    args = ap.parse_args()

    if not args.feedback.exists():
        sys.exit(f"missing feedback: {args.feedback}")
    if not args.source.exists():
        sys.exit(f"missing source: {args.source}")

    rows = merge(args.feedback, args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(args.out, index=False)

    print(f"Merged {len(rows)} feedback rows → {args.out}")
    if not rows.empty:
        cause_dist = rows["probable_delay_cause"].value_counts().to_dict()
        delayed = int(rows["is_departure_delayed_15m"].sum())
        print(f"  delayed: {delayed}/{len(rows)}")
        print(f"  cause distribution: {cause_dist}")
        print(
            "\nNext step (manual): `cp data/feedback/next_round.parquet "
            "data/raw/flight_delays_feedback.parquet`, "
            "extend ingest.py to read both files, then `dvc repro`."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
