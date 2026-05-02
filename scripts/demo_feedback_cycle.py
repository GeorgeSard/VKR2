"""End-to-end demo of the closed ML loop (CLAUDE.md §4 / Stage 10).

Picks N random flights from the held-out test split, sends each to both
predict endpoints, then posts the real outcome (from the same test row)
to /feedback. Finally prints batch accuracy and reads back the persisted
feedback Parquet — proving the full cycle is wired:

    test row → POST /predict/* → X-Request-ID
                                       │
                                       ▼
                  POST /feedback (actual labels) ──► data/feedback/feedback.parquet
                                                              │
                                                              └─► next dvc repro round

Run after `docker compose up -d`:

    python scripts/demo_feedback_cycle.py --n 20

Defaults to localhost:8000. Use --base-url to point at another host.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.api.schemas import FlightFeatures  # noqa: E402

FEATURE_FIELDS = list(FlightFeatures.model_fields.keys())


def pick_flights(n: int, seed: int) -> pd.DataFrame:
    test_path = REPO_ROOT / "data" / "processed" / "test.parquet"
    if not test_path.exists():
        sys.exit(f"missing {test_path} — run `dvc pull` or `dvc repro split`")
    df = pd.read_parquet(test_path)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def to_payload(row: pd.Series) -> dict:
    return {f: (row[f].item() if hasattr(row[f], "item") else row[f]) for f in FEATURE_FIELDS}


def run(base_url: str, n: int, seed: int) -> int:
    sample = pick_flights(n, seed)
    print(f"Sampled {n} test rows (seed={seed}). Hitting {base_url} ...\n")

    binary_correct = 0
    cause_correct = 0
    posted = 0

    for i, row in sample.iterrows():
        payload = to_payload(row)
        actual_delayed = bool(row["is_departure_delayed_15m"])
        actual_cause = str(row["probable_delay_cause"])
        actual_minutes = float(row["dep_delay_minutes"])

        r1 = requests.post(f"{base_url}/predict/delay", json=payload, timeout=10)
        r1.raise_for_status()
        rid = r1.headers["x-request-id"]
        d_pred = r1.json()
        binary_correct += int(d_pred["is_delayed"] == actual_delayed)

        r2 = requests.post(
            f"{base_url}/predict/cause",
            json=payload,
            headers={"X-Request-ID": rid},
            timeout=10,
        )
        r2.raise_for_status()
        c_pred = r2.json()
        cause_correct += int(c_pred["predicted_cause"] == actual_cause)

        fb = requests.post(
            f"{base_url}/feedback",
            json={
                "request_id": rid,
                "actual_is_delayed": actual_delayed,
                "actual_delay_minutes": actual_minutes,
                "actual_cause": actual_cause,
                "notes": f"demo row flight_id={row['flight_id']}",
            },
            headers={"X-Request-ID": rid},
            timeout=10,
        )
        fb.raise_for_status()
        posted += 1

        marker_b = "✓" if d_pred["is_delayed"] == actual_delayed else "✗"
        marker_c = "✓" if c_pred["predicted_cause"] == actual_cause else "✗"
        print(
            f"  [{i+1:>2}/{n}] {row['airline_code']} "
            f"{row['origin_iata']}→{row['destination_iata']}  "
            f"binary {marker_b} (pred={d_pred['is_delayed']!s:5} p={d_pred['delay_probability']:.2f}, "
            f"actual={actual_delayed!s:5})  "
            f"cause  {marker_c} (pred={c_pred['predicted_cause']:>20}, actual={actual_cause})"
        )

    print()
    print(f"Posted {posted} feedback records.")
    print(
        f"Binary head accuracy on this batch: {binary_correct}/{n} "
        f"= {binary_correct / n:.1%}"
    )
    print(
        f"Cause  head accuracy on this batch: {cause_correct}/{n} "
        f"= {cause_correct / n:.1%}"
    )

    fb_path = REPO_ROOT / "data" / "feedback" / "feedback.parquet"
    if fb_path.exists():
        stored = pd.read_parquet(fb_path)
        print(f"\nFeedback parquet: {fb_path} ({len(stored)} total rows)")
        print(stored.tail(3).to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20, help="number of flights to send (default 20)")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed (default 42)")
    ap.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="API base URL (default http://localhost:8000)",
    )
    args = ap.parse_args()
    return run(args.base_url, args.n, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
