"""DVC stage 3 (featurize) — projects raw splits onto the active feature set.

This is the data-axis entry point for the DVC pipeline: it reads the three
processed parquet splits, applies `make_xy` for the feature set + task
declared in params.yaml, and writes per-split X/y parquets plus a manifest.

Why a dedicated stage:
- `dvc dag` shows feature_set as an explicit node between split and train,
  which matches how we describe the experiment in the report (data axis
  before model axis).
- Switching `features.active_set` invalidates only this stage and downstream,
  not ingest/split — so reproductions are cheap.
- The manifest captures the exact column list used; the train and evaluate
  stages read it verbatim, so there's no risk of drift between fit-time and
  inference-time feature lists.

Usage:
    python -m src.models.dvc_featurize
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, load_params
from src.features.build_features import TARGETS, Task, make_xy
from src.features.feature_sets import get_feature_set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("dvc_featurize")

OUT_DIR = PROJECT_ROOT / "data" / "featurized"


def _save_split(name: str, x: pd.DataFrame, y: pd.Series, target_col: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x.to_parquet(OUT_DIR / f"X_{name}.parquet", index=False)
    pd.DataFrame({target_col: y.to_numpy()}).to_parquet(
        OUT_DIR / f"y_{name}.parquet", index=False
    )
    log.info("Saved %s split: X=%s y=%s", name, x.shape, y.shape)


def main() -> int:
    params = load_params()
    set_name: str = params["features"]["active_set"]
    task: Task = params["train"]["task"]
    fs = get_feature_set(set_name)
    target_col = TARGETS[task]

    log.info("feature_set=%s | task=%s | target=%s", set_name, task, target_col)

    train_df = pd.read_parquet(params["split"]["train_path"])
    val_df = pd.read_parquet(params["split"]["val_path"])
    test_df = pd.read_parquet(params["split"]["test_path"])

    x_train, y_train = make_xy(train_df, fs, task)
    x_val, y_val = make_xy(val_df, fs, task)
    x_test, y_test = make_xy(test_df, fs, task)

    _save_split("train", x_train, y_train, target_col)
    _save_split("val", x_val, y_val, target_col)
    _save_split("test", x_test, y_test, target_col)

    manifest = {
        "feature_set": set_name,
        "task": task,
        "target_column": target_col,
        "numeric_columns": list(fs.numeric),
        "categorical_columns": list(fs.categorical),
        "n_train": int(len(x_train)),
        "n_val": int(len(x_val)),
        "n_test": int(len(x_test)),
        "class_balance_train": (
            y_train.value_counts(normalize=True).round(4).to_dict()
            if task != "delay_minutes"
            else None
        ),
    }
    manifest_path = OUT_DIR / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False, default=str)
    log.info("Wrote manifest: %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
