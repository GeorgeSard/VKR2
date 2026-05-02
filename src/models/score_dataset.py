"""Final deliverable — score the test split with both ML heads and emit
a human-readable parquet/CSV/Excel triplet for the report.

Trains the two head winners directly (rather than reconstructing them
from MLflow run artifacts) so this script is fully self-contained and
re-runnable from any clean checkout that has data via DVC:

    head A (delay_binary)  — XGBoost with Run #6 Optuna config
                             (models/best_xgboost_params.yaml)
    head B (delay_cause)   — XGBoost with Cause C5 Optuna config
                             (models/best_xgboost_delay_cause_params.yaml)
                             + balanced sample_weight at fit time
                             Falls back to C4 defaults if C5 file absent.

Output goes to `reports/scored_test_dataset.{parquet,csv,xlsx}`.
The Excel and CSV variants project the dataset down to the columns a
human reader actually wants to see — flight identifiers, schedule,
ground truth and predictions side-by-side — instead of the full 60+
internal columns.

Usage:
    python -m src.models.score_dataset
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from src.config import load_params
from src.features.build_features import make_xy, build_preprocessor
from src.features.feature_sets import get_feature_set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("score_dataset")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "reports"

# Russian display labels for the seven cause classes — what the human
# reader sees in the Excel column. The English snake_case stays in the
# parquet/CSV machine-readable columns for downstream tooling.
CAUSE_LABELS_RU: dict[str, str] = {
    "none": "без задержки",
    "weather": "погода",
    "carrier_operational": "авиакомпания (операционные)",
    "reactionary": "каскадная (от прошлого рейса)",
    "airport_congestion": "загруженность аэропорта",
    "cancelled": "отменён",
    "security": "безопасность",
}


def _load_xgb_best() -> dict:
    """Load Run #6 Optuna best params from disk."""
    p = PROJECT_ROOT / "models" / "best_xgboost_params.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"Run #6 best params file not found at {p}. "
            "Either re-run `python -m src.models.tune --n-trials 30` "
            "or pull the artifact from MLflow run 4227cd7b."
        )
    with p.open() as fh:
        return yaml.safe_load(fh)


def _train_binary_head(
    train_df: pd.DataFrame, seed: int
) -> Pipeline:
    """Head A — XGBoost binary delay predictor (Run #6 Optuna config)."""
    from xgboost import XGBClassifier

    fs = get_feature_set("with_weather")
    x_train, y_train = make_xy(train_df, fs, "delay_binary")
    log.info("[head A] fit on %d rows, %d features", len(x_train), len(fs.all_columns))

    best = _load_xgb_best()
    estimator = XGBClassifier(
        n_estimators=best["n_estimators"],
        max_depth=best["max_depth"],
        learning_rate=best["learning_rate"],
        subsample=best["subsample"],
        colsample_bytree=best["colsample_bytree"],
        min_child_weight=best["min_child_weight"],
        scale_pos_weight=best["scale_pos_weight"],
        random_state=seed,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=-1,
    )
    pipeline = Pipeline(
        steps=[("preprocessor", build_preprocessor(fs)), ("estimator", estimator)]
    )
    pipeline.fit(x_train, y_train)
    return pipeline


def _load_xgb_cause_best() -> dict | None:
    """Load Cause C5 Optuna best params if available; None falls back to C4 defaults."""
    p = PROJECT_ROOT / "models" / "best_xgboost_delay_cause_params.yaml"
    if not p.exists():
        log.warning("Cause C5 params file not found at %s — falling back to C4 defaults", p)
        return None
    with p.open() as fh:
        return yaml.safe_load(fh)


def _train_cause_head(
    train_df: pd.DataFrame, seed: int
) -> tuple[Pipeline, LabelEncoder]:
    """Head B — XGBoost multi-class cause classifier.

    Defaults to Cause C5 Optuna config; falls back to C4 hand-defaults
    when the Optuna best-params file is absent (e.g. fresh checkout
    before tuning has run).
    """
    from xgboost import XGBClassifier

    fs = get_feature_set("with_weather")
    x_train, y_train = make_xy(train_df, fs, "delay_cause")
    log.info("[head B] fit on %d rows, %d features", len(x_train), len(fs.all_columns))

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    sw = compute_sample_weight(class_weight="balanced", y=y_train_enc)

    best = _load_xgb_cause_best()
    if best is not None:
        log.info("[head B] using C5 Optuna config: %s", best)
        estimator = XGBClassifier(
            n_estimators=best["n_estimators"],
            max_depth=best["max_depth"],
            learning_rate=best["learning_rate"],
            subsample=best["subsample"],
            colsample_bytree=best["colsample_bytree"],
            min_child_weight=best["min_child_weight"],
            random_state=seed,
            tree_method="hist",
            eval_metric="mlogloss",
            n_jobs=-1,
        )
    else:
        log.info("[head B] using C4 hand-defaults")
        estimator = XGBClassifier(
            n_estimators=800,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            tree_method="hist",
            eval_metric="mlogloss",
            n_jobs=-1,
        )
    pipeline = Pipeline(
        steps=[("preprocessor", build_preprocessor(fs)), ("estimator", estimator)]
    )
    pipeline.fit(x_train, y_train_enc, estimator__sample_weight=sw)
    return pipeline, label_encoder


def _score(
    test_df: pd.DataFrame,
    binary_pipeline: Pipeline,
    cause_pipeline: Pipeline,
    cause_label_encoder: LabelEncoder,
) -> pd.DataFrame:
    fs = get_feature_set("with_weather")

    # head A — binary
    # We score every row except cancelled flights for which dep_delay is
    # undefined; predictions for cancelled rows are tagged separately.
    is_cancelled = test_df["cancellation_flag"] == 1
    feature_cols = list(fs.all_columns)
    x_test_all = test_df[feature_cols]

    binary_proba = binary_pipeline.predict_proba(x_test_all)[:, 1]
    binary_pred = (binary_proba >= 0.5).astype(int)

    # head B — multi-class cause
    cause_proba_matrix = cause_pipeline.predict_proba(x_test_all)
    cause_pred_idx = cause_proba_matrix.argmax(axis=1)
    cause_classes = cause_label_encoder.classes_
    cause_pred = cause_classes[cause_pred_idx]
    cause_top_proba = cause_proba_matrix[np.arange(len(cause_pred_idx)), cause_pred_idx]

    # Mask cancelled-flight predictions for head A — meaningless when the
    # flight didn't depart. Keep cause prediction (cancelled IS a cause class).
    binary_pred_str = np.where(is_cancelled, "—", np.where(binary_pred == 1, "задержан", "вовремя"))
    binary_proba_display = np.where(is_cancelled, np.nan, binary_proba)

    cause_pred_ru = pd.Series(cause_pred).map(CAUSE_LABELS_RU).to_numpy()

    out = pd.DataFrame(
        {
            # identifiers
            "flight_id": test_df["flight_id"].to_numpy(),
            "flight_date": test_df["flight_date"].to_numpy(),
            "airline_code": test_df["airline_code"].to_numpy(),
            "airline_name": test_df["airline_name"].to_numpy(),
            "flight_number": test_df["flight_number"].to_numpy(),
            # schedule context
            "origin_iata": test_df["origin_iata"].to_numpy(),
            "origin_city": test_df["origin_city"].to_numpy(),
            "destination_iata": test_df["destination_iata"].to_numpy(),
            "destination_city": test_df["destination_city"].to_numpy(),
            "scheduled_departure_local": test_df["scheduled_departure_local"].to_numpy(),
            "distance_km": test_df["distance_km"].to_numpy(),
            "aircraft_family": test_df["aircraft_family"].to_numpy(),
            # ground truth (for inspection / report; would be hidden in production)
            "actual_delay_minutes": test_df["dep_delay_minutes"].to_numpy(),
            "actual_is_delayed": np.where(
                is_cancelled,
                "—",
                np.where(test_df["is_departure_delayed_15m"] == 1, "задержан", "вовремя"),
            ),
            "actual_cause": pd.Series(test_df["probable_delay_cause"].to_numpy())
                .map(CAUSE_LABELS_RU)
                .to_numpy(),
            # head A predictions
            "predicted_delay": binary_pred_str,
            "delay_probability": np.round(binary_proba_display, 3),
            # head B predictions
            "predicted_cause": cause_pred_ru,
            "cause_confidence": np.round(cause_top_proba, 3),
            # eval helpers
            "binary_correct": np.where(
                is_cancelled,
                "н/п",  # неприменимо
                np.where(binary_pred == test_df["is_departure_delayed_15m"].to_numpy(), "✓", "✗"),
            ),
            "cause_correct": np.where(
                cause_pred == test_df["probable_delay_cause"].to_numpy(), "✓", "✗"
            ),
        }
    )
    return out


def _write_outputs(scored: pd.DataFrame) -> dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parquet_path = OUTPUT_DIR / "scored_test_dataset.parquet"
    csv_path = OUTPUT_DIR / "scored_test_dataset.csv"
    xlsx_path = OUTPUT_DIR / "scored_test_dataset.xlsx"

    scored.to_parquet(parquet_path, index=False)
    # Use ; separator + UTF-8 BOM so Excel opens it directly without garbling Russian text.
    scored.to_csv(csv_path, index=False, sep=";", encoding="utf-8-sig")

    # Excel: full sheet + a 100-row preview sheet for quick eyeballing.
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        scored.to_excel(writer, sheet_name="scored_test", index=False)
        scored.head(100).to_excel(writer, sheet_name="preview_first_100", index=False)
        # Confusion-style summary sheet — counts per (actual_cause, predicted_cause)
        confusion = (
            scored.pivot_table(
                index="actual_cause",
                columns="predicted_cause",
                values="flight_id",
                aggfunc="count",
                fill_value=0,
            )
        )
        confusion.to_excel(writer, sheet_name="cause_confusion")

    return {"parquet": parquet_path, "csv": csv_path, "xlsx": xlsx_path}


def _summary(scored: pd.DataFrame) -> dict[str, float | int]:
    n = len(scored)
    binary_evaluable = scored["actual_is_delayed"] != "—"
    binary_acc = float((scored.loc[binary_evaluable, "binary_correct"] == "✓").mean())
    cause_acc = float((scored["cause_correct"] == "✓").mean())
    return {
        "n_rows": n,
        "binary_accuracy_on_evaluable": round(binary_acc, 4),
        "cause_accuracy_overall": round(cause_acc, 4),
    }


def main() -> int:
    params = load_params()
    seed = params["base"]["random_seed"]

    train_df = pd.read_parquet(params["split"]["train_path"])
    test_df = pd.read_parquet(params["split"]["test_path"])
    log.info("Loaded train=%d test=%d rows", len(train_df), len(test_df))

    binary_pipeline = _train_binary_head(train_df, seed)
    cause_pipeline, cause_label_encoder = _train_cause_head(train_df, seed)

    scored = _score(test_df, binary_pipeline, cause_pipeline, cause_label_encoder)
    log.info("Scored %d test rows", len(scored))

    paths = _write_outputs(scored)
    for fmt, path in paths.items():
        size_kb = path.stat().st_size / 1024
        log.info("Wrote %s: %s (%.0f KB)", fmt, path.relative_to(PROJECT_ROOT), size_kb)

    log.info("Headline numbers: %s", _summary(scored))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
