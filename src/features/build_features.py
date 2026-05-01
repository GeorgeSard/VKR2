"""Materialize a sklearn preprocessor for a chosen feature set, plus
helpers that turn a raw split parquet into (X, y) for a chosen task.

Kept deliberately small: no DVC stage of its own. The active feature set
and target task live in params.yaml, and `train.py` calls these helpers
right before `fit`. That way switching the active feature set produces a
fresh MLflow run with no intermediate parquet to manage.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.feature_sets import FeatureSet, assert_no_leakage

Task = Literal["delay_binary", "delay_minutes", "delay_cause"]

TARGETS: dict[Task, str] = {
    "delay_binary": "is_departure_delayed_15m",
    "delay_minutes": "dep_delay_minutes",
    "delay_cause": "probable_delay_cause",
}


def build_preprocessor(fs: FeatureSet) -> ColumnTransformer:
    """Numeric: median-impute → standard-scale. Categorical: one-hot encode.

    handle_unknown='ignore' is critical: validation/test slices contain
    routes/airports that may not appear in train (the dataset has 22
    airports × 11 carriers — sparse combinations exist). Without it,
    one new combination crashes inference.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, list(fs.numeric)),
            ("cat", categorical_pipe, list(fs.categorical)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_xy(
    df: pd.DataFrame, fs: FeatureSet, task: Task
) -> tuple[pd.DataFrame, pd.Series]:
    """Project the dataframe down to the feature set + the chosen target.

    Drops cancelled flights for delay tasks: their dep_delay is undefined.
    """
    target_col = TARGETS[task]
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe")

    feature_cols = list(fs.all_columns)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Feature set '{fs.name}' references missing columns: {missing}")
    assert_no_leakage(feature_cols)

    work = df
    if task in ("delay_binary", "delay_minutes") and "cancellation_flag" in df.columns:
        before = len(work)
        work = work[work["cancellation_flag"] == 0]
        dropped = before - len(work)
        if dropped:
            # Cancelled rows have no dep_delay — including them would break the target.
            # Logged via print so the train script's logger picks it up downstream.
            pass

    if task == "delay_cause":
        work = work[work[target_col].notna()]

    x = work[feature_cols].copy()
    y = work[target_col].copy()
    return x, y
