"""Shared preprocessing helpers for training and inference.

Mirrors behavior from the legacy local training script:
    * Numeric columns -> ``float32``
    * Non-numeric columns -> categorical codes -> ``float32``
    * Remaining ``NaN`` values filled with ``0.0``
"""

from __future__ import annotations
from typing import Tuple, Set, Iterable

import pandas as pd


# Columns that are never features
NON_FEATURE_COLS_BASE: Set[str] = {
    "user_id", "date", "split", "fold", "label", "run_id",
}


def drop_meta_cols(df: pd.DataFrame, extra_drop: Iterable[str] | None = None) -> pd.DataFrame:
    """
    Drop id/meta columns, anything ending with '_id', and provided extras.
    Matches the intent from model_training_local4.py.
    """
    dynamic_id_cols = {c for c in df.columns if c.lower().endswith("_id")}
    drop_cols = set(NON_FEATURE_COLS_BASE) | dynamic_id_cols
    drop_cols |= {c for c in df.columns if c.startswith("y_")}
    if extra_drop:
        drop_cols |= set(extra_drop)
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")


def prepare_xy_compat(df: pd.DataFrame, target_col: str = "y") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Exact replica of feature prep used in model_training_local4.py:

      - Build X, y.
      - Drops id/meta columns and anything ending with '_id'.
      - Drops all y_* columns from features.
      - Cast numerics to float32, categories to codes (then float32).
      - FillNA with 0.0.

    Raises:
        ValueError if target_col is missing.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target {target_col} not in dataframe.")

    # 1) Drop meta/id/y_* columns from features
    dynamic_id_cols = {c for c in df.columns if c.lower().endswith("_id")}
    drop_cols = set(NON_FEATURE_COLS_BASE) | dynamic_id_cols
    drop_cols |= {c for c in df.columns if c.startswith("y_")}
    feat_cols = [c for c in df.columns if c not in drop_cols and c != target_col]

    # 2) Cast numerics to float32; non-numerics to categorical codes -> float32
    X = df[feat_cols].copy()
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
        else:
            X[c] = X[c].astype("category").cat.codes.astype("float32")

    # 3) Fill any remaining NaNs
    X = X.fillna(0.0)

    # 4) y as Int64 then float32 to match original pipeline
    y = df[target_col].astype("Int64").astype("float32")
    return X, y
