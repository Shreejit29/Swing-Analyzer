from __future__ import annotations

import numpy as np
import pandas as pd

from .technical import add_technical_features
from .price_action import add_price_action_features
from .volume import add_volume_features


BASE_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
}

TARGET_COLUMNS = {
    "target",
    "future_return",
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the complete feature set used by the analyzer and AI model.

    Feature generation is centralized here so Stock Analyzer and
    Backtesting use exactly the same feature pipeline.
    """
    if df is None or df.empty:
        raise ValueError("No OHLCV data supplied for feature generation.")

    required = BASE_COLUMNS

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Feature generation requires: "
            + ", ".join(sorted(missing))
        )

    x = df.copy()

    # Keep the pipeline order consistent.
    x = add_technical_features(x)
    x = add_price_action_features(x)
    x = add_volume_features(x)

    # Convert invalid numeric values to NaN.
    x = x.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return x


def feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return numeric columns that can be used as model features.

    Raw OHLCV columns and target/future columns are excluded to prevent
    the model from directly learning the target or future information.
    """
    if df is None or df.empty:
        return []

    excluded = BASE_COLUMNS | TARGET_COLUMNS

    columns = []

    for column in df.columns:
        if column in excluded:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            columns.append(column)

    return columns
