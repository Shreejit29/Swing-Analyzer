"""
Central feature-engineering pipeline.

Pipeline
--------
OHLCV normalization
        ↓
Technical features
        ↓
Price-action features
        ↓
Volume features
        ↓
Regime features
        ↓
Multi-timeframe features
        ↓
Clean model-ready features

Canonical OHLCV columns:
    open
    high
    low
    close
    volume
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.technical import add_technical_features
from src.features.price_action import add_price_action_features
from src.features.volume import add_volume_features
from src.features.regime import add_regime_features
from src.features.multi_timeframe import (
    add_multi_timeframe_features,
)


# ============================================================
# OHLCV NORMALIZATION
# ============================================================

def normalize_ohlcv(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize market-data columns to:

        open
        high
        low
        close
        volume

    Handles common Yahoo Finance MultiIndex layouts and
    case variations.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Input DataFrame is empty."
        )

    out = df.copy()

    # --------------------------------------------------------
    # Flatten MultiIndex columns.
    # --------------------------------------------------------

    if isinstance(
        out.columns,
        pd.MultiIndex,
    ):

        flattened = []

        for column in out.columns:

            parts = [
                str(part).strip()
                for part in column
                if str(part).strip()
                and str(part).lower() != "nan"
            ]

            flattened.append(
                "_".join(parts)
            )

        out.columns = flattened

    # --------------------------------------------------------
    # Build normalized column lookup.
    # --------------------------------------------------------

    normalized = {}

    for column in out.columns:

        clean = (
            str(column)
            .strip()
            .lower()
        )

        clean = (
            clean
            .replace("-", "_")
            .replace(" ", "_")
        )

        normalized[column] = clean

    # --------------------------------------------------------
    # Identify OHLCV columns.
    # --------------------------------------------------------

    aliases = {
        "open": [
            "open",
            "open_price",
        ],
        "high": [
            "high",
            "high_price",
        ],
        "low": [
            "low",
            "low_price",
        ],
        "close": [
            "close",
            "close_price",
            "adj_close",
            "adjusted_close",
        ],
        "volume": [
            "volume",
            "vol",
        ],
    }

    selected = {}

    for target, possible_names in aliases.items():

        # Exact match first.
        for original, clean in normalized.items():

            if clean in possible_names:

                selected[target] = original
                break

        if target in selected:
            continue

        # Yahoo flattened names such as:
        # close_reliance.ns
        # reliance.ns_close
        for original, clean in normalized.items():

            if any(
                clean.startswith(
                    name + "_"
                )
                or clean.endswith(
                    "_" + name
                )
                for name in possible_names
            ):

                selected[target] = original
                break

    missing = [
        column
        for column in [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        if column not in selected
    ]

    if missing:

        raise ValueError(
            "Missing required OHLCV columns: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # Create canonical dataframe.
    # --------------------------------------------------------

    result = pd.DataFrame(
        index=out.index
    )

    for target in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:

        value = out[
            selected[target]
        ]

        # Protect against duplicate columns.
        if isinstance(
            value,
            pd.DataFrame,
        ):
            value = value.iloc[:, 0]

        result[target] = pd.to_numeric(
            value,
            errors="coerce",
        )

    # --------------------------------------------------------
    # Remove invalid rows.
    # --------------------------------------------------------

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    result = result.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    )

    # --------------------------------------------------------
    # Remove duplicate timestamps if possible.
    # --------------------------------------------------------

    if result.index.has_duplicates:

        result = result[
            ~result.index.duplicated(
                keep="last"
            )
        ]

    return result


# ============================================================
# FEATURE PIPELINE
# ============================================================

def build_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the complete model feature set.

    Order:

        1. Normalize OHLCV
        2. Technical indicators
        3. Price action
        4. Volume
        5. Market regime
        6. Multi-timeframe trend
        7. Clean numerical values

    Returns
    -------
    pandas.DataFrame
        Feature-engineered market data.
    """

    data = normalize_ohlcv(
        df
    )

    # --------------------------------------------------------
    # Technical indicators
    # --------------------------------------------------------

    data = add_technical_features(
        data
    )

    # --------------------------------------------------------
    # Price action
    # --------------------------------------------------------

    data = add_price_action_features(
        data
    )

    # --------------------------------------------------------
    # Volume analysis
    # --------------------------------------------------------

    data = add_volume_features(
        data
    )

    # --------------------------------------------------------
    # Market regime
    # --------------------------------------------------------

    data = add_regime_features(
        data
    )

    # --------------------------------------------------------
    # Multi-timeframe analysis
    # --------------------------------------------------------

    data = add_multi_timeframe_features(
        data
    )

    # --------------------------------------------------------
    # Final numerical cleanup
    # --------------------------------------------------------

    numeric_columns = (
        data
        .select_dtypes(
            include=[np.number]
        )
        .columns
    )

    data[numeric_columns] = (
        data[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    return data


# ============================================================
# MODEL FEATURE SELECTION
# ============================================================

def feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Return numerical columns suitable for ML.

    Raw OHLCV columns and target/leakage columns are excluded.
    """

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    excluded = {
        "open",
        "high",
        "low",
        "close",
        "volume",

        # Targets / future information.
        "target",
        "future_return",
        "future_close",
        "label",

        # Date/time fields.
        "date",
        "datetime",
        "timestamp",
    }

    columns = []

    for column in df.columns:

        name = str(
            column
        ).strip().lower()

        if name in excluded:
            continue

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            continue

        columns.append(
            column
        )

    return columns


# ============================================================
# MODEL-READY DATA
# ============================================================

def prepare_model_data(
    df: pd.DataFrame,
    target_column: str = "target",
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Prepare X, y and feature names for machine learning.

    Rows containing missing values in selected features
    or target are removed.
    """

    data = build_features(
        df
    )

    columns = feature_columns(
        data
    )

    if not columns:

        raise ValueError(
            "No usable numerical features were generated."
        )

    if target_column not in data.columns:

        raise ValueError(
            f"Target column '{target_column}' "
            "was not found."
        )

    X = data[
        columns
    ].copy()

    y = data[
        target_column
    ].copy()

    # --------------------------------------------------------
    # Remove invalid numerical values.
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    valid = (
        X.notna().all(axis=1)
        & y.notna()
    )

    X = X.loc[
        valid
    ].copy()

    y = y.loc[
        valid
    ].copy()

    return X, y, columns


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "normalize_ohlcv",
    "build_features",
    "feature_columns",
    "prepare_model_data",
    ]
