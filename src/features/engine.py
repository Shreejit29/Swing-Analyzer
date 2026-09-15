from __future__ import annotations

import numpy as np
import pandas as pd

from .technical import add_technical_features
from .price_action import add_price_action_features
from .volume import add_volume_features


BASE_COLUMNS = {"open", "high", "low", "close", "volume"}
TARGET_COLUMNS = {"target", "future_return"}


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Yahoo Finance MultiIndex columns into simple OHLCV names."""
    x = df.copy()

    if isinstance(x.columns, pd.MultiIndex):
        market_names = {
            "open", "high", "low", "close",
            "adj close", "volume"
        }

        new_columns = []
        for col in x.columns:
            parts = [str(v).strip() for v in col
                     if str(v).strip().lower() not in {"", "nan", "none"}]

            selected = None
            for part in parts:
                if part.lower() in market_names:
                    selected = part
                    break

            new_columns.append(selected if selected else "_".join(parts))

        x.columns = new_columns

    return x


def _find_column(df: pd.DataFrame, aliases: set[str]):
    """Find an OHLCV column case-insensitively, including flattened Yahoo names."""
    aliases = {a.lower() for a in aliases}

    for column in df.columns:
        name = str(column).strip().lower()

        if name in aliases:
            value = df[column]
            if isinstance(value, pd.DataFrame):
                value = value.iloc[:, 0]
            return value

    for column in df.columns:
        name = str(column).strip().lower()

        if any(
            name.startswith(alias + "_") or name.endswith("_" + alias)
            for alias in aliases
        ):
            value = df[column]
            if isinstance(value, pd.DataFrame):
                value = value.iloc[:, 0]
            return value

    return None


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize all incoming market data to exactly:

        open, high, low, close, volume

    This is the single OHLCV contract used by the complete application.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("No OHLCV data supplied for feature generation.")

    x = _flatten_columns(df)

    normalized = pd.DataFrame(index=x.index)

    aliases = {
        "open": {"open"},
        "high": {"high"},
        "low": {"low"},
        "close": {"close", "adj close"},
        "volume": {"volume", "vol"},
    }

    missing = []

    for target, names in aliases.items():
        value = _find_column(x, names)

        if value is None:
            missing.append(target)
        else:
            # Force a genuine 1-D Series.
            normalized[target] = pd.Series(
                value,
                index=x.index,
            ).squeeze()

    if missing:
        raise ValueError(
            "OHLCV normalization failed. Missing: "
            + ", ".join(missing)
            + f". Available columns: {list(x.columns)[:20]}"
        )

    for column in BASE_COLUMNS:
        normalized[column] = pd.to_numeric(
            normalized[column],
            errors="coerce",
        )

    # Preserve only valid market rows.
    normalized = normalized.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    normalized = normalized.dropna(
        subset=["open", "high", "low", "close"]
    )

    if normalized.empty:
        raise ValueError("OHLCV data contains no valid price rows.")

    return normalized


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Central feature pipeline.

    Data flow:
        raw Yahoo/CSV data
        -> OHLCV normalization
        -> technical features
        -> price-action features
        -> volume features
        -> cleaned feature dataframe
    """
    x = normalize_ohlcv(df)

    x = add_technical_features(x)
    x = add_price_action_features(x)
    x = add_volume_features(x)

    x = x.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return x


def feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Return numeric model features while excluding raw OHLCV and targets.
    """
    if df is None or df.empty:
        return []

    excluded = BASE_COLUMNS | TARGET_COLUMNS
    columns = []

    for column in df.columns:
        if str(column).lower() in excluded:
            continue

        value = df[column]

        # Duplicate column names can return a DataFrame.
        if isinstance(value, pd.DataFrame):
            continue

        if pd.api.types.is_numeric_dtype(value):
            columns.append(column)

    return columns
