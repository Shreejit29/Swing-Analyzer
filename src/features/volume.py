"""
Robust volume feature engineering.

This version also handles DataFrames where OHLCV columns arrive with
different casing or Yahoo Finance MultiIndex/flattened naming.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        # Prefer the OHLCV field name rather than ticker name.
        fields = {"open", "high", "low", "close", "adj close", "volume"}
        new_columns = []

        for col in out.columns:
            parts = [str(x).strip() for x in col if str(x).strip().lower() != "nan"]
            chosen = None

            for part in parts:
                if part.lower() in fields:
                    chosen = part
                    break

            new_columns.append(chosen if chosen else "_".join(parts))

        out.columns = new_columns

    return out


def _get_series(df: pd.DataFrame, aliases: list[str]) -> pd.Series | None:
    """Return the first matching column as a guaranteed 1-D numeric Series."""
    aliases_lower = {x.lower() for x in aliases}

    # Exact case-insensitive match.
    for col in df.columns:
        if str(col).strip().lower() in aliases_lower:
            value = df[col]

            # A duplicate column name can return a DataFrame.
            if isinstance(value, pd.DataFrame):
                value = value.iloc[:, 0]

            return pd.to_numeric(value, errors="coerce")

    # Match flattened Yahoo names such as Close_RELIANCE.NS.
    for col in df.columns:
        text = str(col).strip().lower()
        for alias in aliases_lower:
            if text.startswith(alias + "_") or text.endswith("_" + alias):
                value = df[col]
                if isinstance(value, pd.DataFrame):
                    value = value.iloc[:, 0]
                return pd.to_numeric(value, errors="coerce")

    return None


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add robust volume and price-volume features without requiring exact column names."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    out = _flatten_columns(df)

    close = _get_series(out, ["Close", "close", "Adj Close", "adj close"])
    volume = _get_series(out, ["Volume", "volume", "Vol"])

    # Keep the feature pipeline alive even if a source has no volume.
    if close is None:
        close = pd.Series(np.nan, index=out.index, dtype=float)

    if volume is None:
        volume = pd.Series(np.nan, index=out.index, dtype=float)

    # Ensure both are explicitly 1-D Series.
    close = pd.Series(close, index=out.index, dtype=float)
    volume = pd.Series(volume, index=out.index, dtype=float)

    out["Volume_Clean"] = volume

    # Rolling volume
    for window in (5, 10, 20, 50):
        out[f"Volume_SMA{window}"] = volume.rolling(
            window, min_periods=window
        ).mean()

    out["Volume_Change_Pct"] = volume.pct_change() * 100
    out["Volume_ROC5"] = volume.pct_change(5) * 100
    out["Volume_ROC20"] = volume.pct_change(20) * 100

    # Relative volume
    out["Relative_Volume_5"] = volume / out["Volume_SMA5"].replace(0, np.nan)
    out["Relative_Volume_20"] = volume / out["Volume_SMA20"].replace(0, np.nan)

    # Volume trend
    out["Volume_Slope_10"] = (
        out["Volume_SMA10"] - out["Volume_SMA10"].shift(5)
    )
    out["Volume_Slope_20"] = (
        out["Volume_SMA20"] - out["Volume_SMA20"].shift(10)
    )

    # Price-volume relationship
    price_return = close.pct_change()
    volume_change = out["Volume_Change_Pct"]

    out["Price_Return"] = price_return
    out["Price_Volume_Sign"] = np.sign(price_return) * np.sign(volume_change)

    relative_volume = out["Relative_Volume_20"]

    out["Bullish_Volume"] = (
        (price_return > 0) & (relative_volume > 1.0)
    ).astype(int)

    out["Bearish_Volume"] = (
        (price_return < 0) & (relative_volume > 1.0)
    ).astype(int)

    out["Volume_Spike"] = (relative_volume >= 1.5).astype(int)

    # OBV
    direction = np.sign(close.diff()).fillna(0)
    out["OBV"] = (direction * volume.fillna(0)).cumsum()
    out["OBV_SMA20"] = out["OBV"].rolling(20, min_periods=20).mean()
    out["OBV_Slope_10"] = out["OBV"] - out["OBV"].shift(10)

    # Final numeric cleanup.
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)

    return out
