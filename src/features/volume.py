"""
Robust volume feature engineering for the Swing Analyzer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _find_column(df: pd.DataFrame, names: list[str]) -> str | None:
    """Find a column case-insensitively, including simple Yahoo-style names."""
    normalized = {str(c).strip().lower(): c for c in df.columns}

    for name in names:
        if name.lower() in normalized:
            return normalized[name.lower()]

    for col in df.columns:
        text = str(col).strip().lower()
        if any(text.startswith(name.lower() + "_") or text.endswith("_" + name.lower())
               for name in names):
            return col

    return None


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume, relative-volume, volume-trend and price-volume features.

    The function gracefully handles data without a Volume column. In that
    case, volume-derived features are filled with NaN instead of crashing.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    out = df.copy()

    volume_col = _find_column(out, ["Volume", "volume", "Vol"])
    close_col = _find_column(out, ["Close", "close", "Adj Close", "adj close"])

    # No volume data: keep the feature schema but don't stop the analyzer.
    if volume_col is None:
        volume = pd.Series(np.nan, index=out.index, dtype=float)
    else:
        volume = pd.to_numeric(out[volume_col], errors="coerce")

    if close_col is None:
        close = pd.Series(np.nan, index=out.index, dtype=float)
    else:
        close = pd.to_numeric(out[close_col], errors="coerce")

    out["Volume_Clean"] = volume

    # Basic volume statistics
    out["Volume_SMA5"] = volume.rolling(5, min_periods=1).mean()
    out["Volume_SMA10"] = volume.rolling(10, min_periods=1).mean()
    out["Volume_SMA20"] = volume.rolling(20, min_periods=1).mean()
    out["Volume_SMA50"] = volume.rolling(50, min_periods=1).mean()

    out["Volume_Change_Pct"] = volume.pct_change() * 100
    out["Volume_ROC5"] = volume.pct_change(5) * 100
    out["Volume_ROC20"] = volume.pct_change(20) * 100

    # Relative volume against recent average.
    out["Relative_Volume_5"] = (
        volume / out["Volume_SMA5"].replace(0, np.nan)
    )
    out["Relative_Volume_20"] = (
        volume / out["Volume_SMA20"].replace(0, np.nan)
    )

    # Volume trend
    out["Volume_Slope_10"] = (
        out["Volume_SMA10"] - out["Volume_SMA10"].shift(5)
    )
    out["Volume_Slope_20"] = (
        out["Volume_SMA20"] - out["Volume_SMA20"].shift(10)
    )

    # Price-volume confirmation
    price_return = close.pct_change()
    out["Price_Return"] = price_return
    out["Price_Volume_Sign"] = np.sign(price_return) * np.sign(
        out["Volume_Change_Pct"]
    )

    out["Bullish_Volume"] = (
        (price_return > 0) & (out["Relative_Volume_20"] > 1.0)
    ).astype(int)

    out["Bearish_Volume"] = (
        (price_return < 0) & (out["Relative_Volume_20"] > 1.0)
    ).astype(int)

    # OBV
    direction = np.sign(close.diff()).fillna(0)
    out["OBV"] = (direction * volume.fillna(0)).cumsum()
    out["OBV_SMA20"] = out["OBV"].rolling(20, min_periods=1).mean()
    out["OBV_Slope_10"] = out["OBV"] - out["OBV"].shift(10)

    # Simple volume spike flag.
    out["Volume_Spike"] = (
        out["Relative_Volume_20"] >= 1.5
    ).astype(int)

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)

    return out
