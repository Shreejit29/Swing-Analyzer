"""Volume feature engineering. Input columns: open, high, low, close, volume."""

from __future__ import annotations
import numpy as np
import pandas as pd


def _series(df: pd.DataFrame, name: str) -> pd.Series:
    value = df[name]
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    return pd.to_numeric(value, errors="coerce")


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume trend, relative volume, OBV and price-volume features."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    missing = [c for c in ("close", "volume") if c not in df.columns]
    if missing:
        raise ValueError(f"Volume features require lowercase columns. Missing: {missing}")

    out = df.copy()
    close = _series(out, "close")
    volume = _series(out, "volume")

    for window in (5, 10, 20, 50):
        out[f"volume_sma{window}"] = volume.rolling(window, min_periods=window).mean()

    out["volume_change_pct"] = volume.pct_change() * 100
    out["volume_roc5"] = volume.pct_change(5) * 100
    out["volume_roc20"] = volume.pct_change(20) * 100

    out["relative_volume_5"] = volume / out["volume_sma5"].replace(0, np.nan)
    out["relative_volume_20"] = volume / out["volume_sma20"].replace(0, np.nan)

    out["volume_slope_10"] = out["volume_sma10"] - out["volume_sma10"].shift(5)
    out["volume_slope_20"] = out["volume_sma20"] - out["volume_sma20"].shift(10)

    direction = np.sign(close.diff()).fillna(0)
    out["obv"] = (direction * volume.fillna(0)).cumsum()
    out["obv_sma20"] = out["obv"].rolling(20, min_periods=20).mean()
    out["obv_slope_10"] = out["obv"] - out["obv"].shift(10)

    price_return = close.pct_change()
    out["price_volume_sign"] = np.sign(price_return) * np.sign(out["volume_change_pct"])

    out["bullish_volume"] = (
        (price_return > 0) & (out["relative_volume_20"] > 1.0)
    ).astype(int)
    out["bearish_volume"] = (
        (price_return < 0) & (out["relative_volume_20"] > 1.0)
    ).astype(int)

    out["volume_spike"] = (out["relative_volume_20"] >= 1.5).astype(int)
    out["bullish_volume_spike"] = (
        (price_return > 0) & (out["relative_volume_20"] >= 1.5)
    ).astype(int)
    out["bearish_volume_spike"] = (
        (price_return < 0) & (out["relative_volume_20"] >= 1.5)
    ).astype(int)

    numeric = out.select_dtypes(include=[np.number]).columns
    out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan)
    return out
