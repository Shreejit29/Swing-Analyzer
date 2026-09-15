"""
Price-action feature engineering.

Project-wide input contract:
    open, high, low, close, volume

This module never changes the OHLCV column names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _s(df: pd.DataFrame, name: str) -> pd.Series:
    value = df[name]
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    return pd.to_numeric(value, errors="coerce")


def add_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add candle, return, structure, support/resistance and breakout features."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Price-action features require lowercase OHLC columns. "
            f"Missing: {missing}"
        )

    out = df.copy()

    open_ = _s(out, "open")
    high = _s(out, "high")
    low = _s(out, "low")
    close = _s(out, "close")

    previous_close = close.shift(1)

    # Candle structure
    candle_range = (high - low).clip(lower=0)
    body_signed = close - open_
    body = body_signed.abs()

    upper_wick = (
        high - pd.concat([open_, close], axis=1).max(axis=1)
    ).clip(lower=0)

    lower_wick = (
        pd.concat([open_, close], axis=1).min(axis=1) - low
    ).clip(lower=0)

    out["candle_range"] = candle_range
    out["candle_body"] = body
    out["candle_body_signed"] = body_signed
    out["upper_wick"] = upper_wick
    out["lower_wick"] = lower_wick

    out["body_to_range"] = body / candle_range.replace(0, np.nan)
    out["upper_wick_to_range"] = (
        upper_wick / candle_range.replace(0, np.nan)
    )
    out["lower_wick_to_range"] = (
        lower_wick / candle_range.replace(0, np.nan)
    )

    out["bullish_candle"] = (close > open_).astype(int)
    out["bearish_candle"] = (close < open_).astype(int)
    out["doji_candle"] = (
        out["body_to_range"].fillna(1) <= 0.10
    ).astype(int)

    # Where the close sits inside today's range: -1 = low, +1 = high.
    out["close_location"] = (
        2 * (close - low) / candle_range.replace(0, np.nan) - 1
    ).clip(-1, 1)

    # Returns and gaps
    out["gap_pct"] = (
        (open_ - previous_close)
        / previous_close.replace(0, np.nan)
        * 100
    )

    out["intraday_return_pct"] = (
        (close - open_)
        / open_.replace(0, np.nan)
        * 100
    )

    for window in (1, 3, 5, 10, 20):
        out[f"return_{window}d_pct"] = close.pct_change(window) * 100

    out["return_acceleration"] = (
        out["return_1d_pct"] - out["return_1d_pct"].shift(1)
    )

    # Leakage-safe support/resistance.
    for window in (5, 10, 20, 50):
        resistance = high.shift(1).rolling(
            window, min_periods=window
        ).max()

        support = low.shift(1).rolling(
            window, min_periods=window
        ).min()

        out[f"resistance_{window}"] = resistance
        out[f"support_{window}"] = support

        out[f"distance_resistance_{window}_pct"] = (
            (close - resistance)
            / resistance.replace(0, np.nan)
            * 100
        )

        out[f"distance_support_{window}_pct"] = (
            (close - support)
            / support.replace(0, np.nan)
            * 100
        )

        out[f"breakout_{window}"] = (
            close > resistance
        ).astype(int)

        out[f"breakdown_{window}"] = (
            close < support
        ).astype(int)

    # Market structure
    higher_high = (high > high.shift(1)).astype(int)
    lower_high = (high < high.shift(1)).astype(int)
    higher_low = (low > low.shift(1)).astype(int)
    lower_low = (low < low.shift(1)).astype(int)

    out["higher_high"] = higher_high
    out["lower_high"] = lower_high
    out["higher_low"] = higher_low
    out["lower_low"] = lower_low

    out["structure_score"] = (
        higher_high
        + higher_low
        - lower_high
        - lower_low
    )

    out["structure_score_5d"] = (
        out["structure_score"].rolling(5, min_periods=5).sum()
    )

    out["structure_score_10d"] = (
        out["structure_score"].rolling(10, min_periods=10).sum()
    )

    # Price movement relative to ATR.
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr14 = true_range.rolling(14, min_periods=14).mean()

    out["price_action_true_range"] = true_range
    out["price_action_atr14"] = atr14
    out["body_to_atr"] = body / atr14.replace(0, np.nan)
    out["range_to_atr"] = candle_range / atr14.replace(0, np.nan)
    out["close_move_to_atr"] = (
        (close - previous_close).abs()
        / atr14.replace(0, np.nan)
    )

    # Consecutive bullish/bearish candles.
    bullish = (close > open_).astype(int)
    bearish = (close < open_).astype(int)

    out["bullish_streak"] = bullish.groupby(
        bullish.ne(bullish.shift()).cumsum()
    ).cumsum()

    out["bearish_streak"] = bearish.groupby(
        bearish.ne(bearish.shift()).cumsum()
    ).cumsum()

    # Final cleanup.
    numeric = out.select_dtypes(include=[np.number]).columns
    out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan)

    return out
