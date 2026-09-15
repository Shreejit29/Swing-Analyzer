"""
Price-action feature engineering for the Swing Analyzer.

The functions in this module add candle structure, momentum, breakout,
support/resistance, and trend-strength features while keeping the original
OHLCV columns intact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """Element-wise division that avoids infinities and zero-division errors."""
    denominator = b.replace(0, np.nan)
    return a / denominator


def add_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add robust price-action features to an OHLCV dataframe.

    Added feature groups:
    - Candle anatomy: body, range, upper/lower wick, body-to-range
    - Directional candle flags
    - Gaps and close location
    - Multi-day returns and acceleration
    - Rolling support/resistance
    - Breakout/breakdown confirmation
    - Higher-high / lower-low structure
    - ATR-normalized price movement
    """
    out = df.copy()

    required = {"Open", "High", "Low", "Close"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    o = pd.to_numeric(out["Open"], errors="coerce")
    h = pd.to_numeric(out["High"], errors="coerce")
    l = pd.to_numeric(out["Low"], errors="coerce")
    c = pd.to_numeric(out["Close"], errors="coerce")

    # ------------------------------------------------------------------
    # Candle anatomy
    # ------------------------------------------------------------------
    candle_range = (h - l).clip(lower=0)
    body = (c - o).abs()
    signed_body = c - o

    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l

    out["Candle_Range"] = candle_range
    out["Candle_Body"] = body
    out["Candle_Body_Signed"] = signed_body
    out["Upper_Wick"] = upper_wick.clip(lower=0)
    out["Lower_Wick"] = lower_wick.clip(lower=0)
    out["Body_to_Range"] = _safe_div(body, candle_range)
    out["Upper_Wick_to_Range"] = _safe_div(out["Upper_Wick"], candle_range)
    out["Lower_Wick_to_Range"] = _safe_div(out["Lower_Wick"], candle_range)

    out["Bullish_Candle"] = (c > o).astype(int)
    out["Bearish_Candle"] = (c < o).astype(int)
    out["Doji_Candle"] = (
        out["Body_to_Range"].fillna(1) <= 0.10
    ).astype(int)

    # Close location within today's range: +1 near high, -1 near low.
    out["Close_Location"] = (
        2 * _safe_div(c - l, candle_range) - 1
    ).clip(-1, 1)

    # ------------------------------------------------------------------
    # Gap and short-term price movement
    # ------------------------------------------------------------------
    prev_close = c.shift(1)

    out["Gap_Pct"] = _safe_div(o - prev_close, prev_close) * 100
    out["Intraday_Return_Pct"] = _safe_div(c - o, o) * 100
    out["Return_1D_Pct"] = c.pct_change() * 100
    out["Return_3D_Pct"] = c.pct_change(3) * 100
    out["Return_5D_Pct"] = c.pct_change(5) * 100
    out["Return_10D_Pct"] = c.pct_change(10) * 100
    out["Return_20D_Pct"] = c.pct_change(20) * 100

    out["Return_1D_Acceleration"] = (
        out["Return_1D_Pct"] - out["Return_1D_Pct"].shift(1)
    )

    # ------------------------------------------------------------------
    # Rolling support / resistance
    # Use shifted levels so today's close is not used to define today's
    # breakout level, avoiding look-ahead leakage.
    # ------------------------------------------------------------------
    for window in (5, 10, 20, 50):
        prior_high = h.shift(1).rolling(window).max()
        prior_low = l.shift(1).rolling(window).min()

        out[f"Resistance_{window}"] = prior_high
        out[f"Support_{window}"] = prior_low

        out[f"Distance_Resistance_{window}_Pct"] = (
            _safe_div(c - prior_high, prior_high) * 100
        )
        out[f"Distance_Support_{window}_Pct"] = (
            _safe_div(c - prior_low, prior_low) * 100
        )

        out[f"Breakout_{window}"] = (c > prior_high).astype(int)
        out[f"Breakdown_{window}"] = (c < prior_low).astype(int)

    # ------------------------------------------------------------------
    # Market structure
    # ------------------------------------------------------------------
    out["Higher_High_1D"] = (h > h.shift(1)).astype(int)
    out["Lower_High_1D"] = (h < h.shift(1)).astype(int)
    out["Higher_Low_1D"] = (l > l.shift(1)).astype(int)
    out["Lower_Low_1D"] = (l < l.shift(1)).astype(int)

    out["HH_HL_Score"] = (
        out["Higher_High_1D"]
        + out["Higher_Low_1D"]
        - out["Lower_High_1D"]
        - out["Lower_Low_1D"]
    )

    # Rolling structure score gives the model a smoother trend signal.
    out["Structure_Score_5D"] = out["HH_HL_Score"].rolling(5).sum()
    out["Structure_Score_10D"] = out["HH_HL_Score"].rolling(10).sum()

    # ------------------------------------------------------------------
    # Volatility-normalized movement
    # ------------------------------------------------------------------
    true_range = pd.concat(
        [
            h - l,
            (h - prev_close).abs(),
            (l - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr14 = true_range.rolling(14).mean()

    out["True_Range"] = true_range
    out["ATR14_PriceAction"] = atr14
    out["Body_to_ATR"] = _safe_div(body, atr14)
    out["Range_to_ATR"] = _safe_div(candle_range, atr14)
    out["Close_Move_to_ATR"] = _safe_div((c - prev_close).abs(), atr14)

    # ------------------------------------------------------------------
    # Consecutive directional candles
    # ------------------------------------------------------------------
    bullish = (c > o).astype(int)
    bearish = (c < o).astype(int)

    out["Bullish_Streak"] = (
        bullish.groupby((bullish != bullish.shift()).cumsum()).cumsum()
    )
    out["Bearish_Streak"] = (
        bearish.groupby((bearish != bearish.shift()).cumsum()).cumsum()
    )

    # ------------------------------------------------------------------
    # Clean numeric output
    # ------------------------------------------------------------------
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace(
        [np.inf, -np.inf], np.nan
    )

    return out
