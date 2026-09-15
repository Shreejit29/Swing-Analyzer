"""
Robust price-action feature engineering.

Accepts normal OHLCV columns as well as common lowercase/alternative names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def _normalize_ohlc_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common OHLC column naming and simple MultiIndex layouts."""
    out = df.copy()

    # Handle yfinance-style MultiIndex columns if they reach this stage.
    if isinstance(out.columns, pd.MultiIndex):
        flattened = []
        for col in out.columns:
            parts = [str(x) for x in col if str(x).lower() != "nan"]
            flattened.append("_".join(parts))
        out.columns = flattened

    # First try exact canonical names.
    aliases = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adj close": "Adj Close",
        "adj_close": "Adj Close",
        "volume": "Volume",
    }

    rename = {}
    for col in out.columns:
        key = str(col).strip().lower()
        if key in aliases:
            rename[col] = aliases[key]

    out = out.rename(columns=rename)

    # Handle flattened names such as Close_RELIANCE.NS or RELIANCE.NS_Close.
    for target in ("Open", "High", "Low", "Close", "Volume"):
        if target in out.columns:
            continue

        candidates = []
        target_lower = target.lower()
        for col in out.columns:
            text = str(col).lower()
            if text == target_lower or text.startswith(target_lower + "_") or text.endswith("_" + target_lower):
                candidates.append(col)

        if candidates:
            out[target] = out[candidates[0]]

    return out


def add_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add candle structure, returns, support/resistance, breakouts,
    market structure and ATR-normalized price-action features.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    out = _normalize_ohlc_columns(df)

    required = {"Open", "High", "Low", "Close"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Available columns: {list(out.columns)[:15]}"
        )

    o = pd.to_numeric(out["Open"], errors="coerce")
    h = pd.to_numeric(out["High"], errors="coerce")
    l = pd.to_numeric(out["Low"], errors="coerce")
    c = pd.to_numeric(out["Close"], errors="coerce")

    candle_range = (h - l).clip(lower=0)
    body = (c - o).abs()
    signed_body = c - o
    prev_close = c.shift(1)

    upper_wick = (h - pd.concat([o, c], axis=1).max(axis=1)).clip(lower=0)
    lower_wick = (pd.concat([o, c], axis=1).min(axis=1) - l).clip(lower=0)

    # Candle anatomy
    out["Candle_Range"] = candle_range
    out["Candle_Body"] = body
    out["Candle_Body_Signed"] = signed_body
    out["Upper_Wick"] = upper_wick
    out["Lower_Wick"] = lower_wick
    out["Body_to_Range"] = _safe_div(body, candle_range)
    out["Upper_Wick_to_Range"] = _safe_div(upper_wick, candle_range)
    out["Lower_Wick_to_Range"] = _safe_div(lower_wick, candle_range)

    out["Bullish_Candle"] = (c > o).astype(int)
    out["Bearish_Candle"] = (c < o).astype(int)
    out["Doji_Candle"] = (out["Body_to_Range"].fillna(1) <= 0.10).astype(int)
    out["Close_Location"] = (
        2 * _safe_div(c - l, candle_range) - 1
    ).clip(-1, 1)

    # Price movement
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

    # Leakage-safe rolling support/resistance.
    for window in (5, 10, 20, 50):
        resistance = h.shift(1).rolling(window).max()
        support = l.shift(1).rolling(window).min()

        out[f"Resistance_{window}"] = resistance
        out[f"Support_{window}"] = support
        out[f"Distance_Resistance_{window}_Pct"] = (
            _safe_div(c - resistance, resistance) * 100
        )
        out[f"Distance_Support_{window}_Pct"] = (
            _safe_div(c - support, support) * 100
        )
        out[f"Breakout_{window}"] = (c > resistance).astype(int)
        out[f"Breakdown_{window}"] = (c < support).astype(int)

    # Market structure
    hh = (h > h.shift(1)).astype(int)
    lh = (h < h.shift(1)).astype(int)
    hl = (l > l.shift(1)).astype(int)
    ll = (l < l.shift(1)).astype(int)

    out["Higher_High_1D"] = hh
    out["Lower_High_1D"] = lh
    out["Higher_Low_1D"] = hl
    out["Lower_Low_1D"] = ll
    out["HH_HL_Score"] = hh + hl - lh - ll
    out["Structure_Score_5D"] = out["HH_HL_Score"].rolling(5).sum()
    out["Structure_Score_10D"] = out["HH_HL_Score"].rolling(10).sum()

    # ATR-normalized movement
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

    # Consecutive bullish/bearish candles
    out["Bullish_Streak"] = (
        hh * 0  # keeps index/type consistent without using price direction twice
    )
    bullish = (c > o).astype(int)
    bearish = (c < o).astype(int)

    out["Bullish_Streak"] = bullish.groupby(
        (bullish != bullish.shift()).cumsum()
    ).cumsum()
    out["Bearish_Streak"] = bearish.groupby(
        (bearish != bearish.shift()).cumsum()
    ).cumsum()

    # Final cleanup
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)

    return out
