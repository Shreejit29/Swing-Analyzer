# src/data/market.py

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _find_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:

    columns = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in columns:
            return columns[candidate.lower()]

    return None


def _normalise_ohlcv(df: pd.DataFrame) -> pd.DataFrame:

    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # Handle MultiIndex columns
    if isinstance(out.columns, pd.MultiIndex):

        flattened = []

        for col in out.columns:

            parts = [
                str(x).strip()
                for x in col
                if str(x).strip().lower() not in {"nan", "none"}
            ]

            flattened.append("_".join(parts))

        out.columns = flattened

    rename = {}

    for col in out.columns:

        name = str(col).strip().lower()

        if name in {"open", "open_price"}:
            rename[col] = "open"

        elif name in {"high", "high_price"}:
            rename[col] = "high"

        elif name in {"low", "low_price"}:
            rename[col] = "low"

        elif name in {"close", "close_price", "adj close"}:
            rename[col] = "close"

        elif name in {"volume", "vol"}:
            rename[col] = "volume"

    out = out.rename(columns=rename)

    # Remove duplicate columns
    out = out.loc[:, ~out.columns.duplicated()]

    for col in ["open", "high", "low", "close", "volume"]:

        if col in out.columns:
            out[col] = pd.to_numeric(
                out[col],
                errors="coerce",
            )

    return out


# ---------------------------------------------------------------------
# Market Trend
# ---------------------------------------------------------------------

def market_trend(
    df: pd.DataFrame,
) -> str:

    data = _normalise_ohlcv(df)

    if data.empty or "close" not in data.columns:
        return "UNKNOWN"

    close = data["close"].dropna()

    if len(close) < 20:
        return "UNKNOWN"

    ema20 = close.ewm(
        span=20,
        adjust=False,
    ).mean()

    ema50 = close.ewm(
        span=min(50, len(close)),
        adjust=False,
    ).mean()

    latest_close = _safe_float(close.iloc[-1])
    latest_ema20 = _safe_float(ema20.iloc[-1])
    latest_ema50 = _safe_float(ema50.iloc[-1])

    if latest_close > latest_ema20 > latest_ema50:
        return "BULLISH"

    if latest_close < latest_ema20 < latest_ema50:
        return "BEARISH"

    return "SIDEWAYS"


# ---------------------------------------------------------------------
# Market Momentum
# ---------------------------------------------------------------------

def market_momentum(
    df: pd.DataFrame,
) -> str:

    data = _normalise_ohlcv(df)

    if data.empty or "close" not in data.columns:
        return "UNKNOWN"

    close = data["close"].dropna()

    if len(close) < 15:
        return "UNKNOWN"

    returns = close.pct_change(10).iloc[-1]

    if pd.isna(returns):
        return "UNKNOWN"

    returns = float(returns)

    if returns > 0.03:
        return "STRONG POSITIVE"

    if returns > 0:
        return "POSITIVE"

    if returns < -0.03:
        return "STRONG NEGATIVE"

    if returns < 0:
        return "NEGATIVE"

    return "NEUTRAL"


# ---------------------------------------------------------------------
# Market Volatility
# ---------------------------------------------------------------------

def market_volatility(
    df: pd.DataFrame,
) -> str:

    data = _normalise_ohlcv(df)

    if data.empty or "close" not in data.columns:
        return "UNKNOWN"

    close = data["close"].dropna()

    if len(close) < 20:
        return "UNKNOWN"

    returns = close.pct_change()

    volatility = (
        returns.rolling(20)
        .std()
        .iloc[-1]
    )

    if pd.isna(volatility):
        return "UNKNOWN"

    volatility = float(volatility)

    if volatility >= 0.025:
        return "HIGH"

    if volatility >= 0.012:
        return "MEDIUM"

    return "LOW"


# ---------------------------------------------------------------------
# Market Breadth Proxy
# ---------------------------------------------------------------------

def market_strength(
    df: pd.DataFrame,
) -> float:

    data = _normalise_ohlcv(df)

    if data.empty or "close" not in data.columns:
        return 0.0

    close = data["close"].dropna()

    if len(close) < 20:
        return 0.0

    ema20 = close.ewm(
        span=20,
        adjust=False,
    ).mean()

    latest_close = _safe_float(close.iloc[-1])
    latest_ema20 = _safe_float(ema20.iloc[-1])

    if latest_ema20 == 0:
        return 0.0

    strength = (
        (latest_close - latest_ema20)
        / latest_ema20
    ) * 100.0

    return round(float(strength), 3)


# ---------------------------------------------------------------------
# Market Volume
# ---------------------------------------------------------------------

def market_volume(
    df: pd.DataFrame,
) -> str:

    data = _normalise_ohlcv(df)

    if (
        data.empty
        or "volume" not in data.columns
    ):
        return "UNKNOWN"

    volume = data["volume"].dropna()

    if len(volume) < 20:
        return "UNKNOWN"

    avg_volume = (
        volume
        .rolling(20)
        .mean()
        .iloc[-1]
    )

    latest_volume = volume.iloc[-1]

    if pd.isna(avg_volume) or avg_volume <= 0:
        return "UNKNOWN"

    ratio = float(latest_volume / avg_volume)

    if ratio >= 1.50:
        return "VERY HIGH"

    if ratio >= 1.20:
        return "HIGH"

    if ratio >= 0.80:
        return "NORMAL"

    return "LOW"


# ---------------------------------------------------------------------
# Market Summary
# ---------------------------------------------------------------------

def market_summary(
    df: pd.DataFrame,
) -> Dict[str, Any]:

    data = _normalise_ohlcv(df)

    trend = market_trend(data)
    momentum = market_momentum(data)
    volatility = market_volatility(data)
    volume = market_volume(data)
    strength = market_strength(data)

    # Combined market score
    score = 0.0

    if trend == "BULLISH":
        score += 2.0
    elif trend == "BEARISH":
        score -= 2.0

    if momentum == "STRONG POSITIVE":
        score += 2.0
    elif momentum == "POSITIVE":
        score += 1.0
    elif momentum == "NEGATIVE":
        score -= 1.0
    elif momentum == "STRONG NEGATIVE":
        score -= 2.0

    if strength > 2:
        score += 1.0
    elif strength < -2:
        score -= 1.0

    if score >= 3:
        bias = "BULLISH"

    elif score <= -3:
        bias = "BEARISH"

    else:
        bias = "NEUTRAL"

    return {
        "trend": trend,
        "momentum": momentum,
        "volatility": volatility,
        "volume": volume,
        "strength": strength,
        "score": round(score, 2),
        "bias": bias,
    }


# ---------------------------------------------------------------------
# Compatibility aliases
# ---------------------------------------------------------------------

get_market_trend = market_trend
get_market_momentum = market_momentum
get_market_summary = market_summary
get_market_strength = market_strength


__all__ = [
    "market_trend",
    "market_momentum",
    "market_volatility",
    "market_strength",
    "market_volume",
    "market_summary",
    "get_market_trend",
    "get_market_momentum",
    "get_market_summary",
    "get_market_strength",
]
