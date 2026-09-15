from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_value(row: pd.Series, name: str):
    """Return a finite float or None."""
    value = row.get(name)

    if value is None or not np.isscalar(value):
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(value):
        return None

    return value


def market_regime(row: pd.Series) -> str:
    """
    Classify the current market regime from trend structure and ADX.

    Regimes:
        STRONG BULL
        BULL
        SIDEWAYS
        BEAR
        STRONG BEAR
    """
    close = _safe_value(row, "close")
    ema20 = _safe_value(row, "ema20")
    ema50 = _safe_value(row, "ema50")
    ema200 = _safe_value(row, "ema200")
    adx = _safe_value(row, "adx14")

    # If the long-term EMA is not available yet, use a simpler
    # short/medium-term classification.
    if close is None:
        return "UNKNOWN"

    if ema20 is None:
        return "UNKNOWN"

    if ema50 is None:
        if close > ema20:
            return "BULL"
        if close < ema20:
            return "BEAR"
        return "SIDEWAYS"

    trend_strength = adx if adx is not None else 0.0

    if ema200 is not None:
        if (
            close > ema20
            and ema20 > ema50
            and ema50 > ema200
        ):
            if trend_strength >= 25:
                return "STRONG BULL"
            return "BULL"

        if (
            close < ema20
            and ema20 < ema50
            and ema50 < ema200
        ):
            if trend_strength >= 25:
                return "STRONG BEAR"
            return "BEAR"

        if close > ema50 and ema50 >= ema200:
            return "BULL"

        if close < ema50 and ema50 <= ema200:
            return "BEAR"

        return "SIDEWAYS"

    # Fallback when fewer than 200 candles are available.
    if close > ema20 > ema50:
        return "BULL"

    if close < ema20 < ema50:
        return "BEAR"

    return "SIDEWAYS"
