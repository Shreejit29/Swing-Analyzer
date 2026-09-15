"""
Market regime detection.

Input:
    DataFrame containing OHLCV + technical features.

Output:
    DataFrame with regime-related columns.

The module is deliberately defensive because stocks may have
insufficient history for a reliable EMA200.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _series(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a numeric series, case-insensitively."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")

    lookup = {str(c).lower(): c for c in df.columns}

    if name.lower() in lookup:
        return pd.to_numeric(df[lookup[name.lower()]], errors="coerce")

    return pd.Series(np.nan, index=df.index, dtype=float)


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add market-regime features.

    Required:
        close

    Preferred:
        ema20, ema50, ema200

    The function does not modify the input DataFrame.
    """
    out = df.copy()

    close = _series(out, "close")

    ema20 = _series(out, "ema20")
    ema50 = _series(out, "ema50")
    ema200 = _series(out, "ema200")

    # Calculate missing EMAs from close.
    if ema20.isna().all():
        ema20 = close.ewm(span=20, adjust=False, min_periods=1).mean()

    if ema50.isna().all():
        ema50 = close.ewm(span=50, adjust=False, min_periods=1).mean()

    if ema200.isna().all():
        ema200 = close.ewm(span=200, adjust=False, min_periods=1).mean()

    out["ema20"] = ema20
    out["ema50"] = ema50
    out["ema200"] = ema200

    # ---------------------------------------------------------
    # Trend relationships
    # ---------------------------------------------------------
    out["price_vs_ema20"] = close / ema20.replace(0, np.nan) - 1.0
    out["price_vs_ema50"] = close / ema50.replace(0, np.nan) - 1.0
    out["price_vs_ema200"] = close / ema200.replace(0, np.nan) - 1.0

    out["ema20_vs_ema50"] = ema20 / ema50.replace(0, np.nan) - 1.0
    out["ema50_vs_ema200"] = ema50 / ema200.replace(0, np.nan) - 1.0

    # ---------------------------------------------------------
    # EMA slopes
    # ---------------------------------------------------------
    out["ema20_slope"] = ema20.pct_change(5)
    out["ema50_slope"] = ema50.pct_change(10)
    out["ema200_slope"] = ema200.pct_change(20)

    # ---------------------------------------------------------
    # Trend score
    #
    # We use multiple independent conditions rather than a
    # single indicator.
    # ---------------------------------------------------------
    score = pd.Series(0.0, index=out.index)

    score += np.where(close > ema20, 1, -1)
    score += np.where(ema20 > ema50, 1, -1)
    score += np.where(ema50 > ema200, 1, -1)

    score += np.where(out["ema20_slope"] > 0, 1, -1)
    score += np.where(out["ema50_slope"] > 0, 1, -1)

    out["regime_score"] = score

    # ---------------------------------------------------------
    # Volatility context
    # ---------------------------------------------------------
    returns = close.pct_change()

    volatility = returns.rolling(20, min_periods=5).std()

    out["regime_volatility"] = volatility
    out["regime_volatility_rank"] = (
        volatility
        .rolling(100, min_periods=20)
        .rank(pct=True)
    )

    # ---------------------------------------------------------
    # Regime classification
    #
    # Strong trend:
    #     score >= +4 / <= -4
    #
    # Normal trend:
    #     score >= +2 / <= -2
    #
    # Otherwise:
    #     SIDEWAYS
    # ---------------------------------------------------------
    def classify(value: float) -> str:
        if pd.isna(value):
            return "UNKNOWN"

        if value >= 4:
            return "STRONG BULL"

        if value >= 2:
            return "BULL"

        if value <= -4:
            return "STRONG BEAR"

        if value <= -2:
            return "BEAR"

        return "SIDEWAYS"

    out["regime"] = out["regime_score"].apply(classify)

    # ---------------------------------------------------------
    # Numeric trend direction
    # Useful for ML models.
    # ---------------------------------------------------------
    out["regime_direction"] = out["regime_score"].apply(
        lambda x: (
            1
            if pd.notna(x) and x >= 2
            else -1
            if pd.notna(x) and x <= -2
            else 0
        )
    )

    # ---------------------------------------------------------
    # Clean numerical problems
    # ---------------------------------------------------------
    numeric_columns = out.select_dtypes(include=[np.number]).columns
    out[numeric_columns] = out[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return out


def detect_regime(df: pd.DataFrame) -> str:
    """
    Return the latest market regime.

    Examples:
        STRONG BULL
        BULL
        SIDEWAYS
        BEAR
        STRONG BEAR
        UNKNOWN
    """
    features = add_regime_features(df)

    if features.empty or "regime" not in features.columns:
        return "UNKNOWN"

    latest = features["regime"].iloc[-1]

    if pd.isna(latest):
        return "UNKNOWN"

    return str(latest)


def get_regime_score(df: pd.DataFrame) -> float:
    """Return the latest numerical regime score."""
    features = add_regime_features(df)

    if features.empty:
        return 0.0

    value = features["regime_score"].iloc[-1]

    if pd.isna(value):
        return 0.0

    return float(value)


__all__ = [
    "add_regime_features",
    "detect_regime",
    "get_regime_score",
]
