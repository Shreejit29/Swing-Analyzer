"""
Market regime detection for the AI Swing Stock Analyzer.

Canonical OHLCV convention:
    open
    high
    low
    close
    volume

This module provides:
    - add_regime_features()
    - detect_regime()
    - market_regime()       # backward-compatible API for predictor.py
    - get_regime_score()
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _series(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a numeric Series using case-insensitive column matching."""

    if name in df.columns:
        value = df[name]

        # Protect against duplicate columns.
        if isinstance(value, pd.DataFrame):
            value = value.iloc[:, 0]

        return pd.to_numeric(value, errors="coerce")

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    key = name.lower()

    if key in lookup:
        value = df[lookup[key]]

        if isinstance(value, pd.DataFrame):
            value = value.iloc[:, 0]

        return pd.to_numeric(value, errors="coerce")

    return pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )


def _calculate_ema(
    close: pd.Series,
    span: int,
) -> pd.Series:
    """Calculate an EMA safely."""

    return close.ewm(
        span=span,
        adjust=False,
        min_periods=1,
    ).mean()


def add_regime_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add market-regime features.

    Parameters
    ----------
    df:
        DataFrame containing at least a close-price column.

    Returns
    -------
    pandas.DataFrame
        Original data plus regime features.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame"
        )

    if df.empty:
        return df.copy()

    out = df.copy()

    close = _series(out, "close")

    if close.isna().all():
        raise ValueError(
            "Market regime detection requires a valid close column."
        )

    # ---------------------------------------------------------
    # Moving averages
    # ---------------------------------------------------------

    ema20 = _series(out, "ema20")

    if ema20.isna().all():
        ema20 = _calculate_ema(
            close,
            20,
        )

    ema50 = _series(out, "ema50")

    if ema50.isna().all():
        ema50 = _calculate_ema(
            close,
            50,
        )

    ema200 = _series(out, "ema200")

    if ema200.isna().all():
        ema200 = _calculate_ema(
            close,
            200,
        )

    out["ema20"] = ema20
    out["ema50"] = ema50
    out["ema200"] = ema200

    # ---------------------------------------------------------
    # Price vs moving averages
    # ---------------------------------------------------------

    out["price_vs_ema20"] = (
        close / ema20.replace(0, np.nan)
        - 1.0
    )

    out["price_vs_ema50"] = (
        close / ema50.replace(0, np.nan)
        - 1.0
    )

    out["price_vs_ema200"] = (
        close / ema200.replace(0, np.nan)
        - 1.0
    )

    # ---------------------------------------------------------
    # EMA relationships
    # ---------------------------------------------------------

    out["ema20_vs_ema50"] = (
        ema20 / ema50.replace(0, np.nan)
        - 1.0
    )

    out["ema50_vs_ema200"] = (
        ema50 / ema200.replace(0, np.nan)
        - 1.0
    )

    # ---------------------------------------------------------
    # EMA slopes
    # ---------------------------------------------------------

    out["ema20_slope"] = (
        ema20.pct_change(5)
    )

    out["ema50_slope"] = (
        ema50.pct_change(10)
    )

    out["ema200_slope"] = (
        ema200.pct_change(20)
    )

    # ---------------------------------------------------------
    # Trend conditions
    # ---------------------------------------------------------

    price_above_ema20 = (
        close > ema20
    ).astype(int)

    price_above_ema50 = (
        close > ema50
    ).astype(int)

    price_above_ema200 = (
        close > ema200
    ).astype(int)

    ema20_above_ema50 = (
        ema20 > ema50
    ).astype(int)

    ema50_above_ema200 = (
        ema50 > ema200
    ).astype(int)

    out["price_above_ema20"] = (
        price_above_ema20
    )

    out["price_above_ema50"] = (
        price_above_ema50
    )

    out["price_above_ema200"] = (
        price_above_ema200
    )

    out["ema20_above_ema50"] = (
        ema20_above_ema50
    )

    out["ema50_above_ema200"] = (
        ema50_above_ema200
    )

    # ---------------------------------------------------------
    # Trend score
    #
    # Five components:
    #
    # +1 / -1 price vs EMA20
    # +1 / -1 EMA20 vs EMA50
    # +1 / -1 EMA50 vs EMA200
    # +1 / -1 EMA20 slope
    # +1 / -1 EMA50 slope
    #
    # Range:
    #       -5 to +5
    # ---------------------------------------------------------

    score = pd.Series(
        0.0,
        index=out.index,
    )

    score += np.where(
        close > ema20,
        1,
        -1,
    )

    score += np.where(
        ema20 > ema50,
        1,
        -1,
    )

    score += np.where(
        ema50 > ema200,
        1,
        -1,
    )

    score += np.where(
        out["ema20_slope"] > 0,
        1,
        -1,
    )

    score += np.where(
        out["ema50_slope"] > 0,
        1,
        -1,
    )

    out["regime_score"] = score

    # ---------------------------------------------------------
    # Market volatility
    # ---------------------------------------------------------

    returns = close.pct_change()

    volatility = returns.rolling(
        20,
        min_periods=5,
    ).std()

    out["regime_volatility"] = volatility

    out["regime_volatility_rank"] = (
        volatility
        .rolling(
            100,
            min_periods=20,
        )
        .rank(pct=True)
    )

    # ---------------------------------------------------------
    # Regime classification
    # ---------------------------------------------------------

    def classify(value) -> str:

        if pd.isna(value):
            return "UNKNOWN"

        value = float(value)

        if value >= 4:
            return "STRONG BULL"

        if value >= 2:
            return "BULL"

        if value <= -4:
            return "STRONG BEAR"

        if value <= -2:
            return "BEAR"

        return "SIDEWAYS"

    out["regime"] = (
        out["regime_score"]
        .apply(classify)
    )

    # ---------------------------------------------------------
    # Numerical regime direction
    #
    # +1 = bullish
    #  0 = sideways
    # -1 = bearish
    # ---------------------------------------------------------

    out["regime_direction"] = (
        out["regime_score"]
        .apply(
            lambda value:
            1
            if pd.notna(value) and value >= 2
            else -1
            if pd.notna(value) and value <= -2
            else 0
        )
    )

    # ---------------------------------------------------------
    # Clean infinite values
    # ---------------------------------------------------------

    numeric_columns = (
        out
        .select_dtypes(
            include=[np.number]
        )
        .columns
    )

    out[numeric_columns] = (
        out[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    return out


def detect_regime(
    df: pd.DataFrame,
) -> str:
    """
    Return the latest detected market regime.
    """

    if df is None or df.empty:
        return "UNKNOWN"

    try:
        features = add_regime_features(
            df
        )
    except Exception:
        return "UNKNOWN"

    if (
        features.empty
        or "regime" not in features.columns
    ):
        return "UNKNOWN"

    latest = features[
        "regime"
    ].iloc[-1]

    if pd.isna(latest):
        return "UNKNOWN"

    return str(latest)


def market_regime(
    df: pd.DataFrame,
) -> str:
    """
    Backward-compatible function.

    predictor.py currently imports:

        from src.features.regime import market_regime

    Therefore this function must remain available.
    """

    return detect_regime(df)


def get_regime_score(
    df: pd.DataFrame,
) -> float:
    """
    Return the latest numerical regime score.
    """

    if df is None or df.empty:
        return 0.0

    try:
        features = add_regime_features(
            df
        )
    except Exception:
        return 0.0

    if (
        features.empty
        or "regime_score"
        not in features.columns
    ):
        return 0.0

    value = features[
        "regime_score"
    ].iloc[-1]

    if pd.isna(value):
        return 0.0

    return float(value)


__all__ = [
    "add_regime_features",
    "detect_regime",
    "market_regime",
    "get_regime_score",
]
