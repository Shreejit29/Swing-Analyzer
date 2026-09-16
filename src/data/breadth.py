"""
Market Breadth Engine
---------------------

Market-wide participation metrics for the Swing Analyzer.

The engine is deliberately defensive:
- accepts a DataFrame of OHLCV data for multiple symbols;
- tolerates missing columns/symbols;
- calculates advance/decline participation;
- calculates 52-week high/low participation when enough history exists;
- produces normalized breadth scores and a descriptive regime.

This module does not generate a stock-specific BUY/SELL decision.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else float(default)
    except Exception:
        return float(default)


def _normalize_symbol_frame(frame: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Normalize a single symbol OHLCV frame."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None

    df = frame.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = {"close"}
    if not required.issubset(df.columns):
        return None

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    return df


def _classify_change(change: float, threshold: float = 0.0) -> str:
    if change > threshold:
        return "ADVANCE"
    if change < -threshold:
        return "DECLINE"
    return "UNCHANGED"


def _symbol_snapshot(
    frame: pd.DataFrame,
    high_low_lookback: int = 252,
) -> Optional[Dict[str, Any]]:
    df = _normalize_symbol_frame(frame)
    if df is None:
        return None

    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    if len(close) < 2:
        return None

    latest = float(close.iloc[-1])
    previous = float(close.iloc[-2])
    change = latest - previous
    return_pct = (
        (latest / previous) - 1.0
        if previous != 0
        else 0.0
    )

    snapshot: Dict[str, Any] = {
        "close": latest,
        "previous_close": previous,
        "change": float(change),
        "return_pct": float(return_pct),
        "direction": _classify_change(change),
        "new_52w_high": False,
        "new_52w_low": False,
        "history_length": int(len(close)),
    }

    # Exclude the current observation when identifying whether today's close
    # broke the previous 52-week range.
    lookback = close.iloc[-(high_low_lookback + 1):-1]
    if len(lookback) >= min(60, high_low_lookback):
        prior_high = float(lookback.max())
        prior_low = float(lookback.min())

        snapshot["new_52w_high"] = bool(latest > prior_high)
        snapshot["new_52w_low"] = bool(latest < prior_low)

    return snapshot


def _score_from_ratio(value: float) -> float:
    """Map a non-negative ratio to [-1, 1] using a stable bounded transform."""
    value = max(_safe_float(value, 1.0), 0.0)
    score = (value - 1.0) / (value + 1.0)
    return float(np.clip(score, -1.0, 1.0))


def calculate_breadth(
    market_data: Dict[str, pd.DataFrame],
    high_low_lookback: int = 252,
) -> Dict[str, Any]:
    """
    Calculate market breadth from multiple symbol OHLCV DataFrames.

    Parameters
    ----------
    market_data:
        Mapping of symbol -> OHLCV DataFrame.
    high_low_lookback:
        Number of observations used for the 52-week proxy.

    Returns
    -------
    dict
        Breadth statistics, scores, regime, and per-symbol diagnostics.
    """
    if not isinstance(market_data, dict):
        raise TypeError("market_data must be a dictionary of symbol DataFrames.")

    snapshots: Dict[str, Dict[str, Any]] = {}

    for symbol, frame in market_data.items():
        snapshot = _symbol_snapshot(
            frame,
            high_low_lookback=high_low_lookback,
        )
        if snapshot is not None:
            snapshots[str(symbol).upper().strip()] = snapshot

    total = len(snapshots)

    if total == 0:
        return {
            "available": False,
            "total_symbols": 0,
            "advancing": 0,
            "declining": 0,
            "unchanged": 0,
            "advance_decline_ratio": 1.0,
            "advance_decline_score": 0.0,
            "new_highs": 0,
            "new_lows": 0,
            "high_low_ratio": 1.0,
            "high_low_score": 0.0,
            "participation_score": 0.0,
            "breadth_score": 0.0,
            "regime": "UNKNOWN",
            "symbols": {},
        }

    advancing = sum(
        item["direction"] == "ADVANCE"
        for item in snapshots.values()
    )
    declining = sum(
        item["direction"] == "DECLINE"
        for item in snapshots.values()
    )
    unchanged = total - advancing - declining

    ad_ratio = (
        advancing / declining
        if declining > 0
        else float(advancing) if advancing > 0 else 1.0
    )

    participation_score = (
        (advancing - declining) / total
        if total > 0
        else 0.0
    )

    new_highs = sum(
        item["new_52w_high"]
        for item in snapshots.values()
    )
    new_lows = sum(
        item["new_52w_low"]
        for item in snapshots.values()
    )

    high_low_ratio = (
        new_highs / new_lows
        if new_lows > 0
        else float(new_highs) if new_highs > 0 else 1.0
    )

    ad_score = _score_from_ratio(ad_ratio)
    hl_score = _score_from_ratio(high_low_ratio)

    # Breadth is intentionally dominated by current participation.
    breadth_score = float(
        np.clip(
            0.55 * participation_score
            + 0.30 * ad_score
            + 0.15 * hl_score,
            -1.0,
            1.0,
        )
    )

    if breadth_score >= 0.50:
        regime = "STRONG BULLISH BREADTH"
    elif breadth_score >= 0.20:
        regime = "BULLISH BREADTH"
    elif breadth_score <= -0.50:
        regime = "STRONG BEARISH BREADTH"
    elif breadth_score <= -0.20:
        regime = "BEARISH BREADTH"
    else:
        regime = "NEUTRAL BREADTH"

    return {
        "available": True,
        "total_symbols": int(total),
        "advancing": int(advancing),
        "declining": int(declining),
        "unchanged": int(unchanged),
        "advance_decline_ratio": float(ad_ratio),
        "advance_decline_score": float(ad_score),
        "new_highs": int(new_highs),
        "new_lows": int(new_lows),
        "high_low_ratio": float(high_low_ratio),
        "high_low_score": float(hl_score),
        "participation_score": float(
            np.clip(participation_score, -1.0, 1.0)
        ),
        "breadth_score": breadth_score,
        "regime": regime,
        "symbols": snapshots,
    }


def market_breadth(
    market_data: Dict[str, pd.DataFrame],
    high_low_lookback: int = 252,
) -> Dict[str, Any]:
    """Compatibility alias."""
    return calculate_breadth(
        market_data,
        high_low_lookback=high_low_lookback,
    )


def breadth_score(
    market_data: Dict[str, pd.DataFrame],
    high_low_lookback: int = 252,
) -> float:
    """Return only the normalized breadth score."""
    result = calculate_breadth(
        market_data,
        high_low_lookback=high_low_lookback,
    )
    return _safe_float(result.get("breadth_score"), 0.0)


def breadth_summary(
    market_data: Dict[str, pd.DataFrame],
    high_low_lookback: int = 252,
) -> Dict[str, Any]:
    """Return a compact summary suitable for the predictor/UI."""
    result = calculate_breadth(
        market_data,
        high_low_lookback=high_low_lookback,
    )

    return {
        "available": bool(result.get("available", False)),
        "total_symbols": int(result.get("total_symbols", 0)),
        "advancing": int(result.get("advancing", 0)),
        "declining": int(result.get("declining", 0)),
        "unchanged": int(result.get("unchanged", 0)),
        "advance_decline_ratio": _safe_float(
            result.get("advance_decline_ratio"),
            1.0,
        ),
        "new_highs": int(result.get("new_highs", 0)),
        "new_lows": int(result.get("new_lows", 0)),
        "breadth_score": _safe_float(
            result.get("breadth_score"),
            0.0,
        ),
        "regime": str(
            result.get("regime", "UNKNOWN")
        ),
    }


__all__ = [
    "calculate_breadth",
    "market_breadth",
    "breadth_score",
    "breadth_summary",
]
