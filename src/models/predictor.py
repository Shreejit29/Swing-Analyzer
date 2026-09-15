from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from src.features.engine import build_features
from src.features.regime import market_regime
from .classifier import SwingClassifier


@dataclass
class SwingPrediction:
    signal: str
    probability_up: float
    probability_down: float
    confidence: str
    expected_return: float
    entry: float
    stop_loss: float
    target: float
    risk_reward: float
    regime: str
    indicators: dict
    model_training_accuracy: float | None = None
    signal_reason: str = ""
    trend: str = ""
    momentum: str = ""
    volume_status: str = ""


def _safe_float(value):
    """Convert a scalar value to float, otherwise return None."""
    if value is None or not np.isscalar(value):
        return None

    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _describe_trend(last):
    close = _safe_float(last.get("close"))
    ema20 = _safe_float(last.get("ema20"))
    ema50 = _safe_float(last.get("ema50"))
    ema200 = _safe_float(last.get("ema200"))

    if close is None or ema20 is None or ema50 is None:
        return "UNKNOWN"

    if ema200 is not None and close > ema20 > ema50 > ema200:
        return "STRONG BULLISH"

    if ema200 is not None and close < ema20 < ema50 < ema200:
        return "STRONG BEARISH"

    if close > ema20 and close > ema50:
        return "BULLISH"

    if close < ema20 and close < ema50:
        return "BEARISH"

    return "MIXED"


def _describe_momentum(last):
    rsi = _safe_float(last.get("rsi14"))
    macd = _safe_float(last.get("macd"))
    macd_signal = _safe_float(last.get("macd_signal"))

    if rsi is None:
        return "UNKNOWN"

    if rsi >= 70:
        return "OVERBOUGHT"

    if rsi <= 30:
        return "OVERSOLD"

    if macd is not None and macd_signal is not None:
        if rsi >= 55 and macd > macd_signal:
            return "BULLISH"
        if rsi <= 45 and macd < macd_signal:
            return "BEARISH"

    if rsi > 50:
        return "MILDLY BULLISH"

    if rsi < 50:
        return "MILDLY BEARISH"

    return "NEUTRAL"


def _describe_volume(last):
    relative_volume = _safe_float(last.get("relative_volume"))

    if relative_volume is None:
        return "UNKNOWN"

    if relative_volume >= 1.5:
        return "STRONG"

    if relative_volume >= 1.0:
        return "ABOVE AVERAGE"

    if relative_volume >= 0.75:
        return "NORMAL"

    return "LOW"


def analyze_stock(
    df: pd.DataFrame,
    horizon: int = 5,
    probability_threshold: float = 0.60,
    stop_loss_pct: float = 0.03,
    target_pct: float = 0.06,
) -> SwingPrediction:
    """
    Analyze the latest market candle using technical features
    and the Gradient Boosting classifier.

    The model probability is combined with simple trend, momentum,
    and volume diagnostics to make the signal easier to understand.
    """
    if df is None or df.empty:
        raise ValueError("No market data was supplied for analysis.")

    if stop_loss_pct <= 0:
        raise ValueError("Stop loss percentage must be greater than zero.")

    if target_pct <= 0:
        raise ValueError("Target percentage must be greater than zero.")

    features = build_features(df)

    if features.empty:
        raise ValueError("Unable to generate technical indicators from the data.")

    model = SwingClassifier(
        horizon=horizon,
        probability_threshold=probability_threshold,
    )

    model.fit(features)

    probabilities = model.predict_proba(features)
    if probabilities.ndim != 2 or probabilities.shape[0] == 0:
        raise ValueError("Model did not return valid prediction probabilities.")

    latest_probability = probabilities[-1]
    p_up = float(latest_probability[1])
    p_up = min(max(p_up, 0.0), 1.0)
    p_down = 1.0 - p_up

    last = features.iloc[-1]

    close = _safe_float(last.get("close"))

    if close is None or close <= 0:
        raise ValueError("Latest closing price is unavailable.")

    trend = _describe_trend(last)
    momentum = _describe_momentum(last)
    volume_status = _describe_volume(last)
    regime = market_regime(last)

    # The probability is the primary decision variable.
    # Technical diagnostics are reported separately so the user can
    # understand whether the AI signal agrees with the market structure.
    if p_up >= probability_threshold and p_up > p_down:
        signal = "BUY"
        entry = close
        stop = close * (1.0 - stop_loss_pct)
        target = close * (1.0 + target_pct)

        signal_reason = (
            f"AI estimates {p_up:.1%} probability of an upward move "
            f"over the next {horizon} trading day(s)."
        )

    elif p_down >= probability_threshold and p_down > p_up:
        signal = "SELL"
        entry = close
        stop = close * (1.0 + stop_loss_pct)
        target = close * (1.0 - target_pct)

        signal_reason = (
            f"AI estimates {p_down:.1%} probability of a downward move "
            f"over the next {horizon} trading day(s)."
        )

    else:
        signal = "WAIT"
        entry = close
        stop = close
        target = close

        strongest_probability = max(p_up, p_down)

        signal_reason = (
            f"Neither direction reached the {probability_threshold:.0%} "
            f"AI probability threshold. Strongest probability: "
            f"{strongest_probability:.1%}."
        )

    risk_reward = (
        target_pct / stop_loss_pct
        if stop_loss_pct > 0
        else math.inf
    )

    strongest_probability = max(p_up, p_down)

    if strongest_probability >= 0.70:
        confidence = "HIGH"
    elif strongest_probability >= probability_threshold:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    indicator_names = [
        "rsi14",
        "adx",
        "atr14",
        "macd",
        "macd_signal",
        "macd_hist",
        "relative_volume",
        "ema20",
        "ema50",
        "ema200",
        "bb_upper",
        "bb_lower",
        "roc10",
        "obv",
        "volatility20",
    ]

    indicators = {
        name: _safe_float(last.get(name))
        for name in indicator_names
    }

    expected_return = (p_up - p_down) * 0.05

    return SwingPrediction(
        signal=signal,
        probability_up=p_up,
        probability_down=p_down,
        confidence=confidence,
        expected_return=expected_return,
        entry=entry,
        stop_loss=stop,
        target=target,
        risk_reward=risk_reward,
        regime=regime,
        indicators=indicators,
        model_training_accuracy=None,
        signal_reason=signal_reason,
        trend=trend,
        momentum=momentum,
        volume_status=volume_status,
    )
    
