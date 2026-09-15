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


def analyze_stock(df: pd.DataFrame, horizon: int = 5, probability_threshold: float = 0.60, stop_loss_pct: float = 0.03, target_pct: float = 0.06) -> SwingPrediction:
    features = build_features(df)
    model = SwingClassifier(horizon=horizon, probability_threshold=probability_threshold)
    model.fit(df)
    p_up = model.predict_proba(df)
    p_down = 1 - p_up
    last = features.iloc[-1]
    close = float(last["close"])
    expected_return = (p_up - p_down) * 0.05
    if p_up >= probability_threshold and p_up >= p_down:
        signal = "BUY"
        entry = close
        stop = close * (1 - stop_loss_pct)
        target = close * (1 + target_pct)
    elif p_down >= probability_threshold and p_down > p_up:
        signal = "SELL"
        entry = close
        stop = close * (1 + stop_loss_pct)
        target = close * (1 - target_pct)
    else:
        signal = "WAIT"
        entry = close
        stop = close
        target = close
    rr = target_pct / stop_loss_pct if stop_loss_pct > 0 else math.inf
    confidence = "HIGH" if max(p_up, p_down) >= 0.70 else "MEDIUM" if max(p_up, p_down) >= probability_threshold else "LOW"
    indicators = {k: float(last[k]) if pd.notna(last[k]) and np.isscalar(last[k]) else None for k in ["rsi14", "adx14", "atr14", "macd", "macd_signal", "relative_volume", "ema20", "ema50", "ema200", "bb_upper", "bb_lower"]}
    return SwingPrediction(signal, p_up, p_down, confidence, expected_return, entry, stop, target, rr, market_regime(last), indicators, model.training_accuracy)
