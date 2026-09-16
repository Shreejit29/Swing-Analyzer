# src/models/predictor.py

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.features.engine import build_features
from src.features.regime import market_regime
from src.models.classifier import SwingClassifier


class PredictionResult(dict):
    """
    Dictionary-like prediction result that also supports attribute access.

    Example:
        result["probability_up"]
        result.probability_up
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"PredictionResult has no attribute '{name}'"
            ) from None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


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


def _latest_value(
    df: pd.DataFrame,
    columns: list[str],
    default: float = 0.0,
) -> float:
    for col in columns:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if not series.empty:
                return _safe_float(series.iloc[-1], default)
    return default


def _latest_text(
    df: pd.DataFrame,
    columns: list[str],
    default: str = "UNKNOWN",
) -> str:
    for col in columns:
        if col in df.columns:
            series = df[col].dropna()
            if not series.empty:
                value = str(series.iloc[-1]).strip()
                if value and value.lower() not in {"nan", "none"}:
                    return value.upper()
    return default


def _latest_close(df: pd.DataFrame) -> float:
    for col in ["close", "Close"]:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if not series.empty:
                return float(series.iloc[-1])

    raise ValueError("No valid close price found.")


# ---------------------------------------------------------------------
# Market Structure
# ---------------------------------------------------------------------

def _calculate_market_structure(features: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate the market structure displayed by the Stock Analyzer.

    Returns:
        {
            "trend": ...,
            "momentum": ...,
            "volume": ...,
            "trend_score": ...,
            "momentum_score": ...,
            "volume_score": ...,
        }
    """

    if features is None or features.empty:
        return {
            "trend": "UNKNOWN",
            "momentum": "UNKNOWN",
            "volume": "UNKNOWN",
            "trend_score": 0.0,
            "momentum_score": 0.0,
            "volume_score": 0.0,
        }

    row = features.iloc[-1]

    # -------------------------------------------------------------
    # TREND
    # -------------------------------------------------------------

    trend_score = 0.0

    close = _safe_float(row.get("close"), np.nan)

    ema20 = _safe_float(row.get("ema20"), np.nan)
    ema50 = _safe_float(row.get("ema50"), np.nan)
    ema200 = _safe_float(row.get("ema200"), np.nan)

    ema20_slope = _safe_float(
        row.get("ema20_slope", row.get("ema20_slope_pct", 0.0))
    )

    ema50_slope = _safe_float(
        row.get("ema50_slope", row.get("ema50_slope_pct", 0.0))
    )

    adx = _safe_float(row.get("adx14", row.get("adx", 0.0)))

    # Price relative to moving averages
    if np.isfinite(close) and np.isfinite(ema20):
        trend_score += 1.0 if close > ema20 else -1.0

    if np.isfinite(close) and np.isfinite(ema50):
        trend_score += 1.0 if close > ema50 else -1.0

    if np.isfinite(close) and np.isfinite(ema200):
        trend_score += 1.0 if close > ema200 else -1.0

    # EMA alignment
    if np.isfinite(ema20) and np.isfinite(ema50):
        trend_score += 1.0 if ema20 > ema50 else -1.0

    if np.isfinite(ema50) and np.isfinite(ema200):
        trend_score += 1.0 if ema50 > ema200 else -1.0

    # EMA slopes
    if ema20_slope > 0:
        trend_score += 0.75
    elif ema20_slope < 0:
        trend_score -= 0.75

    if ema50_slope > 0:
        trend_score += 0.50
    elif ema50_slope < 0:
        trend_score -= 0.50

    if trend_score >= 3.0:
        trend = "BULLISH"
    elif trend_score <= -3.0:
        trend = "BEARISH"
    else:
        trend = "SIDEWAYS"

    # -------------------------------------------------------------
    # MOMENTUM
    # -------------------------------------------------------------

    momentum_score = 0.0

    rsi = _safe_float(row.get("rsi14", row.get("rsi", 50.0)), 50.0)

    macd_hist = _safe_float(
        row.get("macd_hist", row.get("macd_histogram", 0.0))
    )

    roc10 = _safe_float(row.get("roc10", 0.0))
    roc20 = _safe_float(row.get("roc20", 0.0))

    # RSI
    if rsi >= 55:
        momentum_score += 1.0
    elif rsi <= 45:
        momentum_score -= 1.0

    # MACD histogram
    if macd_hist > 0:
        momentum_score += 1.0
    elif macd_hist < 0:
        momentum_score -= 1.0

    # Rate of change
    if roc10 > 0:
        momentum_score += 0.75
    elif roc10 < 0:
        momentum_score -= 0.75

    if roc20 > 0:
        momentum_score += 0.50
    elif roc20 < 0:
        momentum_score -= 0.50

    if momentum_score >= 1.5:
        momentum = "BULLISH"
    elif momentum_score <= -1.5:
        momentum = "BEARISH"
    else:
        momentum = "NEUTRAL"

    # -------------------------------------------------------------
    # VOLUME
    # -------------------------------------------------------------

    volume_score = 0.0

    relative_volume = _safe_float(
        row.get(
            "relative_volume",
            row.get("volume_ratio", 1.0),
        ),
        1.0,
    )

    volume_change = _safe_float(
        row.get("volume_change", 0.0)
    )

    volume_roc = _safe_float(
        row.get("volume_roc", 0.0)
    )

    obv_slope = _safe_float(
        row.get("obv_slope", 0.0)
    )

    price_volume_sign = _safe_float(
        row.get("price_volume_sign", 0.0)
    )

    # Relative volume
    if relative_volume >= 1.20:
        volume_score += 1.0
    elif relative_volume < 0.80:
        volume_score -= 0.50

    # Volume direction
    if volume_change > 0:
        volume_score += 0.50
    elif volume_change < 0:
        volume_score -= 0.25

    if volume_roc > 0:
        volume_score += 0.50
    elif volume_roc < 0:
        volume_score -= 0.25

    # OBV direction
    if obv_slope > 0:
        volume_score += 0.75
    elif obv_slope < 0:
        volume_score -= 0.75

    # Price-volume confirmation
    if price_volume_sign > 0:
        volume_score += 0.50
    elif price_volume_sign < 0:
        volume_score -= 0.50

    if volume_score >= 1.25:
        volume = "ACCUMULATION"
    elif volume_score <= -1.0:
        volume = "DISTRIBUTION"
    else:
        volume = "NEUTRAL"

    # -------------------------------------------------------------
    # Return
    # -------------------------------------------------------------

    return {
        "trend": trend,
        "momentum": momentum,
        "volume": volume,
        "trend_score": round(float(trend_score), 3),
        "momentum_score": round(float(momentum_score), 3),
        "volume_score": round(float(volume_score), 3),
        "adx": round(float(adx), 3),
    }


# ---------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------

def _generate_signal(
    probability_up: float,
    alignment: float,
    threshold: float,
) -> str:

    probability_up = _safe_float(probability_up, 0.5)
    alignment = _safe_float(alignment, 0.0)

    if probability_up >= threshold and alignment >= 0:
        return "BUY"

    if probability_up <= (1.0 - threshold) and alignment <= 0:
        return "SELL"

    return "WAIT"


# ---------------------------------------------------------------------
# Trade Plan
# ---------------------------------------------------------------------

def _create_trade_plan(
    price: float,
    signal: str,
    stop_loss_pct: float,
    target_pct: float,
) -> Dict[str, Optional[float]]:

    price = _safe_float(price)

    if price <= 0:
        return {
            "entry": None,
            "stop_loss": None,
            "target": None,
        }

    if signal == "BUY":
        return {
            "entry": price,
            "stop_loss": price * (1.0 - stop_loss_pct),
            "target": price * (1.0 + target_pct),
        }

    if signal == "SELL":
        return {
            "entry": price,
            "stop_loss": price * (1.0 + stop_loss_pct),
            "target": price * (1.0 - target_pct),
        }

    return {
        "entry": price,
        "stop_loss": None,
        "target": None,
    }


# ---------------------------------------------------------------------
# Main Analyzer
# ---------------------------------------------------------------------

def analyze_stock(
    df: pd.DataFrame,
    horizon: int = 5,
    probability_threshold: float = 0.60,
    stop_loss_pct: float = 0.03,
    target_pct: float = 0.06,
) -> PredictionResult:

    if df is None or df.empty:
        raise ValueError("Input stock data is empty.")

    horizon = int(horizon)

    if horizon <= 0:
        raise ValueError("Prediction horizon must be greater than zero.")

    if not 0.50 <= probability_threshold < 1.0:
        raise ValueError(
            "Probability threshold must be between 0.50 and 1.00."
        )

    # -------------------------------------------------------------
    # Feature engineering
    # -------------------------------------------------------------

    features = build_features(df)

    if features is None or features.empty:
        raise ValueError("Feature engineering returned no data.")

    # -------------------------------------------------------------
    # Train model
    # -------------------------------------------------------------

    model = SwingClassifier(
        horizon=horizon,
        probability_threshold=probability_threshold,
    )

    model.fit(features)

    # -------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------

    probabilities = model.predict_proba(features)

    if probabilities is None:
        raise ValueError("Model did not return probabilities.")

    probabilities = np.asarray(probabilities)

    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(
            "Unexpected probability output from classifier."
        )

    latest_probability_down = _safe_float(
        probabilities[-1, 0],
        0.5,
    )

    latest_probability_up = _safe_float(
        probabilities[-1, 1],
        0.5,
    )

    # -------------------------------------------------------------
    # Regime
    # -------------------------------------------------------------

    try:
        regime = market_regime(features)
    except Exception:
        regime = "UNKNOWN"

    if isinstance(regime, pd.Series):
        regime = str(regime.iloc[-1])
    elif isinstance(regime, (list, tuple, np.ndarray)):
        regime = str(regime[-1]) if len(regime) else "UNKNOWN"
    else:
        regime = str(regime)

    regime = regime.upper()

    # -------------------------------------------------------------
    # Higher timeframe information
    # -------------------------------------------------------------

    weekly_trend = _latest_text(
        features,
        [
            "weekly_trend",
            "weekly_direction",
            "weekly_regime",
        ],
        "UNKNOWN",
    )

    monthly_trend = _latest_text(
        features,
        [
            "monthly_trend",
            "monthly_direction",
            "monthly_regime",
        ],
        "UNKNOWN",
    )

    higher_timeframe_score = _latest_value(
        features,
        [
            "higher_timeframe_score",
            "mtf_score",
        ],
        0.0,
    )

    three_timeframe_alignment = _latest_value(
        features,
        [
            "three_timeframe_alignment",
            "mtf_alignment",
            "higher_timeframe_alignment",
        ],
        0.0,
    )

    # -------------------------------------------------------------
    # Market Structure
    # -------------------------------------------------------------

    market_structure = _calculate_market_structure(features)

    trend = market_structure["trend"]
    momentum = market_structure["momentum"]
    volume = market_structure["volume"]

    # -------------------------------------------------------------
    # Final signal
    # -------------------------------------------------------------

    signal = _generate_signal(
        latest_probability_up,
        three_timeframe_alignment,
        probability_threshold,
    )

    # -------------------------------------------------------------
    # Current price
    # -------------------------------------------------------------

    current_price = _latest_close(features)

    # -------------------------------------------------------------
    # Trade plan
    # -------------------------------------------------------------

    trade_plan = _create_trade_plan(
        current_price,
        signal,
        float(stop_loss_pct),
        float(target_pct),
    )

    # -------------------------------------------------------------
    # Indicators
    # -------------------------------------------------------------

    indicator_names = [
        "rsi14",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr14",
        "atr_pct",
        "ema20",
        "ema50",
        "ema200",
        "adx14",
        "plus_di",
        "minus_di",
        "relative_volume",
        "obv",
        "roc10",
        "roc20",
    ]

    indicators: Dict[str, float] = {}

    for name in indicator_names:
        if name in features.columns:
            value = _safe_float(features[name].iloc[-1], np.nan)

            if np.isfinite(value):
                indicators[name] = value

    # -------------------------------------------------------------
    # Feature importance
    # -------------------------------------------------------------

    try:
        feature_importance = model.feature_importance()

        if isinstance(feature_importance, pd.DataFrame):
            top_features = feature_importance.head(10)
        elif isinstance(feature_importance, dict):
            top_features = dict(
                sorted(
                    feature_importance.items(),
                    key=lambda x: abs(_safe_float(x[1])),
                    reverse=True,
                )[:10]
            )
        else:
            top_features = feature_importance

    except Exception:
        feature_importance = {}
        top_features = {}

    # -------------------------------------------------------------
    # Model accuracy
    # -------------------------------------------------------------

    training_accuracy = _safe_float(
        getattr(model, "training_accuracy", 0.0),
        0.0,
    )

    # -------------------------------------------------------------
    # Confidence
    # -------------------------------------------------------------

    confidence = abs(
        latest_probability_up - latest_probability_down
    )

    # -------------------------------------------------------------
    # Result
    # -------------------------------------------------------------

    result = PredictionResult()

    result.signal = signal

    result.probability_up = latest_probability_up
    result.probability_down = latest_probability_down
    result.prediction_probability = latest_probability_up
    result.confidence = confidence

    result.horizon = horizon

    result.current_price = current_price
    result.price = current_price

    result.regime = regime

    result.weekly_trend = weekly_trend
    result.monthly_trend = monthly_trend

    result.higher_timeframe_score = higher_timeframe_score
    result.three_timeframe_alignment = three_timeframe_alignment

    # Market Structure
    result.market_structure = market_structure

    # Direct compatibility attributes for the UI
    result.trend = trend
    result.momentum = momentum
    result.volume = volume

    result.trend_score = market_structure["trend_score"]
    result.momentum_score = market_structure["momentum_score"]
    result.volume_score = market_structure["volume_score"]

    # Trade plan
    result.trade_plan = trade_plan

    result.entry = trade_plan.get("entry")
    result.stop_loss = trade_plan.get("stop_loss")
    result.target = trade_plan.get("target")

    # Indicators
    result.indicators = indicators

    # Features / importance
    result.top_features = top_features
    result.feature_importance = feature_importance

    # Model information
    result.model_training_accuracy = training_accuracy
    result.training_accuracy = training_accuracy

    result.n_rows = int(len(features))

    try:
        result.n_features = int(len(model.feature_names_))
    except Exception:
        result.n_features = 0

    result.features = features
    result.model = model

    return result


# ---------------------------------------------------------------------
# Compatibility alias
# ---------------------------------------------------------------------

predict_stock = analyze_stock


__all__ = [
    "PredictionResult",
    "analyze_stock",
    "predict_stock",
]
