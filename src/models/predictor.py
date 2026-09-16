"""
Swing Stock Predictor

Production prediction layer connecting:
    Data
    -> Feature Engineering
    -> Ensemble ML
    -> Market Regime
    -> Multi-Timeframe Analysis
    -> Risk / Trade Plan

The classifier's target construction excludes the most recent
unlabeled observations, so the latest prediction is generated
without using its future outcome during training.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.features.engine import build_features
from src.features.regime import market_regime
from src.models.classifier import SwingClassifier


# =====================================================================
# RESULT OBJECT
# =====================================================================


class PredictionResult(dict):
    """Dictionary with attribute-style access."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


# =====================================================================
# SAFE HELPERS
# =====================================================================


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:
        result = float(value)

        if np.isfinite(result):
            return result

    except Exception:
        pass

    return float(default)


def _latest_value(
    df: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> float:

    if df is None or df.empty:
        return default

    if column not in df.columns:
        return default

    value = df[column].iloc[-1]

    return _safe_float(
        value,
        default,
    )


def _latest_text(
    df: pd.DataFrame,
    column: str,
    default: str = "UNKNOWN",
) -> str:

    if df is None or df.empty:
        return default

    if column not in df.columns:
        return default

    value = df[column].iloc[-1]

    if pd.isna(value):
        return default

    return str(value)


def _latest_close(
    df: pd.DataFrame,
) -> float:

    if "close" not in df.columns:
        raise ValueError(
            "Input data must contain a 'close' column."
        )

    value = pd.to_numeric(
        df["close"],
        errors="coerce",
    ).dropna()

    if value.empty:
        raise ValueError(
            "No valid closing-price data available."
        )

    return float(value.iloc[-1])


# =====================================================================
# MARKET STRUCTURE
# =====================================================================


def _calculate_market_structure(
    features: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Calculate a transparent market-structure summary.

    This is descriptive context and does not override the ML model.
    """

    if features is None or features.empty:
        return {
            "trend": "UNKNOWN",
            "momentum": "UNKNOWN",
            "volume": "UNKNOWN",
            "trend_score": 0.0,
            "momentum_score": 0.0,
            "volume_score": 0.0,
            "adx": 0.0,
        }

    # ---------------------------------------------------------------
    # TREND
    # ---------------------------------------------------------------

    close = _latest_value(
        features,
        "close",
        0.0,
    )

    ema20 = _latest_value(
        features,
        "ema20",
        0.0,
    )

    ema50 = _latest_value(
        features,
        "ema50",
        0.0,
    )

    ema200 = _latest_value(
        features,
        "ema200",
        0.0,
    )

    trend_score = 0.0

    if close > 0 and ema20 > 0:
        trend_score += 1.0 if close > ema20 else -1.0

    if ema20 > 0 and ema50 > 0:
        trend_score += 1.0 if ema20 > ema50 else -1.0

    if ema50 > 0 and ema200 > 0:
        trend_score += 1.0 if ema50 > ema200 else -1.0

    if trend_score >= 2:
        trend = "BULLISH"
    elif trend_score <= -2:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"

    # ---------------------------------------------------------------
    # MOMENTUM
    # ---------------------------------------------------------------

    rsi = _latest_value(
        features,
        "rsi14",
        50.0,
    )

    macd_hist = _latest_value(
        features,
        "macd_hist",
        0.0,
    )

    momentum_score = 0.0

    if rsi >= 55:
        momentum_score += 1.0
    elif rsi <= 45:
        momentum_score -= 1.0

    if macd_hist > 0:
        momentum_score += 1.0
    elif macd_hist < 0:
        momentum_score -= 1.0

    if momentum_score >= 1:
        momentum = "POSITIVE"
    elif momentum_score <= -1:
        momentum = "NEGATIVE"
    else:
        momentum = "NEUTRAL"

    # ---------------------------------------------------------------
    # VOLUME
    # ---------------------------------------------------------------

    relative_volume = _latest_value(
        features,
        "relative_volume",
        np.nan,
    )

    volume_score = 0.0

    if np.isfinite(relative_volume):
        if relative_volume >= 1.20:
            volume_score = 1.0
        elif relative_volume <= 0.80:
            volume_score = -1.0

    if volume_score > 0:
        volume = "EXPANDING"
    elif volume_score < 0:
        volume = "WEAK"
    else:
        volume = "NORMAL"

    # ---------------------------------------------------------------
    # ADX
    # ---------------------------------------------------------------

    adx = _latest_value(
        features,
        "adx",
        0.0,
    )

    return {
        "trend": trend,
        "momentum": momentum,
        "volume": volume,
        "trend_score": float(trend_score),
        "momentum_score": float(momentum_score),
        "volume_score": float(volume_score),
        "adx": float(adx),
    }


# =====================================================================
# TRADE PLAN
# =====================================================================


def _build_trade_plan(
    current_price: float,
    signal: str,
    stop_loss_pct: float,
    target_pct: float,
) -> Dict[str, Any]:

    if current_price <= 0:
        return {
            "entry": None,
            "stop_loss": None,
            "target": None,
            "risk_percent": stop_loss_pct * 100,
            "reward_percent": target_pct * 100,
            "risk_reward": (
                target_pct / stop_loss_pct
                if stop_loss_pct > 0
                else 0.0
            ),
        }

    entry = current_price

    if signal == "BUY":
        stop_loss = entry * (
            1.0 - stop_loss_pct
        )

        target = entry * (
            1.0 + target_pct
        )

    elif signal == "SELL":
        stop_loss = entry * (
            1.0 + stop_loss_pct
        )

        target = entry * (
            1.0 - target_pct
        )

    else:
        stop_loss = None
        target = None

    return {
        "entry": float(entry),
        "stop_loss": (
            float(stop_loss)
            if stop_loss is not None
            else None
        ),
        "target": (
            float(target)
            if target is not None
            else None
        ),
        "risk_percent": float(
            stop_loss_pct * 100
        ),
        "reward_percent": float(
            target_pct * 100
        ),
        "risk_reward": float(
            target_pct / stop_loss_pct
            if stop_loss_pct > 0
            else 0.0
        ),
    }


# =====================================================================
# SIGNAL ENGINE
# =====================================================================


def _generate_signal(
    probability_up: float,
    threshold: float,
) -> str:
    """
    Convert probability into BUY / SELL / WAIT.

    The region between:
        1 - threshold
    and:
        threshold

    is intentionally treated as WAIT.
    """

    probability_up = _safe_float(
        probability_up,
        0.5,
    )

    threshold = min(
        max(float(threshold), 0.50),
        0.95,
    )

    lower_threshold = 1.0 - threshold

    if probability_up >= threshold:
        return "BUY"

    if probability_up <= lower_threshold:
        return "SELL"

    return "WAIT"


# =====================================================================
# MAIN ANALYZER
# =====================================================================


def analyze_stock(
    df: pd.DataFrame,
    horizon: int = 5,
    probability_threshold: float = 0.60,
    stop_loss_pct: float = 0.03,
    target_pct: float = 0.06,
) -> PredictionResult:

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Input DataFrame is empty."
        )

    horizon = int(horizon)

    if horizon < 1:
        raise ValueError(
            "Prediction horizon must be >= 1."
        )

    probability_threshold = float(
        probability_threshold
    )

    if not 0.50 <= probability_threshold < 1.0:
        raise ValueError(
            "Probability threshold must be between "
            "0.50 and 0.99."
        )

    # ---------------------------------------------------------------
    # FEATURE ENGINEERING
    # ---------------------------------------------------------------

    features = build_features(df)

    if features is None or features.empty:
        raise ValueError(
            "Feature engineering produced no data."
        )

    # ---------------------------------------------------------------
    # ENSEMBLE
    # ---------------------------------------------------------------

    model = SwingClassifier(
        horizon=horizon,
        probability_threshold=probability_threshold,
    )

    model.fit(features)

    # ---------------------------------------------------------------
    # LATEST PREDICTION
    # ---------------------------------------------------------------

    probabilities = model.predict_proba(
        features
    )

    if len(probabilities) == 0:
        raise ValueError(
            "The model produced no prediction."
        )

    probability_down = _safe_float(
        probabilities[-1, 0],
        0.5,
    )

    probability_up = _safe_float(
        probabilities[-1, 1],
        0.5,
    )

    # Normalize in case of tiny floating-point error.
    probability_sum = (
        probability_up
        + probability_down
    )

    if probability_sum > 0:
        probability_up /= probability_sum
        probability_down /= probability_sum

    confidence = abs(
        probability_up
        - probability_down
    )

    uncertainty = 1.0 - confidence

    signal = _generate_signal(
        probability_up,
        probability_threshold,
    )

    # ---------------------------------------------------------------
    # MARKET REGIME
    # ---------------------------------------------------------------

    try:
        regime = market_regime(
            features
        )
    except Exception:
        regime = "UNKNOWN"

    if isinstance(regime, pd.Series):
        regime = (
            str(regime.iloc[-1])
            if len(regime)
            else "UNKNOWN"
        )

    if isinstance(regime, pd.DataFrame):
        if not regime.empty:
            regime = str(
                regime.iloc[-1, 0]
            )
        else:
            regime = "UNKNOWN"

    regime = str(regime)

    # ---------------------------------------------------------------
    # MULTI-TIMEFRAME
    # ---------------------------------------------------------------

    weekly_trend = _latest_text(
        features,
        "weekly_trend",
        "UNKNOWN",
    )

    monthly_trend = _latest_text(
        features,
        "monthly_trend",
        "UNKNOWN",
    )

    higher_timeframe_score = _latest_value(
        features,
        "higher_timeframe_score",
        0.0,
    )

    three_timeframe_alignment = _latest_text(
        features,
        "three_timeframe_alignment",
        "UNKNOWN",
    )

    # ---------------------------------------------------------------
    # MARKET STRUCTURE
    # ---------------------------------------------------------------

    structure = _calculate_market_structure(
        features
    )

    # ---------------------------------------------------------------
    # TRADE PLAN
    # ---------------------------------------------------------------

    current_price = _latest_close(
        features
    )

    trade_plan = _build_trade_plan(
        current_price=current_price,
        signal=signal,
        stop_loss_pct=stop_loss_pct,
        target_pct=target_pct,
    )

    # ---------------------------------------------------------------
    # MODEL AGREEMENT
    # ---------------------------------------------------------------

    try:
        model_agreement = (
            model.model_agreement(features)
        )
    except Exception:
        model_agreement = {
            "agreement": 0.0,
            "disagreement": 1.0,
            "bullish_models": 0,
            "bearish_models": 0,
            "total_models": 0,
            "status": "UNKNOWN",
        }

    try:
        component_probabilities = (
            model.component_probabilities(
                features
            )
        )
    except Exception:
        component_probabilities = {}

    # ---------------------------------------------------------------
    # CONFIDENCE
    # ---------------------------------------------------------------

    confidence_metrics = (
        model.confidence_metrics(
            features
        )
    )

    # ---------------------------------------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------------------------------------

    try:
        importance_df = (
            model.feature_importance()
        )

        top_features = (
            importance_df
            .head(10)
            .to_dict("records")
        )

        feature_importance = (
            importance_df.to_dict(
                "records"
            )
        )

    except Exception:
        top_features = []
        feature_importance = []

    # ---------------------------------------------------------------
    # MODEL SUMMARY
    # ---------------------------------------------------------------

    model_summary = model.summary()

    validation_accuracy = (
        model_summary.get(
            "validation_accuracy"
        )
    )

    validation_roc_auc = (
        model_summary.get(
            "validation_roc_auc"
        )
    )

    validation_brier = (
        model_summary.get(
            "validation_brier"
        )
    )

    # ---------------------------------------------------------------
    # RESULT
    # ---------------------------------------------------------------

    result = PredictionResult(
        {
            "signal": signal,

            "probability_up": float(
                probability_up
            ),

            "probability_down": float(
                probability_down
            ),

            # Explicit OOS naming.
            #
            # The classifier creates its target using future close.
            # Therefore the latest row has no target and is not part
            # of the fitted training observations.
            "prediction_probability_oos": float(
                probability_up
            ),

            # Backward compatibility.
            "prediction_probability": float(
                probability_up
            ),

            "confidence": float(
                confidence
            ),

            "uncertainty": float(
                uncertainty
            ),

            "horizon": int(
                horizon
            ),

            "current_price": float(
                current_price
            ),

            "price": float(
                current_price
            ),

            # Market regime.
            "regime": regime,

            # Multi-timeframe.
            "weekly_trend": weekly_trend,
            "monthly_trend": monthly_trend,

            "higher_timeframe_score": float(
                higher_timeframe_score
            ),

            "three_timeframe_alignment": (
                three_timeframe_alignment
            ),

            # Market structure.
            "market_structure": structure,

            "trend": structure[
                "trend"
            ],

            "momentum": structure[
                "momentum"
            ],

            "volume": structure[
                "volume"
            ],

            "trend_score": structure[
                "trend_score"
            ],

            "momentum_score": structure[
                "momentum_score"
            ],

            "volume_score": structure[
                "volume_score"
            ],

            # Trading plan.
            "trade_plan": trade_plan,

            "entry": trade_plan[
                "entry"
            ],

            "stop_loss": trade_plan[
                "stop_loss"
            ],

            "target": trade_plan[
                "target"
            ],

            # Model agreement.
            "model_agreement": model_agreement,

            "component_probabilities": (
                component_probabilities
            ),

            # Confidence diagnostics.
            "confidence_metrics": (
                confidence_metrics
            ),

            # Indicators.
            "indicators": {
                "rsi14": _latest_value(
                    features,
                    "rsi14",
                    np.nan,
                ),
                "macd": _latest_value(
                    features,
                    "macd",
                    np.nan,
                ),
                "macd_signal": _latest_value(
                    features,
                    "macd_signal",
                    np.nan,
                ),
                "macd_hist": _latest_value(
                    features,
                    "macd_hist",
                    np.nan,
                ),
                "atr14": _latest_value(
                    features,
                    "atr14",
                    np.nan,
                ),
                "atr_pct": _latest_value(
                    features,
                    "atr_pct",
                    np.nan,
                ),
                "ema20": _latest_value(
                    features,
                    "ema20",
                    np.nan,
                ),
                "ema50": _latest_value(
                    features,
                    "ema50",
                    np.nan,
                ),
                "ema200": _latest_value(
                    features,
                    "ema200",
                    np.nan,
                ),
                "relative_volume": _latest_value(
                    features,
                    "relative_volume",
                    np.nan,
                ),
                "adx": _latest_value(
                    features,
                    "adx",
                    np.nan,
                ),
            },

            # ML diagnostics.
            "top_features": top_features,

            "feature_importance": (
                feature_importance
            ),

            "model_training_accuracy": (
                model_summary.get(
                    "training_accuracy"
                )
            ),

            "training_accuracy": (
                model_summary.get(
                    "training_accuracy"
                )
            ),

            "validation_accuracy": (
                validation_accuracy
            ),

            "validation_precision": (
                model_summary.get(
                    "validation_precision"
                )
            ),

            "validation_recall": (
                model_summary.get(
                    "validation_recall"
                )
            ),

            "validation_f1": (
                model_summary.get(
                    "validation_f1"
                )
            ),

            "validation_roc_auc": (
                validation_roc_auc
            ),

            "validation_brier": (
                validation_brier
            ),

            "model_summary": model_summary,

            # Dataset diagnostics.
            "n_rows": int(
                len(features)
            ),

            "n_features": int(
                len(
                    model.feature_names_
                )
            ),

            "features": features,

            "model": model,
        }
    )

    return result


# =====================================================================
# BACKWARD COMPATIBILITY
# =====================================================================


predict_stock = analyze_stock


__all__ = [
    "PredictionResult",
    "analyze_stock",
    "predict_stock",
]
