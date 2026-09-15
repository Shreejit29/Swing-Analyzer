"""
Prediction layer for the AI Swing Stock Analyzer.

Pipeline
--------
OHLCV
  ↓
Complete feature engineering
  ↓
Gradient Boosting
  ↓
Probability
  ↓
BUY / WAIT / SELL
  ↓
Trade plan
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.engine import build_features
from src.features.regime import market_regime
from src.models.classifier import SwingClassifier


# ============================================================
# HELPERS
# ============================================================

def _get_close(
    df: pd.DataFrame,
) -> float:
    """Return the latest close price."""

    if "close" in df.columns:

        value = df["close"]

        if isinstance(
            value,
            pd.DataFrame,
        ):
            value = value.iloc[:, 0]

        series = pd.to_numeric(
            value,
            errors="coerce",
        )

        if not series.dropna().empty:
            return float(
                series.dropna().iloc[-1]
            )

    # Case-insensitive fallback.
    for column in df.columns:

        if str(column).strip().lower() in {
            "close",
            "close_price",
            "adj close",
            "adj_close",
        }:

            value = df[column]

            if isinstance(
                value,
                pd.DataFrame,
            ):
                value = value.iloc[:, 0]

            series = pd.to_numeric(
                value,
                errors="coerce",
            )

            if not series.dropna().empty:
                return float(
                    series.dropna().iloc[-1]
                )

    raise ValueError(
        "Could not find a valid close price."
    )


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Safely convert a value to float."""

    try:

        result = float(value)

        if np.isfinite(result):
            return result

    except Exception:
        pass

    return default


def _latest_value(
    df: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> float:
    """Return the latest numeric value of a column."""

    if column not in df.columns:
        return default

    value = df[column]

    if isinstance(
        value,
        pd.DataFrame,
    ):
        value = value.iloc[:, 0]

    series = pd.to_numeric(
        value,
        errors="coerce",
    ).dropna()

    if series.empty:
        return default

    return _safe_float(
        series.iloc[-1],
        default,
    )


# ============================================================
# SIGNAL LOGIC
# ============================================================

def _generate_signal(
    probability: float,
    regime: str,
    alignment: float,
    threshold: float,
) -> str:
    """
    Convert model probability and market context into
    BUY / WAIT / SELL.

    The model probability is the primary signal.

    Higher-timeframe alignment is used as a confirmation
    filter rather than replacing the ML prediction.
    """

    # Strong bearish environment.
    if (
        regime in {
            "STRONG BEAR",
            "BEAR",
        }
        and probability < threshold
    ):
        return "SELL"

    # Strong bullish probability.
    if probability >= threshold:

        # If higher timeframes strongly disagree,
        # avoid forcing a BUY.
        if alignment < 0:
            return "WAIT"

        return "BUY"

    # Very low bullish probability.
    if probability <= (
        1.0 - threshold
    ):

        if alignment > 0:
            return "WAIT"

        return "SELL"

    return "WAIT"


# ============================================================
# TRADE PLAN
# ============================================================

def _trade_plan(
    price: float,
    signal: str,
    stop_loss_pct: float,
    target_pct: float,
) -> dict[str, Any]:
    """Generate a simple long/short trade plan."""

    price = max(
        float(price),
        0.0,
    )

    if signal == "BUY":

        stop_loss = (
            price
            * (1.0 - stop_loss_pct)
        )

        target = (
            price
            * (1.0 + target_pct)
        )

        return {
            "entry": price,
            "stop_loss": stop_loss,
            "target": target,
            "risk_pct": (
                stop_loss_pct * 100.0
            ),
            "reward_pct": (
                target_pct * 100.0
            ),
        }

    if signal == "SELL":

        stop_loss = (
            price
            * (1.0 + stop_loss_pct)
        )

        target = (
            price
            * (1.0 - target_pct)
        )

        return {
            "entry": price,
            "stop_loss": stop_loss,
            "target": target,
            "risk_pct": (
                stop_loss_pct * 100.0
            ),
            "reward_pct": (
                target_pct * 100.0
            ),
        }

    return {
        "entry": price,
        "stop_loss": None,
        "target": None,
        "risk_pct": 0.0,
        "reward_pct": 0.0,
    }


# ============================================================
# MAIN ANALYZER
# ============================================================

def analyze_stock(
    df: pd.DataFrame,
    horizon: int = 5,
    probability_threshold: float = 0.60,
    stop_loss_pct: float = 0.03,
    target_pct: float = 0.06,
    min_train_samples: int = 30,
) -> dict[str, Any]:
    """
    Analyze a stock and generate an ML-based swing signal.

    Parameters
    ----------
    df:
        OHLCV market data.

    horizon:
        Prediction horizon in trading days.

    probability_threshold:
        Probability required for a directional signal.

    stop_loss_pct:
        Internal stop-loss percentage.

    target_pct:
        Internal target percentage.

    min_train_samples:
        Minimum number of samples required for model training.

    Returns
    -------
    dict
        Prediction, probability, regime and trade-plan information.
    """

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "No market data available."
        )

    horizon = int(
        horizon
    )

    if horizon < 1:
        raise ValueError(
            "horizon must be at least 1."
        )

    probability_threshold = float(
        probability_threshold
    )

    if not (
        0.50
        <= probability_threshold
        <= 1.0
    ):
        raise ValueError(
            "probability_threshold must be between 0.50 and 1.0."
        )

    # --------------------------------------------------------
    # Build complete feature set.
    # --------------------------------------------------------

    features = build_features(
        df
    )

    if features.empty:
        raise ValueError(
            "Feature engineering produced no usable data."
        )

    # --------------------------------------------------------
    # Remove infinities.
    # --------------------------------------------------------

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # --------------------------------------------------------
    # Train the classifier.
    #
    # The classifier internally creates the future target.
    # --------------------------------------------------------

    model = SwingClassifier(
        horizon=horizon,
        probability_threshold=(
            probability_threshold
        ),
        random_state=42,
        min_samples=min_train_samples,
    )

    model.fit(
        features
    )

    # --------------------------------------------------------
    # Latest prediction.
    # --------------------------------------------------------

    probability_array = (
        model.predict_proba(
            features
        )
    )

    probability_array = np.asarray(
        probability_array,
        dtype=float,
    )

    if (
        probability_array.ndim == 2
        and probability_array.shape[1] >= 2
    ):

        p_down = float(
            probability_array[
                -1,
                0,
            ]
        )

        p_up = float(
            probability_array[
                -1,
                1,
            ]
        )

    elif probability_array.ndim == 1:

        p_up = float(
            probability_array[-1]
        )

        p_down = (
            1.0 - p_up
        )

    else:

        p_up = 0.5
        p_down = 0.5

    p_up = float(
        np.clip(
            p_up,
            0.0,
            1.0,
        )
    )

    p_down = float(
        np.clip(
            p_down,
            0.0,
            1.0,
        )
    )

    # --------------------------------------------------------
    # Market regime.
    # --------------------------------------------------------

    try:

        regime = market_regime(
            features
        )

    except Exception:

        regime = "UNKNOWN"

    # --------------------------------------------------------
    # Multi-timeframe context.
    # --------------------------------------------------------

    weekly_trend = _latest_value(
        features,
        "weekly_trend",
        0.0,
    )

    monthly_trend = _latest_value(
        features,
        "monthly_trend",
        0.0,
    )

    higher_timeframe_score = _latest_value(
        features,
        "higher_timeframe_score",
        0.0,
    )

    alignment = _latest_value(
        features,
        "three_timeframe_alignment",
        0.0,
    )

    # --------------------------------------------------------
    # Signal.
    # --------------------------------------------------------

    signal = _generate_signal(
        probability=p_up,
        regime=regime,
        alignment=alignment,
        threshold=probability_threshold,
    )

    # --------------------------------------------------------
    # Current price.
    # --------------------------------------------------------

    price = _get_close(
        features
    )

    # --------------------------------------------------------
    # Trade plan.
    # --------------------------------------------------------

    plan = _trade_plan(
        price=price,
        signal=signal,
        stop_loss_pct=stop_loss_pct,
        target_pct=target_pct,
    )

    # --------------------------------------------------------
    # Additional indicators.
    # --------------------------------------------------------

    indicators = {
        "rsi": _latest_value(
            features,
            "rsi14",
            _latest_value(
                features,
                "rsi",
                0.0,
            ),
        ),
        "macd": _latest_value(
            features,
            "macd",
            0.0,
        ),
        "atr_pct": _latest_value(
            features,
            "atr_pct",
            0.0,
        ),
        "relative_volume": _latest_value(
            features,
            "relative_volume",
            _latest_value(
                features,
                "volume_ratio",
                0.0,
            ),
        ),
        "adx": _latest_value(
            features,
            "adx",
            0.0,
        ),
        "ema20": _latest_value(
            features,
            "ema20",
            0.0,
        ),
        "ema50": _latest_value(
            features,
            "ema50",
            0.0,
        ),
        "ema200": _latest_value(
            features,
            "ema200",
            0.0,
        ),
    }

    # --------------------------------------------------------
    # Feature importance.
    # --------------------------------------------------------

    try:

        importance_df = (
            model.feature_importance()
        )

        top_features = (
            importance_df
            .head(10)
            .to_dict(
                orient="records"
            )
        )

    except Exception:

        importance_df = pd.DataFrame()

        top_features = []

    # --------------------------------------------------------
    # Model summary.
    # --------------------------------------------------------

    try:

        training_accuracy = (
            model.training_accuracy
        )

    except Exception:

        training_accuracy = None

    # --------------------------------------------------------
    # Result.
    # --------------------------------------------------------

    return {
        "signal": signal,

        "probability_up": p_up,

        "probability_down": p_down,

        "prediction_probability": p_up,

        "horizon": horizon,

        "current_price": price,

        "regime": regime,

        "weekly_trend": weekly_trend,

        "monthly_trend": monthly_trend,

        "higher_timeframe_score": (
            higher_timeframe_score
        ),

        "three_timeframe_alignment": (
            alignment
        ),

        "trade_plan": plan,

        "entry": plan["entry"],

        "stop_loss": plan["stop_loss"],

        "target": plan["target"],

        "indicators": indicators,

        "top_features": top_features,

        "feature_importance": (
            importance_df
        ),

        "model_training_accuracy": (
            training_accuracy
        ),

        "n_rows": len(features),

        "n_features": len(
            model.feature_names_
        ),

        "features": features,

        "model": model,
    }


# ============================================================
# BACKWARD-COMPATIBLE ALIASES
# ============================================================

predict_stock = analyze_stock


__all__ = [
    "analyze_stock",
    "predict_stock",
]
