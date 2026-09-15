"""
Prediction layer for the AI Swing Stock Analyzer.

Compatible with the existing Stock Analyzer UI.

Returns PredictionResult, which supports both:

    pred.probability_up
    pred["probability_up"]

and:

    pred.confidence
    pred.signal
    pred.regime
    pred.entry
    pred.stop_loss
    pred.target
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.engine import build_features
from src.features.regime import market_regime
from src.models.classifier import SwingClassifier


# ============================================================
# RESULT OBJECT
# ============================================================

class PredictionResult(dict):
    """
    Dictionary supporting attribute-style access.
    """

    def __getattr__(
        self,
        name: str,
    ) -> Any:

        try:
            return self[name]

        except KeyError as exc:

            raise AttributeError(
                f"PredictionResult has no attribute '{name}'"
            ) from exc

    def __setattr__(
        self,
        name: str,
        value: Any,
    ) -> None:

        self[name] = value


# ============================================================
# HELPERS
# ============================================================

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

    return default


def _latest_value(
    df: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> float:

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


def _latest_close(
    df: pd.DataFrame,
) -> float:

    if "close" in df.columns:

        value = df["close"]

        if isinstance(
            value,
            pd.DataFrame,
        ):
            value = value.iloc[:, 0]

        value = pd.to_numeric(
            value,
            errors="coerce",
        ).dropna()

        if not value.empty:
            return float(
                value.iloc[-1]
            )

    for column in df.columns:

        name = (
            str(column)
            .strip()
            .lower()
        )

        if name in {
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

            value = pd.to_numeric(
                value,
                errors="coerce",
            ).dropna()

            if not value.empty:
                return float(
                    value.iloc[-1]
                )

    raise ValueError(
        "Valid close price not found."
    )


# ============================================================
# SIGNAL LOGIC
# ============================================================

def _generate_signal(
    probability_up: float,
    alignment: float,
    threshold: float,
) -> str:
    """
    Generate BUY / SELL / WAIT.

    ML probability is the primary signal.
    Higher-timeframe alignment is used as confirmation.
    """

    probability_down = (
        1.0 - probability_up
    )

    if probability_up >= threshold:

        if alignment < 0:
            return "WAIT"

        return "BUY"

    if probability_down >= threshold:

        if alignment > 0:
            return "WAIT"

        return "SELL"

    return "WAIT"


# ============================================================
# TRADE PLAN
# ============================================================

def _create_trade_plan(
    price: float,
    signal: str,
    stop_loss_pct: float,
    target_pct: float,
) -> dict[str, Any]:

    price = float(price)

    if signal == "BUY":

        stop_loss = (
            price
            * (1.0 - stop_loss_pct)
        )

        target = (
            price
            * (1.0 + target_pct)
        )

    elif signal == "SELL":

        stop_loss = (
            price
            * (1.0 + stop_loss_pct)
        )

        target = (
            price
            * (1.0 - target_pct)
        )

    else:

        stop_loss = None
        target = None

    return {
        "entry": price,
        "stop_loss": stop_loss,
        "target": target,
        "risk_pct": (
            stop_loss_pct * 100.0
            if signal != "WAIT"
            else 0.0
        ),
        "reward_pct": (
            target_pct * 100.0
            if signal != "WAIT"
            else 0.0
        ),
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
) -> PredictionResult:

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

    horizon = int(horizon)

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
            "Probability threshold must be between "
            "0.50 and 1.00."
        )

    # --------------------------------------------------------
    # COMPLETE FEATURE PIPELINE
    # --------------------------------------------------------

    features = build_features(
        df
    )

    if features.empty:
        raise ValueError(
            "Feature engineering produced no data."
        )

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # --------------------------------------------------------
    # MODEL
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
    # PROBABILITY
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        features
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if (
        probabilities.ndim == 2
        and probabilities.shape[1] >= 2
    ):

        probability_down = float(
            probabilities[-1, 0]
        )

        probability_up = float(
            probabilities[-1, 1]
        )

    elif probabilities.ndim == 1:

        probability_up = float(
            probabilities[-1]
        )

        probability_down = (
            1.0 - probability_up
        )

    else:

        probability_up = 0.5
        probability_down = 0.5

    probability_up = float(
        np.clip(
            probability_up,
            0.0,
            1.0,
        )
    )

    probability_down = float(
        np.clip(
            probability_down,
            0.0,
            1.0,
        )
    )

    # --------------------------------------------------------
    # CONFIDENCE
    #
    # Distance from 50%.
    #
    # Example:
    # 50% probability -> 0% confidence
    # 70% probability -> 40% confidence
    # 90% probability -> 80% confidence
    #
    # This is a presentation metric, not a calibrated
    # probability of being correct.
    # --------------------------------------------------------

    confidence = abs(
        probability_up
        - probability_down
    )

    # --------------------------------------------------------
    # REGIME
    # --------------------------------------------------------

    try:

        regime = market_regime(
            features
        )

    except Exception:

        regime = "UNKNOWN"

    # --------------------------------------------------------
    # MULTI-TIMEFRAME
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
    # SIGNAL
    # --------------------------------------------------------

    signal = _generate_signal(
        probability_up=probability_up,
        alignment=alignment,
        threshold=probability_threshold,
    )

    # --------------------------------------------------------
    # CURRENT PRICE
    # --------------------------------------------------------

    current_price = _latest_close(
        features
    )

    # --------------------------------------------------------
    # TRADE PLAN
    # --------------------------------------------------------

    trade_plan = _create_trade_plan(
        price=current_price,
        signal=signal,
        stop_loss_pct=stop_loss_pct,
        target_pct=target_pct,
    )

    # --------------------------------------------------------
    # INDICATORS
    # --------------------------------------------------------

    rsi = _latest_value(
        features,
        "rsi14",
        _latest_value(
            features,
            "rsi",
            0.0,
        ),
    )

    macd = _latest_value(
        features,
        "macd",
        0.0,
    )

    atr_pct = _latest_value(
        features,
        "atr_pct",
        0.0,
    )

    relative_volume = _latest_value(
        features,
        "relative_volume",
        _latest_value(
            features,
            "volume_ratio",
            0.0,
        ),
    )

    adx = _latest_value(
        features,
        "adx",
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

    indicators = {
        "rsi": rsi,
        "rsi14": rsi,
        "macd": macd,
        "atr_pct": atr_pct,
        "relative_volume": relative_volume,
        "volume_ratio": relative_volume,
        "adx": adx,
        "ema20": ema20,
        "ema50": ema50,
        "ema200": ema200,
    }

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
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
    # TRAINING ACCURACY
    # --------------------------------------------------------

    training_accuracy = getattr(
        model,
        "training_accuracy",
        None,
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result = PredictionResult()

    # Core prediction.
    result.signal = signal

    result.probability_up = (
        probability_up
    )

    result.probability_down = (
        probability_down
    )

    result.prediction_probability = (
        probability_up
    )

    # Compatibility field expected by UI.
    result.confidence = confidence

    result.horizon = horizon

    result.current_price = (
        current_price
    )

    result.price = current_price

    # Regime.
    result.regime = regime

    # Multi-timeframe.
    result.weekly_trend = (
        weekly_trend
    )

    result.monthly_trend = (
        monthly_trend
    )

    result.higher_timeframe_score = (
        higher_timeframe_score
    )

    result.three_timeframe_alignment = (
        alignment
    )

    # Trade plan.
    result.trade_plan = trade_plan

    result.entry = trade_plan[
        "entry"
    ]

    result.stop_loss = trade_plan[
        "stop_loss"
    ]

    result.target = trade_plan[
        "target"
    ]

    # Indicators.
    result.indicators = indicators

    # ML information.
    result.top_features = (
        top_features
    )

    result.feature_importance = (
        importance_df
    )

    result.model_training_accuracy = (
        training_accuracy
    )

    result.training_accuracy = (
        training_accuracy
    )

    result.n_rows = len(
        features
    )

    result.n_features = len(
        model.feature_names_
    )

    result.features = features

    result.model = model

    return result


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

predict_stock = analyze_stock


__all__ = [
    "PredictionResult",
    "analyze_stock",
    "predict_stock",
]
