"""
Swing Stock Predictor
=====================

Optimized evidence pipeline:

OHLCV
  -> Technical / Price Action / Volume
  -> ML Ensemble
  -> Market Regime
  -> Multi-Timeframe
  -> News Sentiment
  -> Market Context
  -> Sector Context
  -> Relative Strength
  -> Evidence Fusion
  -> BUY / SELL / WAIT
  -> Risk Plan

Performance principles:
- Predict only the latest row for live analysis.
- Avoid repeated full-history model predictions.
- Cache expensive Streamlit calculations.
- Preserve backward compatibility with the existing UI.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.features.engine import build_features
from src.models.classifier import SwingClassifier
from src.data.sentiment import get_stock_sentiment
from src.data.market import market_summary
from src.data.sector import get_sector, sector_summary


# ---------------------------------------------------------------------
# OPTIONAL STREAMLIT CACHE
# ---------------------------------------------------------------------

try:
    import streamlit as st

    _STREAMLIT_AVAILABLE = True
except Exception:
    st = None
    _STREAMLIT_AVAILABLE = False


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

    return _safe_float(
        df[column].iloc[-1],
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


def _latest_close(df: pd.DataFrame) -> float:

    if "close" not in df.columns:
        raise ValueError(
            "Input data must contain a 'close' column."
        )

    values = pd.to_numeric(
        df["close"],
        errors="coerce",
    ).dropna()

    if values.empty:
        raise ValueError(
            "No valid closing-price data available."
        )

    return float(values.iloc[-1])


# =====================================================================
# CACHE HELPERS
# =====================================================================

# These caches are intentionally short-lived because market/news data
# changes during the trading day.


def _cached_market_summary():
    """
    Cache broad-market context for a short period.

    This avoids repeated Yahoo Finance requests when Streamlit reruns
    because of UI interaction.
    """

    try:
        return market_summary()
    except Exception:
        return {
            "trend": "UNKNOWN",
            "momentum": "UNKNOWN",
            "volatility": "UNKNOWN",
            "strength": "UNKNOWN",
            "volume": "UNKNOWN",
            "score": 0.0,
        }


def _cached_sentiment(
    ticker: str,
    max_items: int = 20,
):
    """
    Cached news sentiment.

    News is intentionally cached only briefly so the application
    does not repeatedly request the same news during Streamlit reruns.
    """

    try:
        return get_stock_sentiment(
            ticker=ticker,
            max_items=max_items,
        )
    except Exception:
        return {
            "ticker": ticker,
            "news": pd.DataFrame(),
            "summary": {
                "score": 0.0,
                "label": "NO DATA",
                "articles": 0,
                "positive": 0,
                "neutral": 0,
                "negative": 0,
            },
            "recent_score": 0.0,
        }


# Apply Streamlit caching only when Streamlit is actually available.
if _STREAMLIT_AVAILABLE:

    try:

        _cached_market_summary = st.cache_data(
            ttl=300,
            show_spinner=False,
        )(_cached_market_summary)

        _cached_sentiment = st.cache_data(
            ttl=300,
            show_spinner=False,
        )(_cached_sentiment)

    except Exception:
        pass


# =====================================================================
# FEATURE CACHE
# =====================================================================


def _build_features_cached(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature-engineering cache.

    Streamlit reruns frequently. Rebuilding hundreds of technical,
    price-action, volume and MTF features unnecessarily can add
    noticeable latency.
    """

    return build_features(df)


if _STREAMLIT_AVAILABLE:

    try:

        _build_features_cached = st.cache_data(
            ttl=1800,
            show_spinner=False,
        )(_build_features_cached)

    except Exception:
        pass


# =====================================================================
# MARKET STRUCTURE
# =====================================================================


def _calculate_market_structure(
    features: pd.DataFrame,
) -> Dict[str, Any]:

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

    # ---------------------------------------------------------------
    # TREND
    # ---------------------------------------------------------------

    trend_score = 0.0

    if close > 0 and ema20 > 0:
        trend_score += (
            1.0
            if close > ema20
            else -1.0
        )

    if ema20 > 0 and ema50 > 0:
        trend_score += (
            1.0
            if ema20 > ema50
            else -1.0
        )

    if ema50 > 0 and ema200 > 0:
        trend_score += (
            1.0
            if ema50 > ema200
            else -1.0
        )

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
# MARKET CONTEXT
# =====================================================================


def _get_market_context() -> Dict[str, Any]:

    try:

        summary = _cached_market_summary()

        if isinstance(summary, dict):
            return summary

    except Exception:
        pass

    return {
        "trend": "UNKNOWN",
        "momentum": "UNKNOWN",
        "volatility": "UNKNOWN",
        "strength": "UNKNOWN",
        "volume": "UNKNOWN",
        "score": 0.0,
    }


# =====================================================================
# SECTOR CONTEXT
# =====================================================================


def _get_sector_context(
    ticker: str,
) -> Dict[str, Any]:

    try:
        sector = get_sector(ticker)

    except Exception:
        sector = "UNKNOWN"

    try:

        summary = sector_summary(sector)

        if not isinstance(summary, dict):
            summary = {}

    except Exception:
        summary = {}

    return {
        "sector": sector,
        "summary": summary,
    }


# =====================================================================
# RELATIVE STRENGTH
# =====================================================================


def _calculate_relative_strength(
    stock_features: pd.DataFrame,
    market_context: Dict[str, Any],
) -> Dict[str, Any]:

    stock_return = 0.0

    for column in [
        "roc20",
        "return_20",
        "returns_20",
        "ret_20",
    ]:

        if column in stock_features.columns:

            value = _latest_value(
                stock_features,
                column,
                np.nan,
            )

            if np.isfinite(value):
                stock_return = value
                break

    market_return = 0.0

    for key in [
        "return_20",
        "market_return_20",
        "roc20",
        "return",
        "performance",
    ]:

        if key in market_context:

            value = _safe_float(
                market_context.get(key),
                np.nan,
            )

            if np.isfinite(value):
                market_return = value
                break

    relative = stock_return - market_return

    if relative > 0.03:
        classification = "OUTPERFORMING"

    elif relative < -0.03:
        classification = "UNDERPERFORMING"

    else:
        classification = "IN LINE"

    return {
        "stock_return": float(stock_return),
        "market_return": float(market_return),
        "relative_return": float(relative),
        "classification": classification,
    }


# =====================================================================
# CONTEXT ADJUSTMENT
# =====================================================================


def _context_adjustment(
    probability_up: float,
    market_context: Dict[str, Any],
    sector_context: Dict[str, Any],
    relative_strength: Dict[str, Any],
) -> Dict[str, Any]:

    p = float(
        np.clip(
            probability_up,
            0.0,
            1.0,
        )
    )

    adjustment = 0.0
    reasons = []

    # ---------------------------------------------------------------
    # MARKET
    # ---------------------------------------------------------------

    market_trend = str(
        market_context.get(
            "trend",
            "UNKNOWN",
        )
    ).upper()

    market_score = _safe_float(
        market_context.get(
            "score",
            0.0,
        ),
        0.0,
    )

    if "BULL" in market_trend:

        adjustment += 0.025
        reasons.append("BROAD MARKET BULLISH")

    elif "BEAR" in market_trend:

        adjustment -= 0.025
        reasons.append("BROAD MARKET BEARISH")

    elif market_score > 0.50:

        adjustment += 0.015
        reasons.append("POSITIVE MARKET SCORE")

    elif market_score < -0.50:

        adjustment -= 0.015
        reasons.append("NEGATIVE MARKET SCORE")

    # ---------------------------------------------------------------
    # SECTOR
    # ---------------------------------------------------------------

    sector_summary_data = sector_context.get(
        "summary",
        {},
    )

    if isinstance(
        sector_summary_data,
        dict,
    ):

        sector_score = _safe_float(
            sector_summary_data.get(
                "score",
                0.0,
            ),
            0.0,
        )

        if sector_score > 0.50:

            adjustment += 0.025
            reasons.append(
                "SECTOR STRENGTH POSITIVE"
            )

        elif sector_score < -0.50:

            adjustment -= 0.025
            reasons.append(
                "SECTOR STRENGTH NEGATIVE"
            )

    # ---------------------------------------------------------------
    # RELATIVE STRENGTH
    # ---------------------------------------------------------------

    classification = str(
        relative_strength.get(
            "classification",
            "IN LINE",
        )
    ).upper()

    if classification == "OUTPERFORMING":

        adjustment += 0.025
        reasons.append(
            "STOCK OUTPERFORMING MARKET"
        )

    elif classification == "UNDERPERFORMING":

        adjustment -= 0.025
        reasons.append(
            "STOCK UNDERPERFORMING MARKET"
        )

    # ---------------------------------------------------------------
    # CAP
    # ---------------------------------------------------------------

    adjustment = float(
        np.clip(
            adjustment,
            -0.075,
            0.075,
        )
    )

    adjusted_probability = float(
        np.clip(
            p + adjustment,
            0.0,
            1.0,
        )
    )

    return {
        "raw_probability": float(p),
        "adjustment": adjustment,
        "adjusted_probability": adjusted_probability,
        "reasons": reasons,
    }


# =====================================================================
# SENTIMENT FUSION
# =====================================================================


def _sentiment_adjustment(
    probability_up: float,
    sentiment_score: float,
    sentiment_articles: int,
) -> Dict[str, Any]:

    p = float(
        np.clip(
            probability_up,
            0.0,
            1.0,
        )
    )

    s = float(
        np.clip(
            sentiment_score,
            -1.0,
            1.0,
        )
    )

    articles = int(
        max(
            sentiment_articles,
            0,
        )
    )

    if articles == 0:

        return {
            "raw_probability": p,
            "sentiment_score": 0.0,
            "adjusted_probability": p,
            "adjustment": 0.0,
            "sentiment_weight": 0.0,
            "reason": "NO NEWS DATA",
        }

    article_confidence = min(
        1.0,
        articles / 10.0,
    )

    sentiment_weight = (
        0.10 * article_confidence
    )

    sentiment_probability = (
        0.50 + 0.50 * s
    )

    adjusted_probability = (
        (1.0 - sentiment_weight) * p
        + sentiment_weight * sentiment_probability
    )

    adjustment = (
        adjusted_probability - p
    )

    if s >= 0.20:
        reason = "POSITIVE NEWS"

    elif s <= -0.20:
        reason = "NEGATIVE NEWS"

    else:
        reason = "NEUTRAL NEWS"

    return {
        "raw_probability": float(p),
        "sentiment_score": float(s),
        "adjusted_probability": float(
            np.clip(
                adjusted_probability,
                0.0,
                1.0,
            )
        ),
        "adjustment": float(adjustment),
        "sentiment_weight": float(
            sentiment_weight
        ),
        "reason": reason,
    }


# =====================================================================
# SIGNAL
# =====================================================================


def _generate_signal(
    probability_up: float,
    threshold: float,
) -> str:

    p = _safe_float(
        probability_up,
        0.5,
    )

    threshold = min(
        max(
            float(threshold),
            0.50,
        ),
        0.95,
    )

    if p >= threshold:
        return "BUY"

    if p <= 1.0 - threshold:
        return "SELL"

    return "WAIT"


def _apply_agreement_filter(
    signal: str,
    probability_up: float,
    agreement: float,
) -> str:

    agreement = float(
        np.clip(
            agreement,
            0.0,
            1.0,
        )
    )

    p = float(
        np.clip(
            probability_up,
            0.0,
            1.0,
        )
    )

    if agreement < 0.60:

        if p > 0.85 or p < 0.15:
            return signal

        return "WAIT"

    if agreement < 0.80:

        if signal == "BUY" and p < 0.65:
            return "WAIT"

        if signal == "SELL" and p > 0.35:
            return "WAIT"

    return signal


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
# MAIN ANALYZER
# =====================================================================


def analyze_stock(
    df: pd.DataFrame,
    horizon: int = 5,
    probability_threshold: float = 0.60,
    stop_loss_pct: float = 0.03,
    target_pct: float = 0.06,
    ticker: str = "",
) -> PredictionResult:

    total_start = time.perf_counter()

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

    # ================================================================
    # FEATURE ENGINEERING
    # ================================================================

    feature_start = time.perf_counter()

    features = _build_features_cached(df)

    if (
        features is None
        or features.empty
    ):
        raise ValueError(
            "Feature engineering produced no data."
        )

    feature_seconds = (
        time.perf_counter() - feature_start
    )

    # ================================================================
    # ML ENSEMBLE
    # ================================================================

    ml_start = time.perf_counter()

    model = SwingClassifier(
        horizon=horizon,
        probability_threshold=(
            probability_threshold
        ),
    )

    model.fit(features)

    # IMPORTANT PERFORMANCE OPTIMIZATION:
    #
    # The old implementation predicted the complete historical
    # dataframe. The live application only needs the latest row.
    #
    # This changes:
    #
    #     predict(all rows)
    #
    # into:
    #
    #     predict(latest row)
    #
    latest_features = features.tail(1)

    probabilities = model.predict_proba(
        latest_features
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

    total_probability = (
        probability_up + probability_down
    )

    if total_probability > 0:

        probability_up /= total_probability
        probability_down /= total_probability

    ml_seconds = (
        time.perf_counter() - ml_start
    )

    # ================================================================
    # MODEL AGREEMENT
    # ================================================================

    try:

        # Only the latest row is required.
        model_agreement = model.model_agreement(
            latest_features
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
                latest_features
            )
        )

    except Exception:

        component_probabilities = {}

    # ================================================================
    # MARKET CONTEXT
    # ================================================================

    context_start = time.perf_counter()

    market_context = _get_market_context()

    # ================================================================
    # SECTOR CONTEXT
    # ================================================================

    sector_context = _get_sector_context(
        ticker
    )

    # ================================================================
    # RELATIVE STRENGTH
    # ================================================================

    relative_strength = (
        _calculate_relative_strength(
            features,
            market_context,
        )
    )

    # ================================================================
    # MARKET / SECTOR ADJUSTMENT
    # ================================================================

    context_fusion = _context_adjustment(
        probability_up=probability_up,
        market_context=market_context,
        sector_context=sector_context,
        relative_strength=relative_strength,
    )

    context_probability = context_fusion[
        "adjusted_probability"
    ]

    context_seconds = (
        time.perf_counter() - context_start
    )

    # ================================================================
    # NEWS SENTIMENT
    # ================================================================

    sentiment_start = time.perf_counter()

    sentiment_data = {
        "ticker": ticker,
        "news": pd.DataFrame(),
        "summary": {
            "score": 0.0,
            "label": "NO DATA",
            "articles": 0,
            "positive": 0,
            "neutral": 0,
            "negative": 0,
        },
        "recent_score": 0.0,
    }

    if ticker:

        sentiment_data = _cached_sentiment(
            ticker=ticker,
            max_items=20,
        )

    sentiment_summary_data = (
        sentiment_data.get(
            "summary",
            {},
        )
    )

    sentiment_score = _safe_float(
        sentiment_data.get(
            "recent_score",
            sentiment_summary_data.get(
                "score",
                0.0,
            ),
        ),
        0.0,
    )

    sentiment_articles = int(
        sentiment_summary_data.get(
            "articles",
            0,
        )
        or 0
    )

    sentiment_label_value = str(
        sentiment_summary_data.get(
            "label",
            "NO DATA",
        )
    )

    sentiment_fusion = _sentiment_adjustment(
        probability_up=context_probability,
        sentiment_score=sentiment_score,
        sentiment_articles=sentiment_articles,
    )

    final_probability = sentiment_fusion[
        "adjusted_probability"
    ]

    sentiment_seconds = (
        time.perf_counter() - sentiment_start
    )

    # ================================================================
    # SIGNAL
    # ================================================================

    base_signal = _generate_signal(
        final_probability,
        probability_threshold,
    )

    final_signal = _apply_agreement_filter(
        signal=base_signal,
        probability_up=final_probability,
        agreement=_safe_float(
            model_agreement.get(
                "agreement",
                0.0,
            ),
            0.0,
        ),
    )

    # ================================================================
    # CONFIDENCE
    # ================================================================

    confidence = abs(
        final_probability
        - (1.0 - final_probability)
    )

    uncertainty = 1.0 - confidence

    # ================================================================
    # MARKET REGIME
    # ================================================================

    # Avoid calling market_regime(features) again.
    #
    # The feature engine already creates regime information.
    # Reading the latest value is substantially cheaper.

    regime = _latest_text(
        features,
        "regime",
        "UNKNOWN",
    )

    # Some versions may use a different regime column.
    if regime == "UNKNOWN":

        for regime_column in [
            "market_regime",
            "regime_label",
            "market_regime_label",
        ]:

            candidate = _latest_text(
                features,
                regime_column,
                "UNKNOWN",
            )

            if candidate != "UNKNOWN":
                regime = candidate
                break

    # ================================================================
    # MULTI-TIMEFRAME
    # ================================================================

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

    # ================================================================
    # STRUCTURE
    # ================================================================

    structure = _calculate_market_structure(
        features
    )

    # ================================================================
    # PRICE / TRADE PLAN
    # ================================================================

    current_price = _latest_close(
        features
    )

    trade_plan = _build_trade_plan(
        current_price=current_price,
        signal=final_signal,
        stop_loss_pct=stop_loss_pct,
        target_pct=target_pct,
    )

    # ================================================================
    # FEATURE IMPORTANCE
    # ================================================================

    try:

        importance_df = model.feature_importance()

        top_features = (
            importance_df
            .head(10)
            .to_dict("records")
        )

        feature_importance = (
            importance_df
            .to_dict("records")
        )

    except Exception:

        top_features = []
        feature_importance = []

    # ================================================================
    # MODEL SUMMARY
    # ================================================================

    model_summary = model.summary()

    # ================================================================
    # NEWS
    # ================================================================

    news_df = sentiment_data.get(
        "news",
        pd.DataFrame(),
    )

    # ================================================================
    # PERFORMANCE DIAGNOSTICS
    # ================================================================

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    performance = {
        "total_seconds": round(
            total_seconds,
            3,
        ),
        "feature_seconds": round(
            feature_seconds,
            3,
        ),
        "ml_seconds": round(
            ml_seconds,
            3,
        ),
        "context_seconds": round(
            context_seconds,
            3,
        ),
        "sentiment_seconds": round(
            sentiment_seconds,
            3,
        ),
        "cached_features": (
            _STREAMLIT_AVAILABLE
        ),
        "latest_row_prediction": True,
    }

    # ================================================================
    # RESULT
    # ================================================================

    return PredictionResult(
        {

            # --------------------------------------------------------
            # SIGNAL
            # --------------------------------------------------------

            "signal": final_signal,

            "base_signal": base_signal,

            # --------------------------------------------------------
            # PROBABILITIES
            # --------------------------------------------------------

            "probability_up": float(
                probability_up
            ),

            "probability_down": float(
                probability_down
            ),

            "prediction_probability_ml": float(
                probability_up
            ),

            "prediction_probability_context": float(
                context_probability
            ),

            "prediction_probability": float(
                final_probability
            ),

            # Kept for backward compatibility.
            #
            # Note:
            # this remains the model probability and should not be
            # interpreted as a formally walk-forward OOS probability.
            "prediction_probability_oos": float(
                probability_up
            ),

            # --------------------------------------------------------
            # CONFIDENCE
            # --------------------------------------------------------

            "confidence": float(
                confidence
            ),

            "uncertainty": float(
                uncertainty
            ),

            # --------------------------------------------------------
            # BASIC
            # --------------------------------------------------------

            "horizon": int(
                horizon
            ),

            "current_price": float(
                current_price
            ),

            "price": float(
                current_price
            ),

            # --------------------------------------------------------
            # MARKET CONTEXT
            # --------------------------------------------------------

            "market_context": market_context,

            "sector_context": sector_context,

            "relative_strength": (
                relative_strength
            ),

            "context_fusion": context_fusion,

            # --------------------------------------------------------
            # SENTIMENT
            # --------------------------------------------------------

            "sentiment": {

                "score": float(
                    sentiment_score
                ),

                "label": (
                    sentiment_label_value
                ),

                "articles": int(
                    sentiment_articles
                ),

                "positive": int(
                    sentiment_summary_data.get(
                        "positive",
                        0,
                    )
                    or 0
                ),

                "neutral": int(
                    sentiment_summary_data.get(
                        "neutral",
                        0,
                    )
                    or 0
                ),

                "negative": int(
                    sentiment_summary_data.get(
                        "negative",
                        0,
                    )
                    or 0
                ),

                "recent_score": float(
                    sentiment_score
                ),

                "raw_probability": float(
                    sentiment_fusion[
                        "raw_probability"
                    ]
                ),

                "adjusted_probability": float(
                    sentiment_fusion[
                        "adjusted_probability"
                    ]
                ),

                "adjustment": float(
                    sentiment_fusion[
                        "adjustment"
                    ]
                ),

                "weight": float(
                    sentiment_fusion[
                        "sentiment_weight"
                    ]
                ),

                "reason": sentiment_fusion[
                    "reason"
                ],

                "news": news_df,
            },

            # --------------------------------------------------------
            # MODEL
            # --------------------------------------------------------

            "model_agreement": (
                model_agreement
            ),

            "component_probabilities": (
                component_probabilities
            ),

            # --------------------------------------------------------
            # REGIME / MTF
            # --------------------------------------------------------

            "regime": regime,

            "weekly_trend": (
                weekly_trend
            ),

            "monthly_trend": (
                monthly_trend
            ),

            "higher_timeframe_score": float(
                higher_timeframe_score
            ),

            "three_timeframe_alignment": (
                three_timeframe_alignment
            ),

            # --------------------------------------------------------
            # STRUCTURE
            # --------------------------------------------------------

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

            # --------------------------------------------------------
            # TRADE PLAN
            # --------------------------------------------------------

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

            # --------------------------------------------------------
            # INDICATORS
            # --------------------------------------------------------

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

            # --------------------------------------------------------
            # MODEL DIAGNOSTICS
            # --------------------------------------------------------

            "top_features": (
                top_features
            ),

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
                model_summary.get(
                    "validation_accuracy"
                )
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
                model_summary.get(
                    "validation_roc_auc"
                )
            ),

            "validation_brier": (
                model_summary.get(
                    "validation_brier"
                )
            ),

            "model_summary": model_summary,

            # --------------------------------------------------------
            # DATA
            # --------------------------------------------------------

            "n_rows": int(
                len(features)
            ),

            "n_features": int(
                len(model.feature_names_)
            ),

            "features": features,

            "model": model,

            # --------------------------------------------------------
            # PERFORMANCE
            # --------------------------------------------------------

            "performance": performance,
        }
    )


# =====================================================================
# BACKWARD COMPATIBILITY
# =====================================================================


predict_stock = analyze_stock


__all__ = [
    "PredictionResult",
    "analyze_stock",
    "predict_stock",
]
