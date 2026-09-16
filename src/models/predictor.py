"""
Swing Stock Predictor

Complete evidence pipeline:

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
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.features.engine import build_features
from src.features.regime import market_regime
from src.models.classifier import SwingClassifier
from src.data.sentiment import get_stock_sentiment
from src.data.market import market_summary
from src.data.sector import get_sector, sector_summary


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


def _latest_close(
    df: pd.DataFrame,
) -> float:

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
        "trend_score": float(
            trend_score
        ),
        "momentum_score": float(
            momentum_score
        ),
        "volume_score": float(
            volume_score
        ),
        "adx": float(adx),
    }


# =====================================================================
# MARKET CONTEXT
# =====================================================================


def _get_market_context() -> Dict[str, Any]:
    """
    Obtain broad-market context.

    The existing market.py module is used as the single source
    for market-level calculations.
    """

    try:

        summary = market_summary()

        if isinstance(
            summary,
            dict,
        ):
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
    """
    Obtain sector classification and sector context.
    """

    try:

        sector = get_sector(
            ticker
        )

    except Exception:

        sector = "UNKNOWN"

    try:

        summary = sector_summary(
            sector
        )

        if not isinstance(
            summary,
            dict,
        ):
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
    """
    Estimate stock momentum relative to the broad market.

    When a market return feature is unavailable, this falls back
    to the stock's own multi-period return.

    This is contextual evidence, not a standalone trading signal.
    """

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

    relative = (
        stock_return
        - market_return
    )

    if relative > 0.03:
        classification = "OUTPERFORMING"

    elif relative < -0.03:
        classification = "UNDERPERFORMING"

    else:
        classification = "IN LINE"

    return {
        "stock_return": float(
            stock_return
        ),
        "market_return": float(
            market_return
        ),
        "relative_return": float(
            relative
        ),
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
    """
    Apply a small contextual adjustment.

    The ML ensemble remains the primary quantitative component.

    Context influence is capped so that market/sector data cannot
    arbitrarily override the ensemble.
    """

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
        reasons.append(
            "BROAD MARKET BULLISH"
        )

    elif "BEAR" in market_trend:

        adjustment -= 0.025
        reasons.append(
            "BROAD MARKET BEARISH"
        )

    elif market_score > 0.50:

        adjustment += 0.015
        reasons.append(
            "POSITIVE MARKET SCORE"
        )

    elif market_score < -0.50:

        adjustment -= 0.015
        reasons.append(
            "NEGATIVE MARKET SCORE"
        )

    # ---------------------------------------------------------------
    # SECTOR
    # ---------------------------------------------------------------

    sector_summary_data = (
        sector_context.get(
            "summary",
            {},
        )
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
        0.10
        * article_confidence
    )

    sentiment_probability = (
        0.50
        + 0.50 * s
    )

    adjusted_probability = (
        (1.0 - sentiment_weight)
        * p
        + sentiment_weight
        * sentiment_probability
    )

    adjustment = (
        adjusted_probability
        - p
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
        "adjustment": float(
            adjustment
        ),
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

        if (
            p > 0.85
            or p < 0.15
        ):
            return signal

        return "WAIT"

    if agreement < 0.80:

        if (
            signal == "BUY"
            and p < 0.65
        ):
            return "WAIT"

        if (
            signal == "SELL"
            and p > 0.35
        ):
            return "WAIT"

    return signal


# =====================================================================
# 90% CONFIDENCE GATE
# =====================================================================


def _extract_validation_reliability(model) -> Dict[str, Any]:
    """Summarize historical validation behavior around the 90% confidence zone."""
    if model is None:
        return {
            "available": False,
            "sample_count": 0,
            "high_confidence_samples": 0,
            "high_confidence_accuracy": None,
            "ece": None,
        }

    probability_names = [
        "validation_probabilities_",
        "validation_probability_",
        "validation_probs_",
    ]
    actual_names = [
        "validation_actuals_",
        "validation_actual_",
        "validation_y_",
        "validation_targets_",
    ]

    probabilities = None
    actuals = None

    for name in probability_names:
        value = getattr(model, name, None)
        if value is not None:
            probabilities = value
            break

    for name in actual_names:
        value = getattr(model, name, None)
        if value is not None:
            actuals = value
            break

    if probabilities is None or actuals is None:
        return {
            "available": False,
            "sample_count": 0,
            "high_confidence_samples": 0,
            "high_confidence_accuracy": None,
            "ece": None,
        }

    try:
        probs = np.asarray(probabilities, dtype=float)
        y = np.asarray(actuals, dtype=float).reshape(-1)

        if probs.ndim == 2 and probs.shape[1] >= 2:
            p_up = probs[:, 1]
            p_down = probs[:, 0]
        elif probs.ndim == 1:
            p_up = probs
            p_down = 1.0 - probs
        else:
            raise ValueError("Unsupported validation probability shape")

        n = min(len(p_up), len(p_down), len(y))
        if n <= 0:
            raise ValueError("No validation observations")

        frame = pd.DataFrame({
            "p_up": p_up[:n],
            "p_down": p_down[:n],
            "actual": y[:n],
        })
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        frame = frame[frame["actual"].isin([0.0, 1.0])].copy()

        if frame.empty:
            raise ValueError("No valid validation observations")

        frame["confidence"] = frame[["p_up", "p_down"]].max(axis=1)
        frame["predicted"] = (frame["p_up"] >= 0.50).astype(int)
        frame["correct"] = (frame["predicted"] == frame["actual"]).astype(float)

        high = frame[frame["confidence"] >= 0.90]
        high_accuracy = (
            float(high["correct"].mean())
            if not high.empty
            else None
        )

        # Confidence calibration error across fixed confidence bands.
        bins = [0.50, 0.60, 0.70, 0.80, 0.90, 1.000001]
        ece = 0.0
        total = len(frame)
        for left, right in zip(bins[:-1], bins[1:]):
            bucket = frame[
                (frame["confidence"] >= left)
                & (frame["confidence"] < right)
            ]
            if bucket.empty:
                continue
            ece += (
                len(bucket) / total
            ) * abs(
                float(bucket["confidence"].mean())
                - float(bucket["correct"].mean())
            )

        return {
            "available": True,
            "sample_count": int(len(frame)),
            "high_confidence_samples": int(len(high)),
            "high_confidence_accuracy": high_accuracy,
            "ece": float(ece),
        }

    except Exception:
        return {
            "available": False,
            "sample_count": 0,
            "high_confidence_samples": 0,
            "high_confidence_accuracy": None,
            "ece": None,
        }


def _build_confidence_gate(
    *,
    signal: str,
    probability_up: float,
    agreement: float,
    weekly_trend: str,
    monthly_trend: str,
    three_timeframe_alignment: str,
    market_context: Dict[str, Any],
    sector_context: Dict[str, Any],
    relative_strength: Dict[str, Any],
    sentiment_score: float,
    validation_reliability: Dict[str, Any],
) -> Dict[str, Any]:
    """Determine whether the current setup qualifies as a 90% evidence-grade setup."""
    p_up = float(np.clip(probability_up, 0.0, 1.0))
    p_down = 1.0 - p_up
    strongest = max(p_up, p_down)
    direction = "BUY" if p_up >= p_down else "SELL"

    checks = []

    checks.append({
        "name": "Model probability",
        "passed": strongest >= 0.90,
        "detail": f"{strongest:.1%} >= 90%",
    })

    checks.append({
        "name": "Ensemble agreement",
        "passed": agreement >= 0.80,
        "detail": f"{agreement:.1%} >= 80%",
    })

    def direction_matches(text: str, bullish: bool) -> bool:
        value = str(text).upper()
        if bullish:
            return "BULL" in value or "POSITIVE" in value or "ALIGNED" in value
        return "BEAR" in value or "NEGATIVE" in value or "ALIGNED" in value

    bullish = direction == "BUY"

    checks.append({
        "name": "Weekly trend",
        "passed": direction_matches(weekly_trend, bullish),
        "detail": str(weekly_trend),
    })
    checks.append({
        "name": "Monthly trend",
        "passed": direction_matches(monthly_trend, bullish),
        "detail": str(monthly_trend),
    })

    alignment = str(three_timeframe_alignment).upper()
    alignment_pass = (
        (bullish and "BULL" in alignment)
        or ((not bullish) and "BEAR" in alignment)
    )
    checks.append({
        "name": "Timeframe alignment",
        "passed": alignment_pass,
        "detail": alignment,
    })

    market_trend = str(market_context.get("trend", "UNKNOWN")).upper()
    market_pass = (
        (bullish and "BEAR" not in market_trend)
        or ((not bullish) and "BULL" not in market_trend)
    )
    checks.append({
        "name": "Market context",
        "passed": market_pass,
        "detail": market_trend,
    })

    sector_summary_data = sector_context.get("summary", {})
    sector_score = _safe_float(
        sector_summary_data.get("score", 0.0)
        if isinstance(sector_summary_data, dict)
        else 0.0,
        0.0,
    )
    sector_pass = sector_score >= 0.0 if bullish else sector_score <= 0.0
    checks.append({
        "name": "Sector context",
        "passed": sector_pass,
        "detail": f"score {sector_score:+.2f}",
    })

    relative_class = str(relative_strength.get("classification", "IN LINE")).upper()
    relative_pass = (
        (bullish and relative_class != "UNDERPERFORMING")
        or ((not bullish) and relative_class != "OUTPERFORMING")
    )
    checks.append({
        "name": "Relative strength",
        "passed": relative_pass,
        "detail": relative_class,
    })

    sentiment_pass = (
        sentiment_score >= -0.20 if bullish
        else sentiment_score <= 0.20
    )
    checks.append({
        "name": "News sentiment",
        "passed": sentiment_pass,
        "detail": f"score {sentiment_score:+.2f}",
    })

    reliability_available = bool(validation_reliability.get("available"))
    high_samples = int(validation_reliability.get("high_confidence_samples", 0))
    high_accuracy = validation_reliability.get("high_confidence_accuracy")

    # A historical 90% bucket is evidence only when enough observations exist.
    # With fewer than 10 observations we do not claim that the 90% zone is validated.
    reliability_pass = (
        reliability_available
        and high_samples >= 10
        and high_accuracy is not None
        and float(high_accuracy) >= 0.90
    )
    checks.append({
        "name": "Historical 90% reliability",
        "passed": reliability_pass,
        "detail": (
            f"{high_accuracy:.1%} over {high_samples} samples"
            if high_accuracy is not None
            else "insufficient validation evidence"
        ),
    })

    passed = sum(bool(item["passed"]) for item in checks)
    total = len(checks)

    return {
        "eligible": bool(signal in {"BUY", "SELL"} and all(item["passed"] for item in checks)),
        "direction": direction,
        "strongest_probability": float(strongest),
        "checks": checks,
        "passed_checks": int(passed),
        "total_checks": int(total),
        "validation_reliability": validation_reliability,
        "label": "HIGH-CONFIDENCE 90% SETUP" if signal in {"BUY", "SELL"} and all(item["passed"] for item in checks) else "STANDARD SETUP",
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
            "risk_percent": (
                stop_loss_pct * 100
            ),
            "reward_percent": (
                target_pct * 100
            ),
            "risk_reward": (
                target_pct / stop_loss_pct
                if stop_loss_pct > 0
                else 0.0
            ),
        }

    entry = current_price

    if signal == "BUY":

        stop_loss = (
            entry
            * (1.0 - stop_loss_pct)
        )

        target = (
            entry
            * (1.0 + target_pct)
        )

    elif signal == "SELL":

        stop_loss = (
            entry
            * (1.0 + stop_loss_pct)
        )

        target = (
            entry
            * (1.0 - target_pct)
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



# ================================================================
# ADAPTIVE CONFIDENCE & EVIDENCE ENGINE
# ================================================================

def _bounded(value, low=0.0, high=1.0, default=0.5):
    """Return a finite value bounded to [low, high]."""
    try:
        value = float(value)
        if not np.isfinite(value):
            return default
        return float(np.clip(value, low, high))
    except Exception:
        return default


def _trend_polarity(value):
    """Map common trend labels to bullish/neutral/bearish polarity."""
    text = str(value).upper()

    if "STRONG BULL" in text or "BULLISH" in text or text == "BULL":
        return 1.0
    if "STRONG BEAR" in text or "BEARISH" in text or text == "BEAR":
        return -1.0
    if "SIDEWAYS" in text or "NEUTRAL" in text:
        return 0.0

    return 0.0


def _context_score(context):
    if not isinstance(context, dict):
        return 0.0

    for key in ("score", "strength", "trend_score"):
        if key in context:
            try:
                value = float(context[key])
                if np.isfinite(value):
                    return float(np.clip(value, -1.0, 1.0))
            except Exception:
                pass

    for key in ("trend", "summary", "regime"):
        if key in context:
            return _trend_polarity(context[key])

    return 0.0


def _extract_reliability_summary(model):
    """Read calibration evidence without inventing missing validation data."""
    summary = {
        "available": False,
        "validation_samples": 0,
        "high_confidence_samples": 0,
        "high_confidence_accuracy": np.nan,
        "ece": np.nan,
    }

    try:
        probabilities = np.asarray(
            getattr(model, "validation_probabilities", []),
            dtype=float,
        ).reshape(-1)

        actual = np.asarray(
            getattr(model, "validation_actuals", []),
            dtype=float,
        ).reshape(-1)

        if len(probabilities) == 0 or len(actual) != len(probabilities):
            return summary

        probabilities = np.clip(probabilities, 0.0, 1.0)
        actual = np.asarray(actual, dtype=int)

        confidence = np.maximum(probabilities, 1.0 - probabilities)
        predicted = (probabilities >= 0.5).astype(int)

        valid = np.isfinite(confidence) & np.isfinite(actual)
        confidence = confidence[valid]
        predicted = predicted[valid]
        actual = actual[valid]

        if len(actual) == 0:
            return summary

        summary["available"] = True
        summary["validation_samples"] = int(len(actual))

        high = confidence >= 0.90
        summary["high_confidence_samples"] = int(high.sum())

        if high.any():
            summary["high_confidence_accuracy"] = float(
                np.mean(predicted[high] == actual[high])
            )

        # Expected Calibration Error using 10 confidence bins.
        edges = np.linspace(0.50, 1.00, 11)
        ece = 0.0

        for left, right in zip(edges[:-1], edges[1:]):
            if right >= 1.0:
                mask = (confidence >= left) & (confidence <= right)
            else:
                mask = (confidence >= left) & (confidence < right)

            if not mask.any():
                continue

            bucket_conf = float(np.mean(confidence[mask]))
            bucket_acc = float(
                np.mean(predicted[mask] == actual[mask])
            )
            ece += (
                float(mask.sum()) / len(actual)
            ) * abs(bucket_acc - bucket_conf)

        summary["ece"] = float(ece)

    except Exception:
        pass

    return summary


def _build_adaptive_confidence(
    probability_up,
    agreement,
    weekly_trend,
    monthly_trend,
    three_timeframe_alignment,
    market_context,
    sector_context,
    relative_strength,
    sentiment_score,
    reliability,
):
    """
    Convert multiple independent evidence sources into an evidence score.

    This does NOT force probability to 90%. It produces a separate
    confidence/evidence assessment and identifies contradictions.
    """
    p_up = _bounded(probability_up)
    agreement = _bounded(agreement)

    direction = 1.0 if p_up >= 0.5 else -1.0

    trend_values = [
        _trend_polarity(weekly_trend),
        _trend_polarity(monthly_trend),
        _trend_polarity(three_timeframe_alignment),
    ]

    market = _context_score(market_context)
    sector = _context_score(sector_context)

    relative = _context_score(relative_strength)

    sentiment = float(
        np.clip(
            _safe_float(sentiment_score, 0.0),
            -1.0,
            1.0,
        )
    )

    # Evidence direction relative to the model's prediction.
    aligned_trends = np.mean(
        [x * direction for x in trend_values]
    )

    context_alignment = np.mean(
        [
            market * direction,
            sector * direction,
            relative * direction,
            sentiment * direction,
        ]
    )

    probability_strength = abs(p_up - 0.5) * 2.0

    # Independent evidence score. Model probability is important,
    # but agreement and external/contextual confirmation matter too.
    evidence_score = (
        0.35 * probability_strength
        + 0.25 * agreement
        + 0.20 * max(0.0, aligned_trends)
        + 0.20 * max(0.0, context_alignment)
    )

    evidence_score = float(np.clip(evidence_score, 0.0, 1.0))

    contradictions = []

    if agreement < 0.60:
        contradictions.append("models disagree")

    if aligned_trends < -0.20:
        contradictions.append("higher timeframes disagree")

    if context_alignment < -0.20:
        contradictions.append("market/sector/news context conflicts")

    reliability_ready = bool(
        reliability.get("available")
        and reliability.get("high_confidence_samples", 0) >= 10
        and np.isfinite(
            reliability.get("high_confidence_accuracy", np.nan)
        )
    )

    high_confidence_supported = False

    if reliability_ready:
        observed_high_accuracy = float(
            reliability["high_confidence_accuracy"]
        )
        high_confidence_supported = bool(
            p_up >= 0.90
            and observed_high_accuracy >= 0.80
            and agreement >= 0.80
            and evidence_score >= 0.80
        )
    else:
        # No historical support means the engine must not claim
        # that a >=90% confidence level is validated.
        high_confidence_supported = False

    if p_up >= 0.90 and not high_confidence_supported:
        contradictions.append(
            "90% confidence is not historically supported by the "
            "available validation evidence"
        )

    if contradictions:
        evidence_status = "CONFLICTED"
    elif high_confidence_supported:
        evidence_status = "HIGH-CONFIDENCE SUPPORTED"
    elif evidence_score >= 0.75:
        evidence_status = "STRONG EVIDENCE"
    elif evidence_score >= 0.55:
        evidence_status = "MODERATE EVIDENCE"
    else:
        evidence_status = "LIMITED EVIDENCE"

    # A conservative confidence estimate: never manufacture a value
    # above the underlying model probability.
    evidence_confidence = float(
        min(
            p_up,
            0.50 + 0.50 * evidence_score,
        )
    )

    return {
        "evidence_score": evidence_score,
        "evidence_confidence": evidence_confidence,
        "evidence_status": evidence_status,
        "contradictions": contradictions,
        "direction": "UP" if direction > 0 else "DOWN",
        "high_confidence_supported": high_confidence_supported,
        "historical_reliability_available": reliability_ready,
        "high_confidence_validation_samples": int(
            reliability.get("high_confidence_samples", 0)
        ),
        "high_confidence_validation_accuracy": (
            float(reliability["high_confidence_accuracy"])
            if np.isfinite(
                reliability.get("high_confidence_accuracy", np.nan)
            )
            else np.nan
        ),
        "ece": (
            float(reliability["ece"])
            if np.isfinite(reliability.get("ece", np.nan))
            else np.nan
        ),
    }



def analyze_stock(
    df: pd.DataFrame,
    horizon: int = 5,
    probability_threshold: float = 0.60,
    stop_loss_pct: float = 0.03,
    target_pct: float = 0.06,
    ticker: str = "",
) -> PredictionResult:

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Input DataFrame is empty."
        )

    horizon = int(
        horizon
    )

    if horizon < 1:
        raise ValueError(
            "Prediction horizon must be >= 1."
        )

    # ================================================================
    # FEATURE ENGINEERING
    # ================================================================

    features = build_features(
        df
    )

    if (
        features is None
        or features.empty
    ):
        raise ValueError(
            "Feature engineering produced no data."
        )

    # ================================================================
    # ML ENSEMBLE
    # ================================================================

    model = SwingClassifier(
        horizon=horizon,
        probability_threshold=(
            probability_threshold
        ),
    )

    model.fit(
        features
    )

    probabilities = (
        model.predict_proba(
            features
        )
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
        probability_up
        + probability_down
    )

    if total_probability > 0:

        probability_up /= (
            total_probability
        )

        probability_down /= (
            total_probability
        )

    # ================================================================
    # MODEL AGREEMENT
    # ================================================================

    try:

        model_agreement = (
            model.model_agreement(
                features
            )
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

    # ================================================================
    # MARKET CONTEXT
    # ================================================================

    market_context = (
        _get_market_context()
    )

    # ================================================================
    # SECTOR CONTEXT
    # ================================================================

    sector_context = (
        _get_sector_context(
            ticker
        )
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

    context_fusion = (
        _context_adjustment(
            probability_up=probability_up,
            market_context=market_context,
            sector_context=sector_context,
            relative_strength=relative_strength,
        )
    )

    context_probability = (
        context_fusion[
            "adjusted_probability"
        ]
    )

    # ================================================================
    # NEWS SENTIMENT
    # ================================================================

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

        try:

            sentiment_data = (
                get_stock_sentiment(
                    ticker=ticker,
                    max_items=20,
                )
            )

        except Exception:
            pass

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

    sentiment_fusion = (
        _sentiment_adjustment(
            probability_up=(
                context_probability
            ),
            sentiment_score=(
                sentiment_score
            ),
            sentiment_articles=(
                sentiment_articles
            ),
        )
    )

    final_probability = (
        sentiment_fusion[
            "adjusted_probability"
        ]
    )

    # ================================================================
    # SIGNAL
    # ================================================================

    base_signal = _generate_signal(
        final_probability,
        probability_threshold,
    )

    final_signal = (
        _apply_agreement_filter(
            signal=base_signal,
            probability_up=(
                final_probability
            ),
            agreement=_safe_float(
                model_agreement.get(
                    "agreement",
                    0.0,
                ),
                0.0,
            ),
        )
    )

    # ================================================================
    # CONFIDENCE
    # ================================================================

    confidence = abs(
        final_probability
        - (1.0 - final_probability)
    )

    uncertainty = (
        1.0 - confidence
    )

    # ================================================================
    # MARKET REGIME
    # ================================================================

    try:

        regime = market_regime(
            features
        )

    except Exception:

        regime = "UNKNOWN"

    if isinstance(
        regime,
        pd.Series,
    ):

        regime = (
            str(regime.iloc[-1])
            if len(regime)
            else "UNKNOWN"
        )

    if isinstance(
        regime,
        pd.DataFrame,
    ):

        if not regime.empty:

            regime = str(
                regime.iloc[-1, 0]
            )

        else:

            regime = "UNKNOWN"

    regime = str(
        regime
    )

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

    higher_timeframe_score = (
        _latest_value(
            features,
            "higher_timeframe_score",
            0.0,
        )
    )

    three_timeframe_alignment = (
        _latest_text(
            features,
            "three_timeframe_alignment",
            "UNKNOWN",
        )
    )

    # ================================================================
    # STRUCTURE
    # ================================================================

    structure = (
        _calculate_market_structure(
            features
        )
    )

    # ================================================================
    # 90% CONFIDENCE EVIDENCE GATE
    # ================================================================

    validation_reliability = _extract_validation_reliability(model)

    confidence_gate = _build_confidence_gate(
        signal=final_signal,
        probability_up=final_probability,
        agreement=_safe_float(
            model_agreement.get("agreement", 0.0),
            0.0,
        ),
        weekly_trend=weekly_trend,
        monthly_trend=monthly_trend,
        three_timeframe_alignment=three_timeframe_alignment,
        market_context=market_context,
        sector_context=sector_context,
        relative_strength=relative_strength,
        sentiment_score=sentiment_score,
        validation_reliability=validation_reliability,
    )

    # ================================================================
    # ADAPTIVE CONFIDENCE & EVIDENCE ENGINE
    # ================================================================

    adaptive_reliability = _extract_reliability_summary(model)

    adaptive_confidence = _build_adaptive_confidence(
        probability_up=final_probability,
        agreement=_safe_float(
            model_agreement.get("agreement", 0.0),
            0.0,
        ),
        weekly_trend=weekly_trend,
        monthly_trend=monthly_trend,
        three_timeframe_alignment=three_timeframe_alignment,
        market_context=market_context,
        sector_context=sector_context,
        relative_strength=relative_strength,
        sentiment_score=sentiment_score,
        reliability=adaptive_reliability,
    )

    # The adaptive engine is deliberately a confidence/evidence layer.
    # It does not rewrite the underlying calibrated ML probability.

    # ================================================================
    # PRICE / TRADE PLAN
    # ================================================================

    current_price = _latest_close(
        features
    )

    trade_plan = (
        _build_trade_plan(
            current_price=current_price,
            signal=final_signal,
            stop_loss_pct=stop_loss_pct,
            target_pct=target_pct,
        )
    )

    # ================================================================
    # FEATURE IMPORTANCE
    # ================================================================

    try:

        importance_df = (
            model.feature_importance()
        )

        top_features = (
            importance_df
            .head(10)
            .to_dict(
                "records"
            )
        )

        feature_importance = (
            importance_df
            .to_dict(
                "records"
            )
        )

    except Exception:

        top_features = []
        feature_importance = []

    # ================================================================
    # MODEL SUMMARY
    # ================================================================

    model_summary = model.summary()

    try:
        model_summary["adaptive_evidence_score"] = float(
            adaptive_confidence.get("evidence_score", 0.0)
        )
        model_summary["adaptive_evidence_status"] = str(
            adaptive_confidence.get(
                "evidence_status",
                "LIMITED EVIDENCE",
            )
        )
        model_summary["high_confidence_supported"] = bool(
            adaptive_confidence.get(
                "high_confidence_supported",
                False,
            )
        )
    except Exception:
        pass


    # ================================================================
    # NEWS
    # ================================================================

    news_df = sentiment_data.get(
        "news",
        pd.DataFrame(),
    )

    # ================================================================
    # RESULT
    # ================================================================

    return PredictionResult(
        {
            "signal": final_signal,

            "base_signal": base_signal,

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

            "prediction_probability_oos": float(
                probability_up
            ),

            "confidence": float(
                confidence
            ),

            "uncertainty": float(
                uncertainty
            ),

            # --------------------------------------------------------
            # ADAPTIVE CONFIDENCE / EVIDENCE
            # --------------------------------------------------------

            "adaptive_confidence": adaptive_confidence,

            "evidence_score": float(
                adaptive_confidence.get(
                    "evidence_score",
                    0.0,
                )
            ),

            "evidence_confidence": float(
                adaptive_confidence.get(
                    "evidence_confidence",
                    0.0,
                )
            ),

            "evidence_status": str(
                adaptive_confidence.get(
                    "evidence_status",
                    "LIMITED EVIDENCE",
                )
            ),

            "evidence_contradictions": list(
                adaptive_confidence.get(
                    "contradictions",
                    [],
                )
            ),

            "high_confidence_supported": bool(
                adaptive_confidence.get(
                    "high_confidence_supported",
                    False,
                )
            ),

            "historical_reliability_available": bool(
                adaptive_confidence.get(
                    "historical_reliability_available",
                    False,
                )
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

            # --------------------------------------------------------
            # CONFIDENCE GATE
            # --------------------------------------------------------

            "confidence_gate": confidence_gate,

            "validation_reliability": validation_reliability,

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


# =====================================================================
# BACKWARD COMPATIBILITY
# =====================================================================


predict_stock = analyze_stock


__all__ = [
    "PredictionResult",
    "analyze_stock",
    "predict_stock",
]
