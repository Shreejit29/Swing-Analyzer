"""
Signal Stability & Decision Engine
----------------------------------

Converts a noisy, frequently changing model probability into a more
stable swing-trading decision.

Design goals:
- Separate ML prediction from the final trading signal.
- Use hysteresis so small probability changes do not flip the signal.
- Require persistence before changing a confirmed signal.
- Combine probability, ensemble agreement, trend/regime, MTF and sentiment.
- Keep the engine deterministic and explainable.
- Do not manufacture confidence or guarantee outcomes.

This module does not train models and does not download market data.
It is intentionally independent so it can be integrated into the predictor
without changing the ensemble itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import math


@dataclass
class SignalState:
    """State of the currently confirmed signal."""

    signal: str = "WAIT"
    pending_signal: str = "WAIT"
    pending_count: int = 0
    observations: int = 0
    last_probability: float = 0.5
    last_score: float = 0.0
    stability: str = "LOW"
    reason: str = "No confirmed signal yet."

    def as_dict(self) -> dict:
        return {
            "signal": self.signal,
            "pending_signal": self.pending_signal,
            "pending_count": int(self.pending_count),
            "observations": int(self.observations),
            "last_probability": float(self.last_probability),
            "last_score": float(self.last_score),
            "stability": self.stability,
            "reason": self.reason,
        }


@dataclass
class SignalDecision:
    """Complete explainable decision returned by the engine."""

    signal: str
    raw_probability: float
    evidence_score: float
    agreement: float
    stability: str
    changed: bool
    confirmed: bool
    pending_signal: str
    pending_count: int
    reason: str
    components: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "signal": self.signal,
            "raw_probability": float(self.raw_probability),
            "evidence_score": float(self.evidence_score),
            "agreement": float(self.agreement),
            "stability": self.stability,
            "changed": bool(self.changed),
            "confirmed": bool(self.confirmed),
            "pending_signal": self.pending_signal,
            "pending_count": int(self.pending_count),
            "reason": self.reason,
            "components": dict(self.components),
        }


class SignalStabilityEngine:
    """
    Stateful decision layer for a swing-trading application.

    Default probability bands intentionally use hysteresis:

        BUY entry:       >= 0.65
        BUY maintained: >= 0.55
        SELL entry:      <= 0.35
        SELL maintained: <= 0.45

    A new direction normally requires two consecutive observations. This
    prevents one transient data refresh from immediately replacing a
    confirmed signal.

    The engine can be reset at a new trading session or when the ticker/
    analysis context changes.
    """

    VALID_SIGNALS = {"BUY", "SELL", "WAIT"}

    def __init__(
        self,
        buy_entry: float = 0.65,
        buy_exit: float = 0.55,
        sell_entry: float = 0.35,
        sell_exit: float = 0.45,
        persistence: int = 2,
        strong_agreement: float = 0.80,
        moderate_agreement: float = 0.60,
    ):
        self.buy_entry = float(buy_entry)
        self.buy_exit = float(buy_exit)
        self.sell_entry = float(sell_entry)
        self.sell_exit = float(sell_exit)
        self.persistence = max(1, int(persistence))
        self.strong_agreement = float(strong_agreement)
        self.moderate_agreement = float(moderate_agreement)

        self._validate_thresholds()
        self.state = SignalState()

    def _validate_thresholds(self) -> None:
        if not (
            0.50 < self.buy_entry <= 1.0
            and 0.50 <= self.buy_exit < self.buy_entry
        ):
            raise ValueError(
                "BUY thresholds must satisfy "
                "0.50 <= buy_exit < buy_entry <= 1.0."
            )

        if not (
            0.0 <= self.sell_entry < 0.50
            and self.sell_entry < self.sell_exit < 0.50
        ):
            raise ValueError(
                "SELL thresholds must satisfy "
                "0.0 <= sell_entry < sell_exit < 0.50."
            )

        if not (0.50 <= self.moderate_agreement <= 1.0):
            raise ValueError("moderate_agreement must be between 0.50 and 1.0.")

        if not (self.moderate_agreement <= self.strong_agreement <= 1.0):
            raise ValueError(
                "strong_agreement must be >= moderate_agreement."
            )

    @staticmethod
    def _clip(value: Any, low: float = 0.0, high: float = 1.0) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return (low + high) / 2.0

        if not math.isfinite(number):
            return (low + high) / 2.0

        return float(max(low, min(high, number)))

    @staticmethod
    def _signed_score(value: Any) -> float:
        """Convert a directional score into [-1, 1]."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0

        if not math.isfinite(number):
            return 0.0

        # Accept either [-1, 1] or [0, 100].
        if 0.0 <= number <= 100.0:
            number = (number - 50.0) / 50.0

        return float(max(-1.0, min(1.0, number)))

    @staticmethod
    def _direction_from_probability(
        probability: float,
        buy_threshold: float,
        sell_threshold: float,
    ) -> str:
        if probability >= buy_threshold:
            return "BUY"
        if probability <= sell_threshold:
            return "SELL"
        return "WAIT"

    def _candidate_signal(
        self,
        probability: float,
        evidence_score: float,
    ) -> str:
        """
        Produce a candidate using both calibrated probability and supporting
        evidence.

        Probability remains the primary driver. Evidence can only reinforce
        a directional signal or move an ambiguous probability toward WAIT;
        it cannot manufacture an extreme probability.
        """
        if probability >= self.buy_entry and evidence_score >= -0.15:
            return "BUY"

        if probability <= self.sell_entry and evidence_score <= 0.15:
            return "SELL"

        return "WAIT"

    def _maintenance_signal(
        self,
        probability: float,
        evidence_score: float,
    ) -> str:
        """
        Hysteresis rule for an already-confirmed signal.

        A confirmed BUY survives moderate probability deterioration, while a
        confirmed SELL survives moderate upward movement. Strong contradictory
        evidence can still force WAIT.
        """
        current = self.state.signal

        if current == "BUY":
            if (
                probability >= self.buy_exit
                and evidence_score > -0.35
            ):
                return "BUY"

            if probability <= self.sell_entry:
                return "SELL"

            return "WAIT"

        if current == "SELL":
            if (
                probability <= self.sell_exit
                and evidence_score < 0.35
            ):
                return "SELL"

            if probability >= self.buy_entry:
                return "BUY"

            return "WAIT"

        return self._candidate_signal(probability, evidence_score)

    def _stability_label(self, agreement: float, pending_count: int) -> str:
        if pending_count >= self.persistence:
            return "HIGH"

        if agreement >= self.strong_agreement:
            return "HIGH"

        if agreement >= self.moderate_agreement:
            return "MEDIUM"

        return "LOW"

    def _build_reason(
        self,
        signal: str,
        probability: float,
        agreement: float,
        evidence_score: float,
        changed: bool,
        pending_signal: str,
        pending_count: int,
    ) -> str:
        p = f"{probability * 100:.1f}%"
        a = f"{agreement * 100:.0f}%"

        if changed:
            return (
                f"Signal confirmed as {signal}. "
                f"Calibrated up probability is {p}; "
                f"ensemble agreement is {a}; "
                f"evidence score is {evidence_score:+.2f}."
            )

        if pending_signal != "WAIT" and pending_count > 0:
            return (
                f"{pending_signal} is pending confirmation "
                f"({pending_count}/{self.persistence} observations). "
                f"Small changes are being filtered by signal persistence."
            )

        if signal == "BUY":
            return (
                f"BUY maintained by hysteresis: probability {p}, "
                f"agreement {a}, evidence {evidence_score:+.2f}."
            )

        if signal == "SELL":
            return (
                f"SELL maintained by hysteresis: probability {p}, "
                f"agreement {a}, evidence {evidence_score:+.2f}."
            )

        return (
            f"WAIT: probability {p}, agreement {a}, "
            f"evidence {evidence_score:+.2f} do not provide a stable "
            f"directional signal."
        )

    def reset(self, reason: str = "State reset.") -> None:
        """Clear the state, normally when the ticker/session changes."""
        self.state = SignalState(reason=str(reason))

    def update(
        self,
        probability: float,
        agreement: float = 0.0,
        trend_score: float = 0.0,
        regime_score: float = 0.0,
        mtf_score: float = 0.0,
        sentiment_score: float = 0.0,
        volume_score: float = 0.0,
        volatility_score: float = 0.0,
    ) -> dict:
        """
        Update the decision with one fresh market/model observation.

        Parameters
        ----------
        probability:
            Calibrated probability of an upward move, in [0, 1].

        agreement:
            Fraction of ensemble models agreeing with the current direction.

        *_score:
            Directional context scores. Each may be [-1, 1] or [0, 100].

        Returns
        -------
        dict
            Stable final signal plus diagnostics.
        """
        probability = self._clip(probability)
        agreement = self._clip(agreement)

        trend = self._signed_score(trend_score)
        regime = self._signed_score(regime_score)
        mtf = self._signed_score(mtf_score)
        sentiment = self._signed_score(sentiment_score)
        volume = self._signed_score(volume_score)
        volatility = self._signed_score(volatility_score)

        probability_score = (probability - 0.50) * 2.0

        # Probability is deliberately dominant. Context is supporting
        # evidence, not a mechanism for manufacturing high confidence.
        evidence_score = (
            0.50 * probability_score
            + 0.15 * trend
            + 0.12 * regime
            + 0.10 * mtf
            + 0.07 * sentiment
            + 0.04 * volume
            + 0.02 * volatility
        )
        evidence_score = max(-1.0, min(1.0, evidence_score))

        if self.state.signal in {"BUY", "SELL"}:
            candidate = self._maintenance_signal(
                probability,
                evidence_score,
            )
        else:
            candidate = self._candidate_signal(
                probability,
                evidence_score,
            )

        previous_signal = self.state.signal
        changed = False
        confirmed = False

        if candidate == previous_signal:
            self.state.pending_signal = "WAIT"
            self.state.pending_count = 0

        elif candidate == "WAIT":
            # A neutral observation does not immediately destroy a confirmed
            # directional signal. The maintenance rule above controls when
            # the signal is actually released.
            if previous_signal in {"BUY", "SELL"}:
                if (
                    (
                        previous_signal == "BUY"
                        and probability < self.buy_exit
                    )
                    or (
                        previous_signal == "SELL"
                        and probability > self.sell_exit
                    )
                ):
                    self.state.pending_signal = "WAIT"
                    self.state.pending_count += 1

                    if self.state.pending_count >= self.persistence:
                        self.state.signal = "WAIT"
                        self.state.pending_count = 0
                        changed = previous_signal != "WAIT"
                        confirmed = changed
                else:
                    self.state.pending_signal = "WAIT"
                    self.state.pending_count = 0
            else:
                self.state.pending_signal = "WAIT"
                self.state.pending_count = 0

        else:
            if self.state.pending_signal == candidate:
                self.state.pending_count += 1
            else:
                self.state.pending_signal = candidate
                self.state.pending_count = 1

            if self.state.pending_count >= self.persistence:
                self.state.signal = candidate
                self.state.pending_signal = "WAIT"
                self.state.pending_count = 0
                changed = previous_signal != candidate
                confirmed = True

        self.state.observations += 1
        self.state.last_probability = probability
        self.state.last_score = evidence_score
        self.state.stability = self._stability_label(
            agreement,
            self.state.pending_count,
        )

        reason = self._build_reason(
            signal=self.state.signal,
            probability=probability,
            agreement=agreement,
            evidence_score=evidence_score,
            changed=changed,
            pending_signal=self.state.pending_signal,
            pending_count=self.state.pending_count,
        )

        self.state.reason = reason

        components = {
            "probability_score": float(probability_score),
            "trend_score": float(trend),
            "regime_score": float(regime),
            "mtf_score": float(mtf),
            "sentiment_score": float(sentiment),
            "volume_score": float(volume),
            "volatility_score": float(volatility),
        }

        decision = SignalDecision(
            signal=self.state.signal,
            raw_probability=probability,
            evidence_score=evidence_score,
            agreement=agreement,
            stability=self.state.stability,
            changed=changed,
            confirmed=confirmed,
            pending_signal=self.state.pending_signal,
            pending_count=self.state.pending_count,
            reason=reason,
            components=components,
        )

        return decision.as_dict()

    def get_state(self) -> dict:
        """Return the current state without changing it."""
        return self.state.as_dict()


def create_signal_engine(**kwargs) -> SignalStabilityEngine:
    """Factory used by the application/predictor."""
    return SignalStabilityEngine(**kwargs)


def stable_signal(
    probability: float,
    agreement: float = 0.0,
    trend_score: float = 0.0,
    regime_score: float = 0.0,
    mtf_score: float = 0.0,
    sentiment_score: float = 0.0,
    volume_score: float = 0.0,
    volatility_score: float = 0.0,
    engine: Optional[SignalStabilityEngine] = None,
) -> dict:
    """
    Stateless-friendly helper.

    For repeated live updates, pass the same engine instance so hysteresis
    and persistence can operate across observations.
    """
    active_engine = engine or SignalStabilityEngine()

    return active_engine.update(
        probability=probability,
        agreement=agreement,
        trend_score=trend_score,
        regime_score=regime_score,
        mtf_score=mtf_score,
        sentiment_score=sentiment_score,
        volume_score=volume_score,
        volatility_score=volatility_score,
    )


__all__ = [
    "SignalState",
    "SignalDecision",
    "SignalStabilityEngine",
    "create_signal_engine",
    "stable_signal",
]
