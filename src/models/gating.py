"""
Final decision and safety gating for AI Swing Analyser.

This module converts model outputs and validation diagnostics into a
conservative research-stage decision.

The gate is intentionally designed to reject weak or contradictory setups.
It does not guarantee profitability and must never override failed
validation, leakage checks, calibration, or range-validation checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, Mapping, Optional, Sequence

import math


class Decision(str, Enum):
    """Possible final trading decisions."""

    BUY = "BUY"
    SELL = "SELL"
    WAIT = "WAIT"
    NO_HIGH_CONFIDENCE_SETUP = "NO_HIGH_CONFIDENCE_SETUP"
    REJECTED = "REJECTED"


class GateStatus(str, Enum):
    """Status of an individual safety gate."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    NOT_EVALUATED = "NOT_EVALUATED"


@dataclass(frozen=True)
class GateResult:
    """Result of one safety gate."""

    name: str
    status: GateStatus
    score: float
    reason: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Gate name cannot be empty.")

        if not math.isfinite(float(self.score)):
            raise ValueError("Gate score must be finite.")

        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("Gate score must be between 0 and 1.")


@dataclass
class DecisionConfig:
    """
    Configuration for the final decision engine.

    Thresholds are intentionally conservative defaults. They should be
    optimized only using leakage-free validation data.
    """

    minimum_probability: float = 0.60
    high_confidence_probability: float = 0.70

    maximum_model_disagreement: float = 0.25
    minimum_model_agreement: float = 0.60

    minimum_timeframe_alignment: float = 0.60

    minimum_risk_reward: float = 2.0

    minimum_range_coverage: float = 0.70

    minimum_validation_accuracy: float = 0.95

    minimum_regime_stability: float = 0.60

    allow_buy: bool = True
    allow_sell: bool = True

    require_validation_pass: bool = True
    require_calibration_pass: bool = True
    require_range_validation_pass: bool = True
    require_regime_validation_pass: bool = True

    # A minimum number of independent positive gates required before
    # a high-confidence setup can be considered.
    minimum_positive_gates: int = 5

    def __post_init__(self) -> None:
        probability_values = (
            self.minimum_probability,
            self.high_confidence_probability,
            self.maximum_model_disagreement,
            self.minimum_model_agreement,
            self.minimum_timeframe_alignment,
            self.minimum_range_coverage,
            self.minimum_regime_stability,
        )

        for value in probability_values:
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    "Probability/ratio thresholds must be between 0 and 1."
                )

        if self.high_confidence_probability < self.minimum_probability:
            raise ValueError(
                "high_confidence_probability must be >= minimum_probability."
            )

        if self.minimum_risk_reward <= 0:
            raise ValueError("minimum_risk_reward must be positive.")

        if self.minimum_validation_accuracy < 0.0:
            raise ValueError(
                "minimum_validation_accuracy cannot be negative."
            )

        if self.minimum_positive_gates < 1:
            raise ValueError("minimum_positive_gates must be at least 1.")


@dataclass
class DecisionInput:
    """
    Inputs required by the final decision engine.

    Validation flags should come from genuinely unseen research results,
    not from the current prediction itself.
    """

    up_probability: float
    down_probability: float

    model_disagreement: float = 0.0
    model_agreement: Optional[float] = None

    timeframe_alignment: Optional[float] = None

    risk_reward: Optional[float] = None

    range_coverage: Optional[float] = None

    validation_accuracy: Optional[float] = None
    validation_passed: bool = False

    calibration_passed: bool = False
    range_validation_passed: bool = False
    regime_validation_passed: bool = False

    regime_stability: Optional[float] = None

    validation_available: bool = False
    calibration_available: bool = False
    range_validation_available: bool = False
    regime_validation_available: bool = False

    market_regime: Optional[str] = None

    target_return: Optional[float] = None
    lower_return: Optional[float] = None
    upper_return: Optional[float] = None

    current_price: Optional[float] = None

    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate_probability(self.up_probability, "up_probability")
        self._validate_probability(self.down_probability, "down_probability")

        if not 0.0 <= float(self.model_disagreement) <= 1.0:
            raise ValueError("model_disagreement must be between 0 and 1.")

        optional_ratios = {
            "model_agreement": self.model_agreement,
            "timeframe_alignment": self.timeframe_alignment,
            "range_coverage": self.range_coverage,
            "regime_stability": self.regime_stability,
        }

        for name, value in optional_ratios.items():
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")

        if self.risk_reward is not None and self.risk_reward < 0:
            raise ValueError("risk_reward cannot be negative.")

        if self.validation_accuracy is not None:
            if self.validation_accuracy < 0:
                raise ValueError("validation_accuracy cannot be negative.")

        if self.current_price is not None and self.current_price <= 0:
            raise ValueError("current_price must be positive.")

    @staticmethod
    def _validate_probability(value: float, name: str) -> None:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")


@dataclass
class DecisionResult:
    """Final decision returned by the gating engine."""

    decision: Decision

    confidence: float

    direction: str

    gates: Sequence[GateResult]

    reasons: Sequence[str]

    warnings: Sequence[str]

    positive_gate_count: int

    failed_gate_count: int

    trade_allowed: bool

    high_confidence: bool

    metadata: Dict[str, object] = field(default_factory=dict)

    def passed_gates(self) -> list[GateResult]:
        """Return all passing gates."""
        return [gate for gate in self.gates if gate.status == GateStatus.PASS]

    def failed_gates(self) -> list[GateResult]:
        """Return all failed gates."""
        return [gate for gate in self.gates if gate.status == GateStatus.FAIL]

    def warning_gates(self) -> list[GateResult]:
        """Return all warning gates."""
        return [
            gate for gate in self.gates if gate.status == GateStatus.WARNING
        ]


class DecisionEngine:
    """
    Conservative final decision engine.

    The engine is deliberately deterministic and rule-based at this stage.
    Machine learning should produce the underlying probabilities/ranges;
    this module decides whether those outputs satisfy the research safety
    constraints.

    Important:
        This engine does not train models and does not claim that a BUY or
        SELL decision will be profitable.
    """

    def __init__(self, config: Optional[DecisionConfig] = None) -> None:
        self.config = config or DecisionConfig()

    def evaluate(
        self,
        inputs: DecisionInput,
    ) -> DecisionResult:
        """Evaluate all available safety gates."""

        gates: list[GateResult] = []
        reasons: list[str] = []
        warnings: list[str] = []

        direction, confidence = self._determine_direction(inputs)

        # --------------------------------------------------------------
        # 1. Probability gate
        # --------------------------------------------------------------
        gates.append(
            self._probability_gate(
                direction=direction,
                confidence=confidence,
            )
        )

        # --------------------------------------------------------------
        # 2. Model agreement/disagreement
        # --------------------------------------------------------------
        gates.append(self._agreement_gate(inputs))

        # --------------------------------------------------------------
        # 3. Multi-timeframe alignment
        # --------------------------------------------------------------
        gates.append(self._timeframe_gate(inputs))

        # --------------------------------------------------------------
        # 4. Risk/reward
        # --------------------------------------------------------------
        gates.append(self._risk_reward_gate(inputs))

        # --------------------------------------------------------------
        # 5. Price-range validation
        # --------------------------------------------------------------
        gates.append(self._range_gate(inputs))

        # --------------------------------------------------------------
        # 6. Historical model validation
        # --------------------------------------------------------------
        gates.append(self._validation_gate(inputs))

        # --------------------------------------------------------------
        # 7. Probability calibration
        # --------------------------------------------------------------
        gates.append(self._calibration_gate(inputs))

        # --------------------------------------------------------------
        # 8. Market-regime stability
        # --------------------------------------------------------------
        gates.append(self._regime_gate(inputs))

        for gate in gates:
            if gate.status == GateStatus.FAIL:
                reasons.append(gate.reason)

            elif gate.status == GateStatus.WARNING:
                warnings.append(gate.reason)

            elif gate.status == GateStatus.PASS:
                reasons.append(gate.reason)

        positive_gate_count = sum(
            gate.status == GateStatus.PASS for gate in gates
        )

        failed_gate_count = sum(
            gate.status == GateStatus.FAIL for gate in gates
        )

        # Critical safety conditions.
        critical_failure = self._has_critical_failure(gates)

        enough_positive_gates = (
            positive_gate_count >= self.config.minimum_positive_gates
        )

        high_confidence = (
            confidence >= self.config.high_confidence_probability
            and enough_positive_gates
            and failed_gate_count == 0
        )

        trade_allowed = (
            not critical_failure
            and enough_positive_gates
            and failed_gate_count == 0
        )

        if critical_failure:
            final_decision = Decision.REJECTED

        elif not trade_allowed:
            final_decision = Decision.NO_HIGH_CONFIDENCE_SETUP

        elif direction == "BUY":
            final_decision = Decision.BUY

        elif direction == "SELL":
            final_decision = Decision.SELL

        else:
            final_decision = Decision.WAIT

        return DecisionResult(
            decision=final_decision,
            confidence=float(confidence),
            direction=direction,
            gates=gates,
            reasons=reasons,
            warnings=warnings,
            positive_gate_count=positive_gate_count,
            failed_gate_count=failed_gate_count,
            trade_allowed=trade_allowed,
            high_confidence=high_confidence,
            metadata={
                "market_regime": inputs.market_regime,
                "up_probability": inputs.up_probability,
                "down_probability": inputs.down_probability,
            },
        )

    # ------------------------------------------------------------------
    # Direction
    # ------------------------------------------------------------------

    def _determine_direction(
        self,
        inputs: DecisionInput,
    ) -> tuple[str, float]:
        """
        Determine dominant direction.

        A probability difference below the minimum threshold produces WAIT.
        """

        up = float(inputs.up_probability)
        down = float(inputs.down_probability)

        if up >= down:
            direction = "BUY"
            confidence = up
        else:
            direction = "SELL"
            confidence = down

        if confidence < self.config.minimum_probability:
            return "WAIT", confidence

        if direction == "BUY" and not self.config.allow_buy:
            return "WAIT", confidence

        if direction == "SELL" and not self.config.allow_sell:
            return "WAIT", confidence

        return direction, confidence

    # ------------------------------------------------------------------
    # Individual gates
    # ------------------------------------------------------------------

    def _probability_gate(
        self,
        direction: str,
        confidence: float,
    ) -> GateResult:
        if direction == "WAIT":
            return GateResult(
                name="direction_probability",
                status=GateStatus.FAIL,
                score=confidence,
                reason=(
                    f"Directional probability {confidence:.1%} is below "
                    f"the minimum threshold of "
                    f"{self.config.minimum_probability:.1%}."
                ),
            )

        if confidence >= self.config.high_confidence_probability:
            status = GateStatus.PASS
        else:
            status = GateStatus.WARNING

        return GateResult(
            name="direction_probability",
            status=status,
            score=confidence,
            reason=(
                f"{direction} probability is {confidence:.1%}."
            ),
        )

    def _agreement_gate(
        self,
        inputs: DecisionInput,
    ) -> GateResult:
        disagreement = float(inputs.model_disagreement)

        if inputs.model_agreement is not None:
            agreement = float(inputs.model_agreement)
        else:
            agreement = 1.0 - disagreement

        if disagreement > self.config.maximum_model_disagreement:
            return GateResult(
                name="model_agreement",
                status=GateStatus.FAIL,
                score=agreement,
                reason=(
                    f"Model disagreement is {disagreement:.1%}, exceeding "
                    f"the maximum allowed "
                    f"{self.config.maximum_model_disagreement:.1%}."
                ),
            )

        if agreement < self.config.minimum_model_agreement:
            return GateResult(
                name="model_agreement",
                status=GateStatus.FAIL,
                score=agreement,
                reason=(
                    f"Model agreement is {agreement:.1%}, below the "
                    f"required {self.config.minimum_model_agreement:.1%}."
                ),
            )

        return GateResult(
            name="model_agreement",
            status=GateStatus.PASS,
            score=agreement,
            reason=f"Models agree at approximately {agreement:.1%}.",
        )

    def _timeframe_gate(
        self,
        inputs: DecisionInput,
    ) -> GateResult:
        if inputs.timeframe_alignment is None:
            return GateResult(
                name="multi_timeframe_alignment",
                status=GateStatus.NOT_EVALUATED,
                score=0.0,
                reason=(
                    "Multi-timeframe alignment was not available. "
                    "No additional confirmation was granted."
                ),
            )

        alignment = float(inputs.timeframe_alignment)

        if alignment < self.config.minimum_timeframe_alignment:
            return GateResult(
                name="multi_timeframe_alignment",
                status=GateStatus.FAIL,
                score=alignment,
                reason=(
                    f"Multi-timeframe alignment is {alignment:.1%}, below "
                    f"the required {self.config.minimum_timeframe_alignment:.1%}."
                ),
            )

        return GateResult(
            name="multi_timeframe_alignment",
            status=GateStatus.PASS,
            score=alignment,
            reason=f"Multi-timeframe alignment is {alignment:.1%}.",
        )

    def _risk_reward_gate(
        self,
        inputs: DecisionInput,
    ) -> GateResult:
        if inputs.risk_reward is None:
            return GateResult(
                name="risk_reward",
                status=GateStatus.NOT_EVALUATED,
                score=0.0,
                reason=(
                    "Risk/reward has not been calculated. "
                    "A trade should not receive high-confidence status."
                ),
            )

        rr = float(inputs.risk_reward)

        if rr < self.config.minimum_risk_reward:
            return GateResult(
                name="risk_reward",
                status=GateStatus.FAIL,
                score=self._ratio_score(
                    rr,
                    self.config.minimum_risk_reward,
                ),
                reason=(
                    f"Risk/reward of {rr:.2f} is below the required "
                    f"{self.config.minimum_risk_reward:.2f}."
                ),
            )

        return GateResult(
            name="risk_reward",
            status=GateStatus.PASS,
            score=1.0,
            reason=f"Risk/reward is {rr:.2f}.",
        )

    def _range_gate(
        self,
        inputs: DecisionInput,
    ) -> GateResult:
        if not inputs.range_validation_available:
            return GateResult(
                name="range_validation",
                status=GateStatus.NOT_EVALUATED,
                score=0.0,
                reason=(
                    "Price-range validation is not available. "
                    "Range reliability has not been established."
                ),
            )

        if not inputs.range_validation_passed:
            return GateResult(
                name="range_validation",
                status=GateStatus.FAIL,
                score=0.0,
                reason="Predicted price-range validation failed.",
            )

        if inputs.range_coverage is not None:
            coverage = float(inputs.range_coverage)

            if coverage < self.config.minimum_range_coverage:
                return GateResult(
                    name="range_validation",
                    status=GateStatus.FAIL,
                    score=coverage,
                    reason=(
                        f"Observed range coverage of {coverage:.1%} is "
                        f"below the minimum "
                        f"{self.config.minimum_range_coverage:.1%}."
                    ),
                )

            return GateResult(
                name="range_validation",
                status=GateStatus.PASS,
                score=coverage,
                reason=(
                    f"Predicted range validation passed with "
                    f"{coverage:.1%} coverage."
                ),
            )

        return GateResult(
            name="range_validation",
            status=GateStatus.PASS,
            score=1.0,
            reason="Predicted range validation passed.",
        )

    def _validation_gate(
        self,
        inputs: DecisionInput,
    ) -> GateResult:
        if not inputs.validation_available:
            if self.config.require_validation_pass:
                return GateResult(
                    name="historical_validation",
                    status=GateStatus.FAIL,
                    score=0.0,
                    reason=(
                        "No qualifying historical out-of-sample validation "
                        "result is available."
                    ),
                )

            return GateResult(
                name="historical_validation",
                status=GateStatus.WARNING,
                score=0.0,
                reason="Historical validation was not available.",
            )

        if not inputs.validation_passed:
            return GateResult(
                name="historical_validation",
                status=GateStatus.FAIL,
                score=0.0,
                reason="Historical validation gate failed.",
            )

        if inputs.validation_accuracy is not None:
            accuracy = float(inputs.validation_accuracy)

            if accuracy < self.config.minimum_validation_accuracy:
                return GateResult(
                    name="historical_validation",
                    status=GateStatus.FAIL,
                    score=min(accuracy, 1.0),
                    reason=(
                        f"Validation accuracy {accuracy:.1%} is below "
                        f"the research threshold "
                        f"{self.config.minimum_validation_accuracy:.1%}."
                    ),
                )

            return GateResult(
                name="historical_validation",
                status=GateStatus.PASS,
                score=min(accuracy, 1.0),
                reason=(
                    f"Historical validation passed with "
                    f"{accuracy:.1%} accuracy."
                ),
            )

        return GateResult(
            name="historical_validation",
            status=GateStatus.PASS,
            score=1.0,
            reason="Historical validation gate passed.",
        )

    def _calibration_gate(
        self,
        inputs: DecisionInput,
    ) -> GateResult:
        if not inputs.calibration_available:
            if self.config.require_calibration_pass:
                return GateResult(
                    name="probability_calibration",
                    status=GateStatus.FAIL,
                    score=0.0,
                    reason=(
                        "Probability calibration has not been independently "
                        "validated."
                    ),
                )

            return GateResult(
                name="probability_calibration",
                status=GateStatus.WARNING,
                score=0.0,
                reason="Probability calibration was not available.",
            )

        if not inputs.calibration_passed:
            return GateResult(
                name="probability_calibration",
                status=GateStatus.FAIL,
                score=0.0,
                reason="Probability calibration failed.",
            )

        return GateResult(
            name="probability_calibration",
            status=GateStatus.PASS,
            score=1.0,
            reason="Probability calibration passed.",
        )

    def _regime_gate(
        self,
        inputs: DecisionInput,
    ) -> GateResult:
        if not inputs.regime_validation_available:
            if self.config.require_regime_validation_pass:
                return GateResult(
                    name="regime_stability",
                    status=GateStatus.FAIL,
                    score=0.0,
                    reason=(
                        "Regime-specific validation is not available."
                    ),
                )

            return GateResult(
                name="regime_stability",
                status=GateStatus.WARNING,
                score=0.0,
                reason="Regime validation was not available.",
            )

        if not inputs.regime_validation_passed:
            return GateResult(
                name="regime_stability",
                status=GateStatus.FAIL,
                score=0.0,
                reason="Regime stability validation failed.",
            )

        stability = (
            float(inputs.regime_stability)
            if inputs.regime_stability is not None
            else 1.0
        )

        if stability < self.config.minimum_regime_stability:
            return GateResult(
                name="regime_stability",
                status=GateStatus.FAIL,
                score=stability,
                reason=(
                    f"Regime stability score {stability:.1%} is below "
                    f"the required {self.config.minimum_regime_stability:.1%}."
                ),
            )

        return GateResult(
            name="regime_stability",
            status=GateStatus.PASS,
            score=stability,
            reason=(
                f"Regime stability score is {stability:.1%}."
            ),
        )

    # ------------------------------------------------------------------
    # Critical safety logic
    # ------------------------------------------------------------------

    def _has_critical_failure(
        self,
        gates: Iterable[GateResult],
    ) -> bool:
        critical_gate_names = {
            "historical_validation",
            "probability_calibration",
            "range_validation",
            "regime_stability",
        }

        return any(
            gate.name in critical_gate_names
            and gate.status == GateStatus.FAIL
            for gate in gates
        )

    @staticmethod
    def _ratio_score(
        actual: float,
        required: float,
    ) -> float:
        if required <= 0:
            return 1.0

        return max(0.0, min(1.0, actual / required))


def calculate_confidence(
    up_probability: float,
    down_probability: float,
    model_disagreement: float = 0.0,
    timeframe_alignment: Optional[float] = None,
) -> float:
    """
    Calculate a diagnostic confidence score.

    This is NOT a calibrated probability.

    The function combines directional strength with agreement and,
    when available, multi-timeframe confirmation.
    """

    if not 0.0 <= up_probability <= 1.0:
        raise ValueError("up_probability must be between 0 and 1.")

    if not 0.0 <= down_probability <= 1.0:
        raise ValueError("down_probability must be between 0 and 1.")

    if not 0.0 <= model_disagreement <= 1.0:
        raise ValueError("model_disagreement must be between 0 and 1.")

    directional_strength = max(
        up_probability,
        down_probability,
    )

    agreement_strength = 1.0 - model_disagreement

    components = [
        directional_strength,
        agreement_strength,
    ]

    if timeframe_alignment is not None:
        if not 0.0 <= timeframe_alignment <= 1.0:
            raise ValueError(
                "timeframe_alignment must be between 0 and 1."
            )

        components.append(timeframe_alignment)

    return float(sum(components) / len(components))


def build_decision_input(
    *,
    up_probability: float,
    down_probability: float,
    model_disagreement: float = 0.0,
    timeframe_alignment: Optional[float] = None,
    risk_reward: Optional[float] = None,
    range_coverage: Optional[float] = None,
    validation_accuracy: Optional[float] = None,
    validation_passed: bool = False,
    calibration_passed: bool = False,
    range_validation_passed: bool = False,
    regime_validation_passed: bool = False,
    regime_stability: Optional[float] = None,
    validation_available: bool = False,
    calibration_available: bool = False,
    range_validation_available: bool = False,
    regime_validation_available: bool = False,
    market_regime: Optional[str] = None,
    target_return: Optional[float] = None,
    lower_return: Optional[float] = None,
    upper_return: Optional[float] = None,
    current_price: Optional[float] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> DecisionInput:
    """
    Convenience constructor for DecisionInput.
    """

    return DecisionInput(
        up_probability=up_probability,
        down_probability=down_probability,
        model_disagreement=model_disagreement,
        timeframe_alignment=timeframe_alignment,
        risk_reward=risk_reward,
        range_coverage=range_coverage,
        validation_accuracy=validation_accuracy,
        validation_passed=validation_passed,
        calibration_passed=calibration_passed,
        range_validation_passed=range_validation_passed,
        regime_validation_passed=regime_validation_passed,
        regime_stability=regime_stability,
        validation_available=validation_available,
        calibration_available=calibration_available,
        range_validation_available=range_validation_available,
        regime_validation_available=regime_validation_available,
        market_regime=market_regime,
        target_return=target_return,
        lower_return=lower_return,
        upper_return=upper_return,
        current_price=current_price,
        metadata=dict(metadata or {}),
    )


def decision_to_dict(
    result: DecisionResult,
) -> Dict[str, object]:
    """Convert a DecisionResult into a serializable dictionary."""

    return {
        "decision": result.decision.value,
        "confidence": result.confidence,
        "direction": result.direction,
        "positive_gate_count": result.positive_gate_count,
        "failed_gate_count": result.failed_gate_count,
        "trade_allowed": result.trade_allowed,
        "high_confidence": result.high_confidence,
        "reasons": list(result.reasons),
        "warnings": list(result.warnings),
        "gates": [
            {
                "name": gate.name,
                "status": gate.status.value,
                "score": gate.score,
                "reason": gate.reason,
            }
            for gate in result.gates
        ],
        "metadata": dict(result.metadata),
    }


def gate_summary(
    result: DecisionResult,
) -> Dict[str, object]:
    """Return a compact summary suitable for a dashboard."""

    return {
        "decision": result.decision.value,
        "direction": result.direction,
        "confidence": round(result.confidence, 4),
        "trade_allowed": result.trade_allowed,
        "high_confidence": result.high_confidence,
        "passed_gates": result.positive_gate_count,
        "failed_gates": result.failed_gate_count,
        "warnings": len(result.warning_gates()),
    }


__all__ = [
    "Decision",
    "GateStatus",
    "GateResult",
    "DecisionConfig",
    "DecisionInput",
    "DecisionResult",
    "DecisionEngine",
    "calculate_confidence",
    "build_decision_input",
    "decision_to_dict",
    "gate_summary",
]
