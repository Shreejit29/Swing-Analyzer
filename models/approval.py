"""
Final production-readiness approval engine.

Purpose
-------
Determine whether a stock-analysis model has passed all required
research and robustness gates before it can be used for live inference.

Approval philosophy
-------------------
A model is NOT production-ready merely because:

    accuracy >= 95%

Instead, approval requires evidence across:

    1. Validation
    2. Final holdout
    3. Leakage checks
    4. Probability calibration
    5. Target-range validation
    6. Regime stability
    7. Trading backtest
    8. Robustness

The final holdout must remain untouched until the research configuration
has been frozen.

A failed critical gate results in REJECTED status.

A warning does not automatically reject a model, but should be visible
to the researcher.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


# ----------------------------------------------------------------------
# Status
# ----------------------------------------------------------------------


class ApprovalStatus(str, Enum):
    """
    Final model lifecycle state.
    """

    APPROVED = "APPROVED"

    RESEARCH = "RESEARCH"

    REJECTED = "REJECTED"

    HOLD = "HOLD"


class GateStatus(str, Enum):
    """
    Individual gate result.
    """

    PASS = "PASS"

    FAIL = "FAIL"

    WARNING = "WARNING"

    NOT_EVALUATED = "NOT_EVALUATED"


# ----------------------------------------------------------------------
# Gate result
# ----------------------------------------------------------------------


@dataclass
class ApprovalGate:
    """
    Result of one production-readiness gate.
    """

    name: str

    status: GateStatus

    value: Optional[float]

    threshold: Optional[float]

    message: str

    critical: bool = True

    evidence: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def passed(self) -> bool:
        return self.status == GateStatus.PASS

    @property
    def failed(self) -> bool:
        return self.status == GateStatus.FAIL


# ----------------------------------------------------------------------
# Approval configuration
# ----------------------------------------------------------------------


@dataclass
class ApprovalConfig:
    """
    Final approval thresholds.

    These values are deliberately conservative starting points.

    They should be evaluated empirically rather than assumed to guarantee
    future performance.
    """

    minimum_validation_accuracy: float = 0.95

    minimum_final_holdout_accuracy: float = 0.95

    maximum_validation_holdout_gap: float = 0.10

    maximum_calibration_brier: float = 0.25

    maximum_calibration_ece: float = 0.15

    minimum_range_coverage: float = 0.70

    maximum_range_coverage: float = 0.95

    minimum_regime_stability: float = 0.60

    minimum_backtest_profit_factor: float = 1.20

    minimum_backtest_sharpe: float = 0.80

    maximum_backtest_drawdown: float = 0.30

    minimum_backtest_trades: int = 30

    minimum_robustness_score: float = 0.60

    require_validation: bool = True

    require_final_holdout: bool = True

    require_no_leakage: bool = True

    require_calibration: bool = True

    require_range_validation: bool = True

    require_regime_validation: bool = True

    require_backtest: bool = True

    require_robustness: bool = True

    allow_warnings: bool = True

    def __post_init__(self) -> None:
        probabilities = [
            self.minimum_validation_accuracy,
            self.minimum_final_holdout_accuracy,
            self.minimum_range_coverage,
            self.maximum_range_coverage,
            self.maximum_calibration_brier,
            self.maximum_calibration_ece,
            self.minimum_regime_stability,
            self.maximum_backtest_drawdown,
            self.minimum_robustness_score,
        ]

        if any(
            not 0.0 <= value <= 1.0
            for value in probabilities
        ):
            raise ValueError(
                "Probability/ratio thresholds must be between 0 and 1."
            )

        if (
            self.minimum_validation_accuracy
            > 1.0
        ):
            raise ValueError(
                "Invalid validation accuracy threshold."
            )

        if (
            self.maximum_validation_holdout_gap
            < 0.0
        ):
            raise ValueError(
                "Validation/holdout gap cannot be negative."
            )

        if (
            self.minimum_range_coverage
            > self.maximum_range_coverage
        ):
            raise ValueError(
                "Invalid range coverage thresholds."
            )

        if (
            self.minimum_backtest_profit_factor
            < 0
        ):
            raise ValueError(
                "Profit-factor threshold cannot be negative."
            )

        if (
            self.minimum_backtest_sharpe
            < 0
        ):
            raise ValueError(
                "Sharpe threshold cannot be negative."
            )

        if (
            self.minimum_backtest_trades
            < 1
        ):
            raise ValueError(
                "minimum_backtest_trades must be positive."
            )


# ----------------------------------------------------------------------
# Approval report
# ----------------------------------------------------------------------


@dataclass
class ApprovalReport:
    """
    Complete production approval report.
    """

    status: ApprovalStatus

    gates: list[
        ApprovalGate
    ]

    critical_failures: list[str]

    warnings: list[str]

    passed_gates: int

    failed_gates: int

    warning_gates: int

    not_evaluated_gates: int

    score: float

    model_id: Optional[str] = None

    experiment_id: Optional[str] = None

    notes: list[str] = field(
        default_factory=list
    )

    @property
    def approved(self) -> bool:
        return self.status == ApprovalStatus.APPROVED

    @property
    def rejected(self) -> bool:
        return self.status == ApprovalStatus.REJECTED

    def summary(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "approved": self.approved,
            "rejected": self.rejected,
            "score": self.score,
            "passed_gates": self.passed_gates,
            "failed_gates": self.failed_gates,
            "warning_gates": self.warning_gates,
            "not_evaluated_gates": (
                self.not_evaluated_gates
            ),
            "critical_failures": (
                self.critical_failures
            ),
            "warnings": self.warnings,
            "model_id": self.model_id,
            "experiment_id": self.experiment_id,
        }


# ----------------------------------------------------------------------
# Approval engine
# ----------------------------------------------------------------------


class ProductionApprovalEngine:
    """
    Final approval decision engine.

    Each gate is independently evaluated.

    Critical gates cannot be bypassed simply because the aggregate score
    is high.
    """

    def __init__(
        self,
        config: Optional[
            ApprovalConfig
        ] = None,
    ) -> None:
        self.config = (
            config
            or ApprovalConfig()
        )

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        *,
        validation_accuracy: Optional[float] = None,
        final_holdout_accuracy: Optional[float] = None,
        leakage_free: Optional[bool] = None,
        calibration_brier: Optional[float] = None,
        calibration_ece: Optional[float] = None,
        range_coverage: Optional[float] = None,
        regime_stability: Optional[float] = None,
        backtest_profit_factor: Optional[float] = None,
        backtest_sharpe: Optional[float] = None,
        backtest_max_drawdown: Optional[float] = None,
        backtest_trade_count: Optional[int] = None,
        robustness_score: Optional[float] = None,
        validation_passed: Optional[bool] = None,
        calibration_passed: Optional[bool] = None,
        range_passed: Optional[bool] = None,
        regime_passed: Optional[bool] = None,
        backtest_passed: Optional[bool] = None,
        robustness_passed: Optional[bool] = None,
        model_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
    ) -> ApprovalReport:
        gates: list[
            ApprovalGate
        ] = []

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        gates.append(
            self._accuracy_gate(
                name="validation_accuracy",
                value=validation_accuracy,
                threshold=(
                    self.config
                    .minimum_validation_accuracy
                ),
                passed_override=(
                    validation_passed
                ),
                critical=self.config.require_validation,
            )
        )

        # --------------------------------------------------------------
        # Final holdout
        # --------------------------------------------------------------

        gates.append(
            self._accuracy_gate(
                name="final_holdout_accuracy",
                value=final_holdout_accuracy,
                threshold=(
                    self.config
                    .minimum_final_holdout_accuracy
                ),
                critical=self.config.require_final_holdout,
            )
        )

        # --------------------------------------------------------------
        # Generalization gap
        # --------------------------------------------------------------

        gates.append(
            self._generalization_gate(
                validation_accuracy=(
                    validation_accuracy
                ),
                holdout_accuracy=(
                    final_holdout_accuracy
                ),
            )
        )

        # --------------------------------------------------------------
        # Leakage
        # --------------------------------------------------------------

        gates.append(
            self._boolean_gate(
                name="no_data_leakage",
                value=leakage_free,
                critical=self.config.require_no_leakage,
            )
        )

        # --------------------------------------------------------------
        # Calibration
        # --------------------------------------------------------------

        gates.append(
            self._threshold_gate(
                name="calibration_brier",
                value=calibration_brier,
                threshold=(
                    self.config
                    .maximum_calibration_brier
                ),
                direction="max",
                critical=self.config.require_calibration,
            )
        )

        gates.append(
            self._threshold_gate(
                name="calibration_ece",
                value=calibration_ece,
                threshold=(
                    self.config
                    .maximum_calibration_ece
                ),
                direction="max",
                critical=self.config.require_calibration,
            )
        )

        if calibration_passed is not None:
            gates.append(
                self._boolean_gate(
                    name="calibration_pipeline",
                    value=calibration_passed,
                    critical=self.config.require_calibration,
                )
            )

        # --------------------------------------------------------------
        # Range
        # --------------------------------------------------------------

        gates.append(
            self._range_gate(
                range_coverage=range_coverage,
                range_passed=range_passed,
            )
        )

        # --------------------------------------------------------------
        # Regime stability
        # --------------------------------------------------------------

        gates.append(
            self._threshold_gate(
                name="regime_stability",
                value=regime_stability,
                threshold=(
                    self.config
                    .minimum_regime_stability
                ),
                direction="min",
                critical=self.config.require_regime_validation,
            )
        )

        if regime_passed is not None:
            gates.append(
                self._boolean_gate(
                    name="regime_validation",
                    value=regime_passed,
                    critical=self.config.require_regime_validation,
                )
            )

        # --------------------------------------------------------------
        # Backtest
        # --------------------------------------------------------------

        gates.append(
            self._threshold_gate(
                name="backtest_profit_factor",
                value=backtest_profit_factor,
                threshold=(
                    self.config
                    .minimum_backtest_profit_factor
                ),
                direction="min",
                critical=self.config.require_backtest,
            )
        )

        gates.append(
            self._threshold_gate(
                name="backtest_sharpe",
                value=backtest_sharpe,
                threshold=(
                    self.config
                    .minimum_backtest_sharpe
                ),
                direction="min",
                critical=self.config.require_backtest,
            )
        )

        gates.append(
            self._threshold_gate(
                name="backtest_max_drawdown",
                value=backtest_max_drawdown,
                threshold=(
                    self.config
                    .maximum_backtest_drawdown
                ),
                direction="max_absolute",
                critical=self.config.require_backtest,
            )
        )

        gates.append(
            self._threshold_gate(
                name="backtest_trade_count",
                value=(
                    float(backtest_trade_count)
                    if backtest_trade_count is not None
                    else None
                ),
                threshold=float(
                    self.config
                    .minimum_backtest_trades
                ),
                direction="min",
                critical=self.config.require_backtest,
            )
        )

        if backtest_passed is not None:
            gates.append(
                self._boolean_gate(
                    name="backtest_pipeline",
                    value=backtest_passed,
                    critical=self.config.require_backtest,
                )
            )

        # --------------------------------------------------------------
        # Robustness
        # --------------------------------------------------------------

        gates.append(
            self._threshold_gate(
                name="robustness_score",
                value=robustness_score,
                threshold=(
                    self.config
                    .minimum_robustness_score
                ),
                direction="min",
                critical=self.config.require_robustness,
            )
        )

        if robustness_passed is not None:
            gates.append(
                self._boolean_gate(
                    name="robustness_pipeline",
                    value=robustness_passed,
                    critical=self.config.require_robustness,
                )
            )

        # --------------------------------------------------------------
        # Final decision
        # --------------------------------------------------------------

        return self._build_report(
            gates=gates,
            model_id=model_id,
            experiment_id=experiment_id,
        )

    # ------------------------------------------------------------------
    # Accuracy gate
    # ------------------------------------------------------------------

    @staticmethod
    def _accuracy_gate(
        *,
        name: str,
        value: Optional[float],
        threshold: float,
        passed_override: Optional[bool] = None,
        critical: bool = True,
    ) -> ApprovalGate:
        if passed_override is not None:
            return ApprovalGate(
                name=name,
                status=(
                    GateStatus.PASS
                    if passed_override
                    else GateStatus.FAIL
                ),
                value=value,
                threshold=threshold,
                message=(
                    "Pipeline gate passed."
                    if passed_override
                    else "Pipeline gate failed."
                ),
                critical=critical,
            )

        if value is None:
            return ApprovalGate(
                name=name,
                status=GateStatus.NOT_EVALUATED,
                value=None,
                threshold=threshold,
                message=(
                    "Metric was not supplied."
                ),
                critical=critical,
            )

        if not np.isfinite(value):
            return ApprovalGate(
                name=name,
                status=GateStatus.FAIL,
                value=value,
                threshold=threshold,
                message=(
                    "Metric is not finite."
                ),
                critical=critical,
            )

        passed = (
            value >= threshold
        )

        return ApprovalGate(
            name=name,
            status=(
                GateStatus.PASS
                if passed
                else GateStatus.FAIL
            ),
            value=float(value),
            threshold=threshold,
            message=(
                f"Accuracy {value:.4f} "
                f"{'meets' if passed else 'does not meet'} "
                f"required {threshold:.4f}."
            ),
            critical=critical,
        )

    # ------------------------------------------------------------------
    # Generalization
    # ------------------------------------------------------------------

    def _generalization_gate(
        self,
        *,
        validation_accuracy: Optional[float],
        holdout_accuracy: Optional[float],
    ) -> ApprovalGate:
        if (
            validation_accuracy is None
            or holdout_accuracy is None
        ):
            return ApprovalGate(
                name="generalization_gap",
                status=GateStatus.NOT_EVALUATED,
                value=None,
                threshold=(
                    self.config
                    .maximum_validation_holdout_gap
                ),
                message=(
                    "Validation and holdout accuracy are required."
                ),
                critical=True,
            )

        gap = abs(
            validation_accuracy
            - holdout_accuracy
        )

        passed = (
            gap
            <= self.config
            .maximum_validation_holdout_gap
        )

        return ApprovalGate(
            name="generalization_gap",
            status=(
                GateStatus.PASS
                if passed
                else GateStatus.FAIL
            ),
            value=float(gap),
            threshold=(
                self.config
                .maximum_validation_holdout_gap
            ),
            message=(
                f"Validation/holdout gap = {gap:.4f}."
            ),
            critical=True,
        )

    # ------------------------------------------------------------------
    # Boolean gate
    # ------------------------------------------------------------------

    @staticmethod
    def _boolean_gate(
        *,
        name: str,
        value: Optional[bool],
        critical: bool,
    ) -> ApprovalGate:
        if value is None:
            return ApprovalGate(
                name=name,
                status=GateStatus.NOT_EVALUATED,
                value=None,
                threshold=None,
                message=(
                    "Boolean gate was not evaluated."
                ),
                critical=critical,
            )

        return ApprovalGate(
            name=name,
            status=(
                GateStatus.PASS
                if value
                else GateStatus.FAIL
            ),
            value=(
                1.0
                if value
                else 0.0
            ),
            threshold=1.0,
            message=(
                "Gate passed."
                if value
                else "Gate failed."
            ),
            critical=critical,
        )

    # ------------------------------------------------------------------
    # Numeric threshold
    # ------------------------------------------------------------------

    @staticmethod
    def _threshold_gate(
        *,
        name: str,
        value: Optional[float],
        threshold: float,
        direction: str,
        critical: bool,
    ) -> ApprovalGate:
        if value is None:
            return ApprovalGate(
                name=name,
                status=GateStatus.NOT_EVALUATED,
                value=None,
                threshold=threshold,
                message=(
                    "Metric was not supplied."
                ),
                critical=critical,
            )

        value = float(value)

        if not np.isfinite(value):
            return ApprovalGate(
                name=name,
                status=GateStatus.FAIL,
                value=value,
                threshold=threshold,
                message=(
                    "Metric is not finite."
                ),
                critical=critical,
            )

        if direction == "min":
            passed = (
                value >= threshold
            )

        elif direction == "max":
            passed = (
                value <= threshold
            )

        elif direction == "max_absolute":
            passed = (
                abs(value)
                <= threshold
            )

        else:
            raise ValueError(
                f"Unknown threshold direction: {direction}"
            )

        return ApprovalGate(
            name=name,
            status=(
                GateStatus.PASS
                if passed
                else GateStatus.FAIL
            ),
            value=value,
            threshold=threshold,
            message=(
                f"{name}: {value:.4f}; "
                f"required {direction} {threshold:.4f}."
            ),
            critical=critical,
        )

    # ------------------------------------------------------------------
    # Range gate
    # ------------------------------------------------------------------

    def _range_gate(
        self,
        *,
        range_coverage: Optional[float],
        range_passed: Optional[bool],
    ) -> ApprovalGate:
        if range_passed is not None:
            return ApprovalGate(
                name="range_validation",
                status=(
                    GateStatus.PASS
                    if range_passed
                    else GateStatus.FAIL
                ),
                value=range_coverage,
                threshold=(
                    self.config
                    .minimum_range_coverage
                ),
                message=(
                    "Range validation passed."
                    if range_passed
                    else "Range validation failed."
                ),
                critical=self.config.require_range_validation,
            )

        if range_coverage is None:
            return ApprovalGate(
                name="range_validation",
                status=GateStatus.NOT_EVALUATED,
                value=None,
                threshold=(
                    self.config
                    .minimum_range_coverage
                ),
                message=(
                    "Range coverage was not supplied."
                ),
                critical=self.config.require_range_validation,
            )

        passed = (
            self.config
            .minimum_range_coverage
            <= range_coverage
            <= self.config
            .maximum_range_coverage
        )

        return ApprovalGate(
            name="range_validation",
            status=(
                GateStatus.PASS
                if passed
                else GateStatus.FAIL
            ),
            value=float(
                range_coverage
            ),
            threshold=(
                self.config
                .minimum_range_coverage
            ),
            message=(
                f"Range coverage = {range_coverage:.4f}; "
                f"accepted interval is "
                f"{self.config.minimum_range_coverage:.2f} "
                f"to "
                f"{self.config.maximum_range_coverage:.2f}."
            ),
            critical=self.config.require_range_validation,
        )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def _build_report(
        self,
        *,
        gates: Sequence[
            ApprovalGate
        ],
        model_id: Optional[str],
        experiment_id: Optional[str],
    ) -> ApprovalReport:
        gates = list(gates)

        passed = [
            gate
            for gate in gates
            if gate.status
            == GateStatus.PASS
        ]

        failed = [
            gate
            for gate in gates
            if gate.status
            == GateStatus.FAIL
        ]

        warnings = [
            gate
            for gate in gates
            if gate.status
            == GateStatus.WARNING
        ]

        not_evaluated = [
            gate
            for gate in gates
            if gate.status
            == GateStatus.NOT_EVALUATED
        ]

        critical_failures = [
            gate.name
            for gate in gates
            if gate.critical
            and gate.status
            in (
                GateStatus.FAIL,
                GateStatus.NOT_EVALUATED,
            )
        ]

        warning_messages = [
            gate.message
            for gate in warnings
        ]

        # --------------------------------------------------------------
        # Score
        # --------------------------------------------------------------

        evaluated = [
            gate
            for gate in gates
            if gate.status
            in (
                GateStatus.PASS,
                GateStatus.FAIL,
            )
        ]

        if evaluated:
            score = float(
                len(passed)
                / len(evaluated)
            )
        else:
            score = 0.0

        # --------------------------------------------------------------
        # Status
        # --------------------------------------------------------------

        if critical_failures:
            status = (
                ApprovalStatus.REJECTED
            )

        elif not_evaluated:
            status = (
                ApprovalStatus.HOLD
            )

        elif warnings and not (
            self.config.allow_warnings
        ):
            status = (
                ApprovalStatus.REJECTED
            )

        else:
            status = (
                ApprovalStatus.APPROVED
            )

        notes = [
            (
                "Approval requires evidence across multiple "
                "independent research gates."
            ),
            (
                "A high accuracy value alone cannot approve a model."
            ),
            (
                "The final holdout must remain untouched until "
                "the research configuration is frozen."
            ),
        ]

        if status == ApprovalStatus.APPROVED:
            notes.append(
                "Model passed all required production gates."
            )

        elif status == ApprovalStatus.HOLD:
            notes.append(
                "Model is on HOLD because required evidence "
                "has not yet been evaluated."
            )

        else:
            notes.append(
                "Model is not approved for production use."
            )

        return ApprovalReport(
            status=status,
            gates=gates,
            critical_failures=(
                critical_failures
            ),
            warnings=warning_messages,
            passed_gates=len(passed),
            failed_gates=len(failed),
            warning_gates=len(warnings),
            not_evaluated_gates=len(
                not_evaluated
            ),
            score=score,
            model_id=model_id,
            experiment_id=experiment_id,
            notes=notes,
        )


# ----------------------------------------------------------------------
# Convenience functions
# ----------------------------------------------------------------------


def evaluate_production_readiness(
    *,
    validation_accuracy: Optional[float] = None,
    final_holdout_accuracy: Optional[float] = None,
    leakage_free: Optional[bool] = None,
    calibration_brier: Optional[float] = None,
    calibration_ece: Optional[float] = None,
    range_coverage: Optional[float] = None,
    regime_stability: Optional[float] = None,
    backtest_profit_factor: Optional[float] = None,
    backtest_sharpe: Optional[float] = None,
    backtest_max_drawdown: Optional[float] = None,
    backtest_trade_count: Optional[int] = None,
    robustness_score: Optional[float] = None,
    validation_passed: Optional[bool] = None,
    calibration_passed: Optional[bool] = None,
    range_passed: Optional[bool] = None,
    regime_passed: Optional[bool] = None,
    backtest_passed: Optional[bool] = None,
    robustness_passed: Optional[bool] = None,
    model_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    config: Optional[
        ApprovalConfig
    ] = None,
) -> ApprovalReport:
    """
    Convenience wrapper for production-readiness evaluation.
    """

    engine = ProductionApprovalEngine(
        config=config
    )

    return engine.evaluate(
        validation_accuracy=(
            validation_accuracy
        ),
        final_holdout_accuracy=(
            final_holdout_accuracy
        ),
        leakage_free=leakage_free,
        calibration_brier=(
            calibration_brier
        ),
        calibration_ece=calibration_ece,
        range_coverage=range_coverage,
        regime_stability=regime_stability,
        backtest_profit_factor=(
            backtest_profit_factor
        ),
        backtest_sharpe=(
            backtest_sharpe
        ),
        backtest_max_drawdown=(
            backtest_max_drawdown
        ),
        backtest_trade_count=(
            backtest_trade_count
        ),
        robustness_score=(
            robustness_score
        ),
        validation_passed=(
            validation_passed
        ),
        calibration_passed=(
            calibration_passed
        ),
        range_passed=range_passed,
        regime_passed=regime_passed,
        backtest_passed=backtest_passed,
        robustness_passed=robustness_passed,
        model_id=model_id,
        experiment_id=experiment_id,
    )


def approval_summary(
    report: ApprovalReport,
) -> Dict[str, Any]:
    """Return a compact approval summary."""

    return report.summary()


__all__ = [
    "ApprovalStatus",
    "GateStatus",
    "ApprovalGate",
    "ApprovalConfig",
    "ApprovalReport",
    "ProductionApprovalEngine",
    "evaluate_production_readiness",
    "approval_summary",
]
