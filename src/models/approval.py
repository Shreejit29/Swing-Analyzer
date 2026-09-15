"""
Final production-readiness approval engine.

Purpose
-------
Determine whether a stock-analysis model has passed all required
research and robustness gates before it can be used for live inference.

Approval philosophy
-------------------
A financial ML model should NOT be approved merely because its accuracy
is high.

Approval should consider evidence across:

    1. Validation
    2. Final holdout
    3. Generalization gap
    4. Leakage checks
    5. Probability calibration
    6. Target-range validation
    7. Regime stability
    8. Trading backtest
    9. Robustness
   10. Economic edge

The final holdout must remain untouched until the research configuration
has been frozen.

A failed critical gate results in REJECTED status.

Missing required evidence results in HOLD.

Warnings are visible but do not automatically reject a model.

This module does not train models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Sequence

import numpy as np


# ----------------------------------------------------------------------
# Status
# ----------------------------------------------------------------------


class ApprovalStatus(str, Enum):
    """Final model lifecycle state."""

    APPROVED = "APPROVED"
    RESEARCH = "RESEARCH"
    REJECTED = "REJECTED"
    HOLD = "HOLD"


class GateStatus(str, Enum):
    """Individual approval-gate status."""

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
    Production approval thresholds.

    These are starting research thresholds, not guarantees of future
    profitability.

    Accuracy is retained for backward compatibility, but should not be
    treated as the primary measure of a financial model.

    The research pipeline should increasingly emphasize:

        balanced accuracy
        ROC-AUC
        PR-AUC
        log loss
        Brier score
        calibration
        walk-forward performance
        trading expectancy
        profit factor
        drawdown
        robustness
        economic edge
    """

    # --------------------------------------------------------------
    # Classification validation
    # --------------------------------------------------------------

    minimum_validation_accuracy: float = 0.60

    minimum_final_holdout_accuracy: float = 0.60

    maximum_validation_holdout_gap: float = 0.10

    # Optional stronger metrics.
    minimum_validation_balanced_accuracy: Optional[
        float
    ] = None

    minimum_final_holdout_balanced_accuracy: Optional[
        float
    ] = None

    minimum_validation_roc_auc: Optional[
        float
    ] = None

    minimum_final_holdout_roc_auc: Optional[
        float
    ] = None

    minimum_validation_pr_auc: Optional[
        float
    ] = None

    minimum_final_holdout_pr_auc: Optional[
        float
    ] = None

    # --------------------------------------------------------------
    # Calibration
    # --------------------------------------------------------------

    maximum_calibration_brier: float = 0.25

    maximum_calibration_ece: float = 0.15

    # --------------------------------------------------------------
    # Range prediction
    # --------------------------------------------------------------

    minimum_range_coverage: float = 0.70

    maximum_range_coverage: float = 0.95

    # --------------------------------------------------------------
    # Regime
    # --------------------------------------------------------------

    minimum_regime_stability: float = 0.60

    # --------------------------------------------------------------
    # Backtest
    # --------------------------------------------------------------

    minimum_backtest_profit_factor: float = 1.20

    minimum_backtest_sharpe: float = 0.80

    maximum_backtest_drawdown: float = 0.30

    minimum_backtest_trades: int = 30

    # --------------------------------------------------------------
    # Robustness
    # --------------------------------------------------------------

    minimum_robustness_score: float = 0.60

    # --------------------------------------------------------------
    # Economic edge
    # --------------------------------------------------------------

    minimum_expected_edge: float = 0.003

    require_expected_edge: bool = False

    # --------------------------------------------------------------
    # Required gates
    # --------------------------------------------------------------

    require_validation: bool = True

    require_final_holdout: bool = True

    require_no_leakage: bool = True

    require_calibration: bool = True

    require_range_validation: bool = True

    require_regime_validation: bool = True

    require_backtest: bool = True

    require_robustness: bool = True

    # --------------------------------------------------------------
    # General
    # --------------------------------------------------------------

    allow_warnings: bool = True

    def __post_init__(self) -> None:
        bounded_values = [
            self.minimum_validation_accuracy,
            self.minimum_final_holdout_accuracy,
            self.maximum_calibration_brier,
            self.maximum_calibration_ece,
            self.minimum_range_coverage,
            self.maximum_range_coverage,
            self.minimum_regime_stability,
            self.maximum_backtest_drawdown,
            self.minimum_robustness_score,
        ]

        optional_bounded_values = [
            self.minimum_validation_balanced_accuracy,
            self.minimum_final_holdout_balanced_accuracy,
            self.minimum_validation_roc_auc,
            self.minimum_final_holdout_roc_auc,
            self.minimum_validation_pr_auc,
            self.minimum_final_holdout_pr_auc,
        ]

        for value in bounded_values:

            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    "Probability/ratio thresholds must be between 0 and 1."
                )

        for value in optional_bounded_values:

            if value is not None:

                if not 0.0 <= float(value) <= 1.0:
                    raise ValueError(
                        "Optional metric thresholds must be between 0 and 1."
                    )

        if (
            self.minimum_range_coverage
            > self.maximum_range_coverage
        ):
            raise ValueError(
                "minimum_range_coverage cannot exceed "
                "maximum_range_coverage."
            )

        if self.maximum_validation_holdout_gap < 0:

            raise ValueError(
                "maximum_validation_holdout_gap cannot be negative."
            )

        if self.minimum_backtest_profit_factor < 0:

            raise ValueError(
                "minimum_backtest_profit_factor cannot be negative."
            )

        if self.minimum_backtest_sharpe < 0:

            raise ValueError(
                "minimum_backtest_sharpe cannot be negative."
            )

        if self.minimum_backtest_trades < 1:

            raise ValueError(
                "minimum_backtest_trades must be positive."
            )

        if self.minimum_expected_edge < 0:

            raise ValueError(
                "minimum_expected_edge cannot be negative."
            )


# ----------------------------------------------------------------------
# Approval report
# ----------------------------------------------------------------------


@dataclass
class ApprovalReport:
    """Complete production approval report."""

    status: ApprovalStatus

    gates: list[ApprovalGate]

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
        """Return a compact serializable summary."""

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
            "critical_failures": list(
                self.critical_failures
            ),
            "warnings": list(
                self.warnings
            ),
            "model_id": self.model_id,
            "experiment_id": self.experiment_id,
        }


# ----------------------------------------------------------------------
# Production approval engine
# ----------------------------------------------------------------------


class ProductionApprovalEngine:
    """
    Final production approval engine.

    Every gate is evaluated independently.

    Critical gates cannot be bypassed by a high aggregate score.

    This class never trains a model.
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
        validation_balanced_accuracy: Optional[float] = None,
        final_holdout_balanced_accuracy: Optional[float] = None,
        validation_roc_auc: Optional[float] = None,
        final_holdout_roc_auc: Optional[float] = None,
        validation_pr_auc: Optional[float] = None,
        final_holdout_pr_auc: Optional[float] = None,
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
        expected_edge: Optional[float] = None,
        validation_passed: Optional[bool] = None,
        calibration_passed: Optional[bool] = None,
        range_passed: Optional[bool] = None,
        regime_passed: Optional[bool] = None,
        backtest_passed: Optional[bool] = None,
        robustness_passed: Optional[bool] = None,
        model_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
    ) -> ApprovalReport:
        """
        Evaluate all production-readiness gates.

        Parameters are optional because research can happen incrementally.
        Missing required evidence produces HOLD/REJECTED rather than
        pretending that the model passed.
        """

        gates: list[ApprovalGate] = []

        # --------------------------------------------------------------
        # 1. Validation accuracy
        # --------------------------------------------------------------

        gates.append(
            self._accuracy_gate(
                name="validation_accuracy",
                value=validation_accuracy,
                threshold=(
                    self.config.minimum_validation_accuracy
                ),
                passed_override=validation_passed,
                critical=self.config.require_validation,
            )
        )

        # --------------------------------------------------------------
        # 2. Final holdout accuracy
        # --------------------------------------------------------------

        gates.append(
            self._accuracy_gate(
                name="final_holdout_accuracy",
                value=final_holdout_accuracy,
                threshold=(
                    self.config.minimum_final_holdout_accuracy
                ),
                critical=self.config.require_final_holdout,
            )
        )

        # --------------------------------------------------------------
        # 3. Generalization gap
        # --------------------------------------------------------------

        gates.append(
            self._generalization_gate(
                validation_accuracy=validation_accuracy,
                holdout_accuracy=final_holdout_accuracy,
            )
        )

        # --------------------------------------------------------------
        # 4. Balanced accuracy
        # --------------------------------------------------------------

        if (
            self.config
            .minimum_validation_balanced_accuracy
            is not None
        ):

            gates.append(
                self._threshold_gate(
                    name="validation_balanced_accuracy",
                    value=validation_balanced_accuracy,
                    threshold=(
                        self.config
                        .minimum_validation_balanced_accuracy
                    ),
                    direction="min",
                    critical=self.config.require_validation,
                )
            )

        if (
            self.config
            .minimum_final_holdout_balanced_accuracy
            is not None
        ):

            gates.append(
                self._threshold_gate(
                    name="final_holdout_balanced_accuracy",
                    value=final_holdout_balanced_accuracy,
                    threshold=(
                        self.config
                        .minimum_final_holdout_balanced_accuracy
                    ),
                    direction="min",
                    critical=self.config.require_final_holdout,
                )
            )

        # --------------------------------------------------------------
        # 5. ROC-AUC
        # --------------------------------------------------------------

        if (
            self.config.minimum_validation_roc_auc
            is not None
        ):

            gates.append(
                self._threshold_gate(
                    name="validation_roc_auc",
                    value=validation_roc_auc,
                    threshold=(
                        self.config
                        .minimum_validation_roc_auc
                    ),
                    direction="min",
                    critical=self.config.require_validation,
                )
            )

        if (
            self.config.minimum_final_holdout_roc_auc
            is not None
        ):

            gates.append(
                self._threshold_gate(
                    name="final_holdout_roc_auc",
                    value=final_holdout_roc_auc,
                    threshold=(
                        self.config
                        .minimum_final_holdout_roc_auc
                    ),
                    direction="min",
                    critical=self.config.require_final_holdout,
                )
            )

        # --------------------------------------------------------------
        # 6. PR-AUC
        # --------------------------------------------------------------

        if (
            self.config.minimum_validation_pr_auc
            is not None
        ):

            gates.append(
                self._threshold_gate(
                    name="validation_pr_auc",
                    value=validation_pr_auc,
                    threshold=(
                        self.config
                        .minimum_validation_pr_auc
                    ),
                    direction="min",
                    critical=self.config.require_validation,
                )
            )

        if (
            self.config.minimum_final_holdout_pr_auc
            is not None
        ):

            gates.append(
                self._threshold_gate(
                    name="final_holdout_pr_auc",
                    value=final_holdout_pr_auc,
                    threshold=(
                        self.config
                        .minimum_final_holdout_pr_auc
                    ),
                    direction="min",
                    critical=self.config.require_final_holdout,
                )
            )

        # --------------------------------------------------------------
        # 7. Leakage
        # --------------------------------------------------------------

        gates.append(
            self._boolean_gate(
                name="no_data_leakage",
                value=leakage_free,
                critical=self.config.require_no_leakage,
            )
        )

        # --------------------------------------------------------------
        # 8. Calibration
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
        # 9. Range validation
        # --------------------------------------------------------------

        gates.append(
            self._range_gate(
                range_coverage=range_coverage,
                range_passed=range_passed,
            )
        )

        # --------------------------------------------------------------
        # 10. Regime stability
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
        # 11. Backtest profit factor
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

        # --------------------------------------------------------------
        # 12. Backtest Sharpe
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # 13. Backtest drawdown
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # 14. Number of trades
        # --------------------------------------------------------------

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
        # 15. Robustness
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
        # 16. Economic edge
        # --------------------------------------------------------------

        gates.append(
            self._edge_gate(
                expected_edge=expected_edge
            )
        )

        # --------------------------------------------------------------
        # Final report
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
        """
        Evaluate an accuracy metric.

        An explicit pipeline result takes precedence over the numeric
        value because the pipeline may incorporate additional validation
        logic.
        """

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
                    "Validation pipeline gate passed."
                    if passed_override
                    else "Validation pipeline gate failed."
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
                    f"{name} was not supplied."
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
                    f"{name} is not finite."
                ),
                critical=critical,
            )

        passed = value >= threshold

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
                f"{name} = {value:.4f}; "
                f"required >= {threshold:.4f}."
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
        """
        Check validation-to-final-holdout degradation.

        This prevents a model from appearing strong during research but
        collapsing on the untouched holdout.
        """

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
                    "Both validation and final holdout "
                    "accuracy are required."
                ),
                critical=True,
            )

        validation_accuracy = float(
            validation_accuracy
        )

        holdout_accuracy = float(
            holdout_accuracy
        )

        if not np.isfinite(
            validation_accuracy
        ) or not np.isfinite(
            holdout_accuracy
        ):

            return ApprovalGate(
                name="generalization_gap",
                status=GateStatus.FAIL,
                value=None,
                threshold=(
                    self.config
                    .maximum_validation_holdout_gap
                ),
                message=(
                    "Validation or holdout accuracy is not finite."
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
                f"Validation/holdout accuracy gap = "
                f"{gap:.4f}; maximum allowed = "
                f"{self.config.maximum_validation_holdout_gap:.4f}."
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
        """
        Evaluate a boolean research gate.

        None means the gate has not been evaluated.
        """

        if value is None:

            return ApprovalGate(
                name=name,
                status=GateStatus.NOT_EVALUATED,
                value=None,
                threshold=None,
                message=(
                    f"{name} was not evaluated."
                ),
                critical=critical,
            )

        passed = bool(value)

        return ApprovalGate(
            name=name,
            status=(
                GateStatus.PASS
                if passed
                else GateStatus.FAIL
            ),
            value=(
                1.0
                if passed
                else 0.0
            ),
            threshold=1.0,
            message=(
                f"{name} passed."
                if passed
                else f"{name} failed."
            ),
            critical=critical,
        )

    # ------------------------------------------------------------------
    # Numeric threshold gate
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
        """
        Evaluate a numeric threshold.

        direction:

            min
                value >= threshold

            max
                value <= threshold

            max_absolute
                abs(value) <= threshold
        """

        if value is None:

            return ApprovalGate(
                name=name,
                status=GateStatus.NOT_EVALUATED,
                value=None,
                threshold=threshold,
                message=(
                    f"{name} was not supplied."
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
                    f"{name} is not finite."
                ),
                critical=critical,
            )

        threshold = float(threshold)

        if direction == "min":

            passed = value >= threshold

        elif direction == "max":

            passed = value <= threshold

        elif direction == "max_absolute":

            passed = abs(value) <= threshold

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
                f"{name} = {value:.4f}; "
                f"required {direction} {threshold:.4f}."
            ),
            critical=critical,
        )

    # ------------------------------------------------------------------
    # Range validation
    # ------------------------------------------------------------------

    def _range_gate(
        self,
        *,
        range_coverage: Optional[float],
        range_passed: Optional[bool],
    ) -> ApprovalGate:
        """
        Evaluate prediction interval coverage.

        Coverage that is unrealistically close to 100% is also rejected
        by the upper bound because a uselessly wide interval can have
        excellent coverage without providing useful information.
        """

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
                    "Range validation pipeline passed."
                    if range_passed
                    else "Range validation pipeline failed."
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

        coverage = float(
            range_coverage
        )

        if not np.isfinite(coverage):

            return ApprovalGate(
                name="range_validation",
                status=GateStatus.FAIL,
                value=coverage,
                threshold=(
                    self.config
                    .minimum_range_coverage
                ),
                message=(
                    "Range coverage is not finite."
                ),
                critical=self.config.require_range_validation,
            )

        passed = (
            self.config.minimum_range_coverage
            <= coverage
            <= self.config.maximum_range_coverage
        )

        return ApprovalGate(
            name="range_validation",
            status=(
                GateStatus.PASS
                if passed
                else GateStatus.FAIL
            ),
            value=coverage,
            threshold=(
                self.config
                .minimum_range_coverage
            ),
            message=(
                f"Range coverage = {coverage:.2%}; "
                f"accepted interval = "
                f"{self.config.minimum_range_coverage:.2%} "
                f"to "
                f"{self.config.maximum_range_coverage:.2%}."
            ),
            critical=self.config.require_range_validation,
        )

    # ------------------------------------------------------------------
    # Economic edge
    # ------------------------------------------------------------------

    def _edge_gate(
        self,
        *,
        expected_edge: Optional[float],
    ) -> ApprovalGate:
        """
        Evaluate expected economic edge.

        expected_edge is a decimal return:

            0.010 = +1.0%
            0.025 = +2.5%
            -0.005 = -0.5%

        The gate is optional at this stage.

        This is intentional: we do not want to fabricate expected
        returns before the return-prediction layer has been properly
        implemented and validated.
        """

        if expected_edge is None:

            if self.config.require_expected_edge:

                return ApprovalGate(
                    name="expected_edge",
                    status=GateStatus.FAIL,
                    value=None,
                    threshold=(
                        self.config
                        .minimum_expected_edge
                    ),
                    message=(
                        "Expected economic edge is required "
                        "but was not supplied."
                    ),
                    critical=True,
                )

            return ApprovalGate(
                name="expected_edge",
                status=GateStatus.NOT_EVALUATED,
                value=None,
                threshold=(
                    self.config
                    .minimum_expected_edge
                ),
                message=(
                    "Expected economic edge is not available; "
                    "economic edge gate is currently optional."
                ),
                critical=False,
            )

        edge = float(
            expected_edge
        )

        if not np.isfinite(edge):

            return ApprovalGate(
                name="expected_edge",
                status=GateStatus.FAIL,
                value=edge,
                threshold=(
                    self.config
                    .minimum_expected_edge
                ),
                message=(
                    "Expected economic edge is not finite."
                ),
                critical=self.config.require_expected_edge,
            )

        threshold = float(
            self.config.minimum_expected_edge
        )

        passed = edge >= threshold

        score = (
            max(
                0.0,
                min(
                    1.0,
                    edge / threshold,
                ),
            )
            if threshold > 0
            else (
                1.0
                if edge >= 0
                else 0.0
            )
        )

        return ApprovalGate(
            name="expected_edge",
            status=(
                GateStatus.PASS
                if passed
                else GateStatus.FAIL
            ),
            value=edge,
            threshold=threshold,
            message=(
                f"Expected net edge = {edge:.2%}; "
                f"required >= {threshold:.2%}."
            ),
            critical=self.config.require_expected_edge,
            evidence={
                "edge_score": score,
            },
        )

    # ------------------------------------------------------------------
    # Report generation
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
        """
        Aggregate all gates into the final lifecycle status.
        """

        gates = list(
            gates
        )

        passed = [
            gate
            for gate in gates
            if gate.status == GateStatus.PASS
        ]

        failed = [
            gate
            for gate in gates
            if gate.status == GateStatus.FAIL
        ]

        warnings = [
            gate
            for gate in gates
            if gate.status == GateStatus.WARNING
        ]

        not_evaluated = [
            gate
            for gate in gates
            if gate.status
            == GateStatus.NOT_EVALUATED
        ]

        # --------------------------------------------------------------
        # Critical failures
        # --------------------------------------------------------------

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
        # Aggregate score
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
        # Final lifecycle status
        # --------------------------------------------------------------

        if critical_failures:

            status = ApprovalStatus.REJECTED

        elif not_evaluated:

            status = ApprovalStatus.HOLD

        elif (
            warnings
            and not self.config.allow_warnings
        ):

            status = ApprovalStatus.REJECTED

        else:

            status = ApprovalStatus.APPROVED

        # --------------------------------------------------------------
        # Notes
        # --------------------------------------------------------------

        notes = [
            (
                "Production approval requires evidence across "
                "multiple independent research gates."
            ),
            (
                "Classification accuracy alone is not sufficient "
                "evidence of financial-model quality."
            ),
            (
                "Final holdout data must remain untouched until "
                "the research configuration is frozen."
            ),
            (
                "Economic performance should ultimately be judged "
                "using leakage-free walk-forward trading results."
            ),
        ]

        if status == ApprovalStatus.APPROVED:

            notes.append(
                "Model passed all required production gates."
            )

        elif status == ApprovalStatus.HOLD:

            notes.append(
                "Model remains on HOLD because required evidence "
                "has not yet been evaluated."
            )

        elif status == ApprovalStatus.REJECTED:

            notes.append(
                "Model is not approved for production use."
            )

        return ApprovalReport(
            status=status,
            gates=gates,
            critical_failures=critical_failures,
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
    validation_balanced_accuracy: Optional[float] = None,
    final_holdout_balanced_accuracy: Optional[float] = None,
    validation_roc_auc: Optional[float] = None,
    final_holdout_roc_auc: Optional[float] = None,
    validation_pr_auc: Optional[float] = None,
    final_holdout_pr_auc: Optional[float] = None,
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
    expected_edge: Optional[float] = None,
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
        validation_balanced_accuracy=(
            validation_balanced_accuracy
        ),
        final_holdout_balanced_accuracy=(
            final_holdout_balanced_accuracy
        ),
        validation_roc_auc=(
            validation_roc_auc
        ),
        final_holdout_roc_auc=(
            final_holdout_roc_auc
        ),
        validation_pr_auc=(
            validation_pr_auc
        ),
        final_holdout_pr_auc=(
            final_holdout_pr_auc
        ),
        leakage_free=leakage_free,
        calibration_brier=(
            calibration_brier
        ),
        calibration_ece=(
            calibration_ece
        ),
        range_coverage=(
            range_coverage
        ),
        regime_stability=(
            regime_stability
        ),
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
        expected_edge=(
            expected_edge
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
    """
    Return a compact approval summary.
    """

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
