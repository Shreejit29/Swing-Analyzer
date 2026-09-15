"""
AI Swing Analyser — Final Holdout Gate.

Protects the final untouched holdout from accidental reuse.

The final holdout is evaluated only after:
    1. candidate experiments are complete,
    2. walk-forward validation is complete,
    3. model selection is frozen.

The holdout must never be used for:
    - feature selection,
    - hyperparameter tuning,
    - model selection,
    - calibration fitting,
    - threshold optimization,
    - trading-rule optimization.

This module does not perform the actual holdout evaluation.
That remains the responsibility of holdout_evaluation.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class HoldoutGateStatus(str, Enum):
    """Status of the final holdout gate."""

    LOCKED = "LOCKED"
    READY = "READY"
    EVALUATED = "EVALUATED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class HoldoutGateInput:
    """
    Evidence required before final holdout evaluation.

    All development-stage activities must already be complete.
    """

    model_id: str

    selection_completed: bool
    walk_forward_completed: bool

    feature_selection_frozen: bool
    hyperparameters_frozen: bool
    preprocessing_frozen: bool

    calibration_fitted: bool = False
    threshold_optimization_completed: bool = False
    final_holdout_used_previously: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HoldoutGateResult:
    """Result of final holdout eligibility evaluation."""

    status: HoldoutGateStatus
    model_id: str

    eligible: bool

    selection_completed: bool
    walk_forward_completed: bool
    feature_selection_frozen: bool
    hyperparameters_frozen: bool
    preprocessing_frozen: bool

    calibration_fitted: bool
    threshold_optimization_completed: bool
    final_holdout_used_previously: bool

    failures: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def safe_to_evaluate(self) -> bool:
        """Whether final holdout evaluation may begin."""
        return self.eligible

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "model_id": self.model_id,
            "eligible": self.eligible,
            "safe_to_evaluate": self.safe_to_evaluate,
            "selection_completed": self.selection_completed,
            "walk_forward_completed": self.walk_forward_completed,
            "feature_selection_frozen": self.feature_selection_frozen,
            "hyperparameters_frozen": self.hyperparameters_frozen,
            "preprocessing_frozen": self.preprocessing_frozen,
            "calibration_fitted": self.calibration_fitted,
            "threshold_optimization_completed": (
                self.threshold_optimization_completed
            ),
            "final_holdout_used_previously": (
                self.final_holdout_used_previously
            ),
            "failure_count": len(self.failures),
            "warning_count": len(self.warnings),
        }


class FinalHoldoutGate:
    """
    Enforces the final holdout boundary.

    The gate is intentionally conservative.

    A model cannot reach the final holdout merely because it has high
    validation accuracy. The development process must be frozen first.
    """

    def evaluate(
        self,
        evidence: HoldoutGateInput,
    ) -> HoldoutGateResult:
        """Evaluate whether the final holdout may be opened."""

        if not isinstance(evidence, HoldoutGateInput):
            raise TypeError(
                "evidence must be a HoldoutGateInput."
            )

        if not evidence.model_id.strip():
            raise ValueError(
                "model_id cannot be empty."
            )

        failures: list[str] = []
        warnings: list[str] = []

        if not evidence.selection_completed:
            failures.append(
                "Model selection has not been completed."
            )

        if not evidence.walk_forward_completed:
            failures.append(
                "Walk-forward validation has not been completed."
            )

        if not evidence.feature_selection_frozen:
            failures.append(
                "Feature selection is not frozen."
            )

        if not evidence.hyperparameters_frozen:
            failures.append(
                "Hyperparameters are not frozen."
            )

        if not evidence.preprocessing_frozen:
            failures.append(
                "Preprocessing is not frozen."
            )

        if evidence.calibration_fitted:
            failures.append(
                "Calibration has already been fitted before "
                "final holdout evaluation."
            )

        if evidence.threshold_optimization_completed:
            failures.append(
                "Prediction/trading thresholds were optimized "
                "using information that may contaminate the holdout."
            )

        if evidence.final_holdout_used_previously:
            failures.append(
                "The final holdout has already been used."
            )

        if failures:
            status = HoldoutGateStatus.BLOCKED
            eligible = False
        else:
            status = HoldoutGateStatus.READY
            eligible = True

            warnings.append(
                "Once final holdout evaluation begins, the holdout "
                "must remain excluded from all model-development decisions."
            )

        metadata = dict(evidence.metadata)

        metadata.update(
            {
                "final_holdout_protected": True,
                "holdout_used_for_selection": False,
                "holdout_used_for_feature_selection": False,
                "holdout_used_for_hyperparameter_tuning": False,
                "holdout_used_for_calibration": False,
                "research_only": True,
            }
        )

        return HoldoutGateResult(
            status=status,
            model_id=evidence.model_id,
            eligible=eligible,
            selection_completed=evidence.selection_completed,
            walk_forward_completed=evidence.walk_forward_completed,
            feature_selection_frozen=evidence.feature_selection_frozen,
            hyperparameters_frozen=evidence.hyperparameters_frozen,
            preprocessing_frozen=evidence.preprocessing_frozen,
            calibration_fitted=evidence.calibration_fitted,
            threshold_optimization_completed=(
                evidence.threshold_optimization_completed
            ),
            final_holdout_used_previously=(
                evidence.final_holdout_used_previously
            ),
            failures=tuple(failures),
            warnings=tuple(warnings),
            metadata=metadata,
        )

    def assert_ready(
        self,
        evidence: HoldoutGateInput,
    ) -> HoldoutGateResult:
        """
        Evaluate and raise if the final holdout is not protected.
        """

        result = self.evaluate(evidence)

        if not result.eligible:
            message = "Final holdout gate blocked evaluation."

            if result.failures:
                message += " " + " ".join(result.failures)

            raise RuntimeError(message)

        return result


def create_holdout_gate(
    model_id: str,
    selection_completed: bool,
    walk_forward_completed: bool,
    feature_selection_frozen: bool,
    hyperparameters_frozen: bool,
    preprocessing_frozen: bool,
    calibration_fitted: bool = False,
    threshold_optimization_completed: bool = False,
    final_holdout_used_previously: bool = False,
    metadata: Optional[dict[str, Any]] = None,
) -> HoldoutGateInput:
    """Convenience constructor for holdout-gate evidence."""

    return HoldoutGateInput(
        model_id=model_id,
        selection_completed=selection_completed,
        walk_forward_completed=walk_forward_completed,
        feature_selection_frozen=feature_selection_frozen,
        hyperparameters_frozen=hyperparameters_frozen,
        preprocessing_frozen=preprocessing_frozen,
        calibration_fitted=calibration_fitted,
        threshold_optimization_completed=(
            threshold_optimization_completed
        ),
        final_holdout_used_previously=(
            final_holdout_used_previously
        ),
        metadata=dict(metadata or {}),
    )


def evaluate_holdout_gate(
    evidence: HoldoutGateInput,
) -> HoldoutGateResult:
    """Convenience API for final holdout protection."""

    return FinalHoldoutGate().evaluate(evidence)


__all__ = [
    "HoldoutGateStatus",
    "HoldoutGateInput",
    "HoldoutGateResult",
    "FinalHoldoutGate",
    "create_holdout_gate",
    "evaluate_holdout_gate",
]
