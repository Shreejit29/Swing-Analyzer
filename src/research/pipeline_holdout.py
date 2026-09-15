"""
AI Swing Analyser — Protected Final Holdout Pipeline.

Execution boundary:

    Model Selection
          ↓
    Freeze Model / Features / Hyperparameters / Preprocessing
          ↓
    Final Holdout Gate
          ↓
    Final Holdout Evaluation
          ↓
    Holdout Evidence
          ↓
    Production Approval Engine

This module does NOT:
    - train a model
    - select features
    - tune hyperparameters
    - fit calibration
    - optimize thresholds
    - approve production deployment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import pandas as pd

from .holdout_evaluation import HoldoutEvaluationResult
from .holdout_gate import (
    FinalHoldoutGate,
    HoldoutGateInput,
    HoldoutGateResult,
    HoldoutGateStatus,
)
from .holdout_stage import (
    HoldoutStageResult,
    run_final_holdout_stage,
)


@dataclass(frozen=True)
class HoldoutPipelineResult:
    """Result of the protected final holdout pipeline."""

    model_id: str
    stage: HoldoutStageResult

    completed: bool
    production_approved: bool

    accuracy: float | None
    passed_accuracy_gate: bool

    errors: tuple[str, ...] = field(
        default_factory=tuple
    )

    warnings: tuple[str, ...] = field(
        default_factory=tuple
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def eligible(self) -> bool:
        """Whether the model was permitted to use the holdout."""
        return self.stage.eligible_for_evaluation

    @property
    def evaluated(self) -> bool:
        """Whether final holdout evaluation actually occurred."""
        return self.stage.evaluated

    @property
    def final_holdout_used(self) -> bool:
        """Whether the final holdout was consumed."""
        return self.stage.final_holdout_used

    @property
    def holdout_result(
        self,
    ) -> HoldoutEvaluationResult | None:
        """Return the underlying holdout evaluation."""
        return self.stage.evaluation

    def summary(self) -> dict[str, Any]:
        """Return a compact, serializable summary."""

        return {
            "model_id": self.model_id,
            "completed": self.completed,
            "eligible": self.eligible,
            "evaluated": self.evaluated,
            "final_holdout_used": (
                self.final_holdout_used
            ),
            "accuracy": self.accuracy,
            "passed_accuracy_gate": (
                self.passed_accuracy_gate
            ),
            "production_approved": (
                self.production_approved
            ),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


class ProtectedHoldoutPipeline:
    """
    Controlled wrapper around final holdout evaluation.

    The pipeline receives an already-frozen model and preprocessor.
    It never fits either object.
    """

    def __init__(
        self,
        accuracy_threshold: float = 0.95,
    ) -> None:
        if not 0.0 <= accuracy_threshold <= 1.0:
            raise ValueError(
                "accuracy_threshold must be between 0 and 1."
            )

        self.accuracy_threshold = (
            float(accuracy_threshold)
        )

        self.gate = FinalHoldoutGate()

    @staticmethod
    def _validate_model_id(
        model_id: str,
    ) -> None:
        if not isinstance(model_id, str):
            raise TypeError(
                "model_id must be a string."
            )

        if not model_id.strip():
            raise ValueError(
                "model_id cannot be empty."
            )

    @staticmethod
    def _validate_gate(
        gate_input: HoldoutGateInput,
    ) -> None:
        if not isinstance(
            gate_input,
            HoldoutGateInput,
        ):
            raise TypeError(
                "gate_input must be a HoldoutGateInput."
            )

    @staticmethod
    def _validate_model_identity(
        model_id: str,
        gate_input: HoldoutGateInput,
    ) -> None:
        if model_id != gate_input.model_id:
            raise ValueError(
                "model_id does not match the holdout gate."
            )

    @staticmethod
    def _validate_data(
        development_data: pd.DataFrame,
        holdout_data: pd.DataFrame,
    ) -> None:
        for data, name in (
            (
                development_data,
                "development_data",
            ),
            (
                holdout_data,
                "holdout_data",
            ),
        ):
            if not isinstance(
                data,
                pd.DataFrame,
            ):
                raise TypeError(
                    f"{name} must be a pandas DataFrame."
                )

            if data.empty:
                raise ValueError(
                    f"{name} cannot be empty."
                )

            if not isinstance(
                data.index,
                pd.DatetimeIndex,
            ):
                raise TypeError(
                    f"{name} must use a DatetimeIndex."
                )

            if data.index.has_duplicates:
                raise ValueError(
                    f"{name} contains duplicate timestamps."
                )

            if not data.index.is_monotonic_increasing:
                raise ValueError(
                    f"{name} must be chronologically sorted."
                )

        if (
            development_data.index.max()
            >= holdout_data.index.min()
        ):
            raise ValueError(
                "Development and final holdout periods overlap."
            )

    @staticmethod
    def _validate_features(
        development_data: pd.DataFrame,
        holdout_data: pd.DataFrame,
        feature_columns: Sequence[str],
    ) -> None:
        if not feature_columns:
            raise ValueError(
                "feature_columns cannot be empty."
            )

        feature_columns = list(
            feature_columns
        )

        if len(feature_columns) != len(
            set(feature_columns)
        ):
            raise ValueError(
                "feature_columns contains duplicates."
            )

        missing_development = [
            column
            for column in feature_columns
            if column not in development_data.columns
        ]

        if missing_development:
            raise ValueError(
                "Features missing from development data: "
                f"{missing_development}"
            )

        missing_holdout = [
            column
            for column in feature_columns
            if column not in holdout_data.columns
        ]

        if missing_holdout:
            raise ValueError(
                "Features missing from holdout data: "
                f"{missing_holdout}"
            )

    @staticmethod
    def _validate_target(
        development_data: pd.DataFrame,
        holdout_data: pd.DataFrame,
        target_column: str,
    ) -> None:
        if not isinstance(
            target_column,
            str,
        ):
            raise TypeError(
                "target_column must be a string."
            )

        if not target_column.strip():
            raise ValueError(
                "target_column cannot be empty."
            )

        if target_column not in development_data.columns:
            raise ValueError(
                "Target missing from development data: "
                f"{target_column}"
            )

        if target_column not in holdout_data.columns:
            raise ValueError(
                "Target missing from holdout data: "
                f"{target_column}"
            )

    @staticmethod
    def _validate_model(
        model: Any,
    ) -> None:
        if model is None:
            raise ValueError(
                "Frozen model cannot be None."
            )

        if not hasattr(
            model,
            "predict",
        ):
            raise TypeError(
                "Frozen model must provide predict()."
            )

    @staticmethod
    def _validate_preprocessor(
        preprocessor: Any,
    ) -> None:
        if preprocessor is None:
            raise ValueError(
                "Frozen preprocessor cannot be None."
            )

        if not hasattr(
            preprocessor,
            "transform",
        ):
            raise TypeError(
                "Frozen preprocessor must provide transform()."
            )

    @staticmethod
    def _validate_gate_metadata(
        gate_input: HoldoutGateInput,
    ) -> None:
        metadata = gate_input.metadata or {}

        forbidden_flags = {
            "holdout_used_for_selection",
            "holdout_used_for_feature_selection",
            "holdout_used_for_hyperparameter_tuning",
            "holdout_used_for_calibration",
            "holdout_used_for_threshold_optimization",
        }

        violations = [
            key
            for key in forbidden_flags
            if bool(metadata.get(key, False))
        ]

        if violations:
            raise ValueError(
                "Holdout contamination detected: "
                f"{violations}"
            )

    @staticmethod
    def _blocked_result(
        model_id: str,
        gate_result: HoldoutGateResult,
        *,
        errors: tuple[str, ...] = (),
    ) -> HoldoutPipelineResult:
        """
        Build a deterministic fail-closed result.

        No holdout evaluation has occurred.
        """

        stage = HoldoutStageResult(
            model_id=model_id,
            gate=gate_result,
            evaluation=None,
            successful=False,
            final_holdout_used=False,
            holdout_accuracy=None,
            passed_accuracy_gate=False,
            warnings=tuple(
                gate_result.warnings
            ),
            metadata={
                "stage": "protected_final_holdout",
                "evaluation_started": False,
                "final_holdout_used": False,
                "research_only": True,
                "production_approved": False,
            },
        )

        return HoldoutPipelineResult(
            model_id=model_id,
            stage=stage,
            completed=False,
            production_approved=False,
            accuracy=None,
            passed_accuracy_gate=False,
            errors=errors,
            warnings=tuple(
                gate_result.warnings
            ),
            metadata={
                "stage": "protected_final_holdout",
                "evaluation_started": False,
                "final_holdout_used": False,
                "research_only": True,
                "production_approved": False,
                "holdout_gate_status": (
                    gate_result.status.value
                ),
            },
        )

    def run(
        self,
        *,
        model_id: str,
        model: Any,
        preprocessor: Any,
        development_data: pd.DataFrame,
        holdout_data: pd.DataFrame,
        feature_columns: Sequence[str],
        target_column: str,
        gate_input: HoldoutGateInput,
    ) -> HoldoutPipelineResult:
        """
        Run the protected final holdout evaluation.

        No model or preprocessing fitting occurs here.
        """

        self._validate_model_id(
            model_id
        )

        self._validate_gate(
            gate_input
        )

        self._validate_model_identity(
            model_id,
            gate_input,
        )

        self._validate_model(
            model
        )

        self._validate_preprocessor(
            preprocessor
        )

        self._validate_data(
            development_data,
            holdout_data,
        )

        self._validate_features(
            development_data,
            holdout_data,
            feature_columns,
        )

        self._validate_target(
            development_data,
            holdout_data,
            target_column,
        )

        self._validate_gate_metadata(
            gate_input
        )

        gate_result = self.gate.evaluate(
            gate_input
        )

        if not gate_result.eligible:
            return self._blocked_result(
                model_id,
                gate_result,
            )

        try:
            stage_result = (
                run_final_holdout_stage(
                    model_id=model_id,
                    model=model,
                    preprocessor=preprocessor,
                    development_data=development_data,
                    holdout_data=holdout_data,
                    feature_columns=feature_columns,
                    target_column=target_column,
                    gate_input=gate_input,
                    accuracy_threshold=(
                        self.accuracy_threshold
                    ),
                )
            )

        except Exception as exc:
            failure_gate = HoldoutGateResult(
                status=HoldoutGateStatus.BLOCKED,
                eligible=False,
                safe_to_evaluate=False,
                failures=(
                    "Final holdout pipeline execution failed.",
                ),
                warnings=(
                    str(exc),
                ),
                metadata={
                    "final_holdout_protected": True,
                    "final_holdout_used": False,
                    "research_only": True,
                    "production_approved": False,
                },
            )

            return self._blocked_result(
                model_id,
                failure_gate,
                errors=(
                    str(exc),
                ),
            )

        accuracy = (
            stage_result.holdout_accuracy
        )

        completed = (
            stage_result.successful
            and stage_result.evaluated
        )

        metadata = {
            "stage": "protected_final_holdout",
            "model_id": model_id,
            "research_only": True,
            "production_approved": False,
            "holdout_used_for_selection": False,
            "holdout_used_for_feature_selection": False,
            "holdout_used_for_hyperparameter_tuning": False,
            "holdout_used_for_calibration": False,
            "holdout_used_for_threshold_optimization": False,
            "model_fitted_on_holdout": False,
            "preprocessor_fitted_on_holdout": False,
            "holdout_gate_status": (
                stage_result.gate.status.value
            ),
            "final_holdout_used": (
                stage_result.final_holdout_used
            ),
        }

        metadata.update(
            stage_result.metadata
        )

        return HoldoutPipelineResult(
            model_id=model_id,
            stage=stage_result,
            completed=completed,
            production_approved=False,
            accuracy=accuracy,
            passed_accuracy_gate=(
                stage_result.passed_accuracy_gate
            ),
            errors=(),
            warnings=tuple(
                stage_result.warnings
            ),
            metadata=metadata,
        )


def run_protected_holdout(
    *,
    model_id: str,
    model: Any,
    preprocessor: Any,
    development_data: pd.DataFrame,
    holdout_data: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    gate_input: HoldoutGateInput,
    accuracy_threshold: float = 0.95,
) -> HoldoutPipelineResult:
    """Convenience API for protected final holdout evaluation."""

    pipeline = ProtectedHoldoutPipeline(
        accuracy_threshold=accuracy_threshold
    )

    return pipeline.run(
        model_id=model_id,
        model=model,
        preprocessor=preprocessor,
        development_data=development_data,
        holdout_data=holdout_data,
        feature_columns=feature_columns,
        target_column=target_column,
        gate_input=gate_input,
    )


__all__ = [
    "HoldoutPipelineResult",
    "ProtectedHoldoutPipeline",
    "run_protected_holdout",
]
