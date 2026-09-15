"""
AI Swing Analyser — Final Holdout Research Stage.

Connects:
    frozen selected model
        ↓
    final holdout gate
        ↓
    isolated final holdout evaluation

The final holdout is never used for:
    - model selection
    - feature selection
    - hyperparameter tuning
    - preprocessing fitting
    - calibration fitting
    - threshold optimization

This stage evaluates the holdout only. It does not approve a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from .holdout_evaluation import (
    FinalHoldoutEvaluator,
    HoldoutEvaluationResult,
)
from .holdout_gate import (
    FinalHoldoutGate,
    HoldoutGateInput,
    HoldoutGateResult,
)


@dataclass(frozen=True)
class HoldoutStageResult:
    """Complete result of the final holdout research stage."""

    model_id: str

    gate: HoldoutGateResult
    evaluation: Optional[HoldoutEvaluationResult]

    successful: bool
    final_holdout_used: bool

    holdout_accuracy: Optional[float] = None
    passed_accuracy_gate: bool = False

    warnings: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def eligible_for_evaluation(self) -> bool:
        return self.gate.eligible

    @property
    def evaluated(self) -> bool:
        return self.evaluation is not None

    def summary(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "gate_status": self.gate.status.value,
            "eligible_for_evaluation": (
                self.eligible_for_evaluation
            ),
            "evaluated": self.evaluated,
            "successful": self.successful,
            "final_holdout_used": self.final_holdout_used,
            "holdout_accuracy": self.holdout_accuracy,
            "passed_accuracy_gate": self.passed_accuracy_gate,
            "warning_count": len(self.warnings),
            "warnings": list(self.warnings),
        }


class FinalHoldoutStage:
    """
    Controlled final holdout evaluation stage.

    The selected model and preprocessing objects are supplied by the
    preceding model-development/selection stages.

    This class never fits them.
    """

    def __init__(
        self,
        accuracy_threshold: float = 0.95,
    ) -> None:
        if not 0.0 <= accuracy_threshold <= 1.0:
            raise ValueError(
                "accuracy_threshold must be between 0 and 1."
            )

        self.accuracy_threshold = accuracy_threshold
        self.gate = FinalHoldoutGate()
        self.evaluator = FinalHoldoutEvaluator(
            accuracy_threshold=accuracy_threshold,
        )

    @staticmethod
    def _validate_dataframe(
        data: pd.DataFrame,
        name: str,
    ) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"{name} must be a pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                f"{name} cannot be empty."
            )

        if not isinstance(data.index, pd.DatetimeIndex):
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

    @staticmethod
    def _validate_partitions(
        development_data: pd.DataFrame,
        holdout_data: pd.DataFrame,
    ) -> None:
        development_end = development_data.index.max()
        holdout_start = holdout_data.index.min()

        if development_end >= holdout_start:
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
        if not target_column:
            raise ValueError(
                "target_column cannot be empty."
            )

        if target_column not in development_data.columns:
            raise ValueError(
                f"Target missing from development data: "
                f"{target_column}"
            )

        if target_column not in holdout_data.columns:
            raise ValueError(
                f"Target missing from holdout data: "
                f"{target_column}"
            )

    @staticmethod
    def _validate_model(model: Any) -> None:
        if model is None:
            raise ValueError(
                "Frozen model cannot be None."
            )

        if not hasattr(model, "predict"):
            raise TypeError(
                "Frozen model must provide predict()."
            )

    @staticmethod
    def _validate_preprocessor(preprocessor: Any) -> None:
        if preprocessor is None:
            raise ValueError(
                "Frozen preprocessor cannot be None."
            )

        if not hasattr(preprocessor, "transform"):
            raise TypeError(
                "Frozen preprocessor must provide transform()."
            )

    @staticmethod
    def _validate_no_holdout_fitting(
        metadata: Optional[dict[str, Any]],
    ) -> None:
        if not metadata:
            return

        forbidden_keys = {
            "holdout_used_for_selection",
            "holdout_used_for_feature_selection",
            "holdout_used_for_hyperparameter_tuning",
            "holdout_used_for_calibration",
        }

        violations = [
            key
            for key in forbidden_keys
            if bool(metadata.get(key, False))
        ]

        if violations:
            raise ValueError(
                "Holdout contamination detected: "
                f"{violations}"
            )

    def evaluate(
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
    ) -> HoldoutStageResult:
        """
        Evaluate the final holdout after the protection gate passes.

        The model and preprocessor are never fitted here.
        """

        if not model_id or not model_id.strip():
            raise ValueError(
                "model_id cannot be empty."
            )

        self._validate_model(model)
        self._validate_preprocessor(preprocessor)

        self._validate_dataframe(
            development_data,
            "development_data",
        )

        self._validate_dataframe(
            holdout_data,
            "holdout_data",
        )

        self._validate_partitions(
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

        self._validate_no_holdout_fitting(
            gate_input.metadata,
        )

        if gate_input.model_id != model_id:
            raise ValueError(
                "Holdout gate model_id does not match supplied model."
            )

        gate_result = self.gate.evaluate(
            gate_input
        )

        if not gate_result.eligible:
            return HoldoutStageResult(
                model_id=model_id,
                gate=gate_result,
                evaluation=None,
                successful=False,
                final_holdout_used=False,
                holdout_accuracy=None,
                passed_accuracy_gate=False,
                warnings=tuple(
                    gate_result.failures
                ),
                metadata={
                    "stage": "final_holdout",
                    "evaluation_started": False,
                    "final_holdout_used": False,
                    "research_only": True,
                    "production_approved": False,
                },
            )

        evaluation = self.evaluator.evaluate(
            model=model,
            preprocessor=preprocessor,
            development_data=development_data,
            holdout_data=holdout_data,
            feature_columns=list(feature_columns),
            target_column=target_column,
        )

        accuracy = getattr(
            evaluation,
            "accuracy",
            None,
        )

        if accuracy is None:
            accuracy = getattr(
                evaluation,
                "classification",
                None,
            )

            if accuracy is not None:
                accuracy = getattr(
                    accuracy,
                    "accuracy",
                    None,
                )

        passed = getattr(
            evaluation,
            "passed_accuracy_gate",
            None,
        )

        if passed is None and accuracy is not None:
            passed = (
                float(accuracy)
                >= self.accuracy_threshold
            )

        passed = bool(passed)

        warnings = list(
            gate_result.warnings
        )

        if not passed:
            warnings.append(
                "Final holdout accuracy did not pass the configured "
                "research threshold."
            )

        return HoldoutStageResult(
            model_id=model_id,
            gate=gate_result,
            evaluation=evaluation,
            successful=True,
            final_holdout_used=True,
            holdout_accuracy=(
                float(accuracy)
                if accuracy is not None
                else None
            ),
            passed_accuracy_gate=passed,
            warnings=tuple(warnings),
            metadata={
                "stage": "final_holdout",
                "evaluation_started": True,
                "final_holdout_used": True,
                "holdout_used_for_selection": False,
                "holdout_used_for_feature_selection": False,
                "holdout_used_for_hyperparameter_tuning": False,
                "holdout_used_for_calibration": False,
                "model_fitted_on_holdout": False,
                "preprocessor_fitted_on_holdout": False,
                "research_only": True,
                "production_approved": False,
            },
        )

    def assert_evaluated(
        self,
        result: HoldoutStageResult,
    ) -> HoldoutStageResult:
        """Require that final holdout evaluation actually occurred."""

        if not isinstance(
            result,
            HoldoutStageResult,
        ):
            raise TypeError(
                "result must be a HoldoutStageResult."
            )

        if not result.evaluated:
            message = (
                "Final holdout evaluation was not completed."
            )

            if result.warnings:
                message += " " + " ".join(
                    result.warnings
                )

            raise RuntimeError(message)

        return result


def run_final_holdout_stage(
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
) -> HoldoutStageResult:
    """Convenience API for final holdout evaluation."""

    stage = FinalHoldoutStage(
        accuracy_threshold=accuracy_threshold,
    )

    return stage.evaluate(
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
    "HoldoutStageResult",
    "FinalHoldoutStage",
    "run_final_holdout_stage",
]
