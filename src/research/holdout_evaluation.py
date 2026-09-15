"""
AI Swing Analyser — Final Holdout Evaluation.

This module evaluates a FROZEN research candidate on the untouched
final holdout period.

The holdout must not be used for:

    - feature selection
    - hyperparameter optimization
    - model selection
    - preprocessing fitting
    - calibration fitting
    - threshold optimization

It is an evaluation-only dataset.

A model passing the holdout accuracy gate is still NOT automatically
approved for live trading. It must subsequently pass calibration,
range, regime, backtest, robustness and governance gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.models.classifier import DirectionClassifier
from src.models.metrics import (
    ClassificationMetrics,
    classification_metrics,
)
from src.models.preprocessing import SafePreprocessor
from src.research.model_development import DevelopmentPartitions


# ---------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------


@dataclass
class HoldoutEvaluationResult:
    """
    Immutable-in-practice record of final holdout performance.

    The result stores the exact timestamps used for the final evaluation
    so the evaluation can be audited later.
    """

    model_name: str
    horizon: int
    target_column: str

    metrics: ClassificationMetrics

    predictions: np.ndarray
    actuals: np.ndarray

    holdout_index: pd.Index

    feature_columns: list[str]

    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp

    threshold: float = 0.95

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def accuracy(self) -> float:
        return float(
            self.metrics.accuracy
        )

    @property
    def passed_accuracy_gate(self) -> bool:
        return (
            self.accuracy
            >= self.threshold
        )

    @property
    def evaluated_samples(self) -> int:
        return len(
            self.holdout_index
        )

    def summary(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "horizon": self.horizon,
            "target_column": self.target_column,
            "accuracy": self.accuracy,
            "accuracy_gate": self.threshold,
            "passed_accuracy_gate": (
                self.passed_accuracy_gate
            ),
            "evaluated_samples": (
                self.evaluated_samples
            ),
            "holdout_start": (
                self.holdout_start
            ),
            "holdout_end": (
                self.holdout_end
            ),
            "feature_count": len(
                self.feature_columns
            ),
            "metadata": dict(
                self.metadata
            ),
        }


# ---------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------


class FinalHoldoutEvaluator:
    """
    Evaluate a frozen model on the untouched final holdout.

    This class intentionally does not contain:

        fit()
        feature selection
        hyperparameter search
        calibration fitting

    Those operations would violate final-holdout isolation.
    """

    def __init__(
        self,
        minimum_accuracy: float = 0.95,
        require_temporal_separation: bool = True,
    ) -> None:

        if not (
            0.0
            <= minimum_accuracy
            <= 1.0
        ):
            raise ValueError(
                "minimum_accuracy must be between 0 and 1."
            )

        self.minimum_accuracy = (
            float(minimum_accuracy)
        )

        self.require_temporal_separation = (
            bool(
                require_temporal_separation
            )
        )

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_dataframe(
        data: pd.DataFrame,
    ) -> None:

        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                "data must not be empty."
            )

        if not isinstance(
            data.index,
            pd.DatetimeIndex,
        ):
            raise TypeError(
                "data must use DatetimeIndex."
            )

        if data.index.has_duplicates:
            raise ValueError(
                "data contains duplicate timestamps."
            )

        if not data.index.is_monotonic_increasing:
            raise ValueError(
                "data must be chronologically sorted."
            )

    @staticmethod
    def _validate_features(
        data: pd.DataFrame,
        feature_columns: Sequence[str],
    ) -> None:

        if not feature_columns:
            raise ValueError(
                "No feature columns supplied."
            )

        missing = [
            column
            for column in feature_columns
            if column not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing feature columns: {missing}"
            )

        for column in feature_columns:
            if not pd.api.types.is_numeric_dtype(
                data[column]
            ):
                raise TypeError(
                    f"Feature '{column}' is not numeric."
                )

    @staticmethod
    def _validate_target(
        data: pd.DataFrame,
        target_column: str,
    ) -> None:

        if target_column not in data.columns:
            raise ValueError(
                f"Target '{target_column}' not found."
            )

        target = data[
            target_column
        ].dropna()

        if target.empty:
            raise ValueError(
                "Holdout target contains no usable observations."
            )

    # -----------------------------------------------------------------
    # Temporal separation
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_temporal_separation(
        partitions: DevelopmentPartitions,
    ) -> None:

        if not partitions.holdout_is_future:
            raise ValueError(
                "Final holdout is not strictly later than "
                "development data."
            )

        if (
            set(
                partitions.development_index
            )
            & set(
                partitions.holdout_index
            )
        ):
            raise ValueError(
                "Development and final holdout overlap."
            )

    # -----------------------------------------------------------------
    # Frozen model evaluation
    # -----------------------------------------------------------------

    def evaluate(
        self,
        model: DirectionClassifier,
        preprocessor: SafePreprocessor,
        partitions: DevelopmentPartitions,
        feature_columns: Sequence[str],
        target_column: str,
        horizon: int,
        model_name: str = "unknown",
    ) -> HoldoutEvaluationResult:
        """
        Evaluate an already-fitted model and preprocessor.

        Neither object is fitted or modified here.

        Parameters
        ----------
        model:
            Already-fitted DirectionClassifier.

        preprocessor:
            Already-fitted SafePreprocessor. It must have been fitted
            exclusively on development/training data.

        partitions:
            Development/holdout partition created before model selection.

        feature_columns:
            Frozen feature schema.

        target_column:
            Frozen direction target.

        horizon:
            Prediction horizon.

        model_name:
            Human-readable model identifier.
        """

        if model is None:
            raise ValueError(
                "A fitted model is required."
            )

        if preprocessor is None:
            raise ValueError(
                "A fitted preprocessor is required."
            )

        self._validate_dataframe(
            partitions.development
        )

        self._validate_dataframe(
            partitions.holdout
        )

        if self.require_temporal_separation:
            self._validate_temporal_separation(
                partitions
            )

        self._validate_features(
            partitions.holdout,
            feature_columns,
        )

        self._validate_target(
            partitions.holdout,
            target_column,
        )

        if horizon <= 0:
            raise ValueError(
                "horizon must be positive."
            )

        # -------------------------------------------------------------
        # IMPORTANT:
        #
        # The preprocessor is TRANSFORMED only.
        #
        # It must never be fitted on the holdout.
        # -------------------------------------------------------------

        holdout = partitions.holdout.loc[
            partitions.holdout[
                target_column
            ].notna()
        ].copy()

        if holdout.empty:
            raise ValueError(
                "No usable holdout observations remain."
            )

        X_holdout = holdout.loc[
            :,
            list(feature_columns),
        ]

        y_holdout = holdout[
            target_column
        ]

        X_holdout_processed = (
            preprocessor.transform(
                X_holdout
            )
        )

        # -------------------------------------------------------------
        # Frozen model prediction.
        # -------------------------------------------------------------

        predictions = model.predict(
            X_holdout_processed
        )

        predictions = np.asarray(
            predictions
        )

        actuals = np.asarray(
            y_holdout
        )

        if len(predictions) != len(
            actuals
        ):
            raise ValueError(
                "Prediction and holdout target lengths differ."
            )

        metrics = classification_metrics(
            actuals,
            predictions,
        )

        warnings: list[str] = []

        if (
            metrics.accuracy
            < self.minimum_accuracy
        ):
            warnings.append(
                "Final holdout accuracy is below the strict "
                "95% research gate."
            )

        warnings.append(
            "Final holdout was evaluated after model-development "
            "selection and was not used for fitting."
        )

        return HoldoutEvaluationResult(
            model_name=model_name,
            horizon=int(horizon),
            target_column=target_column,
            metrics=metrics,
            predictions=predictions,
            actuals=actuals,
            holdout_index=holdout.index.copy(),
            feature_columns=list(
                feature_columns
            ),
            holdout_start=holdout.index.min(),
            holdout_end=holdout.index.max(),
            threshold=self.minimum_accuracy,
            metadata={
                "final_holdout": True,
                "holdout_used_for_fitting": False,
                "holdout_used_for_feature_selection": False,
                "holdout_used_for_hyperparameter_search": False,
                "holdout_used_for_calibration": False,
                "preprocessor_refit_on_holdout": False,
                "model_refit_on_holdout": False,
                "research_only": True,
            },
        )


# ---------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------


def evaluate_final_holdout(
    model: DirectionClassifier,
    preprocessor: SafePreprocessor,
    partitions: DevelopmentPartitions,
    feature_columns: Sequence[str],
    target_column: str,
    horizon: int,
    model_name: str = "unknown",
    minimum_accuracy: float = 0.95,
) -> HoldoutEvaluationResult:

    evaluator = FinalHoldoutEvaluator(
        minimum_accuracy=minimum_accuracy
    )

    return evaluator.evaluate(
        model=model,
        preprocessor=preprocessor,
        partitions=partitions,
        feature_columns=feature_columns,
        target_column=target_column,
        horizon=horizon,
        model_name=model_name,
    )


__all__ = [
    "HoldoutEvaluationResult",
    "FinalHoldoutEvaluator",
    "evaluate_final_holdout",
]
