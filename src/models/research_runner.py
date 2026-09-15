"""
End-to-end research experiment runner for AI Swing Analyser.

This module coordinates the existing research components without mixing
training, validation and production inference.

Design principles:
    - chronological data handling
    - no random shuffling
    - explicit validation
    - explicit final holdout
    - no automatic production approval
    - transparent experiment results
    - reproducible configuration

This is a research orchestrator, not a live trading engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import (
    classification_metrics,
    regression_metrics,
)
from .pipeline import (
    TrainingConfig,
    prepare_dataset,
    split_dataset,
)
from .splitter import TimeSplit, validate_time_split


@dataclass
class ResearchRunConfig:
    """
    Configuration for one end-to-end research run.
    """

    symbol: str
    timeframe: str
    horizon: int

    target_column: str

    minimum_accuracy: float = 0.95

    require_validation: bool = True
    require_holdout: bool = True

    minimum_train_samples: int = 100
    minimum_validation_samples: int = 50
    minimum_test_samples: int = 50

    random_state: int = 42

    experiment_name: str = "swing_model_research"

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError(
                "symbol cannot be empty."
            )

        if not self.timeframe.strip():
            raise ValueError(
                "timeframe cannot be empty."
            )

        if self.horizon <= 0:
            raise ValueError(
                "horizon must be positive."
            )

        if not self.target_column.strip():
            raise ValueError(
                "target_column cannot be empty."
            )

        if not 0.0 <= self.minimum_accuracy <= 1.0:
            raise ValueError(
                "minimum_accuracy must be between 0 and 1."
            )

        if self.minimum_train_samples < 1:
            raise ValueError(
                "minimum_train_samples must be positive."
            )

        if self.minimum_validation_samples < 1:
            raise ValueError(
                "minimum_validation_samples must be positive."
            )

        if self.minimum_test_samples < 1:
            raise ValueError(
                "minimum_test_samples must be positive."
            )


@dataclass
class ResearchPredictions:
    """
    Predictions generated for one dataset partition.
    """

    index: pd.Index

    y_true: np.ndarray

    y_probability: np.ndarray

    y_pred: np.ndarray


@dataclass
class ResearchRunResult:
    """
    Complete result from one research run.
    """

    run_id: str

    config: ResearchRunConfig

    split: TimeSplit

    validation_metrics: Dict[str, float]

    holdout_metrics: Dict[str, float]

    validation_predictions: ResearchPredictions

    holdout_predictions: ResearchPredictions

    feature_names: Sequence[str]

    model_name: str

    started_at: str

    completed_at: str

    validation_passed: bool

    holdout_passed: bool

    leakage_check_passed: bool

    research_candidate: bool

    notes: list[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def passed(self) -> bool:
        """
        Whether the core research gates passed.

        This does NOT mean production approval.
        """

        return (
            self.validation_passed
            and self.holdout_passed
            and self.leakage_check_passed
        )

    def summary(self) -> Dict[str, Any]:
        """Return a compact summary."""

        return {
            "run_id": self.run_id,
            "symbol": self.config.symbol,
            "timeframe": self.config.timeframe,
            "horizon": self.config.horizon,
            "model": self.model_name,
            "validation_accuracy": (
                self.validation_metrics.get(
                    "accuracy"
                )
            ),
            "holdout_accuracy": (
                self.holdout_metrics.get(
                    "accuracy"
                )
            ),
            "validation_passed": self.validation_passed,
            "holdout_passed": self.holdout_passed,
            "leakage_check_passed": (
                self.leakage_check_passed
            ),
            "research_candidate": (
                self.research_candidate
            ),
        }


class ResearchRunner:
    """
    Coordinates model research.

    The runner accepts a training callback so that different model
    families can be evaluated without rewriting the orchestration logic.
    """

    def __init__(
        self,
        config: ResearchRunConfig,
    ) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def run_classifier_experiment(
        self,
        data: pd.DataFrame,
        model_factory: Callable[[], Any],
        *,
        feature_columns: Optional[Sequence[str]] = None,
        model_name: str = "classifier",
        training_config: Optional[TrainingConfig] = None,
    ) -> ResearchRunResult:
        """
        Run a classifier experiment.

        Parameters
        ----------
        data:
            Research dataset containing features and target.

        model_factory:
            Callable returning a fresh, unfitted model.

        feature_columns:
            Optional explicit feature schema.

        model_name:
            Human-readable model name.

        training_config:
            Optional training configuration.

        Returns
        -------
        ResearchRunResult
        """

        started_at = self._utc_now()

        run_id = self._generate_run_id()

        # --------------------------------------------------------------
        # Prepare data
        # --------------------------------------------------------------

        prepared = prepare_dataset(
            data,
            target_column=self.config.target_column,
            feature_columns=feature_columns,
        )

        if len(prepared) < (
            self.config.minimum_train_samples
            + self.config.minimum_validation_samples
            + self.config.minimum_test_samples
        ):
            raise ValueError(
                "Dataset is too small for the configured research run."
            )

        # --------------------------------------------------------------
        # Chronological split
        # --------------------------------------------------------------

        split = split_dataset(
            prepared,
            training_config
            or TrainingConfig(
                target_column=self.config.target_column
            ),
        )

        validate_time_split(split)

        self._validate_partition_sizes(split)

        # --------------------------------------------------------------
        # Leakage audit
        # --------------------------------------------------------------

        leakage_check_passed = (
            self._check_temporal_integrity(split)
            and self._check_feature_target_separation(
                split,
                feature_columns,
            )
        )

        if not leakage_check_passed:
            raise RuntimeError(
                "Research run failed temporal or feature/target "
                "leakage checks."
            )

        # --------------------------------------------------------------
        # Extract data
        # --------------------------------------------------------------

        X_train = split.train.drop(
            columns=[self.config.target_column]
        )

        y_train = split.train[
            self.config.target_column
        ]

        X_validation = split.validation.drop(
            columns=[self.config.target_column]
        )

        y_validation = split.validation[
            self.config.target_column
        ]

        X_test = split.test.drop(
            columns=[self.config.target_column]
        )

        y_test = split.test[
            self.config.target_column
        ]

        # --------------------------------------------------------------
        # Train fresh model
        # --------------------------------------------------------------

        model = model_factory()

        if not hasattr(model, "fit"):
            raise TypeError(
                "model_factory must return an object with fit()."
            )

        if not hasattr(model, "predict"):
            raise TypeError(
                "model_factory must return an object with predict()."
            )

        model.fit(
            X_train,
            y_train,
        )

        # --------------------------------------------------------------
        # Validation predictions
        # --------------------------------------------------------------

        validation_pred = model.predict(
            X_validation
        )

        validation_probability = (
            self._predict_probability(
                model,
                X_validation,
            )
        )

        validation_predictions = (
            ResearchPredictions(
                index=X_validation.index,
                y_true=np.asarray(
                    y_validation
                ),
                y_probability=validation_probability,
                y_pred=np.asarray(
                    validation_pred
                ),
            )
        )

        validation_metrics = (
            classification_metrics(
                y_validation,
                validation_pred,
                validation_probability,
            )
        )

        validation_metrics_dict = (
            self._metrics_to_dict(
                validation_metrics
            )
        )

        # --------------------------------------------------------------
        # Final holdout predictions
        # --------------------------------------------------------------

        holdout_pred = model.predict(
            X_test
        )

        holdout_probability = (
            self._predict_probability(
                model,
                X_test,
            )
        )

        holdout_predictions = ResearchPredictions(
            index=X_test.index,
            y_true=np.asarray(
                y_test
            ),
            y_probability=holdout_probability,
            y_pred=np.asarray(
                holdout_pred
            ),
        )

        holdout_metrics = (
            classification_metrics(
                y_test,
                holdout_pred,
                holdout_probability,
            )
        )

        holdout_metrics_dict = (
            self._metrics_to_dict(
                holdout_metrics
            )
        )

        # --------------------------------------------------------------
        # Gates
        # --------------------------------------------------------------

        validation_passed = (
            validation_metrics_dict.get(
                "accuracy",
                0.0,
            )
            >= self.config.minimum_accuracy
        )

        holdout_passed = (
            holdout_metrics_dict.get(
                "accuracy",
                0.0,
            )
            >= self.config.minimum_accuracy
        )

        research_candidate = (
            validation_passed
            and holdout_passed
            and leakage_check_passed
        )

        notes: list[str] = []

        if validation_passed:
            notes.append(
                "Validation accuracy passed the configured research threshold."
            )
        else:
            notes.append(
                "Validation accuracy did not pass the configured research threshold."
            )

        if holdout_passed:
            notes.append(
                "Final holdout accuracy passed the configured research threshold."
            )
        else:
            notes.append(
                "Final holdout accuracy did not pass the configured research threshold."
            )

        notes.append(
            "This result is not production-approved."
        )

        completed_at = self._utc_now()

        return ResearchRunResult(
            run_id=run_id,
            config=self.config,
            split=split,
            validation_metrics=validation_metrics_dict,
            holdout_metrics=holdout_metrics_dict,
            validation_predictions=validation_predictions,
            holdout_predictions=holdout_predictions,
            feature_names=list(
                X_train.columns
            ),
            model_name=model_name,
            started_at=started_at,
            completed_at=completed_at,
            validation_passed=validation_passed,
            holdout_passed=holdout_passed,
            leakage_check_passed=leakage_check_passed,
            research_candidate=research_candidate,
            notes=notes,
            metadata=dict(
                self.config.metadata
            ),
        )

    # ------------------------------------------------------------------
    # Partition validation
    # ------------------------------------------------------------------

    def _validate_partition_sizes(
        self,
        split: TimeSplit,
    ) -> None:
        if len(split.train) < (
            self.config.minimum_train_samples
        ):
            raise ValueError(
                "Training partition is too small."
            )

        if len(split.validation) < (
            self.config.minimum_validation_samples
        ):
            raise ValueError(
                "Validation partition is too small."
            )

        if len(split.test) < (
            self.config.minimum_test_samples
        ):
            raise ValueError(
                "Final holdout partition is too small."
            )

    # ------------------------------------------------------------------
    # Leakage checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_temporal_integrity(
        split: TimeSplit,
    ) -> bool:
        """
        Verify strict chronological ordering.

        No observation from a later partition may appear in an earlier
        partition.
        """

        train_index = split.train.index
        validation_index = split.validation.index
        test_index = split.test.index

        if len(train_index) == 0:
            return False

        if len(validation_index) == 0:
            return False

        if len(test_index) == 0:
            return False

        train_end = train_index.max()
        validation_start = validation_index.min()
        validation_end = validation_index.max()
        test_start = test_index.min()

        if train_end >= validation_start:
            return False

        if validation_end >= test_start:
            return False

        return True

    @staticmethod
    def _check_feature_target_separation(
        split: TimeSplit,
        feature_columns: Optional[Sequence[str]],
    ) -> bool:
        """
        Verify that explicit feature schemas do not contain target columns.
        """

        if feature_columns is None:
            columns = set(
                split.train.columns
            )
        else:
            columns = set(feature_columns)

        target_like = {
            column
            for column in columns
            if (
                "future" in column.lower()
                or "target" in column.lower()
                or "direction" in column.lower()
            )
        }

        return len(target_like) == 0

    # ------------------------------------------------------------------
    # Probability handling
    # ------------------------------------------------------------------

    @staticmethod
    def _predict_probability(
        model: Any,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Obtain positive-class probabilities.

        Models without predict_proba receive a probability-like fallback
        derived from predictions. Such fallback should not be used for
        calibration.
        """

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)

            probabilities = np.asarray(
                probabilities,
                dtype=float,
            )

            if probabilities.ndim == 2:
                if probabilities.shape[1] == 1:
                    return probabilities[:, 0]

                return probabilities[:, -1]

            return probabilities.reshape(-1)

        predictions = np.asarray(
            model.predict(X)
        )

        return predictions.astype(
            float
        )

    # ------------------------------------------------------------------
    # Metric conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _metrics_to_dict(
        metrics: Any,
    ) -> Dict[str, float]:
        """
        Convert metric dataclass/object into a dictionary.
        """

        if hasattr(metrics, "__dataclass_fields__"):
            values = {
                name: getattr(metrics, name)
                for name in metrics.__dataclass_fields__
            }
        elif isinstance(metrics, dict):
            values = dict(metrics)
        else:
            values = {
                key: value
                for key, value in vars(
                    metrics
                ).items()
                if not key.startswith("_")
            }

        result: Dict[str, float] = {}

        for key, value in values.items():
            if value is None:
                continue

            if isinstance(
                value,
                (int, float, np.integer, np.floating),
            ):
                if np.isfinite(value):
                    result[key] = float(value)

        return result

    # ------------------------------------------------------------------
    # Run ID
    # ------------------------------------------------------------------

    def _generate_run_id(self) -> str:
        timestamp = datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%d%H%M%S"
        )

        symbol = (
            self.config.symbol
            .replace("^", "")
            .replace("/", "_")
            .replace(" ", "_")
        )

        return (
            f"{symbol}_"
            f"{self.config.timeframe}_"
            f"H{self.config.horizon}_"
            f"{timestamp}"
        )

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(
            timezone.utc
        ).isoformat()


def research_result_to_dict(
    result: ResearchRunResult,
) -> Dict[str, Any]:
    """
    Convert a research result into a serializable dictionary.
    """

    return {
        "run_id": result.run_id,
        "config": {
            "symbol": result.config.symbol,
            "timeframe": result.config.timeframe,
            "horizon": result.config.horizon,
            "target_column": result.config.target_column,
            "minimum_accuracy": (
                result.config.minimum_accuracy
            ),
            "experiment_name": (
                result.config.experiment_name
            ),
        },
        "model_name": result.model_name,
        "started_at": result.started_at,
        "completed_at": result.completed_at,
        "validation_metrics": dict(
            result.validation_metrics
        ),
        "holdout_metrics": dict(
            result.holdout_metrics
        ),
        "validation_passed": (
            result.validation_passed
        ),
        "holdout_passed": (
            result.holdout_passed
        ),
        "leakage_check_passed": (
            result.leakage_check_passed
        ),
        "research_candidate": (
            result.research_candidate
        ),
        "feature_names": list(
            result.feature_names
        ),
        "notes": list(result.notes),
        "metadata": dict(result.metadata),
    }


def compare_research_results(
    results: Sequence[ResearchRunResult],
) -> pd.DataFrame:
    """
    Create a comparison table for multiple research runs.
    """

    records = []

    for result in results:
        records.append(
            {
                "run_id": result.run_id,
                "symbol": result.config.symbol,
                "timeframe": result.config.timeframe,
                "horizon": result.config.horizon,
                "model": result.model_name,
                "validation_accuracy": (
                    result.validation_metrics.get(
                        "accuracy"
                    )
                ),
                "holdout_accuracy": (
                    result.holdout_metrics.get(
                        "accuracy"
                    )
                ),
                "validation_f1": (
                    result.validation_metrics.get(
                        "f1"
                    )
                ),
                "holdout_f1": (
                    result.holdout_metrics.get(
                        "f1"
                    )
                ),
                "validation_brier": (
                    result.validation_metrics.get(
                        "brier_score"
                    )
                ),
                "holdout_brier": (
                    result.holdout_metrics.get(
                        "brier_score"
                    )
                ),
                "validation_passed": (
                    result.validation_passed
                ),
                "holdout_passed": (
                    result.holdout_passed
                ),
                "leakage_check_passed": (
                    result.leakage_check_passed
                ),
                "research_candidate": (
                    result.research_candidate
                ),
            }
        )

    return pd.DataFrame(records)


def print_research_summary(
    result: ResearchRunResult,
) -> None:
    """Print a human-readable research summary."""

    print("=" * 70)
    print("AI SWING ANALYSER — RESEARCH RESULT")
    print("=" * 70)

    print(f"Run ID:       {result.run_id}")
    print(f"Symbol:       {result.config.symbol}")
    print(f"Timeframe:    {result.config.timeframe}")
    print(f"Horizon:      {result.config.horizon}")
    print(f"Model:        {result.model_name}")
    print()

    validation_accuracy = (
        result.validation_metrics.get(
            "accuracy"
        )
    )

    holdout_accuracy = (
        result.holdout_metrics.get(
            "accuracy"
        )
    )

    print(
        "Validation accuracy:",
        (
            f"{validation_accuracy:.2%}"
            if validation_accuracy is not None
            else "N/A"
        ),
    )

    print(
        "Holdout accuracy:   ",
        (
            f"{holdout_accuracy:.2%}"
            if holdout_accuracy is not None
            else "N/A"
        ),
    )

    print()
    print(
        "Validation passed:",
        result.validation_passed,
    )

    print(
        "Holdout passed:",
        result.holdout_passed,
    )

    print(
        "Leakage check:",
        result.leakage_check_passed,
    )

    print(
        "Research candidate:",
        result.research_candidate,
    )

    print()
    print(
        "IMPORTANT: Research candidate does NOT mean "
        "production-approved."
    )

    print("=" * 70)


__all__ = [
    "ResearchRunConfig",
    "ResearchPredictions",
    "ResearchRunResult",
    "ResearchRunner",
    "research_result_to_dict",
    "compare_research_results",
    "print_research_summary",
]
