"""
AI Swing Analyser — Research Model Training Pipeline.

Training boundary:

Model-ready research dataset
        ↓
Chronological training subset
        ↓
Train-only preprocessing
        ↓
Candidate direction model
        ↓
Validation predictions / metrics

This module does NOT:
- use the final holdout for fitting
- tune using the final holdout
- calibrate probabilities
- approve production models
- generate trading signals
- claim that 95% accuracy has been achieved

The final holdout remains the responsibility of the protected
holdout/evaluation stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.models.classifier import (
    ClassifierConfig,
    DirectionClassifier,
)
from src.models.metrics import (
    classification_metrics,
)
from src.models.preprocessing import (
    PreprocessorConfig,
    SafePreprocessor,
)
from src.models.splitter import (
    chronological_split,
)
from .research_model_pipeline import (
    ResearchModelDataset,
)


@dataclass(frozen=True)
class ResearchModelTrainingConfig:
    """Configuration for research candidate training."""

    model_type: str = "gradient_boosting"

    train_fraction: float = 0.75

    validation_fraction: float = 0.25

    minimum_training_rows: int = 100

    minimum_validation_rows: int = 30

    preprocessing: PreprocessorConfig = field(
        default_factory=PreprocessorConfig
    )

    classifier: ClassifierConfig = field(
        default_factory=ClassifierConfig
    )

    random_state: int = 42

    def __post_init__(self) -> None:
        if not isinstance(
            self.model_type,
            str,
        ) or not self.model_type.strip():
            raise ValueError(
                "model_type must be a non-empty string."
            )

        if not (
            0.0
            < self.train_fraction
            < 1.0
        ):
            raise ValueError(
                "train_fraction must be between 0 and 1."
            )

        if not (
            0.0
            < self.validation_fraction
            < 1.0
        ):
            raise ValueError(
                "validation_fraction must be between 0 and 1."
            )

        if not np.isclose(
            self.train_fraction
            + self.validation_fraction,
            1.0,
        ):
            raise ValueError(
                "train_fraction + validation_fraction must equal 1."
            )

        if (
            self.minimum_training_rows
            < 1
        ):
            raise ValueError(
                "minimum_training_rows must be positive."
            )

        if (
            self.minimum_validation_rows
            < 1
        ):
            raise ValueError(
                "minimum_validation_rows must be positive."
            )


@dataclass
class ResearchModelTrainingResult:
    """Candidate model training result."""

    model: DirectionClassifier

    preprocessor: SafePreprocessor

    feature_columns: list[str]

    target_column: str

    train_index: pd.DatetimeIndex

    validation_index: pd.DatetimeIndex

    validation_predictions: np.ndarray

    validation_probabilities: np.ndarray

    validation_metrics: dict[str, float]

    validation_accuracy: float

    candidate_passed: bool

    warnings: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, object] = field(
        default_factory=dict
    )

    @property
    def production_ready(self) -> bool:
        """
        Training alone can never make a model production-ready.
        """

        return False

    def summary(self) -> dict[str, object]:
        return {
            "target_column": self.target_column,
            "feature_count": len(
                self.feature_columns
            ),
            "train_rows": len(
                self.train_index
            ),
            "validation_rows": len(
                self.validation_index
            ),
            "validation_accuracy": (
                self.validation_accuracy
            ),
            "candidate_passed": (
                self.candidate_passed
            ),
            "production_ready": False,
            "research_only": True,
            "final_holdout_used": False,
        }


class ResearchModelTrainingPipeline:
    """
    Train a single research candidate using a chronological split.
    """

    def __init__(
        self,
        *,
        config: ResearchModelTrainingConfig
        | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else ResearchModelTrainingConfig()
        )

    def _validate_dataset(
        self,
        dataset: ResearchModelDataset,
    ) -> None:
        if not isinstance(
            dataset,
            ResearchModelDataset,
        ):
            raise TypeError(
                "dataset must be ResearchModelDataset."
            )

        if dataset.rows == 0:
            raise ValueError(
                "Research dataset is empty."
            )

        if not dataset.feature_columns:
            raise ValueError(
                "Research dataset contains no features."
            )

        if not dataset.target_columns:
            raise ValueError(
                "Research dataset contains no targets."
            )

    def _resolve_target(
        self,
        dataset: ResearchModelDataset,
        target_column: str | None,
    ) -> str:
        """
        Resolve a single supervised-learning target.

        Direction targets are preferred because this training
        pipeline is specifically for directional classification.
        """

        if target_column is not None:
            if target_column not in (
                dataset.target_columns
            ):
                raise ValueError(
                    "Unknown target column: "
                    f"{target_column}"
                )

            return target_column

        if dataset.direction_columns:
            return dataset.direction_columns[0]

        raise ValueError(
            "No direction target is available."
        )

    def _prepare_supervised_data(
        self,
        dataset: ResearchModelDataset,
        target_column: str,
    ) -> tuple[
        pd.DataFrame,
        pd.Series,
    ]:
        """Create clean chronological X/y data."""

        subset = dataset.data[
            dataset.feature_columns
            + [target_column]
        ].copy(
            deep=True
        )

        subset = subset.dropna(
            subset=[
                target_column
            ]
        )

        if subset.empty:
            raise ValueError(
                "No supervised rows remain after "
                "removing missing target values."
            )

        X = subset[
            dataset.feature_columns
        ].copy(
            deep=True
        )

        y = subset[
            target_column
        ].copy(
            deep=True
        )

        if not X.index.is_monotonic_increasing:
            raise RuntimeError(
                "Training data is not chronological."
            )

        return X, y

    def _create_model(
        self,
    ) -> DirectionClassifier:
        """Create a fresh classifier."""

        classifier_config = (
            self.config.classifier
        )

        # The existing classifier configuration is authoritative.
        # model_type is exposed here for explicit research metadata.
        model = DirectionClassifier(
            config=classifier_config
        )

        return model

    def train(
        self,
        dataset: ResearchModelDataset,
        *,
        target_column: str | None = None,
    ) -> ResearchModelTrainingResult:
        """
        Train and evaluate one research candidate.

        The newest validation period is used only for validation.
        It is never treated as the final holdout.
        """

        self._validate_dataset(
            dataset
        )

        resolved_target = self._resolve_target(
            dataset,
            target_column,
        )

        X, y = self._prepare_supervised_data(
            dataset,
            resolved_target,
        )

        # -----------------------------------------------------------
        # Chronological split
        # -----------------------------------------------------------

        split = chronological_split(
            X,
            y,
            train_fraction=(
                self.config.train_fraction
            ),
        )

        X_train = split.X_train
        X_validation = split.X_test

        y_train = split.y_train
        y_validation = split.y_test

        if len(X_train) < (
            self.config.minimum_training_rows
        ):
            raise ValueError(
                "Insufficient training rows: "
                f"{len(X_train)} < "
                f"{self.config.minimum_training_rows}"
            )

        if len(X_validation) < (
            self.config.minimum_validation_rows
        ):
            raise ValueError(
                "Insufficient validation rows: "
                f"{len(X_validation)} < "
                f"{self.config.minimum_validation_rows}"
            )

        if (
            X_train.index[-1]
            >= X_validation.index[0]
        ):
            raise RuntimeError(
                "Training and validation periods overlap."
            )

        # -----------------------------------------------------------
        # Train-only preprocessing
        # -----------------------------------------------------------

        preprocessor = SafePreprocessor(
            config=self.config.preprocessing
        )

        X_train_processed = (
            preprocessor.fit_transform(
                X_train
            )
        )

        X_validation_processed = (
            preprocessor.transform(
                X_validation
            )
        )

        # -----------------------------------------------------------
        # Fresh model
        # -----------------------------------------------------------

        model = self._create_model()

        model.fit(
            X_train_processed,
            y_train.to_numpy(),
        )

        # -----------------------------------------------------------
        # Validation prediction
        # -----------------------------------------------------------

        validation_predictions = np.asarray(
            model.predict(
                X_validation_processed
            )
        )

        validation_probabilities = np.asarray(
            model.predict_proba(
                X_validation_processed
            )
        )

        if validation_probabilities.ndim != 2:
            raise RuntimeError(
                "Model probability output must be two-dimensional."
            )

        if validation_probabilities.shape[1] != 2:
            raise RuntimeError(
                "Directional classifier must produce "
                "two-class probabilities."
            )

        # -----------------------------------------------------------
        # Metrics
        # -----------------------------------------------------------

        metrics = classification_metrics(
            y_validation.to_numpy(),
            validation_predictions,
            validation_probabilities,
        )

        validation_accuracy = float(
            metrics.accuracy
        )

        candidate_passed = (
            validation_accuracy
            >= 0.95
        )

        warnings: list[str] = []

        if not candidate_passed:
            warnings.append(
                "Candidate did not meet the strict 95% "
                "validation accuracy research gate."
            )

        warnings.append(
            "This validation result is not a final holdout result."
        )

        metadata = {
            "research_only": True,
            "production_ready": False,
            "production_approved": False,
            "final_holdout_used": False,
            "final_holdout_fitted": False,
            "calibration_fitted": False,
            "threshold_optimization_completed": False,
            "model_selected": False,
            "walk_forward_validated": False,
            "model_type": self.config.model_type,
            "target_column": resolved_target,
            "random_state": self.config.random_state,
            "train_start": str(
                X_train.index[0]
            ),
            "train_end": str(
                X_train.index[-1]
            ),
            "validation_start": str(
                X_validation.index[0]
            ),
            "validation_end": str(
                X_validation.index[-1]
            ),
        }

        return ResearchModelTrainingResult(
            model=model,
            preprocessor=preprocessor,
            feature_columns=list(
                dataset.feature_columns
            ),
            target_column=resolved_target,
            train_index=X_train.index.copy(),
            validation_index=X_validation.index.copy(),
            validation_predictions=(
                validation_predictions.copy()
            ),
            validation_probabilities=(
                validation_probabilities.copy()
            ),
            validation_metrics={
                key: float(value)
                for key, value
                in metrics.__dict__.items()
                if isinstance(
                    value,
                    (int, float, np.number),
                )
            },
            validation_accuracy=(
                validation_accuracy
            ),
            candidate_passed=(
                candidate_passed
            ),
            warnings=warnings,
            metadata=metadata,
        )


def train_research_model(
    dataset: ResearchModelDataset,
    *,
    target_column: str | None = None,
    config: ResearchModelTrainingConfig
    | None = None,
) -> ResearchModelTrainingResult:
    """Convenience function for research model training."""

    pipeline = ResearchModelTrainingPipeline(
        config=config
    )

    return pipeline.train(
        dataset,
        target_column=target_column,
    )


def research_model_training_summary(
    result: ResearchModelTrainingResult,
) -> dict[str, object]:
    """Return a compact training summary."""

    if not isinstance(
        result,
        ResearchModelTrainingResult,
    ):
        raise TypeError(
            "result must be ResearchModelTrainingResult."
        )

    return result.summary()


__all__ = [
    "ResearchModelTrainingConfig",
    "ResearchModelTrainingResult",
    "ResearchModelTrainingPipeline",
    "train_research_model",
    "research_model_training_summary",
]
