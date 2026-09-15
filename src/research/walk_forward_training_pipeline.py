"""
AI Swing Analyser — Walk-Forward Training Pipeline.

Research-only walk-forward training layer.

Purpose
-------
Repeatedly train a fresh model on historical data and evaluate it on
the immediately following unseen period.

For every fold:

    Past data
        ↓
    Train-only preprocessing
        ↓
    Fresh model
        ↓
    Future validation window
        ↓
    Out-of-sample prediction

Important
---------
This module does NOT:

- use the final holdout
- fit preprocessing on validation data
- reuse a fitted model between folds
- calibrate probabilities
- optimize thresholds
- approve production models
- generate live trading signals
- claim 95% accuracy without evidence

Walk-forward performance is evaluated only on development data.
The final holdout remains untouched.
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
    purged_walk_forward_splits,
)
from .research_model_pipeline import (
    ResearchModelDataset,
)


@dataclass(frozen=True)
class WalkForwardTrainingConfig:
    """Configuration for walk-forward model training."""

    model_type: str = "gradient_boosting"

    n_splits: int = 5

    minimum_training_rows: int = 100

    minimum_validation_rows: int = 30

    gap: int = 0

    embargo: int = 0

    expanding: bool = True

    train_size: int | None = None

    validation_size: int | None = None

    preprocessing: PreprocessorConfig = field(
        default_factory=PreprocessorConfig
    )

    classifier: ClassifierConfig = field(
        default_factory=ClassifierConfig
    )

    minimum_accuracy: float = 0.95

    maximum_accuracy_std: float = 0.10

    random_state: int = 42

    def __post_init__(self) -> None:
        if not isinstance(
            self.model_type,
            str,
        ) or not self.model_type.strip():
            raise ValueError(
                "model_type must be a non-empty string."
            )

        if self.n_splits < 1:
            raise ValueError(
                "n_splits must be at least 1."
            )

        if self.minimum_training_rows < 1:
            raise ValueError(
                "minimum_training_rows must be positive."
            )

        if self.minimum_validation_rows < 1:
            raise ValueError(
                "minimum_validation_rows must be positive."
            )

        if self.gap < 0:
            raise ValueError(
                "gap cannot be negative."
            )

        if self.embargo < 0:
            raise ValueError(
                "embargo cannot be negative."
            )

        if not (
            0.0
            <= self.minimum_accuracy
            <= 1.0
        ):
            raise ValueError(
                "minimum_accuracy must be between 0 and 1."
            )

        if not (
            0.0
            <= self.maximum_accuracy_std
        ):
            raise ValueError(
                "maximum_accuracy_std cannot be negative."
            )

        if (
            self.train_size is not None
            and self.train_size < 1
        ):
            raise ValueError(
                "train_size must be positive when supplied."
            )

        if (
            self.validation_size is not None
            and self.validation_size < 1
        ):
            raise ValueError(
                "validation_size must be positive when supplied."
            )


@dataclass
class WalkForwardFoldResult:
    """Result for one walk-forward fold."""

    fold_number: int

    train_index: pd.DatetimeIndex

    validation_index: pd.DatetimeIndex

    predictions: np.ndarray

    probabilities: np.ndarray

    metrics: dict[str, float]

    accuracy: float

    model: DirectionClassifier | None = None

    preprocessor: SafePreprocessor | None = None

    metadata: dict[str, object] = field(
        default_factory=dict
    )

    @property
    def train_rows(self) -> int:
        return len(self.train_index)

    @property
    def validation_rows(self) -> int:
        return len(
            self.validation_index
        )


@dataclass
class WalkForwardTrainingResult:
    """Aggregated walk-forward training result."""

    folds: list[WalkForwardFoldResult]

    feature_columns: list[str]

    target_column: str

    oos_predictions: pd.DataFrame

    mean_accuracy: float

    median_accuracy: float

    minimum_accuracy: float

    maximum_accuracy: float

    accuracy_std: float

    candidate_passed: bool

    warnings: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, object] = field(
        default_factory=dict
    )

    @property
    def fold_count(self) -> int:
        return len(self.folds)

    @property
    def total_oos_rows(self) -> int:
        return len(
            self.oos_predictions
        )

    @property
    def production_ready(self) -> bool:
        """
        Walk-forward training alone never grants production approval.
        """

        return False

    def summary(self) -> dict[str, object]:
        return {
            "target_column": self.target_column,
            "feature_count": len(
                self.feature_columns
            ),
            "fold_count": self.fold_count,
            "total_oos_rows": self.total_oos_rows,
            "mean_accuracy": self.mean_accuracy,
            "median_accuracy": self.median_accuracy,
            "minimum_accuracy": self.minimum_accuracy,
            "maximum_accuracy": self.maximum_accuracy,
            "accuracy_std": self.accuracy_std,
            "candidate_passed": self.candidate_passed,
            "research_only": True,
            "production_ready": False,
            "final_holdout_used": False,
        }


class WalkForwardTrainingPipeline:
    """
    Leakage-aware walk-forward training engine.
    """

    def __init__(
        self,
        *,
        config: WalkForwardTrainingConfig
        | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else WalkForwardTrainingConfig()
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
                "Research model dataset is empty."
            )

        if not dataset.feature_columns:
            raise ValueError(
                "No feature columns are available."
            )

        if not dataset.target_columns:
            raise ValueError(
                "No target columns are available."
            )

        if not dataset.data.index.is_monotonic_increasing:
            raise ValueError(
                "Dataset must be chronologically sorted."
            )

        if dataset.data.index.has_duplicates:
            raise ValueError(
                "Dataset index contains duplicate timestamps."
            )

    def _resolve_target(
        self,
        dataset: ResearchModelDataset,
        target_column: str | None,
    ) -> str:
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
            "No directional target is available."
        )

    def _prepare_data(
        self,
        dataset: ResearchModelDataset,
        target_column: str,
    ) -> tuple[pd.DataFrame, pd.Series]:
        columns = (
            dataset.feature_columns
            + [target_column]
        )

        data = dataset.data[
            columns
        ].copy(
            deep=True
        )

        data = data.dropna(
            subset=[
                target_column
            ]
        )

        if data.empty:
            raise ValueError(
                "No rows remain after removing "
                "missing target values."
            )

        X = data[
            dataset.feature_columns
        ].copy(
            deep=True
        )

        y = data[
            target_column
        ].copy(
            deep=True
        )

        if not X.index.is_monotonic_increasing:
            raise ValueError(
                "Prepared feature data is not chronological."
            )

        if X.index.has_duplicates:
            raise ValueError(
                "Prepared feature data contains duplicate timestamps."
            )

        return X, y

    def _create_model(
        self,
    ) -> DirectionClassifier:
        return DirectionClassifier(
            config=self.config.classifier
        )

    def _create_splits(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ):
        """
        Create purged chronological folds.

        No random shuffling is used.
        """

        return list(
            purged_walk_forward_splits(
                X,
                y,
                n_splits=self.config.n_splits,
                train_size=self.config.train_size,
                test_size=self.config.validation_size,
                gap=self.config.gap,
                embargo=self.config.embargo,
                expanding=self.config.expanding,
            )
        )

    def train(
        self,
        dataset: ResearchModelDataset,
        *,
        target_column: str | None = None,
    ) -> WalkForwardTrainingResult:
        """
        Perform fresh training on every walk-forward fold.
        """

        self._validate_dataset(
            dataset
        )

        resolved_target = self._resolve_target(
            dataset,
            target_column,
        )

        X, y = self._prepare_data(
            dataset,
            resolved_target,
        )

        splits = self._create_splits(
            X,
            y,
        )

        if not splits:
            raise ValueError(
                "No valid walk-forward folds were created."
            )

        fold_results: list[
            WalkForwardFoldResult
        ] = []

        oos_frames: list[
            pd.DataFrame
        ] = []

        for fold_number, split in enumerate(
            splits,
            start=1,
        ):
            train_positions = split.train_indices
            validation_positions = (
                split.test_indices
            )

            train_index = X.index[
                train_positions
            ]

            validation_index = X.index[
                validation_positions
            ]

            if len(train_index) < (
                self.config.minimum_training_rows
            ):
                continue

            if len(validation_index) < (
                self.config.minimum_validation_rows
            ):
                continue

            if train_index[-1] >= validation_index[0]:
                raise RuntimeError(
                    "Walk-forward fold violates temporal ordering."
                )

            if set(train_index) & set(validation_index):
                raise RuntimeError(
                    "Walk-forward fold contains overlapping "
                    "training and validation timestamps."
                )

            X_train = X.iloc[
                train_positions
            ].copy(
                deep=True
            )

            X_validation = X.iloc[
                validation_positions
            ].copy(
                deep=True
            )

            y_train = y.iloc[
                train_positions
            ].copy(
                deep=True
            )

            y_validation = y.iloc[
                validation_positions
            ].copy(
                deep=True
            )

            # -------------------------------------------------------
            # Fresh preprocessing for every fold.
            # -------------------------------------------------------

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

            # -------------------------------------------------------
            # Fresh model for every fold.
            # -------------------------------------------------------

            model = self._create_model()

            model.fit(
                X_train_processed,
                y_train.to_numpy(),
            )

            predictions = np.asarray(
                model.predict(
                    X_validation_processed
                )
            )

            probabilities = np.asarray(
                model.predict_proba(
                    X_validation_processed
                )
            )

            if probabilities.ndim != 2:
                raise RuntimeError(
                    "Probability output must be two-dimensional."
                )

            if probabilities.shape[1] != 2:
                raise RuntimeError(
                    "Directional model must produce two probability columns."
                )

            metrics = classification_metrics(
                y_validation.to_numpy(),
                predictions,
                probabilities,
            )

            accuracy = float(
                metrics.accuracy
            )

            fold_result = (
                WalkForwardFoldResult(
                    fold_number=fold_number,
                    train_index=train_index.copy(),
                    validation_index=validation_index.copy(),
                    predictions=predictions.copy(),
                    probabilities=probabilities.copy(),
                    metrics={
                        key: float(value)
                        for key, value
                        in metrics.__dict__.items()
                        if isinstance(
                            value,
                            (
                                int,
                                float,
                                np.number,
                            ),
                        )
                    },
                    accuracy=accuracy,
                    model=model,
                    preprocessor=preprocessor,
                    metadata={
                        "research_only": True,
                        "final_holdout_used": False,
                        "fresh_model": True,
                        "fresh_preprocessor": True,
                        "train_start": str(
                            train_index[0]
                        ),
                        "train_end": str(
                            train_index[-1]
                        ),
                        "validation_start": str(
                            validation_index[0]
                        ),
                        "validation_end": str(
                            validation_index[-1]
                        ),
                    },
                )
            )

            fold_results.append(
                fold_result
            )

            fold_frame = pd.DataFrame(
                {
                    "Actual": y_validation.to_numpy(),
                    "Prediction": predictions,
                    "Probability_Down": probabilities[:, 0],
                    "Probability_Up": probabilities[:, 1],
                    "Fold": fold_number,
                },
                index=validation_index,
            )

            oos_frames.append(
                fold_frame
            )

        if not fold_results:
            raise ValueError(
                "No walk-forward fold satisfied the minimum "
                "training and validation row requirements."
            )

        oos_predictions = pd.concat(
            oos_frames,
            axis=0,
        ).sort_index()

        if oos_predictions.index.has_duplicates:
            raise RuntimeError(
                "Walk-forward OOS predictions contain duplicate timestamps."
            )

        accuracies = np.asarray(
            [
                fold.accuracy
                for fold in fold_results
            ],
            dtype=float,
        )

        mean_accuracy = float(
            np.mean(accuracies)
        )

        median_accuracy = float(
            np.median(accuracies)
        )

        minimum_accuracy = float(
            np.min(accuracies)
        )

        maximum_accuracy = float(
            np.max(accuracies)
        )

        accuracy_std = float(
            np.std(
                accuracies,
                ddof=0,
            )
        )

        candidate_passed = (
            mean_accuracy
            >= self.config.minimum_accuracy
            and minimum_accuracy
            >= self.config.minimum_accuracy
            and accuracy_std
            <= self.config.maximum_accuracy_std
        )

        warnings: list[str] = [
            "Walk-forward results are development evidence only.",
            "The final holdout was not used.",
            "Walk-forward performance does not imply future trading performance.",
        ]

        if not candidate_passed:
            warnings.append(
                "Walk-forward candidate failed one or more "
                "strict research stability gates."
            )

        metadata = {
            "research_only": True,
            "production_ready": False,
            "production_approved": False,
            "final_holdout_used": False,
            "model_selected": False,
            "calibration_fitted": False,
            "threshold_optimization_completed": False,
            "n_splits_requested": self.config.n_splits,
            "folds_completed": len(
                fold_results
            ),
            "model_type": self.config.model_type,
            "minimum_accuracy": self.config.minimum_accuracy,
            "maximum_accuracy_std": (
                self.config.maximum_accuracy_std
            ),
            "gap": self.config.gap,
            "embargo": self.config.embargo,
            "expanding": self.config.expanding,
        }

        return WalkForwardTrainingResult(
            folds=fold_results,
            feature_columns=list(
                dataset.feature_columns
            ),
            target_column=resolved_target,
            oos_predictions=oos_predictions,
            mean_accuracy=mean_accuracy,
            median_accuracy=median_accuracy,
            minimum_accuracy=minimum_accuracy,
            maximum_accuracy=maximum_accuracy,
            accuracy_std=accuracy_std,
            candidate_passed=candidate_passed,
            warnings=warnings,
            metadata=metadata,
        )


def train_walk_forward(
    dataset: ResearchModelDataset,
    *,
    target_column: str | None = None,
    config: WalkForwardTrainingConfig
    | None = None,
) -> WalkForwardTrainingResult:
    """Convenience function for walk-forward training."""

    pipeline = WalkForwardTrainingPipeline(
        config=config
    )

    return pipeline.train(
        dataset,
        target_column=target_column,
    )


def walk_forward_training_summary(
    result: WalkForwardTrainingResult,
) -> dict[str, object]:
    """Return a compact walk-forward summary."""

    if not isinstance(
        result,
        WalkForwardTrainingResult,
    ):
        raise TypeError(
            "result must be WalkForwardTrainingResult."
        )

    return result.summary()


__all__ = [
    "WalkForwardTrainingConfig",
    "WalkForwardFoldResult",
    "WalkForwardTrainingResult",
    "WalkForwardTrainingPipeline",
    "train_walk_forward",
    "walk_forward_training_summary",
]
