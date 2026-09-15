"""
AI Swing Analyser — Walk-Forward Research Engine.

Runs leakage-safe, chronological walk-forward experiments.

For every fold:

    Past data
       ↓
    Feature selection
       ↓
    Fit preprocessing
       ↓
    Fit model
       ↓
    Future validation window
       ↓
    Out-of-sample prediction

The final holdout is never used by this engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.features.selection import (
    FeatureSelectionConfig,
    select_features,
)
from src.models.classifier import (
    ClassifierConfig,
    DirectionClassifier,
)
from src.models.metrics import (
    ClassificationMetrics,
    classification_metrics,
)
from src.models.preprocessing import (
    PreprocessorConfig,
    SafePreprocessor,
)
from src.research.config import ResearchPipelineConfig
from src.research.temporal_split import (
    TemporalSplit,
    TemporalSplitter,
)


# ---------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------


@dataclass
class WalkForwardFoldResult:
    """
    Result from one walk-forward fold.
    """

    fold_number: int

    train_start: pd.Timestamp
    train_end: pd.Timestamp

    validation_start: pd.Timestamp
    validation_end: pd.Timestamp

    train_samples: int
    validation_samples: int

    validation_metrics: ClassificationMetrics

    predictions: np.ndarray
    actuals: np.ndarray

    validation_index: pd.Index

    selected_features: list[str]

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def accuracy(self) -> float:
        return float(
            self.validation_metrics.accuracy
        )

    def summary(self) -> dict[str, Any]:
        return {
            "fold": self.fold_number,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "train_samples": self.train_samples,
            "validation_samples": self.validation_samples,
            "accuracy": self.accuracy,
            "feature_count": len(
                self.selected_features
            ),
            "metadata": dict(
                self.metadata
            ),
        }


@dataclass
class WalkForwardResearchResult:
    """
    Aggregate walk-forward research result.
    """

    folds: list[WalkForwardFoldResult]

    model_name: str
    horizon: int
    target_column: str

    out_of_sample_predictions: pd.DataFrame

    mean_accuracy: float
    median_accuracy: float
    minimum_accuracy: float
    accuracy_std: float

    final_holdout_used: bool = False

    warnings: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def fold_count(self) -> int:
        return len(self.folds)

    @property
    def passed_95_percent_gate(self) -> bool:
        return (
            self.mean_accuracy
            >= 0.95
            and self.minimum_accuracy
            >= 0.95
        )

    @property
    def stability_passed(self) -> bool:
        return (
            self.accuracy_std
            <= 0.10
        )

    @property
    def research_candidate(self) -> bool:
        """
        This is only a research candidate.

        It is NOT production approval.
        """

        return (
            self.passed_95_percent_gate
            and self.stability_passed
        )

    def summary(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "horizon": self.horizon,
            "target_column": self.target_column,
            "fold_count": self.fold_count,
            "mean_accuracy": self.mean_accuracy,
            "median_accuracy": self.median_accuracy,
            "minimum_accuracy": self.minimum_accuracy,
            "accuracy_std": self.accuracy_std,
            "passed_95_percent_gate": (
                self.passed_95_percent_gate
            ),
            "stability_passed": (
                self.stability_passed
            ),
            "research_candidate": (
                self.research_candidate
            ),
            "final_holdout_used": (
                self.final_holdout_used
            ),
            "warnings": list(
                self.warnings
            ),
            "metadata": dict(
                self.metadata
            ),
        }


# ---------------------------------------------------------------------
# Research engine
# ---------------------------------------------------------------------


class WalkForwardResearchEngine:
    """
    Leakage-safe walk-forward research engine.
    """

    def __init__(
        self,
        config: ResearchPipelineConfig | None = None,
        preprocessor_config: PreprocessorConfig | None = None,
        classifier_config: ClassifierConfig | None = None,
        feature_selection_config: (
            FeatureSelectionConfig | None
        ) = None,
    ) -> None:

        self.config = config

        self.preprocessor_config = (
            preprocessor_config
            if preprocessor_config is not None
            else PreprocessorConfig()
        )

        self.classifier_config = (
            classifier_config
            if classifier_config is not None
            else ClassifierConfig()
        )

        self.feature_selection_config = (
            feature_selection_config
            if feature_selection_config is not None
            else FeatureSelectionConfig()
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
                f"Missing features: {missing}"
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
                "Target contains no usable observations."
            )

        if target.nunique() < 2:
            raise ValueError(
                "Target must contain at least two classes."
            )

    # -----------------------------------------------------------------
    # Fold preparation
    # -----------------------------------------------------------------

    def _prepare_fold(
        self,
        data: pd.DataFrame,
        split: TemporalSplit,
        feature_columns: Sequence[str],
        target_column: str,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
    ]:

        training = data.loc[
            split.train_index
        ].copy()

        validation = data.loc[
            split.validation_index
        ].copy()

        training = training.loc[
            training[
                target_column
            ].notna()
        ]

        validation = validation.loc[
            validation[
                target_column
            ].notna()
        ]

        if training.empty:
            raise ValueError(
                "Walk-forward training fold is empty."
            )

        if validation.empty:
            raise ValueError(
                "Walk-forward validation fold is empty."
            )

        if (
            training.index.max()
            >= validation.index.min()
        ):
            raise ValueError(
                "Walk-forward fold contains temporal overlap."
            )

        return (
            training,
            validation,
        )

    # -----------------------------------------------------------------
    # Single fold
    # -----------------------------------------------------------------

    def _run_fold(
        self,
        fold_number: int,
        data: pd.DataFrame,
        split: TemporalSplit,
        feature_columns: Sequence[str],
        target_column: str,
        model_name: str,
    ) -> WalkForwardFoldResult:

        training, validation = (
            self._prepare_fold(
                data,
                split,
                feature_columns,
                target_column,
            )
        )

        # -------------------------------------------------------------
        # Feature selection on training only.
        # -------------------------------------------------------------

        selection = select_features(
            training.loc[
                :,
                list(feature_columns),
            ],
            training[
                target_column
            ],
            config=self.feature_selection_config,
        )

        selected_features = list(
            selection.selected_features
        )

        if not selected_features:
            raise ValueError(
                "Feature selection returned no features."
            )

        # -------------------------------------------------------------
        # Fit preprocessing on training only.
        # -------------------------------------------------------------

        preprocessor = SafePreprocessor(
            self.preprocessor_config
        )

        X_train = training.loc[
            :,
            selected_features,
        ]

        X_validation = validation.loc[
            :,
            selected_features,
        ]

        y_train = training[
            target_column
        ]

        y_validation = validation[
            target_column
        ]

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

        # -------------------------------------------------------------
        # Fresh model for every fold.
        # -------------------------------------------------------------

        model = DirectionClassifier(
            config=self.classifier_config
        )

        model.fit(
            X_train_processed,
            y_train,
        )

        predictions = model.predict(
            X_validation_processed
        )

        metrics = classification_metrics(
            y_validation.to_numpy(),
            predictions,
        )

        return WalkForwardFoldResult(
            fold_number=fold_number,
            train_start=training.index.min(),
            train_end=training.index.max(),
            validation_start=validation.index.min(),
            validation_end=validation.index.max(),
            train_samples=len(training),
            validation_samples=len(validation),
            validation_metrics=metrics,
            predictions=np.asarray(
                predictions
            ),
            actuals=np.asarray(
                y_validation
            ),
            validation_index=validation.index.copy(),
            selected_features=selected_features,
            metadata={
                "fresh_model": True,
                "fresh_preprocessor": True,
                "feature_selection_training_only": True,
                "final_holdout_used": False,
                "model_name": model_name,
            },
        )

    # -----------------------------------------------------------------
    # Main research run
    # -----------------------------------------------------------------

    def run(
        self,
        data: pd.DataFrame,
        feature_columns: Sequence[str],
        target_column: str,
        horizon: int,
        model_name: str = "gradient_boosting",
    ) -> WalkForwardResearchResult:

        self._validate_dataframe(
            data
        )

        self._validate_features(
            data,
            feature_columns,
        )

        self._validate_target(
            data,
            target_column,
        )

        if horizon <= 0:
            raise ValueError(
                "horizon must be positive."
            )

        # -------------------------------------------------------------
        # IMPORTANT:
        #
        # First reserve the final holdout.
        #
        # Walk-forward research only operates on the development period.
        # -------------------------------------------------------------

        if self.config is not None:
            validation_config = (
                self.config.validation
            )
        else:
            from src.research.config import (
                ResearchValidationConfig,
            )

            validation_config = (
                ResearchValidationConfig()
            )

        splitter = TemporalSplitter(
            validation_config
        )

        holdout = splitter.holdout_split(
            data
        )

        development = data.loc[
            holdout.development_index
        ].copy()

        if development.empty:
            raise ValueError(
                "Development dataset is empty."
            )

        # -------------------------------------------------------------
        # Generate walk-forward folds.
        # -------------------------------------------------------------

        development_splits = list(
            splitter.walk_forward(
                development
            )
        )

        if not development_splits:
            raise ValueError(
                "No valid walk-forward folds were created."
            )

        fold_results: list[
            WalkForwardFoldResult
        ] = []

        errors: list[str] = []

        for fold_number, split in enumerate(
            development_splits,
            start=1,
        ):

            try:
                fold_result = self._run_fold(
                    fold_number=fold_number,
                    data=development,
                    split=split,
                    feature_columns=feature_columns,
                    target_column=target_column,
                    model_name=model_name,
                )

                fold_results.append(
                    fold_result
                )

            except Exception as exc:
                errors.append(
                    f"Fold {fold_number}: {exc}"
                )

        if not fold_results:
            raise RuntimeError(
                "All walk-forward folds failed.\n"
                + "\n".join(errors)
            )

        # -------------------------------------------------------------
        # Aggregate predictions.
        #
        # Exact validation timestamps are preserved.
        # -------------------------------------------------------------

        prediction_frames: list[
            pd.DataFrame
        ] = []

        for fold in fold_results:

            frame = pd.DataFrame(
                {
                    "Prediction": fold.predictions,
                    "Actual": fold.actuals,
                    "Fold": fold.fold_number,
                },
                index=fold.validation_index,
            )

            prediction_frames.append(
                frame
            )

        out_of_sample = pd.concat(
            prediction_frames
        ).sort_index()

        # Guard against accidental overlap.
        if out_of_sample.index.has_duplicates:
            raise ValueError(
                "Walk-forward validation windows overlap."
            )

        accuracies = np.asarray(
            [
                fold.accuracy
                for fold in fold_results
            ],
            dtype=float,
        )

        warnings = list(
            errors
        )

        if accuracies.mean() < 0.95:
            warnings.append(
                "Mean walk-forward accuracy is below "
                "the strict 95% research gate."
            )

        if accuracies.min() < 0.95:
            warnings.append(
                "At least one walk-forward fold is below "
                "the strict 95% research gate."
            )

        if accuracies.std() > 0.10:
            warnings.append(
                "Walk-forward accuracy is unstable across folds."
            )

        warnings.append(
            "Final holdout was reserved and excluded from "
            "walk-forward candidate development."
        )

        return WalkForwardResearchResult(
            folds=fold_results,
            model_name=model_name,
            horizon=int(horizon),
            target_column=target_column,
            out_of_sample_predictions=out_of_sample,
            mean_accuracy=float(
                accuracies.mean()
            ),
            median_accuracy=float(
                np.median(accuracies)
            ),
            minimum_accuracy=float(
                accuracies.min()
            ),
            accuracy_std=float(
                accuracies.std()
            ),
            final_holdout_used=False,
            warnings=warnings,
            metadata={
                "research_only": True,
                "final_holdout_reserved": True,
                "fresh_model_each_fold": True,
                "fresh_preprocessor_each_fold": True,
                "feature_selection_training_only": True,
                "fold_count": len(
                    fold_results
                ),
            },
        )


# ---------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------


def run_walk_forward_research(
    data: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    horizon: int,
    model_name: str = "gradient_boosting",
    config: ResearchPipelineConfig | None = None,
) -> WalkForwardResearchResult:

    engine = WalkForwardResearchEngine(
        config=config
    )

    return engine.run(
        data=data,
        feature_columns=feature_columns,
        target_column=target_column,
        horizon=horizon,
        model_name=model_name,
    )


__all__ = [
    "WalkForwardFoldResult",
    "WalkForwardResearchResult",
    "WalkForwardResearchEngine",
    "run_walk_forward_research",
]
