"""
AI Swing Analyser — Model Development Orchestrator.

This module manages model development using only the development portion
of a research dataset.

Important research rule:

    FINAL HOLDOUT DATA MUST NOT BE USED FOR MODEL SELECTION.

The final holdout is reserved for the final, frozen evaluation stage.

This module therefore focuses on:
    - development/holdout separation
    - training preprocessing
    - optional feature selection
    - candidate model training
    - validation evaluation
    - reproducible model-development records
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

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
    HoldoutSplit,
    TemporalSplitter,
)
from src.features.selection import (
    FeatureSelectionConfig,
    FeatureSelector,
)


# ---------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class DevelopmentPartitions:
    """
    Development and final holdout partitions.

    The holdout is intentionally kept separate from the model-development
    training/validation workflow.
    """

    development: pd.DataFrame
    holdout: pd.DataFrame

    development_index: pd.Index
    holdout_index: pd.Index

    development_end: pd.Timestamp
    holdout_start: pd.Timestamp

    @property
    def holdout_is_future(self) -> bool:
        return (
            self.development_end
            < self.holdout_start
        )


@dataclass
class ModelDevelopmentResult:
    """
    Result of developing one candidate model.
    """

    model_name: str
    horizon: int
    target_column: str

    feature_columns: list[str]

    preprocessor: SafePreprocessor
    model: DirectionClassifier

    validation_metrics: ClassificationMetrics

    feature_selection: FeatureSelector | None = None

    development_start: pd.Timestamp | None = None
    development_end: pd.Timestamp | None = None

    validation_start: pd.Timestamp | None = None
    validation_end: pd.Timestamp | None = None

    warnings: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def validation_accuracy(self) -> float:
        return float(
            self.validation_metrics.accuracy
        )

    @property
    def passed_95_percent_gate(self) -> bool:
        return (
            self.validation_accuracy
            >= 0.95
        )

    @property
    def candidate(self) -> bool:
        """
        Candidate status is deliberately different from approval.

        A model passing 95% validation accuracy is only a research
        candidate. It still needs final holdout testing, calibration,
        range validation, regime testing, backtesting, robustness
        and formal approval.
        """

        return self.passed_95_percent_gate

    def summary(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "horizon": self.horizon,
            "target_column": self.target_column,
            "feature_count": len(
                self.feature_columns
            ),
            "validation_accuracy": (
                self.validation_accuracy
            ),
            "passed_95_percent_gate": (
                self.passed_95_percent_gate
            ),
            "candidate": self.candidate,
            "development_start": (
                self.development_start
            ),
            "development_end": (
                self.development_end
            ),
            "validation_start": (
                self.validation_start
            ),
            "validation_end": (
                self.validation_end
            ),
            "warnings": list(
                self.warnings
            ),
            "metadata": dict(
                self.metadata
            ),
        }


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------


class ModelDevelopmentOrchestrator:
    """
    Controls leakage-safe model development.

    This class intentionally separates:
        development data
        validation data
        final holdout data

    The final holdout is never passed into feature selection,
    preprocessing fitting, or candidate selection.
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
    # Dataset validation
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_dataset(
        data: pd.DataFrame,
    ) -> None:

        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise TypeError(
                "Dataset must be a pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                "Dataset is empty."
            )

        if not isinstance(
            data.index,
            pd.DatetimeIndex,
        ):
            raise TypeError(
                "Dataset must use DatetimeIndex."
            )

        if data.index.has_duplicates:
            raise ValueError(
                "Dataset contains duplicate timestamps."
            )

        if not data.index.is_monotonic_increasing:
            raise ValueError(
                "Dataset must be chronologically sorted."
            )

    # -----------------------------------------------------------------
    # Target validation
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_target(
        data: pd.DataFrame,
        target_column: str,
    ) -> None:

        if target_column not in data.columns:
            raise ValueError(
                f"Target column '{target_column}' "
                "does not exist."
            )

        target = data[target_column]

        if target.isna().all():
            raise ValueError(
                f"Target '{target_column}' "
                "contains no usable values."
            )

        unique = (
            pd.Series(target)
            .dropna()
            .unique()
        )

        if len(unique) < 2:
            raise ValueError(
                f"Target '{target_column}' "
                "contains fewer than two classes."
            )

    # -----------------------------------------------------------------
    # Feature validation
    # -----------------------------------------------------------------

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
                "Feature columns are missing "
                f"from dataset: {missing}"
            )

        values = data.loc[
            :,
            list(feature_columns),
        ].to_numpy(
            dtype=float,
            na_value=np.nan,
        )

        if np.isinf(values).any():
            raise ValueError(
                "Feature matrix contains infinite values."
            )

    # -----------------------------------------------------------------
    # Development / holdout split
    # -----------------------------------------------------------------

    def split_development_holdout(
        self,
        data: pd.DataFrame,
    ) -> DevelopmentPartitions:

        self._validate_dataset(
            data
        )

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

        split: HoldoutSplit = (
            splitter.holdout_split(
                data
            )
        )

        development = data.loc[
            split.development_index
        ].copy()

        holdout = data.loc[
            split.holdout_index
        ].copy()

        if not (
            split.development_end
            < split.holdout_start
        ):
            raise ValueError(
                "Development and holdout periods "
                "are not temporally separated."
            )

        return DevelopmentPartitions(
            development=development,
            holdout=holdout,
            development_index=(
                development.index
            ),
            holdout_index=(
                holdout.index
            ),
            development_end=(
                development.index.max()
            ),
            holdout_start=(
                holdout.index.min()
            ),
        )

    # -----------------------------------------------------------------
    # Development validation split
    # -----------------------------------------------------------------

    def split_training_validation(
        self,
        development: pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
    ]:

        self._validate_dataset(
            development
        )

        if self.config is not None:
            validation_fraction = (
                self.config.validation.validation_fraction
            )
        else:
            validation_fraction = 0.20

        if not (
            0.0
            < validation_fraction
            < 1.0
        ):
            raise ValueError(
                "Validation fraction must be "
                "between 0 and 1."
            )

        validation_size = int(
            len(development)
            * validation_fraction
        )

        if validation_size <= 0:
            raise ValueError(
                "Validation set would be empty."
            )

        train_end = (
            len(development)
            - validation_size
        )

        training = development.iloc[
            :train_end
        ].copy()

        validation = development.iloc[
            train_end:
        ].copy()

        if training.empty:
            raise ValueError(
                "Training dataset is empty."
            )

        if validation.empty:
            raise ValueError(
                "Validation dataset is empty."
            )

        if (
            training.index.max()
            >= validation.index.min()
        ):
            raise ValueError(
                "Training and validation periods overlap."
            )

        return (
            training,
            validation,
        )

    # -----------------------------------------------------------------
    # Feature selection
    # -----------------------------------------------------------------

    def select_development_features(
        self,
        training: pd.DataFrame,
        feature_columns: Sequence[str],
        target_column: str,
    ) -> FeatureSelector:

        self._validate_features(
            training,
            feature_columns,
        )

        if target_column not in training.columns:
            raise ValueError(
                f"Target '{target_column}' "
                "not found in training data."
            )

        # -------------------------------------------------------------
        # Feature selection is deliberately performed only on the
        # training partition.
        #
        # The target is validated above, but it is NOT included in the
        # feature-selection matrix. This prevents the selector from
        # using future target information as a feature.
        # -------------------------------------------------------------

        X = training.loc[
            :,
            list(feature_columns),
        ].copy()

        selector = FeatureSelector(
            config=self.feature_selection_config
        )

        selector.fit(
            X
        )

        if not selector.selected_features:
            raise ValueError(
                "Feature selection returned no features."
            )

        return selector

    # -----------------------------------------------------------------
    # Fit candidate
    # -----------------------------------------------------------------

    def fit_candidate(
        self,
        data: pd.DataFrame,
        feature_columns: Sequence[str],
        target_column: str,
        horizon: int,
        model_name: str = "gradient_boosting",
        apply_feature_selection: bool = True,
    ) -> ModelDevelopmentResult:

        self._validate_dataset(
            data
        )

        self._validate_target(
            data,
            target_column,
        )

        self._validate_features(
            data,
            feature_columns,
        )

        training, validation = (
            self.split_training_validation(
                data
            )
        )

        # -------------------------------------------------------------
        # Remove rows without usable targets.
        #
        # This is done independently for each partition so future
        # observations cannot influence preprocessing statistics.
        # -------------------------------------------------------------

        training = training.loc[
            training[target_column].notna()
        ].copy()

        validation = validation.loc[
            validation[target_column].notna()
        ].copy()

        if training.empty:
            raise ValueError(
                "No usable training observations remain."
            )

        if validation.empty:
            raise ValueError(
                "No usable validation observations remain."
            )

        selected_features = list(
            feature_columns
        )

        feature_selection_result = None

        # -------------------------------------------------------------
        # Feature selection MUST only see training data.
        # -------------------------------------------------------------

        if apply_feature_selection:
            feature_selection_result = (
                self.select_development_features(
                    training,
                    feature_columns,
                    target_column,
                )
            )

            selected_features = list(
                feature_selection_result.selected_features
            )

        if not selected_features:
            raise ValueError(
                "No features remain after feature selection."
            )

        # -------------------------------------------------------------
        # Fit preprocessing ONLY on training data.
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
        # Train a fresh classifier.
        # -------------------------------------------------------------

        classifier_config = (
            self.classifier_config
        )

        if model_name:
            classifier_config = (
                classifier_config.with_model(
                    model_name
                )
                if hasattr(
                    classifier_config,
                    "with_model",
                )
                else classifier_config
            )

        model = DirectionClassifier(
            config=classifier_config
        )

        model.fit(
            X_train_processed,
            y_train,
        )

        # -------------------------------------------------------------
        # Validation predictions.
        # -------------------------------------------------------------

        predictions = model.predict(
            X_validation_processed
        )

        metrics = classification_metrics(
            y_validation.to_numpy(),
            predictions,
        )

        warnings: list[str] = []

        if metrics.accuracy < 0.95:
            warnings.append(
                "Validation accuracy is below the strict "
                "95% research gate."
            )

        if len(training) < 100:
            warnings.append(
                "Training sample size is small."
            )

        if len(validation) < 50:
            warnings.append(
                "Validation sample size is small."
            )

        metadata = {
            "research_only": True,
            "final_holdout_used": False,
            "feature_selection_used": (
                apply_feature_selection
            ),
            "preprocessor_fitted_on": "training_only",
            "model_name": model_name,
            "horizon": int(horizon),
        }

        if feature_selection_result is not None:
            metadata[
                "selected_feature_count"
            ] = len(
                feature_selection_result.selected_features
            )

            metadata[
                "dropped_feature_count"
            ] = len(
                feature_selection_result.dropped_features_
            )

        return ModelDevelopmentResult(
            model_name=model_name,
            horizon=int(horizon),
            target_column=target_column,
            feature_columns=selected_features,
            preprocessor=preprocessor,
            model=model,
            validation_metrics=metrics,
            feature_selection=(
                feature_selection_result
            ),
            development_start=(
                training.index.min()
            ),
            development_end=(
                training.index.max()
            ),
            validation_start=(
                validation.index.min()
            ),
            validation_end=(
                validation.index.max()
            ),
            warnings=warnings,
            metadata=metadata,
        )


__all__ = [
    "DevelopmentPartitions",
    "ModelDevelopmentResult",
    "ModelDevelopmentOrchestrator",
]
