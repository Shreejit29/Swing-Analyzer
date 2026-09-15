"""
End-to-end research training pipeline for AI Swing Analyser.

Connects:

    Features
        ↓
    Targets
        ↓
    Chronological split
        ↓
    Preprocessing
        ↓
    Model
        ↓
    Validation
        ↓
    Final holdout evaluation

IMPORTANT:

This module is research-only.

It does NOT automatically approve a model for production.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .classifier import (
    ClassifierConfig,
    DirectionClassifier,
)
from .metrics import (
    ClassificationMetrics,
    classification_metrics,
)
from .preprocessing import (
    PreprocessorConfig,
    SafePreprocessor,
    ensure_numeric_features,
    remove_infinite_values,
)
from .splitter import (
    TimeSplit,
    chronological_split,
    validate_time_split,
)
from .targets import (
    direction_target,
)


@dataclass
class TrainingConfig:
    """Configuration for one model-training experiment."""

    model_type: str = "gradient_boosting"

    train_fraction: float = 0.60

    validation_fraction: float = 0.20

    test_fraction: float = 0.20

    probability_threshold: float = 0.50

    random_state: int = 42

    minimum_training_samples: int = 100

    minimum_validation_samples: int = 50

    minimum_test_samples: int = 50


@dataclass
class DatasetPartitions:
    """Chronological train/validation/test datasets."""

    train: pd.DataFrame

    validation: pd.DataFrame

    test: pd.DataFrame


@dataclass
class TrainingResult:
    """Result of a complete research training run."""

    model: DirectionClassifier

    preprocessor: SafePreprocessor

    feature_columns: list[str]

    target_column: str

    partitions: DatasetPartitions

    validation_metrics: ClassificationMetrics

    test_metrics: ClassificationMetrics

    validation_probability: np.ndarray

    test_probability: np.ndarray

    validation_prediction: np.ndarray

    test_prediction: np.ndarray


def _validate_training_config(
    config: TrainingConfig,
) -> None:
    """Validate training configuration."""

    fractions = [
        config.train_fraction,
        config.validation_fraction,
        config.test_fraction,
    ]

    if any(
        fraction <= 0
        or fraction >= 1
        for fraction in fractions
    ):
        raise ValueError(
            "All dataset fractions must be between 0 and 1."
        )

    if not np.isclose(
        sum(fractions),
        1.0,
    ):
        raise ValueError(
            "train_fraction + validation_fraction + "
            "test_fraction must equal 1."
        )

    if not 0.0 < (
        config.probability_threshold
    ) < 1.0:
        raise ValueError(
            "probability_threshold must be between 0 and 1."
        )

    if (
        config.minimum_training_samples
        < 1
    ):
        raise ValueError(
            "minimum_training_samples must be positive."
        )

    if (
        config.minimum_validation_samples
        < 1
    ):
        raise ValueError(
            "minimum_validation_samples must be positive."
        )

    if (
        config.minimum_test_samples
        < 1
    ):
        raise ValueError(
            "minimum_test_samples must be positive."
        )


def prepare_dataset(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    """
    Prepare model-ready dataframe.

    Future target columns are never included as features.
    """

    if data.empty:
        raise ValueError(
            "Training dataframe is empty."
        )

    if not feature_columns:
        raise ValueError(
            "feature_columns cannot be empty."
        )

    missing_features = [
        column
        for column in feature_columns
        if column not in data.columns
    ]

    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    if missing_features:
        raise ValueError(
            "Missing feature columns: "
            f"{missing_features}"
        )

    # Explicitly reject future-looking feature names.
    suspicious_features = [
        column
        for column in feature_columns
        if (
            "future" in column.lower()
            or "target" in column.lower()
            or "direction" in column.lower()
        )
    ]

    if suspicious_features:
        raise ValueError(
            "Potential target leakage detected in feature list: "
            f"{suspicious_features}"
        )

    selected = data[
        feature_columns
        + [target_column]
    ].copy()

    selected = remove_infinite_values(
        selected
    )

    selected = selected.dropna(
        subset=[target_column]
    )

    selected = ensure_numeric_features(
        selected,
        feature_columns,
    )

    selected = selected.sort_index()

    if selected.index.has_duplicates:
        raise ValueError(
            "Dataset contains duplicate timestamps."
        )

    return selected


def split_dataset(
    data: pd.DataFrame,
    config: TrainingConfig,
) -> DatasetPartitions:
    """
    Split dataset chronologically.

    No random shuffling.
    """

    split = chronological_split(
        data,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
    )

    validate_time_split(
        split
    )

    if len(split.train) < (
        config.minimum_training_samples
    ):
        raise ValueError(
            "Training partition is too small: "
            f"{len(split.train)}"
        )

    if len(split.validation) < (
        config.minimum_validation_samples
    ):
        raise ValueError(
            "Validation partition is too small: "
            f"{len(split.validation)}"
        )

    if len(split.test) < (
        config.minimum_test_samples
    ):
        raise ValueError(
            "Test partition is too small: "
            f"{len(split.test)}"
        )

    return DatasetPartitions(
        train=split.train,
        validation=split.validation,
        test=split.test,
    )


def create_classifier(
    config: TrainingConfig,
) -> DirectionClassifier:
    """Create a fresh direction classifier."""

    classifier_config = ClassifierConfig(
        model_type=config.model_type,
        random_state=config.random_state,
    )

    return DirectionClassifier(
        config=classifier_config
    )


def fit_preprocessor(
    train: pd.DataFrame,
    feature_columns: list[str],
) -> SafePreprocessor:
    """
    Fit preprocessing ONLY on training data.
    """

    preprocessor = SafePreprocessor(
        PreprocessorConfig()
    )

    preprocessor.fit(
        train[
            feature_columns
        ]
    )

    return preprocessor


def train_classifier(
    train: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    config: TrainingConfig,
) -> tuple[
    DirectionClassifier,
    SafePreprocessor,
]:
    """
    Fit preprocessing and classifier using training data only.
    """

    preprocessor = fit_preprocessor(
        train,
        feature_columns,
    )

    X_train = (
        preprocessor.transform(
            train[
                feature_columns
            ]
        )
    )

    y_train = train[
        target_column
    ].to_numpy()

    model = create_classifier(
        config
    )

    model.fit(
        X_train,
        y_train,
    )

    return (
        model,
        preprocessor,
    )


def predict_dataset(
    model: DirectionClassifier,
    preprocessor: SafePreprocessor,
    data: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    """Generate positive-class probabilities."""

    X = preprocessor.transform(
        data[
            feature_columns
        ]
    )

    probability = (
        model.predict_up_probability(
            X
        )
    )

    probability = np.asarray(
        probability,
        dtype=float,
    ).reshape(-1)

    if not np.isfinite(
        probability
    ).all():
        raise ValueError(
            "Model generated non-finite probabilities."
        )

    return np.clip(
        probability,
        0.0,
        1.0,
    )


def train_research_model(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    config: Optional[
        TrainingConfig
    ] = None,
) -> TrainingResult:
    """
    Train and evaluate a direction model.

    Workflow:

        1. Prepare dataset
        2. Chronological split
        3. Fit preprocessing on train only
        4. Fit model on train only
        5. Evaluate validation
        6. Evaluate final test

    The test set is evaluated but NEVER used to train or
    optimize the model.
    """

    config = (
        config
        if config is not None
        else TrainingConfig()
    )

    _validate_training_config(
        config
    )

    prepared = prepare_dataset(
        data=data,
        feature_columns=feature_columns,
        target_column=target_column,
    )

    partitions = split_dataset(
        prepared,
        config,
    )

    (
        model,
        preprocessor,
    ) = train_classifier(
        train=partitions.train,
        feature_columns=feature_columns,
        target_column=target_column,
        config=config,
    )

    validation_probability = (
        predict_dataset(
            model=model,
            preprocessor=preprocessor,
            data=partitions.validation,
            feature_columns=feature_columns,
        )
    )

    test_probability = (
        predict_dataset(
            model=model,
            preprocessor=preprocessor,
            data=partitions.test,
            feature_columns=feature_columns,
        )
    )

    validation_target = (
        partitions.validation[
            target_column
        ].to_numpy()
    )

    test_target = (
        partitions.test[
            target_column
        ].to_numpy()
    )

    validation_metrics = (
        classification_metrics(
            validation_target,
            validation_probability,
            threshold=(
                config.probability_threshold
            ),
        )
    )

    test_metrics = (
        classification_metrics(
            test_target,
            test_probability,
            threshold=(
                config.probability_threshold
            ),
        )
    )

    validation_prediction = (
        validation_probability
        >= config.probability_threshold
    ).astype(int)

    test_prediction = (
        test_probability
        >= config.probability_threshold
    ).astype(int)

    return TrainingResult(
        model=model,
        preprocessor=preprocessor,
        feature_columns=list(
            feature_columns
        ),
        target_column=target_column,
        partitions=partitions,
        validation_metrics=(
            validation_metrics
        ),
        test_metrics=test_metrics,
        validation_probability=(
            validation_probability
        ),
        test_probability=(
            test_probability
        ),
        validation_prediction=(
            validation_prediction
        ),
        test_prediction=(
            test_prediction
        ),
    )


def validation_summary(
    result: TrainingResult,
) -> pd.DataFrame:
    """
    Create compact validation-vs-test comparison.
    """

    validation = (
        result.validation_metrics
    )

    test = (
        result.test_metrics
    )

    return pd.DataFrame(
        [
            {
                "Dataset": "Validation",
                "Samples": validation.samples,
                "Accuracy": validation.accuracy,
                "Balanced Accuracy": (
                    validation.balanced_accuracy
                ),
                "Precision": validation.precision,
                "Recall": validation.recall,
                "F1": validation.f1,
                "ROC AUC": validation.roc_auc,
                "Brier Score": validation.brier_score,
            },
            {
                "Dataset": "Final Test",
                "Samples": test.samples,
                "Accuracy": test.accuracy,
                "Balanced Accuracy": (
                    test.balanced_accuracy
                ),
                "Precision": test.precision,
                "Recall": test.recall,
                "F1": test.f1,
                "ROC AUC": test.roc_auc,
                "Brier Score": test.brier_score,
            },
        ]
    )


def validation_gap(
    result: TrainingResult,
) -> dict:
    """
    Measure validation-to-test degradation.

    Large degradation can indicate overfitting or instability.
    """

    validation_accuracy = (
        result.validation_metrics.accuracy
    )

    test_accuracy = (
        result.test_metrics.accuracy
    )

    validation_brier = (
        result.validation_metrics.brier_score
    )

    test_brier = (
        result.test_metrics.brier_score
    )

    return {
        "validation_accuracy": (
            validation_accuracy
        ),
        "test_accuracy": (
            test_accuracy
        ),
        "accuracy_gap": (
            validation_accuracy
            - test_accuracy
        ),
        "validation_brier": (
            validation_brier
        ),
        "test_brier": (
            test_brier
        ),
        "brier_gap": (
            test_brier
            - validation_brier
        ),
    }


def model_is_research_candidate(
    result: TrainingResult,
    minimum_accuracy: float = 0.95,
    maximum_accuracy_gap: float = 0.10,
) -> bool:
    """
    Determine whether a model deserves further research.

    This is NOT production approval.

    Both validation and final test accuracy must pass the
    minimum threshold, and validation-to-test degradation
    must remain within the specified limit.
    """

    validation_accuracy = (
        result.validation_metrics.accuracy
    )

    test_accuracy = (
        result.test_metrics.accuracy
    )

    gap = (
        validation_accuracy
        - test_accuracy
    )

    return bool(
        validation_accuracy
        >= minimum_accuracy
        and test_accuracy
        >= minimum_accuracy
        and gap
        <= maximum_accuracy_gap
    )
