"""
Leakage-safe validation framework for AI Swing Analyser.

Provides:

    - Chronological train/validation/test evaluation
    - Walk-forward evaluation
    - Purged validation
    - Embargo support
    - Fold-level metric collection
    - Stability analysis
    - Final validation gate

IMPORTANT:

The final test/holdout set must remain untouched until the
research process is complete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .metrics import (
    classification_metrics,
    regression_metrics,
    range_metrics,
)
from .splitter import (
    TimeSplit,
    chronological_split,
    purged_walk_forward_splits,
    validate_time_split,
)


@dataclass
class FoldResult:
    """Results from one walk-forward validation fold."""

    fold: int

    train_start: pd.Timestamp
    train_end: pd.Timestamp

    test_start: pd.Timestamp
    test_end: pd.Timestamp

    train_samples: int
    test_samples: int

    accuracy: Optional[float] = None
    balanced_accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    roc_auc: Optional[float] = None
    brier_score: Optional[float] = None

    mae: Optional[float] = None
    rmse: Optional[float] = None
    directional_accuracy: Optional[float] = None

    range_coverage: Optional[float] = None
    range_width: Optional[float] = None


@dataclass
class ValidationReport:
    """Complete validation report."""

    folds: List[FoldResult] = field(
        default_factory=list
    )

    mean_accuracy: Optional[float] = None
    median_accuracy: Optional[float] = None
    minimum_accuracy: Optional[float] = None
    accuracy_std: Optional[float] = None

    mean_balanced_accuracy: Optional[float] = None
    mean_f1: Optional[float] = None
    mean_brier_score: Optional[float] = None

    mean_mae: Optional[float] = None
    mean_rmse: Optional[float] = None
    mean_directional_accuracy: Optional[float] = None

    mean_range_coverage: Optional[float] = None
    mean_range_width: Optional[float] = None

    passed_accuracy_gate: bool = False
    passed_stability_gate: bool = False
    passed_sample_gate: bool = False
    passed: bool = False

    notes: List[str] = field(
        default_factory=list
    )


def _safe_mean(
    values: List[Optional[float]],
) -> Optional[float]:
    """Calculate mean while ignoring missing values."""

    valid = [
        value
        for value in values
        if value is not None
        and np.isfinite(value)
    ]

    if not valid:
        return None

    return float(
        np.mean(valid)
    )


def _safe_std(
    values: List[Optional[float]],
) -> Optional[float]:
    """Calculate standard deviation safely."""

    valid = [
        value
        for value in values
        if value is not None
        and np.isfinite(value)
    ]

    if len(valid) < 2:
        return 0.0 if valid else None

    return float(
        np.std(
            valid,
            ddof=1,
        )
    )


def validate_dataset_order(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """
    Verify chronological ordering.

    The last training timestamp must occur before validation,
    and validation before final test.
    """

    if train.empty:
        raise ValueError(
            "Training dataset is empty."
        )

    if validation.empty:
        raise ValueError(
            "Validation dataset is empty."
        )

    if test.empty:
        raise ValueError(
            "Test dataset is empty."
        )

    train_index = pd.DatetimeIndex(
        train.index
    )

    validation_index = pd.DatetimeIndex(
        validation.index
    )

    test_index = pd.DatetimeIndex(
        test.index
    )

    if train_index.max() >= validation_index.min():
        raise ValueError(
            "Training data overlaps or occurs after validation data."
        )

    if validation_index.max() >= test_index.min():
        raise ValueError(
            "Validation data overlaps or occurs after test data."
        )


def chronological_train_validation_test(
    data: pd.DataFrame,
    test_fraction: float = 0.20,
    validation_fraction: float = 0.20,
) -> TimeSplit:
    """
    Create chronological train/validation/test partitions.

    No random shuffling is performed.
    """

    split = chronological_split(
        data,
        train_fraction=(
            1.0
            - test_fraction
            - validation_fraction
        ),
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )

    validate_time_split(
        split
    )

    return split


def calculate_fold_result(
    fold_number: int,
    train_index: pd.DatetimeIndex,
    test_index: pd.DatetimeIndex,
    y_test: np.ndarray,
    predicted_probability: Optional[np.ndarray] = None,
    y_regression: Optional[np.ndarray] = None,
    predicted_return: Optional[np.ndarray] = None,
    actual_range: Optional[np.ndarray] = None,
    predicted_lower: Optional[np.ndarray] = None,
    predicted_upper: Optional[np.ndarray] = None,
) -> FoldResult:
    """
    Convert predictions from one validation fold into metrics.
    """

    result = FoldResult(
        fold=fold_number,
        train_start=train_index.min(),
        train_end=train_index.max(),
        test_start=test_index.min(),
        test_end=test_index.max(),
        train_samples=len(train_index),
        test_samples=len(test_index),
    )

    if (
        predicted_probability is not None
    ):

        metrics = classification_metrics(
            y_test,
            predicted_probability,
        )

        result.accuracy = metrics.accuracy
        result.balanced_accuracy = (
            metrics.balanced_accuracy
        )
        result.precision = metrics.precision
        result.recall = metrics.recall
        result.f1 = metrics.f1
        result.roc_auc = metrics.roc_auc
        result.brier_score = metrics.brier_score

    if (
        y_regression is not None
        and predicted_return is not None
    ):

        metrics = regression_metrics(
            y_regression,
            predicted_return,
        )

        result.mae = metrics.mae
        result.rmse = metrics.rmse
        result.directional_accuracy = (
            metrics.directional_accuracy
        )

    if (
        actual_range is not None
        and predicted_lower is not None
        and predicted_upper is not None
    ):

        metrics = range_metrics(
            actual_range,
            predicted_lower,
            predicted_upper,
        )

        result.range_coverage = (
            metrics.coverage
        )

        result.range_width = (
            metrics.average_width
        )

    return result


def build_validation_report(
    folds: List[FoldResult],
    minimum_accuracy: float = 0.95,
    minimum_test_samples: int = 100,
    maximum_accuracy_std: float = 0.10,
) -> ValidationReport:
    """
    Aggregate fold results and apply validation gates.

    Gates:

        1. Mean accuracy >= minimum_accuracy
        2. Accuracy stability across folds
        3. Sufficient test observations
        4. No automatic production approval
    """

    report = ValidationReport(
        folds=folds
    )

    if not folds:
        report.notes.append(
            "No validation folds were supplied."
        )

        return report

    accuracies = [
        fold.accuracy
        for fold in folds
        if fold.accuracy is not None
    ]

    report.mean_accuracy = (
        _safe_mean(
            accuracies
        )
    )

    report.median_accuracy = (
        float(
            np.median(
                accuracies
            )
        )
        if accuracies
        else None
    )

    report.minimum_accuracy = (
        float(
            np.min(
                accuracies
            )
        )
        if accuracies
        else None
    )

    report.accuracy_std = (
        _safe_std(
            accuracies
        )
    )

    report.mean_balanced_accuracy = (
        _safe_mean(
            [
                fold.balanced_accuracy
                for fold in folds
            ]
        )
    )

    report.mean_f1 = (
        _safe_mean(
            [
                fold.f1
                for fold in folds
            ]
        )
    )

    report.mean_brier_score = (
        _safe_mean(
            [
                fold.brier_score
                for fold in folds
            ]
        )
    )

    report.mean_mae = (
        _safe_mean(
            [
                fold.mae
                for fold in folds
            ]
        )
    )

    report.mean_rmse = (
        _safe_mean(
            [
                fold.rmse
                for fold in folds
            ]
        )
    )

    report.mean_directional_accuracy = (
        _safe_mean(
            [
                fold.directional_accuracy
                for fold in folds
            ]
        )
    )

    report.mean_range_coverage = (
        _safe_mean(
            [
                fold.range_coverage
                for fold in folds
            ]
        )
    )

    report.mean_range_width = (
        _safe_mean(
            [
                fold.range_width
                for fold in folds
            ]
        )
    )

    total_test_samples = sum(
        fold.test_samples
        for fold in folds
    )

    report.passed_sample_gate = (
        total_test_samples
        >= minimum_test_samples
    )

    report.passed_accuracy_gate = (
        report.mean_accuracy is not None
        and report.mean_accuracy
        >= minimum_accuracy
    )

    report.passed_stability_gate = (
        report.accuracy_std is not None
        and report.accuracy_std
        <= maximum_accuracy_std
    )

    report.passed = (
        report.passed_accuracy_gate
        and report.passed_stability_gate
        and report.passed_sample_gate
    )

    if not report.passed_accuracy_gate:
        report.notes.append(
            "Accuracy gate failed."
        )

    if not report.passed_stability_gate:
        report.notes.append(
            "Accuracy stability gate failed."
        )

    if not report.passed_sample_gate:
        report.notes.append(
            "Minimum validation sample gate failed."
        )

    report.notes.append(
        "Passing this validation report does not make the model production-ready."
    )

    return report


def report_to_dataframe(
    report: ValidationReport,
) -> pd.DataFrame:
    """Convert fold results to a dashboard-friendly dataframe."""

    if not report.folds:
        return pd.DataFrame()

    rows = []

    for fold in report.folds:

        rows.append(
            {
                "Fold": fold.fold,
                "Train Start": fold.train_start,
                "Train End": fold.train_end,
                "Test Start": fold.test_start,
                "Test End": fold.test_end,
                "Train Samples": fold.train_samples,
                "Test Samples": fold.test_samples,
                "Accuracy": fold.accuracy,
                "Balanced Accuracy": (
                    fold.balanced_accuracy
                ),
                "Precision": fold.precision,
                "Recall": fold.recall,
                "F1": fold.f1,
                "ROC AUC": fold.roc_auc,
                "Brier Score": fold.brier_score,
                "MAE": fold.mae,
                "RMSE": fold.rmse,
                "Directional Accuracy": (
                    fold.directional_accuracy
                ),
                "Range Coverage": (
                    fold.range_coverage
                ),
                "Range Width": (
                    fold.range_width
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def walk_forward_validation(
    data: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    model_factory: Callable,
    predictor: Callable,
    n_splits: int = 5,
    test_size: int = 100,
    gap: int = 0,
    purge_window: int = 0,
    embargo: int = 0,
) -> ValidationReport:
    """
    Generic walk-forward validation runner.

    Parameters
    ----------
    data:
        Complete feature/target dataframe.

    feature_columns:
        Columns supplied to the model.

    target_column:
        Binary target column.

    model_factory:
        Callable returning a fresh unfitted model.

    predictor:
        Callable with signature:

            predictor(model, X_test) -> probability

    IMPORTANT:

    A NEW model is created for every fold.
    """

    if data.empty:
        raise ValueError(
            "Validation data is empty."
        )

    if not feature_columns:
        raise ValueError(
            "feature_columns cannot be empty."
        )

    missing = [
        column
        for column in feature_columns
        if column not in data.columns
    ]

    if target_column not in data.columns:
        missing.append(
            target_column
        )

    if missing:
        raise ValueError(
            f"Missing validation columns: {missing}"
        )

    clean = data[
        feature_columns
        + [target_column]
    ].copy()

    clean = clean.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(clean) < 200:
        raise ValueError(
            "At least 200 clean observations are recommended "
            "for walk-forward validation."
        )

    splits = list(
        purged_walk_forward_splits(
            clean,
            n_splits=n_splits,
            test_size=test_size,
            gap=gap,
            purge_window=purge_window,
            embargo=embargo,
        )
    )

    folds = []

    for fold_number, (
        train_positions,
        test_positions,
    ) in enumerate(
        splits,
        start=1,
    ):

        train = clean.iloc[
            train_positions
        ]

        test = clean.iloc[
            test_positions
        ]

        if train.empty or test.empty:
            continue

        model = model_factory()

        X_train = train[
            feature_columns
        ]

        y_train = train[
            target_column
        ].to_numpy()

        X_test = test[
            feature_columns
        ]

        y_test = test[
            target_column
        ].to_numpy()

        model.fit(
            X_train,
            y_train,
        )

        probability = predictor(
            model,
            X_test,
        )

        fold = calculate_fold_result(
            fold_number=fold_number,
            train_index=pd.DatetimeIndex(
                train.index
            ),
            test_index=pd.DatetimeIndex(
                test.index
            ),
            y_test=y_test,
            predicted_probability=probability,
        )

        folds.append(
            fold
        )

    return build_validation_report(
        folds
    )


def stability_summary(
    report: ValidationReport,
) -> Dict[str, float]:
    """
    Calculate simple fold-stability statistics.
    """

    accuracies = np.asarray(
        [
            fold.accuracy
            for fold in report.folds
            if fold.accuracy is not None
        ],
        dtype=float,
    )

    if len(accuracies) == 0:
        return {
            "mean_accuracy": np.nan,
            "std_accuracy": np.nan,
            "minimum_accuracy": np.nan,
            "maximum_accuracy": np.nan,
            "accuracy_range": np.nan,
        }

    return {
        "mean_accuracy": float(
            np.mean(accuracies)
        ),
        "std_accuracy": float(
            np.std(
                accuracies,
                ddof=1,
            )
        )
        if len(accuracies) > 1
        else 0.0,
        "minimum_accuracy": float(
            np.min(accuracies)
        ),
        "maximum_accuracy": float(
            np.max(accuracies)
        ),
        "accuracy_range": float(
            np.max(accuracies)
            - np.min(accuracies)
        ),
    }
