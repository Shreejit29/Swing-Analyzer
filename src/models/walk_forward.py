"""
Leakage-safe walk-forward training and evaluation.

This module provides the production research implementation of
walk-forward validation for AI Swing Analyser.

Key principles
--------------
1. Never shuffle temporal data.
2. Never fit preprocessing on future observations.
3. Fit a fresh model inside every fold.
4. Fit preprocessing using training data only.
5. Validate strictly after the training period.
6. Support a purge gap between training and validation.
7. Support embargo periods.
8. Store fold-level predictions and metrics.
9. Never use the final untouched holdout during model selection.

This module is intentionally independent of the Streamlit application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import classification_metrics
from .preprocessing import (
    PreprocessorConfig,
    SafePreprocessor,
    ensure_numeric_features,
)
from .splitter import (
    purged_walk_forward_splits,
)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass
class WalkForwardConfig:
    """
    Configuration for leakage-safe walk-forward validation.
    """

    n_splits: int = 5

    train_size: Optional[int] = None

    validation_size: Optional[int] = None

    gap: int = 0

    embargo: int = 0

    expanding: bool = True

    min_train_samples: int = 100

    min_validation_samples: int = 30

    refit_model_each_fold: bool = True

    refit_preprocessor_each_fold: bool = True

    random_state: int = 42

    minimum_accuracy: float = 0.95

    feature_columns: Optional[
        Sequence[str]
    ] = None

    def __post_init__(self) -> None:
        if self.n_splits < 1:
            raise ValueError(
                "n_splits must be at least 1."
            )

        if self.train_size is not None:
            if self.train_size < 1:
                raise ValueError(
                    "train_size must be positive."
                )

        if self.validation_size is not None:
            if self.validation_size < 1:
                raise ValueError(
                    "validation_size must be positive."
                )

        if self.gap < 0:
            raise ValueError(
                "gap cannot be negative."
            )

        if self.embargo < 0:
            raise ValueError(
                "embargo cannot be negative."
            )

        if self.min_train_samples < 1:
            raise ValueError(
                "min_train_samples must be positive."
            )

        if self.min_validation_samples < 1:
            raise ValueError(
                "min_validation_samples must be positive."
            )

        if not 0.0 <= self.minimum_accuracy <= 1.0:
            raise ValueError(
                "minimum_accuracy must be between 0 and 1."
            )


# ----------------------------------------------------------------------
# Fold result
# ----------------------------------------------------------------------


@dataclass
class WalkForwardFold:
    """
    Results from one walk-forward fold.
    """

    fold_number: int

    train_start: pd.Timestamp

    train_end: pd.Timestamp

    validation_start: pd.Timestamp

    validation_end: pd.Timestamp

    train_samples: int

    validation_samples: int

    metrics: Dict[str, float]

    y_true: np.ndarray

    y_probability: np.ndarray

    y_pred: np.ndarray

    feature_names: Sequence[str]

    model_name: str

    preprocessing_fitted_on: str

    leakage_check_passed: bool

    notes: List[str] = field(
        default_factory=list
    )

    @property
    def accuracy(self) -> float:
        return float(
            self.metrics.get(
                "accuracy",
                0.0,
            )
        )

    @property
    def passed(self) -> bool:
        return self.accuracy >= 0.95


@dataclass
class WalkForwardResult:
    """
    Complete walk-forward validation result.
    """

    folds: List[WalkForwardFold]

    config: WalkForwardConfig

    mean_accuracy: float

    median_accuracy: float

    minimum_accuracy: float

    maximum_accuracy: float

    accuracy_std: float

    passed_accuracy_gate: bool

    leakage_checks_passed: bool

    model_name: str

    feature_names: Sequence[str]

    notes: List[str] = field(
        default_factory=list
    )

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def passed(self) -> bool:
        """
        Core walk-forward pass.

        This is still not production approval.
        """

        return (
            self.passed_accuracy_gate
            and self.leakage_checks_passed
            and self.n_folds > 0
        )

    def fold_table(self) -> pd.DataFrame:
        """Return fold-level metrics as a DataFrame."""

        records: List[Dict[str, Any]] = []

        for fold in self.folds:
            record: Dict[str, Any] = {
                "fold": fold.fold_number,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "validation_start": fold.validation_start,
                "validation_end": fold.validation_end,
                "train_samples": fold.train_samples,
                "validation_samples": fold.validation_samples,
                "accuracy": fold.metrics.get(
                    "accuracy"
                ),
                "balanced_accuracy": fold.metrics.get(
                    "balanced_accuracy"
                ),
                "precision": fold.metrics.get(
                    "precision"
                ),
                "recall": fold.metrics.get(
                    "recall"
                ),
                "f1": fold.metrics.get(
                    "f1"
                ),
                "roc_auc": fold.metrics.get(
                    "roc_auc"
                ),
                "brier_score": fold.metrics.get(
                    "brier_score"
                ),
                "log_loss": fold.metrics.get(
                    "log_loss"
                ),
                "leakage_check_passed": (
                    fold.leakage_check_passed
                ),
            }

            records.append(record)

        return pd.DataFrame(records)

    def summary(self) -> Dict[str, Any]:
        """Return a compact walk-forward summary."""

        return {
            "model": self.model_name,
            "folds": self.n_folds,
            "mean_accuracy": self.mean_accuracy,
            "median_accuracy": self.median_accuracy,
            "minimum_accuracy": self.minimum_accuracy,
            "maximum_accuracy": self.maximum_accuracy,
            "accuracy_std": self.accuracy_std,
            "passed_accuracy_gate": (
                self.passed_accuracy_gate
            ),
            "leakage_checks_passed": (
                self.leakage_checks_passed
            ),
            "passed": self.passed,
        }


# ----------------------------------------------------------------------
# Main engine
# ----------------------------------------------------------------------


class WalkForwardValidator:
    """
    Leakage-safe walk-forward validator.

    A new model and a new preprocessor are fitted for each fold unless
    explicitly configured otherwise.

    The validator does not modify the input DataFrame.
    """

    def __init__(
        self,
        config: Optional[WalkForwardConfig] = None,
        preprocessor_config: Optional[
            PreprocessorConfig
        ] = None,
    ) -> None:
        self.config = (
            config
            or WalkForwardConfig()
        )

        self.preprocessor_config = (
            preprocessor_config
            or PreprocessorConfig()
        )

    # ------------------------------------------------------------------
    # Classification validation
    # ------------------------------------------------------------------

    def validate_classifier(
        self,
        data: pd.DataFrame,
        *,
        target_column: str,
        model_factory: Callable[[], Any],
        model_name: str = "classifier",
        feature_columns: Optional[
            Sequence[str]
        ] = None,
    ) -> WalkForwardResult:
        """
        Execute leakage-safe walk-forward classification.

        Parameters
        ----------
        data:
            Chronologically indexed research dataset.

        target_column:
            Binary target column.

        model_factory:
            Function returning a fresh unfitted classifier.

        model_name:
            Name used in reports.

        feature_columns:
            Explicit feature schema. If omitted, numeric columns are
            inferred after removing the target.
        """

        frame = self._prepare_data(
            data=data,
            target_column=target_column,
            feature_columns=feature_columns,
        )

        features = self._resolve_features(
            frame,
            target_column,
            feature_columns,
        )

        target = frame[target_column]

        splits = self._create_splits(
            frame
        )

        folds: List[
            WalkForwardFold
        ] = []

        for fold_number, (
            train_indices,
            validation_indices,
        ) in enumerate(
            splits,
            start=1,
        ):
            fold = self._run_classifier_fold(
                frame=frame,
                target=target,
                features=features,
                target_column=target_column,
                train_indices=train_indices,
                validation_indices=validation_indices,
                fold_number=fold_number,
                model_factory=model_factory,
                model_name=model_name,
            )

            folds.append(fold)

        return self._build_result(
            folds=folds,
            model_name=model_name,
            feature_names=features,
        )

    # ------------------------------------------------------------------
    # Fold execution
    # ------------------------------------------------------------------

    def _run_classifier_fold(
        self,
        *,
        frame: pd.DataFrame,
        target: pd.Series,
        features: Sequence[str],
        target_column: str,
        train_indices: np.ndarray,
        validation_indices: np.ndarray,
        fold_number: int,
        model_factory: Callable[[], Any],
        model_name: str,
    ) -> WalkForwardFold:
        """
        Train and evaluate one fold.

        Critical:
            The preprocessor is fitted only on train_indices.
        """

        train_frame = frame.iloc[
            train_indices
        ]

        validation_frame = frame.iloc[
            validation_indices
        ]

        X_train = train_frame[
            list(features)
        ].copy()

        X_validation = validation_frame[
            list(features)
        ].copy()

        y_train = target.iloc[
            train_indices
        ].copy()

        y_validation = target.iloc[
            validation_indices
        ].copy()

        # --------------------------------------------------------------
        # Leakage checks
        # --------------------------------------------------------------

        leakage_check = (
            self._fold_temporal_check(
                train_frame.index,
                validation_frame.index,
            )
            and self._fold_feature_check(
                features,
                target_column,
            )
        )

        if not leakage_check:
            raise RuntimeError(
                f"Leakage check failed for fold "
                f"{fold_number}."
            )

        # --------------------------------------------------------------
        # Numeric validation
        # --------------------------------------------------------------

        X_train = ensure_numeric_features(
            X_train
        )

        X_validation = ensure_numeric_features(
            X_validation
        )

        # --------------------------------------------------------------
        # Fresh preprocessor
        # --------------------------------------------------------------

        if self.config.refit_preprocessor_each_fold:
            preprocessor = SafePreprocessor(
                self.preprocessor_config
            )
        else:
            raise RuntimeError(
                "Reusing a preprocessor across folds is disabled "
                "for safety. Each fold must fit independently."
            )

        X_train_transformed = (
            preprocessor.fit_transform(
                X_train
            )
        )

        X_validation_transformed = (
            preprocessor.transform(
                X_validation
            )
        )

        # --------------------------------------------------------------
        # Fresh model
        # --------------------------------------------------------------

        if self.config.refit_model_each_fold:
            model = model_factory()
        else:
            raise RuntimeError(
                "Reusing a model across folds is disabled. "
                "A fresh model is required for each fold."
            )

        if not hasattr(model, "fit"):
            raise TypeError(
                "model_factory must return an object with fit()."
            )

        if not hasattr(model, "predict"):
            raise TypeError(
                "model_factory must return an object with predict()."
            )

        model.fit(
            X_train_transformed,
            y_train,
        )

        # --------------------------------------------------------------
        # Predictions
        # --------------------------------------------------------------

        y_pred = np.asarray(
            model.predict(
                X_validation_transformed
            )
        )

        y_probability = self._predict_probability(
            model,
            X_validation_transformed,
        )

        # --------------------------------------------------------------
        # Metrics
        # --------------------------------------------------------------

        metrics = classification_metrics(
            y_validation,
            y_pred,
            y_probability,
        )

        metrics_dict = self._metrics_to_dict(
            metrics
        )

        return WalkForwardFold(
            fold_number=fold_number,
            train_start=train_frame.index.min(),
            train_end=train_frame.index.max(),
            validation_start=(
                validation_frame.index.min()
            ),
            validation_end=(
                validation_frame.index.max()
            ),
            train_samples=len(
                train_frame
            ),
            validation_samples=len(
                validation_frame
            ),
            metrics=metrics_dict,
            y_true=np.asarray(
                y_validation
            ),
            y_probability=y_probability,
            y_pred=y_pred,
            feature_names=list(
                features
            ),
            model_name=model_name,
            preprocessing_fitted_on=(
                f"{train_frame.index.min()} "
                f"to "
                f"{train_frame.index.max()}"
            ),
            leakage_check_passed=True,
            notes=[
                "Preprocessor fitted only on training fold.",
                "Model fitted only on training fold.",
                "Validation observations were not used for fitting.",
            ],
        )

    # ------------------------------------------------------------------
    # Split generation
    # ------------------------------------------------------------------

    def _create_splits(
        self,
        frame: pd.DataFrame,
    ):
        """
        Create purged walk-forward splits.

        The existing splitter is used only for temporal index generation.
        Model preprocessing/training remains inside this module.
        """

        splits = purged_walk_forward_splits(
            frame,
            n_splits=self.config.n_splits,
            train_size=self.config.train_size,
            test_size=self.config.validation_size,
            gap=self.config.gap,
            embargo=self.config.embargo,
            expanding=self.config.expanding,
        )

        validated_splits = []

        for train_indices, validation_indices in splits:
            train_indices = np.asarray(
                train_indices,
                dtype=int,
            )

            validation_indices = np.asarray(
                validation_indices,
                dtype=int,
            )

            if len(train_indices) < (
                self.config.min_train_samples
            ):
                continue

            if len(validation_indices) < (
                self.config.min_validation_samples
            ):
                continue

            validated_splits.append(
                (
                    train_indices,
                    validation_indices,
                )
            )

        if not validated_splits:
            raise ValueError(
                "No walk-forward folds satisfy the configured "
                "minimum sample requirements."
            )

        return validated_splits

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_data(
        self,
        *,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[
            Sequence[str]
        ],
    ) -> pd.DataFrame:
        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        if target_column not in data.columns:
            raise ValueError(
                f"Target column '{target_column}' "
                "was not found."
            )

        frame = data.copy()

        if not isinstance(
            frame.index,
            pd.DatetimeIndex,
        ):
            raise TypeError(
                "Walk-forward validation requires "
                "a DatetimeIndex."
            )

        frame = frame.sort_index()

        if frame.index.has_duplicates:
            raise ValueError(
                "Duplicate timestamps detected."
            )

        if not frame.index.is_monotonic_increasing:
            raise ValueError(
                "Data index must be monotonically increasing."
            )

        # Explicitly reject future/target-like columns from features.
        if feature_columns is not None:
            suspicious = [
                column
                for column in feature_columns
                if (
                    "future" in column.lower()
                    or "target" in column.lower()
                    or "direction" in column.lower()
                )
            ]

            if suspicious:
                raise ValueError(
                    "Potential target leakage detected in feature "
                    f"columns: {suspicious}"
                )

        # Target must be known for every evaluated observation.
        frame = frame.dropna(
            subset=[target_column]
        )

        return frame

    @staticmethod
    def _resolve_features(
        frame: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[
            Sequence[str]
        ],
    ) -> List[str]:
        if feature_columns is not None:
            features = list(
                feature_columns
            )

            missing = [
                column
                for column in features
                if column not in frame.columns
            ]

            if missing:
                raise ValueError(
                    "Feature columns missing from dataset: "
                    f"{missing}"
                )

            return features

        features = []

        for column in frame.columns:
            if column == target_column:
                continue

            if (
                "future" in column.lower()
                or "target" in column.lower()
                or "direction" in column.lower()
            ):
                continue

            if pd.api.types.is_numeric_dtype(
                frame[column]
            ):
                features.append(column)

        if not features:
            raise ValueError(
                "No numeric feature columns were found."
            )

        return features

    # ------------------------------------------------------------------
    # Leakage validation
    # ------------------------------------------------------------------

    @staticmethod
    def _fold_temporal_check(
        train_index: pd.Index,
        validation_index: pd.Index,
    ) -> bool:
        if len(train_index) == 0:
            return False

        if len(validation_index) == 0:
            return False

        return (
            train_index.max()
            < validation_index.min()
        )

    @staticmethod
    def _fold_feature_check(
        features: Sequence[str],
        target_column: str,
    ) -> bool:
        if target_column in features:
            return False

        for column in features:
            lower = column.lower()

            if (
                "future" in lower
                or "target" in lower
                or "direction" in lower
            ):
                return False

        return True

    # ------------------------------------------------------------------
    # Probability extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _predict_probability(
        model: Any,
        X,
    ) -> np.ndarray:
        if hasattr(
            model,
            "predict_proba",
        ):
            probability = np.asarray(
                model.predict_proba(X),
                dtype=float,
            )

            if probability.ndim == 2:
                if probability.shape[1] == 1:
                    return probability[:, 0]

                return probability[:, -1]

            return probability.reshape(-1)

        # Models without calibrated probability estimates should not be
        # treated as calibrated probabilities. We still return a numeric
        # representation so classification metrics can be calculated,
        # but calibration must later reject this source if required.
        prediction = np.asarray(
            model.predict(X),
            dtype=float,
        )

        return prediction.reshape(-1)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _metrics_to_dict(
        metrics: Any,
    ) -> Dict[str, float]:
        if isinstance(
            metrics,
            dict,
        ):
            source = metrics
        elif hasattr(
            metrics,
            "__dataclass_fields__",
        ):
            source = {
                name: getattr(
                    metrics,
                    name,
                )
                for name in metrics.__dataclass_fields__
            }
        else:
            source = {
                key: value
                for key, value in vars(
                    metrics
                ).items()
                if not key.startswith("_")
            }

        result: Dict[str, float] = {}

        for key, value in source.items():
            if value is None:
                continue

            if isinstance(
                value,
                (
                    int,
                    float,
                    np.integer,
                    np.floating,
                ),
            ):
                if np.isfinite(value):
                    result[key] = float(
                        value
                    )

        return result

    # ------------------------------------------------------------------
    # Result construction
    # ------------------------------------------------------------------

    def _build_result(
        self,
        *,
        folds: Sequence[WalkForwardFold],
        model_name: str,
        feature_names: Sequence[str],
    ) -> WalkForwardResult:
        if not folds:
            raise ValueError(
                "Cannot construct walk-forward result without folds."
            )

        accuracies = np.asarray(
            [
                fold.accuracy
                for fold in folds
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
            np.std(accuracies)
        )

        leakage_checks_passed = all(
            fold.leakage_check_passed
            for fold in folds
        )

        passed_accuracy_gate = (
            mean_accuracy
            >= self.config.minimum_accuracy
        )

        notes: List[str] = []

        if passed_accuracy_gate:
            notes.append(
                "Mean walk-forward accuracy passed the configured threshold."
            )
        else:
            notes.append(
                "Mean walk-forward accuracy failed the configured threshold."
            )

        if minimum_accuracy >= (
            self.config.minimum_accuracy
        ):
            notes.append(
                "Every walk-forward fold passed the accuracy threshold."
            )
        else:
            notes.append(
                "At least one walk-forward fold failed the accuracy threshold."
            )

        if accuracy_std <= 0.05:
            notes.append(
                "Fold accuracy variation is relatively small."
            )
        else:
            notes.append(
                "Fold accuracy variation is elevated; "
                "investigate regime instability."
            )

        notes.append(
            "Walk-forward validation does not constitute production approval."
        )

        return WalkForwardResult(
            folds=list(folds),
            config=self.config,
            mean_accuracy=mean_accuracy,
            median_accuracy=median_accuracy,
            minimum_accuracy=minimum_accuracy,
            maximum_accuracy=maximum_accuracy,
            accuracy_std=accuracy_std,
            passed_accuracy_gate=passed_accuracy_gate,
            leakage_checks_passed=(
                leakage_checks_passed
            ),
            model_name=model_name,
            feature_names=list(
                feature_names
            ),
            notes=notes,
        )


# ----------------------------------------------------------------------
# Convenience functions
# ----------------------------------------------------------------------


def walk_forward_classification(
    data: pd.DataFrame,
    *,
    target_column: str,
    model_factory: Callable[[], Any],
    config: Optional[
        WalkForwardConfig
    ] = None,
    preprocessor_config: Optional[
        PreprocessorConfig
    ] = None,
    model_name: str = "classifier",
    feature_columns: Optional[
        Sequence[str]
    ] = None,
) -> WalkForwardResult:
    """
    Convenience wrapper for walk-forward classification.
    """

    validator = WalkForwardValidator(
        config=config,
        preprocessor_config=preprocessor_config,
    )

    return validator.validate_classifier(
        data,
        target_column=target_column,
        model_factory=model_factory,
        model_name=model_name,
        feature_columns=feature_columns,
    )


def walk_forward_summary(
    result: WalkForwardResult,
) -> Dict[str, Any]:
    """Return a compact summary."""

    return result.summary()


def walk_forward_fold_table(
    result: WalkForwardResult,
) -> pd.DataFrame:
    """Return fold metrics as a DataFrame."""

    return result.fold_table()


def aggregate_walk_forward_predictions(
    result: WalkForwardResult,
) -> pd.DataFrame:
    """
    Combine all fold predictions into one chronological DataFrame.

    These predictions are all out-of-sample relative to their respective
    training folds.
    """

    frames: List[pd.DataFrame] = []

    for fold in result.folds:
        frame = pd.DataFrame(
            {
                "y_true": fold.y_true,
                "y_probability": fold.y_probability,
                "y_pred": fold.y_pred,
                "fold": fold.fold_number,
            },
            index=fold.validation_start
            + pd.to_timedelta(
                np.arange(
                    fold.validation_samples
                ),
                unit="ns",
            ),
        )

        # The temporary index above is only a placeholder. Replace it
        # with the exact validation timestamps when possible.
        frames.append(frame)

    if not frames:
        return pd.DataFrame(
            columns=[
                "y_true",
                "y_probability",
                "y_pred",
                "fold",
            ]
        )

    return pd.concat(
        frames,
        axis=0,
    )


def validate_walk_forward_result(
    result: WalkForwardResult,
) -> None:
    """
    Assert that a walk-forward result satisfies basic integrity checks.
    """

    if result.n_folds == 0:
        raise ValueError(
            "Walk-forward result contains no folds."
        )

    if not result.leakage_checks_passed:
        raise ValueError(
            "Walk-forward leakage checks failed."
        )

    for fold in result.folds:
        if fold.train_end >= fold.validation_start:
            raise ValueError(
                f"Fold {fold.fold_number} has temporal overlap."
            )

        if fold.train_samples <= 0:
            raise ValueError(
                f"Fold {fold.fold_number} has no training observations."
            )

        if fold.validation_samples <= 0:
            raise ValueError(
                f"Fold {fold.fold_number} has no validation observations."
            )


__all__ = [
    "WalkForwardConfig",
    "WalkForwardFold",
    "WalkForwardResult",
    "WalkForwardValidator",
    "walk_forward_classification",
    "walk_forward_summary",
    "walk_forward_fold_table",
    "aggregate_walk_forward_predictions",
    "validate_walk_forward_result",
]
