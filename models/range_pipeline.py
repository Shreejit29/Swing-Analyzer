"""
End-to-end target-price range modelling pipeline.

Purpose
-------
Train and evaluate models that predict a future price/return interval
rather than a single point estimate.

Example:

    Current price:       1200
    Predicted 5D range:  1208 - 1275
    Median estimate:     1242

The range model is evaluated for:
    - interval coverage
    - average interval width
    - lower/upper bound behaviour
    - calibration
    - temporal stability

Temporal safety
---------------
All fitting is performed chronologically.

The final untouched holdout must NOT be used to:
    - choose the range model
    - tune hyperparameters
    - choose quantiles
    - choose features
    - tune the range width
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .range_model import (
    PriceRangeModel,
    RangeModelConfig,
)
from .range_validation import (
    RangeValidationConfig,
    RangeValidationResult,
    evaluate_range,
    range_validation_summary,
)
from .preprocessing import (
    PreprocessorConfig,
    SafePreprocessor,
)
from .splitter import purged_walk_forward_splits


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass
class RangePipelineConfig:
    """
    Configuration for range-model training and validation.
    """

    lower_alpha: float = 0.10

    median_alpha: float = 0.50

    upper_alpha: float = 0.90

    minimum_samples: int = 100

    n_splits: int = 5

    train_size: Optional[int] = None

    validation_size: Optional[int] = None

    gap: int = 0

    embargo: int = 0

    expanding: bool = True

    expected_coverage: float = 0.80

    minimum_coverage: float = 0.70

    maximum_coverage: float = 0.95

    maximum_average_width: Optional[float] = None

    random_state: int = 42

    def __post_init__(self) -> None:
        if not (
            0.0
            < self.lower_alpha
            < self.median_alpha
            < self.upper_alpha
            < 1.0
        ):
            raise ValueError(
                "Range quantiles must satisfy "
                "0 < lower < median < upper < 1."
            )

        if self.minimum_samples < 1:
            raise ValueError(
                "minimum_samples must be positive."
            )

        if self.n_splits < 1:
            raise ValueError(
                "n_splits must be positive."
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

        if not 0.0 < self.expected_coverage < 1.0:
            raise ValueError(
                "expected_coverage must be between 0 and 1."
            )

        if not (
            0.0
            < self.minimum_coverage
            <= self.maximum_coverage
            <= 1.0
        ):
            raise ValueError(
                "Invalid range coverage limits."
            )


# ----------------------------------------------------------------------
# Fold result
# ----------------------------------------------------------------------


@dataclass
class RangeFoldResult:
    """
    Result from one walk-forward range-model fold.
    """

    fold_number: int

    train_start: pd.Timestamp

    train_end: pd.Timestamp

    validation_start: pd.Timestamp

    validation_end: pd.Timestamp

    train_samples: int

    validation_samples: int

    coverage: float

    average_width: float

    median_width: float

    mean_lower_error: float

    mean_upper_error: float

    mean_absolute_interval_error: float

    validation_result: Optional[
        RangeValidationResult
    ] = None

    passed: bool = False

    notes: list[str] = field(
        default_factory=list
    )


# ----------------------------------------------------------------------
# Complete result
# ----------------------------------------------------------------------


@dataclass
class RangePipelineResult:
    """
    Complete range-model research result.
    """

    model: Optional[
        PriceRangeModel
    ]

    folds: list[
        RangeFoldResult
    ]

    aggregate_result: Optional[
        RangeValidationResult
    ]

    feature_names: Sequence[str]

    target_column: str

    horizon: int

    target_type: str

    passed: bool

    notes: list[str] = field(
        default_factory=list
    )

    def fold_table(self) -> pd.DataFrame:
        """Return fold metrics as a DataFrame."""

        rows = []

        for fold in self.folds:
            rows.append(
                {
                    "fold": fold.fold_number,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "validation_start": (
                        fold.validation_start
                    ),
                    "validation_end": (
                        fold.validation_end
                    ),
                    "train_samples": (
                        fold.train_samples
                    ),
                    "validation_samples": (
                        fold.validation_samples
                    ),
                    "coverage": fold.coverage,
                    "average_width": (
                        fold.average_width
                    ),
                    "median_width": (
                        fold.median_width
                    ),
                    "mean_lower_error": (
                        fold.mean_lower_error
                    ),
                    "mean_upper_error": (
                        fold.mean_upper_error
                    ),
                    "mean_absolute_interval_error": (
                        fold.mean_absolute_interval_error
                    ),
                    "passed": fold.passed,
                }
            )

        return pd.DataFrame(rows)

    def summary(self) -> Dict[str, Any]:
        aggregate = None

        if self.aggregate_result is not None:
            aggregate = range_validation_summary(
                self.aggregate_result
            )

        return {
            "target_column": self.target_column,
            "horizon": self.horizon,
            "target_type": self.target_type,
            "folds": len(self.folds),
            "passed_folds": sum(
                fold.passed
                for fold in self.folds
            ),
            "aggregate": aggregate,
            "passed": self.passed,
        }


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------


class RangePipeline:
    """
    Leakage-aware range modelling pipeline.

    Typical workflow:

        historical dataset
              ↓
        chronological development split
              ↓
        walk-forward range training
              ↓
        range validation
              ↓
        select model
              ↓
        freeze configuration
              ↓
        final untouched holdout
    """

    def __init__(
        self,
        config: Optional[
            RangePipelineConfig
        ] = None,
        preprocessor_config: Optional[
            PreprocessorConfig
        ] = None,
        model_config: Optional[
            RangeModelConfig
        ] = None,
        validation_config: Optional[
            RangeValidationConfig
        ] = None,
    ) -> None:
        self.config = (
            config
            or RangePipelineConfig()
        )

        self.preprocessor_config = (
            preprocessor_config
            or PreprocessorConfig()
        )

        self.model_config = (
            model_config
            or RangeModelConfig(
                lower_alpha=(
                    self.config.lower_alpha
                ),
                median_alpha=(
                    self.config.median_alpha
                ),
                upper_alpha=(
                    self.config.upper_alpha
                ),
                random_state=(
                    self.config.random_state
                ),
            )
        )

        self.validation_config = (
            validation_config
            or RangeValidationConfig(
                expected_coverage=(
                    self.config.expected_coverage
                ),
                minimum_coverage=(
                    self.config.minimum_coverage
                ),
                maximum_coverage=(
                    self.config.maximum_coverage
                ),
                maximum_average_width=(
                    self.config.maximum_average_width
                ),
            )
        )

        self.model: Optional[
            PriceRangeModel
        ] = None

    # ------------------------------------------------------------------
    # Walk-forward training
    # ------------------------------------------------------------------

    def walk_forward_validate(
        self,
        data: pd.DataFrame,
        *,
        lower_target: str,
        median_target: str,
        upper_target: str,
        feature_columns: Optional[
            Sequence[str]
        ] = None,
        horizon: int = 5,
        target_type: str = "return",
    ) -> RangePipelineResult:
        """
        Train and evaluate a fresh range model on every walk-forward fold.

        The three target columns should represent the same future horizon
        and should never be included in feature_columns.
        """

        frame = self._prepare_data(
            data=data,
            lower_target=lower_target,
            median_target=median_target,
            upper_target=upper_target,
            feature_columns=feature_columns,
        )

        features = self._resolve_features(
            frame,
            target_columns=[
                lower_target,
                median_target,
                upper_target,
            ],
            feature_columns=feature_columns,
        )

        if len(frame) < (
            self.config.minimum_samples
        ):
            raise ValueError(
                "Insufficient samples for range modelling."
            )

        splits = self._create_splits(
            frame
        )

        folds: list[
            RangeFoldResult
        ] = []

        aggregate_lower = []
        aggregate_median = []
        aggregate_upper = []
        aggregate_actual = []
        aggregate_timestamps = []

        for fold_number, (
            train_indices,
            validation_indices,
        ) in enumerate(
            splits,
            start=1,
        ):
            train = frame.iloc[
                train_indices
            ]

            validation = frame.iloc[
                validation_indices
            ]

            # ----------------------------------------------------------
            # Temporal safety
            # ----------------------------------------------------------

            if train.index.max() >= (
                validation.index.min()
            ):
                raise RuntimeError(
                    "Temporal leakage detected in range fold."
                )

            X_train = train[
                features
            ]

            X_validation = validation[
                features
            ]

            y_train_lower = train[
                lower_target
            ]

            y_train_median = train[
                median_target
            ]

            y_train_upper = train[
                upper_target
            ]

            y_validation_lower = (
                validation[
                    lower_target
                ]
            )

            y_validation_median = (
                validation[
                    median_target
                ]
            )

            y_validation_upper = (
                validation[
                    upper_target
                ]
            )

            # ----------------------------------------------------------
            # Fresh preprocessing per fold
            # ----------------------------------------------------------

            preprocessor = SafePreprocessor(
                self.preprocessor_config
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

            # ----------------------------------------------------------
            # Fresh range model per fold
            # ----------------------------------------------------------

            model = self._new_model()

            model.fit(
                X_train_transformed,
                y_train_lower,
                y_train_median,
                y_train_upper,
            )

            prediction = model.predict_returns(
                X_validation_transformed
            )

            lower_prediction = np.asarray(
                prediction["lower"],
                dtype=float,
            )

            median_prediction = np.asarray(
                prediction["median"],
                dtype=float,
            )

            upper_prediction = np.asarray(
                prediction["upper"],
                dtype=float,
            )

            # ----------------------------------------------------------
            # Guarantee ordered predictions
            # ----------------------------------------------------------

            (
                lower_prediction,
                median_prediction,
                upper_prediction,
            ) = self._order_bounds(
                lower_prediction,
                median_prediction,
                upper_prediction,
            )

            # ----------------------------------------------------------
            # Validate interval against actual outcome
            # ----------------------------------------------------------

            actual = np.asarray(
                y_validation_median,
                dtype=float,
            )

            validation_result = evaluate_range(
                actual=actual,
                lower=lower_prediction,
                upper=upper_prediction,
                config=self.validation_config,
            )

            fold_metrics = (
                self._fold_metrics(
                    actual=actual,
                    lower=lower_prediction,
                    upper=upper_prediction,
                    lower_target=np.asarray(
                        y_validation_lower,
                        dtype=float,
                    ),
                    upper_target=np.asarray(
                        y_validation_upper,
                        dtype=float,
                    ),
                )
            )

            fold_passed = (
                validation_result.passed
            )

            notes = []

            if fold_passed:
                notes.append(
                    "Range coverage passed."
                )
            else:
                notes.append(
                    "Range validation failed."
                )

            fold_result = RangeFoldResult(
                fold_number=fold_number,
                train_start=train.index.min(),
                train_end=train.index.max(),
                validation_start=(
                    validation.index.min()
                ),
                validation_end=(
                    validation.index.max()
                ),
                train_samples=len(train),
                validation_samples=len(
                    validation
                ),
                coverage=(
                    validation_result.coverage
                ),
                average_width=(
                    validation_result.average_width
                ),
                median_width=(
                    validation_result.median_width
                ),
                mean_lower_error=(
                    fold_metrics[
                        "mean_lower_error"
                    ]
                ),
                mean_upper_error=(
                    fold_metrics[
                        "mean_upper_error"
                    ]
                ),
                mean_absolute_interval_error=(
                    fold_metrics[
                        "mean_absolute_interval_error"
                    ]
                ),
                validation_result=(
                    validation_result
                ),
                passed=fold_passed,
                notes=notes,
            )

            folds.append(
                fold_result
            )

            aggregate_lower.extend(
                lower_prediction.tolist()
            )

            aggregate_median.extend(
                median_prediction.tolist()
            )

            aggregate_upper.extend(
                upper_prediction.tolist()
            )

            aggregate_actual.extend(
                actual.tolist()
            )

            aggregate_timestamps.extend(
                validation.index.tolist()
            )

        # --------------------------------------------------------------
        # Aggregate out-of-sample evaluation
        # --------------------------------------------------------------

        aggregate_result = evaluate_range(
            actual=np.asarray(
                aggregate_actual,
                dtype=float,
            ),
            lower=np.asarray(
                aggregate_lower,
                dtype=float,
            ),
            upper=np.asarray(
                aggregate_upper,
                dtype=float,
            ),
            config=self.validation_config,
        )

        passed = (
            len(folds) > 0
            and all(
                fold.passed
                for fold in folds
            )
            and aggregate_result.passed
        )

        notes = [
            (
                "All range predictions were generated "
                "out-of-sample within each walk-forward fold."
            ),
            (
                "Preprocessing was fitted independently "
                "inside each fold."
            ),
            (
                "The final holdout must remain untouched "
                "during range-model selection."
            ),
        ]

        if passed:
            notes.append(
                "Range model passed the configured research validation."
            )
        else:
            notes.append(
                "Range model did not pass all configured research gates."
            )

        return RangePipelineResult(
            model=None,
            folds=folds,
            aggregate_result=aggregate_result,
            feature_names=features,
            target_column=median_target,
            horizon=horizon,
            target_type=target_type,
            passed=passed,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Final model training
    # ------------------------------------------------------------------

    def fit_final_model(
        self,
        data: pd.DataFrame,
        *,
        lower_target: str,
        median_target: str,
        upper_target: str,
        feature_columns: Sequence[str],
    ) -> PriceRangeModel:
        """
        Fit the selected range model.

        This method should only be called after model configuration and
        features have been frozen using development/walk-forward data.

        The caller is responsible for ensuring that the final holdout
        remains excluded.
        """

        frame = self._prepare_data(
            data=data,
            lower_target=lower_target,
            median_target=median_target,
            upper_target=upper_target,
            feature_columns=feature_columns,
        )

        X = frame[
            list(feature_columns)
        ]

        y_lower = frame[
            lower_target
        ]

        y_median = frame[
            median_target
        ]

        y_upper = frame[
            upper_target
        ]

        preprocessor = SafePreprocessor(
            self.preprocessor_config
        )

        X_transformed = (
            preprocessor.fit_transform(
                X
            )
        )

        model = self._new_model()

        model.fit(
            X_transformed,
            y_lower,
            y_median,
            y_upper,
        )

        self.model = model

        return model

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        X: pd.DataFrame,
        *,
        preprocessor: SafePreprocessor,
        current_price: Optional[
            Sequence[float]
        ] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Predict future return ranges.

        If current_price is supplied, price ranges are also returned.
        """

        if self.model is None:
            raise RuntimeError(
                "No final range model is fitted."
            )

        transformed = (
            preprocessor.transform(X)
        )

        prediction = (
            self.model.predict_returns(
                transformed
            )
        )

        lower = np.asarray(
            prediction["lower"],
            dtype=float,
        )

        median = np.asarray(
            prediction["median"],
            dtype=float,
        )

        upper = np.asarray(
            prediction["upper"],
            dtype=float,
        )

        lower, median, upper = (
            self._order_bounds(
                lower,
                median,
                upper,
            )
        )

        result = {
            "lower_return": lower,
            "median_return": median,
            "upper_return": upper,
        }

        if current_price is not None:
            prices = np.asarray(
                current_price,
                dtype=float,
            ).reshape(-1)

            if len(prices) != len(lower):
                raise ValueError(
                    "current_price length does not match predictions."
                )

            if (
                ~np.isfinite(prices)
            ).any() or (
                prices <= 0
            ).any():
                raise ValueError(
                    "current_price contains invalid values."
                )

            result.update(
                {
                    "lower_price": (
                        prices * (1.0 + lower)
                    ),
                    "median_price": (
                        prices * (1.0 + median)
                    ),
                    "upper_price": (
                        prices * (1.0 + upper)
                    ),
                }
            )

        return result

    # ------------------------------------------------------------------
    # Split creation
    # ------------------------------------------------------------------

    def _create_splits(
        self,
        frame: pd.DataFrame,
    ):
        splits = purged_walk_forward_splits(
            frame,
            n_splits=self.config.n_splits,
            train_size=self.config.train_size,
            test_size=self.config.validation_size,
            gap=self.config.gap,
            embargo=self.config.embargo,
            expanding=self.config.expanding,
        )

        valid = []

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
                self.config.minimum_samples
            ):
                continue

            if len(validation_indices) < 1:
                continue

            valid.append(
                (
                    train_indices,
                    validation_indices,
                )
            )

        if not valid:
            raise ValueError(
                "No valid walk-forward range folds were produced."
            )

        return valid

    # ------------------------------------------------------------------
    # Model factory
    # ------------------------------------------------------------------

    def _new_model(
        self,
    ) -> PriceRangeModel:
        return PriceRangeModel(
            config=self.model_config
        )

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_data(
        *,
        data: pd.DataFrame,
        lower_target: str,
        median_target: str,
        upper_target: str,
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

        targets = [
            lower_target,
            median_target,
            upper_target,
        ]

        missing_targets = [
            target
            for target in targets
            if target not in data.columns
        ]

        if missing_targets:
            raise ValueError(
                "Missing range target columns: "
                f"{missing_targets}"
            )

        frame = data.copy()

        if not isinstance(
            frame.index,
            pd.DatetimeIndex,
        ):
            raise TypeError(
                "Range modelling requires a DatetimeIndex."
            )

        frame = frame.sort_index()

        if frame.index.has_duplicates:
            raise ValueError(
                "Duplicate timestamps detected."
            )

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
                    "Potential target leakage in range features: "
                    f"{suspicious}"
                )

        # Range targets must all be available.
        frame = frame.dropna(
            subset=targets
        )

        # Validate target ordering.
        lower = frame[
            lower_target
        ].to_numpy(dtype=float)

        median = frame[
            median_target
        ].to_numpy(dtype=float)

        upper = frame[
            upper_target
        ].to_numpy(dtype=float)

        if not (
            np.isfinite(lower).all()
            and np.isfinite(median).all()
            and np.isfinite(upper).all()
        ):
            raise ValueError(
                "Range targets contain NaN or infinite values."
            )

        if (
            lower > median
        ).any() or (
            median > upper
        ).any():
            raise ValueError(
                "Range targets must satisfy "
                "lower <= median <= upper."
            )

        return frame

    @staticmethod
    def _resolve_features(
        frame: pd.DataFrame,
        target_columns: Sequence[str],
        feature_columns: Optional[
            Sequence[str]
        ],
    ) -> list[str]:
        if feature_columns is not None:
            features = list(
                feature_columns
            )

            missing = [
                feature
                for feature in features
                if feature not in frame.columns
            ]

            if missing:
                raise ValueError(
                    f"Missing feature columns: {missing}"
                )

            return features

        features = []

        target_set = set(
            target_columns
        )

        for column in frame.columns:
            if column in target_set:
                continue

            lower = column.lower()

            if (
                "future" in lower
                or "target" in lower
                or "direction" in lower
            ):
                continue

            if pd.api.types.is_numeric_dtype(
                frame[column]
            ):
                features.append(column)

        if not features:
            raise ValueError(
                "No numeric range-model features found."
            )

        return features

    # ------------------------------------------------------------------
    # Range metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _fold_metrics(
        *,
        actual: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        lower_target: np.ndarray,
        upper_target: np.ndarray,
    ) -> Dict[str, float]:
        lower_error = (
            lower_target - lower
        )

        upper_error = (
            upper_target - upper
        )

        interval_error = np.maximum(
            lower - actual,
            0.0,
        ) + np.maximum(
            actual - upper,
            0.0,
        )

        return {
            "mean_lower_error": float(
                np.mean(
                    np.abs(
                        lower_error
                    )
                )
            ),
            "mean_upper_error": float(
                np.mean(
                    np.abs(
                        upper_error
                    )
                )
            ),
            "mean_absolute_interval_error": float(
                np.mean(
                    interval_error
                )
            ),
        }

    # ------------------------------------------------------------------
    # Bound ordering
    # ------------------------------------------------------------------

    @staticmethod
    def _order_bounds(
        lower: np.ndarray,
        median: np.ndarray,
        upper: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        stacked = np.column_stack(
            [
                lower,
                median,
                upper,
            ]
        )

        ordered = np.sort(
            stacked,
            axis=1,
        )

        return (
            ordered[:, 0],
            ordered[:, 1],
            ordered[:, 2],
        )


# ----------------------------------------------------------------------
# Convenience helpers
# ----------------------------------------------------------------------


def train_range_model_walk_forward(
    data: pd.DataFrame,
    *,
    lower_target: str,
    median_target: str,
    upper_target: str,
    feature_columns: Optional[
        Sequence[str]
    ] = None,
    horizon: int = 5,
    target_type: str = "return",
    config: Optional[
        RangePipelineConfig
    ] = None,
    preprocessor_config: Optional[
        PreprocessorConfig
    ] = None,
    model_config: Optional[
        RangeModelConfig
    ] = None,
    validation_config: Optional[
        RangeValidationConfig
    ] = None,
) -> RangePipelineResult:
    """
    Convenience wrapper for walk-forward range modelling.
    """

    pipeline = RangePipeline(
        config=config,
        preprocessor_config=preprocessor_config,
        model_config=model_config,
        validation_config=validation_config,
    )

    return pipeline.walk_forward_validate(
        data,
        lower_target=lower_target,
        median_target=median_target,
        upper_target=upper_target,
        feature_columns=feature_columns,
        horizon=horizon,
        target_type=target_type,
    )


def range_pipeline_summary(
    result: RangePipelineResult,
) -> Dict[str, Any]:
    """Return a compact range-pipeline summary."""

    return result.summary()


__all__ = [
    "RangePipelineConfig",
    "RangeFoldResult",
    "RangePipelineResult",
    "RangePipeline",
    "train_range_model_walk_forward",
    "range_pipeline_summary",
]
