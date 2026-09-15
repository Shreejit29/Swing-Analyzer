"""
Feature-selection utilities for the AI Swing Analyser.

This module performs conservative, leakage-aware feature filtering.

Selection methods include:
    - numeric validation
    - excessive-missing filtering
    - constant / near-constant filtering
    - infinite-value handling
    - high-correlation pruning
    - feature whitelist / blacklist
    - deterministic feature ordering

IMPORTANT
---------
Feature selection must be fitted only on the training/development
period during model research.

Never calculate correlation, variance, or other selection statistics
using the final untouched holdout set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSelectionConfig:
    """
    Conservative feature-selection configuration.
    """

    max_missing_fraction: float = 0.40

    min_variance: float = 1e-12

    max_correlation: float = 0.95

    correlation_method: str = "spearman"

    drop_infinite: bool = True

    include_columns: Optional[
        tuple[str, ...]
    ] = None

    exclude_columns: tuple[str, ...] = field(
        default_factory=tuple
    )

    min_features: int = 10

    max_features: Optional[int] = None

    def __post_init__(self) -> None:
        if not (
            0.0
            <= self.max_missing_fraction
            < 1.0
        ):
            raise ValueError(
                "max_missing_fraction must be between 0 and 1."
            )

        if self.min_variance < 0:
            raise ValueError(
                "min_variance cannot be negative."
            )

        if not (
            0.0
            < self.max_correlation
            <= 1.0
        ):
            raise ValueError(
                "max_correlation must be greater than 0 and "
                "less than or equal to 1."
            )

        valid_methods = {
            "pearson",
            "spearman",
            "kendall",
        }

        if (
            self.correlation_method
            not in valid_methods
        ):
            raise ValueError(
                "Unsupported correlation method: "
                f"{self.correlation_method}"
            )

        if self.min_features < 0:
            raise ValueError(
                "min_features cannot be negative."
            )

        if (
            self.max_features is not None
            and self.max_features < 1
        ):
            raise ValueError(
                "max_features must be at least 1."
            )

        if (
            self.max_features is not None
            and self.max_features
            < self.min_features
        ):
            raise ValueError(
                "max_features cannot be less than min_features."
            )


# ----------------------------------------------------------------------
# Feature statistics
# ----------------------------------------------------------------------


@dataclass
class FeatureSelectionStatistics:
    """
    Statistics calculated exclusively from the fitting dataset.
    """

    feature: str

    dtype: str

    missing_fraction: float

    variance: float

    unique_values: int

    mean: float

    std: float

    selected: bool = False

    reason: str = "not_evaluated"


# ----------------------------------------------------------------------
# Fitted selector
# ----------------------------------------------------------------------


class FeatureSelector:
    """
    Fit a deterministic feature selector on development data.

    Once fitted, the selected feature list can be applied to validation,
    test, and inference datasets without recalculating statistics.
    """

    def __init__(
        self,
        config: Optional[
            FeatureSelectionConfig
        ] = None,
    ) -> None:
        self.config = (
            config
            or FeatureSelectionConfig()
        )

        self.selected_features_: list[
            str
        ] = []

        self.dropped_features_: dict[
            str,
            str,
        ] = {}

        self.statistics_: list[
            FeatureSelectionStatistics
        ] = []

        self.correlation_pairs_: list[
            tuple[str, str, float]
        ] = []

        self.fitted_: bool = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        data: pd.DataFrame,
    ) -> "FeatureSelector":
        """
        Fit feature-selection rules on the supplied development data.

        The caller is responsible for ensuring that ``data`` does not
        contain the final holdout period.
        """

        frame = self._prepare(
            data
        )

        self.selected_features_ = []
        self.dropped_features_ = {}
        self.statistics_ = []
        self.correlation_pairs_ = []

        # --------------------------------------------------------------
        # Calculate per-feature statistics
        # --------------------------------------------------------------

        candidates: list[
            str
        ] = []

        for column in frame.columns:
            series = frame[column]

            missing_fraction = float(
                series.isna().mean()
            )

            variance = float(
                series.var(
                    ddof=0
                )
            )

            unique_values = int(
                series.nunique(
                    dropna=True
                )
            )

            mean = float(
                series.mean()
            )

            std = float(
                series.std(
                    ddof=0
                )
            )

            statistic = (
                FeatureSelectionStatistics(
                    feature=column,
                    dtype=str(
                        series.dtype
                    ),
                    missing_fraction=missing_fraction,
                    variance=variance,
                    unique_values=unique_values,
                    mean=mean,
                    std=std,
                )
            )

            self.statistics_.append(
                statistic
            )

            # Explicit exclusion
            if column in (
                self.config.exclude_columns
            ):
                self.dropped_features_[
                    column
                ] = "explicitly_excluded"
                continue

            # Explicit inclusion whitelist
            if (
                self.config.include_columns
                is not None
                and column
                not in self.config.include_columns
            ):
                self.dropped_features_[
                    column
                ] = "not_in_include_list"
                continue

            # Missing data
            if (
                missing_fraction
                > self.config.max_missing_fraction
            ):
                self.dropped_features_[
                    column
                ] = "excessive_missing"
                continue

            # Constant / near-constant
            if (
                not np.isfinite(
                    variance
                )
                or variance
                <= self.config.min_variance
            ):
                self.dropped_features_[
                    column
                ] = "zero_or_near_zero_variance"
                continue

            # No usable observations
            if unique_values < 2:
                self.dropped_features_[
                    column
                ] = "insufficient_unique_values"
                continue

            candidates.append(
                column
            )

        # --------------------------------------------------------------
        # Correlation pruning
        # --------------------------------------------------------------

        candidates = self._remove_correlated(
            frame,
            candidates,
        )

        # --------------------------------------------------------------
        # Maximum feature count
        # --------------------------------------------------------------

        if (
            self.config.max_features
            is not None
            and len(candidates)
            > self.config.max_features
        ):
            candidates = self._limit_features(
                frame,
                candidates,
                self.config.max_features,
            )

        # --------------------------------------------------------------
        # Final selection
        # --------------------------------------------------------------

        if (
            len(candidates)
            < self.config.min_features
        ):
            raise ValueError(
                "Feature selection left only "
                f"{len(candidates)} features, below the required "
                f"minimum of {self.config.min_features}."
            )

        self.selected_features_ = sorted(
            candidates
        )

        for statistic in self.statistics_:
            if (
                statistic.feature
                in self.selected_features_
            ):
                statistic.selected = True
                statistic.reason = "selected"

        self.fitted_ = True

        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Select the exact feature set learned during fitting.
        """

        if not self.fitted_:
            raise RuntimeError(
                "FeatureSelector has not been fitted."
            )

        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        missing = [
            column
            for column
            in self.selected_features_
            if column not in data.columns
        ]

        if missing:
            raise ValueError(
                "Input data is missing selected features: "
                f"{missing}"
            )

        result = data[
            self.selected_features_
        ].copy()

        result = result.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        return result

    # ------------------------------------------------------------------
    # Fit-transform
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        self.fit(
            data
        )

        return self.transform(
            data
        )

    # ------------------------------------------------------------------
    # Feature list
    # ------------------------------------------------------------------

    @property
    def selected_features(
        self,
    ) -> list[str]:
        if not self.fitted_:
            raise RuntimeError(
                "FeatureSelector has not been fitted."
            )

        return list(
            self.selected_features_
        )

    # ------------------------------------------------------------------
    # Statistics table
    # ------------------------------------------------------------------

    def statistics_table(
        self,
    ) -> pd.DataFrame:
        if not self.statistics_:
            return pd.DataFrame()

        rows = []

        for statistic in self.statistics_:
            rows.append(
                {
                    "feature": statistic.feature,
                    "dtype": statistic.dtype,
                    "missing_fraction": statistic.missing_fraction,
                    "variance": statistic.variance,
                    "unique_values": statistic.unique_values,
                    "mean": statistic.mean,
                    "std": statistic.std,
                    "selected": statistic.selected,
                    "reason": statistic.reason,
                }
            )

        return pd.DataFrame(
            rows
        ).sort_values(
            "feature"
        ).reset_index(
            drop=True
        )

    # ------------------------------------------------------------------
    # Dropped features
    # ------------------------------------------------------------------

    def dropped_table(
        self,
    ) -> pd.DataFrame:
        rows = [
            {
                "feature": feature,
                "reason": reason,
            }
            for feature, reason
            in sorted(
                self.dropped_features_.items()
            )
        ]

        return pd.DataFrame(
            rows
        )

    # ------------------------------------------------------------------
    # Correlation pruning
    # ------------------------------------------------------------------

    def _remove_correlated(
        self,
        frame: pd.DataFrame,
        candidates: list[str],
    ) -> list[str]:
        if len(candidates) < 2:
            return candidates

        candidate_frame = frame[
            candidates
        ]

        correlation = (
            candidate_frame.corr(
                method=self.config.correlation_method
            )
        )

        correlation = correlation.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        selected = list(
            candidates
        )

        # Deterministic order means repeated research runs produce the
        # same selected feature set.
        for i, left in enumerate(
            candidates
        ):
            if left not in selected:
                continue

            for right in candidates[
                i + 1 :
            ]:
                if right not in selected:
                    continue

                value = correlation.loc[
                    left,
                    right,
                ]

                if pd.isna(value):
                    continue

                absolute = abs(
                    float(value)
                )

                if (
                    absolute
                    > self.config.max_correlation
                ):
                    self.correlation_pairs_.append(
                        (
                            left,
                            right,
                            float(value),
                        )
                    )

                    # Deterministic rule:
                    # keep the lexicographically earlier feature.
                    selected.remove(
                        right
                    )

                    self.dropped_features_[
                        right
                    ] = (
                        f"high_correlation_with:{left}"
                    )

        return selected

    # ------------------------------------------------------------------
    # Feature-count reduction
    # ------------------------------------------------------------------

    @staticmethod
    def _limit_features(
        frame: pd.DataFrame,
        candidates: list[str],
        maximum: int,
    ) -> list[str]:
        """
        Limit feature count deterministically using variance.

        Higher-variance features are retained.

        This is deliberately simple. Model-based feature importance is
        handled later by ``src/models/feature_selection.py`` and must be
        evaluated inside temporal validation.
        """

        variances = (
            frame[candidates]
            .var(
                ddof=0
            )
            .sort_values(
                ascending=False
            )
        )

        selected = list(
            variances.index[
                :maximum
            ]
        )

        for feature in candidates:
            if feature not in selected:
                # Do not overwrite a more informative rejection reason.
                if (
                    feature
                    not in self.dropped_features_
                ):
                    self.dropped_features_[
                        feature
                    ] = "max_features_limit"

        return selected

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def _prepare(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                "Cannot select features from an empty DataFrame."
            )

        if data.columns.has_duplicates:
            raise ValueError(
                "Feature columns must be unique."
            )

        frame = data.copy()

        for column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

        if self.config.drop_infinite:
            frame = frame.replace(
                [np.inf, -np.inf],
                np.nan,
            )

        return frame


# ----------------------------------------------------------------------
# Leakage guard
# ----------------------------------------------------------------------


def validate_feature_selection_inputs(
    train_data: pd.DataFrame,
    *,
    validation_data: Optional[
        pd.DataFrame
    ] = None,
    test_data: Optional[
        pd.DataFrame
    ] = None,
) -> None:
    """
    Validate the intended feature-selection workflow.

    This function intentionally does not combine validation/test data
    with training data.

    Feature-selection statistics must come from ``train_data`` only.
    """

    if not isinstance(
        train_data,
        pd.DataFrame,
    ):
        raise TypeError(
            "train_data must be a pandas DataFrame."
        )

    if train_data.empty:
        raise ValueError(
            "train_data cannot be empty."
        )

    if (
        validation_data is not None
        and not isinstance(
            validation_data,
            pd.DataFrame,
        )
    ):
        raise TypeError(
            "validation_data must be a pandas DataFrame."
        )

    if (
        test_data is not None
        and not isinstance(
            test_data,
            pd.DataFrame,
        )
    ):
        raise TypeError(
            "test_data must be a pandas DataFrame."
        )

    # If timestamps are available, verify temporal ordering.
    for name, frame in (
        (
            "train_data",
            train_data,
        ),
        (
            "validation_data",
            validation_data,
        ),
        (
            "test_data",
            test_data,
        ),
    ):
        if frame is None:
            continue

        if isinstance(
            frame.index,
            pd.DatetimeIndex,
        ) and not frame.index.is_monotonic_increasing:
            raise ValueError(
                f"{name} index must be chronologically ordered."
            )

    if (
        validation_data is not None
        and isinstance(
            train_data.index,
            pd.DatetimeIndex,
        )
        and isinstance(
            validation_data.index,
            pd.DatetimeIndex,
        )
        and not train_data.index.empty
        and not validation_data.index.empty
    ):
        if (
            train_data.index.max()
            >= validation_data.index.min()
        ):
            raise ValueError(
                "Training data must end before validation data begins."
            )

    if (
        test_data is not None
        and validation_data is not None
        and isinstance(
            validation_data.index,
            pd.DatetimeIndex,
        )
        and isinstance(
            test_data.index,
            pd.DatetimeIndex,
        )
        and not validation_data.index.empty
        and not test_data.index.empty
    ):
        if (
            validation_data.index.max()
            >= test_data.index.min()
        ):
            raise ValueError(
                "Validation data must end before test data begins."
            )


# ----------------------------------------------------------------------
# Convenience functions
# ----------------------------------------------------------------------


def select_features(
    train_data: pd.DataFrame,
    *,
    validation_data: Optional[
        pd.DataFrame
    ] = None,
    test_data: Optional[
        pd.DataFrame
    ] = None,
    config: Optional[
        FeatureSelectionConfig
    ] = None,
) -> tuple[
    FeatureSelector,
    pd.DataFrame,
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
]:
    """
    Fit feature selection on training data and apply the resulting
    feature list to validation/test data.

    This is the recommended convenience API.
    """

    validate_feature_selection_inputs(
        train_data,
        validation_data=validation_data,
        test_data=test_data,
    )

    selector = FeatureSelector(
        config=config
    )

    train_selected = (
        selector.fit_transform(
            train_data
        )
    )

    validation_selected = None

    if validation_data is not None:
        validation_selected = (
            selector.transform(
                validation_data
            )
        )

    test_selected = None

    if test_data is not None:
        test_selected = (
            selector.transform(
                test_data
            )
        )

    return (
        selector,
        train_selected,
        validation_selected,
        test_selected,
    )


def feature_selection_summary(
    selector: FeatureSelector,
) -> dict[str, object]:
    """
    Return a compact feature-selection summary.
    """

    if not selector.fitted_:
        raise RuntimeError(
            "FeatureSelector has not been fitted."
        )

    selected = (
        selector.selected_features_
    )

    dropped = (
        selector.dropped_features_
    )

    return {
        "fitted": True,
        "input_features": len(
            selector.statistics_
        ),
        "selected_features": len(
            selected
        ),
        "dropped_features": len(
            dropped
        ),
        "correlated_pairs": len(
            selector.correlation_pairs_
        ),
        "selected_feature_names": selected,
    }


__all__ = [
    "FeatureSelectionConfig",
    "FeatureSelectionStatistics",
    "FeatureSelector",
    "validate_feature_selection_inputs",
    "select_features",
    "feature_selection_summary",
]
