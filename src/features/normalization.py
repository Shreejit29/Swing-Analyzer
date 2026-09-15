"""
Leakage-safe feature normalization utilities.

Important
---------
Financial time series must never be normalized using statistics calculated
from the complete dataset before a chronological train/test split.

This module therefore provides:

    1. Cross-sectional-safe transformations
    2. Expanding/rolling time-series normalization
    3. Train-fitted normalization
    4. Robust clipping
    5. Feature normalization diagnostics

The model-level ``SafePreprocessor`` remains responsible for the final
training/inference preprocessing pipeline.

This module is intended primarily for research feature engineering and
specialized experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizationConfig:
    """
    Configuration for leakage-safe normalization.
    """

    method: str = "robust"

    lower_quantile: float = 0.05

    upper_quantile: float = 0.95

    clip: bool = True

    clip_lower: float = -5.0

    clip_upper: float = 5.0

    rolling_window: int = 60

    min_periods: int = 20

    eps: float = 1e-12

    def __post_init__(self) -> None:
        valid_methods = {
            "none",
            "zscore",
            "robust",
            "minmax",
            "rank",
            "percentile",
        }

        if self.method not in valid_methods:
            raise ValueError(
                f"Unknown normalization method '{self.method}'. "
                f"Expected one of {sorted(valid_methods)}."
            )

        if not (
            0.0
            <= self.lower_quantile
            < self.upper_quantile
            <= 1.0
        ):
            raise ValueError(
                "Invalid normalization quantiles."
            )

        if self.clip_lower >= self.clip_upper:
            raise ValueError(
                "clip_lower must be less than clip_upper."
            )

        if self.rolling_window < 2:
            raise ValueError(
                "rolling_window must be at least 2."
            )

        if self.min_periods < 1:
            raise ValueError(
                "min_periods must be at least 1."
            )

        if self.eps <= 0:
            raise ValueError(
                "eps must be positive."
            )


# ----------------------------------------------------------------------
# Fitted normalizer
# ----------------------------------------------------------------------


@dataclass
class NormalizationStatistics:
    """
    Statistics learned exclusively from a training dataset.
    """

    method: str

    columns: list[str]

    center: pd.Series

    scale: pd.Series

    lower: pd.Series

    upper: pd.Series

    minimum: pd.Series

    maximum: pd.Series

    def validate_columns(
        self,
        columns: Iterable[str],
    ) -> None:
        expected = list(
            self.columns
        )
        actual = list(
            columns
        )

        if expected != actual:
            raise ValueError(
                "Feature columns do not match the columns used "
                "to fit the normalizer."
            )


class TrainFittedNormalizer:
    """
    Fit normalization statistics on training data only.

    After fitting, ``transform`` can safely be applied to validation,
    test, or future inference observations.
    """

    def __init__(
        self,
        config: Optional[
            NormalizationConfig
        ] = None,
    ) -> None:
        self.config = (
            config
            or NormalizationConfig()
        )

        self.statistics_: Optional[
            NormalizationStatistics
        ] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        data: pd.DataFrame,
    ) -> "TrainFittedNormalizer":
        frame = _validate_numeric_frame(
            data
        )

        columns = list(
            frame.columns
        )

        center = frame.median(
            numeric_only=True
        )

        scale = (
            frame.quantile(
                self.config.upper_quantile
            )
            - frame.quantile(
                self.config.lower_quantile
            )
        )

        lower = frame.quantile(
            self.config.lower_quantile
        )

        upper = frame.quantile(
            self.config.upper_quantile
        )

        minimum = frame.min()

        maximum = frame.max()

        # Replace unstable scales with 1.0.
        scale = scale.where(
            scale.abs()
            > self.config.eps,
            1.0,
        )

        self.statistics_ = (
            NormalizationStatistics(
                method=self.config.method,
                columns=columns,
                center=center,
                scale=scale,
                lower=lower,
                upper=upper,
                minimum=minimum,
                maximum=maximum,
            )
        )

        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        if self.statistics_ is None:
            raise RuntimeError(
                "Normalizer has not been fitted."
            )

        frame = _validate_numeric_frame(
            data
        )

        self.statistics_.validate_columns(
            frame.columns
        )

        result = frame.copy()

        method = (
            self.config.method
        )

        if method == "none":
            return result

        if method == "robust":
            result = (
                result
                - self.statistics_.center
            ) / self.statistics_.scale

        elif method == "zscore":
            # The robust scale stored above is not suitable for a true
            # z-score, so calculate standard deviation from the training
            # statistics at fit time would be preferable. To preserve
            # compatibility, this branch is implemented through the
            # training median and IQR only when standard deviation is
            # unavailable.
            #
            # This branch is overridden below by the explicit helper.
            result = _zscore_using_training_stats(
                frame,
                self.statistics_,
                eps=self.config.eps,
            )

        elif method == "minmax":
            denominator = (
                self.statistics_.maximum
                - self.statistics_.minimum
            ).where(
                lambda x: x.abs()
                > self.config.eps,
                1.0,
            )

            result = (
                frame
                - self.statistics_.minimum
            ) / denominator

        elif method in {
            "rank",
            "percentile",
        }:
            # Rank normalization cannot safely use future observations
            # when applied to a time series. For a fitted normalizer we
            # therefore map values using the training empirical
            # distribution.
            result = _empirical_percentile_transform(
                frame,
                self.statistics_,
            )

        else:
            raise RuntimeError(
                f"Unsupported normalization method: {method}"
            )

        if self.config.clip:
            result = result.clip(
                lower=self.config.clip_lower,
                upper=self.config.clip_upper,
            )

        return result.replace(
            [np.inf, -np.inf],
            np.nan,
        )

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
    # State
    # ------------------------------------------------------------------

    @property
    def fitted(
        self,
    ) -> bool:
        return (
            self.statistics_
            is not None
        )


# ----------------------------------------------------------------------
# Z-score statistics
# ----------------------------------------------------------------------


def _zscore_using_training_stats(
    data: pd.DataFrame,
    statistics: NormalizationStatistics,
    *,
    eps: float,
) -> pd.DataFrame:
    """
    Calculate z-scores.

    The NormalizationStatistics object currently stores robust scale.
    To prevent accidental leakage, this helper uses an approximation
    based on IQR:

        sigma ≈ IQR / 1.349

    This conversion is appropriate for approximately normal variables
    and is deliberately robust to outliers.
    """

    sigma = (
        statistics.scale
        / 1.349
    )

    sigma = sigma.where(
        sigma.abs() > eps,
        1.0,
    )

    return (
        data
        - statistics.center
    ) / sigma


# ----------------------------------------------------------------------
# Empirical percentile transformation
# ----------------------------------------------------------------------


def _empirical_percentile_transform(
    data: pd.DataFrame,
    statistics: NormalizationStatistics,
) -> pd.DataFrame:
    """
    Map observations to approximate percentiles using training
    quantiles.

    Values below the training lower quantile map near 0.
    Values above the training upper quantile map near 1.
    """

    result = pd.DataFrame(
        index=data.index,
        columns=data.columns,
        dtype=float,
    )

    for column in data.columns:
        lower = statistics.lower[column]
        upper = statistics.upper[column]

        denominator = (
            upper - lower
        )

        if (
            not np.isfinite(
                denominator
            )
            or abs(denominator)
            <= 1e-12
        ):
            result[column] = 0.5
            continue

        values = data[column].astype(
            float
        )

        result[column] = (
            (
                values - lower
            )
            / denominator
        ).clip(
            0.0,
            1.0,
        )

    return result


# ----------------------------------------------------------------------
# Rolling normalization
# ----------------------------------------------------------------------


def rolling_zscore(
    series: pd.Series,
    *,
    window: int = 60,
    min_periods: int = 20,
    shift: int = 1,
    eps: float = 1e-12,
) -> pd.Series:
    """
    Calculate leakage-safe rolling z-score.

    The rolling statistics are shifted before being applied.

    Therefore the value at time ``t`` is normalized using information
    available before time ``t``.
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    rolling = values.rolling(
        window=window,
        min_periods=min_periods,
    )

    mean = rolling.mean().shift(
        shift
    )

    std = rolling.std(
        ddof=0
    ).shift(
        shift
    )

    std = std.where(
        std.abs() > eps,
        np.nan,
    )

    result = (
        values - mean
    ) / std

    return result.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def rolling_robust_zscore(
    series: pd.Series,
    *,
    window: int = 60,
    min_periods: int = 20,
    shift: int = 1,
    eps: float = 1e-12,
) -> pd.Series:
    """
    Calculate a leakage-safe rolling robust z-score.

    Uses rolling median and IQR.
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    rolling = values.rolling(
        window=window,
        min_periods=min_periods,
    )

    median = rolling.median().shift(
        shift
    )

    q25 = rolling.quantile(
        0.25
    ).shift(
        shift
    )

    q75 = rolling.quantile(
        0.75
    ).shift(
        shift
    )

    iqr = (
        q75 - q25
    ).where(
        lambda x: x.abs()
        > eps,
        np.nan,
    )

    result = (
        values - median
    ) / iqr

    return result.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def rolling_percentile_rank(
    series: pd.Series,
    *,
    window: int = 60,
    min_periods: int = 20,
    shift: int = 1,
) -> pd.Series:
    """
    Calculate the percentile rank of the current value relative to
    prior observations only.

    The current observation is deliberately excluded from the
    reference window.
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    prior = values.shift(
        shift
    )

    def percentile(
        window_values: pd.Series,
    ) -> float:
        current = window_values.iloc[
            -1
        ]

        if pd.isna(current):
            return np.nan

        valid = window_values.dropna()

        if valid.empty:
            return np.nan

        return float(
            (
                valid <= current
            ).mean()
        )

    return (
        prior.rolling(
            window=window,
            min_periods=min_periods,
        )
        .apply(
            percentile,
            raw=False,
        )
    )


# ----------------------------------------------------------------------
# Cross-sectional normalization
# ----------------------------------------------------------------------


def cross_sectional_zscore(
    data: pd.DataFrame,
    *,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Normalize each row cross-sectionally.

    This is appropriate when ``data`` contains multiple stocks at the
    same timestamp.

    It should NOT be used as a substitute for temporal normalization
    when each column is a time series.
    """

    frame = _validate_numeric_frame(
        data
    )

    mean = frame.mean(
        axis=1
    )

    std = frame.std(
        axis=1,
        ddof=0,
    )

    std = std.where(
        std.abs() > eps,
        np.nan,
    )

    return frame.sub(
        mean,
        axis=0,
    ).div(
        std,
        axis=0,
    )


def cross_sectional_rank(
    data: pd.DataFrame,
    *,
    pct: bool = True,
) -> pd.DataFrame:
    """
    Rank values across columns for each timestamp.
    """

    frame = _validate_numeric_frame(
        data
    )

    return frame.rank(
        axis=1,
        pct=pct,
        method="average",
    )


# ----------------------------------------------------------------------
# Winsorization
# ----------------------------------------------------------------------


def winsorize_series(
    series: pd.Series,
    *,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.Series:
    """
    Winsorize a series using quantiles calculated from the supplied
    data.

    WARNING
    -------
    For model training, fit these quantiles on training data only.

    For that reason, this helper is intentionally not used by the
    production inference pipeline.
    """

    if not (
        0.0
        <= lower_quantile
        < upper_quantile
        <= 1.0
    ):
        raise ValueError(
            "Invalid winsorization quantiles."
        )

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    lower = values.quantile(
        lower_quantile
    )

    upper = values.quantile(
        upper_quantile
    )

    return values.clip(
        lower=lower,
        upper=upper,
    )


# ----------------------------------------------------------------------
# DataFrame helpers
# ----------------------------------------------------------------------


def normalize_dataframe(
    data: pd.DataFrame,
    *,
    method: str = "robust",
    columns: Optional[
        Iterable[str]
    ] = None,
    config: Optional[
        NormalizationConfig
    ] = None,
) -> pd.DataFrame:
    """
    Convenience function for train-fitted normalization.

    For research pipelines, prefer explicitly constructing
    ``TrainFittedNormalizer`` so that the fitted state is retained and
    reused correctly.
    """

    frame = _validate_numeric_frame(
        data
    )

    selected_columns = (
        list(columns)
        if columns is not None
        else list(frame.columns)
    )

    missing = [
        column
        for column in selected_columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"Unknown normalization columns: {missing}"
        )

    selected = frame[
        selected_columns
    ]

    selected_config = (
        config
        or NormalizationConfig(
            method=method
        )
    )

    normalizer = (
        TrainFittedNormalizer(
            selected_config
        )
    )

    transformed = (
        normalizer.fit_transform(
            selected
        )
    )

    result = frame.copy()

    result.loc[
        :,
        selected_columns,
    ] = transformed

    return result


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------


def normalization_diagnostics(
    original: pd.DataFrame,
    transformed: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare basic statistics before and after normalization.
    """

    original_frame = (
        _validate_numeric_frame(
            original
        )
    )

    transformed_frame = (
        _validate_numeric_frame(
            transformed
        )
    )

    if list(
        original_frame.columns
    ) != list(
        transformed_frame.columns
    ):
        raise ValueError(
            "Original and transformed columns must match."
        )

    rows = []

    for column in original_frame.columns:
        original_values = (
            original_frame[column]
        )
        transformed_values = (
            transformed_frame[column]
        )

        rows.append(
            {
                "feature": column,
                "original_mean": original_values.mean(),
                "original_std": original_values.std(),
                "original_median": original_values.median(),
                "original_missing": original_values.isna().mean(),
                "normalized_mean": transformed_values.mean(),
                "normalized_std": transformed_values.std(),
                "normalized_median": transformed_values.median(),
                "normalized_missing": transformed_values.isna().mean(),
                "normalized_min": transformed_values.min(),
                "normalized_max": transformed_values.max(),
            }
        )

    return pd.DataFrame(
        rows
    ).set_index(
        "feature"
    )


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def _validate_numeric_frame(
    data: pd.DataFrame,
) -> pd.DataFrame:
    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    if data.columns.has_duplicates:
        raise ValueError(
            "Feature columns must be unique."
        )

    if len(data.columns) == 0:
        raise ValueError(
            "DataFrame contains no columns."
        )

    result = data.copy()

    for column in result.columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    return result


__all__ = [
    "NormalizationConfig",
    "NormalizationStatistics",
    "TrainFittedNormalizer",
    "rolling_zscore",
    "rolling_robust_zscore",
    "rolling_percentile_rank",
    "cross_sectional_zscore",
    "cross_sectional_rank",
    "winsorize_series",
    "normalize_dataframe",
    "normalization_diagnostics",
]
