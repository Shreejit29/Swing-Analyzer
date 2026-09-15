"""
Central feature engineering pipeline.

This module combines all feature engines into one controlled
pipeline.

Feature groups:
    - Technical indicators
    - Price action
    - Volume
    - Market regime
    - Multi-timeframe context

IMPORTANT:
    Features must be calculated before targets are added.
    Future target columns are never passed into the feature engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

from .technical import (
    add_all_technical_features,
)
from .price_action import (
    add_all_price_action_features,
)
from .volume import (
    add_all_volume_features,
)
from .regime import (
    add_all_regime_features,
)
from .multi_timeframe import (
    add_timeframe_alignment_scores,
    build_multi_timeframe_dataset,
)


@dataclass
class FeatureSet:
    """
    Container for engineered features.

    data:
        Complete dataframe containing OHLCV and engineered
        explanatory variables.

    feature_columns:
        Explicit list of columns that may be supplied to a model.

    excluded_columns:
        Columns that must not be used as model inputs.
    """

    data: pd.DataFrame
    feature_columns: list[str]
    excluded_columns: list[str]


BASE_COLUMNS = {
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
}


def _validate_input(
    data: pd.DataFrame,
) -> None:
    """Validate the base OHLCV dataframe."""

    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    missing = [
        column
        for column in BASE_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing OHLCV columns: {missing}"
        )

    if not isinstance(
        data.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "DataFrame index must be a DatetimeIndex."
        )

    if not data.index.is_monotonic_increasing:
        raise ValueError(
            "DataFrame must be chronologically ordered."
        )


def _remove_duplicate_columns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove duplicate column names.

    Feature engines are intentionally modular, so two engines may
    occasionally calculate the same underlying feature. Keeping
    only the first version prevents ambiguous model inputs.
    """

    return data.loc[
        :,
        ~data.columns.duplicated(
            keep="first"
        ),
    ]


def _identify_excluded_columns(
    data: pd.DataFrame,
) -> list[str]:
    """
    Identify columns that must never be supplied to a model.

    This includes:
        - raw target columns
        - future return columns
        - direction labels
        - helper/source columns
        - symbol identifiers
    """

    excluded = []

    for column in data.columns:

        if column in BASE_COLUMNS:
            continue

        if column.startswith(
            "Future_"
        ):
            excluded.append(column)
            continue

        if column.startswith(
            "Direction_"
        ):
            excluded.append(column)
            continue

        if column in {
            "Target",
            "Symbol",
            "_source_time",
            "_base_time",
        }:
            excluded.append(column)

    return excluded


def _identify_feature_columns(
    data: pd.DataFrame,
    excluded: list[str],
) -> list[str]:
    """
    Identify safe model input columns.

    Only numeric columns are returned.
    """

    excluded_set = set(
        excluded
    )

    features = []

    for column in data.columns:

        if column in BASE_COLUMNS:
            # Raw OHLCV can be useful to some models, but we will
            # later decide whether normalized versions are preferable.
            features.append(column)
            continue

        if column in excluded_set:
            continue

        if pd.api.types.is_numeric_dtype(
            data[column]
        ):
            features.append(column)

    return features


def engineer_features(
    data: pd.DataFrame,
    include_technical: bool = True,
    include_price_action: bool = True,
    include_volume: bool = True,
    include_regime: bool = True,
) -> FeatureSet:
    """
    Run the core feature-engineering pipeline.

    No future targets should be present in `data` when this function
    is called.
    """

    _validate_input(data)

    result = data.copy()

    # ---------------------------------------------------------
    # Safety check
    # ---------------------------------------------------------

    future_columns = [
        column
        for column in result.columns
        if (
            column.startswith("Future_")
            or column.startswith("Direction_")
        )
    ]

    if future_columns:
        raise ValueError(
            "Future target columns were supplied to the feature "
            f"engine: {future_columns}"
        )

    # ---------------------------------------------------------
    # Technical indicators
    # ---------------------------------------------------------

    if include_technical:
        result = (
            add_all_technical_features(
                result
            )
        )

    # ---------------------------------------------------------
    # Price action
    # ---------------------------------------------------------

    if include_price_action:
        result = (
            add_all_price_action_features(
                result
            )
        )

    # ---------------------------------------------------------
    # Volume
    # ---------------------------------------------------------

    if include_volume:
        result = (
            add_all_volume_features(
                result
            )
        )

    # ---------------------------------------------------------
    # Market regime
    # ---------------------------------------------------------

    if include_regime:
        result = (
            add_all_regime_features(
                result
            )
        )

    # ---------------------------------------------------------
    # Duplicate-column protection
    # ---------------------------------------------------------

    result = _remove_duplicate_columns(
        result
    )

    # ---------------------------------------------------------
    # Explicit feature list
    # ---------------------------------------------------------

    excluded = (
        _identify_excluded_columns(
            result
        )
    )

    features = (
        _identify_feature_columns(
            result,
            excluded,
        )
    )

    return FeatureSet(
        data=result,
        feature_columns=features,
        excluded_columns=excluded,
    )


def engineer_multitimeframe_features(
    data_4h: pd.DataFrame,
    data_1d: pd.DataFrame,
    data_1w: pd.DataFrame,
    data_1m: pd.DataFrame,
) -> FeatureSet:
    """
    Build leakage-safe multi-timeframe features.

    Base timeframe:
        4H

    Context:
        1D
        1W
        1M

    Only completed higher-timeframe candles are aligned.
    """

    result = build_multi_timeframe_dataset(
        data_4h=data_4h,
        data_1d=data_1d,
        data_1w=data_1w,
        data_1m=data_1m,
    )

    result = (
        add_timeframe_alignment_scores(
            result
        )
    )

    result = _remove_duplicate_columns(
        result
    )

    excluded = (
        _identify_excluded_columns(
            result
        )
    )

    features = (
        _identify_feature_columns(
            result,
            excluded,
        )
    )

    return FeatureSet(
        data=result,
        feature_columns=features,
        excluded_columns=excluded,
    )


def add_feature_metadata(
    feature_set: FeatureSet,
) -> pd.DataFrame:
    """
    Create a compact metadata table describing the feature set.

    Useful later for:
        - feature selection
        - feature importance
        - stability analysis
        - model diagnostics
    """

    rows = []

    for column in feature_set.data.columns:

        is_feature = (
            column
            in feature_set.feature_columns
        )

        is_excluded = (
            column
            in feature_set.excluded_columns
        )

        dtype = str(
            feature_set.data[
                column
            ].dtype
        )

        rows.append(
            {
                "feature": column,
                "dtype": dtype,
                "is_model_feature": is_feature,
                "is_excluded": is_excluded,
                "missing_count": int(
                    feature_set.data[
                        column
                    ].isna().sum()
                ),
                "missing_fraction": float(
                    feature_set.data[
                        column
                    ].isna().mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def remove_high_missing_features(
    feature_set: FeatureSet,
    maximum_missing_fraction: float = 0.40,
) -> FeatureSet:
    """
    Remove features with excessive missing values.

    IMPORTANT:
    This only removes columns. It does not impute values.

    Imputation, if eventually required, must be fitted inside
    each training fold to avoid validation/test leakage.
    """

    data = feature_set.data.copy()

    valid_features = []

    for column in feature_set.feature_columns:

        missing_fraction = (
            data[column]
            .isna()
            .mean()
        )

        if (
            missing_fraction
            <= maximum_missing_fraction
        ):
            valid_features.append(
                column
            )

    removed = set(
        feature_set.feature_columns
    ) - set(valid_features)

    excluded = list(
        set(
            feature_set.excluded_columns
        ).union(removed)
    )

    return FeatureSet(
        data=data,
        feature_columns=valid_features,
        excluded_columns=excluded,
    )


def remove_constant_features(
    feature_set: FeatureSet,
) -> FeatureSet:
    """
    Remove features with no variance.

    Constant features provide no useful information to most
    supervised-learning models.
    """

    data = feature_set.data.copy()

    valid_features = []

    removed = set()

    for column in feature_set.feature_columns:

        non_null = (
            data[column]
            .dropna()
        )

        if non_null.empty:
            removed.add(column)
            continue

        if (
            non_null.nunique()
            <= 1
        ):
            removed.add(column)
            continue

        valid_features.append(
            column
        )

    excluded = list(
        set(
            feature_set.excluded_columns
        ).union(removed)
    )

    return FeatureSet(
        data=data,
        feature_columns=valid_features,
        excluded_columns=excluded,
    )


def prepare_feature_set(
    data: pd.DataFrame,
    maximum_missing_fraction: float = 0.40,
) -> FeatureSet:
    """
    Convenience function for the standard feature pipeline.

    Steps:

        1. Engineer features
        2. Remove high-missing features
        3. Remove constant features

    No imputation is performed here.
    """

    feature_set = engineer_features(
        data
    )

    feature_set = (
        remove_high_missing_features(
            feature_set,
            maximum_missing_fraction=(
                maximum_missing_fraction
            ),
        )
    )

    feature_set = (
        remove_constant_features(
            feature_set
        )
    )

    return feature_set


def feature_matrix(
    feature_set: FeatureSet,
) -> pd.DataFrame:
    """
    Return the model feature matrix.

    No target columns are included.
    """

    return feature_set.data[
        feature_set.feature_columns
    ].copy()


def validate_feature_matrix(
    feature_set: FeatureSet,
) -> None:
    """
    Run strict structural checks before modelling.
    """

    matrix = feature_matrix(
        feature_set
    )

    if matrix.empty:
        raise ValueError(
            "Feature matrix is empty."
        )

    non_numeric = [
        column
        for column in matrix.columns
        if not pd.api.types.is_numeric_dtype(
            matrix[column]
        )
    ]

    if non_numeric:
        raise TypeError(
            "Non-numeric model features found: "
            f"{non_numeric}"
        )

    duplicate_columns = (
        matrix.columns[
            matrix.columns.duplicated()
        ]
        .tolist()
    )

    if duplicate_columns:
        raise ValueError(
            "Duplicate feature columns found: "
            f"{duplicate_columns}"
        )

    future_columns = [
        column
        for column in matrix.columns
        if (
            column.startswith("Future_")
            or column.startswith("Direction_")
        )
    ]

    if future_columns:
        raise ValueError(
            "Potential future leakage detected in feature "
            f"matrix: {future_columns}"
        )
