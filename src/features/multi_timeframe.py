"""
Multi-timeframe feature engine.

Combines information from:
    - 4H
    - 1D
    - 1W
    - 1M

IMPORTANT:
Higher-timeframe information must never be allowed to leak into
an earlier lower-timeframe observation.

Therefore all higher-timeframe values are aligned using the last
COMPLETED candle available before the current timestamp.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


def _validate_dataframe(
    data: pd.DataFrame,
    name: str,
) -> None:
    """Validate a timeframe dataframe."""

    if data is None or data.empty:
        raise ValueError(
            f"{name} cannot be empty."
        )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing columns: {missing}"
        )

    if not isinstance(
        data.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            f"{name} must use a DatetimeIndex."
        )

    if not data.index.is_monotonic_increasing:
        raise ValueError(
            f"{name} must be chronologically ordered."
        )


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Safely divide two series."""

    return numerator / denominator.replace(
        0,
        np.nan,
    )


def _prepare_timeframe(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize timeframe data."""

    result = data.copy()

    result.index = pd.to_datetime(
        result.index
    )

    if result.index.tz is not None:
        result.index = (
            result.index
            .tz_convert("Asia/Kolkata")
            .tz_localize(None)
        )

    result = result.sort_index()

    result = result[
        ~result.index.duplicated(
            keep="last"
        )
    ]

    return result


def _completed_candle_features(
    data: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """
    Generate compact features from a timeframe.

    The values describe the candle itself. When these features are
    subsequently aligned to a lower timeframe, the candle is shifted
    so only completed higher-timeframe candles can be used.
    """

    result = pd.DataFrame(
        index=data.index
    )

    close = data["Close"]
    high = data["High"]
    low = data["Low"]

    ema_20 = (
        close
        .ewm(
            span=20,
            adjust=False,
            min_periods=20,
        )
        .mean()
    )

    ema_50 = (
        close
        .ewm(
            span=50,
            adjust=False,
            min_periods=50,
        )
        .mean()
    )

    ema_200 = (
        close
        .ewm(
            span=200,
            adjust=False,
            min_periods=200,
        )
        .mean()
    )

    returns_1 = (
        close.pct_change()
    )

    returns_5 = (
        close.pct_change(5)
    )

    returns_20 = (
        close.pct_change(20)
    )

    volatility_20 = (
        returns_1
        .rolling(
            20,
            min_periods=20,
        )
        .std()
    )

    rolling_high_20 = (
        high
        .rolling(
            20,
            min_periods=20,
        )
        .max()
    )

    rolling_low_20 = (
        low
        .rolling(
            20,
            min_periods=20,
        )
        .min()
    )

    result[
        f"{prefix}_Close"
    ] = close

    result[
        f"{prefix}_Return_1"
    ] = returns_1

    result[
        f"{prefix}_Return_5"
    ] = returns_5

    result[
        f"{prefix}_Return_20"
    ] = returns_20

    result[
        f"{prefix}_Distance_EMA20"
    ] = (
        _safe_divide(
            close,
            ema_20,
        )
        - 1
    )

    result[
        f"{prefix}_Distance_EMA50"
    ] = (
        _safe_divide(
            close,
            ema_50,
        )
        - 1
    )

    result[
        f"{prefix}_Distance_EMA200"
    ] = (
        _safe_divide(
            close,
            ema_200,
        )
        - 1
    )

    result[
        f"{prefix}_EMA20_EMA50_Spread"
    ] = (
        _safe_divide(
            ema_20,
            ema_50,
        )
        - 1
    )

    result[
        f"{prefix}_EMA50_EMA200_Spread"
    ] = (
        _safe_divide(
            ema_50,
            ema_200,
        )
        - 1
    )

    result[
        f"{prefix}_Volatility_20"
    ] = volatility_20

    result[
        f"{prefix}_Range_Position_20"
    ] = _safe_divide(
        close - rolling_low_20,
        rolling_high_20 - rolling_low_20,
    )

    return result


def _align_completed_features(
    base_index: pd.DatetimeIndex,
    higher_features: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """
    Align higher-timeframe features to a lower timeframe.

    A higher-timeframe candle becomes available only AFTER that
    candle closes.

    Example:

        Daily candle dated Monday
        -> cannot be used during Monday's intraday observations
        -> becomes available from the next valid observation.

    The shift(1) below enforces this rule.
    """

    features = higher_features.copy()

    features = features.shift(1)

    features = features.reset_index()

    index_name = features.columns[0]

    features = features.rename(
        columns={
            index_name: "_source_time"
        }
    )

    base = pd.DataFrame(
        {
            "_base_time": base_index
        }
    )

    features = features.sort_values(
        "_source_time"
    )

    base = base.sort_values(
        "_base_time"
    )

    aligned = pd.merge_asof(
        base,
        features,
        left_on="_base_time",
        right_on="_source_time",
        direction="backward",
        allow_exact_matches=True,
    )

    aligned = aligned.set_index(
        "_base_time"
    )

    if "_source_time" in aligned.columns:
        aligned = aligned.drop(
            columns=["_source_time"]
        )

    return aligned


def add_timeframe_features(
    base_data: pd.DataFrame,
    higher_timeframes: Dict[
        str,
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    """
    Add higher-timeframe context to a base timeframe.

    Parameters
    ----------
    base_data:
        The timeframe on which the model will make observations.

    higher_timeframes:
        Dictionary such as:

            {
                "1D": daily_data,
                "1W": weekly_data,
                "1M": monthly_data,
            }

    The base timeframe itself should normally not be included in
    higher_timeframes.
    """

    _validate_dataframe(
        base_data,
        "base_data",
    )

    result = _prepare_timeframe(
        base_data
    )

    for timeframe, timeframe_data in (
        higher_timeframes.items()
    ):

        _validate_dataframe(
            timeframe_data,
            timeframe,
        )

        prepared = _prepare_timeframe(
            timeframe_data
        )

        features = (
            _completed_candle_features(
                prepared,
                timeframe,
            )
        )

        aligned = (
            _align_completed_features(
                result.index,
                features,
                timeframe,
            )
        )

        result = result.join(
            aligned,
            how="left",
        )

    return result


def add_same_timeframe_features(
    data: pd.DataFrame,
    timeframe: str,
) -> pd.DataFrame:
    """
    Add features from the current timeframe.

    These are shifted by one observation so that the model can
    represent information known before the current candle closes.

    This is especially useful when the prediction is intended to
    be made BEFORE the current candle is complete.
    """

    _validate_dataframe(
        data,
        "data",
    )

    prepared = _prepare_timeframe(
        data
    )

    features = (
        _completed_candle_features(
            prepared,
            timeframe,
        )
    )

    return features.shift(1)


def build_multi_timeframe_dataset(
    data_4h: pd.DataFrame,
    data_1d: pd.DataFrame,
    data_1w: pd.DataFrame,
    data_1m: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the complete multi-timeframe feature dataset.

    Base timeframe:
        4H

    Higher-timeframe context:
        1D
        1W
        1M

    All higher-timeframe values represent COMPLETED candles.
    """

    result = add_same_timeframe_features(
        data_4h,
        "4H",
    )

    higher_timeframes = {
        "1D": data_1d,
        "1W": data_1w,
        "1M": data_1m,
    }

    result = add_timeframe_features(
        base_data=data_4h,
        higher_timeframes=higher_timeframes,
    )

    # Add the current 4H features after alignment.
    current_4h_features = (
        add_same_timeframe_features(
            data_4h,
            "4H",
        )
    )

    for column in current_4h_features.columns:
        result[column] = (
            current_4h_features[column]
        )

    return result


def add_timeframe_alignment_scores(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate agreement between timeframes.

    Positive values indicate bullish alignment.
    Negative values indicate bearish alignment.

    These are descriptive features, not trading signals.
    """

    result = data.copy()

    trend_columns = []

    for timeframe in (
        "4H",
        "1D",
        "1W",
        "1M",
    ):
        column = (
            f"{timeframe}_Distance_EMA50"
        )

        if column in result.columns:
            trend_columns.append(
                np.sign(
                    result[column]
                )
            )

    if trend_columns:

        trend_matrix = pd.concat(
            trend_columns,
            axis=1,
        )

        result[
            "Timeframe_Trend_Alignment"
        ] = (
            trend_matrix.mean(axis=1)
        )

    momentum_columns = []

    for timeframe in (
        "4H",
        "1D",
        "1W",
        "1M",
    ):
        column = (
            f"{timeframe}_Return_5"
        )

        if column in result.columns:
            momentum_columns.append(
                np.sign(
                    result[column]
                )
            )

    if momentum_columns:

        momentum_matrix = pd.concat(
            momentum_columns,
            axis=1,
        )

        result[
            "Timeframe_Momentum_Alignment"
        ] = (
            momentum_matrix.mean(axis=1)
        )

    return result


def multi_timeframe_feature_names(
    data: pd.DataFrame,
) -> list[str]:
    """
    Return multi-timeframe feature names.
    """

    keywords = (
        "4H_",
        "1D_",
        "1W_",
        "1M_",
        "Timeframe_",
    )

    return [
        column
        for column in data.columns
        if column.startswith(keywords)
    ]
