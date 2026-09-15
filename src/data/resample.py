"""
Timeframe resampling utilities.

The model will eventually work across:
    4H -> short-term swing timing
    1D -> primary swing direction
    1W -> medium-term trend
    1M -> long-term market structure

This module converts lower-timeframe OHLCV data into higher-timeframe
candles without using future observations.

IMPORTANT:
4-hour candles must be constructed from intraday data. We must never
create fake historical 4H candles by interpolating daily prices.
"""

from __future__ import annotations

import pandas as pd


OHLCV_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


def resample_4h(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert intraday OHLCV data into 4-hour candles.

    Parameters
    ----------
    data:
        Intraday OHLCV dataframe with a DatetimeIndex.

    Returns
    -------
    pandas.DataFrame
        4-hour OHLCV candles.

    Notes
    -----
    The operation uses:

        Open   -> first observation
        High   -> maximum
        Low    -> minimum
        Close  -> last observation
        Volume -> sum

    Empty/incomplete buckets are removed.
    """

    _validate_input(data)

    result = data.copy()

    result = result.sort_index()

    # Remove duplicate timestamps before aggregation.
    result = result[
        ~result.index.duplicated(keep="first")
    ]

    candles = (
        result[OHLCV_COLUMNS]
        .resample(
            "4h",
            origin="start_day",
            label="left",
            closed="left",
        )
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
    )

    # A candle without a valid open/high/low/close is not usable.
    candles = candles.dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    candles = candles.sort_index()

    return candles


def resample_daily(data: pd.DataFrame) -> pd.DataFrame:
    """
    Resample intraday OHLCV data into daily candles.

    This function is primarily useful when a provider supplies
    intraday data but the research pipeline needs independently
    constructed daily candles.
    """

    _validate_input(data)

    result = data.copy().sort_index()

    candles = (
        result[OHLCV_COLUMNS]
        .resample(
            "1D",
            label="left",
            closed="left",
        )
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
    )

    candles = candles.dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    return candles.sort_index()


def resample_weekly(data: pd.DataFrame) -> pd.DataFrame:
    """
    Resample OHLCV data into weekly candles.

    Weeks are anchored to Monday.
    """

    _validate_input(data)

    result = data.copy().sort_index()

    candles = (
        result[OHLCV_COLUMNS]
        .resample(
            "W-MON",
            label="left",
            closed="left",
        )
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
    )

    candles = candles.dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    return candles.sort_index()


def resample_monthly(data: pd.DataFrame) -> pd.DataFrame:
    """
    Resample OHLCV data into calendar-month candles.
    """

    _validate_input(data)

    result = data.copy().sort_index()

    candles = (
        result[OHLCV_COLUMNS]
        .resample(
            "MS",
            label="left",
            closed="left",
        )
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
    )

    candles = candles.dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    return candles.sort_index()


def _validate_input(data: pd.DataFrame) -> None:
    """
    Validate the dataframe before resampling.
    """

    if data is None:
        raise ValueError("Input data cannot be None.")

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    if data.empty:
        raise ValueError(
            "Cannot resample an empty dataframe."
        )

    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError(
            "Dataframe index must be a pandas DatetimeIndex."
        )

    missing = [
        column
        for column in OHLCV_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing OHLCV columns: {missing}"
        )

    if not data.index.is_monotonic_increasing:
        raise ValueError(
            "Input timestamps must be sorted "
            "in increasing order."
        )
