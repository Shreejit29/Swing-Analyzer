"""
Indian market-context data.

This module downloads benchmark/index data that will later be used
by the stock model to understand the broader market environment.

Current benchmarks:
    NIFTY 50
    SENSEX
    NIFTY Bank

The important design principle is that market information is aligned
to the stock timestamp without using future observations.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from .config import MARKET_SYMBOLS
from .downloader import download_ohlcv


def download_market_data(
    period: str = "max",
) -> Dict[str, pd.DataFrame]:
    """
    Download historical data for major Indian market indices.

    Parameters
    ----------
    period:
        Historical period accepted by the data provider.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Dictionary containing OHLCV data for each benchmark.
    """

    result: Dict[str, pd.DataFrame] = {}

    for name, symbol in MARKET_SYMBOLS.items():
        try:
            result[name] = download_ohlcv(
                symbol=symbol,
                period=period,
                interval="1d",
            )
        except ValueError:
            # Do not silently fabricate missing market information.
            result[name] = pd.DataFrame()

    return result


def close_series(
    market_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Extract benchmark closing prices into one dataframe.

    Output columns:
        NIFTY50_Close
        SENSEX_Close
        NIFTYBANK_Close
    """

    series = []

    for name, data in market_data.items():
        if data.empty or "Close" not in data.columns:
            continue

        close = data["Close"].rename(
            f"{name}_Close"
        )

        series.append(close)

    if not series:
        return pd.DataFrame()

    return pd.concat(
        series,
        axis=1,
    ).sort_index()


def align_market_to_stock(
    stock_data: pd.DataFrame,
    market_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Align market benchmark data to the stock's timestamps.

    Only information that was already available at or before the
    stock observation is used.

    For daily data this means the benchmark series is forward-filled
    only after its latest known observation.

    We intentionally do NOT back-fill market data because back-filling
    could introduce future information into an earlier stock observation.
    """

    if stock_data is None or stock_data.empty:
        raise ValueError(
            "Stock data cannot be empty."
        )

    if not isinstance(
        stock_data.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "Stock data must have a DatetimeIndex."
        )

    benchmark_closes = close_series(
        market_data
    )

    if benchmark_closes.empty:
        return stock_data.copy()

    stock = stock_data.copy().sort_index()

    benchmarks = benchmark_closes.sort_index()

    # Union first, then forward-fill only past-known values.
    combined = pd.concat(
        [stock, benchmarks],
        axis=1,
        join="outer",
    ).sort_index()

    benchmark_columns = [
        column
        for column in benchmarks.columns
        if column in combined.columns
    ]

    combined[benchmark_columns] = (
        combined[benchmark_columns].ffill()
    )

    # Return only rows corresponding to actual stock observations.
    combined = combined.loc[
        combined.index.isin(stock.index)
    ]

    return combined


def add_market_returns(
    data: pd.DataFrame,
    periods: tuple[int, ...] = (1, 3, 5, 10, 20),
) -> pd.DataFrame:
    """
    Add historical returns for each market benchmark.

    These features use only lagged/current information and therefore
    do not directly reference future prices.
    """

    result = data.copy()

    benchmark_columns = [
        column
        for column in result.columns
        if column.endswith("_Close")
    ]

    for column in benchmark_columns:
        prefix = column.removesuffix("_Close")

        for period in periods:
            result[
                f"{prefix}_Return_{period}"
            ] = (
                result[column]
                .pct_change(period)
            )

    return result


def add_market_trend(
    data: pd.DataFrame,
    windows: tuple[int, ...] = (20, 50, 200),
) -> pd.DataFrame:
    """
    Add benchmark trend features.

    Example:

        NIFTY50_Close / NIFTY50_Close_50MA - 1

    Positive values mean the index is above its moving average.
    """

    result = data.copy()

    benchmark_columns = [
        column
        for column in result.columns
        if column.endswith("_Close")
    ]

    for column in benchmark_columns:
        prefix = column.removesuffix("_Close")

        for window in windows:
            moving_average = (
                result[column]
                .rolling(window)
                .mean()
            )

            result[
                f"{prefix}_Trend_{window}"
            ] = (
                result[column]
                / moving_average
                - 1
            )

    return result


def build_market_features(
    stock_data: pd.DataFrame,
    market_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Complete market-feature pipeline.

    Steps:

        Stock data
            ↓
        Benchmark alignment
            ↓
        Market returns
            ↓
        Market trend
    """

    aligned = align_market_to_stock(
        stock_data=stock_data,
        market_data=market_data,
    )

    aligned = add_market_returns(aligned)

    aligned = add_market_trend(aligned)

    return aligned
