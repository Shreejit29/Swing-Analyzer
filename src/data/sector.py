"""
Indian sector-context data.

A stock should not be analysed independently from its sector.

For example, a bullish technical setup in a stock belonging to a
weak sector may behave differently from the same setup in a strong
sector.

This module provides a clean framework for adding sector context
without leaking future information.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from .downloader import download_ohlcv


# Initial sector universe.
#
# These are benchmark symbols used as sector-context proxies.
# The mapping can be expanded as the project develops.
SECTOR_SYMBOLS: Dict[str, str] = {
    "NIFTY_IT": "^CNXIT",
    "NIFTY_BANK": "^NSEBANK",
    "NIFTY_AUTO": "^CNXAUTO",
    "NIFTY_PHARMA": "^CNXPHARMA",
    "NIFTY_FMCG": "^CNXFMCG",
    "NIFTY_METAL": "^CNXMETAL",
    "NIFTY_REALTY": "^CNXREALTY",
    "NIFTY_ENERGY": "^CNXENERGY",
    "NIFTY_INFRA": "^CNXINFRA",
    "NIFTY_PSU_BANK": "^CNXPSUBANK",
}


def download_sector_data(
    period: str = "max",
) -> Dict[str, pd.DataFrame]:
    """
    Download historical OHLCV data for sector indices.

    Failed individual downloads are represented by empty dataframes
    rather than fabricated data.
    """

    result: Dict[str, pd.DataFrame] = {}

    for sector_name, symbol in SECTOR_SYMBOLS.items():
        try:
            result[sector_name] = download_ohlcv(
                symbol=symbol,
                period=period,
                interval="1d",
            )
        except ValueError:
            result[sector_name] = pd.DataFrame()

    return result


def sector_close_series(
    sector_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Extract sector closing prices into one dataframe.
    """

    series = []

    for sector_name, data in sector_data.items():
        if data.empty or "Close" not in data.columns:
            continue

        close = data["Close"].rename(
            f"{sector_name}_Close"
        )

        series.append(close)

    if not series:
        return pd.DataFrame()

    return pd.concat(
        series,
        axis=1,
    ).sort_index()


def align_sector_to_stock(
    stock_data: pd.DataFrame,
    sector_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Align sector data to stock timestamps.

    Forward-filling is allowed only after an index observation is known.

    Back-filling is deliberately avoided because it could introduce
    information from the future into historical stock observations.
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

    sectors = sector_close_series(
        sector_data
    )

    if sectors.empty:
        return stock_data.copy()

    stock = stock_data.copy().sort_index()

    sectors = sectors.sort_index()

    combined = pd.concat(
        [stock, sectors],
        axis=1,
        join="outer",
    ).sort_index()

    sector_columns = [
        column
        for column in sectors.columns
        if column in combined.columns
    ]

    combined[sector_columns] = (
        combined[sector_columns]
        .ffill()
    )

    combined = combined.loc[
        combined.index.isin(stock.index)
    ]

    return combined


def add_sector_returns(
    data: pd.DataFrame,
    periods: tuple[int, ...] = (
        1,
        3,
        5,
        10,
        20,
    ),
) -> pd.DataFrame:
    """
    Add historical sector returns.

    These are explanatory features for the stock model.
    """

    result = data.copy()

    sector_columns = [
        column
        for column in result.columns
        if column.endswith("_Close")
        and column.startswith("NIFTY_")
    ]

    for column in sector_columns:
        prefix = column.removesuffix("_Close")

        for period in periods:
            result[
                f"{prefix}_Return_{period}"
            ] = (
                result[column]
                .pct_change(period)
            )

    return result


def add_sector_trend(
    data: pd.DataFrame,
    windows: tuple[int, ...] = (
        20,
        50,
        200,
    ),
) -> pd.DataFrame:
    """
    Add sector trend features.

    A positive value means the sector index is above its moving average.
    """

    result = data.copy()

    sector_columns = [
        column
        for column in result.columns
        if column.endswith("_Close")
        and column.startswith("NIFTY_")
    ]

    for column in sector_columns:
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


def add_sector_momentum(
    data: pd.DataFrame,
    period: int = 20,
) -> pd.DataFrame:
    """
    Add a simple sector momentum measure.

    This represents the sector's historical return over the selected
    lookback period.
    """

    result = data.copy()

    sector_columns = [
        column
        for column in result.columns
        if column.endswith("_Close")
        and column.startswith("NIFTY_")
    ]

    for column in sector_columns:
        prefix = column.removesuffix("_Close")

        result[
            f"{prefix}_Momentum_{period}"
        ] = (
            result[column]
            / result[column].shift(period)
            - 1
        )

    return result


def build_sector_features(
    stock_data: pd.DataFrame,
    sector_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Complete sector-feature pipeline.

    Steps:

        Stock data
            ↓
        Sector alignment
            ↓
        Sector returns
            ↓
        Sector trend
            ↓
        Sector momentum
    """

    result = align_sector_to_stock(
        stock_data=stock_data,
        sector_data=sector_data,
    )

    result = add_sector_returns(result)

    result = add_sector_trend(result)

    result = add_sector_momentum(result)

    return result
