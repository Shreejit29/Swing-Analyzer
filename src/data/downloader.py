"""
Historical market-data downloader.

This module is responsible ONLY for obtaining raw OHLCV data.
Feature engineering and modelling must happen elsewhere.

Important:
The data source is currently a prototype source. We deliberately keep
the downloader isolated so that a higher-quality historical data provider
can be added later without rewriting the rest of the project.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import yfinance as yf

from .config import DEFAULT_DATA_SOURCE


REQUIRED_COLUMNS = (
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
)


def download_ohlcv(
    symbol: str,
    period: Optional[str] = None,
    interval: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download OHLCV market data for a symbol.

    Parameters
    ----------
    symbol:
        Yahoo Finance ticker symbol, e.g. "RELIANCE.NS".

    period:
        Historical period accepted by the data provider.

    interval:
        Candle interval accepted by the data provider.

    Returns
    -------
    pandas.DataFrame
        Normalised OHLCV dataframe indexed by timestamp.

    Raises
    ------
    ValueError
        If the provider returns no data or required columns are missing.
    """

    period = period or DEFAULT_DATA_SOURCE.daily_period
    interval = interval or "1d"

    if not symbol or not symbol.strip():
        raise ValueError("A valid ticker symbol is required.")

    symbol = symbol.strip().upper()

    download_kwargs = {
        "tickers": symbol,
        "interval": interval,
        "auto_adjust": False,
        "progress": False,
        "group_by": "column",
        "multi_level_index": False,
        "threads": False,
    }

    if start is not None or end is not None:
        download_kwargs["start"] = start
        download_kwargs["end"] = end
    else:
        download_kwargs["period"] = period

    data = yf.download(
        **download_kwargs,
    )

    if data is None or data.empty:
        raise ValueError(
            f"No historical data returned for symbol '{symbol}'."
        )

    data = _normalise_columns(data)

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Historical data for '{symbol}' is missing columns: {missing}"
        )

    data = data.loc[:, REQUIRED_COLUMNS].copy()

    data.index = pd.to_datetime(data.index)

    if data.index.tz is not None:
        data.index = data.index.tz_convert("Asia/Kolkata").tz_localize(None)

    data = data.sort_index()

    # Remove exact duplicate timestamps.
    data = data[~data.index.duplicated(keep="first")]

    # Force numeric OHLCV values.
    for column in REQUIRED_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    return data


def _normalise_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise provider-specific column formats.

    Some versions of yfinance can return MultiIndex columns even for
    a single ticker. This function converts them into simple OHLCV names.
    """

    result = data.copy()

    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [
            column[0] if isinstance(column, tuple) else column
            for column in result.columns
        ]

    result.columns = [
        str(column).strip().title()
        for column in result.columns
    ]

    return result


def download_daily(symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Download daily OHLCV history."""

    return download_ohlcv(
        symbol=symbol,
        period=DEFAULT_DATA_SOURCE.daily_period,
        interval="1d",
        start=start,
        end=end,
    )


def download_weekly(symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Download weekly OHLCV history."""

    return download_ohlcv(
        symbol=symbol,
        period=DEFAULT_DATA_SOURCE.weekly_period,
        interval="1wk",
        start=start,
        end=end,
    )


def download_monthly(symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Download monthly OHLCV history."""

    return download_ohlcv(
        symbol=symbol,
        period=DEFAULT_DATA_SOURCE.monthly_period,
        interval="1mo",
        start=start,
        end=end,
    )


def download_intraday(symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """
    Download prototype intraday history.

    The current provider limits how far historical intraday data can
    be retrieved. This limitation is intentionally exposed rather than
    silently pretending that long-term 4H history exists.
    """

    return download_ohlcv(
        symbol=symbol,
        period=DEFAULT_DATA_SOURCE.intraday_period,
        interval=DEFAULT_DATA_SOURCE.intraday_interval,
        start=start,
        end=end,
    )


class YahooFinanceDownloader:
    """Backward-compatible object adapter over the functional downloader API."""

    def __init__(self, data_source=DEFAULT_DATA_SOURCE) -> None:
        self.data_source = data_source

    def download_daily(self, symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        return download_daily(symbol, start=start, end=end)

    def download_weekly(self, symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        return download_weekly(symbol, start=start, end=end)

    def download_monthly(self, symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        return download_monthly(symbol, start=start, end=end)

    def download_intraday(self, symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        return download_intraday(symbol, start=start, end=end)

    def download_ohlcv(self, symbol: str, period: Optional[str] = None, interval: Optional[str] = None, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        return download_ohlcv(symbol, period=period, interval=interval, start=start, end=end)


def normalize_downloaded_data(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize an already-downloaded OHLCV dataframe."""
    result = data.copy()
    result = _normalise_columns(result)
    result.index = pd.to_datetime(result.index)
    if result.index.tz is not None:
        result.index = result.index.tz_convert("Asia/Kolkata").tz_localize(None)
    result = result.sort_index()
    result = result[~result.index.duplicated(keep="first")]
    for column in REQUIRED_COLUMNS:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.loc[:, [c for c in REQUIRED_COLUMNS if c in result.columns]]
