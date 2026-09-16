"""
Cached market-data downloader for the AI Swing Stock Analyzer.

The Yahoo Finance request is cached for a short period so Streamlit reruns
do not repeatedly make the same network request.
"""

from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st
import yfinance as yf


REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]
DATA_CACHE_TTL = 300  # 5 minutes


def _normalise_column_name(name) -> str:
    return (
        str(name).strip().lower().replace(" ", "_").replace("-", "_")
    )


def _normalise_yahoo_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Yahoo Finance returned no data.")

    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        price_level = None

        for level in range(out.columns.nlevels):
            values = {
                _normalise_column_name(v)
                for v in out.columns.get_level_values(level)
            }
            if len(values.intersection(REQUIRED_COLUMNS)) >= 2:
                price_level = level
                break

        if price_level is None:
            raise ValueError(
                "Unable to identify OHLCV columns returned by Yahoo Finance."
            )

        out.columns = [
            _normalise_column_name(v)
            for v in out.columns.get_level_values(price_level)
        ]
    else:
        out.columns = [
            _normalise_column_name(c) for c in out.columns
        ]

    if "close" not in out.columns and "adj_close" in out.columns:
        out = out.rename(columns={"adj_close": "close"})

    return out


def _get_single_series(df: pd.DataFrame, column_name: str) -> pd.Series:
    matches = [
        c for c in df.columns
        if _normalise_column_name(c) == column_name
    ]

    if not matches:
        raise ValueError(
            f"Yahoo Finance data is missing required column: {column_name}"
        )

    for column in matches:
        value = df[column]

        if isinstance(value, pd.Series):
            return value

        if isinstance(value, pd.DataFrame):
            for sub_column in value.columns:
                series = value[sub_column]
                if isinstance(series, pd.Series):
                    return series

    raise ValueError(
        f"Could not convert Yahoo Finance '{column_name}' to a 1-D Series."
    )


def prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize downloaded or uploaded data into clean OHLCV."""
    if df is None or df.empty:
        raise ValueError("No market data was supplied.")

    normalized = _normalise_yahoo_columns(df)

    out = pd.DataFrame(
        {
            column: _get_single_series(normalized, column)
            for column in REQUIRED_COLUMNS
        },
        index=normalized.index,
    )

    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")

    for column in REQUIRED_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out = out[~out.index.isna()]
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    out = out.dropna(subset=["open", "high", "low", "close"])
    out["volume"] = out["volume"].fillna(0)

    if out.empty:
        raise ValueError("Yahoo Finance returned no usable OHLCV data.")

    return out


@st.cache_data(ttl=DATA_CACHE_TTL, max_entries=128, show_spinner=False)
def _download_cached(
    symbol: str,
    period: str,
    interval: str,
) -> pd.DataFrame:
    """Cached Yahoo Finance network request."""
    raw = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="column",
        timeout=10,
    )
    return prepare_ohlcv(raw)


def download_data(
    symbol: str,
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download market data.

    Identical symbol/period/interval requests are served from the Streamlit
    cache for 5 minutes instead of making another Yahoo Finance request.
    """
    symbol = str(symbol).strip().upper()
    period = str(period).strip()
    interval = str(interval).strip()

    if not symbol:
        raise ValueError("Enter a stock symbol.")

    try:
        return _download_cached(symbol, period, interval)
    except Exception as exc:
        raise RuntimeError(
            f"Yahoo Finance download failed: {exc}"
        ) from exc


def clear_download_cache() -> None:
    """Clear the downloader cache when a fresh download is required."""
    _download_cached.clear()


def load_csv_data(file) -> pd.DataFrame:
    """Load OHLCV data from a CSV file."""
    if file is None:
        raise ValueError("Please provide a CSV file.")

    if isinstance(file, (bytes, bytearray)):
        df = pd.read_csv(BytesIO(file))
    else:
        df = pd.read_csv(file)

    return prepare_ohlcv(df)


__all__ = [
    "download_data",
    "prepare_ohlcv",
    "load_csv_data",
    "clear_download_cache",
]
