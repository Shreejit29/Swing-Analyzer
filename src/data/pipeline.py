"""
Historical data pipeline.

This module connects the individual data components:

    Downloader
        ↓
    Cleaning
        ↓
    Quality checks
        ↓
    Timeframe construction
        ↓
    Research-ready dataset

The pipeline deliberately keeps raw data and processed data separate.

No machine-learning model should download market data directly.
All model data should pass through this pipeline first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from .downloader import (
    download_daily,
    download_weekly,
    download_monthly,
    download_intraday,
)
from .quality import (
    QualityReport,
    clean_ohlcv,
    validate_ohlcv,
)
from .resample import resample_4h


@dataclass
class TimeframeData:
    """
    Container for all available timeframes of one instrument.
    """

    four_hour: pd.DataFrame
    daily: pd.DataFrame
    weekly: pd.DataFrame
    monthly: pd.DataFrame


@dataclass
class PipelineResult:
    """
    Complete output of the historical-data pipeline.
    """

    symbol: str
    data: TimeframeData
    quality_reports: Dict[str, QualityReport]


def prepare_dataframe(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, QualityReport]:
    """
    Clean and validate one OHLCV dataframe.

    Extreme returns are reported but are not automatically removed.
    """

    cleaned = clean_ohlcv(data)

    report = validate_ohlcv(cleaned)

    return cleaned, report


def build_historical_dataset(
    symbol: str,
) -> PipelineResult:
    """
    Download and prepare historical data for one stock.

    Parameters
    ----------
    symbol:
        Yahoo Finance ticker, e.g. "RELIANCE.NS".

    Returns
    -------
    PipelineResult
        Cleaned data for 4H, 1D, 1W and 1M timeframes plus
        data-quality reports.

    Notes
    -----
    The 4H timeframe is constructed from the available intraday
    data. It is NOT fabricated from daily observations.
    """

    symbol = symbol.strip().upper()

    if not symbol:
        raise ValueError("A stock symbol is required.")

    # ---------------------------------------------------------
    # 1. Download raw datasets
    # ---------------------------------------------------------

    raw_daily = download_daily(symbol)
    raw_weekly = download_weekly(symbol)
    raw_monthly = download_monthly(symbol)
    raw_intraday = download_intraday(symbol)

    # ---------------------------------------------------------
    # 2. Clean + validate standard timeframes
    # ---------------------------------------------------------

    daily, daily_report = prepare_dataframe(raw_daily)

    weekly, weekly_report = prepare_dataframe(raw_weekly)

    monthly, monthly_report = prepare_dataframe(raw_monthly)

    # ---------------------------------------------------------
    # 3. Clean intraday data before constructing 4H candles
    # ---------------------------------------------------------

    intraday = clean_ohlcv(raw_intraday)

    intraday_report = validate_ohlcv(intraday)

    # ---------------------------------------------------------
    # 4. Construct 4H candles
    # ---------------------------------------------------------

    if intraday.empty:
        four_hour = pd.DataFrame(
            columns=[
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]
        )
    else:
        four_hour = resample_4h(intraday)

    # ---------------------------------------------------------
    # 5. Validate constructed 4H data
    # ---------------------------------------------------------

    if four_hour.empty:
        four_hour_report = QualityReport(
            rows=0,
            duplicate_timestamps=0,
            missing_values=0,
            invalid_ohlc_rows=0,
            invalid_price_rows=0,
            invalid_volume_rows=0,
            non_monotonic_index=False,
            extreme_return_rows=0,
            passed=False,
            errors=[
                "No usable 4H candles could be constructed "
                "from the available intraday history."
            ],
        )
    else:
        four_hour_report = validate_ohlcv(four_hour)

    # ---------------------------------------------------------
    # 6. Package quality reports
    # ---------------------------------------------------------

    reports = {
        "4H": four_hour_report,
        "1D": daily_report,
        "1W": weekly_report,
        "1M": monthly_report,
        "intraday_source": intraday_report,
    }

    timeframe_data = TimeframeData(
        four_hour=four_hour,
        daily=daily,
        weekly=weekly,
        monthly=monthly,
    )

    return PipelineResult(
        symbol=symbol,
        data=timeframe_data,
        quality_reports=reports,
    )


def quality_summary(
    result: PipelineResult,
) -> pd.DataFrame:
    """
    Convert pipeline quality reports into a dataframe suitable
    for a Streamlit dashboard or research report.
    """

    rows = []

    for timeframe, report in result.quality_reports.items():
        rows.append(
            {
                "Timeframe": timeframe,
                "Rows": report.rows,
                "Duplicates": report.duplicate_timestamps,
                "Missing Values": report.missing_values,
                "Invalid OHLC": report.invalid_ohlc_rows,
                "Invalid Prices": report.invalid_price_rows,
                "Invalid Volume": report.invalid_volume_rows,
                "Non-Monotonic": report.non_monotonic_index,
                "Extreme Returns": report.extreme_return_rows,
                "Passed": report.passed,
            }
        )

    return pd.DataFrame(rows)
