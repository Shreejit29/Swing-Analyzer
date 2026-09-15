"""
Historical market-data quality checks.

No dataset should reach the feature-engineering or machine-learning
pipeline before passing these checks.

The purpose of this module is to detect:
- Missing OHLCV values
- Duplicate timestamps
- Invalid OHLC relationships
- Negative/zero prices
- Invalid volumes
- Non-monotonic timestamps
- Suspiciously large price gaps
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
)


@dataclass
class QualityReport:
    """Result of a historical-data quality inspection."""

    rows: int
    duplicate_timestamps: int
    missing_values: int
    invalid_ohlc_rows: int
    invalid_price_rows: int
    invalid_volume_rows: int
    non_monotonic_index: bool
    extreme_return_rows: int
    passed: bool
    errors: List[str]

    def summary(self) -> str:
        """Return a human-readable quality summary."""

        status = "PASS" if self.passed else "FAIL"

        return (
            f"Data quality: {status}\n"
            f"Rows: {self.rows}\n"
            f"Duplicate timestamps: {self.duplicate_timestamps}\n"
            f"Missing values: {self.missing_values}\n"
            f"Invalid OHLC rows: {self.invalid_ohlc_rows}\n"
            f"Invalid price rows: {self.invalid_price_rows}\n"
            f"Invalid volume rows: {self.invalid_volume_rows}\n"
            f"Non-monotonic index: {self.non_monotonic_index}\n"
            f"Extreme return rows: {self.extreme_return_rows}"
        )


def validate_ohlcv(
    data: pd.DataFrame,
    extreme_return_threshold: float = 0.50,
) -> QualityReport:
    """
    Validate an OHLCV dataframe.

    Parameters
    ----------
    data:
        Historical OHLCV dataframe.

    extreme_return_threshold:
        Absolute one-period return above which an observation is flagged.

        Example:
        0.50 means a return greater than +/-50% is flagged.

    Notes
    -----
    Extreme returns are flagged rather than automatically deleted.
    Corporate actions, exchange events, data errors, and genuine
    market events must not be confused with one another.
    """

    errors: List[str] = []

    if data is None:
        raise ValueError("Dataframe cannot be None.")

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    rows = len(data)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        errors.append(
            f"Missing required columns: {missing_columns}"
        )

        return QualityReport(
            rows=rows,
            duplicate_timestamps=0,
            missing_values=0,
            invalid_ohlc_rows=0,
            invalid_price_rows=0,
            invalid_volume_rows=0,
            non_monotonic_index=False,
            extreme_return_rows=0,
            passed=False,
            errors=errors,
        )

    duplicate_timestamps = int(
        data.index.duplicated(keep=False).sum()
    )

    missing_values = int(
        data.loc[:, REQUIRED_COLUMNS].isna().sum().sum()
    )

    invalid_ohlc = (
        (data["High"] < data["Low"])
        | (data["Open"] > data["High"])
        | (data["Open"] < data["Low"])
        | (data["Close"] > data["High"])
        | (data["Close"] < data["Low"])
    )

    invalid_ohlc_rows = int(invalid_ohlc.fillna(False).sum())

    invalid_price = (
        (data["Open"] <= 0)
        | (data["High"] <= 0)
        | (data["Low"] <= 0)
        | (data["Close"] <= 0)
    )

    invalid_price_rows = int(invalid_price.fillna(False).sum())

    invalid_volume = data["Volume"] < 0

    invalid_volume_rows = int(
        invalid_volume.fillna(False).sum()
    )

    non_monotonic_index = not data.index.is_monotonic_increasing

    returns = data["Close"].pct_change()

    extreme_return_rows = int(
        returns.abs()
        .gt(extreme_return_threshold)
        .fillna(False)
        .sum()
    )

    if duplicate_timestamps > 0:
        errors.append(
            "Duplicate timestamps detected."
        )

    if missing_values > 0:
        errors.append(
            "Missing OHLCV values detected."
        )

    if invalid_ohlc_rows > 0:
        errors.append(
            "Invalid OHLC relationships detected."
        )

    if invalid_price_rows > 0:
        errors.append(
            "Zero or negative prices detected."
        )

    if invalid_volume_rows > 0:
        errors.append(
            "Negative volume detected."
        )

    if non_monotonic_index:
        errors.append(
            "Timestamp index is not monotonically increasing."
        )

    if extreme_return_rows > 0:
        errors.append(
            "Extreme one-period returns were flagged for review."
        )

    # Extreme returns are warnings, not automatic failures.
    hard_fail = any(
        [
            duplicate_timestamps > 0,
            missing_values > 0,
            invalid_ohlc_rows > 0,
            invalid_price_rows > 0,
            invalid_volume_rows > 0,
            non_monotonic_index,
        ]
    )

    return QualityReport(
        rows=rows,
        duplicate_timestamps=duplicate_timestamps,
        missing_values=missing_values,
        invalid_ohlc_rows=invalid_ohlc_rows,
        invalid_price_rows=invalid_price_rows,
        invalid_volume_rows=invalid_volume_rows,
        non_monotonic_index=non_monotonic_index,
        extreme_return_rows=extreme_return_rows,
        passed=not hard_fail,
        errors=errors,
    )


def clean_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform only safe structural cleaning.

    This function deliberately does NOT:
    - Forward-fill prices
    - Interpolate market prices
    - Remove extreme returns
    - Winsorize returns

    Such operations can introduce artificial information and must be
    handled explicitly by the research pipeline.
    """

    if data is None or data.empty:
        raise ValueError("Cannot clean an empty dataframe.")

    result = data.copy()

    result.index = pd.to_datetime(result.index)

    if result.index.tz is not None:
        result.index = (
            result.index
            .tz_convert("Asia/Kolkata")
            .tz_localize(None)
        )

    result = result.sort_index()

    result = result[
        ~result.index.duplicated(keep="first")
    ]

    for column in REQUIRED_COLUMNS:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    # Remove rows where the basic OHLC information cannot be trusted.
    result = result.dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    return result


def assert_quality(data: pd.DataFrame) -> QualityReport:
    """
    Validate data and raise an exception if hard quality checks fail.
    """

    report = validate_ohlcv(data)

    if not report.passed:
        raise ValueError(
            "Historical data failed quality checks:\n"
            + "\n".join(
                f"- {error}"
                for error in report.errors
            )
        )

    return report


# Backward-compatible public name used by older data tests/callers.
def normalize_downloaded_data(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize downloaded OHLCV data using the canonical cleaner."""
    return clean_ohlcv(data)
