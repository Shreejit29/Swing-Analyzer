"""
Indian market trading-session utilities.

The model must understand that Indian equities trade during defined
exchange sessions rather than continuously throughout the day.

This module provides:
    - NSE regular-session boundaries
    - trading-day checks
    - session filtering
    - session-aware timestamp utilities
    - validation helpers for market data

The module intentionally avoids assuming that weekends/holidays are
valid trading sessions.

A future exchange-calendar integration can replace the holiday logic
without changing the rest of the application.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Iterable, Optional

import pandas as pd


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------


INDIA_TIMEZONE = "Asia/Kolkata"

NSE_MARKET_OPEN = time(
    hour=9,
    minute=15,
)

NSE_MARKET_CLOSE = time(
    hour=15,
    minute=30,
)


# ----------------------------------------------------------------------
# Session configuration
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TradingSessionConfig:
    """
    Trading-session configuration.

    The default represents the regular NSE equity session.
    """

    timezone: str = INDIA_TIMEZONE

    market_open: time = NSE_MARKET_OPEN

    market_close: time = NSE_MARKET_CLOSE

    exchange: str = "NSE"

    include_weekends: bool = False

    def __post_init__(self) -> None:
        if self.market_open >= self.market_close:
            raise ValueError(
                "market_open must be earlier than market_close."
            )

        if not self.exchange.strip():
            raise ValueError(
                "exchange cannot be empty."
            )


DEFAULT_SESSION = TradingSessionConfig()


# ----------------------------------------------------------------------
# Holiday calendar
# ----------------------------------------------------------------------


class NSEHolidayCalendar:
    """
    Minimal holiday-calendar interface.

    This class deliberately keeps the interface small so that a proper
    NSE exchange calendar can be plugged in later.

    ``holidays`` may be supplied as strings, dates, or timestamps.
    """

    def __init__(
        self,
        holidays: Optional[
            Iterable[
                date | datetime | str | pd.Timestamp
            ]
        ] = None,
    ) -> None:
        self._holidays = set()

        for value in holidays or []:
            self._holidays.add(
                pd.Timestamp(
                    value
                ).date()
            )

    def is_holiday(
        self,
        value: date | datetime | pd.Timestamp,
    ) -> bool:
        return (
            pd.Timestamp(
                value
            ).date()
            in self._holidays
        )

    def add(
        self,
        value: date | datetime | pd.Timestamp,
    ) -> None:
        self._holidays.add(
            pd.Timestamp(
                value
            ).date()
        )

    def dates(
        self,
    ) -> set[date]:
        return set(
            self._holidays
        )


# ----------------------------------------------------------------------
# Timestamp normalization
# ----------------------------------------------------------------------


def normalize_market_timestamp(
    timestamp: pd.Timestamp | datetime | str,
    *,
    timezone: str = INDIA_TIMEZONE,
) -> pd.Timestamp:
    """
    Convert a timestamp to timezone-naive IST.

    Internally, our historical research datasets use timezone-naive
    timestamps after conversion to Asia/Kolkata.
    """

    ts = pd.Timestamp(
        timestamp
    )

    if ts.tzinfo is not None:
        ts = ts.tz_convert(
            timezone
        ).tz_localize(None)

    return ts


def normalize_market_index(
    index: pd.DatetimeIndex,
    *,
    timezone: str = INDIA_TIMEZONE,
) -> pd.DatetimeIndex:
    """
    Normalize a DatetimeIndex to timezone-naive IST.
    """

    if not isinstance(
        index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "index must be a pandas DatetimeIndex."
        )

    if index.tz is not None:
        index = (
            index.tz_convert(
                timezone
            ).tz_localize(None)
        )

    return index


# ----------------------------------------------------------------------
# Session checks
# ----------------------------------------------------------------------


def is_weekend(
    value: date | datetime | pd.Timestamp,
) -> bool:
    """
    Return True for Saturday or Sunday.
    """

    return (
        pd.Timestamp(
            value
        ).weekday()
        >= 5
    )


def is_regular_session_time(
    timestamp: pd.Timestamp | datetime | str,
    *,
    config: TradingSessionConfig = DEFAULT_SESSION,
) -> bool:
    """
    Check whether a timestamp lies within the regular NSE session.

    Boundary times are included.
    """

    ts = normalize_market_timestamp(
        timestamp,
        timezone=config.timezone,
    )

    current_time = ts.time()

    return (
        config.market_open
        <= current_time
        <= config.market_close
    )


def is_trading_day(
    value: date | datetime | pd.Timestamp,
    *,
    config: TradingSessionConfig = DEFAULT_SESSION,
    holiday_calendar: Optional[
        NSEHolidayCalendar
    ] = None,
) -> bool:
    """
    Determine whether a date is eligible to be an NSE trading day.

    This checks weekends and supplied exchange holidays.
    """

    ts = pd.Timestamp(
        value
    )

    if (
        not config.include_weekends
        and is_weekend(ts)
    ):
        return False

    if (
        holiday_calendar is not None
        and holiday_calendar.is_holiday(
            ts
        )
    ):
        return False

    return True


def is_in_regular_session(
    timestamp: pd.Timestamp | datetime | str,
    *,
    config: TradingSessionConfig = DEFAULT_SESSION,
    holiday_calendar: Optional[
        NSEHolidayCalendar
    ] = None,
) -> bool:
    """
    Return True only when timestamp belongs to a regular trading
    session.
    """

    ts = normalize_market_timestamp(
        timestamp,
        timezone=config.timezone,
    )

    return (
        is_trading_day(
            ts,
            config=config,
            holiday_calendar=holiday_calendar,
        )
        and is_regular_session_time(
            ts,
            config=config,
        )
    )


# ----------------------------------------------------------------------
# Session filtering
# ----------------------------------------------------------------------


def filter_regular_session(
    data: pd.DataFrame,
    *,
    config: TradingSessionConfig = DEFAULT_SESSION,
    holiday_calendar: Optional[
        NSEHolidayCalendar
    ] = None,
) -> pd.DataFrame:
    """
    Keep only candles that fall inside regular NSE trading sessions.

    The input must have a DatetimeIndex.
    """

    if not isinstance(
        data.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "Market data must have a DatetimeIndex."
        )

    frame = data.copy()

    frame.index = normalize_market_index(
        frame.index,
        timezone=config.timezone,
    )

    valid = [
        is_in_regular_session(
            timestamp,
            config=config,
            holiday_calendar=holiday_calendar,
        )
        for timestamp in frame.index
    ]

    return frame.loc[
        valid
    ].copy()


# ----------------------------------------------------------------------
# Trading-day utilities
# ----------------------------------------------------------------------


def trading_days(
    start: date | datetime | str,
    end: date | datetime | str,
    *,
    config: TradingSessionConfig = DEFAULT_SESSION,
    holiday_calendar: Optional[
        NSEHolidayCalendar
    ] = None,
) -> pd.DatetimeIndex:
    """
    Return eligible trading days between start and end, inclusive.
    """

    start_ts = pd.Timestamp(
        start
    ).normalize()

    end_ts = pd.Timestamp(
        end
    ).normalize()

    if start_ts > end_ts:
        raise ValueError(
            "start must be earlier than or equal to end."
        )

    days = pd.date_range(
        start=start_ts,
        end=end_ts,
        freq="D",
    )

    valid = [
        is_trading_day(
            day,
            config=config,
            holiday_calendar=holiday_calendar,
        )
        for day in days
    ]

    return days[
        valid
    ]


def group_by_trading_day(
    data: pd.DataFrame,
    *,
    config: TradingSessionConfig = DEFAULT_SESSION,
    holiday_calendar: Optional[
        NSEHolidayCalendar
    ] = None,
) -> dict[
    pd.Timestamp,
    pd.DataFrame,
]:
    """
    Split intraday data into trading-day DataFrames.

    This is useful for session-aware aggregation.
    """

    filtered = filter_regular_session(
        data,
        config=config,
        holiday_calendar=holiday_calendar,
    )

    if filtered.empty:
        return {}

    groups: dict[
        pd.Timestamp,
        pd.DataFrame,
    ] = {}

    for day, group in filtered.groupby(
        filtered.index.normalize()
    ):
        groups[
            pd.Timestamp(day)
        ] = group.copy()

    return groups


# ----------------------------------------------------------------------
# Session boundaries
# ----------------------------------------------------------------------


def session_open_timestamp(
    day: date | datetime | pd.Timestamp,
    *,
    config: TradingSessionConfig = DEFAULT_SESSION,
) -> pd.Timestamp:
    """
    Return the regular-session opening timestamp for a day.
    """

    ts = pd.Timestamp(
        day
    ).normalize()

    return ts + pd.Timedelta(
        hours=config.market_open.hour,
        minutes=config.market_open.minute,
        seconds=config.market_open.second,
    )


def session_close_timestamp(
    day: date | datetime | pd.Timestamp,
    *,
    config: TradingSessionConfig = DEFAULT_SESSION,
) -> pd.Timestamp:
    """
    Return the regular-session closing timestamp for a day.
    """

    ts = pd.Timestamp(
        day
    ).normalize()

    return ts + pd.Timedelta(
        hours=config.market_close.hour,
        minutes=config.market_close.minute,
        seconds=config.market_close.second,
    )


def session_elapsed_minutes(
    timestamp: pd.Timestamp | datetime | str,
    *,
    config: TradingSessionConfig = DEFAULT_SESSION,
) -> float:
    """
    Return minutes elapsed since the regular-session open.

    Values outside the regular session are clipped to the session
    boundaries.
    """

    ts = normalize_market_timestamp(
        timestamp,
        timezone=config.timezone,
    )

    open_ts = session_open_timestamp(
        ts,
        config=config,
    )

    close_ts = session_close_timestamp(
        ts,
        config=config,
    )

    clipped = min(
        max(
            ts,
            open_ts,
        ),
        close_ts,
    )

    return (
        clipped - open_ts
    ).total_seconds() / 60.0


# ----------------------------------------------------------------------
# Session progress
# ----------------------------------------------------------------------


def session_progress(
    timestamp: pd.Timestamp | datetime | str,
    *,
    config: TradingSessionConfig = DEFAULT_SESSION,
) -> float:
    """
    Return session progress from 0.0 to 1.0.

    0.0 = session open
    1.0 = session close
    """

    total_minutes = (
        config.market_close.hour * 60
        + config.market_close.minute
        - config.market_open.hour * 60
        - config.market_open.minute
    )

    if total_minutes <= 0:
        raise ValueError(
            "Invalid market-session duration."
        )

    elapsed = session_elapsed_minutes(
        timestamp,
        config=config,
    )

    return float(
        min(
            max(
                elapsed
                / total_minutes,
                0.0,
            ),
            1.0,
        )
    )


# ----------------------------------------------------------------------
# Intraday completeness
# ----------------------------------------------------------------------


def session_completeness(
    data: pd.DataFrame,
    *,
    expected_minutes: int = 375,
    config: TradingSessionConfig = DEFAULT_SESSION,
    holiday_calendar: Optional[
        NSEHolidayCalendar
    ] = None,
) -> pd.DataFrame:
    """
    Estimate the completeness of each trading session.

    ``expected_minutes`` defaults to 375 minutes, corresponding to
    09:15 through 15:30.

    For arbitrary bar intervals, the resulting completeness should be
    interpreted as a diagnostic rather than an exact bar-count metric.
    """

    filtered = filter_regular_session(
        data,
        config=config,
        holiday_calendar=holiday_calendar,
    )

    if filtered.empty:
        return pd.DataFrame(
            columns=[
                "rows",
                "first_timestamp",
                "last_timestamp",
                "elapsed_minutes",
                "expected_minutes",
                "completeness",
            ]
        )

    rows = []

    for day, group in filtered.groupby(
        filtered.index.normalize()
    ):
        first = group.index.min()
        last = group.index.max()

        elapsed = (
            last - first
        ).total_seconds() / 60.0

        completeness = (
            elapsed / expected_minutes
            if expected_minutes > 0
            else float("nan")
        )

        rows.append(
            {
                "date": day,
                "rows": len(group),
                "first_timestamp": first,
                "last_timestamp": last,
                "elapsed_minutes": elapsed,
                "expected_minutes": expected_minutes,
                "completeness": min(
                    max(
                        completeness,
                        0.0,
                    ),
                    1.0,
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    if not result.empty:
        result = result.set_index(
            "date"
        ).sort_index()

    return result


# ----------------------------------------------------------------------
# Session-aware 4H bucket assignment
# ----------------------------------------------------------------------


def assign_4h_session_bucket(
    timestamp: pd.Timestamp | datetime | str,
    *,
    config: TradingSessionConfig = DEFAULT_SESSION,
) -> pd.Timestamp:
    """
    Assign an intraday observation to an NSE-session-aware 4H bucket.

    NSE regular trading lasts 6 hours 15 minutes, so four-hour bars
    cannot be represented by ordinary calendar ``4H`` bins.

    The buckets are anchored to 09:15:

        09:15 - 13:15
        13:15 - 15:30

    The second bucket is intentionally shorter because the exchange
    closes at 15:30.

    The returned timestamp represents the bucket start.
    """

    ts = normalize_market_timestamp(
        timestamp,
        timezone=config.timezone,
    )

    if not is_regular_session_time(
        ts,
        config=config,
    ):
        raise ValueError(
            f"Timestamp {ts} is outside the regular trading session."
        )

    open_ts = session_open_timestamp(
        ts,
        config=config,
    )

    elapsed_seconds = (
        ts - open_ts
    ).total_seconds()

    bucket_number = int(
        elapsed_seconds
        // (
            4 * 60 * 60
        )
    )

    bucket_start = (
        open_ts
        + pd.Timedelta(
            hours=4 * bucket_number
        )
    )

    close_ts = session_close_timestamp(
        ts,
        config=config,
    )

    if bucket_start >= close_ts:
        bucket_start = (
            close_ts
            - pd.Timedelta(
                hours=2
            )
        )

    return bucket_start


def add_4h_session_bucket(
    data: pd.DataFrame,
    *,
    config: TradingSessionConfig = DEFAULT_SESSION,
    holiday_calendar: Optional[
        NSEHolidayCalendar
    ] = None,
) -> pd.DataFrame:
    """
    Add an internal ``_4h_bucket`` column for session-aware 4H
    aggregation.
    """

    filtered = filter_regular_session(
        data,
        config=config,
        holiday_calendar=holiday_calendar,
    )

    if filtered.empty:
        result = filtered.copy()
        result["_4h_bucket"] = pd.DatetimeIndex(
            []
        )
        return result

    result = filtered.copy()

    result["_4h_bucket"] = [
        assign_4h_session_bucket(
            timestamp,
            config=config,
        )
        for timestamp in result.index
    ]

    return result


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def validate_session_index(
    data: pd.DataFrame,
    *,
    config: TradingSessionConfig = DEFAULT_SESSION,
) -> dict[str, object]:
    """
    Produce diagnostics for whether an intraday dataset respects
    regular NSE session boundaries.
    """

    if not isinstance(
        data.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "Market data must have a DatetimeIndex."
        )

    normalized = normalize_market_index(
        data.index,
        timezone=config.timezone,
    )

    outside_session = [
        ts
        for ts in normalized
        if not is_regular_session_time(
            ts,
            config=config,
        )
    ]

    weekend_rows = [
        ts
        for ts in normalized
        if is_weekend(ts)
    ]

    return {
        "rows": len(normalized),
        "duplicate_timestamps": int(
            normalized.has_duplicates
        ),
        "non_monotonic": not normalized.is_monotonic_increasing,
        "outside_regular_session": len(
            outside_session
        ),
        "weekend_rows": len(
            weekend_rows
        ),
        "passed": (
            not normalized.has_duplicates
            and normalized.is_monotonic_increasing
            and len(outside_session) == 0
            and len(weekend_rows) == 0
        ),
    }


__all__ = [
    "INDIA_TIMEZONE",
    "NSE_MARKET_OPEN",
    "NSE_MARKET_CLOSE",
    "TradingSessionConfig",
    "NSEHolidayCalendar",
    "DEFAULT_SESSION",
    "normalize_market_timestamp",
    "normalize_market_index",
    "is_weekend",
    "is_regular_session_time",
    "is_trading_day",
    "is_in_regular_session",
    "filter_regular_session",
    "trading_days",
    "group_by_trading_day",
    "session_open_timestamp",
    "session_close_timestamp",
    "session_elapsed_minutes",
    "session_progress",
    "session_completeness",
    "assign_4h_session_bucket",
    "add_4h_session_bucket",
    "validate_session_index",
]
