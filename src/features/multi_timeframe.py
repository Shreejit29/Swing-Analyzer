"""
Multi-timeframe feature generation.

Purpose
-------
Adds higher-timeframe trend context to the daily feature set.

The function is designed to be safe when only daily OHLCV data is
available. It does not require the user to provide additional inputs.

Canonical OHLCV columns:
    open
    high
    low
    close
    volume
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _get_series(
    df: pd.DataFrame,
    name: str,
) -> pd.Series:
    """Get a numeric column using case-insensitive matching."""

    if name in df.columns:
        value = df[name]

        if isinstance(value, pd.DataFrame):
            value = value.iloc[:, 0]

        return pd.to_numeric(
            value,
            errors="coerce",
        )

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    key = name.lower()

    if key in lookup:
        value = df[lookup[key]]

        if isinstance(value, pd.DataFrame):
            value = value.iloc[:, 0]

        return pd.to_numeric(
            value,
            errors="coerce",
        )

    return pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )


def _ema(
    series: pd.Series,
    span: int,
) -> pd.Series:
    """Calculate EMA."""

    return series.ewm(
        span=span,
        adjust=False,
        min_periods=1,
    ).mean()


def _timeframe_features(
    close: pd.Series,
    prefix: str,
    fast: int,
    slow: int,
) -> pd.DataFrame:
    """
    Generate trend features for a resampled timeframe.
    """

    result = pd.DataFrame(
        index=close.index
    )

    ema_fast = _ema(
        close,
        fast,
    )

    ema_slow = _ema(
        close,
        slow,
    )

    result[
        f"{prefix}_ema_fast"
    ] = ema_fast

    result[
        f"{prefix}_ema_slow"
    ] = ema_slow

    result[
        f"{prefix}_trend"
    ] = np.where(
        ema_fast > ema_slow,
        1,
        np.where(
            ema_fast < ema_slow,
            -1,
            0,
        ),
    )

    result[
        f"{prefix}_return"
    ] = close.pct_change(
        max(2, fast // 2)
    )

    result[
        f"{prefix}_slope"
    ] = ema_slow.pct_change(
        max(2, fast // 2)
    )

    result[
        f"{prefix}_strength"
    ] = (
        ema_fast
        / ema_slow.replace(
            0,
            np.nan,
        )
        - 1.0
    )

    return result


def add_multi_timeframe_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add multi-timeframe trend features.

    The input is normally daily OHLCV data.

    Higher timeframes are created internally:

        Weekly
        Monthly

    Features are calculated using only information available up
    to each date and then forward-filled onto the daily index.

    This avoids introducing future information into earlier rows.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if df.empty:
        return df.copy()

    out = df.copy()

    close = _get_series(
        out,
        "close",
    )

    if close.isna().all():
        raise ValueError(
            "Multi-timeframe features require a valid close column."
        )

    # ---------------------------------------------------------
    # Ensure datetime index.
    # ---------------------------------------------------------

    original_index = out.index

    try:

        if not isinstance(
            out.index,
            pd.DatetimeIndex,
        ):
            datetime_index = pd.to_datetime(
                out.index,
                errors="coerce",
            )

            if datetime_index.isna().all():
                raise ValueError(
                    "Index cannot be converted to datetime."
                )

            out.index = datetime_index

    except Exception as exc:

        raise ValueError(
            "Multi-timeframe features require a "
            "datetime-compatible index."
        ) from exc

    # Remove invalid dates.
    valid_index = ~out.index.isna()

    out = out.loc[
        valid_index
    ].copy()

    close = close.loc[
        valid_index
    ].copy()

    close.index = out.index

    # Remove duplicate timestamps.
    duplicate_mask = (
        ~out.index.duplicated(
            keep="last"
        )
    )

    out = out.loc[
        duplicate_mask
    ].copy()

    close = close.loc[
        duplicate_mask
    ].copy()

    close = close.sort_index()

    # ---------------------------------------------------------
    # Weekly timeframe.
    #
    # W-FRI means the weekly candle ends on Friday.
    # ---------------------------------------------------------

    weekly_close = (
        close
        .resample("W-FRI")
        .last()
        .dropna()
    )

    weekly_features = _timeframe_features(
        weekly_close,
        prefix="weekly",
        fast=10,
        slow=30,
    )

    # Shift by one completed weekly candle.
    #
    # This prevents the current incomplete week from being used
    # as if it were already known at the beginning of the week.
    weekly_features = weekly_features.shift(1)

    # Map the completed weekly information to daily rows.
    weekly_daily = (
        weekly_features
        .reindex(
            out.index,
            method="ffill",
        )
    )

    for column in weekly_daily.columns:

        out[column] = weekly_daily[
            column
        ]

    # ---------------------------------------------------------
    # Monthly timeframe.
    # ---------------------------------------------------------

    monthly_close = (
        close
        .resample("ME")
        .last()
        .dropna()
    )

    monthly_features = _timeframe_features(
        monthly_close,
        prefix="monthly",
        fast=3,
        slow=10,
    )

    # Use only completed monthly candles.
    monthly_features = monthly_features.shift(1)

    monthly_daily = (
        monthly_features
        .reindex(
            out.index,
            method="ffill",
        )
    )

    for column in monthly_daily.columns:

        out[column] = monthly_daily[
            column
        ]

    # ---------------------------------------------------------
    # Combined higher-timeframe trend.
    # ---------------------------------------------------------

    weekly_trend = out[
        "weekly_trend"
    ].fillna(0)

    monthly_trend = out[
        "monthly_trend"
    ].fillna(0)

    out[
        "higher_timeframe_score"
    ] = (
        weekly_trend
        + monthly_trend
    )

    out[
        "higher_timeframe_alignment"
    ] = np.where(
        (
            weekly_trend > 0
        )
        & (
            monthly_trend > 0
        ),
        1,
        np.where(
            (
                weekly_trend < 0
            )
            & (
                monthly_trend < 0
            ),
            -1,
            0,
        ),
    )

    # ---------------------------------------------------------
    # Daily trend context.
    # ---------------------------------------------------------

    daily_ema20 = _ema(
        close,
        20,
    )

    daily_ema50 = _ema(
        close,
        50,
    )

    daily_trend = np.where(
        daily_ema20 > daily_ema50,
        1,
        np.where(
            daily_ema20 < daily_ema50,
            -1,
            0,
        ),
    )

    out[
        "daily_trend_direction"
    ] = daily_trend

    # ---------------------------------------------------------
    # Three-timeframe alignment.
    #
    # +1 = daily + weekly + monthly bullish
    # -1 = daily + weekly + monthly bearish
    #  0 = mixed
    # ---------------------------------------------------------

    out[
        "three_timeframe_alignment"
    ] = np.where(
        (
            daily_trend > 0
        )
        & (
            weekly_trend > 0
        )
        & (
            monthly_trend > 0
        ),
        1,
        np.where(
            (
                daily_trend < 0
            )
            & (
                weekly_trend < 0
            )
            & (
                monthly_trend < 0
            ),
            -1,
            0,
        ),
    )

    # ---------------------------------------------------------
    # Trend agreement strength.
    #
    # 0 = no agreement
    # 1 = one timeframe agrees
    # 2 = two timeframes agree
    # 3 = all three agree
    # ---------------------------------------------------------

    bullish_count = (
        (daily_trend > 0).astype(int)
        + (weekly_trend > 0).astype(int)
        + (monthly_trend > 0).astype(int)
    )

    bearish_count = (
        (daily_trend < 0).astype(int)
        + (weekly_trend < 0).astype(int)
        + (monthly_trend < 0).astype(int)
    )

    out[
        "bullish_timeframe_count"
    ] = bullish_count

    out[
        "bearish_timeframe_count"
    ] = bearish_count

    # ---------------------------------------------------------
    # Clean numerical values.
    # ---------------------------------------------------------

    numeric_columns = (
        out
        .select_dtypes(
            include=[np.number]
        )
        .columns
    )

    out[numeric_columns] = (
        out[numeric_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    # Restore original ordering.
    #
    # Feature engineering modules should preserve the user's
    # original row order.
    try:
        out = out.reindex(
            original_index
        )
    except Exception:
        pass

    return out


def add_mtf_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Short alias for add_multi_timeframe_features().
    """

    return add_multi_timeframe_features(
        df
    )


__all__ = [
    "add_multi_timeframe_features",
    "add_mtf_features",
    ]
