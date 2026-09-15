"""
Price-action feature engine.

These features describe candle structure, price behaviour,
breakouts, trend structure and recent support/resistance.

All calculations are strictly backward-looking.
No future candles are used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
]


def _validate(data: pd.DataFrame) -> None:
    """Validate price data."""

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if not isinstance(
        data.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "DataFrame index must be a DatetimeIndex."
        )

    if not data.index.is_monotonic_increasing:
        raise ValueError(
            "DataFrame must be chronologically ordered."
        )


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Safe division."""

    return numerator / denominator.replace(
        0,
        np.nan,
    )


def add_candle_structure(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add candle anatomy features.

    These capture:
        - candle body
        - upper/lower wick
        - range
        - body-to-range ratio
        - bullish/bearish structure
        - gap behaviour
    """

    result = data.copy()

    candle_range = (
        result["High"]
        - result["Low"]
    )

    body = (
        result["Close"]
        - result["Open"]
    )

    body_abs = body.abs()

    upper_wick = (
        result["High"]
        - result[["Open", "Close"]].max(axis=1)
    )

    lower_wick = (
        result[["Open", "Close"]].min(axis=1)
        - result["Low"]
    )

    result["Candle_Range"] = candle_range

    result["Candle_Body"] = body

    result["Candle_Body_Abs"] = body_abs

    result["Upper_Wick"] = upper_wick

    result["Lower_Wick"] = lower_wick

    result["Body_Range_Ratio"] = _safe_divide(
        body_abs,
        candle_range,
    )

    result["Upper_Wick_Ratio"] = _safe_divide(
        upper_wick,
        candle_range,
    )

    result["Lower_Wick_Ratio"] = _safe_divide(
        lower_wick,
        candle_range,
    )

    result["Bullish_Candle"] = (
        body > 0
    ).astype(int)

    result["Bearish_Candle"] = (
        body < 0
    ).astype(int)

    result["Doji_Score"] = (
        1
        - _safe_divide(
            body_abs,
            candle_range,
        )
    )

    previous_close = (
        result["Close"].shift(1)
    )

    result["Gap_Percent"] = (
        _safe_divide(
            result["Open"]
            - previous_close,
            previous_close,
        )
    )

    result["Intraday_Return"] = (
        _safe_divide(
            result["Close"]
            - result["Open"],
            result["Open"],
        )
    )

    return result


def add_candle_patterns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add deterministic candle-pattern features.

    Pattern recognition is intentionally represented as
    numerical features rather than trading decisions.
    """

    result = data.copy()

    body = (
        result["Close"]
        - result["Open"]
    )

    body_abs = body.abs()

    candle_range = (
        result["High"]
        - result["Low"]
    )

    upper_wick = (
        result["High"]
        - result[["Open", "Close"]].max(axis=1)
    )

    lower_wick = (
        result[["Open", "Close"]].min(axis=1)
        - result["Low"]
    )

    # Hammer-like structure.
    result["Hammer_Structure"] = (
        (
            lower_wick
            >= body_abs * 2
        )
        & (
            upper_wick
            <= body_abs
        )
        & (
            body_abs
            <= candle_range * 0.5
        )
    ).astype(int)

    # Shooting-star-like structure.
    result["Shooting_Star_Structure"] = (
        (
            upper_wick
            >= body_abs * 2
        )
        & (
            lower_wick
            <= body_abs
        )
        & (
            body_abs
            <= candle_range * 0.5
        )
    ).astype(int)

    # Strong directional candle.
    result["Strong_Bull_Candle"] = (
        (
            body > 0
        )
        & (
            _safe_divide(
                body_abs,
                candle_range,
            )
            >= 0.7
        )
    ).astype(int)

    result["Strong_Bear_Candle"] = (
        (
            body < 0
        )
        & (
            _safe_divide(
                body_abs,
                candle_range,
            )
            >= 0.7
        )
    ).astype(int)

    # Engulfing structures.
    previous_open = (
        result["Open"].shift(1)
    )

    previous_close = (
        result["Close"].shift(1)
    )

    result["Bullish_Engulfing"] = (
        (previous_close < previous_open)
        & (result["Close"] > result["Open"])
        & (result["Open"] <= previous_close)
        & (result["Close"] >= previous_open)
    ).astype(int)

    result["Bearish_Engulfing"] = (
        (previous_close > previous_open)
        & (result["Close"] < result["Open"])
        & (result["Open"] >= previous_close)
        & (result["Close"] <= previous_open)
    ).astype(int)

    # Inside bar.
    result["Inside_Bar"] = (
        (result["High"] < result["High"].shift(1))
        & (result["Low"] > result["Low"].shift(1))
    ).astype(int)

    # Outside bar.
    result["Outside_Bar"] = (
        (result["High"] > result["High"].shift(1))
        & (result["Low"] < result["Low"].shift(1))
    ).astype(int)

    return result


def add_price_momentum(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Add price acceleration and persistence features."""

    result = data.copy()

    returns = (
        result["Close"]
        .pct_change()
    )

    for period in (
        3,
        5,
        10,
        20,
    ):
        result[
            f"Positive_Return_Ratio_{period}"
        ] = (
            returns.gt(0)
            .rolling(
                period,
                min_periods=period,
            )
            .mean()
        )

        result[
            f"Negative_Return_Ratio_{period}"
        ] = (
            returns.lt(0)
            .rolling(
                period,
                min_periods=period,
            )
            .mean()
        )

    result["Return_Acceleration"] = (
        returns
        - returns.shift(1)
    )

    result["Price_Acceleration_5"] = (
        result["Close"].pct_change(5)
        - result["Close"].pct_change(10)
    )

    return result


def add_support_resistance(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add rolling support/resistance levels.

    CRITICAL:
    Current candle is excluded from the level calculation.

    This prevents the current high/low from leaking into a
    feature that is supposed to represent previously known
    resistance/support.
    """

    result = data.copy()

    for period in (
        10,
        20,
        50,
        100,
        200,
    ):
        previous_high = (
            result["High"]
            .shift(1)
            .rolling(
                period,
                min_periods=period,
            )
            .max()
        )

        previous_low = (
            result["Low"]
            .shift(1)
            .rolling(
                period,
                min_periods=period,
            )
            .min()
        )

        result[
            f"Resistance_{period}"
        ] = previous_high

        result[
            f"Support_{period}"
        ] = previous_low

        result[
            f"Distance_To_Resistance_{period}"
        ] = _safe_divide(
            previous_high
            - result["Close"],
            result["Close"],
        )

        result[
            f"Distance_To_Support_{period}"
        ] = _safe_divide(
            result["Close"]
            - previous_low,
            result["Close"],
        )

    return result


def add_breakout_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detect potential breakouts relative to previously known
    price levels.

    The level is shifted by one candle so the current candle
    cannot define its own breakout threshold.
    """

    result = data.copy()

    for period in (
        10,
        20,
        50,
    ):
        previous_high = (
            result["High"]
            .shift(1)
            .rolling(
                period,
                min_periods=period,
            )
            .max()
        )

        previous_low = (
            result["Low"]
            .shift(1)
            .rolling(
                period,
                min_periods=period,
            )
            .min()
        )

        result[
            f"Breakout_Above_{period}"
        ] = (
            result["Close"]
            > previous_high
        ).astype(int)

        result[
            f"Breakdown_Below_{period}"
        ] = (
            result["Close"]
            < previous_low
        ).astype(int)

        result[
            f"High_Breakout_Strength_{period}"
        ] = _safe_divide(
            result["Close"]
            - previous_high,
            previous_high,
        )

        result[
            f"Low_Breakdown_Strength_{period}"
        ] = _safe_divide(
            previous_low
            - result["Close"],
            previous_low,
        )

    return result


def add_trend_structure(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add simple market-structure features.

    These identify whether price is making higher highs/lows
    or lower highs/lows over recent windows.
    """

    result = data.copy()

    previous_high = (
        result["High"].shift(1)
    )

    previous_low = (
        result["Low"].shift(1)
    )

    result["Higher_High_1"] = (
        result["High"]
        > previous_high
    ).astype(int)

    result["Lower_High_1"] = (
        result["High"]
        < previous_high
    ).astype(int)

    result["Higher_Low_1"] = (
        result["Low"]
        > previous_low
    ).astype(int)

    result["Lower_Low_1"] = (
        result["Low"]
        < previous_low
    ).astype(int)

    for period in (
        5,
        10,
        20,
    ):
        result[
            f"Higher_High_Ratio_{period}"
        ] = (
            result["High"]
            .diff()
            .gt(0)
            .rolling(
                period,
                min_periods=period,
            )
            .mean()
        )

        result[
            f"Higher_Low_Ratio_{period}"
        ] = (
            result["Low"]
            .diff()
            .gt(0)
            .rolling(
                period,
                min_periods=period,
            )
            .mean()
        )

    return result


def add_range_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Add historical trading-range features."""

    result = data.copy()

    daily_range = (
        result["High"]
        - result["Low"]
    )

    for period in (
        5,
        10,
        20,
    ):
        average_range = (
            daily_range
            .rolling(
                period,
                min_periods=period,
            )
            .mean()
        )

        result[
            f"Range_Relative_To_{period}"
        ] = _safe_divide(
            daily_range,
            average_range,
        )

    result["Range_Expansion"] = (
        _safe_divide(
            daily_range,
            daily_range.shift(1),
        )
    )

    return result


def add_all_price_action_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply the complete price-action feature engine.
    """

    _validate(data)

    result = data.copy()

    result = add_candle_structure(result)

    result = add_candle_patterns(result)

    result = add_price_momentum(result)

    result = add_support_resistance(result)

    result = add_breakout_features(result)

    result = add_trend_structure(result)

    result = add_range_features(result)

    return result


def price_action_feature_names(
    data: pd.DataFrame,
) -> list[str]:
    """Return generated price-action feature names."""

    return [
        column
        for column in data.columns
        if column not in REQUIRED_COLUMNS
    ]
