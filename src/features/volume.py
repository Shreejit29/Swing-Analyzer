"""
Advanced volume feature engine.

Volume is treated as confirmation/context rather than a standalone
buy/sell signal.

All calculations are backward-looking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


def _validate(data: pd.DataFrame) -> None:
    """Validate OHLCV input."""

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


def add_relative_volume(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate relative volume against several historical windows.
    """

    result = data.copy()

    for period in (
        5,
        10,
        20,
        50,
    ):
        average_volume = (
            result["Volume"]
            .rolling(
                period,
                min_periods=period,
            )
            .mean()
        )

        result[
            f"RVOL_{period}"
        ] = _safe_divide(
            result["Volume"],
            average_volume,
        )

    return result


def add_volume_trend(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Measure whether volume is expanding or contracting."""

    result = data.copy()

    for period in (
        5,
        10,
        20,
    ):
        volume_change = (
            result["Volume"]
            .pct_change(period)
        )

        result[
            f"Volume_Trend_{period}"
        ] = volume_change

    result["Volume_MA_Ratio_5_20"] = _safe_divide(
        result["Volume"]
        .rolling(5, min_periods=5)
        .mean(),
        result["Volume"]
        .rolling(20, min_periods=20)
        .mean(),
    )

    result["Volume_MA_Ratio_10_50"] = _safe_divide(
        result["Volume"]
        .rolling(10, min_periods=10)
        .mean(),
        result["Volume"]
        .rolling(50, min_periods=50)
        .mean(),
    )

    return result


def add_price_volume_relationship(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Measure whether volume confirms recent price movement.
    """

    result = data.copy()

    returns = (
        result["Close"]
        .pct_change()
    )

    for period in (
        5,
        10,
        20,
    ):
        rolling_return = (
            result["Close"]
            .pct_change(period)
        )

        rolling_volume = (
            result["Volume"]
            .pct_change(period)
        )

        result[
            f"Price_Volume_Confirmation_{period}"
        ] = (
            np.sign(rolling_return)
            * np.sign(rolling_volume)
        )

    result["Return_Volume_Product"] = (
        returns
        * np.log1p(
            result["Volume"]
        )
    )

    return result


def add_obv_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Add On-Balance Volume and its trend."""

    result = data.copy()

    direction = np.sign(
        result["Close"].diff()
    )

    obv = (
        direction
        .fillna(0)
        * result["Volume"]
    ).cumsum()

    result["OBV"] = obv

    for period in (
        5,
        10,
        20,
        50,
    ):
        obv_average = (
            obv
            .rolling(
                period,
                min_periods=period,
            )
            .mean()
        )

        result[
            f"OBV_Distance_MA_{period}"
        ] = _safe_divide(
            obv,
            obv_average,
        ) - 1

    result["OBV_Change_5"] = (
        obv.pct_change(5)
    )

    result["OBV_Change_20"] = (
        obv.pct_change(20)
    )

    return result


def add_money_flow_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add money-flow-style features.

    These are intentionally simple and transparent. More advanced
    volume-flow models can be tested later.
    """

    result = data.copy()

    typical_price = (
        result["High"]
        + result["Low"]
        + result["Close"]
    ) / 3

    money_flow = (
        typical_price
        * result["Volume"]
    )

    for period in (
        5,
        10,
        20,
    ):
        positive_flow = (
            money_flow.where(
                typical_price
                > typical_price.shift(1),
                0,
            )
            .rolling(
                period,
                min_periods=period,
            )
            .sum()
        )

        negative_flow = (
            money_flow.where(
                typical_price
                < typical_price.shift(1),
                0,
            )
            .rolling(
                period,
                min_periods=period,
            )
            .sum()
        )

        result[
            f"Money_Flow_Ratio_{period}"
        ] = _safe_divide(
            positive_flow,
            negative_flow,
        )

    return result


def add_volume_price_pressure(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate buying/selling pressure from candle location
    and volume.

    This is a feature, not an interpretation that guarantees
    institutional buying or selling.
    """

    result = data.copy()

    candle_range = (
        result["High"]
        - result["Low"]
    )

    close_location = _safe_divide(
        (
            result["Close"]
            - result["Low"]
        ),
        candle_range,
    )

    pressure = (
        (close_location * 2)
        - 1
    )

    result["Volume_Price_Pressure"] = (
        pressure
        * result["Volume"]
    )

    result["Normalized_Volume_Pressure"] = (
        pressure
        * _safe_divide(
            result["Volume"],
            result["Volume"]
            .rolling(
                20,
                min_periods=20,
            )
            .mean(),
        )
    )

    return result


def add_volume_breakout_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detect unusually high-volume price expansion.

    Reference values use previous candles only.
    """

    result = data.copy()

    price_return = (
        result["Close"]
        .pct_change()
    )

    for period in (
        20,
        50,
    ):
        volume_average = (
            result["Volume"]
            .shift(1)
            .rolling(
                period,
                min_periods=period,
            )
            .mean()
        )

        return_volatility = (
            price_return
            .shift(1)
            .rolling(
                period,
                min_periods=period,
            )
            .std()
        )

        result[
            f"Unusual_Volume_{period}"
        ] = (
            _safe_divide(
                result["Volume"],
                volume_average,
            )
        )

        result[
            f"Unusual_Price_Move_{period}"
        ] = _safe_divide(
            price_return.abs(),
            return_volatility,
        )

        result[
            f"Volume_Price_Expansion_{period}"
        ] = (
            result[
                f"Unusual_Volume_{period}"
            ]
            * result[
                f"Unusual_Price_Move_{period}"
            ]
        )

    return result


def add_volume_divergence(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detect simple price/volume divergence conditions.

    A divergence is represented numerically so the ML model can
    decide whether it has predictive value.
    """

    result = data.copy()

    price_change_20 = (
        result["Close"]
        .pct_change(20)
    )

    obv_change_20 = (
        result["Volume"]
        * np.sign(
            result["Close"].diff()
        ).fillna(0)
    ).cumsum().pct_change(20)

    result["Price_OBV_Divergence"] = (
        np.sign(price_change_20)
        != np.sign(obv_change_20)
    ).astype(int)

    result["Price_OBV_Divergence_Strength"] = (
        obv_change_20
        - price_change_20
    )

    return result


def add_all_volume_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the complete volume feature engine."""

    _validate(data)

    result = data.copy()

    result = add_relative_volume(result)

    result = add_volume_trend(result)

    result = add_price_volume_relationship(
        result
    )

    result = add_obv_features(result)

    result = add_money_flow_features(
        result
    )

    result = add_volume_price_pressure(
        result
    )

    result = add_volume_breakout_features(
        result
    )

    result = add_volume_divergence(
        result
    )

    return result


def volume_feature_names(
    data: pd.DataFrame,
) -> list[str]:
    """Return volume-derived feature names."""

    return [
        column
        for column in data.columns
        if column not in REQUIRED_COLUMNS
    ]
