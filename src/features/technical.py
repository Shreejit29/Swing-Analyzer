"""
Technical indicator feature engine.

All indicators are calculated using only information available
at or before the current observation timestamp.

No future data is used.
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


def _validate_ohlcv(data: pd.DataFrame) -> None:
    """Validate the input OHLCV dataframe."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing OHLCV columns: {missing}"
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
            "DataFrame index must be chronological."
        )


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Divide safely, returning NaN where denominator is zero."""

    return numerator / denominator.replace(
        0,
        np.nan,
    )


def add_rsi(
    data: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """Add Relative Strength Index."""

    result = data.copy()

    delta = result["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = _safe_divide(
        average_gain,
        average_loss,
    )

    result["RSI_14"] = (
        100
        - (
            100
            / (1 + rs)
        )
    )

    return result


def add_macd(
    data: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Add MACD, signal and histogram."""

    result = data.copy()

    fast_ema = (
        result["Close"]
        .ewm(
            span=fast,
            adjust=False,
            min_periods=fast,
        )
        .mean()
    )

    slow_ema = (
        result["Close"]
        .ewm(
            span=slow,
            adjust=False,
            min_periods=slow,
        )
        .mean()
    )

    macd = fast_ema - slow_ema

    signal_line = (
        macd
        .ewm(
            span=signal,
            adjust=False,
            min_periods=signal,
        )
        .mean()
    )

    result["MACD"] = macd
    result["MACD_Signal"] = signal_line
    result["MACD_Histogram"] = (
        macd - signal_line
    )

    return result


def add_moving_averages(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Add EMA and SMA trend features."""

    result = data.copy()

    for period in (
        9,
        20,
        50,
        100,
        200,
    ):
        result[
            f"EMA_{period}"
        ] = (
            result["Close"]
            .ewm(
                span=period,
                adjust=False,
                min_periods=period,
            )
            .mean()
        )

    for period in (
        20,
        50,
        200,
    ):
        result[
            f"SMA_{period}"
        ] = (
            result["Close"]
            .rolling(
                window=period,
                min_periods=period,
            )
            .mean()
        )

    return result


def add_bollinger_bands(
    data: pd.DataFrame,
    period: int = 20,
    standard_deviations: float = 2.0,
) -> pd.DataFrame:
    """Add Bollinger Band features."""

    result = data.copy()

    middle = (
        result["Close"]
        .rolling(
            period,
            min_periods=period,
        )
        .mean()
    )

    standard_deviation = (
        result["Close"]
        .rolling(
            period,
            min_periods=period,
        )
        .std()
    )

    upper = (
        middle
        + standard_deviations
        * standard_deviation
    )

    lower = (
        middle
        - standard_deviations
        * standard_deviation
    )

    result["BB_Middle"] = middle
    result["BB_Upper"] = upper
    result["BB_Lower"] = lower

    result["BB_Width"] = _safe_divide(
        upper - lower,
        middle,
    )

    result["BB_Position"] = _safe_divide(
        result["Close"] - lower,
        upper - lower,
    )

    return result


def add_atr(
    data: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """Add Average True Range and normalized ATR."""

    result = data.copy()

    previous_close = (
        result["Close"].shift(1)
    )

    true_range = pd.concat(
        [
            result["High"]
            - result["Low"],

            (
                result["High"]
                - previous_close
            ).abs(),

            (
                result["Low"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = (
        true_range
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )

    result["ATR_14"] = atr

    result["ATR_14_Percent"] = (
        _safe_divide(
            atr,
            result["Close"],
        )
        * 100
    )

    return result


def add_adx(
    data: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """Add ADX, +DI and -DI."""

    result = data.copy()

    high = result["High"]
    low = result["Low"]
    close = result["Close"]

    previous_high = high.shift(1)
    previous_low = low.shift(1)
    previous_close = close.shift(1)

    up_move = high - previous_high
    down_move = previous_low - low

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move)
            & (up_move > 0),
            up_move,
            0.0,
        ),
        index=result.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move)
            & (down_move > 0),
            down_move,
            0.0,
        ),
        index=result.index,
    )

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = (
        true_range
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )

    plus_dm_smoothed = (
        plus_dm
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )

    minus_dm_smoothed = (
        minus_dm
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )

    plus_di = (
        _safe_divide(
            plus_dm_smoothed,
            atr,
        )
        * 100
    )

    minus_di = (
        _safe_divide(
            minus_dm_smoothed,
            atr,
        )
        * 100
    )

    denominator = (
        plus_di + minus_di
    )

    dx = (
        _safe_divide(
            (plus_di - minus_di).abs(),
            denominator,
        )
        * 100
    )

    adx = (
        dx
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )

    result["Plus_DI_14"] = plus_di
    result["Minus_DI_14"] = minus_di
    result["ADX_14"] = adx

    return result


def add_stochastic(
    data: pd.DataFrame,
    period: int = 14,
) -> pd.DataFrame:
    """Add stochastic oscillator."""

    result = data.copy()

    lowest_low = (
        result["Low"]
        .rolling(
            period,
            min_periods=period,
        )
        .min()
    )

    highest_high = (
        result["High"]
        .rolling(
            period,
            min_periods=period,
        )
        .max()
    )

    stochastic_k = (
        _safe_divide(
            result["Close"]
            - lowest_low,
            highest_high
            - lowest_low,
        )
        * 100
    )

    stochastic_d = (
        stochastic_k
        .rolling(
            3,
            min_periods=3,
        )
        .mean()
    )

    result["Stochastic_K_14"] = stochastic_k
    result["Stochastic_D_14"] = stochastic_d

    return result


def add_obv(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Add On-Balance Volume."""

    result = data.copy()

    direction = np.sign(
        result["Close"].diff()
    )

    result["OBV"] = (
        direction
        .fillna(0)
        * result["Volume"]
    ).cumsum()

    result["OBV_Change_5"] = (
        result["OBV"]
        .pct_change(5)
    )

    result["OBV_Change_20"] = (
        result["OBV"]
        .pct_change(20)
    )

    return result


def add_roc(
    data: pd.DataFrame,
    period: int = 10,
) -> pd.DataFrame:
    """Add Rate of Change."""

    result = data.copy()

    result[
        f"ROC_{period}"
    ] = (
        result["Close"]
        .pct_change(period)
        * 100
    )

    return result


def add_momentum(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Add momentum features."""

    result = data.copy()

    for period in (
        3,
        5,
        10,
        20,
    ):
        result[
            f"Momentum_{period}"
        ] = (
            result["Close"]
            / result["Close"].shift(period)
            - 1
        )

    return result


def add_volume_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Add volume and relative-volume features."""

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
            f"Relative_Volume_{period}"
        ] = _safe_divide(
            result["Volume"],
            average_volume,
        )

    result["Volume_Change_1"] = (
        result["Volume"]
        .pct_change()
    )

    result["Volume_Change_5"] = (
        result["Volume"]
        .pct_change(5)
    )

    return result


def add_vwap(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add cumulative VWAP.

    VWAP is primarily meaningful for intraday data.
    For daily data it is retained as a price/volume feature,
    but later model selection can determine whether it adds
    predictive value.
    """

    result = data.copy()

    typical_price = (
        result["High"]
        + result["Low"]
        + result["Close"]
    ) / 3

    cumulative_volume = (
        result["Volume"]
        .cumsum()
    )

    cumulative_value = (
        typical_price
        * result["Volume"]
    ).cumsum()

    result["VWAP"] = _safe_divide(
        cumulative_value,
        cumulative_volume,
    )

    result["Distance_From_VWAP"] = (
        _safe_divide(
            result["Close"],
            result["VWAP"],
        )
        - 1
    )

    return result


def add_all_technical_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the complete first-generation technical
    indicator feature set.

    The function applies indicators sequentially so each
    indicator operates on the same chronological dataset.
    """

    _validate_ohlcv(data)

    result = data.copy()

    result = add_rsi(result)
    result = add_macd(result)
    result = add_moving_averages(result)
    result = add_bollinger_bands(result)
    result = add_atr(result)
    result = add_adx(result)
    result = add_stochastic(result)
    result = add_obv(result)
    result = add_roc(result)
    result = add_momentum(result)
    result = add_volume_features(result)
    result = add_vwap(result)

    return result


def technical_feature_names(
    data: pd.DataFrame,
) -> list[str]:
    """
    Return columns generated by the technical engine.

    This allows the modelling layer to explicitly select
    engineered features instead of accidentally selecting
    target columns.
    """

    base_columns = set(
        REQUIRED_COLUMNS
    )

    return [
        column
        for column in data.columns
        if column not in base_columns
    ]
