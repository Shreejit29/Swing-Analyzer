"""
Market regime feature engine.

The purpose of this module is to help the model understand the
environment in which a stock is trading.

Examples:
    - trending vs sideways
    - bullish vs bearish
    - low vs high volatility
    - strong vs weak momentum

All regime calculations are backward-looking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Perform division while avoiding division by zero."""

    return numerator / denominator.replace(
        0,
        np.nan,
    )


def add_trend_regime(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create trend-regime features.

    The model receives continuous measurements rather than a
    hard-coded BUY/SELL regime.
    """

    result = data.copy()

    close = result["Close"]

    ema_20 = (
        close
        .ewm(
            span=20,
            adjust=False,
            min_periods=20,
        )
        .mean()
    )

    ema_50 = (
        close
        .ewm(
            span=50,
            adjust=False,
            min_periods=50,
        )
        .mean()
    )

    ema_200 = (
        close
        .ewm(
            span=200,
            adjust=False,
            min_periods=200,
        )
        .mean()
    )

    result["Trend_Distance_EMA20"] = (
        _safe_divide(
            close,
            ema_20,
        )
        - 1
    )

    result["Trend_Distance_EMA50"] = (
        _safe_divide(
            close,
            ema_50,
        )
        - 1
    )

    result["Trend_Distance_EMA200"] = (
        _safe_divide(
            close,
            ema_200,
        )
        - 1
    )

    result["EMA20_EMA50_Spread"] = (
        _safe_divide(
            ema_20,
            ema_50,
        )
        - 1
    )

    result["EMA50_EMA200_Spread"] = (
        _safe_divide(
            ema_50,
            ema_200,
        )
        - 1
    )

    result["EMA20_Slope_5"] = (
        _safe_divide(
            ema_20,
            ema_20.shift(5),
        )
        - 1
    )

    result["EMA50_Slope_10"] = (
        _safe_divide(
            ema_50,
            ema_50.shift(10),
        )
        - 1
    )

    result["EMA200_Slope_20"] = (
        _safe_divide(
            ema_200,
            ema_200.shift(20),
        )
        - 1
    )

    return result


def add_adx_regime(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create trend-strength regime features.

    Uses ADX when available. If the technical engine has not yet
    been applied, ADX is calculated here.
    """

    result = data.copy()

    if "ADX_14" not in result.columns:

        high = result["High"]
        low = result["Low"]
        close = result["Close"]

        previous_high = high.shift(1)
        previous_low = low.shift(1)
        previous_close = close.shift(1)

        up_move = (
            high - previous_high
        )

        down_move = (
            previous_low - low
        )

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
                (
                    high
                    - previous_close
                ).abs(),
                (
                    low
                    - previous_close
                ).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = (
            true_range
            .ewm(
                alpha=1 / 14,
                adjust=False,
                min_periods=14,
            )
            .mean()
        )

        plus_dm_smooth = (
            plus_dm
            .ewm(
                alpha=1 / 14,
                adjust=False,
                min_periods=14,
            )
            .mean()
        )

        minus_dm_smooth = (
            minus_dm
            .ewm(
                alpha=1 / 14,
                adjust=False,
                min_periods=14,
            )
            .mean()
        )

        plus_di = (
            _safe_divide(
                plus_dm_smooth,
                atr,
            )
            * 100
        )

        minus_di = (
            _safe_divide(
                minus_dm_smooth,
                atr,
            )
            * 100
        )

        dx = (
            _safe_divide(
                (
                    plus_di
                    - minus_di
                ).abs(),
                plus_di + minus_di,
            )
            * 100
        )

        result["ADX_14"] = (
            dx
            .ewm(
                alpha=1 / 14,
                adjust=False,
                min_periods=14,
            )
            .mean()
        )

    result["Strong_Trend_Regime"] = (
        result["ADX_14"] >= 25
    ).astype(int)

    result["Very_Strong_Trend_Regime"] = (
        result["ADX_14"] >= 40
    ).astype(int)

    return result


def add_volatility_regime(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Measure whether current volatility is elevated or compressed
    relative to its historical distribution.
    """

    result = data.copy()

    returns = (
        result["Close"]
        .pct_change()
    )

    volatility_10 = (
        returns
        .rolling(
            10,
            min_periods=10,
        )
        .std()
    )

    volatility_20 = (
        returns
        .rolling(
            20,
            min_periods=20,
        )
        .std()
    )

    volatility_60 = (
        returns
        .rolling(
            60,
            min_periods=60,
        )
        .std()
    )

    result["Volatility_10"] = volatility_10
    result["Volatility_20"] = volatility_20
    result["Volatility_60"] = volatility_60

    result["Volatility_Ratio_10_60"] = (
        _safe_divide(
            volatility_10,
            volatility_60,
        )
    )

    result["Volatility_Ratio_20_60"] = (
        _safe_divide(
            volatility_20,
            volatility_60,
        )
    )

    result["High_Volatility_Regime"] = (
        result["Volatility_Ratio_20_60"]
        > 1.5
    ).astype(int)

    result["Low_Volatility_Regime"] = (
        result["Volatility_Ratio_20_60"]
        < 0.7
    ).astype(int)

    return result


def add_breakout_regime(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Measure whether price is close to or breaking historical
    trading ranges.

    Current candle is excluded from the reference range.
    """

    result = data.copy()

    for period in (
        20,
        50,
        100,
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

        range_width = (
            previous_high
            - previous_low
        )

        result[
            f"Range_Position_{period}"
        ] = _safe_divide(
            result["Close"]
            - previous_low,
            range_width,
        )

        result[
            f"Near_Resistance_{period}"
        ] = (
            result["Close"]
            >= previous_high * 0.98
        ).astype(int)

        result[
            f"Near_Support_{period}"
        ] = (
            result["Close"]
            <= previous_low * 1.02
        ).astype(int)

    return result


def add_momentum_regime(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Measure whether momentum is positive, negative or weakening.
    """

    result = data.copy()

    returns = (
        result["Close"]
        .pct_change()
    )

    result["Momentum_Regime_5"] = (
        result["Close"]
        .pct_change(5)
    )

    result["Momentum_Regime_20"] = (
        result["Close"]
        .pct_change(20)
    )

    result["Momentum_Slope"] = (
        result["Momentum_Regime_5"]
        - result["Momentum_Regime_5"]
        .shift(5)
    )

    result["Momentum_Alignment"] = (
        np.sign(
            result["Momentum_Regime_5"]
        )
        + np.sign(
            result["Momentum_Regime_20"]
        )
    )

    result["Positive_Momentum_Regime"] = (
        (
            result["Momentum_Regime_5"] > 0
        )
        & (
            result["Momentum_Regime_20"] > 0
        )
    ).astype(int)

    result["Negative_Momentum_Regime"] = (
        (
            result["Momentum_Regime_5"] < 0
        )
        & (
            result["Momentum_Regime_20"] < 0
        )
    ).astype(int)

    result["Momentum_Reversal_Risk"] = (
        np.sign(
            result["Momentum_Regime_5"]
        )
        != np.sign(
            result["Momentum_Regime_20"]
        )
    ).astype(int)

    return result


def add_market_state_score(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a continuous composite regime score.

    This is NOT a trading signal.

    It gives the machine-learning models a compact representation
    of the current technical environment.

    Range is approximately -1 to +1.
    """

    result = data.copy()

    components = []

    if "Trend_Distance_EMA50" in result:
        components.append(
            np.tanh(
                result[
                    "Trend_Distance_EMA50"
                ] * 10
            )
        )

    if "EMA50_EMA200_Spread" in result:
        components.append(
            np.tanh(
                result[
                    "EMA50_EMA200_Spread"
                ] * 10
            )
        )

    if "Momentum_Regime_20" in result:
        components.append(
            np.tanh(
                result[
                    "Momentum_Regime_20"
                ] * 5
            )
        )

    if components:
        result["Market_State_Score"] = (
            pd.concat(
                components,
                axis=1,
            )
            .mean(axis=1)
        )

    return result


def add_all_regime_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply the complete regime feature engine.
    """

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing = [
        column
        for column in required
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

    result = data.copy()

    result = add_trend_regime(result)
    result = add_adx_regime(result)
    result = add_volatility_regime(result)
    result = add_breakout_regime(result)
    result = add_momentum_regime(result)
    result = add_market_state_score(result)

    return result


def regime_feature_names(
    data: pd.DataFrame,
) -> list[str]:
    """
    Return regime-derived feature names.
    """

    base_columns = {
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    }

    return [
        column
        for column in data.columns
        if column not in base_columns
    ]
