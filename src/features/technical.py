from __future__ import annotations

import numpy as np
import pandas as pd


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the technical features used by the swing analyzer.

    The indicators are deliberately based only on current and historical
    OHLCV data so they can also be used safely during walk-forward
    backtesting.
    """
    if df is None or df.empty:
        raise ValueError("No OHLCV data supplied for technical analysis.")

    required = ["open", "high", "low", "close", "volume"]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Technical analysis requires: "
            + ", ".join(missing)
        )

    x = df.copy()

    # Ensure all market fields are numeric 1-D Series.
    for column in required:
        x[column] = pd.to_numeric(
            x[column],
            errors="coerce",
        )

    close = x["close"]
    high = x["high"]
    low = x["low"]
    volume = x["volume"]

    # ---------------------------------------------------------------
    # Trend
    # ---------------------------------------------------------------
    x["ema20"] = close.ewm(
        span=20,
        adjust=False,
        min_periods=20,
    ).mean()

    x["ema50"] = close.ewm(
        span=50,
        adjust=False,
        min_periods=50,
    ).mean()

    x["ema200"] = close.ewm(
        span=200,
        adjust=False,
        min_periods=200,
    ).mean()

    x["sma20"] = close.rolling(
        20,
        min_periods=20,
    ).mean()

    x["sma50"] = close.rolling(
        50,
        min_periods=50,
    ).mean()

    # Trend relationships.
    x["ema20_slope"] = x["ema20"].pct_change(5)
    x["ema50_slope"] = x["ema50"].pct_change(10)

    x["ema20_above_ema50"] = (
        x["ema20"] > x["ema50"]
    ).astype(int)

    x["ema50_above_ema200"] = (
        x["ema50"] > x["ema200"]
    ).astype(int)

    # ---------------------------------------------------------------
    # RSI
    # ---------------------------------------------------------------
    delta = close.diff()

    gain = delta.clip(
        lower=0
    ).rolling(
        14,
        min_periods=14,
    ).mean()

    loss = (
        -delta.clip(upper=0)
    ).rolling(
        14,
        min_periods=14,
    ).mean()

    rs = gain / loss.replace(
        0,
        np.nan,
    )

    x["rsi14"] = 100 - (
        100 / (1 + rs)
    )

    # ---------------------------------------------------------------
    # MACD
    # ---------------------------------------------------------------
    ema12 = close.ewm(
        span=12,
        adjust=False,
        min_periods=12,
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False,
        min_periods=26,
    ).mean()

    x["macd"] = ema12 - ema26

    x["macd_signal"] = x["macd"].ewm(
        span=9,
        adjust=False,
        min_periods=9,
    ).mean()

    x["macd_hist"] = (
        x["macd"] - x["macd_signal"]
    )

    x["macd_bullish"] = (
        x["macd"] > x["macd_signal"]
    ).astype(int)

    # ---------------------------------------------------------------
    # ATR
    # ---------------------------------------------------------------
    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    x["atr14"] = true_range.rolling(
        14,
        min_periods=14,
    ).mean()

    x["atr_pct"] = (
        x["atr14"] / close.replace(0, np.nan)
    )

    # ---------------------------------------------------------------
    # ADX / directional movement
    # ---------------------------------------------------------------
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where(
        (up_move > down_move)
        & (up_move > 0),
        0.0,
    )

    minus_dm = down_move.where(
        (down_move > up_move)
        & (down_move > 0),
        0.0,
    )

    atr = x["atr14"].replace(
        0,
        np.nan,
    )

    plus_di = (
        100
        * plus_dm.rolling(
            14,
            min_periods=14,
        ).mean()
        / atr
    )

    minus_di = (
        100
        * minus_dm.rolling(
            14,
            min_periods=14,
        ).mean()
        / atr
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(
            0,
            np.nan,
        )
    )

    x["plus_di14"] = plus_di
    x["minus_di14"] = minus_di

    x["adx14"] = dx.rolling(
        14,
        min_periods=14,
    ).mean()

    # ---------------------------------------------------------------
    # Bollinger Bands
    # ---------------------------------------------------------------
    bb_mid = close.rolling(
        20,
        min_periods=20,
    ).mean()

    bb_std = close.rolling(
        20,
        min_periods=20,
    ).std()

    x["bb_mid"] = bb_mid
    x["bb_upper"] = bb_mid + 2 * bb_std
    x["bb_lower"] = bb_mid - 2 * bb_std

    x["bb_width"] = (
        x["bb_upper"] - x["bb_lower"]
    ) / bb_mid.replace(0, np.nan)

    x["bb_position"] = (
        (close - x["bb_lower"])
        / (
            x["bb_upper"]
            - x["bb_lower"]
        ).replace(0, np.nan)
    )

    # ---------------------------------------------------------------
    # Momentum
    # ---------------------------------------------------------------
    x["roc10"] = close.pct_change(
        10
    )

    x["roc20"] = close.pct_change(
        20
    )

    # ---------------------------------------------------------------
    # Volume
    # ---------------------------------------------------------------
    x["volume_sma20"] = volume.rolling(
        20,
        min_periods=20,
    ).mean()

    x["relative_volume"] = (
        volume
        / x["volume_sma20"].replace(
            0,
            np.nan,
        )
    )

    x["volume_change"] = volume.pct_change()

    # OBV.
    direction = np.sign(
        close.diff()
    ).fillna(0)

    x["obv"] = (
        direction * volume
    ).cumsum()

    x["obv_slope"] = x["obv"].pct_change(
        10
    )

    # ---------------------------------------------------------------
    # Volatility
    # ---------------------------------------------------------------
    x["daily_return"] = close.pct_change()

    x["volatility20"] = (
        x["daily_return"]
        .rolling(
            20,
            min_periods=20,
        )
        .std()
    )

    x["volatility20_pct"] = (
        x["volatility20"]
        * np.sqrt(252)
    )

    # ---------------------------------------------------------------
    # Price relative to trend
    # ---------------------------------------------------------------
    x["distance_ema20"] = (
        close / x["ema20"] - 1
    )

    x["distance_ema50"] = (
        close / x["ema50"] - 1
    )

    x["distance_ema200"] = (
        close / x["ema200"] - 1
    )

    # ---------------------------------------------------------------
    # Support / resistance and breakouts
    # ---------------------------------------------------------------
    x["high_20"] = high.rolling(
        20,
        min_periods=20,
    ).max()

    x["low_20"] = low.rolling(
        20,
        min_periods=20,
    ).min()

    x["high_50"] = high.rolling(
        50,
        min_periods=50,
    ).max()

    x["low_50"] = low.rolling(
        50,
        min_periods=50,
    ).min()

    # Shift first to avoid treating today's high as a breakout level.
    x["breakout_20"] = (
        close > x["high_20"].shift(1)
    ).astype(int)

    x["breakdown_20"] = (
        close < x["low_20"].shift(1)
    ).astype(int)

    x["breakout_50"] = (
        close > x["high_50"].shift(1)
    ).astype(int)

    x["breakdown_50"] = (
        close < x["low_50"].shift(1)
    ).astype(int)

    # Distance from recent high/low.
    x["distance_high20"] = (
        close / x["high_20"] - 1
    )

    x["distance_low20"] = (
        close / x["low_20"] - 1
    )

    # ---------------------------------------------------------------
    # Candle range / volatility structure
    # ---------------------------------------------------------------
    x["range_pct"] = (
        (high - low)
        / close.replace(0, np.nan)
    )

    x["gap_pct"] = (
        x["open"] / close.shift(1) - 1
    )

    # ---------------------------------------------------------------
    # Clean invalid numeric values.
    # ---------------------------------------------------------------
    x = x.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return x
