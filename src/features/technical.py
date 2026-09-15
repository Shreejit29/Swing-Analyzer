"""
Technical indicators for the Swing Analyzer.

Input contract:
    lowercase OHLCV columns: open, high, low, close, volume

The function is deliberately dependency-light and works with pandas/numpy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _series(df: pd.DataFrame, name: str) -> pd.Series:
    value = df[name]
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    return pd.to_numeric(value, errors="coerce")


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add robust technical-indicator features using the project OHLCV contract."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    required = {"open", "high", "low", "close", "volume"}
    missing = required - {str(c).lower() for c in df.columns}

    if missing:
        raise ValueError(
            f"Technical features require lowercase OHLCV columns. "
            f"Missing: {sorted(missing)}"
        )

    out = df.copy()

    # Ensure canonical lowercase columns if casing differs.
    for name in required:
        matching = [c for c in out.columns if str(c).lower() == name]
        if name not in out.columns and matching:
            out[name] = out[matching[0]]

    close = _series(out, "close")
    high = _series(out, "high")
    low = _series(out, "low")
    volume = _series(out, "volume")

    # Moving averages
    out["ema20"] = _ema(close, 20)
    out["ema50"] = _ema(close, 50)
    out["ema200"] = _ema(close, 200)
    out["sma20"] = close.rolling(20, min_periods=20).mean()
    out["sma50"] = close.rolling(50, min_periods=50).mean()

    out["ema20_slope"] = out["ema20"].pct_change(5) * 100
    out["ema50_slope"] = out["ema50"].pct_change(10) * 100
    out["ema200_slope"] = out["ema200"].pct_change(20) * 100

    out["close_above_ema20"] = (close > out["ema20"]).astype(int)
    out["close_above_ema50"] = (close > out["ema50"]).astype(int)
    out["close_above_ema200"] = (close > out["ema200"]).astype(int)
    out["ema20_above_ema50"] = (out["ema20"] > out["ema50"]).astype(int)
    out["ema50_above_ema200"] = (out["ema50"] > out["ema200"]).astype(int)

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi14"] = 100 - (100 / (1 + rs))
    out["rsi14"] = out["rsi14"].clip(0, 100)

    # MACD
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(
        span=9, adjust=False, min_periods=9
    ).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    out["macd_bullish"] = (out["macd"] > out["macd_signal"]).astype(int)

    # ATR / volatility
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    out["atr14"] = true_range.rolling(14, min_periods=14).mean()
    out["atr_pct"] = out["atr14"] / close.replace(0, np.nan) * 100

    # Bollinger Bands
    bb_mid = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std()

    out["bb_mid"] = bb_mid
    out["bb_upper"] = bb_mid + 2 * bb_std
    out["bb_lower"] = bb_mid - 2 * bb_std
    out["bb_width"] = (
        (out["bb_upper"] - out["bb_lower"])
        / bb_mid.replace(0, np.nan)
    )
    out["bb_position"] = (
        (close - out["bb_lower"])
        / (out["bb_upper"] - out["bb_lower"]).replace(0, np.nan)
    )

    # Rate of change
    out["roc10"] = close.pct_change(10) * 100
    out["roc20"] = close.pct_change(20) * 100

    # Volume
    out["volume_sma20"] = volume.rolling(20, min_periods=20).mean()
    out["relative_volume"] = (
        volume / out["volume_sma20"].replace(0, np.nan)
    )
    out["volume_change_pct"] = volume.pct_change() * 100

    # OBV
    direction = np.sign(close.diff()).fillna(0)
    out["obv"] = (direction * volume.fillna(0)).cumsum()
    out["obv_slope"] = out["obv"] - out["obv"].shift(10)

    # Returns and volatility
    out["daily_return"] = close.pct_change()
    out["volatility20"] = out["daily_return"].rolling(
        20, min_periods=20
    ).std()
    out["annualized_volatility"] = out["volatility20"] * np.sqrt(252)

    # Distances from moving averages
    out["distance_ema20_pct"] = (
        close / out["ema20"].replace(0, np.nan) - 1
    ) * 100
    out["distance_ema50_pct"] = (
        close / out["ema50"].replace(0, np.nan) - 1
    ) * 100
    out["distance_ema200_pct"] = (
        close / out["ema200"].replace(0, np.nan) - 1
    ) * 100

    # Rolling highs/lows and breakout flags.
    for window in (20, 50):
        previous_high = high.shift(1).rolling(
            window, min_periods=window
        ).max()
        previous_low = low.shift(1).rolling(
            window, min_periods=window
        ).min()

        out[f"high_{window}"] = previous_high
        out[f"low_{window}"] = previous_low
        out[f"breakout_{window}"] = (close > previous_high).astype(int)
        out[f"breakdown_{window}"] = (close < previous_low).astype(int)

    # Price range / gap
    out["range_pct"] = (high - low) / close.replace(0, np.nan) * 100
    out["gap_pct"] = (
        (out["open"] - previous_close)
        / previous_close.replace(0, np.nan)
        * 100
    )

    # ADX / directional movement
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0,
        ),
        index=out.index,
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0,
        ),
        index=out.index,
    )

    atr = out["atr14"].replace(0, np.nan)
    plus_di = 100 * plus_dm.rolling(14, min_periods=14).mean() / atr
    minus_di = 100 * minus_dm.rolling(14, min_periods=14).mean() / atr

    dx = (
        (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
        * 100
    )

    out["plus_di"] = plus_di
    out["minus_di"] = minus_di
    out["adx"] = dx.rolling(14, min_periods=14).mean()
    out["di_bullish"] = (plus_di > minus_di).astype(int)

    # Clean infinities but don't destroy the warm-up NaNs.
    numeric_columns = out.select_dtypes(include=[np.number]).columns
    out[numeric_columns] = out[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return out
