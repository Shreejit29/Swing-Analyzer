from __future__ import annotations

import numpy as np
import pandas as pd


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    close, high, low, volume = x["close"], x["high"], x["low"], x["volume"]
    x["ema20"] = close.ewm(span=20, adjust=False).mean()
    x["ema50"] = close.ewm(span=50, adjust=False).mean()
    x["ema200"] = close.ewm(span=200, adjust=False).mean()
    x["sma20"] = close.rolling(20).mean()
    x["sma50"] = close.rolling(50).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi14"] = 100 - (100 / (1 + rs))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    x["macd"] = ema12 - ema26
    x["macd_signal"] = x["macd"].ewm(span=9, adjust=False).mean()
    x["macd_hist"] = x["macd"] - x["macd_signal"]
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    x["atr14"] = tr.rolling(14).mean()
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr = x["atr14"].replace(0, np.nan)
    plus_di = 100 * plus_dm.rolling(14).mean() / atr
    minus_di = 100 * minus_dm.rolling(14).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    x["adx14"] = dx.rolling(14).mean()
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    x["bb_mid"] = mid
    x["bb_upper"] = mid + 2 * std
    x["bb_lower"] = mid - 2 * std
    x["bb_width"] = (x["bb_upper"] - x["bb_lower"]) / mid.replace(0, np.nan)
    x["roc10"] = close.pct_change(10)
    x["volume_sma20"] = volume.rolling(20).mean()
    x["relative_volume"] = volume / x["volume_sma20"].replace(0, np.nan)
    x["obv"] = (np.sign(close.diff()).fillna(0) * volume).cumsum()
    x["daily_return"] = close.pct_change()
    x["volatility20"] = x["daily_return"].rolling(20).std()
    x["distance_ema20"] = close / x["ema20"] - 1
    x["distance_ema50"] = close / x["ema50"] - 1
    x["distance_ema200"] = close / x["ema200"] - 1
    x["high_20"] = high.rolling(20).max()
    x["low_20"] = low.rolling(20).min()
    x["breakout_20"] = (close > x["high_20"].shift(1)).astype(int)
    x["range_pct"] = (high - low) / close.replace(0, np.nan)
    return x
