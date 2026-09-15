from __future__ import annotations
import pandas as pd


def add_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["higher_high"] = (x["high"] > x["high"].shift(1)).astype(int)
    x["higher_low"] = (x["low"] > x["low"].shift(1)).astype(int)
    x["bullish_candle"] = (x["close"] > x["open"]).astype(int)
    x["body_pct"] = (x["close"] - x["open"]) / x["open"].replace(0, pd.NA)
    x["close_location"] = (x["close"] - x["low"]) / (x["high"] - x["low"]).replace(0, pd.NA)
    return x
