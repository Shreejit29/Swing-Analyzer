from __future__ import annotations
import pandas as pd


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["volume_change"] = x["volume"].pct_change()
    x["volume_z20"] = (x["volume"] - x["volume"].rolling(20).mean()) / x["volume"].rolling(20).std().replace(0, pd.NA)
    typical = (x["high"] + x["low"] + x["close"]) / 3
    x["vwap20"] = (typical * x["volume"]).rolling(20).sum() / x["volume"].rolling(20).sum().replace(0, pd.NA)
    x["distance_vwap20"] = x["close"] / x["vwap20"] - 1
    return x
