from __future__ import annotations
import pandas as pd


def resample_ohlcv(df: pd.DataFrame, rule: str = "4h") -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df.resample(rule).agg(agg).dropna(subset=["open", "high", "low", "close"])
