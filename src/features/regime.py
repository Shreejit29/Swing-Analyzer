from __future__ import annotations
import pandas as pd


def market_regime(row: pd.Series) -> str:
    close = float(row.get("close", 0))
    e20 = float(row.get("ema20", close))
    e50 = float(row.get("ema50", close))
    e200 = float(row.get("ema200", close))
    adx = float(row.get("adx14", 0) or 0)
    if close > e20 > e50 > e200 and adx >= 25:
        return "STRONG BULL"
    if close > e50 and e50 >= e200:
        return "BULL"
    if close < e20 < e50 < e200 and adx >= 25:
        return "STRONG BEAR"
    if close < e50 and e50 <= e200:
        return "BEAR"
    return "SIDEWAYS"
