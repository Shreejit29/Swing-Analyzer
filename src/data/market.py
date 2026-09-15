from __future__ import annotations

import pandas as pd
import yfinance as yf


def download_market_context(period: str = "1y") -> pd.DataFrame:
    symbols = {
        "NIFTY50": "^NSEI",
        "NIFTY_BANK": "^NSEBANK",
        "SENSEX": "^BSESN",
        "INDIA_VIX": "^INDIAVIX",
    }
    frames = []
    for name, ticker in symbols.items():
        try:
            df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            frames.append(pd.DataFrame({name: df["Close"].squeeze()}))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)
