from __future__ import annotations
import yfinance as yf

SECTOR_ETFS = {
    "BANK": "^NSEBANK",
    "IT": "^CNXIT",
    "PHARMA": "^CNXPHARMA",
    "AUTO": "^CNXAUTO",
    "FINANCE": "^CNXFIN",
    "METAL": "^CNXMETAL",
}


def sector_snapshot(period: str = "1mo"):
    result = {}
    for name, ticker in SECTOR_ETFS.items():
        try:
            df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
            if not df.empty:
                close = df["Close"].squeeze()
                result[name] = float(close.iloc[-1] / close.iloc[0] - 1)
        except Exception:
            pass
    return result
