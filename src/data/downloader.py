from __future__ import annotations

from io import BytesIO
from typing import Optional

import pandas as pd
import yfinance as yf


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    aliases = {"adj_close": "close"}
    df = df.rename(columns=aliases)
    return df


def prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("No market data was supplied.")
    out = _flatten_columns(df.copy())
    if "date" in out.columns and not isinstance(out.index, pd.DatetimeIndex):
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.set_index("date")
    elif "datetime" in out.columns and not isinstance(out.index, pd.DatetimeIndex):
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
        out = out.set_index("datetime")
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {', '.join(missing)}")
    out = out[required].copy()
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[~out.index.isna()].sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.dropna(subset=["open", "high", "low", "close"])
    out["volume"] = out["volume"].fillna(0)
    return out


def download_data(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    symbol = symbol.strip().upper()
    if not symbol:
        raise ValueError("Enter a stock symbol.")
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        return prepare_ohlcv(df)
    except Exception as exc:
        raise RuntimeError(f"Yahoo Finance download failed: {exc}") from exc


def load_csv_data(file) -> pd.DataFrame:
    if isinstance(file, (bytes, bytearray)):
        df = pd.read_csv(BytesIO(file))
    else:
        df = pd.read_csv(file)
    return prepare_ohlcv(df)
