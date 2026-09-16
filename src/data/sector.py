# src/data/sector.py

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Sector mapping
# ---------------------------------------------------------------------

SECTOR_MAP = {
    # Banking / Financial Services
    "HDFCBANK.NS": "BANKING",
    "ICICIBANK.NS": "BANKING",
    "SBIN.NS": "BANKING",
    "AXISBANK.NS": "BANKING",
    "KOTAKBANK.NS": "BANKING",
    "INDUSINDBK.NS": "BANKING",
    "BANKBARODA.NS": "BANKING",

    "BAJFINANCE.NS": "FINANCIAL SERVICES",
    "BAJAJFINSV.NS": "FINANCIAL SERVICES",
    "SHRIRAMFIN.NS": "FINANCIAL SERVICES",

    # IT
    "TCS.NS": "IT",
    "INFY.NS": "IT",
    "HCLTECH.NS": "IT",
    "WIPRO.NS": "IT",
    "TECHM.NS": "IT",
    "LTIM.NS": "IT",

    # Energy
    "RELIANCE.NS": "ENERGY",
    "ONGC.NS": "ENERGY",
    "BPCL.NS": "ENERGY",
    "IOC.NS": "ENERGY",
    "NTPC.NS": "POWER",
    "POWERGRID.NS": "POWER",

    # Auto
    "MARUTI.NS": "AUTOMOBILE",
    "TATAMOTORS.NS": "AUTOMOBILE",
    "M&M.NS": "AUTOMOBILE",
    "EICHERMOT.NS": "AUTOMOBILE",
    "HEROMOTOCO.NS": "AUTOMOBILE",
    "BAJAJ-AUTO.NS": "AUTOMOBILE",

    # FMCG
    "HINDUNILVR.NS": "FMCG",
    "ITC.NS": "FMCG",
    "NESTLEIND.NS": "FMCG",
    "BRITANNIA.NS": "FMCG",
    "TATACONSUM.NS": "FMCG",

    # Pharma / Healthcare
    "SUNPHARMA.NS": "PHARMA",
    "CIPLA.NS": "PHARMA",
    "DRREDDY.NS": "PHARMA",
    "DIVISLAB.NS": "PHARMA",
    "APOLLOHOSP.NS": "HEALTHCARE",

    # Metals
    "TATASTEEL.NS": "METALS",
    "HINDALCO.NS": "METALS",
    "JSWSTEEL.NS": "METALS",
    "COALINDIA.NS": "METALS",

    # Telecom
    "BHARTIARTL.NS": "TELECOM",

    # Infrastructure / Construction
    "LT.NS": "INFRASTRUCTURE",
    "ADANIENT.NS": "INFRASTRUCTURE",
    "ADANIPORTS.NS": "INFRASTRUCTURE",

    # Consumer / Retail
    "TITAN.NS": "CONSUMER",
    "TRENT.NS": "CONSUMER",

    # Cement
    "ULTRACEMCO.NS": "CEMENT",
    "GRASIM.NS": "CEMENT",

    # Conglomerate
    "ADANIENT.NS": "CONGLOMERATE",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:
        if pd.isna(value):
            return default

        return float(value)

    except (TypeError, ValueError):
        return default


def _normalise_symbol(
    symbol: Optional[str],
) -> str:

    if symbol is None:
        return ""

    return str(symbol).strip().upper()


def _normalise_ohlcv(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # Handle Yahoo MultiIndex
    if isinstance(out.columns, pd.MultiIndex):

        flattened = []

        for col in out.columns:

            parts = [
                str(x).strip()
                for x in col
                if str(x).strip().lower()
                not in {"nan", "none"}
            ]

            flattened.append("_".join(parts))

        out.columns = flattened

    rename = {}

    for col in out.columns:

        name = str(col).strip().lower()

        if name in {"open", "open_price"}:
            rename[col] = "open"

        elif name in {"high", "high_price"}:
            rename[col] = "high"

        elif name in {"low", "low_price"}:
            rename[col] = "low"

        elif name in {
            "close",
            "close_price",
            "adj close",
        }:
            rename[col] = "close"

        elif name in {
            "volume",
            "vol",
        }:
            rename[col] = "volume"

    out = out.rename(columns=rename)

    out = out.loc[
        :,
        ~out.columns.duplicated(),
    ]

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:

        if col in out.columns:

            out[col] = pd.to_numeric(
                out[col],
                errors="coerce",
            )

    return out


# ---------------------------------------------------------------------
# Sector lookup
# ---------------------------------------------------------------------

def get_sector(
    symbol: Optional[str],
) -> str:

    symbol = _normalise_symbol(symbol)

    if not symbol:
        return "UNKNOWN"

    if symbol in SECTOR_MAP:
        return SECTOR_MAP[symbol]

    # Some common naming patterns
    symbol_without_suffix = symbol.replace(".NS", "")

    pattern_map = {
        "BANK": "BANKING",
        "FINANCE": "FINANCIAL SERVICES",
        "TECH": "IT",
        "PHARMA": "PHARMA",
        "AUTO": "AUTOMOBILE",
        "POWER": "POWER",
        "STEEL": "METALS",
        "CEMENT": "CEMENT",
    }

    for pattern, sector in pattern_map.items():

        if pattern in symbol_without_suffix:
            return sector

    return "UNKNOWN"


# ---------------------------------------------------------------------
# Sector information
# ---------------------------------------------------------------------

def sector_info(
    symbol: Optional[str],
) -> Dict[str, str]:

    normalized_symbol = _normalise_symbol(symbol)

    return {
        "symbol": normalized_symbol,
        "sector": get_sector(normalized_symbol),
    }


# ---------------------------------------------------------------------
# Sector performance
# ---------------------------------------------------------------------

def sector_performance(
    df: pd.DataFrame,
) -> Dict[str, Any]:

    data = _normalise_ohlcv(df)

    if data.empty or "close" not in data.columns:
        return {
            "return_5d": 0.0,
            "return_20d": 0.0,
            "trend": "UNKNOWN",
            "momentum": "UNKNOWN",
            "strength": 0.0,
        }

    close = data["close"].dropna()

    if len(close) < 5:

        return {
            "return_5d": 0.0,
            "return_20d": 0.0,
            "trend": "UNKNOWN",
            "momentum": "UNKNOWN",
            "strength": 0.0,
        }

    return_5d = (
        close.iloc[-1] / close.iloc[-5] - 1
    ) * 100.0

    if len(close) >= 20:

        return_20d = (
            close.iloc[-1] / close.iloc[-20] - 1
        ) * 100.0

    else:
        return_20d = return_5d

    # Trend
    if len(close) >= 20:

        ema20 = close.ewm(
            span=20,
            adjust=False,
        ).mean()

        latest_close = _safe_float(
            close.iloc[-1]
        )

        latest_ema20 = _safe_float(
            ema20.iloc[-1]
        )

        if latest_close > latest_ema20:
            trend = "BULLISH"

        elif latest_close < latest_ema20:
            trend = "BEARISH"

        else:
            trend = "SIDEWAYS"

    else:
        trend = "UNKNOWN"

    # Momentum
    if return_5d > 2:
        momentum = "STRONG POSITIVE"

    elif return_5d > 0:
        momentum = "POSITIVE"

    elif return_5d < -2:
        momentum = "STRONG NEGATIVE"

    elif return_5d < 0:
        momentum = "NEGATIVE"

    else:
        momentum = "NEUTRAL"

    # Combined strength
    strength = (
        0.4 * return_5d
        + 0.6 * return_20d
    )

    return {
        "return_5d": round(
            float(return_5d),
            3,
        ),
        "return_20d": round(
            float(return_20d),
            3,
        ),
        "trend": trend,
        "momentum": momentum,
        "strength": round(
            float(strength),
            3,
        ),
    }


# ---------------------------------------------------------------------
# Sector score
# ---------------------------------------------------------------------

def sector_score(
    df: pd.DataFrame,
) -> float:

    result = sector_performance(df)

    strength = _safe_float(
        result.get("strength"),
        0.0,
    )

    # Convert return-based strength
    # approximately to -100 ... +100
    score = np.tanh(strength / 5.0) * 100.0

    return round(
        float(score),
        2,
    )


# ---------------------------------------------------------------------
# Sector summary
# ---------------------------------------------------------------------

def sector_summary(
    symbol: Optional[str],
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:

    sector = get_sector(symbol)

    result: Dict[str, Any] = {
        "symbol": _normalise_symbol(symbol),
        "sector": sector,
        "trend": "UNKNOWN",
        "momentum": "UNKNOWN",
        "return_5d": 0.0,
        "return_20d": 0.0,
        "strength": 0.0,
        "score": 0.0,
    }

    if df is None or df.empty:
        return result

    performance = sector_performance(df)

    result.update(
        {
            "trend": performance["trend"],
            "momentum": performance["momentum"],
            "return_5d": performance["return_5d"],
            "return_20d": performance["return_20d"],
            "strength": performance["strength"],
            "score": sector_score(df),
        }
    )

    return result


# ---------------------------------------------------------------------
# Compatibility aliases
# ---------------------------------------------------------------------

get_sector_name = get_sector
get_sector_info = sector_info
get_sector_performance = sector_performance
get_sector_score = sector_score
get_sector_summary = sector_summary


__all__ = [
    "SECTOR_MAP",
    "get_sector",
    "sector_info",
    "sector_performance",
    "sector_score",
    "sector_summary",
    "get_sector_name",
    "get_sector_info",
    "get_sector_performance",
    "get_sector_score",
    "get_sector_summary",
]
