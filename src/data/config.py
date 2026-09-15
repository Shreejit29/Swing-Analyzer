"""
Central configuration for the historical market-data engine.

Keeping data-source settings in one place makes it easier to replace
the prototype provider later without changing the rest of the system.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class DataSourceConfig:
    """Configuration for a market-data source."""

    provider: str = "yfinance"

    # Daily/weekly/monthly history can be downloaded over long periods.
    daily_period: str = "max"
    weekly_period: str = "max"
    monthly_period: str = "max"

    # Prototype limitation: intraday history is intentionally restricted.
    intraday_period: str = "60d"

    # We currently use 1-hour data and resample it into 4-hour candles.
    intraday_interval: str = "1h"


MARKET_SYMBOLS: Dict[str, str] = {
    "NIFTY50": "^NSEI",
    "SENSEX": "^BSESN",
    "NIFTYBANK": "^NSEBANK",
}


DEFAULT_DATA_SOURCE = DataSourceConfig()


SUPPORTED_TIMEFRAMES = (
    "4H",
    "1D",
    "1W",
    "1M",
)
