"""
Data package for the AI Swing Stock Analyzer.

Provides:
- Yahoo Finance data downloading
- Market context analysis
- Sector analysis
"""

from .downloader import (
    download_data,
)

from .market import (
    market_trend,
    market_momentum,
    market_volatility,
    market_strength,
    market_volume,
    market_summary,
    get_market_trend,
    get_market_momentum,
    get_market_summary,
    get_market_strength,
)

from .sector import (
    get_sector,
    sector_info,
    sector_performance,
    sector_score,
    sector_summary,
    get_sector_name,
    get_sector_info,
    get_sector_performance,
    get_sector_score,
    get_sector_summary,
)


__all__ = [
    # Downloader
    "download_data",

    # Market
    "market_trend",
    "market_momentum",
    "market_volatility",
    "market_strength",
    "market_volume",
    "market_summary",
    "get_market_trend",
    "get_market_momentum",
    "get_market_summary",
    "get_market_strength",

    # Sector
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
