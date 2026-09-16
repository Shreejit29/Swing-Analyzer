"""
Feature engineering package for the AI Swing Stock Analyzer.

The package contains independent feature modules for:

- Technical indicators
- Price action
- Volume analysis
- Market regime
- Multi-timeframe analysis

The main feature pipeline is exposed through ``engine.py``.
"""

from .engine import (
    build_features,
    feature_columns,
    normalize_ohlcv,
    prepare_model_data,
)

from .technical import add_technical_features

from .price_action import add_price_action_features

from .volume import add_volume_features

from .regime import (
    add_regime_features,
    detect_regime,
    market_regime,
    get_regime_score,
)

from .multi_timeframe import (
    add_multi_timeframe_features,
    add_mtf_features,
)


__all__ = [
    # Engine
    "build_features",
    "feature_columns",
    "normalize_ohlcv",
    "prepare_model_data",

    # Technical
    "add_technical_features",

    # Price action
    "add_price_action_features",

    # Volume
    "add_volume_features",

    # Regime
    "add_regime_features",
    "detect_regime",
    "market_regime",
    "get_regime_score",

    # Multi-timeframe
    "add_multi_timeframe_features",
    "add_mtf_features",
]
