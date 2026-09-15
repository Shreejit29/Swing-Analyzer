"""
AI Swing Analyser — Indian Market Context.

Provides market-level context for stock prediction.

Supported benchmarks:
    NIFTY 50
    SENSEX
    NIFTY Bank

The module generates causal market features such as:
- benchmark returns
- moving-average trend
- momentum
- rolling volatility
- relative strength
- market breadth-style score

Important:
- Features are calculated from historical benchmark data only.
- No future values are used.
- The current stock prediction model is not trained here.
- This module does not generate BUY/SELL decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd


DEFAULT_MARKET_SYMBOLS: dict[str, str] = {
    "NIFTY50": "^NSEI",
    "SENSEX": "^BSESN",
    "NIFTYBANK": "^NSEBANK",
}


@dataclass(frozen=True)
class MarketContextConfig:
    """Configuration for market-context feature generation."""

    benchmark_names: tuple[str, ...] = (
        "NIFTY50",
        "SENSEX",
        "NIFTYBANK",
    )

    return_windows: tuple[int, ...] = (
        1,
        3,
        5,
        10,
        20,
    )

    moving_average_windows: tuple[int, ...] = (
        20,
        50,
        200,
    )

    volatility_windows: tuple[int, ...] = (
        10,
        20,
        60,
    )

    momentum_windows: tuple[int, ...] = (
        5,
        10,
        20,
    )

    relative_strength_window: int = 20

    def __post_init__(self) -> None:
        if not self.benchmark_names:
            raise ValueError(
                "At least one benchmark is required."
            )

        if any(
            not isinstance(window, int)
            or isinstance(window, bool)
            or window <= 0
            for window in (
                self.return_windows
                + self.moving_average_windows
                + self.volatility_windows
                + self.momentum_windows
            )
        ):
            raise ValueError(
                "All market-context windows must be positive integers."
            )

        if (
            not isinstance(
                self.relative_strength_window,
                int,
            )
            or isinstance(
                self.relative_strength_window,
                bool,
            )
            or self.relative_strength_window <= 0
        ):
            raise ValueError(
                "relative_strength_window must be a positive integer."
            )


@dataclass
class MarketContextResult:
    """Result of market-context feature generation."""

    data: pd.DataFrame

    benchmark_features: dict[
        str,
        list[str],
    ] = field(
        default_factory=dict
    )

    feature_names: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, object] = field(
        default_factory=dict
    )

    warnings: list[str] = field(
        default_factory=list
    )

    def summary(self) -> dict[str, object]:
        return {
            "rows": len(self.data),
            "features": len(
                self.feature_names
            ),
            "benchmarks": list(
                self.benchmark_features.keys()
            ),
            "feature_names": list(
                self.feature_names
            ),
            "warning_count": len(
                self.warnings
            ),
        }


def _validate_price_series(
    series: pd.Series,
    name: str,
) -> pd.Series:
    if not isinstance(
        series,
        pd.Series,
    ):
        raise TypeError(
            f"{name} must be a pandas Series."
        )

    if not isinstance(
        series.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            f"{name} must have a DatetimeIndex."
        )

    if series.index.has_duplicates:
        raise ValueError(
            f"{name} contains duplicate timestamps."
        )

    if not series.index.is_monotonic_increasing:
        raise ValueError(
            f"{name} must be chronologically sorted."
        )

    result = pd.to_numeric(
        series,
        errors="coerce",
    ).astype(float)

    if result.isna().all():
        raise ValueError(
            f"{name} contains no valid numeric prices."
        )

    if (result.dropna() <= 0).any():
        raise ValueError(
            f"{name} contains non-positive prices."
        )

    return result


def _returns(
    close: pd.Series,
    windows: tuple[int, ...],
) -> pd.DataFrame:
    result = pd.DataFrame(
        index=close.index
    )

    for window in windows:
        result[
            f"Market_Return_{window}"
        ] = close.pct_change(
            window
        )

    return result


def _moving_averages(
    close: pd.Series,
    windows: tuple[int, ...],
) -> pd.DataFrame:
    result = pd.DataFrame(
        index=close.index
    )

    for window in windows:
        ma = close.rolling(
            window=window,
            min_periods=window,
        ).mean()

        result[
            f"Market_MA_{window}"
        ] = ma

        result[
            f"Market_Distance_MA_{window}"
        ] = (
            close / ma
        ) - 1.0

    return result


def _volatility(
    close: pd.Series,
    windows: tuple[int, ...],
) -> pd.DataFrame:
    result = pd.DataFrame(
        index=close.index
    )

    returns = close.pct_change()

    for window in windows:
        result[
            f"Market_Volatility_{window}"
        ] = returns.rolling(
            window=window,
            min_periods=window,
        ).std()

    return result


def _momentum(
    close: pd.Series,
    windows: tuple[int, ...],
) -> pd.DataFrame:
    result = pd.DataFrame(
        index=close.index
    )

    for window in windows:
        result[
            f"Market_Momentum_{window}"
        ] = (
            close
            / close.shift(window)
        ) - 1.0

    return result


def _trend_score(
    close: pd.Series,
) -> pd.Series:
    ma20 = close.rolling(
        20,
        min_periods=20,
    ).mean()

    ma50 = close.rolling(
        50,
        min_periods=50,
    ).mean()

    ma200 = close.rolling(
        200,
        min_periods=200,
    ).mean()

    score = pd.Series(
        np.nan,
        index=close.index,
        dtype=float,
    )

    score = score.mask(
        close > ma20,
        1.0,
    )

    score = score.mask(
        close > ma50,
        score.fillna(0.0) + 1.0,
    )

    score = score.mask(
        close > ma200,
        score.fillna(0.0) + 1.0,
    )

    bearish = (
        (close < ma20)
        & (close < ma50)
        & (close < ma200)
    )

    bullish = (
        (close > ma20)
        & (close > ma50)
        & (close > ma200)
    )

    score = pd.Series(
        0.0,
        index=close.index,
    )

    score = score.mask(
        bullish,
        1.0,
    )

    score = score.mask(
        bearish,
        -1.0,
    )

    return score


def _market_regime(
    trend_score: pd.Series,
    volatility: pd.Series,
) -> pd.Series:
    """
    Simple causal market regime classification.

    The regime is intentionally conservative:
        BULL
        BEAR
        NEUTRAL
    """

    result = pd.Series(
        "UNKNOWN",
        index=trend_score.index,
        dtype="object",
    )

    result = result.mask(
        trend_score > 0,
        "BULL",
    )

    result = result.mask(
        trend_score < 0,
        "BEAR",
    )

    result = result.mask(
        trend_score == 0,
        "NEUTRAL",
    )

    result = result.where(
        volatility.notna(),
        "UNKNOWN",
    )

    return result


def build_benchmark_features(
    close: pd.Series,
    *,
    benchmark_name: str,
    config: MarketContextConfig | None = None,
) -> pd.DataFrame:
    """
    Build causal features for one market benchmark.
    """

    config = (
        config
        if config is not None
        else MarketContextConfig()
    )

    close = _validate_price_series(
        close,
        benchmark_name,
    )

    result = pd.DataFrame(
        index=close.index
    )

    result[
        f"{benchmark_name}_Close"
    ] = close

    returns = _returns(
        close,
        config.return_windows,
    )

    returns.columns = [
        f"{benchmark_name}_{column}"
        for column in returns.columns
    ]

    result = pd.concat(
        [
            result,
            returns,
        ],
        axis=1,
    )

    moving = _moving_averages(
        close,
        config.moving_average_windows,
    )

    moving.columns = [
        f"{benchmark_name}_{column}"
        for column in moving.columns
    ]

    result = pd.concat(
        [
            result,
            moving,
        ],
        axis=1,
    )

    volatility = _volatility(
        close,
        config.volatility_windows,
    )

    volatility.columns = [
        f"{benchmark_name}_{column}"
        for column in volatility.columns
    ]

    result = pd.concat(
        [
            result,
            volatility,
        ],
        axis=1,
    )

    momentum = _momentum(
        close,
        config.momentum_windows,
    )

    momentum.columns = [
        f"{benchmark_name}_{column}"
        for column in momentum.columns
    ]

    result = pd.concat(
        [
            result,
            momentum,
        ],
        axis=1,
    )

    trend = _trend_score(
        close
    )

    result[
        f"{benchmark_name}_Trend_Score"
    ] = trend

    volatility20 = close.pct_change().rolling(
        20,
        min_periods=20,
    ).std()

    result[
        f"{benchmark_name}_Regime"
    ] = _market_regime(
        trend,
        volatility20,
    )

    return result


def build_market_context(
    benchmark_data: Mapping[
        str,
        pd.Series | pd.DataFrame,
    ],
    *,
    config: MarketContextConfig | None = None,
) -> MarketContextResult:
    """
    Build a combined market-context DataFrame.

    benchmark_data maps benchmark names to either:
        - a Close-price Series
        - an OHLCV DataFrame containing Close
    """

    config = (
        config
        if config is not None
        else MarketContextConfig()
    )

    if not benchmark_data:
        raise ValueError(
            "benchmark_data cannot be empty."
        )

    combined: pd.DataFrame | None = None

    benchmark_features: dict[
        str,
        list[str],
    ] = {}

    warnings: list[str] = []

    for benchmark_name in config.benchmark_names:
        if benchmark_name not in benchmark_data:
            warnings.append(
                f"Benchmark {benchmark_name} was not supplied."
            )
            continue

        source = benchmark_data[
            benchmark_name
        ]

        if isinstance(
            source,
            pd.DataFrame,
        ):
            if "Close" not in source.columns:
                raise ValueError(
                    f"{benchmark_name} DataFrame must contain Close."
                )

            close = source["Close"]

        elif isinstance(
            source,
            pd.Series,
        ):
            close = source

        else:
            raise TypeError(
                f"Unsupported data type for {benchmark_name}."
            )

        features = build_benchmark_features(
            close,
            benchmark_name=benchmark_name,
            config=config,
        )

        benchmark_features[
            benchmark_name
        ] = list(
            features.columns
        )

        if combined is None:
            combined = features.copy()
        else:
            combined = combined.join(
                features,
                how="outer",
            )

    if combined is None:
        raise ValueError(
            "No configured benchmark data was supplied."
        )

    combined = combined.sort_index()

    return MarketContextResult(
        data=combined,
        benchmark_features=benchmark_features,
        feature_names=list(
            combined.columns
        ),
        metadata={
            "research_only": True,
            "future_values_used": False,
            "feature_generation": "causal",
        },
        warnings=warnings,
    )


def add_relative_strength(
    stock_close: pd.Series,
    market_close: pd.Series,
    *,
    window: int = 20,
    prefix: str = "Relative_Strength",
) -> pd.DataFrame:
    """
    Calculate stock-vs-market relative strength.

    Both series are aligned by timestamp. The calculation uses only
    historical returns up to the current timestamp.
    """

    if (
        not isinstance(
            stock_close,
            pd.Series,
        )
        or not isinstance(
            market_close,
            pd.Series,
        )
    ):
        raise TypeError(
            "stock_close and market_close must be Series."
        )

    if window <= 0:
        raise ValueError(
            "window must be positive."
        )

    stock = _validate_price_series(
        stock_close,
        "stock_close",
    )

    market = _validate_price_series(
        market_close,
        "market_close",
    )

    aligned = pd.concat(
        [
            stock.rename("Stock"),
            market.rename("Market"),
        ],
        axis=1,
        join="inner",
    )

    if aligned.empty:
        raise ValueError(
            "Stock and market series have no overlapping timestamps."
        )

    stock_return = aligned[
        "Stock"
    ].pct_change(window)

    market_return = aligned[
        "Market"
    ].pct_change(window)

    result = pd.DataFrame(
        index=aligned.index
    )

    result[
        f"{prefix}_{window}"
    ] = (
        stock_return
        - market_return
    )

    result[
        f"{prefix}_Ratio_{window}"
    ] = (
        (1.0 + stock_return)
        / (1.0 + market_return)
    ) - 1.0

    return result


def market_strength_score(
    market_context: pd.DataFrame,
) -> pd.Series:
    """
    Create a simple aggregate market-strength score.

    The score is based on available benchmark trend scores.
    """

    if not isinstance(
        market_context,
        pd.DataFrame,
    ):
        raise TypeError(
            "market_context must be a DataFrame."
        )

    trend_columns = [
        column
        for column in market_context.columns
        if column.endswith(
            "_Trend_Score"
        )
    ]

    if not trend_columns:
        raise ValueError(
            "No benchmark trend-score columns were found."
        )

    score = market_context[
        trend_columns
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    ).mean(
        axis=1,
        skipna=True,
    )

    return score.rename(
        "Market_Strength_Score"
    )


def market_context_feature_names(
    context: MarketContextResult,
) -> list[str]:
    """Return generated market feature names."""

    if not isinstance(
        context,
        MarketContextResult,
    ):
        raise TypeError(
            "context must be a MarketContextResult."
        )

    return list(
        context.feature_names
    )


__all__ = [
    "DEFAULT_MARKET_SYMBOLS",
    "MarketContextConfig",
    "MarketContextResult",
    "build_benchmark_features",
    "build_market_context",
    "add_relative_strength",
    "market_strength_score",
    "market_context_feature_names",
]
