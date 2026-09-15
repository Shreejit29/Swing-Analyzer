"""
AI Swing Analyser — Market Data Context Loader.

Connects the existing MarketDataProvider layer with the market-context
feature engine.

Responsibilities:
- Fetch benchmark OHLCV data.
- Normalize benchmark close series.
- Build market-context features.
- Optionally calculate stock-vs-market relative strength.
- Keep downloading separate from feature engineering.

This module does NOT:
- train models
- select models
- calibrate probabilities
- generate BUY/SELL decisions
- approve models
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd

from src.data.config import MARKET_SYMBOLS
from src.data.providers import (
    DataRequest,
    MarketDataProvider,
    YahooFinanceProvider,
)
from .market_context import (
    MarketContextConfig,
    MarketContextResult,
    add_relative_strength,
    build_market_context,
)


@dataclass(frozen=True)
class MarketContextRequest:
    """Request for benchmark market context."""

    start: str | pd.Timestamp
    end: str | pd.Timestamp

    benchmarks: tuple[str, ...] = (
        "NIFTY50",
        "SENSEX",
        "NIFTYBANK",
    )

    interval: str = "1d"

    def __post_init__(self) -> None:
        if not self.benchmarks:
            raise ValueError(
                "At least one benchmark is required."
            )

        if any(
            not isinstance(name, str)
            or not name.strip()
            for name in self.benchmarks
        ):
            raise ValueError(
                "Benchmark names must be non-empty strings."
            )

        if len(self.benchmarks) != len(
            set(self.benchmarks)
        ):
            raise ValueError(
                "Benchmark names must be unique."
            )

        if not isinstance(
            self.interval,
            str,
        ) or not self.interval.strip():
            raise ValueError(
                "interval must be a non-empty string."
            )


@dataclass
class MarketContextData:
    """Fetched benchmark data and generated features."""

    raw_data: dict[
        str,
        pd.DataFrame,
    ]

    context: MarketContextResult

    metadata: dict[str, object] = field(
        default_factory=dict
    )

    warnings: list[str] = field(
        default_factory=list
    )

    def summary(self) -> dict[str, object]:
        return {
            "benchmarks": list(
                self.raw_data.keys()
            ),
            "rows_by_benchmark": {
                name: len(frame)
                for name, frame
                in self.raw_data.items()
            },
            "context_rows": len(
                self.context.data
            ),
            "context_features": len(
                self.context.feature_names
            ),
            "warning_count": len(
                self.warnings
            ),
        }


def _extract_close(
    frame: pd.DataFrame,
    benchmark: str,
) -> pd.Series:
    """Extract and validate Close from benchmark OHLCV."""

    if not isinstance(
        frame,
        pd.DataFrame,
    ):
        raise TypeError(
            f"{benchmark} data must be a DataFrame."
        )

    if "Close" not in frame.columns:
        raise ValueError(
            f"{benchmark} data does not contain a Close column."
        )

    close = pd.to_numeric(
        frame["Close"],
        errors="coerce",
    )

    if close.dropna().empty:
        raise ValueError(
            f"{benchmark} contains no valid Close prices."
        )

    return close.astype(float)


def _build_data_request(
    benchmark: str,
    symbol: str,
    request: MarketContextRequest,
) -> DataRequest:
    """
    Build a provider request.

    DataRequest is deliberately created here so provider-specific
    fetching remains outside the feature-engineering layer.
    """

    return DataRequest(
        symbol=symbol,
        start=request.start,
        end=request.end,
        interval=request.interval,
    )


class MarketContextDataLoader:
    """
    Load Indian benchmark data and construct market features.
    """

    def __init__(
        self,
        provider: MarketDataProvider | None = None,
        *,
        context_config: MarketContextConfig | None = None,
        symbol_map: Mapping[str, str] | None = None,
    ) -> None:
        self.provider = (
            provider
            if provider is not None
            else YahooFinanceProvider()
        )

        self.context_config = (
            context_config
            if context_config is not None
            else MarketContextConfig()
        )

        self.symbol_map = dict(
            symbol_map
            if symbol_map is not None
            else MARKET_SYMBOLS
        )

    def fetch_benchmarks(
        self,
        request: MarketContextRequest,
    ) -> dict[
        str,
        pd.DataFrame,
    ]:
        """
        Fetch all requested benchmark OHLCV datasets.

        Each benchmark is fetched independently so a provider failure
        can be identified precisely.
        """

        if not isinstance(
            request,
            MarketContextRequest,
        ):
            raise TypeError(
                "request must be a MarketContextRequest."
            )

        data: dict[
            str,
            pd.DataFrame,
        ] = {}

        for benchmark in request.benchmarks:
            if benchmark not in self.symbol_map:
                raise ValueError(
                    f"Unknown benchmark: {benchmark}"
                )

            symbol = self.symbol_map[
                benchmark
            ]

            provider_request = _build_data_request(
                benchmark,
                symbol,
                request,
            )

            frame = self.provider.fetch(
                provider_request
            )

            if not isinstance(
                frame,
                pd.DataFrame,
            ):
                raise TypeError(
                    f"Provider returned invalid data for {benchmark}."
                )

            if frame.empty:
                raise ValueError(
                    f"Provider returned empty data for {benchmark}."
                )

            data[benchmark] = frame.copy()

        return data

    def build(
        self,
        request: MarketContextRequest,
    ) -> MarketContextData:
        """
        Fetch benchmarks and build market-context features.
        """

        raw_data = self.fetch_benchmarks(
            request
        )

        context_input: dict[
            str,
            pd.Series,
        ] = {}

        for benchmark, frame in raw_data.items():
            context_input[
                benchmark
            ] = _extract_close(
                frame,
                benchmark,
            )

        context = build_market_context(
            context_input,
            config=self.context_config,
        )

        warnings = list(
            context.warnings
        )

        return MarketContextData(
            raw_data=raw_data,
            context=context,
            metadata={
                "research_only": True,
                "future_values_used": False,
                "provider": type(
                    self.provider
                ).__name__,
                "interval": request.interval,
                "benchmarks": list(
                    request.benchmarks
                ),
            },
            warnings=warnings,
        )

    def add_stock_relative_strength(
        self,
        market_data: MarketContextData,
        stock_close: pd.Series,
        *,
        benchmark: str = "NIFTY50",
        window: int | None = None,
    ) -> pd.DataFrame:
        """
        Add stock-vs-benchmark relative strength.

        The returned DataFrame is indexed only where the stock and
        benchmark timestamps overlap.
        """

        if not isinstance(
            market_data,
            MarketContextData,
        ):
            raise TypeError(
                "market_data must be MarketContextData."
            )

        if benchmark not in market_data.raw_data:
            raise ValueError(
                f"Benchmark {benchmark} was not loaded."
            )

        benchmark_close = _extract_close(
            market_data.raw_data[
                benchmark
            ],
            benchmark,
        )

        selected_window = (
            window
            if window is not None
            else self.context_config.relative_strength_window
        )

        return add_relative_strength(
            stock_close,
            benchmark_close,
            window=selected_window,
            prefix=f"{benchmark}_Relative_Strength",
        )


def load_market_context(
    request: MarketContextRequest,
    *,
    provider: MarketDataProvider | None = None,
    config: MarketContextConfig | None = None,
) -> MarketContextData:
    """Convenience function for loading market context."""

    loader = MarketContextDataLoader(
        provider=provider,
        context_config=config,
    )

    return loader.build(
        request
    )


def build_stock_market_context(
    stock_close: pd.Series,
    market_data: MarketContextData,
    *,
    benchmark: str = "NIFTY50",
    window: int | None = None,
) -> pd.DataFrame:
    """
    Convenience function for stock-vs-market relative strength.
    """

    if not isinstance(
        market_data,
        MarketContextData,
    ):
        raise TypeError(
            "market_data must be MarketContextData."
        )

    if benchmark not in market_data.raw_data:
        raise ValueError(
            f"Benchmark {benchmark} is not available."
        )

    benchmark_close = _extract_close(
        market_data.raw_data[
            benchmark
        ],
        benchmark,
    )

    selected_window = (
        window
        if window is not None
        else market_data.context.metadata.get(
            "relative_strength_window",
            20,
        )
    )

    return add_relative_strength(
        stock_close,
        benchmark_close,
        window=int(
            selected_window
        ),
        prefix=f"{benchmark}_Relative_Strength",
    )


__all__ = [
    "MarketContextRequest",
    "MarketContextData",
    "MarketContextDataLoader",
    "load_market_context",
    "build_stock_market_context",
]
