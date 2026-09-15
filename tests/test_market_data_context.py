"""
Tests for the market-data context loader.

No live market-data network calls are used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.market_context import (
    MarketContextConfig,
)
from src.research.market_data_context import (
    MarketContextData,
    MarketContextDataLoader,
    MarketContextRequest,
    build_stock_market_context,
)


class FakeProvider:
    """Deterministic in-memory market-data provider."""

    def __init__(
        self,
        data: dict[str, pd.DataFrame],
    ) -> None:
        self.data = data
        self.requests = []

    def fetch(self, request):
        self.requests.append(request)

        if request.symbol == "^NSEI":
            return self.data["NIFTY50"].copy()

        if request.symbol == "^BSESN":
            return self.data["SENSEX"].copy()

        if request.symbol == "^NSEBANK":
            return self.data["NIFTYBANK"].copy()

        raise ValueError(
            f"Unknown fake symbol: {request.symbol}"
        )


class InvalidProvider:
    def fetch(self, request):
        return "invalid"


class EmptyProvider:
    def fetch(self, request):
        return pd.DataFrame()


def make_ohlcv(
    n: int = 260,
    start_price: float = 100.0,
    slope: float = 0.25,
) -> pd.DataFrame:
    index = pd.date_range(
        "2024-01-01",
        periods=n,
        freq="D",
    )

    close = (
        start_price
        + np.arange(n) * slope
        + np.sin(
            np.arange(n) / 8.0
        )
    )

    frame = pd.DataFrame(
        index=index
    )

    frame["Open"] = (
        close * 0.998
    )

    frame["High"] = (
        close * 1.01
    )

    frame["Low"] = (
        close * 0.99
    )

    frame["Close"] = close

    frame["Volume"] = (
        1_000_000
        + np.arange(n) * 1000
    )

    return frame


def make_provider() -> FakeProvider:
    return FakeProvider(
        {
            "NIFTY50": make_ohlcv(
                start_price=100
            ),
            "SENSEX": make_ohlcv(
                start_price=50_000,
                slope=50,
            ),
            "NIFTYBANK": make_ohlcv(
                start_price=40_000,
                slope=40,
            ),
        }
    )


def make_request(
    benchmarks=(
        "NIFTY50",
        "SENSEX",
        "NIFTYBANK",
    ),
) -> MarketContextRequest:
    return MarketContextRequest(
        start="2024-01-01",
        end="2025-12-31",
        benchmarks=benchmarks,
        interval="1d",
    )


# ---------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------


def test_market_context_request():
    request = make_request()

    assert request.start == "2024-01-01"
    assert request.end == "2025-12-31"

    assert request.benchmarks == (
        "NIFTY50",
        "SENSEX",
        "NIFTYBANK",
    )

    assert request.interval == "1d"


def test_empty_benchmarks_fail():
    with pytest.raises(ValueError):
        MarketContextRequest(
            start="2024-01-01",
            end="2025-01-01",
            benchmarks=(),
        )


def test_blank_benchmark_fails():
    with pytest.raises(ValueError):
        MarketContextRequest(
            start="2024-01-01",
            end="2025-01-01",
            benchmarks=(
                "NIFTY50",
                "",
            ),
        )


def test_duplicate_benchmarks_fail():
    with pytest.raises(ValueError):
        MarketContextRequest(
            start="2024-01-01",
            end="2025-01-01",
            benchmarks=(
                "NIFTY50",
                "NIFTY50",
            ),
        )


def test_blank_interval_fails():
    with pytest.raises(ValueError):
        MarketContextRequest(
            start="2024-01-01",
            end="2025-01-01",
            interval="",
        )


# ---------------------------------------------------------------------
# Loader construction
# ---------------------------------------------------------------------


def test_loader_construction():
    provider = make_provider()

    loader = MarketContextDataLoader(
        provider=provider
    )

    assert loader.provider is provider

    assert isinstance(
        loader.context_config,
        MarketContextConfig,
    )


def test_custom_context_config():
    provider = make_provider()

    config = MarketContextConfig(
        benchmark_names=(
            "NIFTY50",
        )
    )

    loader = MarketContextDataLoader(
        provider=provider,
        context_config=config,
    )

    assert (
        loader.context_config
        is config
    )


# ---------------------------------------------------------------------
# Benchmark fetching
# ---------------------------------------------------------------------


def test_fetch_all_benchmarks():
    provider = make_provider()

    loader = MarketContextDataLoader(
        provider=provider
    )

    data = loader.fetch_benchmarks(
        make_request()
    )

    assert set(
        data.keys()
    ) == {
        "NIFTY50",
        "SENSEX",
        "NIFTYBANK",
    }

    assert all(
        isinstance(
            frame,
            pd.DataFrame,
        )
        for frame in data.values()
    )


def test_provider_requests_are_created_for_each_benchmark():
    provider = make_provider()

    loader = MarketContextDataLoader(
        provider=provider
    )

    loader.fetch_benchmarks(
        make_request()
    )

    assert len(
        provider.requests
    ) == 3


def test_requested_symbols_are_correct():
    provider = make_provider()

    loader = MarketContextDataLoader(
        provider=provider
    )

    loader.fetch_benchmarks(
        make_request()
    )

    symbols = {
        request.symbol
        for request in provider.requests
    }

    assert symbols == {
        "^NSEI",
        "^BSESENSEX"
        if False
        else "^BSESENSEX",
        "^NSEBANK",
    } or symbols == {
        "^NSEI",
        "^BSESENSEX",
        "^NSEBANK",
    }


def test_subset_of_benchmarks():
    provider = make_provider()

    loader = MarketContextDataLoader(
        provider=provider
    )

    request = make_request(
        benchmarks=("NIFTY50",)
    )

    data = loader.fetch_benchmarks(
        request
    )

    assert list(
        data.keys()
    ) == ["NIFTY50"]

    assert len(
        provider.requests
    ) == 1


def test_unknown_benchmark_fails():
    provider = make_provider()

    loader = MarketContextDataLoader(
        provider=provider
    )

    request = make_request(
        benchmarks=("UNKNOWN",)
    )

    with pytest.raises(ValueError):
        loader.fetch_benchmarks(
            request
        )


def test_wrong_request_type_fails():
    provider = make_provider()

    loader = MarketContextDataLoader(
        provider=provider
    )

    with pytest.raises(TypeError):
        loader.fetch_benchmarks(
            object()
        )


# ---------------------------------------------------------------------
# Build market context
# ---------------------------------------------------------------------


def test_build_market_context_data():
    provider = make_provider()

    loader = MarketContextDataLoader(
        provider=provider
    )

    result = loader.build(
        make_request()
    )

    assert isinstance(
        result,
        MarketContextData,
    )

    assert isinstance(
        result.context,
        object,
    )

    assert set(
        result.raw_data.keys()
    ) == {
        "NIFTY50",
        "SENSEX",
        "NIFTYBANK",
    }


def test_build_contains_market_features():
    provider = make_provider()

    loader = MarketContextDataLoader(
        provider=provider
    )

    result = loader.build(
        make_request()
    )

    columns = result.context.data.columns

    assert (
        "NIFTY50_Close"
        in columns
    )

    assert (
        "NIFTY50_Trend_Score"
        in columns
    )

    assert (
        "SENSEX_Close"
        in columns
    )

    assert (
        "NIFTYBANK_Close"
        in columns
    )


def test_build_contains_metadata():
    provider = make_provider()

    loader = MarketContextDataLoader(
        provider=provider
    )

    result = loader.build(
        make_request()
    )

    assert (
        result.metadata[
            "research_only"
        ]
        is True
    )

    assert (
        result.metadata[
            "future_values_used"
        ]
        is False
    )

    assert (
        result.metadata[
            "provider"
        ]
        == "FakeProvider"
    )


def test_build_summary():
    provider = make_provider()

    loader = MarketContextDataLoader(
        provider=provider
    )

    result = loader.build(
        make_request()
    )

    summary = result.summary()

    assert set(
        summary["benchmarks"]
    ) == {
        "NIFTY50",
        "SENSEX",
        "NIFTYBANK",
    }

    assert (
        summary["context_rows"]
        > 0
    )

    assert (
        summary["context_features"]
        > 0
    )


# ---------------------------------------------------------------------
# Provider failures
# ---------------------------------------------------------------------


def test_invalid_provider_result_fails():
    loader = MarketContextDataLoader(
        provider=InvalidProvider()
    )

    with pytest.raises(TypeError):
        loader.fetch_benchmarks(
            make_request(
                benchmarks=("NIFTY50",)
            )
        )


def test_empty_provider_result_fails():
    loader = MarketContextDataLoader(
        provider=EmptyProvider()
    )

    with pytest.raises(ValueError):
        loader.fetch_benchmarks(
            make_request(
                benchmarks=("NIFTY50",)
            )
        )


def test_missing_close_column_fails():
    frame = make_ohlcv().drop(
        columns=["Close"]
    )

    provider = FakeProvider(
        {
            "NIFTY50": frame,
        }
    )

    loader = MarketContextDataLoader(
        provider=provider,
        context_config=MarketContextConfig(
            benchmark_names=(
                "NIFTY50",
            )
        ),
    )

    with pytest.raises(ValueError):
        loader.build(
            make_request(
                benchmarks=("NIFTY50",)
            )
        )


# ---------------------------------------------------------------------
# Close extraction
# ---------------------------------------------------------------------


def test_close_is_numeric():
    provider = make_provider()

    loader = MarketContextDataLoader(
        provider=provider
    )

    result = loader.build(
        make_request()
    )

    close = result.raw_data[
        "NIFTY50"
    ]["Close"]

    assert pd.api.types.is_numeric_dtype(
        close
    )


# ---------------------------------------------------------------------
# Relative strength
# ---------------------------------------------------------------------


def test_add_stock_relative_strength():
    provider = make_provider()

    loader = MarketContextDataLoader(
        provider=provider
    )

    market_data = loader.build(
        make_request()
    )

    stock = make_ohlcv(
        start_price=200,
        slope=1.0,
    )["Close"]

    result = loader.add_stock_relative_strength(
        market_data,
        stock,
        benchmark="NIFTY50",
        window=20,
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )

    assert (
        "NIFTY50_Relative_Strength_20"
        in result.columns
    )

    assert (
        "NIFTY50_Relative_Strength_Ratio_20"
        in result.columns
    )


def test_relative_strength_requires_loaded_benchmark():
    provider = make_provider()

    loader = MarketContextDataLoader(
        provider=provider,
        context_config=MarketContextConfig(
            benchmark_names=(
                "NIFTY50",
            )
        ),
    )

    market_data = loader.build(
        make_request(
            benchmarks=("NIFTY50",)
        )
    )

    stock = make_ohlcv()["Close"]

    with pytest.raises(ValueError):
        loader.add_stock_relative_strength(
            market_data,
            stock,
            benchmark="SENSEX",
        )


def test_relative_strength_custom_window():
    provider = make_provider()

    loader = MarketContextDataLoader(
        provider=provider
    )

    market_data = loader.build(
        make_request()
    )

    stock = make_ohlcv()["Close"]

    result = loader.add_stock_relative_strength(
        market_data,
        stock,
        benchmark="NIFTY50",
        window=10,
    )

    assert (
        "NIFTY50_Relative_Strength_10"
        in result.columns
    )


def test_relative_strength_input_is_not_modified():
    provider = make_provider()

    loader = MarketContextDataLoader(
        provider=provider
    )

    market_data = loader.build(
        make_request()
    )

    stock = make_ohlcv()["Close"]

    original = stock.copy(
        deep=True
    )

    loader.add_stock_relative_strength(
        market_data,
        stock,
        benchmark="NIFTY50",
    )

    pd.testing.assert_series_equal(
        stock,
        original,
    )


# ---------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------


def test_build_stock_market_context():
    provider = make_provider()

    loader = MarketContextDataLoader(
        provider=provider
    )

    market_data = loader.build(
        make_request()
    )

    stock = make_ohlcv(
        start_price=200,
        slope=0.8,
    )["Close"]

    result = build_stock_market_context(
        stock,
        market_data,
        benchmark="NIFTY50",
        window=20,
    )

    assert (
        "NIFTY50_Relative_Strength_20"
        in result.columns
    )


def test_convenience_function_requires_market_context():
    with pytest.raises(TypeError):
        build_stock_market_context(
            make_ohlcv()["Close"],
            object(),
        )


def test_convenience_function_requires_benchmark():
    provider = make_provider()

    loader = MarketContextDataLoader(
        provider=provider,
        context_config=MarketContextConfig(
            benchmark_names=(
                "NIFTY50",
            )
        ),
    )

    market_data = loader.build(
        make_request(
            benchmarks=("NIFTY50",)
        )
    )

    with pytest.raises(ValueError):
        build_stock_market_context(
            make_ohlcv()["Close"],
            market_data,
            benchmark="SENSEX",
        )


# ---------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------


def test_loader_output_is_deterministic():
    provider_1 = make_provider()
    provider_2 = make_provider()

    loader_1 = MarketContextDataLoader(
        provider=provider_1
    )

    loader_2 = MarketContextDataLoader(
        provider=provider_2
    )

    result_1 = loader_1.build(
        make_request()
    )

    result_2 = loader_2.build(
        make_request()
    )

    pd.testing.assert_frame_equal(
        result_1.context.data,
        result_2.context.data,
    )


# ---------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------


def test_loader_market_features_do_not_use_future_values():
    provider = make_provider()

    loader = MarketContextDataLoader(
        provider=provider
    )

    original = loader.build(
        make_request(
            benchmarks=("NIFTY50",)
        )
    )

    mutated_frame = make_ohlcv()

    mutated_frame.loc[
        mutated_frame.index[150:],
        "Close",
    ] *= 100.0

    mutated_provider = FakeProvider(
        {
            "NIFTY50": mutated_frame,
        }
    )

    mutated_loader = MarketContextDataLoader(
        provider=mutated_provider,
        context_config=MarketContextConfig(
            benchmark_names=(
                "NIFTY50",
            )
        ),
    )

    mutated = mutated_loader.build(
        make_request(
            benchmarks=("NIFTY50",)
        )
    )

    timestamp = original.context.data.index[
        100
    ]

    columns = [
        "NIFTY50_Market_Return_20",
        "NIFTY50_Market_MA_20",
        "NIFTY50_Market_Volatility_20",
        "NIFTY50_Trend_Score",
    ]

    for column in columns:
        assert original.context.data.loc[
            timestamp,
            column,
        ] == pytest.approx(
            mutated.context.data.loc[
                timestamp,
                column,
            ]
        )


# ---------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------


def test_build_does_not_modify_provider_data():
    source = make_ohlcv()

    provider = FakeProvider(
        {
            "NIFTY50": source,
        }
    )

    original = source.copy(
        deep=True
    )

    loader = MarketContextDataLoader(
        provider=provider,
        context_config=MarketContextConfig(
            benchmark_names=(
                "NIFTY50",
            )
        ),
    )

    loader.build(
        make_request(
            benchmarks=("NIFTY50",)
        )
    )

    pd.testing.assert_frame_equal(
        source,
        original,
    )


# ---------------------------------------------------------------------
# Research boundary
# ---------------------------------------------------------------------


def test_loader_does_not_approve_models():
    provider = make_provider()

    loader = MarketContextDataLoader(
        provider=provider
    )

    result = loader.build(
        make_request()
    )

    assert (
        result.metadata.get(
            "production_approved"
        )
        is None
    )


def test_loader_is_research_only():
    provider = make_provider()

    loader = MarketContextDataLoader(
        provider=provider
    )

    result = loader.build(
        make_request()
    )

    assert (
        result.metadata[
            "research_only"
        ]
        is True
    )
