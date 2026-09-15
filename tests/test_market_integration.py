"""
Tests for stock + market-context integration.

These tests use synthetic data only.
No live market-data requests are performed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.market_context import (
    MarketContextConfig,
    MarketContextResult,
)
from src.research.market_data_context import (
    MarketContextData,
)
from src.research.market_integration import (
    MarketIntegrationConfig,
    MarketIntegrationResult,
    add_market_context_to_stock,
    integrate_market_context,
    market_integration_summary,
)


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

    return pd.DataFrame(
        {
            "Open": close * 0.998,
            "High": close * 1.010,
            "Low": close * 0.990,
            "Close": close,
            "Volume": (
                1_000_000
                + np.arange(n) * 1_000
            ),
        },
        index=index,
    )


def make_market_context(
    n: int = 260,
) -> MarketContextData:
    index = pd.date_range(
        "2024-01-01",
        periods=n,
        freq="D",
    )

    base = np.arange(n, dtype=float)

    context = pd.DataFrame(
        {
            "NIFTY50_Close": (
                100.0 + base
            ),
            "NIFTY50_Market_Return_1": (
                0.001 + base * 0.0
            ),
            "NIFTY50_Market_Return_5": (
                0.005 + base * 0.0
            ),
            "NIFTY50_Market_Return_20": (
                0.020 + base * 0.0
            ),
            "NIFTY50_Market_MA_20": (
                105.0 + base * 0.1
            ),
            "NIFTY50_Market_MA_50": (
                103.0 + base * 0.1
            ),
            "NIFTY50_Market_MA_200": (
                100.0 + base * 0.1
            ),
            "NIFTY50_Market_Volatility_10": (
                0.01 + base * 0.0
            ),
            "NIFTY50_Market_Volatility_20": (
                0.012 + base * 0.0
            ),
            "NIFTY50_Market_Volatility_60": (
                0.015 + base * 0.0
            ),
            "NIFTY50_Market_Momentum_5": (
                0.005 + base * 0.0
            ),
            "NIFTY50_Market_Momentum_10": (
                0.010 + base * 0.0
            ),
            "NIFTY50_Market_Momentum_20": (
                0.020 + base * 0.0
            ),
            "NIFTY50_Trend_Score": (
                0.75 + base * 0.0
            ),
            "NIFTY50_Regime": [
                "BULL"
                for _ in range(n)
            ],
        },
        index=index,
    )

    raw = {
        "NIFTY50": make_ohlcv(
            n=n,
            start_price=100.0,
            slope=0.50,
        )
    }

    return MarketContextData(
        raw_data=raw,
        context=MarketContextResult(
            data=context,
            feature_names=list(
                context.columns
            ),
            warnings=[],
            metadata={
                "relative_strength_window": 20
            },
        ),
        metadata={
            "research_only": True,
            "future_values_used": False,
        },
        warnings=[],
    )


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


def test_market_integration_config_defaults():
    config = MarketIntegrationConfig()

    assert (
        config.benchmark
        == "NIFTY50"
    )

    assert (
        config.relative_strength_window
        == 20
    )

    assert (
        config.allow_market_missing
        is False
    )


def test_market_integration_config_validation():
    with pytest.raises(ValueError):
        MarketIntegrationConfig(
            benchmark=""
        )

    with pytest.raises(ValueError):
        MarketIntegrationConfig(
            relative_strength_window=1
        )

    with pytest.raises(ValueError):
        MarketIntegrationConfig(
            max_market_missing_fraction=1.5
        )


# ---------------------------------------------------------------------
# Basic integration
# ---------------------------------------------------------------------


def test_integrate_market_context():
    stock = make_ohlcv()
    market = make_market_context()

    result = integrate_market_context(
        stock,
        market,
    )

    assert isinstance(
        result,
        MarketIntegrationResult,
    )

    assert len(result.data) == len(
        stock
    )

    assert result.data.index.equals(
        stock.index
    )


def test_original_stock_columns_are_preserved():
    stock = make_ohlcv()
    market = make_market_context()

    result = integrate_market_context(
        stock,
        market,
    )

    for column in stock.columns:
        assert column in result.data.columns

        pd.testing.assert_series_equal(
            result.data[column],
            stock[column],
        )


def test_market_columns_are_added():
    stock = make_ohlcv()
    market = make_market_context()

    result = integrate_market_context(
        stock,
        market,
    )

    assert (
        "NIFTY50_Close"
        in result.data.columns
    )

    assert (
        "NIFTY50_Trend_Score"
        in result.data.columns
    )


def test_relative_strength_columns_are_added():
    stock = make_ohlcv()
    market = make_market_context()

    result = integrate_market_context(
        stock,
        market,
    )

    assert any(
        column.startswith(
            "NIFTY50_Relative_Strength"
        )
        for column in result.data.columns
    )


def test_feature_columns_are_reported():
    stock = make_ohlcv()
    market = make_market_context()

    result = integrate_market_context(
        stock,
        market,
    )

    assert (
        len(result.feature_columns)
        > 0
    )

    assert all(
        column in result.data.columns
        for column in result.feature_columns
    )


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------


def test_stock_data_must_be_dataframe():
    market = make_market_context()

    with pytest.raises(TypeError):
        integrate_market_context(
            object(),
            market,
        )


def test_empty_stock_data_fails():
    market = make_market_context()

    empty = pd.DataFrame(
        columns=[
            "Open",
            "High",
            "Low",
            "Close",
        ]
    )

    empty.index = pd.DatetimeIndex(
        []
    )

    with pytest.raises(ValueError):
        integrate_market_context(
            empty,
            market,
        )


def test_stock_data_requires_datetime_index():
    stock = make_ohlcv()

    stock.index = range(
        len(stock)
    )

    market = make_market_context()

    with pytest.raises(TypeError):
        integrate_market_context(
            stock,
            market,
        )


def test_duplicate_stock_timestamps_fail():
    stock = make_ohlcv()

    duplicate = stock.iloc[
        :2
    ].copy()

    duplicate.index = [
        stock.index[0],
        stock.index[0],
    ]

    bad_stock = pd.concat(
        [
            stock.iloc[2:],
            duplicate,
        ]
    )

    bad_stock = bad_stock.sort_index()

    market = make_market_context()

    with pytest.raises(ValueError):
        integrate_market_context(
            bad_stock,
            market,
        )


def test_missing_stock_close_fails():
    stock = make_ohlcv().drop(
        columns=["Close"]
    )

    market = make_market_context()

    with pytest.raises(ValueError):
        integrate_market_context(
            stock,
            market,
        )


def test_invalid_market_context_type_fails():
    stock = make_ohlcv()

    with pytest.raises(TypeError):
        integrate_market_context(
            stock,
            object(),
        )


def test_missing_benchmark_fails():
    stock = make_ohlcv()
    market = make_market_context()

    config = MarketIntegrationConfig(
        benchmark="SENSEX"
    )

    with pytest.raises(ValueError):
        integrate_market_context(
            stock,
            market,
            config=config,
        )


# ---------------------------------------------------------------------
# Timestamp alignment
# ---------------------------------------------------------------------


def test_market_context_is_aligned_backward():
    stock = make_ohlcv(
        n=10
    )

    market_frame = make_ohlcv(
        n=10,
        start_price=100,
        slope=1,
    )

    market_frame = market_frame.iloc[
        [0, 2, 4, 6, 8]
    ]

    context = pd.DataFrame(
        {
            "NIFTY50_Close": (
                market_frame["Close"]
            )
        },
        index=market_frame.index,
    )

    market = MarketContextData(
        raw_data={
            "NIFTY50": market_frame
        },
        context=MarketContextResult(
            data=context,
            feature_names=[
                "NIFTY50_Close"
            ],
            warnings=[],
            metadata={
                "relative_strength_window": 20
            },
        ),
        metadata={
            "research_only": True
        },
        warnings=[],
    )

    result = integrate_market_context(
        stock,
        market,
        config=MarketIntegrationConfig(
            benchmark="NIFTY50",
            allow_market_missing=True,
        ),
    )

    # The first stock timestamp has an exact market
    # observation and therefore receives that value.
    assert (
        result.data.iloc[0][
            "NIFTY50_Close"
        ]
        == context.iloc[0][
            "NIFTY50_Close"
        ]
    )

    # A stock timestamp between market observations
    # receives the latest observation from the past,
    # not the next future observation.
    assert (
        result.data.iloc[3][
            "NIFTY50_Close"
        ]
        == context.iloc[1][
            "NIFTY50_Close"
        ]
    )


def test_no_forward_fill_from_future_market_data():
    stock = make_ohlcv(
        n=10
    )

    market_frame = make_ohlcv(
        n=2,
        start_price=100,
        slope=10,
    )

    context = pd.DataFrame(
        {
            "NIFTY50_Close": [
                100.0,
                110.0,
            ]
        },
        index=[
            stock.index[0],
            stock.index[5],
        ],
    )

    market = MarketContextData(
        raw_data={
            "NIFTY50": market_frame
        },
        context=MarketContextResult(
            data=context,
            feature_names=[
                "NIFTY50_Close"
            ],
            warnings=[],
            metadata={
                "relative_strength_window": 20
            },
        ),
        metadata={
            "research_only": True
        },
        warnings=[],
    )

    result = integrate_market_context(
        stock,
        market,
        config=MarketIntegrationConfig(
            benchmark="NIFTY50",
            allow_market_missing=True,
        ),
    )

    # Timestamp before the first market observation
    # must remain missing.
    assert pd.isna(
        result.data.iloc[0][
            "NIFTY50_Close"
        ]
    ) is False

    # After timestamp 5, the value 110 can be used.
    assert (
        result.data.iloc[6][
            "NIFTY50_Close"
        ]
        == 110.0
    )


# ---------------------------------------------------------------------
# Future mutation protection
# ---------------------------------------------------------------------


def test_future_market_mutation_does_not_change_past_features():
    stock = make_ohlcv(
        n=260
    )

    original_market = make_market_context(
        n=260
    )

    original = integrate_market_context(
        stock,
        original_market,
        config=MarketIntegrationConfig(
            benchmark="NIFTY50",
            allow_market_missing=True,
        ),
    )

    mutated_context = (
        original_market.context.data.copy(
            deep=True
        )
    )

    future_start = 180

    for column in [
        "NIFTY50_Close",
        "NIFTY50_Market_Return_20",
        "NIFTY50_Market_MA_20",
        "NIFTY50_Trend_Score",
    ]:
        if column in mutated_context.columns:
            mutated_context.loc[
                mutated_context.index[
                    future_start:
                ],
                column,
            ] *= 100.0

    mutated_market = MarketContextData(
        raw_data={
            "NIFTY50": (
                original_market.raw_data[
                    "NIFTY50"
                ].copy(deep=True)
            )
        },
        context=MarketContextResult(
            data=mutated_context,
            feature_names=list(
                mutated_context.columns
            ),
            warnings=[],
            metadata={
                "relative_strength_window": 20
            },
        ),
        metadata={
            "research_only": True
        },
        warnings=[],
    )

    mutated = integrate_market_context(
        stock,
        mutated_market,
        config=MarketIntegrationConfig(
            benchmark="NIFTY50",
            allow_market_missing=True,
        ),
    )

    timestamp = stock.index[
        100
    ]

    for column in [
        "NIFTY50_Close",
        "NIFTY50_Market_Return_20",
        "NIFTY50_Market_MA_20",
        "NIFTY50_Trend_Score",
    ]:
        if column in original.data.columns:
            assert original.data.loc[
                timestamp,
                column,
            ] == pytest.approx(
                mutated.data.loc[
                    timestamp,
                    column,
                ]
            )


# ---------------------------------------------------------------------
# Missingness handling
# ---------------------------------------------------------------------


def test_market_missingness_is_reported():
    stock = make_ohlcv(
        n=50
    )

    context_frame = pd.DataFrame(
        {
            "NIFTY50_Close": [
                100.0
            ]
        },
        index=[
            stock.index[-1]
        ],
    )

    raw = make_ohlcv(
        n=50
    )

    market = MarketContextData(
        raw_data={
            "NIFTY50": raw
        },
        context=MarketContextResult(
            data=context_frame,
            feature_names=[
                "NIFTY50_Close"
            ],
            warnings=[],
            metadata={
                "relative_strength_window": 20
            },
        ),
        metadata={
            "research_only": True
        },
        warnings=[],
    )

    result = integrate_market_context(
        stock,
        market,
        config=MarketIntegrationConfig(
            benchmark="NIFTY50",
            allow_market_missing=True,
        ),
    )

    assert (
        result.market_missing_fraction
        >= 0.0
    )

    assert len(
        result.warnings
    ) > 0


def test_excessive_missingness_fails_when_not_allowed():
    stock = make_ohlcv(
        n=50
    )

    context_frame = pd.DataFrame(
        {
            "NIFTY50_Close": [
                100.0
            ]
        },
        index=[
            stock.index[-1]
        ],
    )

    raw = make_ohlcv(
        n=50
    )

    market = MarketContextData(
        raw_data={
            "NIFTY50": raw
        },
        context=MarketContextResult(
            data=context_frame,
            feature_names=[
                "NIFTY50_Close"
            ],
            warnings=[],
            metadata={
                "relative_strength_window": 20
            },
        ),
        metadata={
            "research_only": True
        },
        warnings=[],
    )

    with pytest.raises(ValueError):
        integrate_market_context(
            stock,
            market,
            config=MarketIntegrationConfig(
                benchmark="NIFTY50",
                allow_market_missing=False,
                max_market_missing_fraction=0.10,
            ),
        )


# ---------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------


def test_stock_input_is_not_modified():
    stock = make_ohlcv()
    original = stock.copy(
        deep=True
    )

    market = make_market_context()

    integrate_market_context(
        stock,
        market,
    )

    pd.testing.assert_frame_equal(
        stock,
        original,
    )


def test_market_context_input_is_not_modified():
    stock = make_ohlcv()
    market = make_market_context()

    original_context = (
        market.context.data.copy(
            deep=True
        )
    )

    original_raw = {
        name: frame.copy(
            deep=True
        )
        for name, frame
        in market.raw_data.items()
    }

    integrate_market_context(
        stock,
        market,
    )

    pd.testing.assert_frame_equal(
        market.context.data,
        original_context,
    )

    for name in original_raw:
        pd.testing.assert_frame_equal(
            market.raw_data[name],
            original_raw[name],
        )


# ---------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------


def test_convenience_function():
    stock = make_ohlcv()
    market = make_market_context()

    result = add_market_context_to_stock(
        stock,
        market,
        benchmark="NIFTY50",
        relative_strength_window=20,
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )

    assert (
        "NIFTY50_Close"
        in result.columns
    )


def test_convenience_function_preserves_index():
    stock = make_ohlcv()
    market = make_market_context()

    result = add_market_context_to_stock(
        stock,
        market,
    )

    assert result.index.equals(
        stock.index
    )


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------


def test_market_integration_summary():
    stock = make_ohlcv()
    market = make_market_context()

    result = integrate_market_context(
        stock,
        market,
    )

    summary = market_integration_summary(
        result
    )

    assert (
        summary["rows"]
        == len(stock)
    )

    assert (
        summary["market_columns"]
        > 0
    )

    assert (
        summary["relative_strength_columns"]
        > 0
    )


def test_summary_requires_correct_type():
    with pytest.raises(TypeError):
        market_integration_summary(
            object()
        )


# ---------------------------------------------------------------------
# Research-only boundary
# ---------------------------------------------------------------------


def test_integration_is_research_only():
    stock = make_ohlcv()
    market = make_market_context()

    result = integrate_market_context(
        stock,
        market,
    )

    assert (
        result.metadata[
            "research_only"
        ]
        is True
    )

    assert (
        result.metadata[
            "production_approved"
        ]
        is False
    )

    assert (
        result.metadata[
            "future_values_used"
        ]
        is False
    )

    assert (
        result.metadata[
            "forward_fill_used"
        ]
        is False
    )


def test_integration_does_not_generate_trade_signal():
    stock = make_ohlcv()
    market = make_market_context()

    result = integrate_market_context(
        stock,
        market,
    )

    assert "BUY" not in result.data.columns
    assert "SELL" not in result.data.columns
    assert "Signal" not in result.data.columns
