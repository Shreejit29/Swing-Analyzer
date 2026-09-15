"""
Tests for the Indian market-context feature engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.market_context import (
    DEFAULT_MARKET_SYMBOLS,
    MarketContextConfig,
    MarketContextResult,
    add_relative_strength,
    build_benchmark_features,
    build_market_context,
    market_context_feature_names,
    market_strength_score,
)


def make_index(
    n: int = 260,
    start: str = "2024-01-01",
) -> pd.DatetimeIndex:
    return pd.date_range(
        start,
        periods=n,
        freq="D",
    )


def make_close(
    n: int = 260,
    start_price: float = 100.0,
    slope: float = 0.25,
) -> pd.Series:
    index = make_index(n)

    values = (
        start_price
        + np.arange(n) * slope
        + np.sin(
            np.arange(n) / 8.0
        )
    )

    return pd.Series(
        values,
        index=index,
        name="Close",
    )


def make_ohlcv(
    n: int = 260,
) -> pd.DataFrame:
    close = make_close(
        n=n
    )

    frame = pd.DataFrame(
        index=close.index
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


# ---------------------------------------------------------------------
# Constants and configuration
# ---------------------------------------------------------------------


def test_default_market_symbols():
    assert DEFAULT_MARKET_SYMBOLS[
        "NIFTY50"
    ] == "^NSEI"

    assert DEFAULT_MARKET_SYMBOLS[
        "SENSEX"
    ] == "^BSESN"

    assert DEFAULT_MARKET_SYMBOLS[
        "NIFTYBANK"
    ] == "^NSEBANK"


def test_default_config():
    config = MarketContextConfig()

    assert (
        "NIFTY50"
        in config.benchmark_names
    )

    assert (
        "SENSEX"
        in config.benchmark_names
    )

    assert (
        "NIFTYBANK"
        in config.benchmark_names
    )

    assert config.return_windows == (
        1,
        3,
        5,
        10,
        20,
    )


def test_empty_benchmark_names_fail():
    with pytest.raises(ValueError):
        MarketContextConfig(
            benchmark_names=()
        )


def test_invalid_return_window_fails():
    with pytest.raises(ValueError):
        MarketContextConfig(
            return_windows=(1, 0, 5)
        )


def test_invalid_moving_average_window_fails():
    with pytest.raises(ValueError):
        MarketContextConfig(
            moving_average_windows=(20, -50)
        )


def test_invalid_volatility_window_fails():
    with pytest.raises(ValueError):
        MarketContextConfig(
            volatility_windows=(0,)
        )


def test_invalid_momentum_window_fails():
    with pytest.raises(ValueError):
        MarketContextConfig(
            momentum_windows=(-5,)
        )


def test_invalid_relative_strength_window_fails():
    with pytest.raises(ValueError):
        MarketContextConfig(
            relative_strength_window=0
        )


# ---------------------------------------------------------------------
# Benchmark feature generation
# ---------------------------------------------------------------------


def test_build_benchmark_features():
    close = make_close()

    result = build_benchmark_features(
        close,
        benchmark_name="NIFTY50",
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )

    assert len(result) == len(
        close
    )

    assert (
        "NIFTY50_Close"
        in result.columns
    )

    assert (
        "NIFTY50_Market_Return_1"
        in result.columns
    )

    assert (
        "NIFTY50_Market_Return_20"
        in result.columns
    )

    assert (
        "NIFTY50_Market_MA_20"
        in result.columns
    )

    assert (
        "NIFTY50_Market_MA_50"
        in result.columns
    )

    assert (
        "NIFTY50_Market_MA_200"
        in result.columns
    )


def test_benchmark_features_include_volatility():
    result = build_benchmark_features(
        make_close(),
        benchmark_name="NIFTY50",
    )

    assert (
        "NIFTY50_Market_Volatility_10"
        in result.columns
    )

    assert (
        "NIFTY50_Market_Volatility_20"
        in result.columns
    )

    assert (
        "NIFTY50_Market_Volatility_60"
        in result.columns
    )


def test_benchmark_features_include_momentum():
    result = build_benchmark_features(
        make_close(),
        benchmark_name="NIFTY50",
    )

    assert (
        "NIFTY50_Market_Momentum_5"
        in result.columns
    )

    assert (
        "NIFTY50_Market_Momentum_10"
        in result.columns
    )

    assert (
        "NIFTY50_Market_Momentum_20"
        in result.columns
    )


def test_benchmark_features_include_trend():
    result = build_benchmark_features(
        make_close(),
        benchmark_name="NIFTY50",
    )

    assert (
        "NIFTY50_Trend_Score"
        in result.columns
    )

    assert (
        "NIFTY50_Regime"
        in result.columns
    )


def test_first_return_observation_is_nan():
    result = build_benchmark_features(
        make_close(),
        benchmark_name="NIFTY50",
    )

    assert pd.isna(
        result.loc[
            result.index[0],
            "NIFTY50_Market_Return_1",
        ]
    )


def test_long_moving_average_has_warmup_period():
    result = build_benchmark_features(
        make_close(
            n=260
        ),
        benchmark_name="NIFTY50",
    )

    assert pd.isna(
        result.iloc[0][
            "NIFTY50_Market_MA_200"
        ]
    )

    assert pd.isna(
        result.iloc[198][
            "NIFTY50_Market_MA_200"
        ]
    )

    assert not pd.isna(
        result.iloc[199][
            "NIFTY50_Market_MA_200"
        ]
    )


def test_benchmark_close_is_preserved():
    close = make_close()

    result = build_benchmark_features(
        close,
        benchmark_name="NIFTY50",
    )

    pd.testing.assert_series_equal(
        result["NIFTY50_Close"],
        close.rename(
            "NIFTY50_Close"
        ),
    )


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------


def test_benchmark_requires_series():
    with pytest.raises(TypeError):
        build_benchmark_features(
            [1, 2, 3],
            benchmark_name="NIFTY50",
        )


def test_benchmark_requires_datetime_index():
    series = pd.Series(
        [100, 101, 102]
    )

    with pytest.raises(TypeError):
        build_benchmark_features(
            series,
            benchmark_name="NIFTY50",
        )


def test_duplicate_timestamps_fail():
    index = pd.DatetimeIndex(
        [
            "2024-01-01",
            "2024-01-02",
            "2024-01-02",
        ]
    )

    close = pd.Series(
        [100, 101, 102],
        index=index,
    )

    with pytest.raises(ValueError):
        build_benchmark_features(
            close,
            benchmark_name="NIFTY50",
        )


def test_unsorted_timestamps_fail():
    index = pd.DatetimeIndex(
        [
            "2024-01-03",
            "2024-01-01",
            "2024-01-02",
        ]
    )

    close = pd.Series(
        [100, 101, 102],
        index=index,
    )

    with pytest.raises(ValueError):
        build_benchmark_features(
            close,
            benchmark_name="NIFTY50",
        )


def test_nonpositive_price_fails():
    close = make_close()

    close.iloc[20] = 0

    with pytest.raises(ValueError):
        build_benchmark_features(
            close,
            benchmark_name="NIFTY50",
        )


def test_all_invalid_prices_fail():
    close = pd.Series(
        np.nan,
        index=make_index(20),
    )

    with pytest.raises(ValueError):
        build_benchmark_features(
            close,
            benchmark_name="NIFTY50",
        )


# ---------------------------------------------------------------------
# Combined market context
# ---------------------------------------------------------------------


def test_build_market_context():
    data = {
        "NIFTY50": make_close(),
        "SENSEX": make_close(
            start_price=50000,
            slope=50,
        ),
        "NIFTYBANK": make_close(
            start_price=40000,
            slope=40,
        ),
    }

    result = build_market_context(
        data
    )

    assert isinstance(
        result,
        MarketContextResult,
    )

    assert len(result.data) > 0

    assert (
        "NIFTY50_Close"
        in result.data.columns
    )

    assert (
        "SENSEX_Close"
        in result.data.columns
    )

    assert (
        "NIFTYBANK_Close"
        in result.data.columns
    )


def test_market_context_feature_names():
    result = build_market_context(
        {
            "NIFTY50": make_close(),
        },
        config=MarketContextConfig(
            benchmark_names=(
                "NIFTY50",
            )
        ),
    )

    names = market_context_feature_names(
        result
    )

    assert isinstance(
        names,
        list,
    )

    assert (
        "NIFTY50_Close"
        in names
    )


def test_market_context_summary():
    result = build_market_context(
        {
            "NIFTY50": make_close(),
        },
        config=MarketContextConfig(
            benchmark_names=(
                "NIFTY50",
            )
        ),
    )

    summary = result.summary()

    assert summary["rows"] == len(
        result.data
    )

    assert summary[
        "features"
    ] == len(
        result.feature_names
    )

    assert summary[
        "benchmarks"
    ] == ["NIFTY50"]


def test_missing_benchmark_is_warning():
    config = MarketContextConfig(
        benchmark_names=(
            "NIFTY50",
            "SENSEX",
        )
    )

    result = build_market_context(
        {
            "NIFTY50": make_close(),
        },
        config=config,
    )

    assert (
        "SENSEX"
        in result.warnings[0]
    )


def test_no_configured_benchmark_fails():
    config = MarketContextConfig(
        benchmark_names=(
            "NIFTY50",
        )
    )

    with pytest.raises(ValueError):
        build_market_context(
            {},
            config=config,
        )


def test_dataframe_benchmark_source():
    frame = make_ohlcv()

    result = build_market_context(
        {
            "NIFTY50": frame,
        },
        config=MarketContextConfig(
            benchmark_names=(
                "NIFTY50",
            )
        ),
    )

    assert (
        "NIFTY50_Close"
        in result.data.columns
    )


def test_dataframe_without_close_fails():
    frame = make_ohlcv().drop(
        columns=["Close"]
    )

    with pytest.raises(ValueError):
        build_market_context(
            {
                "NIFTY50": frame,
            },
            config=MarketContextConfig(
                benchmark_names=(
                    "NIFTY50",
                )
            ),
        )


def test_invalid_benchmark_source_type_fails():
    with pytest.raises(TypeError):
        build_market_context(
            {
                "NIFTY50": [1, 2, 3],
            },
            config=MarketContextConfig(
                benchmark_names=(
                    "NIFTY50",
                )
            ),
        )


# ---------------------------------------------------------------------
# Market strength
# ---------------------------------------------------------------------


def test_market_strength_score():
    result = build_market_context(
        {
            "NIFTY50": make_close(),
            "SENSEX": make_close(
                start_price=50000,
                slope=50,
            ),
        },
        config=MarketContextConfig(
            benchmark_names=(
                "NIFTY50",
                "SENSEX",
            )
        ),
    )

    score = market_strength_score(
        result.data
    )

    assert isinstance(
        score,
        pd.Series,
    )

    assert (
        score.index.equals(
            result.data.index
        )
    )


def test_market_strength_requires_trend_columns():
    frame = pd.DataFrame(
        {
            "Close": make_close(20)
        }
    )

    with pytest.raises(ValueError):
        market_strength_score(
            frame
        )


def test_market_strength_requires_dataframe():
    with pytest.raises(TypeError):
        market_strength_score(
            make_close()
        )


# ---------------------------------------------------------------------
# Relative strength
# ---------------------------------------------------------------------


def test_relative_strength():
    stock = make_close(
        start_price=100,
        slope=1.0,
    )

    market = make_close(
        start_price=100,
        slope=0.2,
    )

    result = add_relative_strength(
        stock,
        market,
        window=20,
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )

    assert (
        "Relative_Strength_20"
        in result.columns
    )

    assert (
        "Relative_Strength_Ratio_20"
        in result.columns
    )


def test_relative_strength_uses_overlapping_dates():
    stock = make_close(
        n=100
    )

    market = make_close(
        n=150
    )

    result = add_relative_strength(
        stock,
        market,
        window=20,
    )

    assert (
        len(result)
        == len(stock)
    )


def test_relative_strength_warmup():
    result = add_relative_strength(
        make_close(),
        make_close(),
        window=20,
    )

    assert pd.isna(
        result.iloc[0][
            "Relative_Strength_20"
        ]
    )

    assert not pd.isna(
        result.iloc[20][
            "Relative_Strength_20"
        ]
    )


def test_relative_strength_window_must_be_positive():
    with pytest.raises(ValueError):
        add_relative_strength(
            make_close(),
            make_close(),
            window=0,
        )


def test_relative_strength_requires_series():
    with pytest.raises(TypeError):
        add_relative_strength(
            [1, 2, 3],
            make_close(),
        )


def test_relative_strength_requires_overlap():
    stock = pd.Series(
        [100, 101],
        index=pd.date_range(
            "2020-01-01",
            periods=2,
        ),
    )

    market = pd.Series(
        [100, 101],
        index=pd.date_range(
            "2030-01-01",
            periods=2,
        ),
    )

    with pytest.raises(ValueError):
        add_relative_strength(
            stock,
            market,
        )


# ---------------------------------------------------------------------
# Causality / future-leakage tests
# ---------------------------------------------------------------------


def test_market_return_is_causal():
    close = make_close()

    result = build_benchmark_features(
        close,
        benchmark_name="NIFTY50",
    )

    original = result.loc[
        result.index[100],
        "NIFTY50_Market_Return_20",
    ]

    mutated = close.copy()

    mutated.iloc[101:] *= 10.0

    result_mutated = build_benchmark_features(
        mutated,
        benchmark_name="NIFTY50",
    )

    mutated_value = result_mutated.loc[
        result_mutated.index[100],
        "NIFTY50_Market_Return_20",
    ]

    assert original == pytest.approx(
        mutated_value
    )


def test_market_moving_average_is_causal():
    close = make_close()

    result = build_benchmark_features(
        close,
        benchmark_name="NIFTY50",
    )

    original = result.loc[
        result.index[100],
        "NIFTY50_Market_MA_20",
    ]

    mutated = close.copy()

    mutated.iloc[101:] *= 20.0

    result_mutated = build_benchmark_features(
        mutated,
        benchmark_name="NIFTY50",
    )

    mutated_value = result_mutated.loc[
        result_mutated.index[100],
        "NIFTY50_Market_MA_20",
    ]

    assert original == pytest.approx(
        mutated_value
    )


def test_market_volatility_is_causal():
    close = make_close()

    result = build_benchmark_features(
        close,
        benchmark_name="NIFTY50",
    )

    original = result.loc[
        result.index[100],
        "NIFTY50_Market_Volatility_20",
    ]

    mutated = close.copy()

    mutated.iloc[101:] *= 20.0

    result_mutated = build_benchmark_features(
        mutated,
        benchmark_name="NIFTY50",
    )

    mutated_value = result_mutated.loc[
        result_mutated.index[100],
        "NIFTY50_Market_Volatility_20",
    ]

    assert original == pytest.approx(
        mutated_value
    )


def test_relative_strength_is_causal():
    stock = make_close(
        slope=0.5
    )

    market = make_close(
        slope=0.2
    )

    result = add_relative_strength(
        stock,
        market,
        window=20,
    )

    original = result.loc[
        result.index[100],
        "Relative_Strength_20",
    ]

    stock_mutated = stock.copy()
    market_mutated = market.copy()

    stock_mutated.iloc[101:] *= 50
    market_mutated.iloc[101:] *= 0.1

    mutated_result = add_relative_strength(
        stock_mutated,
        market_mutated,
        window=20,
    )

    mutated = mutated_result.loc[
        mutated_result.index[100],
        "Relative_Strength_20",
    ]

    assert original == pytest.approx(
        mutated
    )


# ---------------------------------------------------------------------
# Input immutability
# ---------------------------------------------------------------------


def test_benchmark_input_is_not_modified():
    close = make_close()

    original = close.copy(
        deep=True
    )

    build_benchmark_features(
        close,
        benchmark_name="NIFTY50",
    )

    pd.testing.assert_series_equal(
        close,
        original,
    )


def test_market_context_inputs_are_not_modified():
    nifty = make_close()
    sensex = make_close(
        start_price=50000
    )

    nifty_original = nifty.copy(
        deep=True
    )

    sensex_original = sensex.copy(
        deep=True
    )

    build_market_context(
        {
            "NIFTY50": nifty,
            "SENSEX": sensex,
        },
        config=MarketContextConfig(
            benchmark_names=(
                "NIFTY50",
                "SENSEX",
            )
        ),
    )

    pd.testing.assert_series_equal(
        nifty,
        nifty_original,
    )

    pd.testing.assert_series_equal(
        sensex,
        sensex_original,
    )


# ---------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------


def test_benchmark_features_are_deterministic():
    close = make_close()

    first = build_benchmark_features(
        close,
        benchmark_name="NIFTY50",
    )

    second = build_benchmark_features(
        close,
        benchmark_name="NIFTY50",
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )


def test_market_context_is_deterministic():
    data = {
        "NIFTY50": make_close(),
        "SENSEX": make_close(
            start_price=50000
        ),
    }

    config = MarketContextConfig(
        benchmark_names=(
            "NIFTY50",
            "SENSEX",
        )
    )

    first = build_market_context(
        data,
        config=config,
    )

    second = build_market_context(
        data,
        config=config,
    )

    pd.testing.assert_frame_equal(
        first.data,
        second.data,
    )

    assert (
        first.feature_names
        == second.feature_names
    )


def test_relative_strength_is_deterministic():
    stock = make_close(
        slope=0.5
    )

    market = make_close(
        slope=0.2
    )

    first = add_relative_strength(
        stock,
        market,
    )

    second = add_relative_strength(
        stock,
        market,
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )


# ---------------------------------------------------------------------
# Research safety metadata
# ---------------------------------------------------------------------


def test_market_context_metadata_is_causal():
    result = build_market_context(
        {
            "NIFTY50": make_close(),
        },
        config=MarketContextConfig(
            benchmark_names=(
                "NIFTY50",
            )
        ),
    )

    assert (
        result.metadata[
            "future_values_used"
        ]
        is False
    )

    assert (
        result.metadata[
            "feature_generation"
        ]
        == "causal"
    )


def test_market_context_is_research_only():
    result = build_market_context(
        {
            "NIFTY50": make_close(),
        },
        config=MarketContextConfig(
            benchmark_names=(
                "NIFTY50",
            )
        ),
    )

    assert (
        result.metadata[
            "research_only"
        ]
        is True
    )
