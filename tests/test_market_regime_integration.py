"""
Tests for market and regime integration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.market_regime_integration import (
    DEFAULT_MARKET_REGIMES,
    DEFAULT_VOLATILITY_REGIMES,
    MarketRegimeIntegration,
    MarketRegimeIntegrationConfig,
    MarketRegimeIntegrationResult,
    integrate_market_regime,
    market_regime_integration_summary,
)


def make_stock_data(
    rows: int = 500,
) -> pd.DataFrame:
    index = pd.date_range(
        "2020-01-01",
        periods=rows,
        freq="D",
    )

    t = np.arange(rows)

    close = (
        100
        + 0.15 * t
        + 2.0 * np.sin(t / 15)
    )

    open_price = (
        close
        + 0.25 * np.sin(t / 7)
    )

    high = (
        np.maximum(
            open_price,
            close,
        )
        + 1.0
    )

    low = (
        np.minimum(
            open_price,
            close,
        )
        - 1.0
    )

    volume = (
        100_000
        + 5_000 * np.sin(t / 20)
        + 500 * (t % 20)
    )

    return pd.DataFrame(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )


def make_market_series(
    rows: int = 500,
    *,
    trend: float = 0.10,
    start: float = 100.0,
) -> pd.Series:
    index = pd.date_range(
        "2020-01-01",
        periods=rows,
        freq="D",
    )

    t = np.arange(rows)

    values = (
        start
        + trend * t
        + 1.5 * np.sin(t / 17)
    )

    return pd.Series(
        values,
        index=index,
        name="Close",
    )


def make_market_data(
    rows: int = 500,
) -> dict[str, pd.Series]:
    return {
        "NIFTY50": make_market_series(
            rows,
            trend=0.12,
            start=100.0,
        ),
        "SENSEX": make_market_series(
            rows,
            trend=0.14,
            start=200.0,
        ),
        "NIFTYBANK": make_market_series(
            rows,
            trend=0.08,
            start=150.0,
        ),
    }


def make_sector_data(
    rows: int = 500,
) -> dict[str, pd.Series]:
    return {
        "NIFTY_IT": make_market_series(
            rows,
            trend=0.18,
            start=120.0,
        ),
        "NIFTY_AUTO": make_market_series(
            rows,
            trend=0.11,
            start=110.0,
        ),
    }


def make_config(
    **kwargs,
) -> MarketRegimeIntegrationConfig:
    defaults = {
        "minimum_market_features": 1,
        "minimum_observations": 50,
    }

    defaults.update(kwargs)

    return MarketRegimeIntegrationConfig(
        **defaults
    )


def test_default_market_regimes():
    assert "BULL" in DEFAULT_MARKET_REGIMES
    assert "BEAR" in DEFAULT_MARKET_REGIMES
    assert "NEUTRAL" in DEFAULT_MARKET_REGIMES
    assert "UNKNOWN" in DEFAULT_MARKET_REGIMES


def test_default_volatility_regimes():
    assert "LOW" in DEFAULT_VOLATILITY_REGIMES
    assert "NORMAL" in DEFAULT_VOLATILITY_REGIMES
    assert "HIGH" in DEFAULT_VOLATILITY_REGIMES
    assert "UNKNOWN" in DEFAULT_VOLATILITY_REGIMES


def test_config_construction():
    config = make_config()

    assert isinstance(
        config,
        MarketRegimeIntegrationConfig,
    )

    assert (
        config.minimum_market_features
        >= 1
    )


def test_invalid_market_feature_count():
    with pytest.raises(ValueError):
        MarketRegimeIntegrationConfig(
            minimum_market_features=0
        )


def test_invalid_minimum_observations():
    with pytest.raises(ValueError):
        MarketRegimeIntegrationConfig(
            minimum_observations=0
        )


def test_pipeline_construction():
    pipeline = MarketRegimeIntegration(
        config=make_config()
    )

    assert isinstance(
        pipeline,
        MarketRegimeIntegration,
    )


def test_basic_integration():
    result = integrate_market_regime(
        make_stock_data(),
        make_market_data(),
        config=make_config(),
    )

    assert isinstance(
        result,
        MarketRegimeIntegrationResult,
    )

    assert result.success


def test_stock_timeline_is_preserved():
    stock = make_stock_data()

    result = integrate_market_regime(
        stock,
        make_market_data(),
        config=make_config(),
    )

    assert result.data.index.equals(
        stock.index
    )


def test_stock_columns_are_preserved():
    stock = make_stock_data()

    result = integrate_market_regime(
        stock,
        make_market_data(),
        config=make_config(),
    )

    for column in stock.columns:
        assert column in result.data.columns


def test_market_features_are_created():
    result = integrate_market_regime(
        make_stock_data(),
        make_market_data(),
        config=make_config(),
    )

    assert result.market_columns

    for column in result.market_columns:
        assert column in result.data.columns


def test_market_return_features_exist():
    result = integrate_market_regime(
        make_stock_data(),
        make_market_data(),
        config=make_config(),
    )

    expected_patterns = (
        "Market_NIFTY50_Return_1",
        "Market_NIFTY50_Return_5",
        "Market_NIFTY50_Return_20",
    )

    for column in expected_patterns:
        assert column in result.data.columns


def test_market_regime_is_created():
    result = integrate_market_regime(
        make_stock_data(),
        make_market_data(),
        config=make_config(),
    )

    assert result.market_regime is not None

    assert (
        "Market_Regime"
        in result.data.columns
    )


def test_volatility_regime_is_created():
    result = integrate_market_regime(
        make_stock_data(),
        make_market_data(),
        config=make_config(),
    )

    assert (
        result.volatility_regime
        is not None
    )

    assert (
        "Market_Volatility_Regime"
        in result.data.columns
    )


def test_relative_strength_is_created():
    result = integrate_market_regime(
        make_stock_data(),
        make_market_data(),
        config=make_config(),
    )

    assert result.relative_strength_columns

    assert (
        "Relative_Strength_20"
        in result.data.columns
    )

    assert (
        "Relative_Strength_Ratio_20"
        in result.data.columns
    )


def test_sector_features_are_created():
    result = integrate_market_regime(
        make_stock_data(),
        make_market_data(),
        sector_data=make_sector_data(),
        config=make_config(),
    )

    assert result.sector_columns

    assert any(
        column.startswith("Sector_")
        for column in result.sector_columns
    )


def test_sector_features_can_be_disabled():
    config = make_config(
        include_sector_features=False
    )

    result = integrate_market_regime(
        make_stock_data(),
        make_market_data(),
        sector_data=make_sector_data(),
        config=config,
    )

    assert not result.sector_columns


def test_relative_strength_can_be_disabled():
    config = make_config(
        include_relative_strength=False
    )

    result = integrate_market_regime(
        make_stock_data(),
        make_market_data(),
        config=config,
    )

    assert not result.relative_strength_columns


def test_regime_features_can_be_disabled():
    config = make_config(
        include_trend_regime=False,
        include_volatility_regime=False,
    )

    result = integrate_market_regime(
        make_stock_data(),
        make_market_data(),
        config=config,
    )

    assert not result.regime_columns


def test_market_strength_is_created():
    result = integrate_market_regime(
        make_stock_data(),
        make_market_data(),
        config=make_config(),
    )

    assert (
        result.market_strength
        is not None
    )

    assert (
        "Market_Strength_Score"
        in result.data.columns
    )


def test_market_strength_is_numeric():
    result = integrate_market_regime(
        make_stock_data(),
        make_market_data(),
        config=make_config(),
    )

    series = result.data[
        "Market_Strength_Score"
    ]

    assert pd.api.types.is_numeric_dtype(
        series
    )


def test_regime_values_are_valid():
    result = integrate_market_regime(
        make_stock_data(),
        make_market_data(),
        config=make_config(),
    )

    regimes = set(
        result.data[
            "Market_Regime"
        ].dropna().unique()
    )

    assert regimes.issubset(
        set(DEFAULT_MARKET_REGIMES)
    )


def test_volatility_regime_values_are_valid():
    result = integrate_market_regime(
        make_stock_data(),
        make_market_data(),
        config=make_config(),
    )

    regimes = set(
        result.data[
            "Market_Volatility_Regime"
        ].dropna().unique()
    )

    assert regimes.issubset(
        set(DEFAULT_VOLATILITY_REGIMES)
    )


def test_market_data_is_backward_aligned():
    stock = make_stock_data(
        rows=20
    )

    market_index = pd.date_range(
        "2020-01-01",
        periods=10,
        freq="2D",
    )

    market = pd.Series(
        np.arange(100, 110),
        index=market_index,
    )

    result = integrate_market_regime(
        stock,
        {"NIFTY50": market},
        config=make_config(
            minimum_market_features=1,
            minimum_observations=1,
        ),
    )

    for timestamp in result.data.index:
        available = market[
            market.index <= timestamp
        ]

        if available.empty:
            assert pd.isna(
                result.data.loc[
                    timestamp,
                    "Market_NIFTY50_Close",
                ]
            )
        else:
            expected = available.iloc[-1]

            actual = result.data.loc[
                timestamp,
                "Market_NIFTY50_Close",
            ]

            assert actual == expected


def test_future_market_observation_is_not_used():
    stock = make_stock_data(
        rows=20
    )

    market_index = pd.date_range(
        "2020-01-01",
        periods=5,
        freq="4D",
    )

    market = pd.Series(
        [100, 101, 102, 103, 104],
        index=market_index,
    )

    result = integrate_market_regime(
        stock,
        {"NIFTY50": market},
        config=make_config(
            minimum_market_features=1,
            minimum_observations=1,
        ),
    )

    timestamp = stock.index[2]

    available = market[
        market.index <= timestamp
    ]

    expected = available.iloc[-1]

    actual = result.data.loc[
        timestamp,
        "Market_NIFTY50_Close",
    ]

    assert actual == expected


def test_future_market_mutation_does_not_change_earlier_alignment():
    stock = make_stock_data(
        rows=30
    )

    market = make_market_series(
        rows=30
    )

    original = integrate_market_regime(
        stock,
        {"NIFTY50": market},
        config=make_config(
            minimum_market_features=1,
            minimum_observations=1,
        ),
    )

    mutated_market = market.copy()

    cutoff = stock.index[15]

    mutated_market.loc[
        mutated_market.index > cutoff
    ] *= 1000.0

    mutated = integrate_market_regime(
        stock,
        {"NIFTY50": mutated_market},
        config=make_config(
            minimum_market_features=1,
            minimum_observations=1,
        ),
    )

    early = stock.index[
        :15
    ]

    pd.testing.assert_frame_equal(
        original.data.loc[
            early,
            [
                "Market_NIFTY50_Close",
            ],
        ],
        mutated.data.loc[
            early,
            [
                "Market_NIFTY50_Close",
            ],
        ],
    )


def test_no_forward_fill_is_used():
    stock = make_stock_data(
        rows=10
    )

    market_index = pd.DatetimeIndex(
        [
            stock.index[3],
            stock.index[7],
        ]
    )

    market = pd.Series(
        [100.0, 200.0],
        index=market_index,
    )

    result = integrate_market_regime(
        stock,
        {"NIFTY50": market},
        config=make_config(
            minimum_market_features=1,
            minimum_observations=1,
        ),
    )

    assert pd.isna(
        result.data.loc[
            stock.index[0],
            "Market_NIFTY50_Close",
        ]
    )

    assert (
        result.data.loc[
            stock.index[3],
            "Market_NIFTY50_Close",
        ]
        == 100.0
    )

    assert (
        result.data.loc[
            stock.index[6],
            "Market_NIFTY50_Close",
        ]
        == 100.0
    )

    assert (
        result.data.loc[
            stock.index[7],
            "Market_NIFTY50_Close",
        ]
        == 200.0
    )


def test_market_columns_are_numeric():
    result = integrate_market_regime(
        make_stock_data(),
        make_market_data(),
        config=make_config(),
    )

    for column in result.market_columns:
        assert pd.api.types.is_numeric_dtype(
            result.data[column]
        )


def test_input_stock_data_is_not_mutated():
    stock = make_stock_data()

    original = stock.copy(
        deep=True
    )

    integrate_market_regime(
        stock,
        make_market_data(),
        sector_data=make_sector_data(),
        config=make_config(),
    )

    pd.testing.assert_frame_equal(
        stock,
        original,
    )


def test_market_series_are_not_mutated():
    market = make_market_data()

    originals = {
        name: series.copy(
            deep=True
        )
        for name, series
        in market.items()
    }

    integrate_market_regime(
        make_stock_data(),
        market,
        config=make_config(),
    )

    for name in market:
        pd.testing.assert_series_equal(
            market[name],
            originals[name],
        )


def test_missing_market_data_is_handled():
    stock = make_stock_data()

    market = {
        "NIFTY50": make_market_series()
    }

    result = integrate_market_regime(
        stock,
        market,
        config=make_config(
            minimum_market_features=1
        ),
    )

    assert isinstance(
        result,
        MarketRegimeIntegrationResult,
    )

    assert result.success


def test_empty_market_data_fails_closed():
    result = integrate_market_regime(
        make_stock_data(),
        {},
        config=make_config(),
    )

    assert not result.success
    assert result.errors


def test_wrong_stock_type_is_rejected():
    with pytest.raises(TypeError):
        integrate_market_regime(
            None,
            make_market_data(),
            config=make_config(),
        )


def test_empty_stock_data_is_rejected():
    empty = make_stock_data().iloc[
        0:0
    ]

    with pytest.raises(ValueError):
        integrate_market_regime(
            empty,
            make_market_data(),
            config=make_config(),
        )


def test_missing_ohlcv_is_rejected():
    stock = make_stock_data().drop(
        columns=["Volume"]
    )

    with pytest.raises(ValueError):
        integrate_market_regime(
            stock,
            make_market_data(),
            config=make_config(),
        )


def test_duplicate_stock_timestamps_are_rejected():
    stock = make_stock_data()

    duplicate = pd.concat(
        [
            stock,
            stock.iloc[[0]],
        ]
    )

    with pytest.raises(ValueError):
        integrate_market_regime(
            duplicate,
            make_market_data(),
            config=make_config(),
        )


def test_unsorted_stock_data_is_rejected():
    stock = make_stock_data().sample(
        frac=1.0,
        random_state=42,
    )

    with pytest.raises(ValueError):
        integrate_market_regime(
            stock,
            make_market_data(),
            config=make_config(),
        )


def test_invalid_market_series_type_is_rejected():
    with pytest.raises(TypeError):
        integrate_market_regime(
            make_stock_data(),
            {
                "NIFTY50": [1, 2, 3]
            },
            config=make_config(),
        )


def test_invalid_sector_series_type_is_rejected():
    with pytest.raises(TypeError):
        integrate_market_regime(
            make_stock_data(),
            make_market_data(),
            sector_data={
                "NIFTY_IT": [1, 2, 3]
            },
            config=make_config(),
        )


def test_non_numeric_market_series_fails():
    index = pd.date_range(
        "2020-01-01",
        periods=10,
        freq="D",
    )

    market = pd.Series(
        ["bad"] * 10,
        index=index,
    )

    result = integrate_market_regime(
        make_stock_data(
            rows=10
        ),
        {"NIFTY50": market},
        config=make_config(
            minimum_market_features=1,
            minimum_observations=1,
        ),
    )

    assert not result.success
    assert result.errors


def test_market_regime_summary():
    result = integrate_market_regime(
        make_stock_data(),
        make_market_data(),
        sector_data=make_sector_data(),
        config=make_config(),
    )

    summary = market_regime_integration_summary(
        result
    )

    assert isinstance(
        summary,
        dict,
    )

    assert (
        summary["research_only"]
        is True
    )

    assert (
        summary["production_ready"]
        is False
    )

    assert (
        summary["final_holdout_used"]
        is False
    )


def test_summary_rejects_wrong_type():
    with pytest.raises(TypeError):
        market_regime_integration_summary(
            None
        )


def test_research_only_metadata():
    result = integrate_market_regime(
        make_stock_data(),
        make_market_data(),
        config=make_config(),
    )

    assert (
        result.metadata[
            "research_only"
        ]
        is True
    )

    assert (
        result.metadata[
            "production_ready"
        ]
        is False
    )

    assert (
        result.metadata[
            "production_approved"
        ]
        is False
    )

    assert (
        result.metadata[
            "final_holdout_used"
        ]
        is False
    )


def test_future_market_data_flag_is_false():
    result = integrate_market_regime(
        make_stock_data(),
        make_market_data(),
        config=make_config(),
    )

    assert (
        result.metadata[
            "future_market_data_used"
        ]
        is False
    )


def test_forward_fill_flag_is_false():
    result = integrate_market_regime(
        make_stock_data(),
        make_market_data(),
        config=make_config(),
    )

    assert (
        result.metadata[
            "forward_fill_used"
        ]
        is False
    )


def test_deterministic_output():
    stock = make_stock_data()
    market = make_market_data()
    sector = make_sector_data()

    first = integrate_market_regime(
        stock,
        market,
        sector_data=sector,
        config=make_config(),
    )

    second = integrate_market_regime(
        stock,
        market,
        sector_data=sector,
        config=make_config(),
    )

    pd.testing.assert_frame_equal(
        first.data,
        second.data,
    )

    assert (
        first.market_columns
        == second.market_columns
    )

    assert (
        first.regime_columns
        == second.regime_columns
    )

    assert (
        first.relative_strength_columns
        == second.relative_strength_columns
    )

    assert (
        first.sector_columns
        == second.sector_columns
    )


def test_result_summary_is_consistent():
    result = integrate_market_regime(
        make_stock_data(),
        make_market_data(),
        config=make_config(),
    )

    summary = result.summary()

    assert (
        summary["rows"]
        == len(result.data)
    )

    assert (
        summary["market_features"]
        == len(result.market_columns)
    )

    assert (
        summary["regime_features"]
        == len(result.regime_columns)
    )


def test_production_ready_always_false():
    result = integrate_market_regime(
        make_stock_data(),
        make_market_data(),
        config=make_config(),
    )

    assert (
        result.production_ready
        is False
    )
