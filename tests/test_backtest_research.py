"""
Tests for the research backtest engine.

The tests verify:

- valid OHLC input
- exact prediction timestamp alignment
- no forward filling of predictions
- invalid probability protection
- chronological protection
- final-holdout protection
- transaction-cost/slippage validation
- return-distribution helpers
- input immutability
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.backtest import BacktestConfig
from src.research.backtest_research import (
    BacktestResearchEngine,
    BacktestResearchResult,
    calculate_return_distribution,
    stress_trade_returns,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def market_data() -> pd.DataFrame:
    index = pd.date_range(
        "2020-01-01",
        periods=150,
        freq="D",
    )

    close = (
        100.0
        + np.linspace(
            0.0,
            20.0,
            len(index),
        )
    )

    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(
                len(index),
                100_000,
                dtype=float,
            ),
        },
        index=index,
    )


@pytest.fixture
def probabilities(market_data):
    values = np.full(
        len(market_data),
        0.65,
        dtype=float,
    )

    values[::10] = 0.40

    return values


@pytest.fixture
def engine():
    return BacktestResearchEngine(
        BacktestConfig(
            probability_threshold=0.60,
            transaction_cost=0.001,
            slippage=0.0005,
        )
    )


# ---------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------


def test_engine_can_be_created():
    engine = BacktestResearchEngine()

    assert engine is not None


def test_custom_config_is_preserved():
    config = BacktestConfig(
        probability_threshold=0.65
    )

    engine = BacktestResearchEngine(
        config
    )

    assert engine.config is config


def test_invalid_probability_threshold_fails():
    with pytest.raises(ValueError):
        BacktestResearchEngine(
            BacktestConfig(
                probability_threshold=1.5
            )
        )


def test_negative_transaction_cost_fails():
    with pytest.raises(ValueError):
        BacktestResearchEngine(
            BacktestConfig(
                transaction_cost_pct=-0.01
            )
        )


def test_negative_slippage_fails():
    with pytest.raises(ValueError):
        BacktestResearchEngine(
            BacktestConfig(
                slippage_pct=-0.01
            )
        )


# ---------------------------------------------------------------------
# Market-data validation
# ---------------------------------------------------------------------


def test_empty_market_data_fails(
    engine,
):
    with pytest.raises(ValueError):
        engine.run(
            market_data=pd.DataFrame(),
            probabilities=np.array([]),
            prediction_timestamps=pd.DatetimeIndex([]),
        )


def test_missing_ohlc_column_fails(
    engine,
    market_data,
    probabilities,
):
    broken = market_data.drop(
        columns=["High"]
    )

    with pytest.raises(ValueError):
        engine.run(
            market_data=broken,
            probabilities=probabilities,
            prediction_timestamps=market_data.index,
        )


def test_non_datetime_market_index_fails(
    engine,
    market_data,
    probabilities,
):
    broken = market_data.copy()
    broken.index = np.arange(
        len(broken)
    )

    with pytest.raises(TypeError):
        engine.run(
            market_data=broken,
            probabilities=probabilities,
            prediction_timestamps=pd.DatetimeIndex(
                market_data.index
            ),
        )


def test_duplicate_market_timestamps_fail(
    engine,
    market_data,
    probabilities,
):
    broken = market_data.copy()

    duplicate_index = list(
        broken.index
    )

    duplicate_index[-1] = (
        duplicate_index[-2]
    )

    broken.index = pd.DatetimeIndex(
        duplicate_index
    )

    with pytest.raises(ValueError):
        engine.run(
            market_data=broken,
            probabilities=probabilities,
            prediction_timestamps=market_data.index,
        )


def test_unsorted_market_data_fails(
    engine,
    market_data,
    probabilities,
):
    broken = market_data.iloc[
        ::-1
    ].copy()

    with pytest.raises(ValueError):
        engine.run(
            market_data=broken,
            probabilities=probabilities,
            prediction_timestamps=market_data.index,
        )


def test_invalid_high_price_fails(
    engine,
    market_data,
    probabilities,
):
    broken = market_data.copy()

    broken.iloc[
        0,
        broken.columns.get_loc(
            "High"
        ),
    ] = broken.iloc[
        0,
        broken.columns.get_loc(
            "Close"
        ),
    ] - 10

    with pytest.raises(ValueError):
        engine.run(
            market_data=broken,
            probabilities=probabilities,
            prediction_timestamps=market_data.index,
        )


def test_invalid_low_price_fails(
    engine,
    market_data,
    probabilities,
):
    broken = market_data.copy()

    broken.iloc[
        0,
        broken.columns.get_loc(
            "Low"
        ),
    ] = broken.iloc[
        0,
        broken.columns.get_loc(
            "Close"
        ),
    ] + 10

    with pytest.raises(ValueError):
        engine.run(
            market_data=broken,
            probabilities=probabilities,
            prediction_timestamps=market_data.index,
        )


# ---------------------------------------------------------------------
# Prediction validation
# ---------------------------------------------------------------------


def test_empty_predictions_fail(
    engine,
    market_data,
):
    with pytest.raises(ValueError):
        engine.run(
            market_data=market_data,
            probabilities=np.array([]),
            prediction_timestamps=pd.DatetimeIndex([]),
        )


def test_probability_length_mismatch_fails(
    engine,
    market_data,
):
    with pytest.raises(ValueError):
        engine.run(
            market_data=market_data,
            probabilities=np.full(
                len(market_data) - 1,
                0.65,
            ),
            prediction_timestamps=market_data.index,
        )


def test_probability_below_zero_fails(
    engine,
    market_data,
    probabilities,
):
    broken = probabilities.copy()
    broken[0] = -0.1

    with pytest.raises(ValueError):
        engine.run(
            market_data=market_data,
            probabilities=broken,
            prediction_timestamps=market_data.index,
        )


def test_probability_above_one_fails(
    engine,
    market_data,
    probabilities,
):
    broken = probabilities.copy()
    broken[0] = 1.1

    with pytest.raises(ValueError):
        engine.run(
            market_data=market_data,
            probabilities=broken,
            prediction_timestamps=market_data.index,
        )


def test_nan_probability_fails(
    engine,
    market_data,
    probabilities,
):
    broken = probabilities.copy()
    broken[0] = np.nan

    with pytest.raises(ValueError):
        engine.run(
            market_data=market_data,
            probabilities=broken,
            prediction_timestamps=market_data.index,
        )


def test_duplicate_prediction_timestamps_fail(
    engine,
    market_data,
    probabilities,
):
    timestamps = list(
        market_data.index
    )

    timestamps[-1] = timestamps[-2]

    with pytest.raises(ValueError):
        engine.run(
            market_data=market_data,
            probabilities=probabilities,
            prediction_timestamps=pd.DatetimeIndex(
                timestamps
            ),
        )


def test_unsorted_prediction_timestamps_fail(
    engine,
    market_data,
    probabilities,
):
    with pytest.raises(ValueError):
        engine.run(
            market_data=market_data,
            probabilities=probabilities,
            prediction_timestamps=(
                market_data.index[::-1]
            ),
        )


# ---------------------------------------------------------------------
# Prediction alignment
# ---------------------------------------------------------------------


def test_alignment_preserves_exact_timestamps(
    engine,
    market_data,
    probabilities,
):
    timestamps = market_data.index[
        10:30
    ]

    values = probabilities[
        10:30
    ]

    aligned = engine.align_predictions(
        market_data=market_data,
        probabilities=values,
        prediction_timestamps=timestamps,
    )

    assert (
        aligned.loc[
            timestamps,
            "Probability",
        ].notna().all()
    )

    np.testing.assert_allclose(
        aligned.loc[
            timestamps,
            "Probability",
        ].to_numpy(),
        values,
    )


def test_alignment_does_not_forward_fill(
    engine,
    market_data,
):
    timestamps = market_data.index[
        [10, 20, 30]
    ]

    probabilities = np.array(
        [0.80, 0.70, 0.90]
    )

    aligned = engine.align_predictions(
        market_data=market_data,
        probabilities=probabilities,
        prediction_timestamps=timestamps,
    )

    between = aligned.loc[
        market_data.index[11:20],
        "Probability",
    ]

    assert between.isna().all()


def test_prediction_timestamp_outside_market_data_fails(
    engine,
    market_data,
):
    timestamps = pd.DatetimeIndex(
        [
            market_data.index[0],
            pd.Timestamp(
                "2035-01-01"
            ),
        ]
    )

    probabilities = np.array(
        [0.70, 0.80]
    )

    with pytest.raises(ValueError):
        engine.run(
            market_data=market_data,
            probabilities=probabilities,
            prediction_timestamps=timestamps,
        )


# ---------------------------------------------------------------------
# Successful backtest
# ---------------------------------------------------------------------


def test_run_returns_result(
    engine,
    market_data,
    probabilities,
):
    result = engine.run(
        market_data=market_data,
        probabilities=probabilities,
        prediction_timestamps=market_data.index,
    )

    assert isinstance(
        result,
        BacktestResearchResult,
    )


def test_sample_count_is_correct(
    engine,
    market_data,
    probabilities,
):
    result = engine.run(
        market_data=market_data,
        probabilities=probabilities,
        prediction_timestamps=market_data.index,
    )

    assert (
        result.samples
        == len(market_data)
    )


def test_prediction_period_is_recorded(
    engine,
    market_data,
    probabilities,
):
    result = engine.run(
        market_data=market_data,
        probabilities=probabilities,
        prediction_timestamps=market_data.index,
    )

    assert (
        result.prediction_start
        == market_data.index.min()
    )

    assert (
        result.prediction_end
        == market_data.index.max()
    )


def test_threshold_is_recorded(
    engine,
    market_data,
    probabilities,
):
    result = engine.run(
        market_data=market_data,
        probabilities=probabilities,
        prediction_timestamps=market_data.index,
    )

    assert (
        result.probability_threshold
        == 0.60
    )


# ---------------------------------------------------------------------
# Holdout safety
# ---------------------------------------------------------------------


def test_final_holdout_is_rejected(
    engine,
    market_data,
    probabilities,
):
    with pytest.raises(ValueError):
        engine.run(
            market_data=market_data,
            probabilities=probabilities,
            prediction_timestamps=market_data.index,
            final_holdout_used=True,
        )


def test_result_records_holdout_unused(
    engine,
    market_data,
    probabilities,
):
    result = engine.run(
        market_data=market_data,
        probabilities=probabilities,
        prediction_timestamps=market_data.index,
    )

    assert (
        result.final_holdout_used
        is False
    )

    assert (
        result.metadata[
            "final_holdout_used"
        ]
        is False
    )


# ---------------------------------------------------------------------
# Research-only metadata
# ---------------------------------------------------------------------


def test_backtest_is_marked_research_only(
    engine,
    market_data,
    probabilities,
):
    result = engine.run(
        market_data=market_data,
        probabilities=probabilities,
        prediction_timestamps=market_data.index,
    )

    assert (
        result.metadata[
            "research_only"
        ]
        is True
    )

    assert (
        result.metadata[
            "model_fitted"
        ]
        is False
    )


def test_forward_fill_is_explicitly_disabled(
    engine,
    market_data,
    probabilities,
):
    result = engine.run(
        market_data=market_data,
        probabilities=probabilities,
        prediction_timestamps=market_data.index,
    )

    assert (
        result.metadata[
            "forward_fill_predictions"
        ]
        is False
    )


# ---------------------------------------------------------------------
# Input immutability
# ---------------------------------------------------------------------


def test_market_data_is_not_modified(
    engine,
    market_data,
    probabilities,
):
    original = market_data.copy(
        deep=True
    )

    engine.run(
        market_data=market_data,
        probabilities=probabilities,
        prediction_timestamps=market_data.index,
    )

    pd.testing.assert_frame_equal(
        market_data,
        original,
    )


def test_probabilities_are_not_modified(
    engine,
    market_data,
    probabilities,
):
    original = probabilities.copy()

    engine.run(
        market_data=market_data,
        probabilities=probabilities,
        prediction_timestamps=market_data.index,
    )

    np.testing.assert_array_equal(
        probabilities,
        original,
    )


# ---------------------------------------------------------------------
# Return-distribution helper
# ---------------------------------------------------------------------


def test_return_distribution():
    returns = np.array(
        [
            -0.05,
            -0.02,
            0.01,
            0.03,
            0.10,
        ]
    )

    result = calculate_return_distribution(
        returns
    )

    required = {
        "mean",
        "median",
        "std",
        "minimum",
        "maximum",
        "p05",
        "p25",
        "p75",
        "p95",
    }

    assert required.issubset(
        result.keys()
    )

    assert (
        result["minimum"]
        == -0.05
    )

    assert (
        result["maximum"]
        == 0.10
    )


def test_empty_return_distribution_fails():
    with pytest.raises(ValueError):
        calculate_return_distribution(
            np.array([])
        )


def test_nonfinite_return_distribution_fails():
    with pytest.raises(ValueError):
        calculate_return_distribution(
            np.array(
                [0.01, np.nan]
            )
        )


# ---------------------------------------------------------------------
# Stress helper
# ---------------------------------------------------------------------


def test_stress_trade_returns():
    returns = np.array(
        [
            0.05,
            -0.02,
            0.03,
        ]
    )

    stressed = stress_trade_returns(
        returns,
        additional_cost=0.01,
    )

    np.testing.assert_allclose(
        stressed,
        [
            0.04,
            -0.03,
            0.02,
        ],
    )


def test_negative_stress_cost_fails():
    with pytest.raises(ValueError):
        stress_trade_returns(
            np.array([0.01]),
            additional_cost=-0.01,
        )


def test_stress_does_not_modify_input():
    returns = np.array(
        [
            0.05,
            -0.02,
            0.03,
        ]
    )

    original = returns.copy()

    stress_trade_returns(
        returns,
        additional_cost=0.01,
    )

    np.testing.assert_array_equal(
        returns,
        original,
    )


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------


def test_summary_contains_key_fields(
    engine,
    market_data,
    probabilities,
):
    result = engine.run(
        market_data=market_data,
        probabilities=probabilities,
        prediction_timestamps=market_data.index,
    )

    summary = result.summary()

    required = {
        "samples",
        "prediction_start",
        "prediction_end",
        "probability_threshold",
        "transaction_cost",
        "slippage",
        "final_holdout_used",
    }

    assert required.issubset(
        summary.keys()
    )
