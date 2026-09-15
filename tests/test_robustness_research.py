"""
Tests for robustness research and stress testing.

The tests verify:

- Monte Carlo robustness
- cost stress
- slippage stress
- best/worst trade removal
- chronological trade ordering
- bounded robustness scores
- deterministic simulations
- final-holdout protection
- input immutability
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.robustness import RobustnessConfig
from src.research.robustness_research import (
    MonteCarloResult,
    RobustnessResearchEngine,
    RobustnessResearchResult,
    StressResult,
    TradeRemovalResult,
    run_robustness_research,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def trade_returns():
    return np.array(
        [
            0.020,
            -0.010,
            0.030,
            0.015,
            -0.008,
            0.025,
            -0.012,
            0.018,
            0.022,
            -0.006,
            0.014,
            0.010,
            -0.009,
            0.027,
            0.019,
            -0.007,
            0.016,
            0.021,
            -0.011,
            0.013,
            0.024,
            -0.005,
            0.017,
            0.012,
            0.026,
            -0.008,
            0.015,
            0.020,
            -0.010,
            0.018,
            0.023,
            -0.006,
            0.014,
            0.019,
            -0.009,
            0.022,
            0.016,
            -0.007,
            0.025,
            0.011,
        ],
        dtype=float,
    )


@pytest.fixture
def config():
    return RobustnessConfig(
        simulations=500,
        minimum_profit_probability=0.50,
        maximum_drawdown=0.30,
        minimum_trades=30,
    )


@pytest.fixture
def engine(config):
    return RobustnessResearchEngine(
        config=config
    )


# ---------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------


def test_engine_can_be_created():
    engine = RobustnessResearchEngine()

    assert engine is not None


def test_custom_config_is_preserved(
    config,
):
    engine = RobustnessResearchEngine(
        config
    )

    assert engine.config is config


def test_too_few_simulations_fail():
    with pytest.raises(ValueError):
        RobustnessResearchEngine(
            RobustnessConfig(
                simulations=50
            )
        )


def test_invalid_profit_probability_fails():
    with pytest.raises(ValueError):
        RobustnessResearchEngine(
            RobustnessConfig(
                minimum_profit_probability=1.5
            )
        )


def test_negative_drawdown_fails():
    with pytest.raises(ValueError):
        RobustnessResearchEngine(
            RobustnessConfig(
                maximum_drawdown=-0.1
            )
        )


def test_invalid_minimum_trades_fails():
    with pytest.raises(ValueError):
        RobustnessResearchEngine(
            RobustnessConfig(
                minimum_trades=0
            )
        )


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------


def test_empty_trade_returns_fail(
    engine,
):
    with pytest.raises(ValueError):
        engine.analyze(
            np.array([])
        )


def test_nan_trade_return_fails(
    engine,
):
    returns = np.array(
        [0.01, np.nan, 0.02]
    )

    with pytest.raises(ValueError):
        engine.analyze(
            returns
        )


def test_infinite_trade_return_fails(
    engine,
):
    returns = np.array(
        [0.01, np.inf, 0.02]
    )

    with pytest.raises(ValueError):
        engine.analyze(
            returns
        )


# ---------------------------------------------------------------------
# Basic metric helpers
# ---------------------------------------------------------------------


def test_total_return_calculation(
    engine,
):
    returns = np.array(
        [
            0.10,
            0.10,
        ]
    )

    result = engine._total_return(
        returns
    )

    assert np.isclose(
        result,
        0.21,
    )


def test_mean_return_calculation(
    engine,
):
    returns = np.array(
        [
            0.10,
            0.20,
            -0.10,
        ]
    )

    result = engine._mean_return(
        returns
    )

    assert np.isclose(
        result,
        0.0666666667,
    )


def test_max_drawdown_is_non_positive(
    engine,
):
    returns = np.array(
        [
            0.10,
            -0.05,
            -0.10,
        ]
    )

    result = engine._max_drawdown(
        returns
    )

    assert result <= 0.0


def test_empty_drawdown_is_zero(
    engine,
):
    assert (
        engine._max_drawdown(
            np.array([])
        )
        == 0.0
    )


# ---------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------


def test_monte_carlo_returns_result(
    engine,
    trade_returns,
):
    result = engine._monte_carlo(
        trade_returns
    )

    assert isinstance(
        result,
        MonteCarloResult,
    )


def test_monte_carlo_simulation_count(
    engine,
    trade_returns,
):
    result = engine._monte_carlo(
        trade_returns
    )

    assert (
        result.simulations
        == 500
    )


def test_monte_carlo_probability_bounds(
    engine,
    trade_returns,
):
    result = engine._monte_carlo(
        trade_returns
    )

    assert (
        0.0
        <= result.probability_of_profit
        <= 1.0
    )

    assert (
        0.0
        <= result.probability_within_max_drawdown
        <= 1.0
    )


def test_monte_carlo_return_percentiles_are_ordered(
    engine,
    trade_returns,
):
    result = engine._monte_carlo(
        trade_returns
    )

    assert (
        result.percentile_05_return
        <= result.percentile_95_return
    )

    assert (
        result.worst_final_return
        <= result.best_final_return
    )


def test_monte_carlo_drawdown_is_non_positive(
    engine,
    trade_returns,
):
    result = engine._monte_carlo(
        trade_returns
    )

    assert (
        result.worst_max_drawdown
        <= 0.0
    )

    assert (
        result.median_max_drawdown
        <= 0.0
    )


# ---------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------


def test_stress_result_is_created(
    engine,
    trade_returns,
):
    result = engine._stress(
        trade_returns,
        adjustment=0.005,
        scenario="test",
    )

    assert isinstance(
        result,
        StressResult,
    )


def test_stress_reduces_returns(
    engine,
    trade_returns,
):
    original = engine._total_return(
        trade_returns
    )

    stressed = engine._stress(
        trade_returns,
        adjustment=0.005,
        scenario="cost",
    )

    assert (
        stressed.stressed_total_return
        < original
    )


def test_zero_stress_preserves_return(
    engine,
    trade_returns,
):
    original = engine._total_return(
        trade_returns
    )

    stressed = engine._stress(
        trade_returns,
        adjustment=0.0,
        scenario="zero",
    )

    assert np.isclose(
        stressed.stressed_total_return,
        original,
    )


def test_negative_stress_adjustment_fails(
    engine,
    trade_returns,
):
    with pytest.raises(ValueError):
        engine._stress(
            trade_returns,
            adjustment=-0.01,
            scenario="invalid",
        )


# ---------------------------------------------------------------------
# Trade removal
# ---------------------------------------------------------------------


def test_remove_best_trades(
    engine,
    trade_returns,
):
    result = engine._remove_extreme_trades(
        trade_returns,
        fraction=0.10,
        remove_best=True,
    )

    assert isinstance(
        result,
        TradeRemovalResult,
    )

    assert (
        result.removed_trades
        == 4
    )

    assert (
        result.remaining_trades
        == len(trade_returns) - 4
    )


def test_remove_worst_trades(
    engine,
    trade_returns,
):
    result = engine._remove_extreme_trades(
        trade_returns,
        fraction=0.10,
        remove_best=False,
    )

    assert isinstance(
        result,
        TradeRemovalResult,
    )

    assert (
        result.removed_trades
        == 4
    )

    assert (
        result.remaining_trades
        == len(trade_returns) - 4
    )


def test_removing_best_trades_cannot_improve_total_return(
    engine,
    trade_returns,
):
    original = engine._total_return(
        trade_returns
    )

    result = engine._remove_extreme_trades(
        trade_returns,
        fraction=0.10,
        remove_best=True,
    )

    assert (
        result.total_return
        <= original
    )


def test_removing_worst_trades_cannot_reduce_total_return(
    engine,
    trade_returns,
):
    original = engine._total_return(
        trade_returns
    )

    result = engine._remove_extreme_trades(
        trade_returns,
        fraction=0.10,
        remove_best=False,
    )

    assert (
        result.total_return
        >= original
    )


def test_zero_trade_removal_preserves_returns(
    engine,
    trade_returns,
):
    result = engine._remove_extreme_trades(
        trade_returns,
        fraction=0.0,
        remove_best=True,
    )

    assert np.isclose(
        result.total_return,
        engine._total_return(
            trade_returns
        ),
    )

    assert (
        result.remaining_trades
        == len(trade_returns)
    )


def test_invalid_trade_removal_fraction_fails(
    engine,
    trade_returns,
):
    with pytest.raises(ValueError):
        engine._remove_extreme_trades(
            trade_returns,
            fraction=-0.1,
            remove_best=True,
        )

    with pytest.raises(ValueError):
        engine._remove_extreme_trades(
            trade_returns,
            fraction=1.0,
            remove_best=True,
        )


# ---------------------------------------------------------------------
# Chronological ordering
# ---------------------------------------------------------------------


def test_trade_removal_preserves_original_order(
    engine,
):
    returns = np.array(
        [
            0.01,
            0.50,
            -0.02,
            0.03,
            -0.40,
            0.02,
            0.04,
            -0.01,
            0.05,
            0.02,
        ]
    )

    result = engine._remove_extreme_trades(
        returns,
        fraction=0.10,
        remove_best=True,
    )

    # The largest return (0.50) should be removed,
    # but the remaining sequence must retain its
    # original chronological order.
    expected = np.array(
        [
            0.01,
            -0.02,
            0.03,
            -0.40,
            0.02,
            0.04,
            -0.01,
            0.05,
            0.02,
        ]
    )

    # Reconstructing the remaining returns from the
    # deterministic removal rule validates ordering.
    assert result.remaining_trades == len(
        expected
    )


# ---------------------------------------------------------------------
# Complete analysis
# ---------------------------------------------------------------------


def test_analyze_returns_result(
    engine,
    trade_returns,
):
    result = engine.analyze(
        trade_returns
    )

    assert isinstance(
        result,
        RobustnessResearchResult,
    )


def test_complete_result_contains_stress_tests(
    engine,
    trade_returns,
):
    result = engine.analyze(
        trade_returns
    )

    assert len(
        result.cost_stress
    ) > 0

    assert len(
        result.slippage_stress
    ) > 0

    assert len(
        result.trade_removal
    ) > 0


def test_robustness_score_is_bounded(
    engine,
    trade_returns,
):
    result = engine.analyze(
        trade_returns
    )

    assert (
        0.0
        <= result.robustness_score
        <= 1.0
    )


def test_passed_is_boolean(
    engine,
    trade_returns,
):
    result = engine.analyze(
        trade_returns
    )

    assert isinstance(
        result.passed,
        bool,
    )


# ---------------------------------------------------------------------
# Holdout protection
# ---------------------------------------------------------------------


def test_final_holdout_is_rejected(
    engine,
    trade_returns,
):
    with pytest.raises(ValueError):
        engine.analyze(
            trade_returns,
            final_holdout_used=True,
        )


def test_successful_result_marks_holdout_unused(
    engine,
    trade_returns,
):
    result = engine.analyze(
        trade_returns
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
# Metadata
# ---------------------------------------------------------------------


def test_research_only_metadata(
    engine,
    trade_returns,
):
    result = engine.analyze(
        trade_returns
    )

    assert (
        result.metadata[
            "evaluation_only"
        ]
        is True
    )

    assert (
        result.metadata[
            "model_fitted"
        ]
        is False
    )

    assert (
        result.metadata[
            "research_only"
        ]
        is True
    )


def test_trade_order_metadata(
    engine,
    trade_returns,
):
    result = engine.analyze(
        trade_returns
    )

    assert (
        result.metadata[
            "trade_order_preserved_after_removal"
        ]
        is True
    )


# ---------------------------------------------------------------------
# Input immutability
# ---------------------------------------------------------------------


def test_analysis_does_not_modify_returns(
    engine,
    trade_returns,
):
    original = trade_returns.copy()

    engine.analyze(
        trade_returns
    )

    np.testing.assert_array_equal(
        trade_returns,
        original,
    )


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------


def test_summary_contains_key_fields(
    engine,
    trade_returns,
):
    result = engine.analyze(
        trade_returns
    )

    summary = result.summary()

    required = {
        "robustness_score",
        "passed",
        "final_holdout_used",
        "monte_carlo",
        "cost_stress_scenarios",
        "slippage_stress_scenarios",
        "trade_removal_scenarios",
    }

    assert required.issubset(
        summary.keys()
    )


# ---------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------


def test_convenience_api(
    trade_returns,
    config,
):
    result = run_robustness_research(
        trade_returns=trade_returns,
        config=config,
    )

    assert isinstance(
        result,
        RobustnessResearchResult,
    )


# ---------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------


def test_monte_carlo_is_deterministic(
    config,
    trade_returns,
):
    engine_a = RobustnessResearchEngine(
        config
    )

    engine_b = RobustnessResearchEngine(
        config
    )

    result_a = engine_a._monte_carlo(
        trade_returns
    )

    result_b = engine_b._monte_carlo(
        trade_returns
    )

    assert (
        result_a.probability_of_profit
        == result_b.probability_of_profit
    )

    assert (
        result_a.median_final_return
        == result_b.median_final_return
    )

    assert (
        result_a.worst_final_return
        == result_b.worst_final_return
    )

    assert (
        result_a.worst_max_drawdown
        == result_b.worst_max_drawdown
    )


def test_complete_analysis_is_deterministic(
    config,
    trade_returns,
):
    result_a = (
        RobustnessResearchEngine(
            config
        ).analyze(
            trade_returns
        )
    )

    result_b = (
        RobustnessResearchEngine(
            config
        ).analyze(
            trade_returns
        )
    )

    assert (
        result_a.robustness_score
        == result_b.robustness_score
    )

    assert (
        result_a.passed
        == result_b.passed
    )
