"""
Tests for multi-horizon robustness integration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.backtest_integration import (
    HorizonBacktestResult,
    MultiHorizonBacktestIntegrationResult,
)
from src.research.robustness_integration import (
    DEFAULT_ROBUSTNESS_HORIZONS,
    HorizonRobustnessResult,
    MultiHorizonRobustnessIntegrationResult,
    RobustnessIntegration,
    RobustnessIntegrationConfig,
    integrate_robustness,
    robustness_integration_summary,
)


def make_backtest_result(
    horizons=(1, 3, 5),
    *,
    passed=True,
):
    results = {}

    for horizon in horizons:
        results[horizon] = HorizonBacktestResult(
            horizon=horizon,
            model_id=f"model_{horizon}d",
            result=None,
            evaluated=True,
            passed=passed,
            total_return=0.20,
            annualized_return=0.30,
            sharpe_ratio=1.20,
            max_drawdown=-0.15,
            win_rate=0.60,
            profit_factor=1.80,
            trade_count=100,
            warnings=[],
            metadata={
                "research_only": True,
                "final_holdout_used": False,
                "production_approved": False,
            },
        )

    return MultiHorizonBacktestIntegrationResult(
        horizons=tuple(horizons),
        results=results,
        evaluated_horizons=tuple(horizons),
        passed_horizons=tuple(
            horizons if passed else ()
        ),
        failed_horizons=tuple(
            () if passed else horizons
        ),
        candidate_passed=passed,
        production_ready=False,
        warnings=[],
        errors=[],
        metadata={
            "research_only": True,
            "production_ready": False,
            "production_approved": False,
            "final_holdout_used": False,
        },
    )


def make_trade_returns(
    n=100,
    *,
    seed=42,
):
    rng = np.random.default_rng(seed)

    returns = (
        rng.normal(
            loc=0.01,
            scale=0.025,
            size=n,
        )
    )

    return pd.Series(
        returns,
        index=pd.RangeIndex(n),
        name="Trade_Return",
    )


def make_trade_return_map(
    horizons=(1, 3, 5),
):
    return {
        horizon: make_trade_returns(
            seed=42 + horizon
        )
        for horizon in horizons
    }


def make_config(
    horizons=(1, 3, 5),
    **kwargs,
):
    defaults = {
        "horizons": tuple(horizons),
        "simulations": 200,
        "minimum_trades": 30,
        "minimum_robustness_score": 0.0,
    }

    defaults.update(kwargs)

    return RobustnessIntegrationConfig(
        **defaults
    )


def test_default_horizons():
    assert DEFAULT_ROBUSTNESS_HORIZONS == (
        1,
        3,
        5,
        10,
        20,
    )


def test_config_construction():
    config = make_config()

    assert isinstance(
        config,
        RobustnessIntegrationConfig,
    )

    assert (
        config.simulations
        >= 100
    )


def test_empty_horizons_are_rejected():
    with pytest.raises(ValueError):
        RobustnessIntegrationConfig(
            horizons=()
        )


def test_non_positive_horizon_is_rejected():
    with pytest.raises(ValueError):
        RobustnessIntegrationConfig(
            horizons=(1, 0, 5)
        )


def test_duplicate_horizons_are_rejected():
    with pytest.raises(ValueError):
        RobustnessIntegrationConfig(
            horizons=(1, 3, 3)
        )


def test_too_few_simulations_are_rejected():
    with pytest.raises(ValueError):
        RobustnessIntegrationConfig(
            simulations=99
        )


def test_negative_cost_multiplier_is_rejected():
    with pytest.raises(ValueError):
        RobustnessIntegrationConfig(
            transaction_cost_multiplier=-1
        )


def test_negative_slippage_multiplier_is_rejected():
    with pytest.raises(ValueError):
        RobustnessIntegrationConfig(
            slippage_multiplier=-1
        )


def test_invalid_removal_fraction_is_rejected():
    with pytest.raises(ValueError):
        RobustnessIntegrationConfig(
            removal_fraction=-0.1
        )

    with pytest.raises(ValueError):
        RobustnessIntegrationConfig(
            removal_fraction=1.0
        )


def test_invalid_minimum_trade_count_is_rejected():
    with pytest.raises(ValueError):
        RobustnessIntegrationConfig(
            minimum_trades=0
        )


def test_invalid_robustness_threshold_is_rejected():
    with pytest.raises(ValueError):
        RobustnessIntegrationConfig(
            minimum_robustness_score=-0.1
        )

    with pytest.raises(ValueError):
        RobustnessIntegrationConfig(
            minimum_robustness_score=1.1
        )


def test_pipeline_construction():
    pipeline = RobustnessIntegration(
        config=make_config()
    )

    assert isinstance(
        pipeline,
        RobustnessIntegration,
    )


def test_backtest_result_type_is_required():
    pipeline = RobustnessIntegration(
        config=make_config()
    )

    with pytest.raises(TypeError):
        pipeline.run(
            backtest_result=None,
            trade_returns={},
        )


def test_backtest_final_holdout_usage_is_rejected():
    backtest = make_backtest_result(
        horizons=(1,)
    )

    backtest.metadata[
        "final_holdout_used"
    ] = True

    pipeline = RobustnessIntegration(
        config=make_config(
            horizons=(1,)
        )
    )

    with pytest.raises(ValueError):
        pipeline.run(
            backtest_result=backtest,
            trade_returns={
                1: make_trade_returns()
            },
        )


def test_production_approved_backtest_is_rejected():
    backtest = make_backtest_result(
        horizons=(1,)
    )

    backtest.metadata[
        "production_approved"
    ] = True

    pipeline = RobustnessIntegration(
        config=make_config(
            horizons=(1,)
        )
    )

    with pytest.raises(ValueError):
        pipeline.run(
            backtest_result=backtest,
            trade_returns={
                1: make_trade_returns()
            },
        )


def test_trade_returns_must_be_dictionary():
    pipeline = RobustnessIntegration(
        config=make_config(
            horizons=(1,)
        )
    )

    with pytest.raises(TypeError):
        pipeline.run(
            backtest_result=make_backtest_result(
                horizons=(1,)
            ),
            trade_returns=None,
        )


def test_basic_robustness_run():
    result = integrate_robustness(
        backtest_result=make_backtest_result(),
        trade_returns=make_trade_return_map(),
        config=make_config(),
    )

    assert isinstance(
        result,
        MultiHorizonRobustnessIntegrationResult,
    )


def test_each_horizon_is_evaluated_independently():
    horizons = (1, 3, 5)

    result = integrate_robustness(
        backtest_result=make_backtest_result(
            horizons=horizons
        ),
        trade_returns=make_trade_return_map(
            horizons
        ),
        config=make_config(
            horizons=horizons
        ),
    )

    assert set(
        result.evaluated_horizons
    ) == set(horizons)

    assert set(
        result.results.keys()
    ) == set(horizons)


def test_horizon_result_type():
    result = integrate_robustness(
        backtest_result=make_backtest_result(
            horizons=(5,)
        ),
        trade_returns={
            5: make_trade_returns()
        },
        config=make_config(
            horizons=(5,)
        ),
    )

    assert isinstance(
        result.get(5),
        HorizonRobustnessResult,
    )


def test_horizon_identity_is_preserved():
    result = integrate_robustness(
        backtest_result=make_backtest_result(
            horizons=(1, 5)
        ),
        trade_returns=make_trade_return_map(
            horizons=(1, 5)
        ),
        config=make_config(
            horizons=(1, 5)
        ),
    )

    for horizon in (1, 5):
        assert (
            result.get(horizon).horizon
            == horizon
        )


def test_trade_count_is_recorded():
    returns = make_trade_returns(
        n=75
    )

    result = integrate_robustness(
        backtest_result=make_backtest_result(
            horizons=(5,)
        ),
        trade_returns={
            5: returns
        },
        config=make_config(
            horizons=(5,)
        ),
    )

    assert (
        result.get(5).trade_count
        == 75
    )


def test_robustness_score_is_between_zero_and_one():
    result = integrate_robustness(
        backtest_result=make_backtest_result(
            horizons=(5,)
        ),
        trade_returns={
            5: make_trade_returns()
        },
        config=make_config(
            horizons=(5,)
        ),
    )

    score = result.get(
        5
    ).robustness_score

    if score is not None:
        assert (
            0.0
            <= score
            <= 1.0
        )


def test_monte_carlo_profit_probability_is_valid():
    result = integrate_robustness(
        backtest_result=make_backtest_result(
            horizons=(5,)
        ),
        trade_returns={
            5: make_trade_returns()
        },
        config=make_config(
            horizons=(5,)
        ),
    )

    probability = (
        result.get(5)
        .monte_carlo_profit_probability
    )

    if probability is not None:
        assert (
            0.0
            <= probability
            <= 1.0
        )


def test_monte_carlo_drawdown_probability_is_valid():
    result = integrate_robustness(
        backtest_result=make_backtest_result(
            horizons=(5,)
        ),
        trade_returns={
            5: make_trade_returns()
        },
        config=make_config(
            horizons=(5,)
        ),
    )

    probability = (
        result.get(5)
        .monte_carlo_drawdown_probability
    )

    if probability is not None:
        assert (
            0.0
            <= probability
            <= 1.0
        )


def test_missing_trade_returns_are_fail_closed():
    horizons = (1, 5)

    result = integrate_robustness(
        backtest_result=make_backtest_result(
            horizons=horizons
        ),
        trade_returns={
            1: make_trade_returns()
        },
        config=make_config(
            horizons=horizons
        ),
    )

    assert 5 in (
        result.failed_horizons
    )

    assert any(
        "5D" in error
        for error in result.errors
    )


def test_missing_backtest_result_is_fail_closed():
    backtest = make_backtest_result(
        horizons=(1,)
    )

    result = integrate_robustness(
        backtest_result=backtest,
        trade_returns={
            1: make_trade_returns()
        },
        config=make_config(
            horizons=(1, 5)
        ),
    )

    assert 5 in (
        result.failed_horizons
    )


def test_failed_backtest_can_block_robustness():
    backtest = make_backtest_result(
        horizons=(5,),
        passed=False,
    )

    result = integrate_robustness(
        backtest_result=backtest,
        trade_returns={
            5: make_trade_returns()
        },
        config=make_config(
            horizons=(5,),
            require_backtest_pass=True,
        ),
    )

    assert 5 in (
        result.failed_horizons
    )

    assert 5 not in (
        result.evaluated_horizons
    )


def test_failed_backtest_can_be_allowed():
    backtest = make_backtest_result(
        horizons=(5,),
        passed=False,
    )

    result = integrate_robustness(
        backtest_result=backtest,
        trade_returns={
            5: make_trade_returns()
        },
        config=make_config(
            horizons=(5,),
            require_backtest_pass=False,
        ),
    )

    assert 5 in (
        result.evaluated_horizons
    )


def test_short_trade_history_generates_warning():
    returns = make_trade_returns(
        n=10
    )

    result = integrate_robustness(
        backtest_result=make_backtest_result(
            horizons=(5,)
        ),
        trade_returns={
            5: returns
        },
        config=make_config(
            horizons=(5,),
            minimum_trades=30,
        ),
    )

    assert result.get(
        5
    ).warnings


def test_list_trade_returns_are_supported():
    result = integrate_robustness(
        backtest_result=make_backtest_result(
            horizons=(5,)
        ),
        trade_returns={
            5: list(
                make_trade_returns()
            )
        },
        config=make_config(
            horizons=(5,)
        ),
    )

    assert 5 in (
        result.evaluated_horizons
    )


def test_numpy_trade_returns_are_supported():
    result = integrate_robustness(
        backtest_result=make_backtest_result(
            horizons=(5,)
        ),
        trade_returns={
            5: make_trade_returns().to_numpy()
        },
        config=make_config(
            horizons=(5,)
        ),
    )

    assert 5 in (
        result.evaluated_horizons
    )


def test_invalid_trade_return_values_are_rejected():
    pipeline = RobustnessIntegration(
        config=make_config(
            horizons=(5,)
        )
    )

    with pytest.raises(
        (TypeError, ValueError)
    ):
        pipeline._validate_returns(
            ["bad", "value"],
            5,
        )


def test_nan_trade_returns_are_rejected():
    pipeline = RobustnessIntegration(
        config=make_config(
            horizons=(5,)
        )
    )

    with pytest.raises(ValueError):
        pipeline._validate_returns(
            pd.Series(
                [0.01, np.nan, 0.02]
            ),
            5,
        )


def test_infinite_trade_returns_are_rejected():
    pipeline = RobustnessIntegration(
        config=make_config(
            horizons=(5,)
        )
    )

    with pytest.raises(ValueError):
        pipeline._validate_returns(
            pd.Series(
                [0.01, np.inf, 0.02]
            ),
            5,
        )


def test_trade_order_is_preserved_by_input():
    returns = pd.Series(
        [
            0.10,
            -0.05,
            0.03,
            -0.08,
            0.06,
        ],
        name="Trade_Return",
    )

    original = returns.copy(
        deep=True
    )

    pipeline = RobustnessIntegration(
        config=make_config(
            horizons=(5,),
            minimum_trades=1,
        )
    )

    validated = pipeline._validate_returns(
        returns,
        5,
    )

    pd.testing.assert_series_equal(
        validated,
        original,
    )


def test_input_trade_returns_are_not_mutated():
    trade_returns = make_trade_return_map(
        horizons=(1, 5)
    )

    originals = {
        horizon: values.copy(
            deep=True
        )
        for horizon, values
        in trade_returns.items()
    }

    integrate_robustness(
        backtest_result=make_backtest_result(
            horizons=(1, 5)
        ),
        trade_returns=trade_returns,
        config=make_config(
            horizons=(1, 5)
        ),
    )

    for horizon in trade_returns:
        pd.testing.assert_series_equal(
            trade_returns[horizon],
            originals[horizon],
        )


def test_final_holdout_is_always_false():
    result = integrate_robustness(
        backtest_result=make_backtest_result(
            horizons=(1, 3, 5)
        ),
        trade_returns=make_trade_return_map(),
        config=make_config(),
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

    for item in result.results.values():
        assert (
            item.metadata[
                "final_holdout_used"
            ]
            is False
        )


def test_model_fitting_does_not_occur():
    result = integrate_robustness(
        backtest_result=make_backtest_result(
            horizons=(5,)
        ),
        trade_returns={
            5: make_trade_returns()
        },
        config=make_config(
            horizons=(5,)
        ),
    )

    assert (
        result.metadata[
            "model_fitted_here"
        ]
        is False
    )


def test_threshold_optimization_does_not_occur():
    result = integrate_robustness(
        backtest_result=make_backtest_result(
            horizons=(5,)
        ),
        trade_returns={
            5: make_trade_returns()
        },
        config=make_config(
            horizons=(5,)
        ),
    )

    assert (
        result.metadata[
            "threshold_optimized_here"
        ]
        is False
    )


def test_trade_order_metadata_is_true():
    result = integrate_robustness(
        backtest_result=make_backtest_result(
            horizons=(5,)
        ),
        trade_returns={
            5: make_trade_returns()
        },
        config=make_config(
            horizons=(5,)
        ),
    )

    assert (
        result.metadata[
            "trade_order_preserved"
        ]
        is True
    )


def test_research_only_metadata():
    result = integrate_robustness(
        backtest_result=make_backtest_result(
            horizons=(5,)
        ),
        trade_returns={
            5: make_trade_returns()
        },
        config=make_config(
            horizons=(5,)
        ),
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


def test_production_ready_is_false():
    result = integrate_robustness(
        backtest_result=make_backtest_result(
            horizons=(5,)
        ),
        trade_returns={
            5: make_trade_returns()
        },
        config=make_config(
            horizons=(5,)
        ),
    )

    assert (
        result.production_ready
        is False
    )


def test_candidate_gate_requires_passed_horizon():
    result = integrate_robustness(
        backtest_result=make_backtest_result(
            horizons=(5,),
            passed=True,
        ),
        trade_returns={
            5: make_trade_returns()
        },
        config=make_config(
            horizons=(5,),
            minimum_robustness_score=0.0,
        ),
    )

    if result.get(5).passed:
        assert 5 in (
            result.passed_horizons
        )


def test_summary_returns_dictionary():
    result = integrate_robustness(
        backtest_result=make_backtest_result(
            horizons=(1, 5)
        ),
        trade_returns=make_trade_return_map(
            horizons=(1, 5)
        ),
        config=make_config(
            horizons=(1, 5)
        ),
    )

    summary = robustness_integration_summary(
        result
    )

    assert isinstance(
        summary,
        dict,
    )


def test_summary_contains_horizons():
    result = integrate_robustness(
        backtest_result=make_backtest_result(
            horizons=(1, 5)
        ),
        trade_returns=make_trade_return_map(
            horizons=(1, 5)
        ),
        config=make_config(
            horizons=(1, 5)
        ),
    )

    summary = robustness_integration_summary(
        result
    )

    assert (
        summary["horizons"]
        == [1, 5]
    )


def test_summary_contains_horizon_results():
    result = integrate_robustness(
        backtest_result=make_backtest_result(
            horizons=(1, 5)
        ),
        trade_returns=make_trade_return_map(
            horizons=(1, 5)
        ),
        config=make_config(
            horizons=(1, 5)
        ),
    )

    summary = robustness_integration_summary(
        result
    )

    assert (
        "horizon_results"
        in summary
    )

    assert (
        "1"
        in summary["horizon_results"]
    )

    assert (
        "5"
        in summary["horizon_results"]
    )


def test_summary_is_research_only():
    result = integrate_robustness(
        backtest_result=make_backtest_result(
            horizons=(5,)
        ),
        trade_returns={
            5: make_trade_returns()
        },
        config=make_config(
            horizons=(5,)
        ),
    )

    summary = robustness_integration_summary(
        result
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
        robustness_integration_summary(
            None
        )


def test_get_unknown_horizon_fails():
    result = integrate_robustness(
        backtest_result=make_backtest_result(
            horizons=(1,)
        ),
        trade_returns={
            1: make_trade_returns()
        },
        config=make_config(
            horizons=(1,)
        ),
    )

    with pytest.raises(KeyError):
        result.get(20)


def test_deterministic_with_same_seed():
    backtest = make_backtest_result(
        horizons=(5,)
    )

    returns = {
        5: make_trade_returns(
            seed=123
        )
    }

    config = make_config(
        horizons=(5,),
        random_state=42,
    )

    first = integrate_robustness(
        backtest_result=backtest,
        trade_returns=returns,
        config=config,
    )

    second = integrate_robustness(
        backtest_result=backtest,
        trade_returns=returns,
        config=config,
    )

    first_item = first.get(5)
    second_item = second.get(5)

    assert (
        first_item.passed
        == second_item.passed
    )

    assert (
        first_item.trade_count
        == second_item.trade_count
    )

    if (
        first_item.robustness_score
        is not None
        and second_item.robustness_score
        is not None
    ):
        assert np.isclose(
            first_item.robustness_score,
            second_item.robustness_score,
        )


def test_different_horizons_use_different_trade_returns():
    returns = {
        1: pd.Series(
            [0.20] * 50
        ),
        5: pd.Series(
            [-0.10] * 50
        ),
    }

    result = integrate_robustness(
        backtest_result=make_backtest_result(
            horizons=(1, 5)
        ),
        trade_returns=returns,
        config=make_config(
            horizons=(1, 5)
        ),
    )

    first = result.get(1)
    second = result.get(5)

    assert first.trade_count == 50
    assert second.trade_count == 50

    if (
        first.robustness_score is not None
        and second.robustness_score is not None
    ):
        assert (
            first.robustness_score
            != second.robustness_score
            or first.passed
            != second.passed
        )


def test_result_type_is_correct():
    result = integrate_robustness(
        backtest_result=make_backtest_result(
            horizons=(1,)
        ),
        trade_returns={
            1: make_trade_returns()
        },
        config=make_config(
            horizons=(1,)
        ),
    )

    assert isinstance(
        result,
        MultiHorizonRobustnessIntegrationResult,
    )


def test_horizon_result_metadata_is_research_only():
    result = integrate_robustness(
        backtest_result=make_backtest_result(
            horizons=(5,)
        ),
        trade_returns={
            5: make_trade_returns()
        },
        config=make_config(
            horizons=(5,)
        ),
    )

    item = result.get(5)

    assert (
        item.metadata[
            "research_only"
        ]
        is True
    )

    assert (
        item.metadata[
            "production_approved"
        ]
        is False
    )


def test_no_holdout_approval_shortcut():
    result = integrate_robustness(
        backtest_result=make_backtest_result(
            horizons=(1, 3, 5)
        ),
        trade_returns=make_trade_return_map(),
        config=make_config(),
    )

    assert (
        result.candidate_passed
        is False
        or result.production_ready
        is False
    )
