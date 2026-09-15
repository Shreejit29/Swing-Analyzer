"""
Tests for regime-aware research evaluation.

These tests verify:

- trend regime construction
- volatility regime construction
- combined regimes
- chronological behaviour
- future-information resistance
- regime-specific evaluation
- stability scoring
- final-holdout protection
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.regime_validation import (
    RegimeValidationConfig,
)
from src.research.regime_research import (
    RegimeResearchEngine,
    RegimeResearchResult,
    run_regime_research,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def price_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)

    index = pd.date_range(
        "2020-01-01",
        periods=500,
        freq="D",
    )

    # A deterministic but varying price path.
    returns = rng.normal(
        0.001,
        0.015,
        len(index),
    )

    close = (
        100
        * np.cumprod(
            1 + returns
        )
    )

    return pd.DataFrame(
        {
            "Close": close,
        },
        index=index,
    )


@pytest.fixture
def predictions_data():
    rng = np.random.default_rng(42)

    index = pd.date_range(
        "2020-01-01",
        periods=400,
        freq="D",
    )

    actuals = rng.integers(
        0,
        2,
        size=len(index),
    )

    predictions = actuals.copy()

    # Add controlled errors.
    error_indices = rng.choice(
        len(index),
        size=80,
        replace=False,
    )

    predictions[
        error_indices
    ] = 1 - predictions[
        error_indices
    ]

    probabilities = np.where(
        predictions == 1,
        0.75,
        0.25,
    )

    return (
        actuals,
        predictions,
        probabilities,
        index,
    )


@pytest.fixture
def regime_config():
    return RegimeValidationConfig(
        minimum_samples=10,
        minimum_accuracy=0.50,
        maximum_accuracy_std=0.50,
    )


@pytest.fixture
def regime_data(
    price_data,
):
    engine = RegimeResearchEngine()

    return engine.add_all_regimes(
        price_data
    )


# ---------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------


def test_engine_can_be_created():
    engine = RegimeResearchEngine()

    assert engine is not None


def test_custom_config_is_accepted(
    regime_config,
):
    engine = RegimeResearchEngine(
        regime_config
    )

    assert (
        engine.config
        is regime_config
    )


def test_invalid_minimum_samples_fails():
    with pytest.raises(ValueError):
        RegimeResearchEngine(
            RegimeValidationConfig(
                minimum_samples=0
            )
        )


def test_invalid_minimum_accuracy_fails():
    with pytest.raises(ValueError):
        RegimeResearchEngine(
            RegimeValidationConfig(
                minimum_accuracy=1.5
            )
        )


def test_negative_accuracy_std_fails():
    with pytest.raises(ValueError):
        RegimeResearchEngine(
            RegimeValidationConfig(
                maximum_accuracy_std=-0.1
            )
        )


# ---------------------------------------------------------------------
# Trend regime
# ---------------------------------------------------------------------


def test_trend_regime_requires_close():
    data = pd.DataFrame(
        {
            "Open": [1, 2, 3],
        }
    )

    with pytest.raises(ValueError):
        RegimeResearchEngine.add_trend_regime(
            data,
            fast_period=2,
            slow_period=3,
        )


def test_trend_regime_is_created(
    price_data,
):
    result = (
        RegimeResearchEngine.add_trend_regime(
            price_data,
            fast_period=20,
            slow_period=50,
        )
    )

    assert (
        "Trend_Regime"
        in result.columns
    )


def test_trend_regime_values_are_valid(
    price_data,
):
    result = (
        RegimeResearchEngine.add_trend_regime(
            price_data,
            fast_period=20,
            slow_period=50,
        )
    )

    valid = {
        "BULL",
        "BEAR",
        "NEUTRAL",
        "UNKNOWN",
    }

    assert set(
        result[
            "Trend_Regime"
        ].unique()
    ).issubset(valid)


def test_trend_regime_does_not_modify_input(
    price_data,
):
    original = price_data.copy(
        deep=True
    )

    RegimeResearchEngine.add_trend_regime(
        price_data,
        fast_period=20,
        slow_period=50,
    )

    pd.testing.assert_frame_equal(
        price_data,
        original,
    )


def test_trend_regime_rejects_invalid_periods(
    price_data,
):
    with pytest.raises(ValueError):
        RegimeResearchEngine.add_trend_regime(
            price_data,
            fast_period=0,
            slow_period=50,
        )

    with pytest.raises(ValueError):
        RegimeResearchEngine.add_trend_regime(
            price_data,
            fast_period=50,
            slow_period=20,
        )


# ---------------------------------------------------------------------
# Volatility regime
# ---------------------------------------------------------------------


def test_volatility_regime_requires_close():
    data = pd.DataFrame(
        {
            "Open": [1, 2, 3],
        }
    )

    with pytest.raises(ValueError):
        RegimeResearchEngine.add_volatility_regime(
            data
        )


def test_volatility_regime_is_created(
    price_data,
):
    result = (
        RegimeResearchEngine.add_volatility_regime(
            price_data,
            return_period=20,
        )
    )

    assert (
        "Volatility_Regime"
        in result.columns
    )

    assert (
        "Regime_Volatility"
        in result.columns
    )


def test_volatility_regime_values_are_valid(
    price_data,
):
    result = (
        RegimeResearchEngine.add_volatility_regime(
            price_data,
            return_period=20,
        )
    )

    valid = {
        "LOW",
        "NORMAL",
        "HIGH",
        "UNKNOWN",
    }

    assert set(
        result[
            "Volatility_Regime"
        ].unique()
    ).issubset(valid)


def test_volatility_regime_does_not_modify_input(
    price_data,
):
    original = price_data.copy(
        deep=True
    )

    RegimeResearchEngine.add_volatility_regime(
        price_data
    )

    pd.testing.assert_frame_equal(
        price_data,
        original,
    )


def test_invalid_volatility_period_fails(
    price_data,
):
    with pytest.raises(ValueError):
        RegimeResearchEngine.add_volatility_regime(
            price_data,
            return_period=1,
        )


def test_invalid_volatility_quantiles_fail(
    price_data,
):
    with pytest.raises(ValueError):
        RegimeResearchEngine.add_volatility_regime(
            price_data,
            low_quantile=0.8,
            high_quantile=0.2,
        )


# ---------------------------------------------------------------------
# Combined regimes
# ---------------------------------------------------------------------


def test_all_regimes_are_created(
    price_data,
):
    result = (
        RegimeResearchEngine.add_all_regimes(
            price_data
        )
    )

    required = {
        "Trend_Regime",
        "Volatility_Regime",
        "Combined_Regime",
    }

    assert required.issubset(
        result.columns
    )


def test_combined_regime_is_consistent(
    price_data,
):
    result = (
        RegimeResearchEngine.add_all_regimes(
            price_data
        )
    )

    expected = (
        result["Trend_Regime"].astype(str)
        + "_"
        + result["Volatility_Regime"].astype(str)
    )

    pd.testing.assert_series_equal(
        result["Combined_Regime"],
        expected,
        check_names=False,
    )


# ---------------------------------------------------------------------
# Future information resistance
# ---------------------------------------------------------------------


def test_future_price_mutation_does_not_change_past_trend_regime(
    price_data,
):
    original = (
        RegimeResearchEngine.add_trend_regime(
            price_data,
            fast_period=20,
            slow_period=50,
        )
    )

    mutated = price_data.copy()

    cutoff = 300

    mutated.iloc[
        cutoff:,
        mutated.columns.get_loc(
            "Close"
        ),
    ] *= 10.0

    changed = (
        RegimeResearchEngine.add_trend_regime(
            mutated,
            fast_period=20,
            slow_period=50,
        )
    )

    pd.testing.assert_series_equal(
        original[
            "Trend_Regime"
        ].iloc[:cutoff],
        changed[
            "Trend_Regime"
        ].iloc[:cutoff],
        check_names=False,
    )


def test_future_price_mutation_does_not_change_past_volatility_regime(
    price_data,
):
    original = (
        RegimeResearchEngine.add_volatility_regime(
            price_data,
            return_period=20,
        )
    )

    mutated = price_data.copy()

    cutoff = 300

    mutated.iloc[
        cutoff:,
        mutated.columns.get_loc(
            "Close"
        ),
    ] *= 10.0

    changed = (
        RegimeResearchEngine.add_volatility_regime(
            mutated,
            return_period=20,
        )
    )

    pd.testing.assert_series_equal(
        original[
            "Volatility_Regime"
        ].iloc[:cutoff],
        changed[
            "Volatility_Regime"
        ].iloc[:cutoff],
        check_names=False,
    )


def test_future_price_mutation_does_not_change_past_volatility_values(
    price_data,
):
    original = (
        RegimeResearchEngine.add_volatility_regime(
            price_data,
            return_period=20,
        )
    )

    mutated = price_data.copy()

    cutoff = 300

    mutated.iloc[
        cutoff:,
        mutated.columns.get_loc(
            "Close"
        ),
    ] *= 10.0

    changed = (
        RegimeResearchEngine.add_volatility_regime(
            mutated,
            return_period=20,
        )
    )

    pd.testing.assert_series_equal(
        original[
            "Regime_Volatility"
        ].iloc[:cutoff],
        changed[
            "Regime_Volatility"
        ].iloc[:cutoff],
        check_names=False,
    )


# ---------------------------------------------------------------------
# Prediction validation
# ---------------------------------------------------------------------


def test_prediction_length_mismatch_fails(
    regime_data,
):
    engine = RegimeResearchEngine(
        RegimeValidationConfig(
            minimum_samples=5
        )
    )

    timestamps = regime_data.index

    actuals = np.zeros(
        len(timestamps),
        dtype=int,
    )

    predictions = np.zeros(
        len(timestamps) - 1,
        dtype=int,
    )

    with pytest.raises(ValueError):
        engine.evaluate(
            actuals=actuals,
            predictions=predictions,
            timestamps=timestamps,
            regime_data=regime_data,
        )


def test_probability_length_mismatch_fails(
    regime_data,
):
    engine = RegimeResearchEngine(
        RegimeValidationConfig(
            minimum_samples=5
        )
    )

    timestamps = regime_data.index

    actuals = np.zeros(
        len(timestamps),
        dtype=int,
    )

    predictions = actuals.copy()

    probability = np.zeros(
        len(timestamps) - 1
    )

    with pytest.raises(ValueError):
        engine.evaluate(
            actuals=actuals,
            predictions=predictions,
            timestamps=timestamps,
            regime_data=regime_data,
            probability=probability,
        )


def test_probability_outside_range_fails(
    regime_data,
):
    engine = RegimeResearchEngine(
        RegimeValidationConfig(
            minimum_samples=5
        )
    )

    timestamps = regime_data.index

    actuals = np.tile(
        [0, 1],
        len(timestamps) // 2 + 1,
    )[: len(timestamps)]

    predictions = actuals.copy()

    probability = np.full(
        len(timestamps),
        0.5,
    )

    probability[0] = 2.0

    with pytest.raises(ValueError):
        engine.evaluate(
            actuals=actuals,
            predictions=predictions,
            timestamps=timestamps,
            regime_data=regime_data,
            probability=probability,
        )


def test_duplicate_prediction_timestamps_fail(
    regime_data,
):
    engine = RegimeResearchEngine(
        RegimeValidationConfig(
            minimum_samples=5
        )
    )

    timestamps = pd.DatetimeIndex(
        list(regime_data.index[:-1])
        + [regime_data.index[-2]]
    )

    actuals = np.tile(
        [0, 1],
        len(timestamps) // 2 + 1,
    )[: len(timestamps)]

    predictions = actuals.copy()

    with pytest.raises(ValueError):
        engine.evaluate(
            actuals=actuals,
            predictions=predictions,
            timestamps=timestamps,
            regime_data=regime_data.iloc[
                : len(timestamps)
            ],
        )


# ---------------------------------------------------------------------
# Successful evaluation
# ---------------------------------------------------------------------


def test_evaluation_returns_result(
    predictions_data,
    regime_config,
    price_data,
):
    actuals, predictions, probabilities, index = (
        predictions_data
    )

    regime_data = (
        RegimeResearchEngine.add_all_regimes(
            price_data.loc[index]
        )
    )

    result = run_regime_research(
        actuals=actuals,
        predictions=predictions,
        timestamps=index,
        regime_data=regime_data,
        probability=probabilities,
        config=regime_config,
    )

    assert isinstance(
        result,
        RegimeResearchResult,
    )


def test_total_samples_are_recorded(
    predictions_data,
    regime_config,
    price_data,
):
    actuals, predictions, probabilities, index = (
        predictions_data
    )

    regime_data = (
        RegimeResearchEngine.add_all_regimes(
            price_data.loc[index]
        )
    )

    result = run_regime_research(
        actuals=actuals,
        predictions=predictions,
        timestamps=index,
        regime_data=regime_data,
        probability=probabilities,
        config=regime_config,
    )

    assert (
        result.total_samples
        == len(actuals)
    )


def test_regime_names_are_recorded(
    predictions_data,
    regime_config,
    price_data,
):
    actuals, predictions, probabilities, index = (
        predictions_data
    )

    regime_data = (
        RegimeResearchEngine.add_all_regimes(
            price_data.loc[index]
        )
    )

    result = run_regime_research(
        actuals=actuals,
        predictions=predictions,
        timestamps=index,
        regime_data=regime_data,
        probability=probabilities,
        config=regime_config,
    )

    assert isinstance(
        result.trend_regimes,
        list,
    )

    assert isinstance(
        result.volatility_regimes,
        list,
    )

    assert isinstance(
        result.combined_regimes,
        list,
    )


# ---------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------


def test_stability_score_is_bounded(
    predictions_data,
    regime_config,
    price_data,
):
    actuals, predictions, probabilities, index = (
        predictions_data
    )

    regime_data = (
        RegimeResearchEngine.add_all_regimes(
            price_data.loc[index]
        )
    )

    result = run_regime_research(
        actuals=actuals,
        predictions=predictions,
        timestamps=index,
        regime_data=regime_data,
        probability=probabilities,
        config=regime_config,
    )

    assert (
        0.0
        <= result.stability_score
        <= 1.0
    )


def test_stability_property_matches_gate(
    predictions_data,
    regime_config,
    price_data,
):
    actuals, predictions, probabilities, index = (
        predictions_data
    )

    regime_data = (
        RegimeResearchEngine.add_all_regimes(
            price_data.loc[index]
        )
    )

    result = run_regime_research(
        actuals=actuals,
        predictions=predictions,
        timestamps=index,
        regime_data=regime_data,
        probability=probabilities,
        config=regime_config,
    )

    assert (
        result.stable
        == result.passed_stability_gate
    )


# ---------------------------------------------------------------------
# Holdout safety
# ---------------------------------------------------------------------


def test_final_holdout_flag_is_rejected(
    predictions_data,
    regime_config,
    price_data,
):
    actuals, predictions, probabilities, index = (
        predictions_data
    )

    regime_data = (
        RegimeResearchEngine.add_all_regimes(
            price_data.loc[index]
        )
    )

    engine = RegimeResearchEngine(
        regime_config
    )

    with pytest.raises(ValueError):
        engine.evaluate(
            actuals=actuals,
            predictions=predictions,
            timestamps=index,
            regime_data=regime_data,
            probability=probabilities,
            final_holdout_used=True,
        )


def test_result_records_holdout_unused(
    predictions_data,
    regime_config,
    price_data,
):
    actuals, predictions, probabilities, index = (
        predictions_data
    )

    regime_data = (
        RegimeResearchEngine.add_all_regimes(
            price_data.loc[index]
        )
    )

    result = run_regime_research(
        actuals=actuals,
        predictions=predictions,
        timestamps=index,
        regime_data=regime_data,
        probability=probabilities,
        config=regime_config,
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
# Input immutability
# ---------------------------------------------------------------------


def test_regime_data_is_not_modified(
    predictions_data,
    regime_config,
    price_data,
):
    actuals, predictions, probabilities, index = (
        predictions_data
    )

    regime_data = (
        RegimeResearchEngine.add_all_regimes(
            price_data.loc[index]
        )
    )

    original = regime_data.copy(
        deep=True
    )

    run_regime_research(
        actuals=actuals,
        predictions=predictions,
        timestamps=index,
        regime_data=regime_data,
        probability=probabilities,
        config=regime_config,
    )

    pd.testing.assert_frame_equal(
        regime_data,
        original,
    )


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------


def test_summary_contains_key_fields(
    predictions_data,
    regime_config,
    price_data,
):
    actuals, predictions, probabilities, index = (
        predictions_data
    )

    regime_data = (
        RegimeResearchEngine.add_all_regimes(
            price_data.loc[index]
        )
    )

    result = run_regime_research(
        actuals=actuals,
        predictions=predictions,
        timestamps=index,
        regime_data=regime_data,
        probability=probabilities,
        config=regime_config,
    )

    summary = result.summary()

    required = {
        "total_samples",
        "trend_regimes",
        "volatility_regimes",
        "combined_regimes",
        "minimum_regime_accuracy",
        "maximum_regime_accuracy",
        "regime_accuracy_std",
        "stability_score",
        "passed_stability_gate",
        "final_holdout_used",
    }

    assert required.issubset(
        summary.keys()
    )


# ---------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------


def test_regime_research_is_deterministic(
    predictions_data,
    regime_config,
    price_data,
):
    actuals, predictions, probabilities, index = (
        predictions_data
    )

    regime_data = (
        RegimeResearchEngine.add_all_regimes(
            price_data.loc[index]
        )
    )

    result_a = run_regime_research(
        actuals=actuals,
        predictions=predictions,
        timestamps=index,
        regime_data=regime_data,
        probability=probabilities,
        config=regime_config,
    )

    result_b = run_regime_research(
        actuals=actuals,
        predictions=predictions,
        timestamps=index,
        regime_data=regime_data,
        probability=probabilities,
        config=regime_config,
    )

    assert (
        result_a.stability_score
        == result_b.stability_score
    )

    assert (
        result_a.minimum_regime_accuracy
        == result_b.minimum_regime_accuracy
    )

    assert (
        result_a.regime_accuracy_std
        == result_b.regime_accuracy_std
    )
