"""
Tests for probabilistic target-price range research.

The tests verify:

- valid range evaluation
- lower <= median <= upper
- interval coverage calculation
- interval width calculation
- conversion from returns to prices
- invalid-input protection
- final-holdout protection
- input immutability
- deterministic results
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.range_validation import (
    RangeValidationConfig,
)
from src.research.range_research import (
    RangeResearchEngine,
    RangeResearchResult,
    run_range_research,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def range_data():
    timestamps = pd.date_range(
        "2020-01-01",
        periods=200,
        freq="D",
    )

    actual = np.linspace(
        100.0,
        120.0,
        len(timestamps),
    )

    lower = actual - 5.0
    median = actual
    upper = actual + 5.0

    return (
        actual,
        lower,
        median,
        upper,
        timestamps,
    )


@pytest.fixture
def config():
    return RangeValidationConfig(
        expected_coverage=0.80,
        minimum_coverage=0.70,
        maximum_coverage=0.95,
        minimum_samples=50,
    )


@pytest.fixture
def engine(config):
    return RangeResearchEngine(
        config=config
    )


# ---------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------


def test_engine_can_be_created():
    engine = RangeResearchEngine()

    assert engine is not None


def test_custom_config_is_preserved(
    config,
):
    engine = RangeResearchEngine(
        config
    )

    assert engine.config is config


def test_invalid_coverage_configuration_fails():
    with pytest.raises(ValueError):
        RangeResearchEngine(
            RangeValidationConfig(
                expected_coverage=0.50,
                minimum_coverage=0.80,
                maximum_coverage=0.95,
            )
        )


def test_invalid_minimum_samples_fails():
    with pytest.raises(ValueError):
        RangeResearchEngine(
            RangeValidationConfig(
                minimum_samples=0
            )
        )


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------


def test_empty_range_data_fails(
    engine,
):
    timestamps = pd.DatetimeIndex([])

    with pytest.raises(ValueError):
        engine.evaluate(
            actual=np.array([]),
            lower=np.array([]),
            median=np.array([]),
            upper=np.array([]),
            timestamps=timestamps,
            horizon=5,
        )


def test_length_mismatch_fails(
    engine,
    range_data,
):
    actual, lower, median, upper, timestamps = (
        range_data
    )

    with pytest.raises(ValueError):
        engine.evaluate(
            actual=actual[:-1],
            lower=lower,
            median=median,
            upper=upper,
            timestamps=timestamps,
            horizon=5,
        )


def test_lower_greater_than_median_fails(
    engine,
    range_data,
):
    actual, lower, median, upper, timestamps = (
        range_data
    )

    lower = lower.copy()
    lower[0] = median[0] + 10.0

    with pytest.raises(ValueError):
        engine.evaluate(
            actual=actual,
            lower=lower,
            median=median,
            upper=upper,
            timestamps=timestamps,
            horizon=5,
        )


def test_median_greater_than_upper_fails(
    engine,
    range_data,
):
    actual, lower, median, upper, timestamps = (
        range_data
    )

    upper = upper.copy()
    upper[0] = median[0] - 10.0

    with pytest.raises(ValueError):
        engine.evaluate(
            actual=actual,
            lower=lower,
            median=median,
            upper=upper,
            timestamps=timestamps,
            horizon=5,
        )


def test_nonfinite_actual_fails(
    engine,
    range_data,
):
    actual, lower, median, upper, timestamps = (
        range_data
    )

    actual = actual.copy()
    actual[0] = np.nan

    with pytest.raises(ValueError):
        engine.evaluate(
            actual=actual,
            lower=lower,
            median=median,
            upper=upper,
            timestamps=timestamps,
            horizon=5,
        )


def test_nonfinite_range_prediction_fails(
    engine,
    range_data,
):
    actual, lower, median, upper, timestamps = (
        range_data
    )

    upper = upper.copy()
    upper[0] = np.inf

    with pytest.raises(ValueError):
        engine.evaluate(
            actual=actual,
            lower=lower,
            median=median,
            upper=upper,
            timestamps=timestamps,
            horizon=5,
        )


def test_duplicate_timestamps_fail(
    engine,
    range_data,
):
    actual, lower, median, upper, timestamps = (
        range_data
    )

    timestamps = pd.DatetimeIndex(
        list(timestamps[:-1])
        + [timestamps[-2]]
    )

    with pytest.raises(ValueError):
        engine.evaluate(
            actual=actual,
            lower=lower,
            median=median,
            upper=upper,
            timestamps=timestamps,
            horizon=5,
        )


def test_unsorted_timestamps_fail(
    engine,
    range_data,
):
    actual, lower, median, upper, timestamps = (
        range_data
    )

    with pytest.raises(ValueError):
        engine.evaluate(
            actual=actual,
            lower=lower,
            median=median,
            upper=upper,
            timestamps=timestamps[::-1],
            horizon=5,
        )


def test_non_positive_horizon_fails(
    engine,
    range_data,
):
    actual, lower, median, upper, timestamps = (
        range_data
    )

    with pytest.raises(ValueError):
        engine.evaluate(
            actual=actual,
            lower=lower,
            median=median,
            upper=upper,
            timestamps=timestamps,
            horizon=0,
        )


# ---------------------------------------------------------------------
# Minimum samples
# ---------------------------------------------------------------------


def test_too_few_samples_fail(
    range_data,
):
    actual, lower, median, upper, timestamps = (
        range_data
    )

    engine = RangeResearchEngine(
        RangeValidationConfig(
            minimum_samples=500
        )
    )

    with pytest.raises(ValueError):
        engine.evaluate(
            actual=actual,
            lower=lower,
            median=median,
            upper=upper,
            timestamps=timestamps,
            horizon=5,
        )


# ---------------------------------------------------------------------
# Successful evaluation
# ---------------------------------------------------------------------


def test_evaluation_returns_result(
    engine,
    range_data,
):
    actual, lower, median, upper, timestamps = (
        range_data
    )

    result = engine.evaluate(
        actual=actual,
        lower=lower,
        median=median,
        upper=upper,
        timestamps=timestamps,
        horizon=5,
    )

    assert isinstance(
        result,
        RangeResearchResult,
    )


def test_sample_count_is_correct(
    engine,
    range_data,
):
    actual, lower, median, upper, timestamps = (
        range_data
    )

    result = engine.evaluate(
        actual=actual,
        lower=lower,
        median=median,
        upper=upper,
        timestamps=timestamps,
        horizon=5,
    )

    assert (
        result.samples
        == len(actual)
    )


def test_horizon_is_recorded(
    engine,
    range_data,
):
    actual, lower, median, upper, timestamps = (
        range_data
    )

    result = engine.evaluate(
        actual=actual,
        lower=lower,
        median=median,
        upper=upper,
        timestamps=timestamps,
        horizon=10,
    )

    assert result.horizon == 10


# ---------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------


def test_perfect_containment_has_full_coverage(
    engine,
    range_data,
):
    actual, _, _, _, timestamps = (
        range_data
    )

    lower = actual - 10
    median = actual
    upper = actual + 10

    result = engine.evaluate(
        actual=actual,
        lower=lower,
        median=median,
        upper=upper,
        timestamps=timestamps,
        horizon=5,
    )

    assert (
        result.actual_coverage
        == 1.0
    )


def test_zero_containment_has_zero_coverage(
    engine,
    range_data,
):
    actual, _, _, _, timestamps = (
        range_data
    )

    lower = actual + 10
    median = actual + 15
    upper = actual + 20

    result = engine.evaluate(
        actual=actual,
        lower=lower,
        median=median,
        upper=upper,
        timestamps=timestamps,
        horizon=5,
    )

    assert (
        result.actual_coverage
        == 0.0
    )


def test_coverage_is_bounded(
    engine,
    range_data,
):
    actual, lower, median, upper, timestamps = (
        range_data
    )

    result = engine.evaluate(
        actual=actual,
        lower=lower,
        median=median,
        upper=upper,
        timestamps=timestamps,
        horizon=5,
    )

    assert (
        0.0
        <= result.actual_coverage
        <= 1.0
    )


# ---------------------------------------------------------------------
# Width
# ---------------------------------------------------------------------


def test_average_width_is_correct(
    engine,
    range_data,
):
    actual, _, _, _, timestamps = (
        range_data
    )

    lower = actual - 5
    median = actual
    upper = actual + 5

    result = engine.evaluate(
        actual=actual,
        lower=lower,
        median=median,
        upper=upper,
        timestamps=timestamps,
        horizon=5,
    )

    assert np.isclose(
        result.average_width,
        10.0,
    )


def test_median_width_is_correct(
    engine,
    range_data,
):
    actual, _, _, _, timestamps = (
        range_data
    )

    lower = actual - 7
    median = actual
    upper = actual + 3

    result = engine.evaluate(
        actual=actual,
        lower=lower,
        median=median,
        upper=upper,
        timestamps=timestamps,
        horizon=5,
    )

    assert np.isclose(
        result.median_width,
        10.0,
    )


# ---------------------------------------------------------------------
# Miss rates
# ---------------------------------------------------------------------


def test_lower_and_upper_miss_rates_are_bounded(
    engine,
    range_data,
):
    actual, lower, median, upper, timestamps = (
        range_data
    )

    result = engine.evaluate(
        actual=actual,
        lower=lower,
        median=median,
        upper=upper,
        timestamps=timestamps,
        horizon=5,
    )

    assert (
        0.0
        <= result.lower_miss_rate
        <= 1.0
    )

    assert (
        0.0
        <= result.upper_miss_rate
        <= 1.0
    )


def test_miss_rates_sum_with_coverage(
    engine,
    range_data,
):
    actual, lower, median, upper, timestamps = (
        range_data
    )

    result = engine.evaluate(
        actual=actual,
        lower=lower,
        median=median,
        upper=upper,
        timestamps=timestamps,
        horizon=5,
    )

    total = (
        result.actual_coverage
        + result.lower_miss_rate
        + result.upper_miss_rate
    )

    assert np.isclose(
        total,
        1.0,
    )


# ---------------------------------------------------------------------
# Holdout safety
# ---------------------------------------------------------------------


def test_final_holdout_is_rejected(
    engine,
    range_data,
):
    actual, lower, median, upper, timestamps = (
        range_data
    )

    with pytest.raises(ValueError):
        engine.evaluate(
            actual=actual,
            lower=lower,
            median=median,
            upper=upper,
            timestamps=timestamps,
            horizon=5,
            final_holdout_used=True,
        )


def test_successful_result_records_holdout_unused(
    engine,
    range_data,
):
    actual, lower, median, upper, timestamps = (
        range_data
    )

    result = engine.evaluate(
        actual=actual,
        lower=lower,
        median=median,
        upper=upper,
        timestamps=timestamps,
        horizon=5,
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
# Return-to-price conversion
# ---------------------------------------------------------------------


def test_returns_to_price_range():
    result = (
        RangeResearchEngine.returns_to_price_range(
            current_price=100.0,
            lower_return=-0.05,
            median_return=0.02,
            upper_return=0.10,
        )
    )

    lower, median, upper = result

    assert np.isclose(
        lower,
        95.0,
    )

    assert np.isclose(
        median,
        102.0,
    )

    assert np.isclose(
        upper,
        110.0,
    )


def test_returns_to_price_range_supports_arrays():
    current = np.array(
        [100.0, 200.0]
    )

    lower_return = np.array(
        [-0.05, -0.10]
    )

    median_return = np.array(
        [0.02, 0.05]
    )

    upper_return = np.array(
        [0.10, 0.15]
    )

    lower, median, upper = (
        RangeResearchEngine.returns_to_price_range(
            current,
            lower_return,
            median_return,
            upper_return,
        )
    )

    np.testing.assert_allclose(
        lower,
        [95.0, 180.0],
    )

    np.testing.assert_allclose(
        median,
        [102.0, 210.0],
    )

    np.testing.assert_allclose(
        upper,
        [110.0, 230.0],
    )


def test_non_positive_current_price_fails():
    with pytest.raises(ValueError):
        RangeResearchEngine.returns_to_price_range(
            current_price=0.0,
            lower_return=-0.05,
            median_return=0.0,
            upper_return=0.05,
        )


def test_nonfinite_current_price_fails():
    with pytest.raises(ValueError):
        RangeResearchEngine.returns_to_price_range(
            current_price=np.nan,
            lower_return=-0.05,
            median_return=0.0,
            upper_return=0.05,
        )


# ---------------------------------------------------------------------
# Range-width percentage
# ---------------------------------------------------------------------


def test_range_width_percentage():
    width = (
        RangeResearchEngine.range_width_percentage(
            lower_price=95.0,
            upper_price=105.0,
            current_price=100.0,
        )
    )

    assert np.isclose(
        width,
        0.10,
    )


def test_range_width_percentage_supports_arrays():
    width = (
        RangeResearchEngine.range_width_percentage(
            lower_price=np.array(
                [95.0, 190.0]
            ),
            upper_price=np.array(
                [105.0, 210.0]
            ),
            current_price=np.array(
                [100.0, 200.0]
            ),
        )
    )

    np.testing.assert_allclose(
        width,
        [0.10, 0.10],
    )


def test_range_width_rejects_invalid_order():
    with pytest.raises(ValueError):
        RangeResearchEngine.range_width_percentage(
            lower_price=110.0,
            upper_price=105.0,
            current_price=100.0,
        )


# ---------------------------------------------------------------------
# Input immutability
# ---------------------------------------------------------------------


def test_inputs_are_not_modified(
    engine,
    range_data,
):
    actual, lower, median, upper, timestamps = (
        range_data
    )

    actual_copy = actual.copy()
    lower_copy = lower.copy()
    median_copy = median.copy()
    upper_copy = upper.copy()
    timestamps_copy = timestamps.copy()

    engine.evaluate(
        actual=actual,
        lower=lower,
        median=median,
        upper=upper,
        timestamps=timestamps,
        horizon=5,
    )

    np.testing.assert_array_equal(
        actual,
        actual_copy,
    )

    np.testing.assert_array_equal(
        lower,
        lower_copy,
    )

    np.testing.assert_array_equal(
        median,
        median_copy,
    )

    np.testing.assert_array_equal(
        upper,
        upper_copy,
    )

    assert timestamps.equals(
        timestamps_copy
    )


# ---------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------


def test_compare_ranges(
    engine,
    range_data,
):
    actual, lower, median, upper, timestamps = (
        range_data
    )

    first = engine.evaluate(
        actual=actual,
        lower=lower - 2,
        median=median,
        upper=upper + 2,
        timestamps=timestamps,
        horizon=5,
    )

    second = engine.evaluate(
        actual=actual,
        lower=lower,
        median=median,
        upper=upper,
        timestamps=timestamps,
        horizon=5,
    )

    comparison = (
        engine.compare_ranges(
            first,
            second,
        )
    )

    assert (
        "coverage_difference"
        in comparison
    )

    assert (
        "average_width_difference"
        in comparison
    )

    assert (
        "stability_difference"
        in comparison
    )


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------


def test_summary_contains_key_fields(
    engine,
    range_data,
):
    actual, lower, median, upper, timestamps = (
        range_data
    )

    result = engine.evaluate(
        actual=actual,
        lower=lower,
        median=median,
        upper=upper,
        timestamps=timestamps,
        horizon=5,
    )

    summary = result.summary()

    required = {
        "horizon",
        "samples",
        "expected_coverage",
        "actual_coverage",
        "average_width",
        "median_width",
        "lower_miss_rate",
        "upper_miss_rate",
        "passed_coverage_gate",
        "passed_width_gate",
        "passed",
        "final_holdout_used",
    }

    assert required.issubset(
        summary.keys()
    )


# ---------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------


def test_convenience_api(
    range_data,
    config,
):
    actual, lower, median, upper, timestamps = (
        range_data
    )

    result = run_range_research(
        actual=actual,
        lower=lower,
        median=median,
        upper=upper,
        timestamps=timestamps,
        horizon=5,
        config=config,
    )

    assert isinstance(
        result,
        RangeResearchResult,
    )


# ---------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------


def test_range_research_is_deterministic(
    engine,
    range_data,
):
    actual, lower, median, upper, timestamps = (
        range_data
    )

    result_a = engine.evaluate(
        actual=actual,
        lower=lower,
        median=median,
        upper=upper,
        timestamps=timestamps,
        horizon=5,
    )

    result_b = engine.evaluate(
        actual=actual,
        lower=lower,
        median=median,
        upper=upper,
        timestamps=timestamps,
        horizon=5,
    )

    assert (
        result_a.actual_coverage
        == result_b.actual_coverage
    )

    assert (
        result_a.average_width
        == result_b.average_width
    )

    assert (
        result_a.lower_miss_rate
        == result_b.lower_miss_rate
    )

    assert (
        result_a.upper_miss_rate
        == result_b.upper_miss_rate
    )
