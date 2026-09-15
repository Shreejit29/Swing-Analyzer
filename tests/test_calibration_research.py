"""
Tests for probability calibration research.

Calibration must:

- use only a dedicated calibration period
- remain strictly after training
- never use the final holdout
- reject invalid probabilities
- reject one-class calibration data
- preserve chronological ordering
- expose calibration diagnostics
- provide deterministic results
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.calibration import CalibrationConfig
from src.research.calibration_research import (
    CalibrationResearchEngine,
    CalibrationResearchResult,
    calibrate_research_probabilities,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def calibration_data():
    rng = np.random.default_rng(42)

    timestamps = pd.date_range(
        "2024-01-01",
        periods=200,
        freq="D",
    )

    latent = rng.normal(
        0,
        1,
        len(timestamps),
    )

    probabilities = (
        1.0
        / (
            1.0
            + np.exp(
                -(
                    latent
                    + rng.normal(
                        0,
                        0.5,
                        len(timestamps),
                    )
                )
            )
        )
    )

    actuals = (
        probabilities
        + rng.normal(
            0,
            0.20,
            len(timestamps),
        )
        > 0.5
    ).astype(int)

    return (
        probabilities,
        actuals,
        timestamps,
    )


@pytest.fixture
def engine():
    return CalibrationResearchEngine(
        CalibrationConfig(
            method="sigmoid",
            min_samples=50,
            high_confidence_threshold=0.70,
        )
    )


# ---------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------


def test_engine_can_be_created():
    engine = CalibrationResearchEngine()

    assert engine is not None


def test_invalid_confidence_threshold_fails():
    with pytest.raises(ValueError):
        CalibrationResearchEngine(
            CalibrationConfig(
                high_confidence_threshold=1.5
            )
        )


def test_negative_confidence_threshold_fails():
    with pytest.raises(ValueError):
        CalibrationResearchEngine(
            CalibrationConfig(
                high_confidence_threshold=-0.1
            )
        )


# ---------------------------------------------------------------------
# Calibration split
# ---------------------------------------------------------------------


def test_calibration_split_is_chronological():
    rng = np.random.default_rng(1)

    index = pd.date_range(
        "2020-01-01",
        periods=100,
        freq="D",
    )

    data = pd.DataFrame(
        {
            "Feature": rng.normal(
                size=100
            )
        },
        index=index,
    )

    training, calibration = (
        CalibrationResearchEngine.create_calibration_split(
            data,
            calibration_fraction=0.20,
        )
    )

    assert (
        training.index.max()
        < calibration.index.min()
    )


def test_calibration_split_has_no_overlap():
    index = pd.date_range(
        "2020-01-01",
        periods=100,
        freq="D",
    )

    data = pd.DataFrame(
        {"Feature": np.arange(100)},
        index=index,
    )

    training, calibration = (
        CalibrationResearchEngine.create_calibration_split(
            data,
            calibration_fraction=0.25,
        )
    )

    assert set(
        training.index
    ).isdisjoint(
        set(calibration.index)
    )


def test_invalid_calibration_fraction_fails():
    index = pd.date_range(
        "2020-01-01",
        periods=100,
        freq="D",
    )

    data = pd.DataFrame(
        {"Feature": np.arange(100)},
        index=index,
    )

    with pytest.raises(ValueError):
        CalibrationResearchEngine.create_calibration_split(
            data,
            calibration_fraction=0.0,
        )

    with pytest.raises(ValueError):
        CalibrationResearchEngine.create_calibration_split(
            data,
            calibration_fraction=1.0,
        )


def test_unsorted_calibration_split_fails():
    index = pd.date_range(
        "2020-01-01",
        periods=100,
        freq="D",
    )[::-1]

    data = pd.DataFrame(
        {"Feature": np.arange(100)},
        index=index,
    )

    with pytest.raises(ValueError):
        CalibrationResearchEngine.create_calibration_split(
            data
        )


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------


def test_empty_probabilities_fail(engine):
    with pytest.raises(ValueError):
        engine.fit(
            np.array([]),
            np.array([]),
            pd.DatetimeIndex([]),
        )


def test_length_mismatch_fails(
    engine,
    calibration_data,
):
    probabilities, actuals, timestamps = (
        calibration_data
    )

    with pytest.raises(ValueError):
        engine.fit(
            probabilities[:-1],
            actuals,
            timestamps,
        )


def test_probability_above_one_fails(
    engine,
    calibration_data,
):
    probabilities, actuals, timestamps = (
        calibration_data
    )

    probabilities = probabilities.copy()
    probabilities[0] = 1.5

    with pytest.raises(ValueError):
        engine.fit(
            probabilities,
            actuals,
            timestamps,
        )


def test_negative_probability_fails(
    engine,
    calibration_data,
):
    probabilities, actuals, timestamps = (
        calibration_data
    )

    probabilities = probabilities.copy()
    probabilities[0] = -0.1

    with pytest.raises(ValueError):
        engine.fit(
            probabilities,
            actuals,
            timestamps,
        )


def test_nonfinite_probability_fails(
    engine,
    calibration_data,
):
    probabilities, actuals, timestamps = (
        calibration_data
    )

    probabilities = probabilities.copy()
    probabilities[0] = np.nan

    with pytest.raises(ValueError):
        engine.fit(
            probabilities,
            actuals,
            timestamps,
        )


def test_one_class_targets_fail(
    engine,
    calibration_data,
):
    probabilities, _, timestamps = (
        calibration_data
    )

    actuals = np.ones(
        len(probabilities),
        dtype=int,
    )

    with pytest.raises(ValueError):
        engine.fit(
            probabilities,
            actuals,
            timestamps,
        )


def test_duplicate_timestamps_fail(
    engine,
    calibration_data,
):
    probabilities, actuals, timestamps = (
        calibration_data
    )

    timestamps = timestamps.copy()

    timestamps = pd.DatetimeIndex(
        list(timestamps[:-1])
        + [timestamps[-2]]
    )

    with pytest.raises(ValueError):
        engine.fit(
            probabilities,
            actuals,
            timestamps,
        )


def test_unsorted_timestamps_fail(
    engine,
    calibration_data,
):
    probabilities, actuals, timestamps = (
        calibration_data
    )

    timestamps = timestamps[::-1]

    with pytest.raises(ValueError):
        engine.fit(
            probabilities,
            actuals,
            timestamps,
        )


def test_non_datetime_index_fails(
    engine,
    calibration_data,
):
    probabilities, actuals, _ = (
        calibration_data
    )

    timestamps = pd.Index(
        range(
            len(probabilities)
        )
    )

    with pytest.raises(TypeError):
        engine.fit(
            probabilities,
            actuals,
            timestamps,
        )


def test_too_few_samples_fail(
    calibration_data,
):
    probabilities, actuals, timestamps = (
        calibration_data
    )

    engine = CalibrationResearchEngine(
        CalibrationConfig(
            min_samples=500
        )
    )

    with pytest.raises(ValueError):
        engine.fit(
            probabilities,
            actuals,
            timestamps,
        )


# ---------------------------------------------------------------------
# Temporal separation
# ---------------------------------------------------------------------


def test_calibration_must_follow_training(
    engine,
    calibration_data,
):
    probabilities, actuals, timestamps = (
        calibration_data
    )

    training_end = timestamps[100]

    with pytest.raises(ValueError):
        engine.fit(
            probabilities,
            actuals,
            timestamps,
            training_end=training_end,
        )


def test_calibration_after_training_is_allowed(
    engine,
    calibration_data,
):
    probabilities, actuals, timestamps = (
        calibration_data
    )

    training_end = timestamps[0] - pd.Timedelta(days=1)

    result = engine.fit(
        probabilities,
        actuals,
        timestamps,
        training_end=training_end,
    )

    assert isinstance(
        result,
        CalibrationResearchResult,
    )


# ---------------------------------------------------------------------
# Successful calibration
# ---------------------------------------------------------------------


def test_fit_returns_result(
    engine,
    calibration_data,
):
    probabilities, actuals, timestamps = (
        calibration_data
    )

    result = engine.fit(
        probabilities,
        actuals,
        timestamps,
    )

    assert isinstance(
        result,
        CalibrationResearchResult,
    )


def test_calibration_sample_count(
    engine,
    calibration_data,
):
    probabilities, actuals, timestamps = (
        calibration_data
    )

    result = engine.fit(
        probabilities,
        actuals,
        timestamps,
    )

    assert (
        result.calibration_samples
        == len(probabilities)
    )


def test_calibration_period_is_preserved(
    engine,
    calibration_data,
):
    probabilities, actuals, timestamps = (
        calibration_data
    )

    result = engine.fit(
        probabilities,
        actuals,
        timestamps,
    )

    assert (
        result.calibration_start
        == timestamps.min()
    )

    assert (
        result.calibration_end
        == timestamps.max()
    )


def test_brier_scores_are_valid(
    engine,
    calibration_data,
):
    probabilities, actuals, timestamps = (
        calibration_data
    )

    result = engine.fit(
        probabilities,
        actuals,
        timestamps,
    )

    assert (
        result.raw_brier_score
        >= 0.0
    )

    assert (
        result.calibrated_brier_score
        >= 0.0
    )


def test_ece_scores_are_valid(
    engine,
    calibration_data,
):
    probabilities, actuals, timestamps = (
        calibration_data
    )

    result = engine.fit(
        probabilities,
        actuals,
        timestamps,
    )

    assert (
        0.0
        <= result.raw_ece
        <= 1.0
    )

    assert (
        0.0
        <= result.calibrated_ece
        <= 1.0
    )


# ---------------------------------------------------------------------
# Transformation
# ---------------------------------------------------------------------


def test_transform_returns_probability_array(
    engine,
    calibration_data,
):
    probabilities, actuals, timestamps = (
        calibration_data
    )

    result = engine.fit(
        probabilities,
        actuals,
        timestamps,
    )

    transformed = engine.transform(
        result,
        probabilities,
    )

    assert isinstance(
        transformed,
        np.ndarray,
    )

    assert len(transformed) == len(
        probabilities
    )

    assert np.isfinite(
        transformed
    ).all()

    assert (
        transformed.min()
        >= 0.0
    )

    assert (
        transformed.max()
        <= 1.0
    )


def test_transform_rejects_invalid_probabilities(
    engine,
    calibration_data,
):
    probabilities, actuals, timestamps = (
        calibration_data
    )

    result = engine.fit(
        probabilities,
        actuals,
        timestamps,
    )

    invalid = probabilities.copy()
    invalid[0] = 2.0

    with pytest.raises(ValueError):
        engine.transform(
            result,
            invalid,
        )


def test_transform_requires_result(
    engine,
    calibration_data,
):
    probabilities, _, _ = (
        calibration_data
    )

    with pytest.raises(ValueError):
        engine.transform(
            None,
            probabilities,
        )


# ---------------------------------------------------------------------
# High-confidence diagnostics
# ---------------------------------------------------------------------


def test_high_confidence_threshold_is_recorded(
    engine,
    calibration_data,
):
    probabilities, actuals, timestamps = (
        calibration_data
    )

    result = engine.fit(
        probabilities,
        actuals,
        timestamps,
    )

    assert (
        result.high_confidence_threshold
        == 0.70
    )


def test_high_confidence_coverage_is_valid(
    engine,
    calibration_data,
):
    probabilities, actuals, timestamps = (
        calibration_data
    )

    result = engine.fit(
        probabilities,
        actuals,
        timestamps,
    )

    assert (
        0.0
        <= result.raw_high_confidence_coverage
        <= 1.0
    )

    assert (
        0.0
        <= result.calibrated_high_confidence_coverage
        <= 1.0
    )


# ---------------------------------------------------------------------
# Metadata / safety
# ---------------------------------------------------------------------


def test_final_holdout_is_never_used(
    engine,
    calibration_data,
):
    probabilities, actuals, timestamps = (
        calibration_data
    )

    result = engine.fit(
        probabilities,
        actuals,
        timestamps,
    )

    assert (
        result.metadata[
            "final_holdout_used"
        ]
        is False
    )

    assert (
        result.metadata[
            "calibrator_fitted_on_holdout"
        ]
        is False
    )


def test_result_is_research_only(
    engine,
    calibration_data,
):
    probabilities, actuals, timestamps = (
        calibration_data
    )

    result = engine.fit(
        probabilities,
        actuals,
        timestamps,
    )

    assert (
        result.metadata[
            "research_only"
        ]
        is True
    )


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------


def test_summary_contains_key_metrics(
    engine,
    calibration_data,
):
    probabilities, actuals, timestamps = (
        calibration_data
    )

    result = engine.fit(
        probabilities,
        actuals,
        timestamps,
    )

    summary = result.summary()

    required = {
        "method",
        "raw_brier_score",
        "calibrated_brier_score",
        "raw_log_loss",
        "calibrated_log_loss",
        "raw_ece",
        "calibrated_ece",
        "calibration_samples",
        "calibration_start",
        "calibration_end",
        "brier_improved",
        "ece_improved",
    }

    assert required.issubset(
        summary.keys()
    )


# ---------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------


def test_convenience_api(
    calibration_data,
):
    probabilities, actuals, timestamps = (
        calibration_data
    )

    result = calibrate_research_probabilities(
        probabilities=probabilities,
        actuals=actuals,
        timestamps=timestamps,
        config=CalibrationConfig(
            method="sigmoid",
            min_samples=50,
        ),
    )

    assert isinstance(
        result,
        CalibrationResearchResult,
    )


# ---------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------


def test_calibration_is_deterministic(
    calibration_data,
):
    probabilities, actuals, timestamps = (
        calibration_data
    )

    config = CalibrationConfig(
        method="sigmoid",
        min_samples=50,
    )

    result_a = (
        CalibrationResearchEngine(
            config
        ).fit(
            probabilities,
            actuals,
            timestamps,
        )
    )

    result_b = (
        CalibrationResearchEngine(
            config
        ).fit(
            probabilities,
            actuals,
            timestamps,
        )
    )

    assert (
        result_a.raw_brier_score
        == result_b.raw_brier_score
    )

    assert (
        result_a.calibrated_brier_score
        == result_b.calibrated_brier_score
    )

    assert (
        result_a.raw_ece
        == result_b.raw_ece
    )

    assert (
        result_a.calibrated_ece
        == result_b.calibrated_ece
    )
