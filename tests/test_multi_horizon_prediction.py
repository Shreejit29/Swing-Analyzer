"""
Tests for production multi-horizon prediction.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.research.multi_horizon_prediction import (
    DEFAULT_HORIZONS,
    MultiHorizonConfig,
    MultiHorizonPrediction,
    MultiHorizonPredictionEngine,
    build_multi_horizon_engine,
    multi_horizon_summary,
)
from src.research.production_prediction import (
    ProductionPredictionGateway,
    ProductionPredictionStatus,
)


class FakePreprocessor:
    def transform(self, frame):
        return frame.to_numpy(dtype=float)


class FakeClassifier:
    classes_ = np.array([0, 1])

    def __init__(self, probability_up):
        self.probability_up = probability_up

    def predict_proba(self, X):
        return np.tile(
            [
                1.0 - self.probability_up,
                self.probability_up,
            ],
            (len(X), 1),
        )


def make_gateway(
    probability_up=0.80,
    model_id="model_test",
):
    feature_names = [
        "RSI14",
        "EMA_20_Distance",
        "ATR_PCT",
    ]

    metadata = SimpleNamespace(
        model_id=model_id,
        feature_names=feature_names,
        production_approved=True,
        artifact_version="1.0",
    )

    artifact = SimpleNamespace(
        model=FakeClassifier(
            probability_up
        ),
        preprocessor=FakePreprocessor(),
        metadata=metadata,
    )

    return ProductionPredictionGateway(
        artifact
    )


def make_gateways(
    probabilities=None,
):
    if probabilities is None:
        probabilities = {
            1: 0.80,
            3: 0.78,
            5: 0.82,
            10: 0.75,
            20: 0.72,
        }

    return {
        horizon: make_gateway(
            probability_up=probability,
            model_id=f"model_{horizon}D",
        )
        for horizon, probability
        in probabilities.items()
    }


def make_features():
    return {
        "RSI14": 58.0,
        "EMA_20_Distance": 0.025,
        "ATR_PCT": 0.018,
    }


def make_features_by_horizon():
    return {
        horizon: make_features()
        for horizon in DEFAULT_HORIZONS
    }


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


def test_default_horizons():
    assert DEFAULT_HORIZONS == (
        1,
        3,
        5,
        10,
        20,
    )


def test_default_config():
    config = MultiHorizonConfig()

    assert config.horizons == DEFAULT_HORIZONS
    assert config.preferred_horizon == 5
    assert (
        config.minimum_agreement_fraction
        == pytest.approx(0.60)
    )


def test_custom_config():
    config = MultiHorizonConfig(
        horizons=(1, 5, 10),
        preferred_horizon=5,
        minimum_ready_horizons=2,
    )

    assert config.horizons == (
        1,
        5,
        10,
    )


def test_empty_horizons_fail():
    with pytest.raises(ValueError):
        MultiHorizonConfig(
            horizons=()
        )


def test_duplicate_horizons_fail():
    with pytest.raises(ValueError):
        MultiHorizonConfig(
            horizons=(1, 1, 5)
        )


def test_invalid_horizon_fails():
    with pytest.raises(ValueError):
        MultiHorizonConfig(
            horizons=(1, 0, 5)
        )


def test_invalid_agreement_fraction():
    with pytest.raises(ValueError):
        MultiHorizonConfig(
            minimum_agreement_fraction=1.5
        )


def test_invalid_ready_horizon_count():
    with pytest.raises(ValueError):
        MultiHorizonConfig(
            minimum_ready_horizons=0
        )


def test_ready_horizon_count_cannot_exceed_horizons():
    with pytest.raises(ValueError):
        MultiHorizonConfig(
            horizons=(1, 5),
            minimum_ready_horizons=3,
        )


def test_preferred_horizon_must_exist():
    with pytest.raises(ValueError):
        MultiHorizonConfig(
            preferred_horizon=7
        )


# ---------------------------------------------------------------------
# Engine construction
# ---------------------------------------------------------------------


def test_engine_construction():
    engine = MultiHorizonPredictionEngine(
        make_gateways()
    )

    assert engine.horizons == DEFAULT_HORIZONS


def test_engine_requires_gateways():
    with pytest.raises(ValueError):
        MultiHorizonPredictionEngine(
            {}
        )


def test_engine_requires_all_configured_horizons():
    gateways = make_gateways()

    gateways.pop(5)

    with pytest.raises(ValueError):
        MultiHorizonPredictionEngine(
            gateways
        )


def test_engine_rejects_invalid_gateway():
    gateways = make_gateways()

    gateways[5] = object()

    with pytest.raises(TypeError):
        MultiHorizonPredictionEngine(
            gateways
        )


def test_engine_rejects_invalid_gateway_horizon():
    gateways = make_gateways()

    gateways[-1] = gateways[1]

    with pytest.raises(ValueError):
        MultiHorizonPredictionEngine(
            gateways
        )


def test_builder():
    engine = build_multi_horizon_engine(
        make_gateways()
    )

    assert isinstance(
        engine,
        MultiHorizonPredictionEngine,
    )


# ---------------------------------------------------------------------
# Successful multi-horizon prediction
# ---------------------------------------------------------------------


def test_all_horizons_are_predicted():
    engine = MultiHorizonPredictionEngine(
        make_gateways()
    )

    result = engine.predict(
        make_features_by_horizon()
    )

    assert set(
        result.horizon_predictions.keys()
    ) == set(DEFAULT_HORIZONS)


def test_all_horizons_are_independent():
    probabilities = {
        1: 0.80,
        3: 0.20,
        5: 0.75,
        10: 0.30,
        20: 0.70,
    }

    engine = MultiHorizonPredictionEngine(
        make_gateways(probabilities)
    )

    result = engine.predict(
        make_features_by_horizon()
    )

    assert (
        result.horizon_predictions[
            1
        ].predicted_direction
        == "UP"
    )

    assert (
        result.horizon_predictions[
            3
        ].predicted_direction
        == "DOWN"
    )

    assert (
        result.horizon_predictions[
            5
        ].predicted_direction
        == "UP"
    )

    assert (
        result.horizon_predictions[
            10
        ].predicted_direction
        == "DOWN"
    )

    assert (
        result.horizon_predictions[
            20
        ].predicted_direction
        == "UP"
    )


def test_strong_up_consensus():
    engine = MultiHorizonPredictionEngine(
        make_gateways()
    )

    result = engine.predict(
        make_features_by_horizon()
    )

    assert (
        result.consensus_direction
        == "UP"
    )

    assert (
        result.trade_allowed
        is True
    )

    assert (
        result.status
        == ProductionPredictionStatus.READY
    )


def test_consensus_probability_is_calculated():
    engine = MultiHorizonPredictionEngine(
        make_gateways()
    )

    result = engine.predict(
        make_features_by_horizon()
    )

    assert result.consensus_probability is not None

    assert (
        0.60
        <= result.consensus_probability
        <= 1.0
    )


def test_agreement_fraction():
    probabilities = {
        1: 0.80,
        3: 0.78,
        5: 0.82,
        10: 0.75,
        20: 0.72,
    }

    engine = MultiHorizonPredictionEngine(
        make_gateways(probabilities)
    )

    result = engine.predict(
        make_features_by_horizon()
    )

    assert result.agreement_fraction == pytest.approx(
        1.0
    )


def test_preferred_horizon_is_five_days():
    engine = MultiHorizonPredictionEngine(
        make_gateways()
    )

    result = engine.predict(
        make_features_by_horizon()
    )

    assert (
        result.preferred_horizon
        == 5
    )


# ---------------------------------------------------------------------
# Mixed horizon behavior
# ---------------------------------------------------------------------


def test_mixed_directions_can_still_produce_consensus():
    probabilities = {
        1: 0.80,
        3: 0.75,
        5: 0.70,
        10: 0.25,
        20: 0.20,
    }

    engine = MultiHorizonPredictionEngine(
        make_gateways(probabilities)
    )

    result = engine.predict(
        make_features_by_horizon()
    )

    assert (
        result.consensus_direction
        == "UP"
    )

    assert (
        result.agreement_fraction
        == pytest.approx(0.60)
    )

    assert (
        result.trade_allowed
        is True
    )


def test_equal_directional_split_waits():
    probabilities = {
        1: 0.80,
        3: 0.80,
        5: 0.20,
        10: 0.20,
        20: 0.50,
    }

    engine = MultiHorizonPredictionEngine(
        make_gateways(probabilities)
    )

    result = engine.predict(
        make_features_by_horizon()
    )

    assert (
        result.consensus_direction
        == "NEUTRAL"
    )

    assert (
        result.trade_allowed
        is False
    )

    assert (
        result.status
        == ProductionPredictionStatus.WAIT
    )


def test_low_probability_horizons_are_not_forced_into_trade():
    probabilities = {
        1: 0.55,
        3: 0.56,
        5: 0.57,
        10: 0.58,
        20: 0.59,
    }

    engine = MultiHorizonPredictionEngine(
        make_gateways(probabilities)
    )

    result = engine.predict(
        make_features_by_horizon()
    )

    assert (
        result.trade_allowed
        is False
    )

    assert (
        result.status
        == ProductionPredictionStatus.WAIT
    )


# ---------------------------------------------------------------------
# Missing horizon data
# ---------------------------------------------------------------------


def test_missing_horizon_features_fail_closed_for_that_horizon():
    engine = MultiHorizonPredictionEngine(
        make_gateways()
    )

    features = make_features_by_horizon()
    features.pop(5)

    result = engine.predict(
        features
    )

    prediction = (
        result.horizon_predictions[5]
    )

    assert (
        prediction.status
        == ProductionPredictionStatus.BLOCKED
    )

    assert (
        prediction.trade_allowed
        is False
    )


def test_missing_horizon_does_not_reuse_another_horizon():
    engine = MultiHorizonPredictionEngine(
        make_gateways()
    )

    features = make_features_by_horizon()
    features.pop(10)

    result = engine.predict(
        features
    )

    assert (
        result.horizon_predictions[10]
        .status
        == ProductionPredictionStatus.BLOCKED
    )


def test_no_valid_horizons_blocks():
    engine = MultiHorizonPredictionEngine(
        make_gateways()
    )

    result = engine.predict(
        {}
    )

    assert (
        result.status
        == ProductionPredictionStatus.BLOCKED
    )

    assert (
        result.trade_allowed
        is False
    )


def test_missing_horizon_is_reported():
    engine = MultiHorizonPredictionEngine(
        make_gateways()
    )

    features = make_features_by_horizon()
    features.pop(20)

    result = engine.predict(
        features
    )

    assert any(
        "20D" in error
        for error in result.errors
    )


# ---------------------------------------------------------------------
# Same-feature convenience method
# ---------------------------------------------------------------------


def test_predict_same_features():
    engine = MultiHorizonPredictionEngine(
        make_gateways()
    )

    result = engine.predict_same_features(
        make_features()
    )

    assert set(
        result.horizon_predictions.keys()
    ) == set(DEFAULT_HORIZONS)


def test_same_features_does_not_modify_input():
    engine = MultiHorizonPredictionEngine(
        make_gateways()
    )

    features = make_features()
    original = dict(features)

    engine.predict_same_features(
        features
    )

    assert features == original


# ---------------------------------------------------------------------
# Current price
# ---------------------------------------------------------------------


def test_current_price_propagates_to_horizons():
    engine = MultiHorizonPredictionEngine(
        make_gateways()
    )

    result = engine.predict(
        make_features_by_horizon(),
        current_price=1500.0,
    )

    for prediction in (
        result.horizon_predictions.values()
    ):
        assert (
            prediction.current_price
            == pytest.approx(1500.0)
        )


# ---------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------


def test_ready_horizon_count():
    engine = MultiHorizonPredictionEngine(
        make_gateways()
    )

    result = engine.predict(
        make_features_by_horizon()
    )

    assert (
        result.ready_horizon_count
        == 5
    )


def test_high_confidence_horizon_count():
    probabilities = {
        1: 0.80,
        3: 0.75,
        5: 0.65,
        10: 0.71,
        20: 0.55,
    }

    engine = MultiHorizonPredictionEngine(
        make_gateways(probabilities)
    )

    result = engine.predict(
        make_features_by_horizon()
    )

    assert (
        result.high_confidence_horizon_count
        == 3
    )


# ---------------------------------------------------------------------
# Configuration gates
# ---------------------------------------------------------------------


def test_minimum_ready_horizons_can_block_trade():
    config = MultiHorizonConfig(
        minimum_ready_horizons=5
    )

    engine = MultiHorizonPredictionEngine(
        make_gateways(),
        config=config,
    )

    features = make_features_by_horizon()
    features.pop(5)

    result = engine.predict(
        features
    )

    assert (
        result.ready_horizon_count
        == 4
    )

    assert (
        result.trade_allowed
        is False
    )


def test_agreement_threshold_can_block_trade():
    config = MultiHorizonConfig(
        minimum_agreement_fraction=0.80
    )

    probabilities = {
        1: 0.80,
        3: 0.80,
        5: 0.80,
        10: 0.20,
        20: 0.20,
    }

    engine = MultiHorizonPredictionEngine(
        make_gateways(probabilities),
        config=config,
    )

    result = engine.predict(
        make_features_by_horizon()
    )

    assert (
        result.agreement_fraction
        == pytest.approx(0.60)
    )

    assert (
        result.trade_allowed
        is False
    )

    assert (
        result.status
        == ProductionPredictionStatus.WAIT
    )


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------


def test_multi_horizon_summary():
    engine = MultiHorizonPredictionEngine(
        make_gateways()
    )

    result = engine.predict(
        make_features_by_horizon()
    )

    summary = result.summary()

    assert (
        summary["consensus_direction"]
        == "UP"
    )

    assert (
        summary["preferred_horizon"]
        == 5
    )

    assert (
        summary["trade_allowed"]
        is True
    )

    assert set(
        summary["horizons"]
    ) == set(DEFAULT_HORIZONS)


def test_summary_helper():
    engine = MultiHorizonPredictionEngine(
        make_gateways()
    )

    result = engine.predict(
        make_features_by_horizon()
    )

    summary = multi_horizon_summary(
        result
    )

    assert (
        summary["model_id"]
        == "multi_horizon"
    )


def test_summary_helper_rejects_wrong_type():
    with pytest.raises(TypeError):
        multi_horizon_summary(
            object()
        )


# ---------------------------------------------------------------------
# Safety metadata
# ---------------------------------------------------------------------


def test_multi_horizon_never_fits_models():
    engine = MultiHorizonPredictionEngine(
        make_gateways()
    )

    result = engine.predict(
        make_features_by_horizon()
    )

    assert (
        result.metadata[
            "model_fitted"
        ]
        is False
    )

    assert (
        result.metadata[
            "preprocessor_fitted"
        ]
        is False
    )

    assert (
        result.metadata[
            "calibration_fitted"
        ]
        is False
    )

    assert (
        result.metadata[
            "final_holdout_fitted"
        ]
        is False
    )


def test_multi_horizon_does_not_approve_model():
    engine = MultiHorizonPredictionEngine(
        make_gateways()
    )

    result = engine.predict(
        make_features_by_horizon()
    )

    assert (
        result.metadata[
            "production_approved"
        ]
        is False
    )


# ---------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------


def test_multi_horizon_prediction_is_deterministic():
    gateways_1 = make_gateways()
    gateways_2 = make_gateways()

    engine_1 = MultiHorizonPredictionEngine(
        gateways_1
    )

    engine_2 = MultiHorizonPredictionEngine(
        gateways_2
    )

    result_1 = engine_1.predict(
        make_features_by_horizon()
    )

    result_2 = engine_2.predict(
        make_features_by_horizon()
    )

    assert (
        result_1.consensus_direction
        == result_2.consensus_direction
    )

    assert (
        result_1.agreement_fraction
        == pytest.approx(
            result_2.agreement_fraction
        )
    )

    assert (
        result_1.consensus_probability
        == pytest.approx(
            result_2.consensus_probability
        )
    )


# ---------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------


def test_features_by_horizon_are_not_modified():
    engine = MultiHorizonPredictionEngine(
        make_gateways()
    )

    features = make_features_by_horizon()

    original = {
        horizon: dict(values)
        for horizon, values
        in features.items()
    }

    engine.predict(
        features
    )

    assert features == original


# ---------------------------------------------------------------------
# Research boundary
# ---------------------------------------------------------------------


def test_multi_horizon_result_is_not_automatically_a_trade():
    probabilities = {
        1: 0.80,
        3: 0.80,
        5: 0.80,
        10: 0.80,
        20: 0.80,
    }

    engine = MultiHorizonPredictionEngine(
        make_gateways(probabilities)
    )

    result = engine.predict(
        make_features_by_horizon()
    )

    # Consensus is only one upstream condition.
    # Later MTF, market regime, sector, RR, range,
    # calibration and approval gates still apply.
    assert (
        result.consensus_direction
        == "UP"
    )

    assert (
        result.metadata[
            "production_approved"
        ]
        is False
    )
