"""
Tests for the production-safe prediction gateway.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.research.production_prediction import (
    ProductionPrediction,
    ProductionPredictionConfig,
    ProductionPredictionGateway,
    ProductionPredictionStatus,
    production_prediction_summary,
)


class FakePreprocessor:
    """Minimal fitted preprocessor for production tests."""

    def __init__(self) -> None:
        self.transform_calls = 0

    def transform(
        self,
        frame: pd.DataFrame,
    ) -> np.ndarray:
        self.transform_calls += 1

        return frame.to_numpy(
            dtype=float
        )


class FakeClassifier:
    """Deterministic binary classifier."""

    classes_ = np.array([0, 1])

    def __init__(
        self,
        probability_up: float = 0.80,
    ) -> None:
        self.probability_up = float(
            probability_up
        )
        self.predict_proba_calls = 0

    def predict_proba(
        self,
        X,
    ) -> np.ndarray:
        self.predict_proba_calls += 1

        n = len(X)

        return np.tile(
            [
                1.0 - self.probability_up,
                self.probability_up,
            ],
            (n, 1),
        )


def make_artifact(
    *,
    approved: bool = True,
    probability_up: float = 0.80,
    model_id: str = "model_test",
):
    feature_names = [
        "RSI14",
        "EMA_20_Distance",
        "ATR_PCT",
    ]

    metadata = SimpleNamespace(
        model_id=model_id,
        feature_names=feature_names,
        production_approved=approved,
        artifact_version="1.0",
    )

    return SimpleNamespace(
        model=FakeClassifier(
            probability_up
        ),
        preprocessor=FakePreprocessor(),
        metadata=metadata,
    )


def make_features() -> dict[str, float]:
    return {
        "RSI14": 58.0,
        "EMA_20_Distance": 0.025,
        "ATR_PCT": 0.018,
    }


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


def test_default_config():
    config = ProductionPredictionConfig()

    assert (
        config.require_approved_model
        is True
    )

    assert (
        config.minimum_probability
        == pytest.approx(0.60)
    )

    assert (
        config.high_confidence_probability
        == pytest.approx(0.70)
    )


def test_invalid_minimum_probability():
    with pytest.raises(ValueError):
        ProductionPredictionConfig(
            minimum_probability=1.5
        )


def test_invalid_high_confidence_probability():
    with pytest.raises(ValueError):
        ProductionPredictionConfig(
            high_confidence_probability=-0.1
        )


def test_high_confidence_cannot_be_below_minimum():
    with pytest.raises(ValueError):
        ProductionPredictionConfig(
            minimum_probability=0.80,
            high_confidence_probability=0.70,
        )


def test_invalid_missing_fraction():
    with pytest.raises(ValueError):
        ProductionPredictionConfig(
            maximum_missing_fraction=1.0
        )


# ---------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------


def test_gateway_accepts_approved_artifact():
    artifact = make_artifact(
        approved=True
    )

    gateway = ProductionPredictionGateway(
        artifact
    )

    assert gateway.artifact is artifact


def test_gateway_rejects_unapproved_artifact():
    artifact = make_artifact(
        approved=False
    )

    with pytest.raises(RuntimeError):
        ProductionPredictionGateway(
            artifact
        )


def test_gateway_can_allow_unapproved_when_explicitly_configured():
    artifact = make_artifact(
        approved=False
    )

    config = ProductionPredictionConfig(
        require_approved_model=False
    )

    gateway = ProductionPredictionGateway(
        artifact,
        config=config,
    )

    assert gateway.artifact is artifact


def test_gateway_requires_artifact_type():
    with pytest.raises(TypeError):
        ProductionPredictionGateway(
            object()
        )


def test_gateway_requires_feature_schema():
    artifact = make_artifact()

    artifact.metadata.feature_names = []

    with pytest.raises(RuntimeError):
        ProductionPredictionGateway(
            artifact
        )


def test_gateway_rejects_duplicate_feature_schema():
    artifact = make_artifact()

    artifact.metadata.feature_names = [
        "RSI14",
        "RSI14",
        "ATR_PCT",
    ]

    with pytest.raises(RuntimeError):
        ProductionPredictionGateway(
            artifact
        )


def test_expected_features():
    artifact = make_artifact()

    gateway = ProductionPredictionGateway(
        artifact
    )

    assert gateway.expected_features() == (
        "RSI14",
        "EMA_20_Distance",
        "ATR_PCT",
    )


# ---------------------------------------------------------------------
# Feature validation
# ---------------------------------------------------------------------


def test_mapping_features_are_accepted():
    gateway = ProductionPredictionGateway(
        make_artifact()
    )

    frame, warnings = (
        gateway.validate_features(
            make_features()
        )
    )

    assert isinstance(
        frame,
        pd.DataFrame,
    )

    assert list(
        frame.columns
    ) == list(
        gateway.expected_features()
    )

    assert warnings == []


def test_one_row_dataframe_is_accepted():
    gateway = ProductionPredictionGateway(
        make_artifact()
    )

    frame = pd.DataFrame(
        [make_features()]
    )

    validated, _ = (
        gateway.validate_features(
            frame
        )
    )

    assert len(validated) == 1


def test_multiple_rows_are_rejected_for_single_validation():
    gateway = ProductionPredictionGateway(
        make_artifact()
    )

    frame = pd.DataFrame(
        [
            make_features(),
            make_features(),
        ]
    )

    with pytest.raises(ValueError):
        gateway.validate_features(
            frame
        )


def test_empty_dataframe_is_rejected():
    gateway = ProductionPredictionGateway(
        make_artifact()
    )

    frame = pd.DataFrame(
        columns=[
            "RSI14",
            "EMA_20_Distance",
            "ATR_PCT",
        ]
    )

    with pytest.raises(ValueError):
        gateway.validate_features(
            frame
        )


def test_missing_feature_is_rejected():
    gateway = ProductionPredictionGateway(
        make_artifact()
    )

    features = make_features()
    features.pop("ATR_PCT")

    with pytest.raises(ValueError):
        gateway.validate_features(
            features
        )


def test_extra_feature_is_rejected():
    gateway = ProductionPredictionGateway(
        make_artifact()
    )

    features = make_features()
    features["Unexpected"] = 1.0

    with pytest.raises(ValueError):
        gateway.validate_features(
            features
        )


def test_feature_order_is_normalized():
    gateway = ProductionPredictionGateway(
        make_artifact()
    )

    features = {
        "ATR_PCT": 0.018,
        "RSI14": 58.0,
        "EMA_20_Distance": 0.025,
    }

    frame, _ = gateway.validate_features(
        features
    )

    assert list(
        frame.columns
    ) == list(
        gateway.expected_features()
    )


def test_non_numeric_feature_is_rejected():
    gateway = ProductionPredictionGateway(
        make_artifact()
    )

    features = make_features()
    features["RSI14"] = "high"

    with pytest.raises(TypeError):
        gateway.validate_features(
            features
        )


def test_nan_feature_is_rejected():
    gateway = ProductionPredictionGateway(
        make_artifact()
    )

    features = make_features()
    features["RSI14"] = np.nan

    with pytest.raises(ValueError):
        gateway.validate_features(
            features
        )


def test_infinite_feature_is_rejected():
    gateway = ProductionPredictionGateway(
        make_artifact()
    )

    features = make_features()
    features["RSI14"] = np.inf

    with pytest.raises(ValueError):
        gateway.validate_features(
            features
        )


def test_wrong_input_type_is_rejected():
    gateway = ProductionPredictionGateway(
        make_artifact()
    )

    with pytest.raises(TypeError):
        gateway.validate_features(
            [1, 2, 3]
        )


# ---------------------------------------------------------------------
# Preprocessor behavior
# ---------------------------------------------------------------------


def test_saved_preprocessor_is_used():
    artifact = make_artifact()

    gateway = ProductionPredictionGateway(
        artifact
    )

    gateway.predict(
        make_features()
    )

    assert (
        artifact.preprocessor.transform_calls
        == 1
    )


def test_preprocessor_is_not_fitted():
    artifact = make_artifact()

    original = artifact.preprocessor

    gateway = ProductionPredictionGateway(
        artifact
    )

    gateway.predict(
        make_features()
    )

    assert (
        gateway.artifact.preprocessor
        is original
    )


def test_missing_preprocessor_blocks_prediction():
    artifact = make_artifact()

    artifact.preprocessor = None

    gateway = ProductionPredictionGateway(
        artifact
    )

    result = gateway.predict(
        make_features()
    )

    assert (
        result.status
        == ProductionPredictionStatus.BLOCKED
    )

    assert (
        result.trade_allowed
        is False
    )


# ---------------------------------------------------------------------
# Probability mapping
# ---------------------------------------------------------------------


def test_probability_up_is_correct():
    artifact = make_artifact(
        probability_up=0.80
    )

    gateway = ProductionPredictionGateway(
        artifact
    )

    result = gateway.predict(
        make_features()
    )

    assert result.probability_up == pytest.approx(
        0.80
    )

    assert result.probability_down == pytest.approx(
        0.20
    )


def test_probabilities_sum_to_one():
    gateway = ProductionPredictionGateway(
        make_artifact(
            probability_up=0.73
        )
    )

    result = gateway.predict(
        make_features()
    )

    assert (
        result.probability_up
        + result.probability_down
        == pytest.approx(1.0)
    )


def test_up_prediction():
    gateway = ProductionPredictionGateway(
        make_artifact(
            probability_up=0.80
        )
    )

    result = gateway.predict(
        make_features()
    )

    assert (
        result.predicted_direction
        == "UP"
    )


def test_down_prediction():
    gateway = ProductionPredictionGateway(
        make_artifact(
            probability_up=0.20
        )
    )

    result = gateway.predict(
        make_features()
    )

    assert (
        result.predicted_direction
        == "DOWN"
    )


def test_neutral_when_probability_is_below_threshold():
    gateway = ProductionPredictionGateway(
        make_artifact(
            probability_up=0.55
        )
    )

    result = gateway.predict(
        make_features()
    )

    assert (
        result.predicted_direction
        == "NEUTRAL"
    )

    assert (
        result.status
        == ProductionPredictionStatus.WAIT
    )

    assert (
        result.trade_allowed
        is False
    )


def test_exact_minimum_probability_is_allowed():
    gateway = ProductionPredictionGateway(
        make_artifact(
            probability_up=0.60
        )
    )

    result = gateway.predict(
        make_features()
    )

    assert (
        result.predicted_direction
        == "UP"
    )

    assert (
        result.trade_allowed
        is True
    )


def test_high_confidence_threshold():
    gateway = ProductionPredictionGateway(
        make_artifact(
            probability_up=0.70
        )
    )

    result = gateway.predict(
        make_features()
    )

    assert (
        result.high_confidence
        is True
    )


def test_probability_below_high_confidence():
    gateway = ProductionPredictionGateway(
        make_artifact(
            probability_up=0.65
        )
    )

    result = gateway.predict(
        make_features()
    )

    assert (
        result.high_confidence
        is False
    )

    assert (
        result.trade_allowed
        is True
    )


# ---------------------------------------------------------------------
# Current price
# ---------------------------------------------------------------------


def test_current_price_is_preserved():
    gateway = ProductionPredictionGateway(
        make_artifact()
    )

    result = gateway.predict(
        make_features(),
        current_price=1250.50,
    )

    assert result.current_price == pytest.approx(
        1250.50
    )


def test_zero_current_price_blocks():
    gateway = ProductionPredictionGateway(
        make_artifact()
    )

    result = gateway.predict(
        make_features(),
        current_price=0,
    )

    assert (
        result.status
        == ProductionPredictionStatus.BLOCKED
    )


def test_negative_current_price_blocks():
    gateway = ProductionPredictionGateway(
        make_artifact()
    )

    result = gateway.predict(
        make_features(),
        current_price=-100,
    )

    assert (
        result.status
        == ProductionPredictionStatus.BLOCKED
    )


def test_nonfinite_current_price_blocks():
    gateway = ProductionPredictionGateway(
        make_artifact()
    )

    result = gateway.predict(
        make_features(),
        current_price=np.inf,
    )

    assert (
        result.status
        == ProductionPredictionStatus.BLOCKED
    )


# ---------------------------------------------------------------------
# Fail-closed prediction
# ---------------------------------------------------------------------


def test_invalid_features_return_blocked_result():
    gateway = ProductionPredictionGateway(
        make_artifact()
    )

    result = gateway.predict(
        {
            "RSI14": 50.0,
        }
    )

    assert (
        result.status
        == ProductionPredictionStatus.BLOCKED
    )

    assert (
        result.trade_allowed
        is False
    )

    assert len(
        result.errors
    ) >= 1


def test_blocked_result_contains_error():
    gateway = ProductionPredictionGateway(
        make_artifact()
    )

    result = gateway.predict(
        {}
    )

    assert result.errors
    assert isinstance(
        result.errors[0],
        str,
    )


def test_missing_model_probability_method_blocks():
    artifact = make_artifact()

    artifact.model = object()

    gateway = ProductionPredictionGateway(
        artifact
    )

    result = gateway.predict(
        make_features()
    )

    assert (
        result.status
        == ProductionPredictionStatus.BLOCKED
    )


def test_invalid_classes_block():
    artifact = make_artifact()

    artifact.model.classes_ = np.array(
        [0, 2]
    )

    gateway = ProductionPredictionGateway(
        artifact
    )

    result = gateway.predict(
        make_features()
    )

    assert (
        result.status
        == ProductionPredictionStatus.BLOCKED
    )


def test_probability_output_with_wrong_shape_blocks():
    class BadModel:
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            return np.array(
                [0.2, 0.8]
            )

    artifact = make_artifact()
    artifact.model = BadModel()

    gateway = ProductionPredictionGateway(
        artifact
    )

    result = gateway.predict(
        make_features()
    )

    assert (
        result.status
        == ProductionPredictionStatus.BLOCKED
    )


# ---------------------------------------------------------------------
# Multiple predictions
# ---------------------------------------------------------------------


def test_predict_many():
    gateway = ProductionPredictionGateway(
        make_artifact(
            probability_up=0.80
        )
    )

    rows = [
        make_features(),
        make_features(),
        make_features(),
    ]

    results = gateway.predict_many(
        rows
    )

    assert len(results) == 3

    assert all(
        isinstance(
            result,
            ProductionPrediction,
        )
        for result in results
    )


def test_predict_many_dataframe():
    gateway = ProductionPredictionGateway(
        make_artifact(
            probability_up=0.80
        )
    )

    frame = pd.DataFrame(
        [
            make_features(),
            make_features(),
        ]
    )

    results = gateway.predict_many(
        frame
    )

    assert len(results) == 2


def test_predict_many_does_not_fit():
    artifact = make_artifact()

    gateway = ProductionPredictionGateway(
        artifact
    )

    gateway.predict_many(
        [
            make_features(),
            make_features(),
        ]
    )

    assert (
        artifact.preprocessor.transform_calls
        == 2
    )


# ---------------------------------------------------------------------
# Prediction object
# ---------------------------------------------------------------------


def test_prediction_ready_property():
    prediction = ProductionPrediction(
        model_id="model_test",
        status=ProductionPredictionStatus.READY,
    )

    assert prediction.is_ready is True
    assert prediction.should_wait is False


def test_prediction_wait_property():
    prediction = ProductionPrediction(
        model_id="model_test",
        status=ProductionPredictionStatus.WAIT,
    )

    assert prediction.is_ready is False
    assert prediction.should_wait is True


def test_prediction_blocked():
    prediction = ProductionPrediction(
        model_id="model_test",
        status=ProductionPredictionStatus.BLOCKED,
    )

    assert prediction.is_ready is False
    assert prediction.should_wait is False


def test_prediction_summary():
    gateway = ProductionPredictionGateway(
        make_artifact(
            probability_up=0.80
        )
    )

    result = gateway.predict(
        make_features(),
        current_price=1000,
    )

    summary = result.summary()

    assert summary["model_id"] == (
        "model_test"
    )

    assert summary["status"] == "READY"

    assert summary[
        "predicted_direction"
    ] == "UP"

    assert summary[
        "trade_allowed"
    ] is True


def test_production_prediction_summary_helper():
    prediction = ProductionPrediction(
        model_id="model_test",
        status=ProductionPredictionStatus.WAIT,
    )

    summary = production_prediction_summary(
        prediction
    )

    assert summary[
        "model_id"
    ] == "model_test"

    assert summary[
        "status"
    ] == "WAIT"


def test_summary_rejects_wrong_type():
    with pytest.raises(TypeError):
        production_prediction_summary(
            object()
        )


# ---------------------------------------------------------------------
# Production safety metadata
# ---------------------------------------------------------------------


def test_production_prediction_metadata():
    gateway = ProductionPredictionGateway(
        make_artifact()
    )

    result = gateway.predict(
        make_features()
    )

    assert (
        result.metadata[
            "production_inference"
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
            "preprocessor_fitted"
        ]
        is False
    )

    assert (
        result.metadata[
            "final_holdout_fitted"
        ]
        is False
    )


def test_unapproved_artifact_never_predicts_by_default():
    artifact = make_artifact(
        approved=False
    )

    with pytest.raises(RuntimeError):
        ProductionPredictionGateway(
            artifact
        )


def test_prediction_does_not_modify_features():
    gateway = ProductionPredictionGateway(
        make_artifact()
    )

    features = make_features()
    original = dict(features)

    gateway.predict(
        features
    )

    assert features == original


def test_dataframe_input_is_not_modified():
    gateway = ProductionPredictionGateway(
        make_artifact()
    )

    frame = pd.DataFrame(
        [make_features()]
    )

    original = frame.copy(
        deep=True
    )

    gateway.predict(
        frame
    )

    pd.testing.assert_frame_equal(
        frame,
        original,
    )


# ---------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------


def test_prediction_is_deterministic():
    gateway = ProductionPredictionGateway(
        make_artifact(
            probability_up=0.82
        )
    )

    result_1 = gateway.predict(
        make_features()
    )

    result_2 = gateway.predict(
        make_features()
    )

    assert (
        result_1.predicted_direction
        == result_2.predicted_direction
    )

    assert (
        result_1.probability_up
        == pytest.approx(
            result_2.probability_up
        )
    )

    assert (
        result_1.probability_down
        == pytest.approx(
            result_2.probability_down
        )
    )


# ---------------------------------------------------------------------
# WAIT behavior
# ---------------------------------------------------------------------


def test_low_confidence_does_not_force_trade():
    gateway = ProductionPredictionGateway(
        make_artifact(
            probability_up=0.51
        )
    )

    result = gateway.predict(
        make_features()
    )

    assert (
        result.status
        == ProductionPredictionStatus.WAIT
    )

    assert (
        result.trade_allowed
        is False
    )

    assert (
        result.high_confidence
        is False
    )


def test_wait_is_allowed_production_behavior():
    gateway = ProductionPredictionGateway(
        make_artifact(
            probability_up=0.55
        )
    )

    result = gateway.predict(
        make_features()
    )

    assert result.should_wait is True
    assert result.trade_allowed is False
