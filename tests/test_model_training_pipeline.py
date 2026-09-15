"""
Tests for the controlled research model-training pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.data_preparation import (
    DataPreparationConfig,
    prepare_market_data,
)
from src.research.feature_integration import (
    UnifiedFeatureConfig,
)
from src.research.research_feature_pipeline import (
    ResearchFeaturePipeline,
    ResearchFeaturePipelineConfig,
)
from src.research.research_model_pipeline import (
    ResearchModelPipeline,
    ResearchModelPipelineConfig,
)
from src.research.research_target_pipeline import (
    ResearchTargetPipelineConfig,
)
from src.research.model_training_pipeline import (
    ResearchModelTrainingConfig,
    ResearchModelTrainingPipeline,
    ResearchModelTrainingResult,
    research_model_training_summary,
    train_research_model,
)


def make_ohlcv(
    rows: int = 900,
) -> pd.DataFrame:
    index = pd.date_range(
        "2020-01-01",
        periods=rows,
        freq="D",
    )

    t = np.arange(rows)

    close = (
        100.0
        + np.linspace(
            0,
            50,
            rows,
        )
        + 2.0
        * np.sin(t / 13.0)
        + 0.7
        * np.sin(t / 31.0)
    )

    open_price = (
        close
        + 0.25
        * np.sin(t / 7.0)
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
        + 2_000
        * (t % 25)
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


def make_model_dataset():
    prepared = prepare_market_data(
        make_ohlcv(),
        symbol="TCS.NS",
        timeframe="1D",
        config=DataPreparationConfig(
            minimum_rows=100,
        ),
    )

    feature_pipeline = (
        ResearchFeaturePipeline(
            config=ResearchFeaturePipelineConfig(
                unified=UnifiedFeatureConfig(
                    include_market_context=False,
                    include_relative_strength=False,
                    include_multi_timeframe=False,
                ),
                minimum_features=10,
            )
        )
    )

    features = feature_pipeline.build(
        prepared
    )

    model_pipeline = ResearchModelPipeline(
        config=ResearchModelPipelineConfig(
            target=ResearchTargetPipelineConfig(
                horizons=(1,)
            ),
            minimum_rows=100,
        )
    )

    return model_pipeline.build(
        features
    )


def training_config():
    return ResearchModelTrainingConfig(
        train_fraction=0.75,
        validation_fraction=0.25,
        minimum_training_rows=100,
        minimum_validation_rows=30,
    )


def test_config_defaults():
    config = ResearchModelTrainingConfig()

    assert config.model_type
    assert 0 < config.train_fraction < 1
    assert 0 < config.validation_fraction < 1
    assert np.isclose(
        config.train_fraction
        + config.validation_fraction,
        1.0,
    )


def test_invalid_model_type_is_rejected():
    with pytest.raises(ValueError):
        ResearchModelTrainingConfig(
            model_type=""
        )


def test_invalid_train_fraction_is_rejected():
    with pytest.raises(ValueError):
        ResearchModelTrainingConfig(
            train_fraction=0
        )

    with pytest.raises(ValueError):
        ResearchModelTrainingConfig(
            train_fraction=1
        )


def test_invalid_validation_fraction_is_rejected():
    with pytest.raises(ValueError):
        ResearchModelTrainingConfig(
            validation_fraction=0
        )

    with pytest.raises(ValueError):
        ResearchModelTrainingConfig(
            validation_fraction=1
        )


def test_train_and_validation_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        ResearchModelTrainingConfig(
            train_fraction=0.80,
            validation_fraction=0.10,
        )


def test_invalid_minimum_rows_are_rejected():
    with pytest.raises(ValueError):
        ResearchModelTrainingConfig(
            minimum_training_rows=0
        )

    with pytest.raises(ValueError):
        ResearchModelTrainingConfig(
            minimum_validation_rows=0
        )


def test_pipeline_construction():
    pipeline = (
        ResearchModelTrainingPipeline(
            config=training_config()
        )
    )

    assert isinstance(
        pipeline,
        ResearchModelTrainingPipeline,
    )


def test_model_dataset_can_be_created():
    dataset = make_model_dataset()

    assert dataset.rows > 0
    assert dataset.feature_count > 0
    assert dataset.target_count > 0


def test_training_returns_correct_result():
    dataset = make_model_dataset()

    result = train_research_model(
        dataset,
        config=training_config(),
    )

    assert isinstance(
        result,
        ResearchModelTrainingResult,
    )


def test_training_creates_model():
    dataset = make_model_dataset()

    result = train_research_model(
        dataset,
        config=training_config(),
    )

    assert result.model is not None
    assert result.preprocessor is not None


def test_training_has_features():
    dataset = make_model_dataset()

    result = train_research_model(
        dataset,
        config=training_config(),
    )

    assert result.feature_columns
    assert len(
        result.feature_columns
    ) == dataset.feature_count


def test_target_column_is_recorded():
    dataset = make_model_dataset()

    result = train_research_model(
        dataset,
        target_column=dataset.direction_columns[0],
        config=training_config(),
    )

    assert (
        result.target_column
        == dataset.direction_columns[0]
    )


def test_training_and_validation_are_chronological():
    dataset = make_model_dataset()

    result = train_research_model(
        dataset,
        config=training_config(),
    )

    assert (
        result.train_index.is_monotonic_increasing
    )

    assert (
        result.validation_index.is_monotonic_increasing
    )

    assert (
        result.train_index[-1]
        < result.validation_index[0]
    )


def test_training_and_validation_do_not_overlap():
    dataset = make_model_dataset()

    result = train_research_model(
        dataset,
        config=training_config(),
    )

    assert not (
        set(result.train_index)
        & set(result.validation_index)
    )


def test_validation_predictions_have_correct_length():
    dataset = make_model_dataset()

    result = train_research_model(
        dataset,
        config=training_config(),
    )

    assert len(
        result.validation_predictions
    ) == len(
        result.validation_index
    )


def test_validation_probabilities_have_correct_length():
    dataset = make_model_dataset()

    result = train_research_model(
        dataset,
        config=training_config(),
    )

    assert (
        result.validation_probabilities.shape[0]
        == len(result.validation_index)
    )


def test_binary_probability_output():
    dataset = make_model_dataset()

    result = train_research_model(
        dataset,
        config=training_config(),
    )

    probabilities = (
        result.validation_probabilities
    )

    assert probabilities.ndim == 2
    assert probabilities.shape[1] == 2

    assert np.isfinite(
        probabilities
    ).all()

    assert (
        probabilities >= 0
    ).all()

    assert (
        probabilities <= 1
    ).all()

    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-6,
    )


def test_validation_predictions_are_binary():
    dataset = make_model_dataset()

    result = train_research_model(
        dataset,
        config=training_config(),
    )

    unique = set(
        np.asarray(
            result.validation_predictions
        ).tolist()
    )

    assert unique.issubset(
        {0, 1}
    )


def test_validation_accuracy_is_between_zero_and_one():
    dataset = make_model_dataset()

    result = train_research_model(
        dataset,
        config=training_config(),
    )

    assert 0.0 <= (
        result.validation_accuracy
    ) <= 1.0


def test_validation_metrics_are_present():
    dataset = make_model_dataset()

    result = train_research_model(
        dataset,
        config=training_config(),
    )

    assert isinstance(
        result.validation_metrics,
        dict,
    )

    assert len(
        result.validation_metrics
    ) > 0


def test_candidate_passes_only_if_accuracy_reaches_95_percent():
    dataset = make_model_dataset()

    result = train_research_model(
        dataset,
        config=training_config(),
    )

    expected = (
        result.validation_accuracy
        >= 0.95
    )

    assert (
        result.candidate_passed
        == expected
    )


def test_failed_95_percent_candidate_gets_warning():
    dataset = make_model_dataset()

    result = train_research_model(
        dataset,
        config=training_config(),
    )

    if not result.candidate_passed:
        assert any(
            "95%"
            in warning
            for warning in result.warnings
        )


def test_validation_is_not_called_final_holdout():
    dataset = make_model_dataset()

    result = train_research_model(
        dataset,
        config=training_config(),
    )

    assert any(
        "final holdout"
        in warning.lower()
        for warning in result.warnings
    )


def test_final_holdout_is_not_used():
    dataset = make_model_dataset()

    result = train_research_model(
        dataset,
        config=training_config(),
    )

    assert (
        result.metadata[
            "final_holdout_used"
        ]
        is False
    )

    assert (
        result.metadata[
            "final_holdout_fitted"
        ]
        is False
    )


def test_calibration_is_not_performed():
    dataset = make_model_dataset()

    result = train_research_model(
        dataset,
        config=training_config(),
    )

    assert (
        result.metadata[
            "calibration_fitted"
        ]
        is False
    )


def test_model_selection_is_not_performed():
    dataset = make_model_dataset()

    result = train_research_model(
        dataset,
        config=training_config(),
    )

    assert (
        result.metadata[
            "model_selected"
        ]
        is False
    )


def test_walk_forward_is_not_claimed():
    dataset = make_model_dataset()

    result = train_research_model(
        dataset,
        config=training_config(),
    )

    assert (
        result.metadata[
            "walk_forward_validated"
        ]
        is False
    )


def test_production_approval_is_false():
    dataset = make_model_dataset()

    result = train_research_model(
        dataset,
        config=training_config(),
    )

    assert (
        result.metadata[
            "production_approved"
        ]
        is False
    )

    assert (
        result.metadata[
            "production_ready"
        ]
        is False
    )


def test_production_ready_property_is_false():
    dataset = make_model_dataset()

    result = train_research_model(
        dataset,
        config=training_config(),
    )

    assert result.production_ready is False


def test_training_does_not_modify_dataset():
    dataset = make_model_dataset()

    original = dataset.data.copy(
        deep=True
    )

    train_research_model(
        dataset,
        config=training_config(),
    )

    pd.testing.assert_frame_equal(
        dataset.data,
        original,
    )


def test_training_does_not_modify_feature_columns():
    dataset = make_model_dataset()

    original = list(
        dataset.feature_columns
    )

    train_research_model(
        dataset,
        config=training_config(),
    )

    assert (
        dataset.feature_columns
        == original
    )


def test_explicit_target_must_exist():
    dataset = make_model_dataset()

    with pytest.raises(ValueError):
        train_research_model(
            dataset,
            target_column="NOT_A_TARGET",
            config=training_config(),
        )


def test_direction_target_is_selected_by_default():
    dataset = make_model_dataset()

    result = train_research_model(
        dataset,
        config=training_config(),
    )

    assert result.target_column in (
        dataset.direction_columns
    )


def test_wrong_dataset_type_is_rejected():
    pipeline = (
        ResearchModelTrainingPipeline(
            config=training_config()
        )
    )

    with pytest.raises(TypeError):
        pipeline.train(
            None
        )


def test_empty_dataset_is_rejected():
    dataset = make_model_dataset()

    dataset.data = dataset.data.iloc[
        0:0
    ]

    with pytest.raises(ValueError):
        train_research_model(
            dataset,
            config=training_config(),
        )


def test_missing_features_are_rejected():
    dataset = make_model_dataset()

    dataset.feature_columns = []

    with pytest.raises(ValueError):
        train_research_model(
            dataset,
            config=training_config(),
        )


def test_missing_targets_are_rejected():
    dataset = make_model_dataset()

    dataset.target_columns = []

    with pytest.raises(ValueError):
        train_research_model(
            dataset,
            config=training_config(),
        )


def test_insufficient_training_rows_are_rejected():
    dataset = make_model_dataset()

    config = ResearchModelTrainingConfig(
        train_fraction=0.75,
        validation_fraction=0.25,
        minimum_training_rows=10_000,
        minimum_validation_rows=30,
    )

    with pytest.raises(ValueError):
        train_research_model(
            dataset,
            config=config,
        )


def test_insufficient_validation_rows_are_rejected():
    dataset = make_model_dataset()

    config = ResearchModelTrainingConfig(
        train_fraction=0.99,
        validation_fraction=0.01,
        minimum_training_rows=100,
        minimum_validation_rows=100,
    )

    with pytest.raises(ValueError):
        train_research_model(
            dataset,
            config=config,
        )


def test_training_is_deterministic_for_same_input():
    dataset = make_model_dataset()

    config = training_config()

    first = train_research_model(
        dataset,
        config=config,
    )

    second = train_research_model(
        dataset,
        config=config,
    )

    assert (
        first.train_index.equals(
            second.train_index
        )
    )

    assert (
        first.validation_index.equals(
            second.validation_index
        )
    )

    assert np.array_equal(
        first.validation_predictions,
        second.validation_predictions,
    )

    assert np.allclose(
        first.validation_probabilities,
        second.validation_probabilities,
    )

    assert np.isclose(
        first.validation_accuracy,
        second.validation_accuracy,
    )


def test_summary_contains_expected_fields():
    dataset = make_model_dataset()

    result = train_research_model(
        dataset,
        config=training_config(),
    )

    summary = research_model_training_summary(
        result
    )

    assert (
        summary["target_column"]
        == result.target_column
    )

    assert (
        summary["feature_count"]
        == len(result.feature_columns)
    )

    assert (
        summary["train_rows"]
        == len(result.train_index)
    )

    assert (
        summary["validation_rows"]
        == len(result.validation_index)
    )

    assert (
        summary["validation_accuracy"]
        == result.validation_accuracy
    )

    assert (
        summary["production_ready"]
        is False
    )

    assert (
        summary["research_only"]
        is True
    )


def test_summary_rejects_wrong_type():
    with pytest.raises(TypeError):
        research_model_training_summary(
            None
        )


def test_model_and_preprocessor_are_distinct_objects():
    dataset = make_model_dataset()

    result = train_research_model(
        dataset,
        config=training_config(),
    )

    assert result.model is not result.preprocessor


def test_training_preprocessor_is_fitted():
    dataset = make_model_dataset()

    result = train_research_model(
        dataset,
        config=training_config(),
    )

    assert getattr(
        result.preprocessor,
        "fitted_",
        True,
    )


def test_validation_predictions_are_aligned_to_validation_index():
    dataset = make_model_dataset()

    result = train_research_model(
        dataset,
        config=training_config(),
    )

    assert len(
        result.validation_predictions
    ) == len(
        result.validation_index
    )

    assert len(
        result.validation_probabilities
    ) == len(
        result.validation_index
    )


def test_training_result_contains_model_type_metadata():
    dataset = make_model_dataset()

    config = training_config()

    result = train_research_model(
        dataset,
        config=config,
    )

    assert (
        result.metadata[
            "model_type"
        ]
        == config.model_type
    )


def test_random_state_is_recorded():
    dataset = make_model_dataset()

    config = ResearchModelTrainingConfig(
        random_state=123,
    )

    result = train_research_model(
        dataset,
        config=config,
    )

    assert (
        result.metadata[
            "random_state"
        ]
        == 123
    )
