"""
Tests for the multi-horizon research training pipeline.
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
from src.research.multi_horizon_training_pipeline import (
    DEFAULT_HORIZONS,
    HorizonTrainingResult,
    MultiHorizonTrainingConfig,
    MultiHorizonTrainingPipeline,
    MultiHorizonTrainingResult,
    multi_horizon_training_summary,
    train_multi_horizon,
)


def make_ohlcv(
    rows: int = 1500,
) -> pd.DataFrame:
    """Create deterministic synthetic OHLCV data."""

    index = pd.date_range(
        "2018-01-01",
        periods=rows,
        freq="D",
    )

    t = np.arange(rows)

    close = (
        100.0
        + np.linspace(
            0.0,
            100.0,
            rows,
        )
        + 3.0 * np.sin(t / 13.0)
        + 1.5 * np.sin(t / 31.0)
        + 0.5 * np.sin(t / 71.0)
    )

    open_price = (
        close
        + 0.25 * np.sin(t / 9.0)
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
        + 2_000 * (t % 30)
        + 500 * np.sin(t / 17.0)
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
                horizons=DEFAULT_HORIZONS
            ),
            minimum_rows=100,
        )
    )

    return model_pipeline.build(
        features
    )


def make_config(
    horizons=DEFAULT_HORIZONS,
):
    normalized = tuple(
        horizons
    )

    preferred = (
        5
        if 5 in normalized
        else normalized[0]
    )

    return MultiHorizonTrainingConfig(
        horizons=normalized,
        preferred_horizon=preferred,
        minimum_successful_horizons=1,
    )


def test_default_horizons():
    assert DEFAULT_HORIZONS == (
        1,
        3,
        5,
        10,
        20,
    )


def test_config_defaults():
    config = MultiHorizonTrainingConfig()

    assert config.horizons
    assert config.minimum_successful_horizons >= 1
    assert (
        config.preferred_horizon
        in config.horizons
    )


def test_empty_horizons_are_rejected():
    with pytest.raises(ValueError):
        MultiHorizonTrainingConfig(
            horizons=()
        )


def test_non_positive_horizon_is_rejected():
    with pytest.raises(ValueError):
        MultiHorizonTrainingConfig(
            horizons=(1, 0, 5)
        )

    with pytest.raises(ValueError):
        MultiHorizonTrainingConfig(
            horizons=(1, -5)
        )


def test_duplicate_horizons_are_rejected():
    with pytest.raises(ValueError):
        MultiHorizonTrainingConfig(
            horizons=(1, 3, 3, 5)
        )


def test_invalid_successful_horizon_count_is_rejected():
    with pytest.raises(ValueError):
        MultiHorizonTrainingConfig(
            horizons=(1, 5),
            minimum_successful_horizons=0,
        )

    with pytest.raises(ValueError):
        MultiHorizonTrainingConfig(
            horizons=(1, 5),
            minimum_successful_horizons=3,
        )


def test_invalid_preferred_horizon_is_rejected():
    with pytest.raises(ValueError):
        MultiHorizonTrainingConfig(
            horizons=(1, 5),
            preferred_horizon=20,
        )


def test_pipeline_construction():
    pipeline = (
        MultiHorizonTrainingPipeline(
            config=make_config(
                horizons=(1, 3)
            )
        )
    )

    assert isinstance(
        pipeline,
        MultiHorizonTrainingPipeline,
    )


def test_model_dataset_contains_direction_targets():
    dataset = make_model_dataset()

    assert dataset.direction_columns


def test_multi_horizon_training_returns_result():
    dataset = make_model_dataset()

    result = train_multi_horizon(
        dataset,
        horizons=(1, 3),
    )

    assert isinstance(
        result,
        MultiHorizonTrainingResult,
    )


def test_requested_horizons_are_preserved():
    dataset = make_model_dataset()

    horizons = (
        1,
        3,
        5,
    )

    result = train_multi_horizon(
        dataset,
        horizons=horizons,
    )

    assert (
        result.horizons
        == horizons
    )


def test_each_successful_horizon_has_result():
    dataset = make_model_dataset()

    result = train_multi_horizon(
        dataset,
        horizons=(1, 3),
    )

    for horizon in result.successful_horizons:
        assert horizon in result.results

        assert isinstance(
            result.results[horizon],
            HorizonTrainingResult,
        )


def test_horizon_result_has_correct_horizon():
    dataset = make_model_dataset()

    result = train_multi_horizon(
        dataset,
        horizons=(1, 5),
    )

    for horizon, horizon_result in (
        result.results.items()
    ):
        assert (
            horizon_result.horizon
            == horizon
        )


def test_horizon_result_target_is_directional():
    dataset = make_model_dataset()

    result = train_multi_horizon(
        dataset,
        horizons=(1, 5),
    )

    for horizon_result in (
        result.results.values()
    ):
        assert (
            horizon_result.target_column
            in dataset.direction_columns
        )


def test_horizon_results_are_independent():
    dataset = make_model_dataset()

    result = train_multi_horizon(
        dataset,
        horizons=(1, 3, 5),
    )

    successful_results = [
        item
        for item in result.results.values()
        if item.successful
    ]

    models = [
        item.result.folds[0].model
        for item in successful_results
        if item.result.folds
    ]

    models = [
        model
        for model in models
        if model is not None
    ]

    assert len(models) == len(
        {
            id(model)
            for model in models
        }
    )


def test_horizon_preprocessors_are_independent():
    dataset = make_model_dataset()

    result = train_multi_horizon(
        dataset,
        horizons=(1, 3, 5),
    )

    preprocessors = []

    for horizon_result in (
        result.results.values()
    ):
        for fold in horizon_result.result.folds:
            if fold.preprocessor is not None:
                preprocessors.append(
                    fold.preprocessor
                )

    assert len(preprocessors) == len(
        {
            id(processor)
            for processor in preprocessors
        }
    )


def test_successful_horizon_count_is_correct():
    dataset = make_model_dataset()

    result = train_multi_horizon(
        dataset,
        horizons=(1, 3),
    )

    assert (
        result.successful_count
        == len(result.successful_horizons)
    )


def test_failed_horizon_count_is_correct():
    dataset = make_model_dataset()

    result = train_multi_horizon(
        dataset,
        horizons=(1, 3),
    )

    assert (
        result.failed_count
        == len(result.failed_horizons)
    )


def test_successful_and_failed_horizons_do_not_overlap():
    dataset = make_model_dataset()

    result = train_multi_horizon(
        dataset,
        horizons=(1, 3, 5),
    )

    assert not (
        set(result.successful_horizons)
        & set(result.failed_horizons)
    )


def test_all_requested_horizons_are_accounted_for():
    dataset = make_model_dataset()

    horizons = (
        1,
        3,
        5,
    )

    result = train_multi_horizon(
        dataset,
        horizons=horizons,
    )

    accounted = (
        set(result.successful_horizons)
        | set(result.failed_horizons)
    )

    assert accounted == set(
        horizons
    )


def test_get_returns_requested_horizon():
    dataset = make_model_dataset()

    result = train_multi_horizon(
        dataset,
        horizons=(1, 5),
    )

    if 5 in result.results:
        horizon_result = result.get(5)

        assert (
            horizon_result.horizon
            == 5
        )


def test_get_unknown_horizon_is_rejected():
    dataset = make_model_dataset()

    result = train_multi_horizon(
        dataset,
        horizons=(1,),
    )

    with pytest.raises(KeyError):
        result.get(20)


def test_horizon_walk_forward_predictions_exist():
    dataset = make_model_dataset()

    result = train_multi_horizon(
        dataset,
        horizons=(1, 5),
    )

    for horizon_result in (
        result.results.values()
    ):
        if horizon_result.successful:
            assert not (
                horizon_result.result.oos_predictions.empty
            )


def test_horizon_accuracy_is_valid():
    dataset = make_model_dataset()

    result = train_multi_horizon(
        dataset,
        horizons=(1, 5),
    )

    for horizon_result in (
        result.results.values()
    ):
        if horizon_result.successful:
            assert (
                0.0
                <= horizon_result.mean_accuracy
                <= 1.0
            )

            assert (
                0.0
                <= horizon_result.minimum_accuracy
                <= 1.0
            )

            assert (
                horizon_result.accuracy_std
                >= 0.0
            )


def test_candidate_pass_requires_walk_forward_candidate():
    dataset = make_model_dataset()

    result = train_multi_horizon(
        dataset,
        horizons=(1, 5),
    )

    for horizon_result in (
        result.results.values()
    ):
        assert (
            horizon_result.candidate_passed
            == horizon_result.result.candidate_passed
        )


def test_multi_horizon_candidate_gate_is_conservative():
    dataset = make_model_dataset()

    config = make_config(
        horizons=(1, 5)
    )

    result = train_multi_horizon(
        dataset,
        config=config,
    )

    expected = (
        len(result.successful_horizons)
        >= config.minimum_successful_horizons
        and sum(
            item.candidate_passed
            for item in result.results.values()
        )
        >= config.minimum_successful_horizons
    )

    assert (
        result.candidate_passed
        == expected
    )


def test_final_holdout_is_never_used():
    dataset = make_model_dataset()

    result = train_multi_horizon(
        dataset,
        horizons=(1, 3, 5),
    )

    assert (
        result.metadata[
            "final_holdout_used"
        ]
        is False
    )

    for horizon_result in (
        result.results.values()
    ):
        assert (
            horizon_result.metadata[
                "final_holdout_used"
            ]
            is False
        )

        assert (
            horizon_result.result.metadata[
                "final_holdout_used"
            ]
            is False
        )

        for fold in (
            horizon_result.result.folds
        ):
            assert (
                fold.metadata[
                    "final_holdout_used"
                ]
                is False
            )


def test_calibration_is_not_performed():
    dataset = make_model_dataset()

    result = train_multi_horizon(
        dataset,
        horizons=(1, 5),
    )

    assert (
        result.metadata[
            "calibration_fitted"
        ]
        is False
    )

    for horizon_result in (
        result.results.values()
    ):
        assert (
            horizon_result.metadata[
                "calibration_fitted"
            ]
            is False
        )


def test_model_selection_is_not_performed():
    dataset = make_model_dataset()

    result = train_multi_horizon(
        dataset,
        horizons=(1, 5),
    )

    assert (
        result.metadata[
            "model_selected"
        ]
        is False
    )


def test_threshold_optimization_is_not_performed():
    dataset = make_model_dataset()

    result = train_multi_horizon(
        dataset,
        horizons=(1, 5),
    )

    assert (
        result.metadata[
            "threshold_optimization_completed"
        ]
        is False
    )


def test_production_approval_is_false():
    dataset = make_model_dataset()

    result = train_multi_horizon(
        dataset,
        horizons=(1, 5),
    )

    assert (
        result.metadata[
            "production_approved"
        ]
        is False
    )

    assert (
        result.production_ready
        is False
    )


def test_research_only_boundary():
    dataset = make_model_dataset()

    result = train_multi_horizon(
        dataset,
        horizons=(1, 3, 5),
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


def test_preferred_horizon_is_preserved():
    dataset = make_model_dataset()

    config = MultiHorizonTrainingConfig(
        horizons=(1, 3, 5),
        preferred_horizon=3,
    )

    result = train_multi_horizon(
        dataset,
        config=config,
    )

    assert (
        result.preferred_horizon
        == 3
    )


def test_horizon_specific_convenience_api():
    dataset = make_model_dataset()

    result = train_multi_horizon(
        dataset,
        horizons=(5,),
    )

    assert result.horizons == (
        5,
    )


def test_config_and_horizons_cannot_both_be_supplied():
    dataset = make_model_dataset()

    config = make_config(
        horizons=(1, 5)
    )

    with pytest.raises(ValueError):
        train_multi_horizon(
            dataset,
            horizons=(3,),
            config=config,
        )


def test_wrong_dataset_type_is_rejected():
    pipeline = MultiHorizonTrainingPipeline(
        config=make_config(
            horizons=(1,)
        )
    )

    with pytest.raises(TypeError):
        pipeline.train(
            None
        )


def test_dataset_without_direction_targets_is_rejected():
    dataset = make_model_dataset()

    dataset.direction_columns = []

    pipeline = MultiHorizonTrainingPipeline(
        config=make_config(
            horizons=(1,)
        )
    )

    with pytest.raises(ValueError):
        pipeline.train(
            dataset
        )


def test_empty_dataset_is_rejected():
    dataset = make_model_dataset()

    dataset.data = dataset.data.iloc[
        0:0
    ]

    pipeline = MultiHorizonTrainingPipeline(
        config=make_config(
            horizons=(1,)
        )
    )

    with pytest.raises(ValueError):
        pipeline.train(
            dataset
        )


def test_no_feature_dataset_is_rejected():
    dataset = make_model_dataset()

    dataset.feature_columns = []

    pipeline = MultiHorizonTrainingPipeline(
        config=make_config(
            horizons=(1,)
        )
    )

    with pytest.raises(ValueError):
        pipeline.train(
            dataset
        )


def test_input_dataset_is_not_modified():
    dataset = make_model_dataset()

    original = dataset.data.copy(
        deep=True
    )

    train_multi_horizon(
        dataset,
        horizons=(1, 3),
    )

    pd.testing.assert_frame_equal(
        dataset.data,
        original,
    )


def test_feature_columns_are_not_modified():
    dataset = make_model_dataset()

    original = list(
        dataset.feature_columns
    )

    train_multi_horizon(
        dataset,
        horizons=(1, 3),
    )

    assert (
        dataset.feature_columns
        == original
    )


def test_deterministic_for_same_input():
    dataset = make_model_dataset()

    config = make_config(
        horizons=(1, 3)
    )

    first = train_multi_horizon(
        dataset,
        config=config,
    )

    second = train_multi_horizon(
        dataset,
        config=config,
    )

    assert (
        first.horizons
        == second.horizons
    )

    assert (
        first.successful_horizons
        == second.successful_horizons
    )

    assert (
        first.failed_horizons
        == second.failed_horizons
    )

    assert (
        first.candidate_passed
        == second.candidate_passed
    )

    for horizon in first.results:
        first_result = first.results[
            horizon
        ]

        second_result = second.results[
            horizon
        ]

        assert np.isclose(
            first_result.mean_accuracy,
            second_result.mean_accuracy,
        )

        assert np.isclose(
            first_result.minimum_accuracy,
            second_result.minimum_accuracy,
        )

        assert np.isclose(
            first_result.accuracy_std,
            second_result.accuracy_std,
        )

        pd.testing.assert_frame_equal(
            first_result.result.oos_predictions,
            second_result.result.oos_predictions,
        )


def test_summary_returns_dictionary():
    dataset = make_model_dataset()

    result = train_multi_horizon(
        dataset,
        horizons=(1, 3),
    )

    summary = multi_horizon_training_summary(
        result
    )

    assert isinstance(
        summary,
        dict,
    )


def test_summary_contains_horizons():
    dataset = make_model_dataset()

    result = train_multi_horizon(
        dataset,
        horizons=(1, 3, 5),
    )

    summary = multi_horizon_training_summary(
        result
    )

    assert (
        summary["horizons"]
        == [1, 3, 5]
    )


def test_summary_contains_horizon_results():
    dataset = make_model_dataset()

    result = train_multi_horizon(
        dataset,
        horizons=(1, 5),
    )

    summary = multi_horizon_training_summary(
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


def test_summary_contains_research_boundary():
    dataset = make_model_dataset()

    result = train_multi_horizon(
        dataset,
        horizons=(1,),
    )

    summary = multi_horizon_training_summary(
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
        multi_horizon_training_summary(
            None
        )
