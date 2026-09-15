"""
Tests for the leakage-safe walk-forward training pipeline.
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
from src.research.walk_forward_training_pipeline import (
    WalkForwardFoldResult,
    WalkForwardTrainingConfig,
    WalkForwardTrainingPipeline,
    WalkForwardTrainingResult,
    train_walk_forward,
    walk_forward_training_summary,
)


def make_ohlcv(
    rows: int = 1200,
) -> pd.DataFrame:
    """Create deterministic synthetic OHLCV data."""

    index = pd.date_range(
        "2019-01-01",
        periods=rows,
        freq="D",
    )

    t = np.arange(rows)

    close = (
        100.0
        + np.linspace(
            0.0,
            80.0,
            rows,
        )
        + 3.0 * np.sin(t / 13.0)
        + 1.2 * np.sin(t / 37.0)
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
    """Build a deterministic model-ready research dataset."""

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


def make_config() -> WalkForwardTrainingConfig:
    return WalkForwardTrainingConfig(
        n_splits=5,
        minimum_training_rows=100,
        minimum_validation_rows=30,
        gap=0,
        embargo=0,
        expanding=True,
        minimum_accuracy=0.95,
        maximum_accuracy_std=0.10,
        random_state=42,
    )


def test_config_defaults():
    config = WalkForwardTrainingConfig()

    assert config.n_splits >= 1
    assert config.minimum_training_rows > 0
    assert config.minimum_validation_rows > 0
    assert 0.0 <= config.minimum_accuracy <= 1.0
    assert config.maximum_accuracy_std >= 0


def test_invalid_model_type_is_rejected():
    with pytest.raises(ValueError):
        WalkForwardTrainingConfig(
            model_type=""
        )


def test_invalid_number_of_splits_is_rejected():
    with pytest.raises(ValueError):
        WalkForwardTrainingConfig(
            n_splits=0
        )


def test_invalid_training_rows_are_rejected():
    with pytest.raises(ValueError):
        WalkForwardTrainingConfig(
            minimum_training_rows=0
        )


def test_invalid_validation_rows_are_rejected():
    with pytest.raises(ValueError):
        WalkForwardTrainingConfig(
            minimum_validation_rows=0
        )


def test_negative_gap_is_rejected():
    with pytest.raises(ValueError):
        WalkForwardTrainingConfig(
            gap=-1
        )


def test_negative_embargo_is_rejected():
    with pytest.raises(ValueError):
        WalkForwardTrainingConfig(
            embargo=-1
        )


def test_invalid_accuracy_threshold_is_rejected():
    with pytest.raises(ValueError):
        WalkForwardTrainingConfig(
            minimum_accuracy=-0.1
        )

    with pytest.raises(ValueError):
        WalkForwardTrainingConfig(
            minimum_accuracy=1.1
        )


def test_negative_accuracy_std_threshold_is_rejected():
    with pytest.raises(ValueError):
        WalkForwardTrainingConfig(
            maximum_accuracy_std=-0.1
        )


def test_invalid_train_size_is_rejected():
    with pytest.raises(ValueError):
        WalkForwardTrainingConfig(
            train_size=0
        )


def test_invalid_validation_size_is_rejected():
    with pytest.raises(ValueError):
        WalkForwardTrainingConfig(
            validation_size=0
        )


def test_pipeline_construction():
    pipeline = WalkForwardTrainingPipeline(
        config=make_config()
    )

    assert isinstance(
        pipeline,
        WalkForwardTrainingPipeline,
    )


def test_model_dataset_is_available():
    dataset = make_model_dataset()

    assert dataset.rows > 0
    assert dataset.feature_count > 0
    assert dataset.direction_columns


def test_walk_forward_returns_result():
    dataset = make_model_dataset()

    result = train_walk_forward(
        dataset,
        config=make_config(),
    )

    assert isinstance(
        result,
        WalkForwardTrainingResult,
    )


def test_walk_forward_creates_multiple_folds():
    dataset = make_model_dataset()

    result = train_walk_forward(
        dataset,
        config=make_config(),
    )

    assert result.fold_count >= 1
    assert result.fold_count <= 5


def test_each_fold_has_correct_result_type():
    dataset = make_model_dataset()

    result = train_walk_forward(
        dataset,
        config=make_config(),
    )

    for fold in result.folds:
        assert isinstance(
            fold,
            WalkForwardFoldResult,
        )


def test_each_fold_is_chronological():
    dataset = make_model_dataset()

    result = train_walk_forward(
        dataset,
        config=make_config(),
    )

    for fold in result.folds:
        assert (
            fold.train_index.is_monotonic_increasing
        )

        assert (
            fold.validation_index.is_monotonic_increasing
        )

        assert (
            fold.train_index[-1]
            < fold.validation_index[0]
        )


def test_each_fold_has_no_train_validation_overlap():
    dataset = make_model_dataset()

    result = train_walk_forward(
        dataset,
        config=make_config(),
    )

    for fold in result.folds:
        assert not (
            set(fold.train_index)
            & set(fold.validation_index)
        )


def test_each_fold_contains_a_fresh_model():
    dataset = make_model_dataset()

    result = train_walk_forward(
        dataset,
        config=make_config(),
    )

    models = [
        fold.model
        for fold in result.folds
    ]

    assert all(
        model is not None
        for model in models
    )

    assert len(
        {
            id(model)
            for model in models
        }
    ) == len(models)


def test_each_fold_contains_a_fresh_preprocessor():
    dataset = make_model_dataset()

    result = train_walk_forward(
        dataset,
        config=make_config(),
    )

    preprocessors = [
        fold.preprocessor
        for fold in result.folds
    ]

    assert all(
        processor is not None
        for processor in preprocessors
    )

    assert len(
        {
            id(processor)
            for processor in preprocessors
        }
    ) == len(preprocessors)


def test_fold_metadata_confirms_fresh_training():
    dataset = make_model_dataset()

    result = train_walk_forward(
        dataset,
        config=make_config(),
    )

    for fold in result.folds:
        assert (
            fold.metadata[
                "fresh_model"
            ]
            is True
        )

        assert (
            fold.metadata[
                "fresh_preprocessor"
            ]
            is True
        )


def test_fold_predictions_match_validation_rows():
    dataset = make_model_dataset()

    result = train_walk_forward(
        dataset,
        config=make_config(),
    )

    for fold in result.folds:
        assert len(
            fold.predictions
        ) == fold.validation_rows

        assert (
            fold.probabilities.shape[0]
            == fold.validation_rows
        )


def test_probability_output_is_binary():
    dataset = make_model_dataset()

    result = train_walk_forward(
        dataset,
        config=make_config(),
    )

    for fold in result.folds:
        probabilities = fold.probabilities

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


def test_fold_predictions_are_binary():
    dataset = make_model_dataset()

    result = train_walk_forward(
        dataset,
        config=make_config(),
    )

    for fold in result.folds:
        unique = set(
            np.asarray(
                fold.predictions
            ).tolist()
        )

        assert unique.issubset(
            {0, 1}
        )


def test_fold_accuracy_is_valid():
    dataset = make_model_dataset()

    result = train_walk_forward(
        dataset,
        config=make_config(),
    )

    for fold in result.folds:
        assert 0.0 <= (
            fold.accuracy
        ) <= 1.0


def test_aggregate_accuracy_statistics_are_valid():
    dataset = make_model_dataset()

    result = train_walk_forward(
        dataset,
        config=make_config(),
    )

    assert 0.0 <= result.mean_accuracy <= 1.0
    assert 0.0 <= result.median_accuracy <= 1.0
    assert 0.0 <= result.minimum_accuracy <= 1.0
    assert 0.0 <= result.maximum_accuracy <= 1.0
    assert result.accuracy_std >= 0.0


def test_accuracy_statistics_have_correct_order():
    dataset = make_model_dataset()

    result = train_walk_forward(
        dataset,
        config=make_config(),
    )

    assert (
        result.minimum_accuracy
        <= result.mean_accuracy
        <= result.maximum_accuracy
    )


def test_oos_predictions_exist():
    dataset = make_model_dataset()

    result = train_walk_forward(
        dataset,
        config=make_config(),
    )

    assert not result.oos_predictions.empty


def test_oos_predictions_are_chronological():
    dataset = make_model_dataset()

    result = train_walk_forward(
        dataset,
        config=make_config(),
    )

    assert (
        result.oos_predictions.index
        .is_monotonic_increasing
    )


def test_oos_predictions_have_unique_timestamps():
    dataset = make_model_dataset()

    result = train_walk_forward(
        dataset,
        config=make_config(),
    )

    assert not (
        result.oos_predictions.index.has_duplicates
    )


def test_oos_rows_equal_fold_validation_rows():
    dataset = make_model_dataset()

    result = train_walk_forward(
        dataset,
        config=make_config(),
    )

    expected = sum(
        fold.validation_rows
        for fold in result.folds
    )

    assert (
        len(result.oos_predictions)
        == expected
    )


def test_oos_predictions_contain_required_columns():
    dataset = make_model_dataset()

    result = train_walk_forward(
        dataset,
        config=make_config(),
    )

    expected_columns = {
        "Actual",
        "Prediction",
        "Probability_Down",
        "Probability_Up",
        "Fold",
    }

    assert expected_columns.issubset(
        set(
            result.oos_predictions.columns
        )
    )


def test_oos_fold_numbers_are_valid():
    dataset = make_model_dataset()

    result = train_walk_forward(
        dataset,
        config=make_config(),
    )

    folds = set(
        result.oos_predictions[
            "Fold"
        ].astype(int)
    )

    assert folds.issubset(
        set(
            range(
                1,
                result.fold_count + 1,
            )
        )
    )


def test_oos_predictions_do_not_include_future_targets_as_features():
    dataset = make_model_dataset()

    result = train_walk_forward(
        dataset,
        config=make_config(),
    )

    for column in result.feature_columns:
        lowered = column.lower()

        assert not lowered.startswith(
            "future_"
        )

        assert not lowered.startswith(
            "direction_"
        )

        assert not lowered.startswith(
            "target_"
        )

        assert lowered not in {
            "target",
            "label",
        }


def test_final_holdout_is_never_used():
    dataset = make_model_dataset()

    result = train_walk_forward(
        dataset,
        config=make_config(),
    )

    assert (
        result.metadata[
            "final_holdout_used"
        ]
        is False
    )

    for fold in result.folds:
        assert (
            fold.metadata[
                "final_holdout_used"
            ]
            is False
        )


def test_calibration_is_not_performed():
    dataset = make_model_dataset()

    result = train_walk_forward(
        dataset,
        config=make_config(),
    )

    assert (
        result.metadata[
            "calibration_fitted"
        ]
        is False
    )


def test_model_selection_is_not_performed():
    dataset = make_model_dataset()

    result = train_walk_forward(
        dataset,
        config=make_config(),
    )

    assert (
        result.metadata[
            "model_selected"
        ]
        is False
    )


def test_threshold_optimization_is_not_performed():
    dataset = make_model_dataset()

    result = train_walk_forward(
        dataset,
        config=make_config(),
    )

    assert (
        result.metadata[
            "threshold_optimization_completed"
        ]
        is False
    )


def test_walk_forward_is_research_only():
    dataset = make_model_dataset()

    result = train_walk_forward(
        dataset,
        config=make_config(),
    )

    assert (
        result.metadata[
            "research_only"
        ]
        is True
    )

    assert (
        result.production_ready
        is False
    )


def test_production_approval_is_false():
    dataset = make_model_dataset()

    result = train_walk_forward(
        dataset,
        config=make_config(),
    )

    assert (
        result.metadata[
            "production_approved"
        ]
        is False
    )


def test_candidate_gate_matches_statistics():
    dataset = make_model_dataset()

    config = make_config()

    result = train_walk_forward(
        dataset,
        config=config,
    )

    expected = (
        result.mean_accuracy
        >= config.minimum_accuracy
        and result.minimum_accuracy
        >= config.minimum_accuracy
        and result.accuracy_std
        <= config.maximum_accuracy_std
    )

    assert (
        result.candidate_passed
        == expected
    )


def test_candidate_gate_is_not_based_on_mean_accuracy_alone():
    dataset = make_model_dataset()

    config = make_config()

    result = train_walk_forward(
        dataset,
        config=config,
    )

    if (
        result.mean_accuracy
        >= config.minimum_accuracy
        and (
            result.minimum_accuracy
            < config.minimum_accuracy
            or result.accuracy_std
            > config.maximum_accuracy_std
        )
    ):
        assert (
            result.candidate_passed
            is False
        )


def test_summary_matches_result():
    dataset = make_model_dataset()

    result = train_walk_forward(
        dataset,
        config=make_config(),
    )

    summary = walk_forward_training_summary(
        result
    )

    assert (
        summary["fold_count"]
        == result.fold_count
    )

    assert (
        summary["total_oos_rows"]
        == result.total_oos_rows
    )

    assert (
        summary["mean_accuracy"]
        == result.mean_accuracy
    )

    assert (
        summary["median_accuracy"]
        == result.median_accuracy
    )

    assert (
        summary["minimum_accuracy"]
        == result.minimum_accuracy
    )

    assert (
        summary["maximum_accuracy"]
        == result.maximum_accuracy
    )

    assert (
        summary["accuracy_std"]
        == result.accuracy_std
    )


def test_summary_rejects_wrong_type():
    with pytest.raises(TypeError):
        walk_forward_training_summary(
            None
        )


def test_explicit_target_is_supported():
    dataset = make_model_dataset()

    target = dataset.direction_columns[0]

    result = train_walk_forward(
        dataset,
        target_column=target,
        config=make_config(),
    )

    assert (
        result.target_column
        == target
    )


def test_invalid_target_is_rejected():
    dataset = make_model_dataset()

    with pytest.raises(ValueError):
        train_walk_forward(
            dataset,
            target_column="NOT_A_TARGET",
            config=make_config(),
        )


def test_input_dataset_is_not_modified():
    dataset = make_model_dataset()

    original = dataset.data.copy(
        deep=True
    )

    train_walk_forward(
        dataset,
        config=make_config(),
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

    train_walk_forward(
        dataset,
        config=make_config(),
    )

    assert (
        dataset.feature_columns
        == original
    )


def test_deterministic_for_same_input():
    dataset = make_model_dataset()

    config = make_config()

    first = train_walk_forward(
        dataset,
        config=config,
    )

    second = train_walk_forward(
        dataset,
        config=config,
    )

    assert (
        first.feature_columns
        == second.feature_columns
    )

    assert (
        first.target_column
        == second.target_column
    )

    assert (
        first.mean_accuracy
        == second.mean_accuracy
    )

    assert (
        first.minimum_accuracy
        == second.minimum_accuracy
    )

    assert (
        first.accuracy_std
        == second.accuracy_std
    )

    pd.testing.assert_frame_equal(
        first.oos_predictions,
        second.oos_predictions,
    )


def test_wrong_dataset_type_is_rejected():
    pipeline = (
        WalkForwardTrainingPipeline(
            config=make_config()
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

    pipeline = (
        WalkForwardTrainingPipeline(
            config=make_config()
        )
    )

    with pytest.raises(ValueError):
        pipeline.train(
            dataset
        )


def test_unsorted_dataset_is_rejected():
    dataset = make_model_dataset()

    dataset.data = dataset.data.iloc[
        ::-1
    ]

    pipeline = (
        WalkForwardTrainingPipeline(
            config=make_config()
        )
    )

    with pytest.raises(ValueError):
        pipeline.train(
            dataset
        )


def test_duplicate_timestamp_dataset_is_rejected():
    dataset = make_model_dataset()

    duplicated = pd.concat(
        [
            dataset.data,
            dataset.data.iloc[
                [0]
            ],
        ]
    )

    duplicated = duplicated.sort_index()

    dataset.data = duplicated

    pipeline = (
        WalkForwardTrainingPipeline(
            config=make_config()
        )
    )

    with pytest.raises(ValueError):
        pipeline.train(
            dataset
        )


def test_summary_reports_research_only_status():
    dataset = make_model_dataset()

    result = train_walk_forward(
        dataset,
        config=make_config(),
    )

    summary = result.summary()

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
