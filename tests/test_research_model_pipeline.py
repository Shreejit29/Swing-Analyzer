"""
Tests for the model-ready research dataset pipeline.
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
    ResearchModelDataset,
    ResearchModelPipeline,
    ResearchModelPipelineConfig,
    build_horizon_dataset,
    build_model_dataset,
    research_model_dataset_summary,
)
from src.research.research_target_pipeline import (
    ResearchTargetPipelineConfig,
)


def make_ohlcv(
    rows: int = 800,
) -> pd.DataFrame:
    index = pd.date_range(
        "2020-01-01",
        periods=rows,
        freq="D",
    )

    close = (
        100.0
        + np.linspace(
            0,
            40,
            rows,
        )
        + np.sin(
            np.arange(rows) / 13.0
        ) * 2.0
    )

    open_price = (
        close
        + np.sin(
            np.arange(rows) / 9.0
        )
        * 0.25
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
        + np.arange(rows)
        * 100
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


def make_feature_result():
    prepared = prepare_market_data(
        make_ohlcv(),
        symbol="TCS.NS",
        timeframe="1D",
        config=DataPreparationConfig(
            minimum_rows=100,
        ),
    )

    pipeline = ResearchFeaturePipeline(
        config=ResearchFeaturePipelineConfig(
            unified=UnifiedFeatureConfig(
                include_market_context=False,
                include_relative_strength=False,
                include_multi_timeframe=False,
            ),
            minimum_features=10,
        )
    )

    return pipeline.build(
        prepared
    )


def test_config_defaults():
    config = ResearchModelPipelineConfig()

    assert isinstance(
        config.target,
        ResearchTargetPipelineConfig,
    )

    assert config.minimum_rows > 0
    assert config.drop_rows_with_missing_targets is False
    assert config.reject_future_features is True


def test_config_rejects_invalid_target_config():
    with pytest.raises(TypeError):
        ResearchModelPipelineConfig(
            target="invalid"
        )


def test_config_rejects_invalid_minimum_rows():
    with pytest.raises(ValueError):
        ResearchModelPipelineConfig(
            minimum_rows=0
        )


def test_pipeline_construction():
    pipeline = ResearchModelPipeline()

    assert isinstance(
        pipeline,
        ResearchModelPipeline,
    )


def test_model_dataset_is_created():
    features = make_feature_result()

    result = build_model_dataset(
        features,
        config=ResearchModelPipelineConfig(
            target=ResearchTargetPipelineConfig(
                horizons=(
                    1,
                    5,
                )
            ),
            minimum_rows=100,
        ),
    )

    assert isinstance(
        result,
        ResearchModelDataset,
    )

    assert result.rows > 0
    assert result.feature_count > 0
    assert result.target_count > 0


def test_feature_target_overlap_is_zero():
    features = make_feature_result()

    result = build_model_dataset(
        features,
        config=ResearchModelPipelineConfig(
            target=ResearchTargetPipelineConfig(
                horizons=(
                    1,
                    5,
                )
            )
        ),
    )

    assert not (
        set(result.feature_columns)
        & set(result.target_columns)
    )


def test_feature_matrix_contains_only_features():
    features = make_feature_result()

    result = build_model_dataset(
        features
    )

    X = result.feature_matrix()

    assert list(
        X.columns
    ) == result.feature_columns

    for target in result.target_columns:
        assert target not in X.columns


def test_target_matrix_contains_only_targets():
    features = make_feature_result()

    result = build_model_dataset(
        features
    )

    y = result.target_matrix()

    assert list(
        y.columns
    ) == result.target_columns


def test_supervised_dataset_removes_missing_targets():
    features = make_feature_result()

    result = build_model_dataset(
        features,
        config=ResearchModelPipelineConfig(
            target=ResearchTargetPipelineConfig(
                horizons=(20,)
            )
        ),
    )

    X, y = result.supervised_dataset()

    assert len(X) == len(y)

    assert (
        y.notna()
        .all()
        .all()
    )

    assert len(X) <= result.rows


def test_supervised_dataset_can_select_specific_target():
    features = make_feature_result()

    result = build_model_dataset(
        features,
        config=ResearchModelPipelineConfig(
            target=ResearchTargetPipelineConfig(
                horizons=(
                    1,
                    5,
                )
            )
        ),
    )

    direction_targets = (
        result.direction_columns
    )

    assert direction_targets

    X, y = result.supervised_dataset(
        target_columns=[
            direction_targets[0]
        ]
    )

    assert len(X) == len(y)
    assert list(
        y.columns
    ) == [
        direction_targets[0]
    ]


def test_unknown_target_is_rejected():
    features = make_feature_result()

    result = build_model_dataset(
        features
    )

    with pytest.raises(ValueError):
        result.supervised_dataset(
            target_columns=[
                "NOT_A_REAL_TARGET"
            ]
        )


def test_empty_target_selection_is_rejected():
    features = make_feature_result()

    result = build_model_dataset(
        features
    )

    with pytest.raises(ValueError):
        result.supervised_dataset(
            target_columns=[]
        )


def test_model_dataset_preserves_timeline():
    features = make_feature_result()

    result = build_model_dataset(
        features
    )

    assert result.data.index.equals(
        features.features.data.index
    )


def test_model_dataset_is_chronologically_sorted():
    features = make_feature_result()

    result = build_model_dataset(
        features
    )

    assert (
        result.data.index
        .is_monotonic_increasing
    )

    assert not (
        result.data.index
        .has_duplicates
    )


def test_no_future_features_are_present():
    features = make_feature_result()

    result = build_model_dataset(
        features
    )

    suspicious = [
        column
        for column in result.feature_columns
        if any(
            token in column.lower()
            for token in (
                "future_",
                "direction_",
                "target_",
                "target",
                "label",
            )
        )
    ]

    assert suspicious == []


def test_targets_are_present_as_targets():
    features = make_feature_result()

    result = build_model_dataset(
        features
    )

    assert any(
        column.lower().startswith(
            "direction_"
        )
        for column in result.target_columns
    )


def test_future_target_rows_are_retained_by_default():
    features = make_feature_result()

    result = build_model_dataset(
        features,
        config=ResearchModelPipelineConfig(
            target=ResearchTargetPipelineConfig(
                horizons=(20,)
            ),
            drop_rows_with_missing_targets=False,
        ),
    )

    assert result.rows == len(
        features.features.data
    )

    assert (
        result.complete_target_rows
        < result.rows
    )


def test_future_target_rows_can_be_removed():
    features = make_feature_result()

    result = build_model_dataset(
        features,
        config=ResearchModelPipelineConfig(
            target=ResearchTargetPipelineConfig(
                horizons=(20,)
            ),
            drop_rows_with_missing_targets=True,
        ),
    )

    assert result.rows <= len(
        features.features.data
    )


def test_model_dataset_is_independent_copy():
    features = make_feature_result()

    result = build_model_dataset(
        features
    )

    column = result.feature_columns[0]
    index = result.data.index[0]

    original = result.data.loc[
        index,
        column,
    ]

    result.data.loc[
        index,
        column,
    ] = 999999.0

    assert (
        features.features.data.loc[
            index,
            column,
        ]
        == original
    )


def test_feature_matrix_is_independent_copy():
    features = make_feature_result()

    result = build_model_dataset(
        features
    )

    X = result.feature_matrix()

    column = X.columns[0]
    index = X.index[0]

    original = X.loc[
        index,
        column,
    ]

    X.loc[
        index,
        column,
    ] = 999999.0

    assert (
        result.data.loc[
            index,
            column,
        ]
        == original
    )


def test_target_matrix_is_independent_copy():
    features = make_feature_result()

    result = build_model_dataset(
        features
    )

    y = result.target_matrix()

    column = y.columns[0]
    index = y.index[0]

    original = y.loc[
        index,
        column,
    ]

    y.loc[
        index,
        column,
    ] = 999999.0

    assert (
        result.data.loc[
            index,
            column,
        ]
        == original
    )


def test_horizon_specific_dataset():
    features = make_feature_result()

    result = build_horizon_dataset(
        features,
        5,
    )

    assert result.horizons == (
        5,
    )

    assert result.target_count > 0


def test_invalid_horizon_is_rejected():
    features = make_feature_result()

    with pytest.raises(ValueError):
        build_horizon_dataset(
            features,
            0,
        )

    with pytest.raises(ValueError):
        build_horizon_dataset(
            features,
            -5,
        )


def test_horizon_dataset_contains_only_requested_horizon():
    features = make_feature_result()

    result = build_horizon_dataset(
        features,
        5,
    )

    for column in result.target_columns:
        assert "5" in column


def test_multiple_horizons_are_supported():
    features = make_feature_result()

    result = build_model_dataset(
        features,
        config=ResearchModelPipelineConfig(
            target=ResearchTargetPipelineConfig(
                horizons=(
                    1,
                    3,
                    5,
                    10,
                    20,
                )
            )
        ),
    )

    assert result.horizons == (
        1,
        3,
        5,
        10,
        20,
    )

    assert result.target_count > 0


def test_direction_return_range_and_path_categories():
    features = make_feature_result()

    result = build_model_dataset(
        features
    )

    assert len(
        result.direction_columns
    ) > 0

    assert len(
        result.return_columns
    ) > 0


def test_all_feature_columns_are_numeric():
    features = make_feature_result()

    result = build_model_dataset(
        features
    )

    for column in result.feature_columns:
        assert pd.api.types.is_numeric_dtype(
            result.data[column]
        )


def test_all_target_columns_are_numeric():
    features = make_feature_result()

    result = build_model_dataset(
        features
    )

    for column in result.target_columns:
        assert pd.api.types.is_numeric_dtype(
            result.data[column]
        )


def test_feature_values_do_not_contain_infinity():
    features = make_feature_result()

    result = build_model_dataset(
        features
    )

    values = (
        result.feature_matrix()
        .fillna(0.0)
        .to_numpy()
    )

    assert np.isfinite(
        values
    ).all()


def test_target_values_do_not_contain_infinity():
    features = make_feature_result()

    result = build_model_dataset(
        features
    )

    values = (
        result.target_matrix()
        .dropna()
        .to_numpy()
    )

    assert np.isfinite(
        values
    ).all()


def test_research_only_metadata():
    features = make_feature_result()

    result = build_model_dataset(
        features
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

    assert (
        result.metadata[
            "final_holdout_used"
        ]
        is False
    )

    assert (
        result.metadata[
            "model_fitted"
        ]
        is False
    )

    assert (
        result.metadata[
            "model_selected"
        ]
        is False
    )

    assert (
        result.metadata[
            "calibration_fitted"
        ]
        is False
    )


def test_production_ready_is_false():
    features = make_feature_result()

    result = build_model_dataset(
        features
    )

    assert result.production_ready is False


def test_summary_contains_expected_values():
    features = make_feature_result()

    result = build_model_dataset(
        features
    )

    summary = research_model_dataset_summary(
        result
    )

    assert (
        summary["rows"]
        == result.rows
    )

    assert (
        summary["feature_count"]
        == result.feature_count
    )

    assert (
        summary["target_count"]
        == result.target_count
    )

    assert (
        summary["complete_target_rows"]
        == result.complete_target_rows
    )

    assert (
        summary["research_only"]
        is True
    )

    assert (
        summary["production_ready"]
        is False
    )


def test_summary_rejects_wrong_type():
    with pytest.raises(TypeError):
        research_model_dataset_summary(
            None
        )


def test_wrong_feature_result_type_is_rejected():
    pipeline = ResearchModelPipeline()

    with pytest.raises(TypeError):
        pipeline.build(
            None
        )


def test_failed_feature_result_is_rejected():
    features = make_feature_result()

    features.success = False

    pipeline = ResearchModelPipeline()

    with pytest.raises(ValueError):
        pipeline.build(
            features
        )


def test_feature_target_timeline_mismatch_is_rejected():
    features = make_feature_result()

    pipeline = ResearchModelPipeline()

    original_build = (
        pipeline
    )

    # The normal pipeline constructs both datasets from
    # the same timeline. This test verifies the public object
    # remains protected against an externally altered feature
    # timeline.
    features.feature_matrix = (
        features.feature_matrix.iloc[
            :-1
        ]
    )

    with pytest.raises(
        (ValueError, AttributeError)
    ):
        original_build.build(
            features
        )


def test_model_dataset_is_deterministic():
    features = make_feature_result()

    config = ResearchModelPipelineConfig(
        target=ResearchTargetPipelineConfig(
            horizons=(
                1,
                5,
            )
        )
    )

    first = build_model_dataset(
        features,
        config=config,
    )

    second = build_model_dataset(
        features,
        config=config,
    )

    assert (
        first.feature_columns
        == second.feature_columns
    )

    assert (
        first.target_columns
        == second.target_columns
    )

    pd.testing.assert_frame_equal(
        first.data,
        second.data,
    )


def test_original_feature_data_is_unchanged():
    features = make_feature_result()

    original = features.features.data.copy(
        deep=True
    )

    build_model_dataset(
        features
    )

    pd.testing.assert_frame_equal(
        features.features.data,
        original,
    )


def test_model_dataset_does_not_fit_models():
    features = make_feature_result()

    result = build_model_dataset(
        features
    )

    assert (
        result.metadata[
            "model_fitted"
        ]
        is False
    )


def test_model_dataset_does_not_approve_models():
    features = make_feature_result()

    result = build_model_dataset(
        features
    )

    assert (
        result.metadata[
            "approval_granted"
        ]
        is False
    )
