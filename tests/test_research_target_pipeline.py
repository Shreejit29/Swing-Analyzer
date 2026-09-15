"""
Tests for the controlled research target pipeline.
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
from src.research.research_target_pipeline import (
    DEFAULT_RESEARCH_HORIZONS,
    ResearchTargetPipeline,
    ResearchTargetPipelineConfig,
    ResearchTargetPipelineResult,
    build_research_targets,
    research_target_pipeline_summary,
)


def make_ohlcv(
    rows: int = 800,
) -> pd.DataFrame:
    index = pd.date_range(
        "2020-01-01",
        periods=rows,
        freq="D",
    )

    trend = np.linspace(
        0,
        40,
        rows,
    )

    cycle = (
        np.sin(
            np.arange(rows) / 12.0
        )
        * 2.0
    )

    close = (
        100.0
        + trend
        + cycle
    )

    open_price = (
        close
        + np.sin(
            np.arange(rows) / 8.0
        )
        * 0.3
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


def test_default_horizons():
    assert DEFAULT_RESEARCH_HORIZONS == (
        1,
        3,
        5,
        10,
        20,
    )


def test_config_defaults():
    config = ResearchTargetPipelineConfig()

    assert config.horizons == (
        1,
        3,
        5,
        10,
        20,
    )

    assert config.direction_threshold == 0.0
    assert config.reject_future_features is True


def test_custom_horizons():
    config = ResearchTargetPipelineConfig(
        horizons=(
            1,
            5,
            10,
        )
    )

    assert config.horizons == (
        1,
        5,
        10,
    )


def test_empty_horizons_are_rejected():
    with pytest.raises(ValueError):
        ResearchTargetPipelineConfig(
            horizons=()
        )


def test_non_positive_horizon_is_rejected():
    with pytest.raises(ValueError):
        ResearchTargetPipelineConfig(
            horizons=(
                1,
                0,
                5,
            )
        )


def test_duplicate_horizons_are_rejected():
    with pytest.raises(ValueError):
        ResearchTargetPipelineConfig(
            horizons=(
                1,
                5,
                5,
            )
        )


def test_negative_direction_threshold_is_rejected():
    with pytest.raises(ValueError):
        ResearchTargetPipelineConfig(
            direction_threshold=-0.01
        )


def test_pipeline_construction():
    pipeline = ResearchTargetPipeline()

    assert isinstance(
        pipeline,
        ResearchTargetPipeline,
    )


def test_target_pipeline_returns_correct_type():
    features = make_feature_result()

    result = build_research_targets(
        features
    )

    assert isinstance(
        result,
        ResearchTargetPipelineResult,
    )


def test_target_pipeline_creates_targets():
    features = make_feature_result()

    result = build_research_targets(
        features,
        config=ResearchTargetPipelineConfig(
            horizons=(
                1,
                3,
                5,
            )
        ),
    )

    assert result.target_count > 0
    assert len(
        result.target_columns
    ) > 0


def test_requested_horizons_are_recorded():
    features = make_feature_result()

    horizons = (
        1,
        3,
        5,
        10,
        20,
    )

    result = build_research_targets(
        features,
        config=ResearchTargetPipelineConfig(
            horizons=horizons
        ),
    )

    assert result.horizons == horizons


def test_feature_target_overlap_is_zero():
    features = make_feature_result()

    result = build_research_targets(
        features
    )

    assert not (
        set(result.feature_columns)
        & set(result.target_columns)
    )


def test_future_targets_are_not_features():
    features = make_feature_result()

    result = build_research_targets(
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


def test_feature_matrix_excludes_targets():
    features = make_feature_result()

    result = build_research_targets(
        features
    )

    matrix = result.feature_matrix()

    assert list(
        matrix.columns
    ) == result.feature_columns

    for target in result.target_columns:
        assert target not in matrix.columns


def test_target_matrix_contains_only_targets():
    features = make_feature_result()

    result = build_research_targets(
        features
    )

    matrix = result.target_matrix()

    assert list(
        matrix.columns
    ) == result.target_columns


def test_direction_columns_are_identified():
    features = make_feature_result()

    result = build_research_targets(
        features,
        config=ResearchTargetPipelineConfig(
            horizons=(
                1,
                5,
            )
        ),
    )

    assert len(
        result.direction_columns
    ) > 0

    for column in result.direction_columns:
        assert column.lower().startswith(
            "direction_"
        )


def test_return_columns_are_identified():
    features = make_feature_result()

    result = build_research_targets(
        features,
        config=ResearchTargetPipelineConfig(
            horizons=(
                1,
                5,
            )
        ),
    )

    assert len(
        result.return_columns
    ) > 0


def test_target_columns_are_numeric():
    features = make_feature_result()

    result = build_research_targets(
        features
    )

    for column in result.target_columns:
        assert pd.api.types.is_numeric_dtype(
            result.data[column]
        )


def test_target_dataframe_preserves_timeline():
    features = make_feature_result()

    result = build_research_targets(
        features
    )

    assert result.data.index.equals(
        features.features.data.index
    )


def test_target_pipeline_does_not_modify_feature_result():
    features = make_feature_result()

    original = features.features.data.copy(
        deep=True
    )

    build_research_targets(
        features
    )

    pd.testing.assert_frame_equal(
        features.features.data,
        original,
    )


def test_target_values_are_finite_when_present():
    features = make_feature_result()

    result = build_research_targets(
        features
    )

    for column in result.target_columns:
        values = (
            result.data[column]
            .dropna()
            .to_numpy()
        )

        assert np.isfinite(
            values
        ).all()


def test_future_targets_create_terminal_missing_values():
    features = make_feature_result()

    result = build_research_targets(
        features,
        config=ResearchTargetPipelineConfig(
            horizons=(
                20,
            )
        ),
    )

    missing_rows = (
        result.data[
            result.target_columns
        ]
        .isna()
        .any(axis=1)
    )

    assert missing_rows.any()


def test_multiple_horizons_produce_distinct_targets():
    features = make_feature_result()

    result = build_research_targets(
        features,
        config=ResearchTargetPipelineConfig(
            horizons=(
                1,
                5,
            )
        ),
    )

    assert len(
        result.target_columns
    ) > 0

    horizon_columns = {}

    for horizon in (
        1,
        5,
    ):
        horizon_columns[horizon] = [
            column
            for column
            in result.target_columns
            if str(horizon)
            in column
        ]

    assert any(
        horizon_columns[1]
    )

    assert any(
        horizon_columns[5]
    )


def test_target_pipeline_is_deterministic():
    features = make_feature_result()

    config = ResearchTargetPipelineConfig(
        horizons=(
            1,
            3,
            5,
        )
    )

    first = build_research_targets(
        features,
        config=config,
    )

    second = build_research_targets(
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


def test_summary_contains_expected_fields():
    features = make_feature_result()

    result = build_research_targets(
        features
    )

    summary = research_target_pipeline_summary(
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
        summary["research_only"]
        is True
    )

    assert (
        summary["production_ready"]
        is False
    )


def test_summary_rejects_wrong_type():
    with pytest.raises(TypeError):
        research_target_pipeline_summary(
            None
        )


def test_production_ready_is_always_false():
    features = make_feature_result()

    result = build_research_targets(
        features
    )

    assert result.production_ready is False


def test_research_only_metadata():
    features = make_feature_result()

    result = build_research_targets(
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
            "future_targets_created"
        ]
        is True
    )

    assert (
        result.metadata[
            "future_targets_used_as_features"
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


def test_failed_feature_pipeline_is_rejected():
    features = make_feature_result()

    features.success = False

    pipeline = ResearchTargetPipeline()

    with pytest.raises(ValueError):
        pipeline.build(
            features
        )


def test_wrong_input_type_is_rejected():
    pipeline = ResearchTargetPipeline()

    with pytest.raises(TypeError):
        pipeline.build(
            None
        )


def test_feature_matrix_is_independent_copy():
    features = make_feature_result()

    result = build_research_targets(
        features
    )

    matrix = result.feature_matrix()

    assert not matrix.empty

    column = matrix.columns[0]
    index = matrix.index[0]

    original = result.data.loc[
        index,
        column,
    ]

    matrix.loc[
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

    result = build_research_targets(
        features
    )

    matrix = result.target_matrix()

    assert not matrix.empty

    column = matrix.columns[0]
    index = matrix.index[0]

    original = result.data.loc[
        index,
        column,
    ]

    matrix.loc[
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


def test_target_construction_does_not_add_future_features():
    features = make_feature_result()

    result = build_research_targets(
        features
    )

    feature_matrix = result.feature_matrix()

    suspicious = [
        column
        for column in feature_matrix.columns
        if (
            "future"
            in column.lower()
            or "direction_"
            in column.lower()
            or "target"
            in column.lower()
        )
    ]

    assert suspicious == []


def test_original_feature_count_is_preserved():
    features = make_feature_result()

    result = build_research_targets(
        features
    )

    assert (
        result.feature_count
        == features.features.feature_count
    )


def test_target_pipeline_keeps_original_ohlcv_columns_available():
    features = make_feature_result()

    result = build_research_targets(
        features
    )

    for column in (
        "Open",
        "High",
        "Low",
        "Close",
    ):
        assert column in result.data.columns


def test_target_pipeline_does_not_fit_a_model():
    features = make_feature_result()

    result = build_research_targets(
        features
    )

    assert (
        result.metadata[
            "model_fitted"
        ]
        is False
    )


def test_target_pipeline_does_not_calibrate():
    features = make_feature_result()

    result = build_research_targets(
        features
    )

    assert (
        result.metadata[
            "calibration_fitted"
        ]
        is False
    )


def test_target_pipeline_does_not_select_model():
    features = make_feature_result()

    result = build_research_targets(
        features
    )

    assert (
        result.metadata[
            "model_selected"
        ]
        is False
    )


def test_target_pipeline_does_not_approve_model():
    features = make_feature_result()

    result = build_research_targets(
        features
    )

    assert (
        result.metadata[
            "approval_granted"
        ]
        is False
    )
