"""
Tests for the controlled research feature pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.data_preparation import (
    DataPreparationConfig,
    PreparedMarketData,
    prepare_market_data,
)
from src.research.feature_integration import (
    UnifiedFeatureConfig,
)
from src.research.research_feature_pipeline import (
    ResearchFeaturePipeline,
    ResearchFeaturePipelineConfig,
    ResearchFeaturePipelineResult,
    build_research_features,
    research_feature_pipeline_summary,
)


def make_ohlcv(
    rows: int = 700,
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
            35,
            rows,
        )
        + np.sin(
            np.arange(rows) / 14.0
        )
    )

    open_price = (
        close
        + np.sin(
            np.arange(rows) / 8.0
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


def prepared_data(
    rows: int = 700,
) -> PreparedMarketData:
    return prepare_market_data(
        make_ohlcv(rows),
        symbol="TCS.NS",
        timeframe="1D",
        config=DataPreparationConfig(
            minimum_rows=100,
        ),
    )


def basic_config() -> ResearchFeaturePipelineConfig:
    return ResearchFeaturePipelineConfig(
        unified=UnifiedFeatureConfig(
            include_market_context=False,
            include_relative_strength=False,
            include_multi_timeframe=False,
        ),
        minimum_features=10,
    )


def test_config_construction():
    config = ResearchFeaturePipelineConfig()

    assert isinstance(
        config.unified,
        UnifiedFeatureConfig,
    )

    assert config.minimum_features > 0
    assert config.require_numeric_features is True
    assert config.reject_future_features is True


def test_invalid_config_is_rejected():
    with pytest.raises(TypeError):
        ResearchFeaturePipelineConfig(
            unified="invalid"
        )

    with pytest.raises(ValueError):
        ResearchFeaturePipelineConfig(
            minimum_features=0
        )


def test_pipeline_construction():
    pipeline = ResearchFeaturePipeline()

    assert isinstance(
        pipeline,
        ResearchFeaturePipeline,
    )


def test_build_returns_correct_result_type():
    prepared = prepared_data()

    pipeline = ResearchFeaturePipeline(
        config=basic_config()
    )

    result = pipeline.build(
        prepared
    )

    assert isinstance(
        result,
        ResearchFeaturePipelineResult,
    )


def test_feature_pipeline_succeeds():
    prepared = prepared_data()

    pipeline = ResearchFeaturePipeline(
        config=basic_config()
    )

    result = pipeline.build(
        prepared
    )

    assert result.success is True
    assert result.feature_count >= 10
    assert result.rows == len(
        prepared.data
    )


def test_feature_matrix_matches_declared_features():
    prepared = prepared_data()

    pipeline = ResearchFeaturePipeline(
        config=basic_config()
    )

    result = pipeline.build(
        prepared
    )

    assert list(
        result.feature_matrix.columns
    ) == result.features.feature_names


def test_feature_matrix_is_numeric():
    prepared = prepared_data()

    pipeline = ResearchFeaturePipeline(
        config=basic_config()
    )

    result = pipeline.build(
        prepared
    )

    for column in (
        result.feature_matrix.columns
    ):
        assert pd.api.types.is_numeric_dtype(
            result.feature_matrix[column]
        )


def test_feature_index_is_chronological():
    prepared = prepared_data()

    pipeline = ResearchFeaturePipeline(
        config=basic_config()
    )

    result = pipeline.build(
        prepared
    )

    assert (
        result.feature_matrix.index
        .is_monotonic_increasing
    )

    assert not (
        result.feature_matrix.index
        .has_duplicates
    )


def test_feature_timeline_matches_prepared_data():
    prepared = prepared_data()

    pipeline = ResearchFeaturePipeline(
        config=basic_config()
    )

    result = pipeline.build(
        prepared
    )

    assert (
        result.feature_matrix.index.equals(
            prepared.data.index
        )
    )


def test_no_future_features():
    prepared = prepared_data()

    pipeline = ResearchFeaturePipeline(
        config=basic_config()
    )

    result = pipeline.build(
        prepared
    )

    suspicious = [
        column
        for column
        in result.feature_matrix.columns
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


def test_input_prepared_data_is_not_modified():
    prepared = prepared_data()

    original = prepared.data.copy(
        deep=True
    )

    pipeline = ResearchFeaturePipeline(
        config=basic_config()
    )

    pipeline.build(
        prepared
    )

    pd.testing.assert_frame_equal(
        prepared.data,
        original,
    )


def test_research_only_metadata():
    prepared = prepared_data()

    pipeline = ResearchFeaturePipeline(
        config=basic_config()
    )

    result = pipeline.build(
        prepared
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
            "calibration_fitted"
        ]
        is False
    )


def test_production_ready_is_always_false():
    prepared = prepared_data()

    result = build_research_features(
        prepared,
        config=basic_config(),
    )

    assert result.production_ready is False


def test_market_context_can_be_enabled():
    """
    This test verifies that enabling market context does not
    bypass the feature-pipeline boundary.

    A context object is intentionally not constructed here;
    the no-context path is separately tested.
    """

    prepared = prepared_data()

    config = ResearchFeaturePipelineConfig(
        unified=UnifiedFeatureConfig(
            include_market_context=True,
            include_relative_strength=True,
            include_multi_timeframe=False,
        ),
        minimum_features=10,
    )

    result = build_research_features(
        prepared,
        market_context=None,
        config=config,
    )

    assert isinstance(
        result,
        ResearchFeaturePipelineResult,
    )

    assert result.feature_count >= 10

    assert any(
        "market context"
        in warning.lower()
        for warning in result.warnings
    )


def test_multi_timeframe_disabled_does_not_require_data():
    prepared = prepared_data()

    config = ResearchFeaturePipelineConfig(
        unified=UnifiedFeatureConfig(
            include_market_context=False,
            include_relative_strength=False,
            include_multi_timeframe=False,
        ),
        minimum_features=10,
    )

    result = build_research_features(
        prepared,
        timeframe_data=None,
        config=config,
    )

    assert result.success is True


def test_feature_count_requirement_is_enforced():
    prepared = prepared_data()

    config = ResearchFeaturePipelineConfig(
        unified=UnifiedFeatureConfig(
            include_market_context=False,
            include_relative_strength=False,
            include_multi_timeframe=False,
        ),
        minimum_features=10_000,
    )

    with pytest.raises(
        ValueError
    ):
        build_research_features(
            prepared,
            config=config,
        )


def test_failed_preparation_is_rejected():
    prepared = prepared_data()

    prepared.quality_report.passed = False

    pipeline = ResearchFeaturePipeline(
        config=basic_config()
    )

    with pytest.raises(
        ValueError
    ):
        pipeline.build(
            prepared
        )


def test_wrong_prepared_type_is_rejected():
    pipeline = ResearchFeaturePipeline(
        config=basic_config()
    )

    with pytest.raises(
        TypeError
    ):
        pipeline.build(
            "invalid"
        )


def test_feature_pipeline_is_deterministic():
    prepared = prepared_data()

    config = basic_config()

    first = build_research_features(
        prepared,
        config=config,
    )

    second = build_research_features(
        prepared,
        config=config,
    )

    assert (
        first.features.feature_names
        == second.features.feature_names
    )

    pd.testing.assert_frame_equal(
        first.feature_matrix,
        second.feature_matrix,
    )


def test_summary_contains_expected_fields():
    prepared = prepared_data()

    result = build_research_features(
        prepared,
        config=basic_config(),
    )

    summary = research_feature_pipeline_summary(
        result
    )

    assert (
        summary["success"]
        == result.success
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
        summary["production_ready"]
        is False
    )

    assert (
        summary["research_only"]
        is True
    )


def test_summary_rejects_wrong_type():
    with pytest.raises(
        TypeError
    ):
        research_feature_pipeline_summary(
            None
        )


def test_feature_matrix_is_independent_copy():
    prepared = prepared_data()

    result = build_research_features(
        prepared,
        config=basic_config(),
    )

    matrix = result.feature_matrix

    assert not matrix.empty

    column = matrix.columns[0]
    index = matrix.index[0]

    original_value = matrix.loc[
        index,
        column,
    ]

    matrix.loc[
        index,
        column,
    ] = 999999.0

    assert (
        result.features.data.loc[
            index,
            column,
        ]
        == original_value
    )


def test_feature_matrix_contains_no_infinite_values():
    prepared = prepared_data()

    result = build_research_features(
        prepared,
        config=basic_config(),
    )

    values = (
        result.feature_matrix
        .fillna(0.0)
        .to_numpy()
    )

    assert np.isfinite(
        values
    ).all()


def test_convenience_function_matches_pipeline():
    prepared = prepared_data()

    config = basic_config()

    pipeline = ResearchFeaturePipeline(
        config=config
    )

    direct = pipeline.build(
        prepared
    )

    convenience = build_research_features(
        prepared,
        config=config,
    )

    assert (
        direct.features.feature_names
        == convenience.features.feature_names
    )

    pd.testing.assert_frame_equal(
        direct.feature_matrix,
        convenience.feature_matrix,
    )


def test_final_holdout_is_not_used():
    prepared = prepared_data()

    result = build_research_features(
        prepared,
        config=basic_config(),
    )

    assert (
        result.metadata[
            "final_holdout_used"
        ]
        is False
    )


def test_model_selection_not_performed():
    prepared = prepared_data()

    result = build_research_features(
        prepared,
        config=basic_config(),
    )

    assert (
        result.metadata[
            "feature_selection_completed"
        ]
        is False
    )


def test_model_training_not_performed():
    prepared = prepared_data()

    result = build_research_features(
        prepared,
        config=basic_config(),
    )

    assert (
        result.metadata[
            "model_fitted"
        ]
        is False
    )


def test_target_generation_not_performed():
    prepared = prepared_data()

    result = build_research_features(
        prepared,
        config=basic_config(),
    )

    assert (
        result.metadata[
            "targets_created"
        ]
        is False
    )
