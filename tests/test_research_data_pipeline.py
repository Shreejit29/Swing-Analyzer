"""
Tests for the controlled research data pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.data_preparation import (
    DataPreparationConfig,
)
from src.research.research_data_pipeline import (
    ResearchDataPipeline,
    ResearchDataPipelineConfig,
    ResearchDataPipelineResult,
    research_data_pipeline_summary,
    run_research_data_pipeline,
)


def make_ohlcv(
    rows: int = 500,
) -> pd.DataFrame:
    index = pd.date_range(
        "2020-01-01",
        periods=rows,
        freq="D",
    )

    close = (
        100
        + np.linspace(
            0,
            30,
            rows,
        )
        + np.sin(
            np.arange(rows) / 15
        )
    )

    open_price = (
        close
        + np.sin(
            np.arange(rows) / 9
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


def test_config_construction():
    config = ResearchDataPipelineConfig()

    assert isinstance(
        config.preparation,
        DataPreparationConfig,
    )

    assert (
        config.require_research_dataset
        is True
    )


def test_custom_config_is_preserved():
    preparation = DataPreparationConfig(
        minimum_rows=100,
        require_volume=True,
    )

    config = ResearchDataPipelineConfig(
        preparation=preparation,
        require_research_dataset=True,
    )

    assert (
        config.preparation
        is preparation
    )


def test_invalid_config_is_rejected():
    with pytest.raises(TypeError):
        ResearchDataPipelineConfig(
            preparation="invalid"
        )


def test_pipeline_construction():
    pipeline = ResearchDataPipeline()

    assert isinstance(
        pipeline,
        ResearchDataPipeline,
    )


def test_prepare_stage():
    data = make_ohlcv()

    pipeline = ResearchDataPipeline()

    result = pipeline.prepare(
        data,
        symbol="TCS.NS",
        timeframe="1D",
    )

    assert result.passed is True
    assert result.symbol == "TCS.NS"
    assert result.timeframe == "1D"
    assert result.rows == len(data)


def test_prepare_stage_does_not_modify_input():
    data = make_ohlcv()

    original = data.copy(
        deep=True
    )

    pipeline = ResearchDataPipeline()

    pipeline.prepare(
        data,
        symbol="TCS.NS",
        timeframe="1D",
    )

    pd.testing.assert_frame_equal(
        data,
        original,
    )


def test_full_pipeline_returns_result():
    data = make_ohlcv()

    pipeline = ResearchDataPipeline()

    result = pipeline.run(
        data,
        symbol="RELIANCE.NS",
        timeframe="1D",
        horizons=[
            1,
            3,
            5,
        ],
    )

    assert isinstance(
        result,
        ResearchDataPipelineResult,
    )


def test_full_pipeline_can_build_dataset():
    data = make_ohlcv(
        rows=700
    )

    pipeline = ResearchDataPipeline()

    result = pipeline.run(
        data,
        symbol="RELIANCE.NS",
        timeframe="1D",
        horizons=[
            1,
            3,
            5,
        ],
    )

    assert result.success is True
    assert result.dataset is not None
    assert result.research_ready is True


def test_pipeline_preserves_identity():
    data = make_ohlcv(
        rows=700
    )

    result = run_research_data_pipeline(
        data,
        symbol="INFY.NS",
        timeframe="1D",
        horizons=[
            1,
            5,
        ],
    )

    assert (
        result.metadata["symbol"]
        == "INFY.NS"
    )

    assert (
        result.metadata["timeframe"]
        == "1D"
    )

    assert (
        result.metadata["horizons"]
        == [1, 5]
    )


def test_empty_horizons_fail():
    data = make_ohlcv()

    pipeline = ResearchDataPipeline()

    result = pipeline.run(
        data,
        symbol="TCS.NS",
        timeframe="1D",
        horizons=[],
    )

    assert result.success is False
    assert len(
        result.errors
    ) > 0


def test_invalid_market_data_fails_closed():
    data = make_ohlcv()

    data.loc[
        data.index[20],
        "Close",
    ] = -100

    pipeline = ResearchDataPipeline()

    with pytest.raises(ValueError):
        pipeline.run(
            data,
            symbol="TCS.NS",
            timeframe="1D",
            horizons=[
                1,
                5,
            ],
        )


def test_invalid_ohlc_relationship_fails_closed():
    data = make_ohlcv()

    data.loc[
        data.index[20],
        "High",
    ] = 10

    data.loc[
        data.index[20],
        "Low",
    ] = 20

    pipeline = ResearchDataPipeline()

    with pytest.raises(ValueError):
        pipeline.run(
            data,
            symbol="TCS.NS",
            timeframe="1D",
            horizons=[
                1,
                5,
            ],
        )


def test_dataset_build_failure_is_captured():
    data = make_ohlcv()

    pipeline = ResearchDataPipeline()

    result = pipeline.run(
        data,
        symbol="",
        timeframe="1D",
        horizons=[
            1,
        ],
    )

    assert result.success is False
    assert result.dataset is None
    assert len(
        result.errors
    ) > 0


def test_research_ready_requires_dataset():
    data = make_ohlcv(
        rows=700
    )

    pipeline = ResearchDataPipeline()

    result = pipeline.run(
        data,
        symbol="TCS.NS",
        timeframe="1D",
        horizons=[
            1,
            3,
        ],
    )

    if result.dataset is not None:
        assert result.research_ready is True


def test_production_ready_is_always_false():
    data = make_ohlcv(
        rows=700
    )

    result = run_research_data_pipeline(
        data,
        symbol="TCS.NS",
        timeframe="1D",
        horizons=[
            1,
            3,
        ],
    )

    assert result.production_ready is False


def test_production_approval_is_false():
    data = make_ohlcv(
        rows=700
    )

    result = run_research_data_pipeline(
        data,
        symbol="TCS.NS",
        timeframe="1D",
        horizons=[
            1,
            3,
        ],
    )

    assert (
        result.metadata[
            "production_approved"
        ]
        is False
    )

    assert (
        result.metadata[
            "approval_granted"
        ]
        is False
    )


def test_final_holdout_is_not_used():
    data = make_ohlcv(
        rows=700
    )

    result = run_research_data_pipeline(
        data,
        symbol="TCS.NS",
        timeframe="1D",
        horizons=[
            1,
            5,
        ],
    )

    assert (
        result.metadata[
            "final_holdout_used"
        ]
        is False
    )


def test_model_is_not_fitted():
    data = make_ohlcv(
        rows=700
    )

    result = run_research_data_pipeline(
        data,
        symbol="TCS.NS",
        timeframe="1D",
        horizons=[
            1,
            5,
        ],
    )

    assert (
        result.metadata[
            "model_fitted"
        ]
        is False
    )


def test_no_future_values_are_used():
    data = make_ohlcv(
        rows=700
    )

    result = run_research_data_pipeline(
        data,
        symbol="TCS.NS",
        timeframe="1D",
        horizons=[
            1,
            5,
        ],
    )

    assert (
        result.metadata[
            "future_values_used"
        ]
        is False
    )


def test_input_remains_unchanged_after_full_pipeline():
    data = make_ohlcv(
        rows=700
    )

    original = data.copy(
        deep=True
    )

    run_research_data_pipeline(
        data,
        symbol="TCS.NS",
        timeframe="1D",
        horizons=[
            1,
            5,
        ],
    )

    pd.testing.assert_frame_equal(
        data,
        original,
    )


def test_pipeline_is_deterministic():
    data = make_ohlcv(
        rows=700
    )

    first = run_research_data_pipeline(
        data,
        symbol="INFY.NS",
        timeframe="1D",
        horizons=[
            1,
            3,
            5,
        ],
    )

    second = run_research_data_pipeline(
        data,
        symbol="INFY.NS",
        timeframe="1D",
        horizons=[
            1,
            3,
            5,
        ],
    )

    assert (
        first.success
        == second.success
    )

    assert (
        first.warnings
        == second.warnings
    )

    assert (
        first.errors
        == second.errors
    )

    assert (
        first.metadata
        == second.metadata
    )

    if (
        first.dataset is not None
        and second.dataset is not None
    ):
        pd.testing.assert_frame_equal(
            first.dataset.data,
            second.dataset.data,
        )


def test_summary_contains_expected_fields():
    data = make_ohlcv(
        rows=700
    )

    result = run_research_data_pipeline(
        data,
        symbol="HDFCBANK.NS",
        timeframe="1D",
        horizons=[
            1,
            5,
        ],
    )

    summary = research_data_pipeline_summary(
        result
    )

    assert summary["success"] == (
        result.success
    )

    assert (
        summary["production_ready"]
        is False
    )

    assert (
        summary["prepared_rows"]
        == result.prepared_data.rows
    )

    assert (
        summary["research_only"]
        is True
    )


def test_summary_rejects_wrong_type():
    with pytest.raises(TypeError):
        research_data_pipeline_summary(
            None
        )


def test_dataset_is_based_on_prepared_data():
    data = make_ohlcv(
        rows=700
    )

    pipeline = ResearchDataPipeline()

    prepared = pipeline.prepare(
        data,
        symbol="TCS.NS",
        timeframe="1D",
    )

    dataset = pipeline.build_dataset(
        prepared,
        symbol="TCS.NS",
        timeframe="1D",
        horizons=[
            1,
            5,
        ],
    )

    assert (
        dataset.symbol
        == "TCS.NS"
    )

    assert (
        dataset.timeframe
        == "1D"
    )


def test_build_dataset_requires_prepared_result():
    pipeline = ResearchDataPipeline()

    with pytest.raises(TypeError):
        pipeline.build_dataset(
            "invalid",
            symbol="TCS.NS",
            timeframe="1D",
            horizons=[
                1,
            ],
        )


def test_pipeline_does_not_claim_approval_after_success():
    data = make_ohlcv(
        rows=700
    )

    result = run_research_data_pipeline(
        data,
        symbol="RELIANCE.NS",
        timeframe="1D",
        horizons=[
            1,
            3,
            5,
        ],
    )

    assert result.success is True

    assert (
        result.production_ready
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
            "approval_granted"
        ]
        is False
    )
