"""
Tests for the research orchestration layer.

These tests focus on:

    - configuration validation
    - stage ordering
    - feature/target separation
    - chronological integrity
    - leakage detection
    - deterministic dataset preparation
    - failure handling

No live market-data connection is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engine import FeatureSet
from src.models.targets import TargetSpec
from src.research.config import (
    ResearchApprovalConfig,
    ResearchBacktestConfig,
    ResearchCalibrationConfig,
    ResearchFeatureConfig,
    ResearchHyperparameterConfig,
    ResearchPipelineConfig,
    ResearchRangeConfig,
    ResearchRobustnessConfig,
    ResearchValidationConfig,
)
from src.research.pipeline import (
    ResearchPipeline,
    ResearchPipelineResult,
    ResearchStage,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """
    Deterministic OHLCV dataset.
    """

    index = pd.date_range(
        "2020-01-01",
        periods=300,
        freq="D",
    )

    rng = np.random.default_rng(
        42
    )

    close = (
        100
        + np.cumsum(
            rng.normal(
                0.15,
                1.0,
                len(index),
            )
        )
    )

    close = np.maximum(
        close,
        10,
    )

    open_price = (
        close
        + rng.normal(
            0,
            0.5,
            len(index),
        )
    )

    high = (
        np.maximum(
            open_price,
            close,
        )
        + rng.uniform(
            0.1,
            1.5,
            len(index),
        )
    )

    low = (
        np.minimum(
            open_price,
            close,
        )
        - rng.uniform(
            0.1,
            1.5,
            len(index),
        )
    )

    volume = rng.integers(
        100_000,
        1_000_000,
        len(index),
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


@pytest.fixture
def config() -> ResearchPipelineConfig:
    return ResearchPipelineConfig(
        symbol="RELIANCE",
        timeframe="1D",
        horizons=(
            1,
            3,
            5,
        ),
    )


# ----------------------------------------------------------------------
# Configuration tests
# ----------------------------------------------------------------------


def test_default_validation_config_is_strict() -> None:
    config = ResearchValidationConfig()

    assert config.train_fraction == 0.60
    assert config.validation_fraction == 0.20
    assert config.test_fraction == 0.20

    assert config.minimum_accuracy == 0.95
    assert config.require_walk_forward is True
    assert config.require_final_holdout is True
    assert config.require_no_leakage is True


def test_validation_fractions_must_sum_to_one() -> None:
    with pytest.raises(
        ValueError,
        match="sum to 1.0",
    ):
        ResearchValidationConfig(
            train_fraction=0.50,
            validation_fraction=0.20,
            test_fraction=0.10,
        )


def test_invalid_timeframe_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported timeframe",
    ):
        ResearchPipelineConfig(
            symbol="RELIANCE",
            timeframe="2D",
        )


def test_empty_symbol_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="symbol cannot be empty",
    ):
        ResearchPipelineConfig(
            symbol="   ",
        )


def test_horizons_are_normalized() -> None:
    config = ResearchPipelineConfig(
        symbol="RELIANCE",
        horizons=(
            20,
            5,
            5,
            1,
            10,
        ),
    )

    assert config.horizons == (
        1,
        5,
        10,
        20,
    )


def test_negative_horizon_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="horizons must be positive",
    ):
        ResearchPipelineConfig(
            symbol="RELIANCE",
            horizons=(
                1,
                -5,
            ),
        )


def test_approval_defaults_use_strict_gate() -> None:
    config = ResearchApprovalConfig()

    assert (
        config.minimum_validation_accuracy
        == 0.95
    )

    assert (
        config.minimum_final_holdout_accuracy
        == 0.95
    )

    assert (
        config.maximum_generalization_gap
        == 0.10
    )


def test_range_quantiles_must_be_ordered() -> None:
    with pytest.raises(
        ValueError,
        match="Range quantiles must be ordered",
    ):
        ResearchRangeConfig(
            lower_quantile=0.50,
            median_quantile=0.10,
            upper_quantile=0.90,
        )


def test_calibration_method_is_validated() -> None:
    with pytest.raises(
        ValueError,
        match="Calibration method",
    ):
        ResearchCalibrationConfig(
            method="random_forest",
        )


def test_backtest_probability_threshold_is_validated() -> None:
    with pytest.raises(
        ValueError,
        match="probability_threshold",
    ):
        ResearchBacktestConfig(
            probability_threshold=0.20,
        )


def test_robustness_simulations_are_validated() -> None:
    with pytest.raises(
        ValueError,
        match="robustness simulations",
    ):
        ResearchRobustnessConfig(
            simulations=10,
        )


def test_feature_configuration_is_validated() -> None:
    with pytest.raises(
        ValueError,
        match="minimum_features",
    ):
        ResearchFeatureConfig(
            minimum_features=0,
        )


def test_hyperparameter_trials_are_validated() -> None:
    with pytest.raises(
        ValueError,
        match="max_trials",
    ):
        ResearchHyperparameterConfig(
            max_trials=0,
        )


# ----------------------------------------------------------------------
# Configuration serialization
# ----------------------------------------------------------------------


def test_configuration_summary(config) -> None:
    summary = config.summary()

    assert summary[
        "symbol"
    ] == "RELIANCE"

    assert summary[
        "timeframe"
    ] == "1D"

    assert (
        summary[
            "minimum_accuracy"
        ]
        == 0.95
    )


def test_configuration_to_dict(config) -> None:
    data = config.to_dict()

    assert data[
        "symbol"
    ] == "RELIANCE"

    assert data[
        "timeframe"
    ] == "1D"

    assert data[
        "horizons"
    ] == [
        1,
        3,
        5,
    ]

    assert (
        "validation"
        in data
    )

    assert (
        "approval"
        in data
    )


def test_configuration_from_mapping() -> None:
    config = (
        ResearchPipelineConfig.from_mapping(
            "TCS",
            {
                "prediction": {
                    "horizons": [
                        1,
                        5,
                        10,
                    ]
                },
                "validation": {
                    "minimum_accuracy": 0.95,
                    "test_fraction": 0.20,
                    "validation_fraction": 0.20,
                    "train_fraction": 0.60,
                },
            },
        )
    )

    assert config.symbol == "TCS"

    assert config.horizons == (
        1,
        5,
        10,
    )

    assert (
        config.validation.minimum_accuracy
        == 0.95
    )


# ----------------------------------------------------------------------
# Pipeline result tests
# ----------------------------------------------------------------------


def test_initial_result_is_not_successful(
    config,
) -> None:
    result = ResearchPipelineResult(
        config=config,
        symbol=config.symbol,
        timeframe=config.timeframe,
    )

    assert result.stage == ResearchStage.CREATED
    assert result.successful is False
    assert result.production_ready is False


def test_failed_result_is_not_successful(
    config,
) -> None:
    result = ResearchPipelineResult(
        config=config,
        symbol=config.symbol,
        timeframe=config.timeframe,
        stage=ResearchStage.FAILED,
        errors=[
            "test error"
        ],
    )

    assert result.successful is False
    assert result.production_ready is False


def test_result_summary(config) -> None:
    result = ResearchPipelineResult(
        config=config,
        symbol="RELIANCE",
        timeframe="1D",
    )

    summary = result.summary()

    assert summary[
        "symbol"
    ] == "RELIANCE"

    assert summary[
        "stage"
    ] == "CREATED"

    assert summary[
        "production_ready"
    ] is False


# ----------------------------------------------------------------------
# Fake builders
# ----------------------------------------------------------------------


def fake_data_builder(
    symbol: str,
    timeframe: str = "1D",
) -> pd.DataFrame:
    index = pd.date_range(
        "2020-01-01",
        periods=200,
        freq="D",
    )

    close = np.linspace(
        100,
        200,
        len(index),
    )

    return pd.DataFrame(
        {
            "Open": close - 1,
            "High": close + 2,
            "Low": close - 2,
            "Close": close,
            "Volume": np.full(
                len(index),
                100_000,
            ),
        },
        index=index,
    )


def fake_feature_builder(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe.copy()

    result[
        "RSI_14"
    ] = 50.0

    result[
        "Momentum_5"
    ] = result[
        "Close"
    ].pct_change(
        5
    )

    return result


def fake_target_builder(
    dataframe: pd.DataFrame,
    target_spec: TargetSpec,
) -> pd.DataFrame:
    horizon = target_spec.horizon

    result = pd.DataFrame(
        index=dataframe.index
    )

    result[
        f"Future_Return_{horizon}"
    ] = (
        dataframe[
            "Close"
        ].shift(
            -horizon
        )
        / dataframe[
            "Close"
        ]
        - 1.0
    )

    result[
        f"Direction_{horizon}"
    ] = (
        result[
            f"Future_Return_{horizon}"
        ]
        > 0
    ).astype(
        float
    )

    return result


# ----------------------------------------------------------------------
# Pipeline stage tests
# ----------------------------------------------------------------------


def build_test_pipeline(
    config,
) -> ResearchPipeline:
    return ResearchPipeline(
        config,
        data_builder=fake_data_builder,
        feature_builder=fake_feature_builder,
        target_builder=fake_target_builder,
    )


def test_data_stage(
    config,
) -> None:
    pipeline = build_test_pipeline(
        config
    )

    result = ResearchPipelineResult(
        config=config,
        symbol=config.symbol,
        timeframe=config.timeframe,
    )

    result = pipeline.run_data_stage(
        result
    )

    assert result.stage == ResearchStage.DATA
    assert result.raw_data is not None
    assert not result.raw_data.empty


def test_feature_stage(
    config,
) -> None:
    pipeline = build_test_pipeline(
        config
    )

    result = ResearchPipelineResult(
        config=config,
        symbol=config.symbol,
        timeframe=config.timeframe,
    )

    result = pipeline.run_data_stage(
        result
    )

    result = pipeline.run_feature_stage(
        result
    )

    assert (
        result.stage
        == ResearchStage.FEATURES
    )

    assert result.dataset is not None

    assert (
        "RSI_14"
        in result.dataset.columns
    )


def test_target_stage(
    config,
) -> None:
    pipeline = build_test_pipeline(
        config
    )

    result = ResearchPipelineResult(
        config=config,
        symbol=config.symbol,
        timeframe=config.timeframe,
    )

    result = pipeline.run_data_stage(
        result
    )

    result = pipeline.run_feature_stage(
        result
    )

    result = pipeline.run_target_stage(
        result
    )

    assert (
        result.stage
        == ResearchStage.TARGETS
    )

    assert result.target_columns

    assert any(
        column.startswith(
            "Future_"
        )
        for column in result.target_columns
    )

    assert any(
        column.startswith(
            "Direction_"
        )
        for column in result.target_columns
    )


def test_dataset_preparation(
    config,
) -> None:
    pipeline = build_test_pipeline(
        config
    )

    result = pipeline.run()

    assert (
        result.stage
        == ResearchStage.DATASET_READY
    )

    assert result.dataset is not None

    assert len(
        result.feature_columns
    ) > 0

    assert len(
        result.target_columns
    ) > 0


def test_full_dataset_stage_sequence(
    config,
) -> None:
    pipeline = build_test_pipeline(
        config
    )

    result = pipeline.run()

    assert result.stage == (
        ResearchStage.DATASET_READY
    )

    assert result.errors == []

    assert result.successful is False

    assert result.production_ready is False


# ----------------------------------------------------------------------
# Leakage tests
# ----------------------------------------------------------------------


def test_target_columns_are_not_features(
    config,
) -> None:
    pipeline = build_test_pipeline(
        config
    )

    result = pipeline.run()

    overlap = (
        set(
            result.feature_columns
        )
        & set(
            result.target_columns
        )
    )

    assert overlap == set()


def test_future_named_feature_is_rejected(
    config,
) -> None:
    pipeline = build_test_pipeline(
        config
    )

    result = ResearchPipelineResult(
        config=config,
        symbol=config.symbol,
        timeframe=config.timeframe,
    )

    bad_dataframe = fake_data_builder(
        "RELIANCE"
    )

    bad_dataframe[
        "Future_Return"
    ] = 0.0

    result.dataset = bad_dataframe

    with pytest.raises(
        ValueError,
        match="future/target leakage",
    ):
        pipeline._assert_no_future_feature_names(
            [
                "Close",
                "Future_Return",
            ]
        )


def test_duplicate_timestamps_are_rejected(
    config,
) -> None:
    pipeline = build_test_pipeline(
        config
    )

    result = ResearchPipelineResult(
        config=config,
        symbol=config.symbol,
        timeframe=config.timeframe,
    )

    result.dataset = pd.concat(
        [
            fake_data_builder(
                "RELIANCE"
            ),
            fake_data_builder(
                "RELIANCE"
            ).iloc[:1],
        ]
    )

    with pytest.raises(
        ValueError,
        match="duplicate timestamps",
    ):
        pipeline._assert_no_duplicate_index(
            result.dataset
        )


def test_unsorted_timestamps_are_rejected(
    config,
) -> None:
    pipeline = build_test_pipeline(
        config
    )

    result = ResearchPipelineResult(
        config=config,
        symbol=config.symbol,
        timeframe=config.timeframe,
    )

    dataframe = fake_data_builder(
        "RELIANCE"
    ).iloc[
        ::-1
    ]

    with pytest.raises(
        ValueError,
        match="not chronological",
    ):
        pipeline._assert_sorted_index(
            dataframe
        )


def test_non_numeric_features_are_rejected(
    config,
) -> None:
    pipeline = build_test_pipeline(
        config
    )

    dataframe = fake_data_builder(
        "RELIANCE"
    )

    dataframe[
        "BadFeature"
    ] = "not numeric"

    with pytest.raises(
        TypeError,
        match="Non-numeric feature",
    ):
        pipeline._assert_numeric_features(
            dataframe,
            [
                "BadFeature"
            ],
        )


# ----------------------------------------------------------------------
# Stage-order tests
# ----------------------------------------------------------------------


def test_feature_stage_cannot_run_before_data(
    config,
) -> None:
    pipeline = build_test_pipeline(
        config
    )

    result = ResearchPipelineResult(
        config=config,
        symbol=config.symbol,
        timeframe=config.timeframe,
    )

    with pytest.raises(
        RuntimeError,
        match="Invalid pipeline stage",
    ):
        pipeline.run_feature_stage(
            result
        )


def test_target_stage_cannot_run_before_features(
    config,
) -> None:
    pipeline = build_test_pipeline(
        config
    )

    result = ResearchPipelineResult(
        config=config,
        symbol=config.symbol,
        timeframe=config.timeframe,
        stage=ResearchStage.DATA,
        dataset=fake_data_builder(
            "RELIANCE"
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Invalid pipeline stage",
    ):
        pipeline.run_target_stage(
            result
        )


def test_dataset_preparation_cannot_run_before_targets(
    config,
) -> None:
    pipeline = build_test_pipeline(
        config
    )

    result = ResearchPipelineResult(
        config=config,
        symbol=config.symbol,
        timeframe=config.timeframe,
        stage=ResearchStage.FEATURES,
        dataset=fake_data_builder(
            "RELIANCE"
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Invalid pipeline stage",
    ):
        pipeline.prepare_dataset(
            result
        )


# ----------------------------------------------------------------------
# Failure handling
# ----------------------------------------------------------------------


def test_data_failure_is_captured(
    config,
) -> None:
    def failing_data_builder(
        *args,
        **kwargs,
    ):
        raise RuntimeError(
            "synthetic data failure"
        )

    pipeline = ResearchPipeline(
        config,
        data_builder=failing_data_builder,
        feature_builder=fake_feature_builder,
        target_builder=fake_target_builder,
    )

    result = pipeline.run()

    assert (
        result.stage
        == ResearchStage.FAILED
    )

    assert len(
        result.errors
    ) == 1

    assert (
        "synthetic data failure"
        in result.errors[0]
    )


def test_feature_failure_is_captured(
    config,
) -> None:
    def failing_feature_builder(
        dataframe,
    ):
        raise RuntimeError(
            "synthetic feature failure"
        )

    pipeline = ResearchPipeline(
        config,
        data_builder=fake_data_builder,
        feature_builder=failing_feature_builder,
        target_builder=fake_target_builder,
    )

    result = pipeline.run()

    assert (
        result.stage
        == ResearchStage.FAILED
    )

    assert (
        "synthetic feature failure"
        in result.errors[0]
    )


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------


def test_research_dataset_is_deterministic(
    config,
) -> None:
    pipeline_1 = build_test_pipeline(
        config
    )

    pipeline_2 = build_test_pipeline(
        config
    )

    result_1 = pipeline_1.run()
    result_2 = pipeline_2.run()

    assert result_1.stage == result_2.stage

    pd.testing.assert_frame_equal(
        result_1.dataset,
        result_2.dataset,
    )

    assert (
        result_1.feature_columns
        == result_2.feature_columns
    )

    assert (
        result_1.target_columns
        == result_2.target_columns
    )


# ----------------------------------------------------------------------
# Production safety
# ----------------------------------------------------------------------


def test_dataset_ready_does_not_mean_production_ready(
    config,
) -> None:
    pipeline = build_test_pipeline(
        config
    )

    result = pipeline.run()

    assert (
        result.stage
        == ResearchStage.DATASET_READY
    )

    assert result.production_ready is False


def test_approval_status_controls_production_ready(
    config,
) -> None:
    class ApprovedReport:
        status = "APPROVED"

    result = ResearchPipelineResult(
        config=config,
        symbol=config.symbol,
        timeframe=config.timeframe,
        stage=ResearchStage.COMPLETE,
        approval_report=ApprovedReport(),
    )

    assert result.production_ready is True
