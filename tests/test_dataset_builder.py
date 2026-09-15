"""
Tests for the research dataset builder.

The tests are intentionally synthetic and deterministic.

They verify:

    - OHLCV validation
    - feature generation
    - target generation
    - feature/target separation
    - future-column detection
    - chronological ordering
    - duplicate handling
    - numeric feature enforcement
    - deterministic output
    - warning generation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.targets import TargetSpec
from src.research.config import ResearchPipelineConfig
from src.research.dataset_builder import (
    ResearchDatasetBuilder,
    ResearchDatasetResult,
    build_research_dataset,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """
    Create deterministic OHLCV data.
    """

    index = pd.date_range(
        "2020-01-01",
        periods=600,
        freq="D",
    )

    rng = np.random.default_rng(
        123
    )

    close = (
        100
        + np.cumsum(
            rng.normal(
                0.10,
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
            0.4,
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
            1.2,
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
            1.2,
            len(index),
        )
    )

    volume = rng.integers(
        100_000,
        2_000_000,
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
# Fake builders
# ----------------------------------------------------------------------


def fake_data_builder(
    symbol: str,
    timeframe: str = "1D",
) -> pd.DataFrame:

    index = pd.date_range(
        "2020-01-01",
        periods=600,
        freq="D",
    )

    close = np.linspace(
        100,
        250,
        len(index),
    )

    return pd.DataFrame(
        {
            "Open": close - 1.0,
            "High": close + 2.0,
            "Low": close - 2.0,
            "Close": close,
            "Volume": np.full(
                len(index),
                500_000,
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
    ] = (
        result[
            "Close"
        ].pct_change(
            5
        )
    )

    result[
        "Volume_Ratio"
    ] = (
        result[
            "Volume"
        ]
        / result[
            "Volume"
        ].rolling(
            20
        ).mean()
    )

    return result


def fake_target_builder(
    dataframe: pd.DataFrame,
    target_spec: TargetSpec,
) -> pd.DataFrame:

    horizon = int(
        target_spec.horizon
    )

    future_return = (
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

    result = pd.DataFrame(
        index=dataframe.index
    )

    result[
        f"Future_Return_{horizon}"
    ] = future_return

    result[
        f"Direction_{horizon}"
    ] = (
        future_return
        > 0
    ).astype(
        float
    )

    return result


def build_builder(
    config: ResearchPipelineConfig,
) -> ResearchDatasetBuilder:

    return ResearchDatasetBuilder(
        config,
        data_builder=fake_data_builder,
        feature_builder=fake_feature_builder,
        target_builder=fake_target_builder,
    )


# ----------------------------------------------------------------------
# Basic builder tests
# ----------------------------------------------------------------------


def test_builder_returns_result(
    config,
) -> None:

    builder = build_builder(
        config
    )

    result = builder.build()

    assert isinstance(
        result,
        ResearchDatasetResult,
    )


def test_builder_creates_non_empty_dataset(
    config,
) -> None:

    result = build_builder(
        config
    ).build()

    assert not result.dataframe.empty
    assert result.rows > 0


def test_symbol_and_timeframe_are_preserved(
    config,
) -> None:

    result = build_builder(
        config
    ).build()

    assert result.symbol == "RELIANCE"
    assert result.timeframe == "1D"


def test_horizons_are_preserved(
    config,
) -> None:

    result = build_builder(
        config
    ).build()

    assert result.horizons == (
        1,
        3,
        5,
    )


# ----------------------------------------------------------------------
# Feature tests
# ----------------------------------------------------------------------


def test_feature_columns_are_detected(
    config,
) -> None:

    result = build_builder(
        config
    ).build()

    assert (
        "Open"
        in result.feature_columns
    )

    assert (
        "Close"
        in result.feature_columns
    )

    assert (
        "RSI_14"
        in result.feature_columns
    )

    assert (
        "Momentum_5"
        in result.feature_columns
    )


def test_target_columns_are_not_features(
    config,
) -> None:

    result = build_builder(
        config
    ).build()

    overlap = (
        set(
            result.feature_columns
        )
        & set(
            result.target_columns
        )
    )

    assert overlap == set()


def test_future_columns_are_targets(
    config,
) -> None:

    result = build_builder(
        config
    ).build()

    future_targets = [
        column
        for column in result.target_columns
        if column.startswith(
            "Future_"
        )
    ]

    assert future_targets

    for column in future_targets:
        assert (
            column
            not in result.feature_columns
        )


def test_direction_columns_are_targets(
    config,
) -> None:

    result = build_builder(
        config
    ).build()

    direction_targets = [
        column
        for column in result.target_columns
        if column.startswith(
            "Direction_"
        )
    ]

    assert direction_targets

    for column in direction_targets:
        assert (
            column
            not in result.feature_columns
        )


# ----------------------------------------------------------------------
# Target horizon tests
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "horizon",
    [
        1,
        3,
        5,
    ],
)
def test_each_horizon_creates_targets(
    config,
    horizon,
) -> None:

    result = build_builder(
        config
    ).build()

    assert (
        f"Future_Return_{horizon}"
        in result.target_columns
    )

    assert (
        f"Direction_{horizon}"
        in result.target_columns
    )


def test_target_count_matches_horizons(
    config,
) -> None:

    result = build_builder(
        config
    ).build()

    expected_minimum = (
        len(
            config.horizons
        )
        * 2
    )

    assert (
        result.target_count
        >= expected_minimum
    )


# ----------------------------------------------------------------------
# Chronological integrity
# ----------------------------------------------------------------------


def test_dataset_is_chronological(
    config,
) -> None:

    result = build_builder(
        config
    ).build()

    assert result.dataframe.index.is_monotonic_increasing


def test_dataset_has_no_duplicate_timestamps(
    config,
) -> None:

    result = build_builder(
        config
    ).build()

    assert not result.dataframe.index.has_duplicates


def test_dataset_uses_datetime_index(
    config,
) -> None:

    result = build_builder(
        config
    ).build()

    assert isinstance(
        result.dataframe.index,
        pd.DatetimeIndex,
    )


# ----------------------------------------------------------------------
# Invalid data tests
# ----------------------------------------------------------------------


def test_empty_data_is_rejected(
    config,
) -> None:

    def empty_builder(
        *args,
        **kwargs,
    ):
        return pd.DataFrame()

    builder = ResearchDatasetBuilder(
        config,
        data_builder=empty_builder,
        feature_builder=fake_feature_builder,
        target_builder=fake_target_builder,
    )

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        builder.build()


def test_missing_ohlcv_column_is_rejected(
    config,
) -> None:

    def invalid_builder(
        *args,
        **kwargs,
    ):

        dataframe = fake_data_builder(
            "RELIANCE"
        )

        return dataframe.drop(
            columns=[
                "Volume"
            ]
        )

    builder = ResearchDatasetBuilder(
        config,
        data_builder=invalid_builder,
        feature_builder=fake_feature_builder,
        target_builder=fake_target_builder,
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        builder.build()


def test_duplicate_raw_timestamps_are_rejected(
    config,
) -> None:

    def duplicate_builder(
        *args,
        **kwargs,
    ):

        dataframe = fake_data_builder(
            "RELIANCE"
        )

        duplicate = pd.concat(
            [
                dataframe,
                dataframe.iloc[
                    :1
                ],
            ]
        )

        return duplicate

    builder = ResearchDatasetBuilder(
        config,
        data_builder=duplicate_builder,
        feature_builder=fake_feature_builder,
        target_builder=fake_target_builder,
    )

    with pytest.raises(
        ValueError,
        match="duplicate timestamps",
    ):
        builder.build()


def test_unsorted_raw_data_is_rejected(
    config,
) -> None:

    def unsorted_builder(
        *args,
        **kwargs,
    ):

        dataframe = fake_data_builder(
            "RELIANCE"
        )

        return dataframe.iloc[
            ::-1
        ]

    builder = ResearchDatasetBuilder(
        config,
        data_builder=unsorted_builder,
        feature_builder=fake_feature_builder,
        target_builder=fake_target_builder,
    )

    with pytest.raises(
        ValueError,
        match="not chronological",
    ):
        builder.build()


# ----------------------------------------------------------------------
# Feature validation tests
# ----------------------------------------------------------------------


def test_non_numeric_feature_is_rejected(
    config,
) -> None:

    def invalid_feature_builder(
        dataframe,
    ):

        result = dataframe.copy()

        result[
            "InvalidFeature"
        ] = "text"

        return result

    builder = ResearchDatasetBuilder(
        config,
        data_builder=fake_data_builder,
        feature_builder=invalid_feature_builder,
        target_builder=fake_target_builder,
    )

    with pytest.raises(
        TypeError,
        match="is not numeric",
    ):
        builder.build()


def test_infinite_feature_is_rejected(
    config,
) -> None:

    def invalid_feature_builder(
        dataframe,
    ):

        result = dataframe.copy()

        result[
            "InfiniteFeature"
        ] = np.inf

        return result

    builder = ResearchDatasetBuilder(
        config,
        data_builder=fake_data_builder,
        feature_builder=invalid_feature_builder,
        target_builder=fake_target_builder,
    )

    with pytest.raises(
        ValueError,
        match="infinite",
    ):
        builder.build()


def test_future_named_feature_is_rejected(
    config,
) -> None:

    def leaking_feature_builder(
        dataframe,
    ):

        result = dataframe.copy()

        result[
            "Future_Close_Return"
        ] = (
            result[
                "Close"
            ].shift(
                -1
            )
            / result[
                "Close"
            ]
            - 1
        )

        return result

    builder = ResearchDatasetBuilder(
        config,
        data_builder=fake_data_builder,
        feature_builder=leaking_feature_builder,
        target_builder=fake_target_builder,
    )

    with pytest.raises(
        ValueError,
        match="forward-looking feature",
    ):
        builder.build()


# ----------------------------------------------------------------------
# Target builder tests
# ----------------------------------------------------------------------


def test_target_builder_must_return_dataframe(
    config,
) -> None:

    def invalid_target_builder(
        dataframe,
        target_spec,
    ):
        return None

    builder = ResearchDatasetBuilder(
        config,
        data_builder=fake_data_builder,
        feature_builder=fake_feature_builder,
        target_builder=invalid_target_builder,
    )

    with pytest.raises(
        TypeError,
        match="Target builder must return a DataFrame",
    ):
        builder.build()


def test_empty_target_frame_is_rejected(
    config,
) -> None:

    def empty_target_builder(
        dataframe,
        target_spec,
    ):
        return pd.DataFrame(
            index=dataframe.index
        )

    builder = ResearchDatasetBuilder(
        config,
        data_builder=fake_data_builder,
        feature_builder=fake_feature_builder,
        target_builder=empty_target_builder,
    )

    with pytest.raises(
        ValueError,
        match="Target builder returned an empty",
    ):
        builder.build()


# ----------------------------------------------------------------------
# Metadata tests
# ----------------------------------------------------------------------


def test_result_contains_start_and_end(
    config,
) -> None:

    result = build_builder(
        config
    ).build()

    assert result.start is not None
    assert result.end is not None

    assert (
        result.start
        <= result.end
    )


def test_result_metadata_is_populated(
    config,
) -> None:

    result = build_builder(
        config
    ).build()

    assert (
        result.metadata[
            "symbol"
        ]
        == "RELIANCE"
    )

    assert (
        result.metadata[
            "timeframe"
        ]
        == "1D"
    )

    assert (
        result.metadata[
            "feature_count"
        ]
        == result.feature_count
    )


def test_summary_is_consistent(
    config,
) -> None:

    result = build_builder(
        config
    ).build()

    summary = result.summary()

    assert (
        summary[
            "rows"
        ]
        == result.rows
    )

    assert (
        summary[
            "features"
        ]
        == result.feature_count
    )

    assert (
        summary[
            "targets"
        ]
        == result.target_count
    )


# ----------------------------------------------------------------------
# Warning tests
# ----------------------------------------------------------------------


def test_small_dataset_generates_warning(
    config,
) -> None:

    def small_data_builder(
        *args,
        **kwargs,
    ):

        index = pd.date_range(
            "2020-01-01",
            periods=100,
            freq="D",
        )

        close = np.linspace(
            100,
            120,
            len(index),
        )

        return pd.DataFrame(
            {
                "Open": close,
                "High": close + 1,
                "Low": close - 1,
                "Close": close,
                "Volume": 100_000,
            },
            index=index,
        )

    builder = ResearchDatasetBuilder(
        config,
        data_builder=small_data_builder,
        feature_builder=fake_feature_builder,
        target_builder=fake_target_builder,
    )

    result = builder.build()

    assert any(
        "fewer than 500"
        in warning
        for warning in result.warnings
    )


# ----------------------------------------------------------------------
# Determinism tests
# ----------------------------------------------------------------------


def test_builder_is_deterministic(
    config,
) -> None:

    result_1 = build_builder(
        config
    ).build()

    result_2 = build_builder(
        config
    ).build()

    pd.testing.assert_frame_equal(
        result_1.dataframe,
        result_2.dataframe,
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
# Convenience API
# ----------------------------------------------------------------------


def test_convenience_function(
    config,
) -> None:

    result = build_research_dataset(
        config,
        data_builder=fake_data_builder,
        feature_builder=fake_feature_builder,
        target_builder=fake_target_builder,
    )

    assert isinstance(
        result,
        ResearchDatasetResult,
    )

    assert result.symbol == "RELIANCE"


# ----------------------------------------------------------------------
# Target causality test
# ----------------------------------------------------------------------


def test_future_target_does_not_become_feature(
    config,
) -> None:

    result = build_builder(
        config
    ).build()

    future_columns = [
        column
        for column in result.dataframe.columns
        if column.startswith(
            "Future_"
        )
    ]

    assert future_columns

    for column in future_columns:
        assert (
            column
            not in result.feature_columns
        )


# ----------------------------------------------------------------------
# Feature count sanity
# ----------------------------------------------------------------------


def test_feature_count_is_positive(
    config,
) -> None:

    result = build_builder(
        config
    ).build()

    assert result.feature_count > 0


def test_target_count_is_positive(
    config,
) -> None:

    result = build_builder(
        config
    ).build()

    assert result.target_count > 0
