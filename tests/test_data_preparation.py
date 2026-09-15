"""
Tests for the research market-data preparation layer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.data_preparation import (
    DataPreparationConfig,
    PreparedMarketData,
    prepare_market_data,
    preparation_summary,
)


def make_ohlcv(
    rows: int = 300,
) -> pd.DataFrame:
    index = pd.date_range(
        "2022-01-03",
        periods=rows,
        freq="D",
    )

    close = (
        100.0
        + np.linspace(0, 20, rows)
        + np.sin(
            np.arange(rows) / 10.0
        )
    )

    open_price = close + 0.2
    high = np.maximum(
        open_price,
        close,
    ) + 1.0
    low = np.minimum(
        open_price,
        close,
    ) - 1.0

    volume = (
        100_000
        + np.arange(rows) * 100
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


def test_default_config():
    config = DataPreparationConfig()

    assert config.minimum_rows > 0
    assert config.require_volume is True
    assert config.reject_negative_prices is True
    assert 0 <= config.max_missing_fraction <= 1


def test_invalid_minimum_rows():
    with pytest.raises(ValueError):
        DataPreparationConfig(
            minimum_rows=0
        )

    with pytest.raises(ValueError):
        DataPreparationConfig(
            minimum_rows=-10
        )


def test_invalid_missing_fraction():
    with pytest.raises(ValueError):
        DataPreparationConfig(
            max_missing_fraction=-0.1
        )

    with pytest.raises(ValueError):
        DataPreparationConfig(
            max_missing_fraction=1.1
        )


def test_basic_preparation():
    data = make_ohlcv()

    result = prepare_market_data(
        data,
        symbol="RELIANCE.NS",
        timeframe="1D",
    )

    assert isinstance(
        result,
        PreparedMarketData,
    )

    assert result.rows == len(data)
    assert result.symbol == "RELIANCE.NS"
    assert result.timeframe == "1D"
    assert result.start == data.index[0]
    assert result.end == data.index[-1]
    assert result.passed is True


def test_output_is_chronologically_sorted():
    data = make_ohlcv()

    result = prepare_market_data(
        data
    )

    assert (
        result.data.index.is_monotonic_increasing
    )

    assert not (
        result.data.index.has_duplicates
    )


def test_duplicate_timestamps_are_cleaned():
    data = make_ohlcv()

    duplicate = data.iloc[
        [10]
    ].copy()

    duplicate.index = [
        data.index[20]
    ]

    combined = pd.concat(
        [
            data,
            duplicate,
        ]
    )

    result = prepare_market_data(
        combined
    )

    assert not (
        result.data.index.has_duplicates
    )


def test_unsorted_data_is_sorted():
    data = make_ohlcv()

    shuffled = data.iloc[
        ::-1
    ].copy()

    result = prepare_market_data(
        shuffled
    )

    assert (
        result.data.index.is_monotonic_increasing
    )


def test_missing_ohlc_column_is_rejected():
    data = make_ohlcv()

    data = data.drop(
        columns=["Close"]
    )

    with pytest.raises(ValueError):
        prepare_market_data(
            data
        )


def test_missing_volume_is_rejected_by_default():
    data = make_ohlcv()

    data = data.drop(
        columns=["Volume"]
    )

    with pytest.raises(ValueError):
        prepare_market_data(
            data
        )


def test_missing_volume_can_be_allowed():
    data = make_ohlcv()

    data = data.drop(
        columns=["Volume"]
    )

    result = prepare_market_data(
        data,
        config=DataPreparationConfig(
            require_volume=False
        ),
    )

    assert result.rows > 0
    assert result.passed is True


def test_negative_price_is_rejected():
    data = make_ohlcv()

    data.loc[
        data.index[50],
        "Close",
    ] = -10

    with pytest.raises(ValueError):
        prepare_market_data(
            data
        )


def test_zero_price_is_rejected():
    data = make_ohlcv()

    data.loc[
        data.index[50],
        "Close",
    ] = 0

    with pytest.raises(ValueError):
        prepare_market_data(
            data
        )


def test_high_less_than_low_is_rejected():
    data = make_ohlcv()

    data.loc[
        data.index[50],
        "High",
    ] = 90

    data.loc[
        data.index[50],
        "Low",
    ] = 100

    with pytest.raises(ValueError):
        prepare_market_data(
            data
        )


def test_high_below_open_is_rejected():
    data = make_ohlcv()

    data.loc[
        data.index[50],
        "High",
    ] = (
        data.loc[
            data.index[50],
            "Open",
        ]
        - 1
    )

    with pytest.raises(ValueError):
        prepare_market_data(
            data
        )


def test_low_above_close_is_rejected():
    data = make_ohlcv()

    data.loc[
        data.index[50],
        "Low",
    ] = (
        data.loc[
            data.index[50],
            "Close",
        ]
        + 1
    )

    with pytest.raises(ValueError):
        prepare_market_data(
            data
        )


def test_negative_volume_is_rejected():
    data = make_ohlcv()

    data.loc[
        data.index[50],
        "Volume",
    ] = -100

    with pytest.raises(ValueError):
        prepare_market_data(
            data
        )


def test_zero_volume_can_be_removed():
    data = make_ohlcv()

    data.loc[
        data.index[50],
        "Volume",
    ] = 0

    result = prepare_market_data(
        data,
        config=DataPreparationConfig(
            remove_zero_volume=True
        ),
    )

    assert (
        data.index[50]
        not in result.data.index
    )


def test_minimum_rows_generates_warning():
    data = make_ohlcv(
        rows=50
    )

    result = prepare_market_data(
        data,
        config=DataPreparationConfig(
            minimum_rows=250
        ),
    )

    assert result.rows == 50
    assert any(
        "minimum recommended"
        in warning.lower()
        for warning in result.warnings
    )


def test_high_missing_column_is_rejected():
    data = make_ohlcv()

    data["Extra"] = np.nan

    with pytest.raises(ValueError):
        prepare_market_data(
            data,
            config=DataPreparationConfig(
                max_missing_fraction=0.40,
                reject_high_missing_columns=True,
            ),
        )


def test_high_missing_column_can_generate_warning():
    data = make_ohlcv()

    data["Extra"] = np.nan

    result = prepare_market_data(
        data,
        config=DataPreparationConfig(
            max_missing_fraction=0.40,
            reject_high_missing_columns=False,
        ),
    )

    assert any(
        "missing-value"
        in warning.lower()
        for warning in result.warnings
    )


def test_input_is_not_mutated():
    data = make_ohlcv()

    original = data.copy(
        deep=True
    )

    prepare_market_data(
        data
    )

    pd.testing.assert_frame_equal(
        data,
        original,
    )


def test_output_is_independent_copy():
    data = make_ohlcv()

    result = prepare_market_data(
        data
    )

    original_value = result.data.loc[
        result.data.index[0],
        "Close",
    ]

    result.data.loc[
        result.data.index[0],
        "Close",
    ] = 999999

    assert (
        data.loc[
            data.index[0],
            "Close",
        ]
        != 999999
    )

    assert (
        result.data.loc[
            result.data.index[0],
            "Close",
        ]
        != original_value
    )


def test_non_dataframe_input_is_rejected():
    with pytest.raises(TypeError):
        prepare_market_data(
            [1, 2, 3]
        )


def test_non_datetime_index_is_rejected():
    data = make_ohlcv()

    data.index = range(
        len(data)
    )

    with pytest.raises(TypeError):
        prepare_market_data(
            data
        )


def test_empty_dataframe_is_rejected():
    data = make_ohlcv(
        rows=2
    ).iloc[0:0]

    with pytest.raises(ValueError):
        prepare_market_data(
            data
        )


def test_non_numeric_price_is_rejected():
    data = make_ohlcv()

    data["Close"] = "invalid"

    with pytest.raises(ValueError):
        prepare_market_data(
            data
        )


def test_non_finite_price_is_rejected():
    data = make_ohlcv()

    data.loc[
        data.index[20],
        "Close",
    ] = np.inf

    with pytest.raises(ValueError):
        prepare_market_data(
            data
        )


def test_summary_contains_expected_fields():
    data = make_ohlcv()

    result = prepare_market_data(
        data,
        symbol="TCS.NS",
        timeframe="1D",
    )

    summary = preparation_summary(
        result
    )

    assert summary["symbol"] == "TCS.NS"
    assert summary["timeframe"] == "1D"
    assert summary["rows"] == len(data)
    assert (
        summary["quality_passed"]
        is True
    )
    assert (
        summary["research_only"]
        is True
    )


def test_research_only_metadata():
    data = make_ohlcv()

    result = prepare_market_data(
        data
    )

    assert (
        result.metadata[
            "research_only"
        ]
        is True
    )

    assert (
        result.metadata[
            "production_approved"
        ]
        is False
    )

    assert (
        result.metadata[
            "future_values_used"
        ]
        is False
    )

    assert (
        result.metadata[
            "targets_created"
        ]
        is False
    )

    assert (
        result.metadata[
            "features_created"
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
            "final_holdout_used"
        ]
        is False
    )


def test_quality_report_is_attached():
    data = make_ohlcv()

    result = prepare_market_data(
        data
    )

    assert result.quality_report is not None
    assert result.quality_report.passed is True


def test_preparation_is_deterministic():
    data = make_ohlcv()

    first = prepare_market_data(
        data,
        symbol="INFY.NS",
        timeframe="1D",
    )

    second = prepare_market_data(
        data,
        symbol="INFY.NS",
        timeframe="1D",
    )

    pd.testing.assert_frame_equal(
        first.data,
        second.data,
    )

    assert (
        first.warnings
        == second.warnings
    )

    assert (
        first.metadata
        == second.metadata
    )


def test_future_columns_are_not_created():
    data = make_ohlcv()

    result = prepare_market_data(
        data
    )

    future_columns = [
        column
        for column in result.data.columns
        if column.lower().startswith(
            (
                "future_",
                "direction_",
                "target_",
            )
        )
    ]

    assert future_columns == []


def test_preparation_does_not_create_features():
    data = make_ohlcv()

    result = prepare_market_data(
        data
    )

    assert (
        result.metadata[
            "features_created"
        ]
        is False
    )


def test_symbol_and_timeframe_are_optional():
    data = make_ohlcv()

    result = prepare_market_data(
        data
    )

    assert result.symbol is None
    assert result.timeframe is None
