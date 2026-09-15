"""
Tests for unified feature integration.

The tests focus on:
- deterministic feature construction
- target/future leakage protection
- market-context integration
- multi-timeframe integration
- numeric model features
- timeline preservation
- input immutability
- future-data mutation resistance
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.feature_integration import (
    UnifiedFeatureConfig,
    UnifiedFeatureResult,
    build_unified_features,
    feature_matrix,
    unified_feature_names,
    unified_feature_summary,
)
from src.research.market_context import (
    MarketContextConfig,
    build_market_context,
)
from src.research.market_data_context import (
    MarketContextData,
)


def make_stock_data(
    rows: int = 500,
) -> pd.DataFrame:
    index = pd.date_range(
        "2022-01-03",
        periods=rows,
        freq="D",
    )

    close = (
        100
        + np.cumsum(
            np.sin(
                np.arange(rows) / 17.0
            )
            * 0.35
        )
        + np.linspace(
            0,
            20,
            rows,
        )
    )

    open_price = (
        close
        + np.sin(
            np.arange(rows) / 9.0
        )
        * 0.4
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
        + (
            np.arange(rows)
            % 20
        )
        * 2_000
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


def make_market_context(
    rows: int = 500,
) -> MarketContextData:
    index = pd.date_range(
        "2022-01-03",
        periods=rows,
        freq="D",
    )

    base = (
        100
        + np.cumsum(
            np.sin(
                np.arange(rows) / 20.0
            )
            * 0.25
        )
    )

    benchmark = pd.DataFrame(
        {
            "NIFTY50_Close": base,
            "NIFTY50_Return_1": pd.Series(
                base,
                index=index,
            ).pct_change(),
        },
        index=index,
    )

    context = build_market_context(
        {
            "NIFTY50": pd.DataFrame(
                {
                    "Close": base,
                },
                index=index,
            ),
            "SENSEX": pd.DataFrame(
                {
                    "Close": base * 1.05,
                },
                index=index,
            ),
            "NIFTYBANK": pd.DataFrame(
                {
                    "Close": base * 0.98,
                },
                index=index,
            ),
        },
        config=MarketContextConfig(),
    )

    assert isinstance(
        context,
        MarketContextData,
    )

    return context


def make_timeframe_data(
    stock: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    daily = stock.copy(
        deep=True
    )

    weekly = (
        stock.resample("W")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
    )

    monthly = (
        stock.resample("MS")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
    )

    four_hour = stock.copy(
        deep=True
    )

    return {
        "4H": four_hour,
        "1D": daily,
        "1W": weekly,
        "1M": monthly,
    }


def test_config_defaults_are_safe():
    config = UnifiedFeatureConfig()

    assert config.include_market_context is True
    assert config.include_relative_strength is True
    assert config.include_multi_timeframe is True
    assert config.benchmark == "NIFTY50"
    assert config.relative_strength_window >= 2
    assert 0 <= config.max_missing_fraction <= 1


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        UnifiedFeatureConfig(
            relative_strength_window=1
        )

    with pytest.raises(ValueError):
        UnifiedFeatureConfig(
            max_missing_fraction=1.5
        )

    with pytest.raises(ValueError):
        UnifiedFeatureConfig(
            benchmark=""
        )


def test_basic_unified_feature_construction():
    stock = make_stock_data()

    result = build_unified_features(
        stock,
        config=UnifiedFeatureConfig(
            include_market_context=False,
            include_multi_timeframe=False,
        ),
    )

    assert isinstance(
        result,
        UnifiedFeatureResult,
    )

    assert result.rows == len(stock)
    assert result.feature_count > 0
    assert len(
        result.feature_names
    ) > 0


def test_feature_matrix_contains_only_declared_features():
    stock = make_stock_data()

    result = build_unified_features(
        stock,
        config=UnifiedFeatureConfig(
            include_market_context=False,
            include_multi_timeframe=False,
        ),
    )

    matrix = feature_matrix(
        result
    )

    assert list(
        matrix.columns
    ) == result.feature_names

    assert len(matrix) == len(
        stock
    )


def test_feature_names_are_unique():
    stock = make_stock_data()

    result = build_unified_features(
        stock,
        config=UnifiedFeatureConfig(
            include_market_context=False,
            include_multi_timeframe=False,
        ),
    )

    assert len(
        result.feature_names
    ) == len(
        set(result.feature_names)
    )


def test_features_are_numeric():
    stock = make_stock_data()

    result = build_unified_features(
        stock,
        config=UnifiedFeatureConfig(
            include_market_context=False,
            include_multi_timeframe=False,
        ),
    )

    matrix = feature_matrix(
        result
    )

    for column in matrix.columns:
        assert pd.api.types.is_numeric_dtype(
            matrix[column]
        )


def test_no_future_or_target_features():
    stock = make_stock_data()

    result = build_unified_features(
        stock,
        config=UnifiedFeatureConfig(
            include_market_context=False,
            include_multi_timeframe=False,
        ),
    )

    suspicious = [
        name
        for name in result.feature_names
        if any(
            token.lower()
            in name.lower()
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


def test_target_like_original_column_is_not_used_as_feature():
    stock = make_stock_data()

    stock["Target_Test"] = (
        stock["Close"]
        .shift(-5)
    )

    result = build_unified_features(
        stock,
        config=UnifiedFeatureConfig(
            include_market_context=False,
            include_multi_timeframe=False,
        ),
    )

    assert "Target_Test" not in (
        result.feature_names
    )


def test_label_like_original_column_is_not_used_as_feature():
    stock = make_stock_data()

    stock["Label"] = (
        stock["Close"]
        .shift(-5)
        .gt(stock["Close"])
        .astype(int)
    )

    result = build_unified_features(
        stock,
        config=UnifiedFeatureConfig(
            include_market_context=False,
            include_multi_timeframe=False,
        ),
    )

    assert "Label" not in (
        result.feature_names
    )


def test_input_dataframe_is_not_mutated():
    stock = make_stock_data()

    original = stock.copy(
        deep=True
    )

    build_unified_features(
        stock,
        config=UnifiedFeatureConfig(
            include_market_context=False,
            include_multi_timeframe=False,
        ),
    )

    pd.testing.assert_frame_equal(
        stock,
        original,
    )


def test_timeline_is_preserved():
    stock = make_stock_data()

    result = build_unified_features(
        stock,
        config=UnifiedFeatureConfig(
            include_market_context=False,
            include_multi_timeframe=False,
        ),
    )

    assert result.data.index.equals(
        stock.index
    )


def test_market_context_can_be_integrated():
    stock = make_stock_data()
    market = make_market_context()

    result = build_unified_features(
        stock,
        market_context=market,
        config=UnifiedFeatureConfig(
            include_market_context=True,
            include_relative_strength=True,
            include_multi_timeframe=False,
        ),
    )

    assert result.feature_count > 0

    assert len(
        result.market_feature_names
    ) > 0


def test_market_context_is_not_required_when_disabled():
    stock = make_stock_data()

    result = build_unified_features(
        stock,
        market_context=None,
        config=UnifiedFeatureConfig(
            include_market_context=False,
            include_multi_timeframe=False,
        ),
    )

    assert result.feature_count > 0


def test_missing_requested_market_context_generates_warning():
    stock = make_stock_data()

    result = build_unified_features(
        stock,
        market_context=None,
        config=UnifiedFeatureConfig(
            include_market_context=True,
            include_multi_timeframe=False,
        ),
    )

    assert any(
        "market context"
        in warning.lower()
        for warning in result.warnings
    )


def test_multi_timeframe_features_can_be_integrated():
    stock = make_stock_data()

    timeframe_data = (
        make_timeframe_data(
            stock
        )
    )

    result = build_unified_features(
        stock,
        timeframe_data=timeframe_data,
        config=UnifiedFeatureConfig(
            include_market_context=False,
            include_multi_timeframe=True,
        ),
    )

    assert result.feature_count > 0

    assert len(
        result.multi_timeframe_feature_names
    ) > 0


def test_multitimeframe_disabled_does_not_require_timeframe_data():
    stock = make_stock_data()

    result = build_unified_features(
        stock,
        timeframe_data=None,
        config=UnifiedFeatureConfig(
            include_market_context=False,
            include_multi_timeframe=False,
        ),
    )

    assert result.feature_count > 0


def test_future_market_mutation_does_not_change_earlier_features():
    stock = make_stock_data()
    market = make_market_context()

    baseline = build_unified_features(
        stock,
        market_context=market,
        config=UnifiedFeatureConfig(
            include_market_context=True,
            include_relative_strength=True,
            include_multi_timeframe=False,
        ),
    )

    mutated_market = make_market_context()

    mutation_start = len(
        mutated_market.context
    ) // 2

    mutated_market.context.iloc[
        mutation_start:,
        :
    ] = (
        mutated_market.context.iloc[
            mutation_start:,
            :
        ]
        * 10.0
    )

    mutated = build_unified_features(
        stock,
        market_context=mutated_market,
        config=UnifiedFeatureConfig(
            include_market_context=True,
            include_relative_strength=True,
            include_multi_timeframe=False,
        ),
    )

    common_index = baseline.data.index[
        :mutation_start
    ]

    common_features = [
        column
        for column in baseline.feature_names
        if column in mutated.feature_names
    ]

    pd.testing.assert_frame_equal(
        baseline.data.loc[
            common_index,
            common_features,
        ],
        mutated.data.loc[
            common_index,
            common_features,
        ],
        check_dtype=False,
    )


def test_feature_generation_is_deterministic():
    stock = make_stock_data()

    config = UnifiedFeatureConfig(
        include_market_context=False,
        include_multi_timeframe=False,
    )

    first = build_unified_features(
        stock,
        config=config,
    )

    second = build_unified_features(
        stock,
        config=config,
    )

    assert first.feature_names == (
        second.feature_names
    )

    pd.testing.assert_frame_equal(
        first.data,
        second.data,
    )


def test_feature_name_helper_returns_copy():
    stock = make_stock_data()

    result = build_unified_features(
        stock,
        config=UnifiedFeatureConfig(
            include_market_context=False,
            include_multi_timeframe=False,
        ),
    )

    names = unified_feature_names(
        result
    )

    names.append(
        "SHOULD_NOT_CHANGE_RESULT"
    )

    assert (
        "SHOULD_NOT_CHANGE_RESULT"
        not in result.feature_names
    )


def test_summary_contains_expected_fields():
    stock = make_stock_data()

    result = build_unified_features(
        stock,
        config=UnifiedFeatureConfig(
            include_market_context=False,
            include_multi_timeframe=False,
        ),
    )

    summary = unified_feature_summary(
        result
    )

    assert summary["rows"] == len(
        stock
    )

    assert (
        summary["feature_count"]
        == result.feature_count
    )

    assert (
        summary["research_only"]
        is True
    )


def test_research_only_metadata_is_explicit():
    stock = make_stock_data()

    result = build_unified_features(
        stock,
        config=UnifiedFeatureConfig(
            include_market_context=False,
            include_multi_timeframe=False,
        ),
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
            "final_holdout_used"
        ]
        is False
    )

    assert (
        result.metadata[
            "future_values_used"
        ]
        is False
    )


def test_invalid_stock_input_is_rejected():
    with pytest.raises(TypeError):
        build_unified_features(
            pd.DataFrame()
        )


def test_missing_close_is_rejected():
    stock = make_stock_data()

    stock = stock.drop(
        columns=["Close"]
    )

    with pytest.raises(ValueError):
        build_unified_features(
            stock
        )


def test_duplicate_timestamps_are_rejected():
    stock = make_stock_data()

    duplicate = stock.iloc[
        [0]
    ].copy()

    duplicate.index = [
        stock.index[1]
    ]

    bad = pd.concat(
        [
            stock,
            duplicate,
        ]
    )

    with pytest.raises(ValueError):
        build_unified_features(
            bad
        )


def test_unsorted_index_is_rejected():
    stock = make_stock_data()

    bad = stock.iloc[
        ::-1
    ]

    with pytest.raises(ValueError):
        build_unified_features(
            bad
        )


def test_non_datetime_index_is_rejected():
    stock = make_stock_data()

    stock.index = range(
        len(stock)
    )

    with pytest.raises(TypeError):
        build_unified_features(
            stock
        )


def test_non_numeric_feature_from_market_context_is_rejected():
    stock = make_stock_data()

    market = make_market_context()

    market.context[
        "Artificial_Text_Feature"
    ] = "text"

    with pytest.raises(TypeError):
        build_unified_features(
            stock,
            market_context=market,
            config=UnifiedFeatureConfig(
                include_market_context=True,
                include_multi_timeframe=False,
            ),
        )


def test_feature_matrix_does_not_modify_result():
    stock = make_stock_data()

    result = build_unified_features(
        stock,
        config=UnifiedFeatureConfig(
            include_market_context=False,
            include_multi_timeframe=False,
        ),
    )

    matrix = feature_matrix(
        result
    )

    if not matrix.empty:
        first_column = matrix.columns[0]
        original_value = (
            result.data.loc[
                matrix.index[0],
                first_column,
            ]
        )

        matrix.loc[
            matrix.index[0],
            first_column,
        ] = 999999.0

        assert (
            result.data.loc[
                matrix.index[0],
                first_column,
            ]
            == original_value
        )
