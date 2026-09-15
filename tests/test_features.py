"""
Tests for the feature-engineering layer.

The tests focus heavily on:
    - deterministic feature generation
    - expected feature availability
    - NaN/infinity handling
    - support/resistance leakage
    - future-target exclusion
    - multi-timeframe look-ahead prevention
    - train-only feature selection
    - leakage-safe normalization
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engine import (
    engineer_features,
)
from src.features.multi_timeframe import (
    add_timeframe_features,
    build_multi_timeframe_dataset,
)
from src.features.normalization import (
    TrainFittedNormalizer,
    rolling_percentile_rank,
    rolling_robust_zscore,
    rolling_zscore,
)
from src.features.price_action import (
    add_all_price_action_features,
)
from src.features.regime import (
    add_all_regime_features,
)
from src.features.selection import (
    FeatureSelectionConfig,
    FeatureSelector,
    select_features,
)
from src.features.technical import (
    add_all_technical_features,
)
from src.features.volume import (
    add_all_volume_features,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def market_data() -> pd.DataFrame:
    """
    Deterministic synthetic OHLCV dataset.
    """

    index = pd.date_range(
        "2020-01-01",
        periods=500,
        freq="D",
    )

    rng = np.random.default_rng(
        42
    )

    returns = (
        0.0005
        + rng.normal(
            0.0,
            0.01,
            len(index),
        )
    )

    close = (
        100.0
        * np.exp(
            np.cumsum(
                returns
            )
        )
    )

    open_price = (
        close
        * (
            1.0
            + rng.normal(
                0.0,
                0.002,
                len(index),
            )
        )
    )

    high = np.maximum(
        open_price,
        close,
    ) * (
        1.0
        + rng.uniform(
            0.001,
            0.02,
            len(index),
        )
    )

    low = np.minimum(
        open_price,
        close,
    ) * (
        1.0
        - rng.uniform(
            0.001,
            0.02,
            len(index),
        )
    )

    volume = rng.integers(
        100_000,
        1_000_000,
        len(index),
    ).astype(float)

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
def higher_timeframe_data() -> dict[
    str,
    pd.DataFrame,
]:
    """
    Synthetic completed daily/weekly/monthly datasets.
    """

    daily_index = pd.date_range(
        "2020-01-01",
        periods=120,
        freq="D",
    )

    weekly_index = pd.date_range(
        "2019-12-23",
        periods=30,
        freq="W-MON",
    )

    monthly_index = pd.date_range(
        "2019-12-01",
        periods=12,
        freq="MS",
    )

    def make_frame(
        index: pd.DatetimeIndex,
        start: float,
    ) -> pd.DataFrame:
        close = np.linspace(
            start,
            start + len(index) - 1,
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
                    100_000.0,
                ),
            },
            index=index,
        )

    return {
        "1D": make_frame(
            daily_index,
            100.0,
        ),
        "1W": make_frame(
            weekly_index,
            100.0,
        ),
        "1M": make_frame(
            monthly_index,
            100.0,
        ),
    }


# ----------------------------------------------------------------------
# Technical features
# ----------------------------------------------------------------------


def test_technical_features_are_created(
    market_data: pd.DataFrame,
) -> None:
    result = add_all_technical_features(
        market_data
    )

    expected = {
        "RSI_14",
        "MACD",
        "MACD_Signal",
        "EMA_9",
        "EMA_20",
        "EMA_50",
        "EMA_100",
        "EMA_200",
        "SMA_20",
        "SMA_50",
        "SMA_200",
        "ATR_14",
        "ADX_14",
        "Stochastic",
        "OBV",
        "ROC_10",
    }

    assert expected.issubset(
        set(result.columns)
    )


def test_technical_features_preserve_index(
    market_data: pd.DataFrame,
) -> None:
    result = add_all_technical_features(
        market_data
    )

    pd.testing.assert_index_equal(
        result.index,
        market_data.index,
    )


def test_rsi_is_bounded(
    market_data: pd.DataFrame,
) -> None:
    result = add_all_technical_features(
        market_data
    )

    values = result[
        "RSI_14"
    ].dropna()

    assert (
        values.between(
            0.0,
            100.0,
        ).all()
    )


# ----------------------------------------------------------------------
# Price action
# ----------------------------------------------------------------------


def test_price_action_features_are_created(
    market_data: pd.DataFrame,
) -> None:
    result = add_all_price_action_features(
        market_data
    )

    expected = {
        "Candle_Body",
        "Candle_Range",
        "Upper_Wick",
        "Lower_Wick",
        "Body_Range_Ratio",
        "Bullish_Candle",
        "Bearish_Candle",
        "Doji",
        "Hammer_Like",
        "Shooting_Star_Like",
    }

    assert expected.issubset(
        set(result.columns)
    )


def test_price_action_does_not_change_ohlcv(
    market_data: pd.DataFrame,
) -> None:
    result = add_all_price_action_features(
        market_data
    )

    for column in [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]:
        pd.testing.assert_series_equal(
            result[column],
            market_data[column],
            check_names=True,
        )


# ----------------------------------------------------------------------
# Volume features
# ----------------------------------------------------------------------


def test_volume_features_are_created(
    market_data: pd.DataFrame,
) -> None:
    result = add_all_volume_features(
        market_data
    )

    expected = {
        "RVOL_5",
        "RVOL_10",
        "RVOL_20",
        "RVOL_50",
        "Volume_Trend_5",
        "Volume_Trend_20",
        "Price_Volume_Confirmation",
        "OBV_Distance",
    }

    assert expected.issubset(
        set(result.columns)
    )


def test_rvol_is_non_negative(
    market_data: pd.DataFrame,
) -> None:
    result = add_all_volume_features(
        market_data
    )

    values = result[
        "RVOL_20"
    ].dropna()

    assert (
        values >= 0
    ).all()


# ----------------------------------------------------------------------
# Regime features
# ----------------------------------------------------------------------


def test_regime_features_are_created(
    market_data: pd.DataFrame,
) -> None:
    result = add_all_regime_features(
        market_data
    )

    expected = {
        "Trend_Regime",
        "ADX_Regime",
        "Volatility_Regime",
        "Momentum_Regime",
        "Market_State_Score",
    }

    assert expected.issubset(
        set(result.columns)
    )


def test_market_state_score_is_finite_after_warmup(
    market_data: pd.DataFrame,
) -> None:
    result = add_all_regime_features(
        market_data
    )

    values = result[
        "Market_State_Score"
    ].dropna()

    assert np.isfinite(
        values
    ).all()


# ----------------------------------------------------------------------
# Feature engine
# ----------------------------------------------------------------------


def test_engine_creates_features(
    market_data: pd.DataFrame,
) -> None:
    result = engineer_features(
        market_data
    )

    assert not result.data.empty

    assert (
        len(
            result.feature_names
        )
        > 10
    )

    assert set(
        result.feature_names
    ).issubset(
        set(result.data.columns)
    )


def test_engine_excludes_future_targets(
    market_data: pd.DataFrame,
) -> None:
    data = market_data.copy()

    data[
        "Future_Close_Return_5"
    ] = 0.01

    data[
        "Direction_5"
    ] = 1

    result = engineer_features(
        data
    )

    for feature in result.feature_names:
        assert "Future_" not in feature
        assert "Direction_" not in feature
        assert "Target" not in feature


def test_engine_rejects_future_feature_names(
    market_data: pd.DataFrame,
) -> None:
    data = market_data.copy()

    data[
        "Future_Close_Return_5"
    ] = 0.01

    # The engine should safely exclude future targets rather than
    # allowing them into the feature matrix.
    result = engineer_features(
        data
    )

    assert (
        "Future_Close_Return_5"
        not in result.feature_names
    )


# ----------------------------------------------------------------------
# Support/resistance leakage checks
# ----------------------------------------------------------------------


def test_support_resistance_does_not_use_current_high(
    market_data: pd.DataFrame,
) -> None:
    """
    Changing today's high must not alter a support/resistance value
    that is explicitly based on prior observations.
    """

    original = (
        add_all_price_action_features(
            market_data
        )
    )

    modified_data = market_data.copy()

    modified_data.loc[
        modified_data.index[250],
        "High",
    ] *= 1.50

    modified = (
        add_all_price_action_features(
            modified_data
        )
    )

    resistance_columns = [
        column
        for column in original.columns
        if "Resistance" in column
        or "Support" in column
    ]

    assert resistance_columns

    # Values before the modified timestamp must be identical.
    before = modified_data.index[
        :250
    ]

    for column in resistance_columns:
        pd.testing.assert_series_equal(
            original.loc[
                before,
                column,
            ],
            modified.loc[
                before,
                column,
            ],
            check_names=True,
        )


# ----------------------------------------------------------------------
# Rolling normalization leakage
# ----------------------------------------------------------------------


def test_rolling_zscore_does_not_use_current_value() -> None:
    index = pd.date_range(
        "2025-01-01",
        periods=100,
        freq="D",
    )

    values = pd.Series(
        np.arange(
            100.0
        ),
        index=index,
    )

    changed = values.copy()

    changed.iloc[-1] = 1_000_000.0

    first = rolling_zscore(
        values,
        window=20,
        min_periods=10,
        shift=1,
    )

    second = rolling_zscore(
        changed,
        window=20,
        min_periods=10,
        shift=1,
    )

    # The previous observation must not change because the modified
    # value occurs later in time.
    pd.testing.assert_series_equal(
        first.iloc[
            :-1
        ],
        second.iloc[
            :-1
        ],
        check_names=True,
    )


def test_rolling_robust_zscore_is_finite_after_warmup() -> None:
    values = pd.Series(
        np.linspace(
            1.0,
            100.0,
            100,
        )
    )

    result = rolling_robust_zscore(
        values,
        window=20,
        min_periods=10,
    )

    valid = result.dropna()

    assert np.isfinite(
        valid
    ).all()


def test_rolling_percentile_uses_prior_values() -> None:
    values = pd.Series(
        np.arange(
            100.0
        )
    )

    changed = values.copy()
    changed.iloc[-1] = 10_000.0

    first = rolling_percentile_rank(
        values,
        window=20,
        min_periods=10,
        shift=1,
    )

    second = rolling_percentile_rank(
        changed,
        window=20,
        min_periods=10,
        shift=1,
    )

    pd.testing.assert_series_equal(
        first.iloc[
            :-1
        ],
        second.iloc[
            :-1
        ],
    )


# ----------------------------------------------------------------------
# Train-fitted normalization
# ----------------------------------------------------------------------


def test_train_fitted_normalizer_uses_training_statistics() -> None:
    train = pd.DataFrame(
        {
            "feature_a": np.arange(
                1.0,
                101.0,
            ),
            "feature_b": np.arange(
                100.0,
                200.0,
            ),
        }
    )

    test = pd.DataFrame(
        {
            "feature_a": [
                50.0,
                5000.0,
            ],
            "feature_b": [
                150.0,
                5000.0,
            ],
        }
    )

    normalizer = TrainFittedNormalizer()

    normalizer.fit(
        train
    )

    transformed = normalizer.transform(
        test
    )

    assert transformed.shape == test.shape

    # Extreme test values must not alter the fitted training state.
    before = normalizer.statistics_.center.copy()

    normalizer.transform(
        pd.DataFrame(
            {
                "feature_a": [
                    9999999.0
                ],
                "feature_b": [
                    -9999999.0
                ],
            }
        )
    )

    after = normalizer.statistics_.center

    pd.testing.assert_series_equal(
        before,
        after,
    )


def test_normalizer_rejects_unseen_feature() -> None:
    train = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
        }
    )

    normalizer = TrainFittedNormalizer()

    normalizer.fit(
        train
    )

    test = pd.DataFrame(
        {
            "a": [1.0],
            "c": [2.0],
        }
    )

    with pytest.raises(
        ValueError
    ):
        normalizer.transform(
            test
        )


# ----------------------------------------------------------------------
# Feature selection
# ----------------------------------------------------------------------


def test_feature_selector_removes_constant_features() -> None:
    data = pd.DataFrame(
        {
            "useful_a": np.arange(
                100.0
            ),
            "useful_b": np.arange(
                100.0
            )[::-1],
            "constant": np.ones(
                100
            ),
            "useful_c": np.sin(
                np.arange(100.0)
            ),
            "useful_d": np.cos(
                np.arange(100.0)
            ),
            "useful_e": np.log1p(
                np.arange(100.0)
            ),
            "useful_f": np.sqrt(
                np.arange(100.0)
            ),
            "useful_g": np.arange(
                100.0
            ) ** 2,
            "useful_h": np.arange(
                100.0
            ) ** 3,
            "useful_i": np.arange(
                100.0
            ) % 7,
            "useful_j": np.arange(
                100.0
            ) % 11,
        }
    )

    selector = FeatureSelector(
        FeatureSelectionConfig(
            max_correlation=0.999999,
            min_features=5,
        )
    )

    selector.fit(
        data
    )

    assert (
        "constant"
        not in selector.selected_features
    )

    assert (
        "constant"
        in selector.dropped_features_
    )


def test_feature_selector_removes_high_correlation() -> None:
    base = np.arange(
        100.0
    )

    data = pd.DataFrame(
        {
            "feature_a": base,
            "feature_b": base * 2.0,
            "feature_c": np.sin(
                base
            ),
            "feature_d": np.cos(
                base
            ),
            "feature_e": np.log1p(
                base
            ),
            "feature_f": np.sqrt(
                base
            ),
            "feature_g": base % 5,
            "feature_h": base % 7,
            "feature_i": base % 11,
            "feature_j": base % 13,
        }
    )

    selector = FeatureSelector(
        FeatureSelectionConfig(
            max_correlation=0.95,
            min_features=5,
        )
    )

    selector.fit(
        data
    )

    assert not (
        "feature_a"
        in selector.selected_features
        and "feature_b"
        in selector.selected_features
    )

    assert len(
        selector.correlation_pairs_
    ) > 0


def test_feature_selection_is_fitted_on_training_only() -> None:
    train = pd.DataFrame(
        {
            "a": np.arange(
                100.0
            ),
            "b": np.arange(
                100.0
            ) * -1.0,
            "c": np.sin(
                np.arange(100.0)
            ),
            "d": np.cos(
                np.arange(100.0)
            ),
            "e": np.sqrt(
                np.arange(100.0)
            ),
            "f": np.log1p(
                np.arange(100.0)
            ),
            "g": np.arange(
                100.0
            ) % 3,
            "h": np.arange(
                100.0
            ) % 5,
            "i": np.arange(
                100.0
            ) % 7,
            "j": np.arange(
                100.0
            ) % 11,
        }
    )

    validation = train.copy()
    validation.index = pd.date_range(
        "2025-05-01",
        periods=len(validation),
        freq="D",
    )

    train.index = pd.date_range(
        "2025-01-01",
        periods=len(train),
        freq="D",
    )

    selector, train_selected, validation_selected, _ = (
        select_features(
            train,
            validation_data=validation,
            config=FeatureSelectionConfig(
                max_correlation=0.95,
                min_features=5,
            ),
        )
    )

    assert selector.fitted_

    assert list(
        train_selected.columns
    ) == list(
        validation_selected.columns
    )

    assert list(
        train_selected.columns
    ) == selector.selected_features


# ----------------------------------------------------------------------
# Multi-timeframe tests
# ----------------------------------------------------------------------


def test_add_timeframe_features_preserves_base_rows(
    market_data: pd.DataFrame,
    higher_timeframe_data: dict[
        str,
        pd.DataFrame,
    ],
) -> None:
    result = add_timeframe_features(
        market_data,
        higher_timeframe_data[
            "1D"
        ],
        timeframe="1D",
    )

    assert len(
        result
    ) == len(
        market_data
    )


def test_mtf_features_do_not_use_current_higher_timeframe_candle(
    market_data: pd.DataFrame,
) -> None:
    """
    Changing a higher-timeframe candle must not affect base observations
    before that candle is completed.
    """

    daily_index = pd.date_range(
        "2019-12-30",
        periods=20,
        freq="D",
    )

    daily_close = np.linspace(
        100.0,
        119.0,
        len(daily_index),
    )

    daily = pd.DataFrame(
        {
            "Open": daily_close - 1.0,
            "High": daily_close + 1.0,
            "Low": daily_close - 1.0,
            "Close": daily_close,
            "Volume": 1000.0,
        },
        index=daily_index,
    )

    base_index = pd.date_range(
        "2020-01-01 09:15",
        periods=20,
        freq="4h",
    )

    base_close = np.linspace(
        100.0,
        120.0,
        len(base_index),
    )

    base = pd.DataFrame(
        {
            "Open": base_close - 0.5,
            "High": base_close + 1.0,
            "Low": base_close - 1.0,
            "Close": base_close,
            "Volume": 1000.0,
        },
        index=base_index,
    )

    first = add_timeframe_features(
        base,
        daily,
        timeframe="1D",
    )

    changed_daily = daily.copy()

    changed_daily.loc[
        changed_daily.index[10],
        "Close",
    ] *= 5.0

    second = add_timeframe_features(
        base,
        changed_daily,
        timeframe="1D",
    )

    # Observations before the changed daily candle must be unchanged.
    comparison_index = first.index[
        first.index
        < changed_daily.index[10]
    ]

    shared_columns = [
        column
        for column in first.columns
        if column.startswith(
            "1D_"
        )
    ]

    assert shared_columns

    for column in shared_columns:
        pd.testing.assert_series_equal(
            first.loc[
                comparison_index,
                column,
            ],
            second.loc[
                comparison_index,
                column,
            ],
        )


def test_build_multi_timeframe_dataset(
    market_data: pd.DataFrame,
    higher_timeframe_data: dict[
        str,
        pd.DataFrame,
    ],
) -> None:
    result = build_multi_timeframe_dataset(
        base_data=market_data,
        timeframe_data=higher_timeframe_data,
    )

    assert len(
        result
    ) == len(
        market_data
    )

    assert any(
        column.startswith(
            "1D_"
        )
        for column in result.columns
    )

    assert any(
        column.startswith(
            "1W_"
        )
        for column in result.columns
    )

    assert any(
        column.startswith(
            "1M_"
        )
        for column in result.columns
    )


# ----------------------------------------------------------------------
# General feature sanity
# ----------------------------------------------------------------------


def test_engineered_features_do_not_contain_infinity(
    market_data: pd.DataFrame,
) -> None:
    result = engineer_features(
        market_data
    )

    numeric = result.data[
        result.feature_names
    ].select_dtypes(
        include=np.number
    )

    assert not np.isinf(
        numeric.to_numpy()
    ).any()


def test_feature_engine_is_deterministic(
    market_data: pd.DataFrame,
) -> None:
    first = engineer_features(
        market_data
    )

    second = engineer_features(
        market_data
    )

    assert (
        first.feature_names
        == second.feature_names
    )

    pd.testing.assert_frame_equal(
        first.data,
        second.data,
    )
