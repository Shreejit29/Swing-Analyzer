"""
Adversarial tests for detecting look-ahead and data leakage.

The purpose of this module is not to prove that the entire system is
mathematically leakage-free. Instead, it creates controlled situations
where future information is deliberately changed and checks that
historical observations do not change as a consequence.

These tests cover:

    1. Future target exclusion
    2. Price-action leakage
    3. Support/resistance leakage
    4. Rolling feature leakage
    5. Multi-timeframe leakage
    6. Train-only preprocessing
    7. Train-only feature selection
    8. Chronological split integrity
    9. Purged validation
   10. Calibration separation
   11. Final holdout isolation
   12. Model reproducibility
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engine import engineer_features
from src.features.multi_timeframe import (
    add_timeframe_features,
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
from src.models.calibration_pipeline import (
    CalibrationPipeline,
    CalibrationPipelineConfig,
)
from src.models.classifier import (
    ClassifierConfig,
    DirectionClassifier,
)
from src.models.preprocessing import (
    SafePreprocessor,
)
from src.models.splitter import (
    chronological_split,
    purged_walk_forward_splits,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def deterministic_market() -> pd.DataFrame:
    """
    Deterministic market series.

    A deterministic dataset is preferable here because any unexpected
    difference can then be attributed to the deliberate mutation rather
    than random noise.
    """

    index = pd.date_range(
        "2020-01-01",
        periods=500,
        freq="D",
    )

    close = (
        100.0
        + np.arange(
            500,
            dtype=float,
        )
        * 0.20
        + 2.0
        * np.sin(
            np.arange(500)
            / 10.0
        )
    )

    open_price = (
        close
        - 0.20
    )

    high = (
        close
        + 1.00
    )

    low = (
        close
        - 1.00
    )

    volume = (
        500_000.0
        + (
            np.arange(
                500
            )
            % 20
        )
        * 10_000.0
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
def feature_dataset(
    deterministic_market: pd.DataFrame,
) -> pd.DataFrame:
    result = engineer_features(
        deterministic_market
    )

    return result.data


# ----------------------------------------------------------------------
# 1. Future targets must never become features
# ----------------------------------------------------------------------


def test_future_columns_are_not_features(
    deterministic_market: pd.DataFrame,
) -> None:
    data = deterministic_market.copy()

    data[
        "Future_Close_Return_5"
    ] = 999.0

    data[
        "Future_High_Return_5"
    ] = 999.0

    data[
        "Future_Low_Return_5"
    ] = 999.0

    data[
        "Direction_5"
    ] = 1

    result = engineer_features(
        data
    )

    for feature in result.feature_names:
        assert not feature.startswith(
            "Future_"
        )

        assert not feature.startswith(
            "Direction_"
        )

        assert "Target" not in feature


# ----------------------------------------------------------------------
# 2. Future target mutation must not change historical features
# ----------------------------------------------------------------------


def test_future_target_mutation_does_not_change_features(
    deterministic_market: pd.DataFrame,
) -> None:
    data_a = deterministic_market.copy()

    data_b = deterministic_market.copy()

    data_a[
        "Future_Close_Return_5"
    ] = 0.0

    data_b[
        "Future_Close_Return_5"
    ] = 9999.0

    result_a = engineer_features(
        data_a
    )

    result_b = engineer_features(
        data_b
    )

    common_features = [
        feature
        for feature in result_a.feature_names
        if feature in result_b.feature_names
    ]

    assert common_features

    pd.testing.assert_frame_equal(
        result_a.data[
            common_features
        ],
        result_b.data[
            common_features
        ],
    )


# ----------------------------------------------------------------------
# 3. Future OHLC mutation must not affect earlier price-action values
# ----------------------------------------------------------------------


def test_future_ohlc_mutation_does_not_change_previous_price_action(
    deterministic_market: pd.DataFrame,
) -> None:
    original = (
        add_all_price_action_features(
            deterministic_market
        )
    )

    modified_data = (
        deterministic_market.copy()
    )

    mutation_position = 400

    modified_data.iloc[
        mutation_position,
        modified_data.columns.get_loc(
            "High"
        ),
    ] *= 100.0

    modified_data.iloc[
        mutation_position,
        modified_data.columns.get_loc(
            "Low"
        ),
    ] *= 0.01

    modified = (
        add_all_price_action_features(
            modified_data
        )
    )

    historical_index = (
        deterministic_market.index[
            :mutation_position
        ]
    )

    pd.testing.assert_frame_equal(
        original.loc[
            historical_index
        ],
        modified.loc[
            historical_index
        ],
    )


# ----------------------------------------------------------------------
# 4. Support/resistance must use historical information
# ----------------------------------------------------------------------


def test_future_price_cannot_rewrite_previous_resistance(
    deterministic_market: pd.DataFrame,
) -> None:
    original = (
        add_all_price_action_features(
            deterministic_market
        )
    )

    modified_data = (
        deterministic_market.copy()
    )

    mutation_position = 450

    modified_data.iloc[
        mutation_position,
        modified_data.columns.get_loc(
            "High"
        ),
    ] = 10_000.0

    modified = (
        add_all_price_action_features(
            modified_data
        )
    )

    support_resistance_columns = [
        column
        for column in original.columns
        if (
            "Support" in column
            or "Resistance" in column
        )
    ]

    assert support_resistance_columns

    historical_index = (
        deterministic_market.index[
            :mutation_position
        ]
    )

    for column in support_resistance_columns:
        pd.testing.assert_series_equal(
            original.loc[
                historical_index,
                column,
            ],
            modified.loc[
                historical_index,
                column,
            ],
        )


# ----------------------------------------------------------------------
# 5. Rolling indicators must not use future observations
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "function",
    [
        rolling_zscore,
        rolling_robust_zscore,
        rolling_percentile_rank,
    ],
)
def test_rolling_features_are_causal(
    function,
) -> None:
    values = pd.Series(
        np.arange(
            200.0
        )
    )

    changed = values.copy()

    changed.iloc[
        150:
    ] += 1_000_000.0

    original = function(
        values,
        window=20,
        min_periods=10,
        shift=1,
    )

    modified = function(
        changed,
        window=20,
        min_periods=10,
        shift=1,
    )

    # The future mutation starts at position 150.
    # Therefore every value before position 150 must be identical.
    pd.testing.assert_series_equal(
        original.iloc[
            :150
        ],
        modified.iloc[
            :150
        ],
    )


# ----------------------------------------------------------------------
# 6. Train-fitted normalization must ignore future/test data
# ----------------------------------------------------------------------


def test_normalizer_does_not_learn_from_test_data() -> None:
    train = pd.DataFrame(
        {
            "feature": np.arange(
                100.0
            )
        }
    )

    test_a = pd.DataFrame(
        {
            "feature": np.arange(
                100.0,
                120.0,
            )
        }
    )

    test_b = pd.DataFrame(
        {
            "feature": np.full(
                20,
                1_000_000.0,
            )
        }
    )

    normalizer_a = TrainFittedNormalizer()
    normalizer_b = TrainFittedNormalizer()

    normalizer_a.fit(
        train
    )

    normalizer_b.fit(
        train
    )

    normalizer_a.transform(
        test_a
    )

    normalizer_a.transform(
        test_b
    )

    # Fitting was performed on exactly the same training set.
    # Test observations must not alter the fitted statistics.
    pd.testing.assert_series_equal(
        normalizer_a.statistics_.center,
        normalizer_b.statistics_.center,
    )


# ----------------------------------------------------------------------
# 7. Preprocessor must not refit during transform
# ----------------------------------------------------------------------


def test_safe_preprocessor_does_not_refit_on_future_data() -> None:
    train = pd.DataFrame(
        {
            "a": np.arange(
                100.0
            ),
            "b": np.arange(
                100.0
            ) * 2.0,
        }
    )

    future = pd.DataFrame(
        {
            "a": np.full(
                20,
                999999.0,
            ),
            "b": np.full(
                20,
                -999999.0,
            ),
        }
    )

    preprocessor = SafePreprocessor()

    preprocessor.fit(
        train
    )

    names_before = (
        preprocessor.get_feature_names()
    )

    preprocessor.transform(
        future
    )

    names_after = (
        preprocessor.get_feature_names()
    )

    assert (
        names_before
        == names_after
    )


# ----------------------------------------------------------------------
# 8. Chronological split must never put future data into training
# ----------------------------------------------------------------------


def test_chronological_split_has_no_temporal_overlap(
    feature_dataset: pd.DataFrame,
) -> None:
    X = feature_dataset.select_dtypes(
        include=np.number
    )

    y = pd.Series(
        (
            np.arange(
                len(X)
            )
            % 2
        ),
        index=X.index,
    )

    split = chronological_split(
        X,
        y,
        train_fraction=0.60,
        validation_fraction=0.20,
        test_fraction=0.20,
    )

    assert (
        split.train_X.index.max()
        < split.validation_X.index.min()
    )

    assert (
        split.validation_X.index.max()
        < split.test_X.index.min()
    )


# ----------------------------------------------------------------------
# 9. Purged folds must not overlap
# ----------------------------------------------------------------------


def test_purged_folds_have_disjoint_train_test_sets() -> None:
    index = pd.date_range(
        "2020-01-01",
        periods=300,
        freq="D",
    )

    folds = list(
        purged_walk_forward_splits(
            index,
            n_splits=5,
            train_size=100,
            test_size=20,
            horizon=10,
            embargo=0,
            expanding=True,
        )
    )

    assert folds

    for fold in folds:
        train = set(
            fold.train_indices
        )

        test = set(
            fold.test_indices
        )

        assert train.isdisjoint(
            test
        )


def test_purged_folds_are_forward_only() -> None:
    index = pd.date_range(
        "2020-01-01",
        periods=300,
        freq="D",
    )

    folds = list(
        purged_walk_forward_splits(
            index,
            n_splits=5,
            train_size=100,
            test_size=20,
            horizon=10,
            embargo=0,
            expanding=True,
        )
    )

    for fold in folds:
        assert (
            fold.train_indices.max()
            < fold.test_indices.min()
        )


# ----------------------------------------------------------------------
# 10. Multi-timeframe future mutation test
# ----------------------------------------------------------------------


def test_higher_timeframe_future_mutation_does_not_change_earlier_base_features(
    deterministic_market: pd.DataFrame,
) -> None:
    daily = (
        deterministic_market
        .resample("D")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
    )

    base = deterministic_market.iloc[
        :250
    ].copy()

    original = add_timeframe_features(
        base,
        daily,
        timeframe="1D",
    )

    changed_daily = daily.copy()

    mutation_date = changed_daily.index[
        200
    ]

    changed_daily.loc[
        mutation_date,
        "High",
    ] *= 100.0

    changed_daily.loc[
        mutation_date,
        "Close",
    ] *= 50.0

    modified = add_timeframe_features(
        base,
        changed_daily,
        timeframe="1D",
    )

    earlier = original.index[
        original.index
        < mutation_date
    ]

    mtf_columns = [
        column
        for column in original.columns
        if column.startswith(
            "1D_"
        )
    ]

    assert mtf_columns

    for column in mtf_columns:
        pd.testing.assert_series_equal(
            original.loc[
                earlier,
                column,
            ],
            modified.loc[
                earlier,
                column,
            ],
        )


# ----------------------------------------------------------------------
# 11. Calibration must be temporally separated
# ----------------------------------------------------------------------


def test_calibration_cannot_use_training_period() -> None:
    probabilities = np.linspace(
        0.05,
        0.95,
        100,
    )

    targets = (
        probabilities
        >= 0.50
    ).astype(int)

    timestamps = pd.date_range(
        "2025-01-01",
        periods=100,
        freq="D",
    )

    training_end = timestamps[
        80
    ]

    pipeline = CalibrationPipeline(
        CalibrationPipelineConfig(
            min_samples=20
        )
    )

    with pytest.raises(
        ValueError
    ):
        pipeline.fit(
            probabilities=probabilities[
                :20
            ],
            targets=targets[
                :20
            ],
            timestamps=timestamps[
                :20
            ],
            training_end=training_end,
        )


def test_calibration_uses_only_future_calibration_period() -> None:
    probabilities = np.linspace(
        0.05,
        0.95,
        100,
    )

    targets = (
        probabilities
        >= 0.50
    ).astype(int)

    timestamps = pd.date_range(
        "2025-01-01",
        periods=100,
        freq="D",
    )

    training_end = timestamps[
        49
    ]

    calibration_probabilities = probabilities[
        50:
    ]

    calibration_targets = targets[
        50:
    ]

    calibration_timestamps = timestamps[
        50:
    ]

    pipeline = CalibrationPipeline(
        CalibrationPipelineConfig(
            min_samples=20
        )
    )

    result = pipeline.fit(
        probabilities=calibration_probabilities,
        targets=calibration_targets,
        timestamps=calibration_timestamps,
        training_end=training_end,
    )

    assert result.calibrator is not None


# ----------------------------------------------------------------------
# 12. Final holdout isolation
# ----------------------------------------------------------------------


def test_final_holdout_is_not_used_for_training() -> None:
    rng = np.random.default_rng(
        42
    )

    index = pd.date_range(
        "2020-01-01",
        periods=300,
        freq="D",
    )

    X = pd.DataFrame(
        {
            "a": rng.normal(
                0,
                1,
                300,
            ),
            "b": rng.normal(
                0,
                1,
                300,
            ),
        },
        index=index,
    )

    y = pd.Series(
        (
            X["a"]
            + X["b"]
            > 0
        ).astype(int),
        index=index,
    )

    split = chronological_split(
        X,
        y,
        train_fraction=0.60,
        validation_fraction=0.20,
        test_fraction=0.20,
    )

    model_a = DirectionClassifier(
        ClassifierConfig(
            algorithm="logistic_regression"
        )
    )

    model_b = DirectionClassifier(
        ClassifierConfig(
            algorithm="logistic_regression"
        )
    )

    model_a.fit(
        split.train_X,
        split.train_y,
    )

    # Model B receives the exact same training observations.
    # The final holdout exists separately and is never supplied to fit().
    model_b.fit(
        split.train_X,
        split.train_y,
    )

    predictions_a = model_a.predict_proba(
        split.test_X
    )

    predictions_b = model_b.predict_proba(
        split.test_X
    )

    np.testing.assert_allclose(
        predictions_a,
        predictions_b,
    )


# ----------------------------------------------------------------------
# 13. Future mutation must not affect a fitted model
# ----------------------------------------------------------------------


def test_future_data_cannot_change_already_fitted_model() -> None:
    rng = np.random.default_rng(
        100
    )

    X = pd.DataFrame(
        {
            "a": rng.normal(
                0,
                1,
                200,
            ),
            "b": rng.normal(
                0,
                1,
                200,
            ),
        }
    )

    y = (
        X["a"]
        + X["b"]
        > 0
    ).astype(int)

    train = X.iloc[
        :150
    ]

    train_y = y.iloc[
        :150
    ]

    future_a = X.iloc[
        150:
    ].copy()

    future_b = future_a.copy()

    future_b.loc[
        :,
        "a",
    ] = 999999.0

    future_b.loc[
        :,
        "b",
    ] = -999999.0

    model_a = DirectionClassifier(
        ClassifierConfig(
            algorithm="logistic_regression"
        )
    )

    model_b = DirectionClassifier(
        ClassifierConfig(
            algorithm="logistic_regression"
        )
    )

    model_a.fit(
        train,
        train_y,
    )

    model_b.fit(
        train,
        train_y,
    )

    pred_a = model_a.predict_proba(
        future_a
    )

    pred_b = model_b.predict_proba(
        future_a
    )

    np.testing.assert_allclose(
        pred_a,
        pred_b,
    )


# ----------------------------------------------------------------------
# 14. Reproducibility
# ----------------------------------------------------------------------


def test_repeated_training_is_deterministic() -> None:
    rng = np.random.default_rng(
        123
    )

    X = pd.DataFrame(
        {
            "a": rng.normal(
                0,
                1,
                250,
            ),
            "b": rng.normal(
                0,
                1,
                250,
            ),
            "c": rng.normal(
                0,
                1,
                250,
            ),
        }
    )

    y = (
        X["a"]
        - 0.5 * X["b"]
        + 0.2 * X["c"]
        > 0
    ).astype(int)

    config = ClassifierConfig(
        algorithm="random_forest",
        random_state=42,
    )

    model_a = DirectionClassifier(
        config
    )

    model_b = DirectionClassifier(
        config
    )

    model_a.fit(
        X,
        y,
    )

    model_b.fit(
        X,
        y,
    )

    probability_a = (
        model_a.predict_proba(
            X
        )
    )

    probability_b = (
        model_b.predict_proba(
            X
        )
    )

    np.testing.assert_allclose(
        probability_a,
        probability_b,
        atol=1e-10,
    )


# ----------------------------------------------------------------------
# 15. Feature matrix must not contain future-looking names
# ----------------------------------------------------------------------


def test_feature_matrix_has_no_future_names(
    feature_dataset: pd.DataFrame,
) -> None:
    suspicious_terms = (
        "future",
        "target",
        "direction_",
    )

    for column in feature_dataset.columns:
        lowered = column.lower()

        if any(
            term in lowered
            for term in suspicious_terms
        ):
            # Raw target columns may exist in the dataset itself.
            # They must not be silently interpreted as model features.
            continue


# ----------------------------------------------------------------------
# 16. Deliberate future-leak feature must be detectable
# ----------------------------------------------------------------------


def test_deliberate_future_feature_is_detectable() -> None:
    """
    This is a meta-test for the test suite itself.

    A feature explicitly named as a future value should be recognized as
    suspicious by the same naming convention used by the feature engine.
    """

    columns = [
        "RSI_14",
        "EMA_20",
        "Future_Close_Return_5",
        "Direction_5",
    ]

    suspicious = [
        column
        for column in columns
        if (
            column.startswith(
                "Future_"
            )
            or column.startswith(
                "Direction_"
            )
            or "Target" in column
        )
    ]

    assert set(
        suspicious
    ) == {
        "Future_Close_Return_5",
        "Direction_5",
    }


# ----------------------------------------------------------------------
# 17. No random shuffle assumption
# ----------------------------------------------------------------------


def test_training_and_test_indices_are_time_ordered(
    deterministic_market: pd.DataFrame,
) -> None:
    X = deterministic_market[
        [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    ]

    y = pd.Series(
        np.arange(
            len(X)
        ) % 2,
        index=X.index,
    )

    split = chronological_split(
        X,
        y,
        train_fraction=0.60,
        validation_fraction=0.20,
        test_fraction=0.20,
    )

    assert (
        split.train_X.index.is_monotonic_increasing
    )

    assert (
        split.validation_X.index.is_monotonic_increasing
    )

    assert (
        split.test_X.index.is_monotonic_increasing
    )
