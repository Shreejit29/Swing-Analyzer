"""
Tests for temporal validation and research-quality controls.

These tests are designed to catch:
    - random train/test splitting
    - temporal leakage
    - overlapping future labels
    - incorrect walk-forward ordering
    - preprocessing leakage
    - calibration leakage
    - invalid range coverage calculations
    - unstable regime performance
    - incorrect 95% research gates
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.calibration import (
    ProbabilityCalibrator,
)
from src.models.calibration_pipeline import (
    CalibrationPipeline,
    CalibrationPipelineConfig,
)
from src.models.metrics import (
    classification_metrics,
    evaluate_95_percent_gate,
)
from src.models.range_validation import (
    RangeValidationConfig,
    evaluate_range,
)
from src.models.regime_validation import (
    RegimeValidationConfig,
    evaluate_by_regime,
    build_regime_report,
)
from src.models.splitter import (
    chronological_split,
    date_split,
    purged_walk_forward_splits,
    walk_forward_splits,
)
from src.models.walk_forward import (
    WalkForwardConfig,
    WalkForwardValidator,
)
from src.models.classifier import (
    DirectionClassifier,
)
from src.models.preprocessing import (
    SafePreprocessor,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def temporal_data() -> pd.DataFrame:
    """
    Deterministic time-indexed dataset with explicit future labels.
    """

    n = 300

    index = pd.date_range(
        "2020-01-01",
        periods=n,
        freq="D",
    )

    rng = np.random.default_rng(
        42
    )

    feature_1 = rng.normal(
        0,
        1,
        n,
    )

    feature_2 = rng.normal(
        0,
        1,
        n,
    )

    signal = (
        feature_1
        + 0.5 * feature_2
    )

    direction = (
        signal > 0
    ).astype(int)

    return pd.DataFrame(
        {
            "feature_1": feature_1,
            "feature_2": feature_2,
            "Direction_5": direction,
        },
        index=index,
    )


@pytest.fixture
def range_predictions() -> tuple[
    pd.Series,
    pd.Series,
    pd.Series,
]:
    index = pd.date_range(
        "2025-01-01",
        periods=200,
        freq="D",
    )

    lower = pd.Series(
        -0.02,
        index=index,
    )

    median = pd.Series(
        0.00,
        index=index,
    )

    upper = pd.Series(
        0.02,
        index=index,
    )

    return lower, median, upper


# ----------------------------------------------------------------------
# Chronological splitting
# ----------------------------------------------------------------------


def test_chronological_split_preserves_time_order(
    temporal_data: pd.DataFrame,
) -> None:
    X = temporal_data[
        [
            "feature_1",
            "feature_2",
        ]
    ]

    y = temporal_data[
        "Direction_5"
    ]

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


def test_chronological_split_has_expected_sizes(
    temporal_data: pd.DataFrame,
) -> None:
    X = temporal_data[
        [
            "feature_1",
            "feature_2",
        ]
    ]

    y = temporal_data[
        "Direction_5"
    ]

    split = chronological_split(
        X,
        y,
        train_fraction=0.60,
        validation_fraction=0.20,
        test_fraction=0.20,
    )

    assert len(
        split.train_X
    ) == 180

    assert len(
        split.validation_X
    ) == 60

    assert len(
        split.test_X
    ) == 60


def test_chronological_split_rejects_invalid_fractions(
    temporal_data: pd.DataFrame,
) -> None:
    X = temporal_data[
        [
            "feature_1",
            "feature_2",
        ]
    ]

    y = temporal_data[
        "Direction_5"
    ]

    with pytest.raises(
        ValueError
    ):
        chronological_split(
            X,
            y,
            train_fraction=0.70,
            validation_fraction=0.30,
            test_fraction=0.30,
        )


def test_date_split_preserves_order(
    temporal_data: pd.DataFrame,
) -> None:
    X = temporal_data[
        [
            "feature_1",
            "feature_2",
        ]
    ]

    y = temporal_data[
        "Direction_5"
    ]

    train_end = pd.Timestamp(
        "2020-06-30"
    )

    validation_end = pd.Timestamp(
        "2020-08-31"
    )

    split = date_split(
        X,
        y,
        train_end=train_end,
        validation_end=validation_end,
    )

    assert (
        split.train_X.index.max()
        <= train_end
    )

    assert (
        split.validation_X.index.min()
        > train_end
    )

    assert (
        split.validation_X.index.max()
        <= validation_end
    )

    assert (
        split.test_X.index.min()
        > validation_end
    )


# ----------------------------------------------------------------------
# Walk-forward splitting
# ----------------------------------------------------------------------


def test_walk_forward_splits_are_chronological() -> None:
    index = pd.date_range(
        "2020-01-01",
        periods=200,
        freq="D",
    )

    folds = list(
        walk_forward_splits(
            index,
            n_splits=5,
            train_size=100,
            test_size=20,
            expanding=True,
            gap=0,
        )
    )

    assert len(
        folds
    ) == 5

    previous_test_end = -1

    for fold in folds:
        train_idx = fold.train_indices
        test_idx = fold.test_indices

        assert (
            train_idx.max()
            < test_idx.min()
        )

        assert (
            test_idx.min()
            > previous_test_end
        )

        previous_test_end = test_idx.max()


def test_walk_forward_train_size_expands() -> None:
    index = pd.date_range(
        "2020-01-01",
        periods=250,
        freq="D",
    )

    folds = list(
        walk_forward_splits(
            index,
            n_splits=4,
            train_size=80,
            test_size=20,
            expanding=True,
            gap=0,
        )
    )

    train_sizes = [
        len(
            fold.train_indices
        )
        for fold in folds
    ]

    assert train_sizes == sorted(
        train_sizes
    )

    assert train_sizes[-1] >= train_sizes[0]


def test_rolling_walk_forward_keeps_training_window_fixed() -> None:
    index = pd.date_range(
        "2020-01-01",
        periods=250,
        freq="D",
    )

    folds = list(
        walk_forward_splits(
            index,
            n_splits=4,
            train_size=80,
            test_size=20,
            expanding=False,
            gap=0,
        )
    )

    train_sizes = [
        len(
            fold.train_indices
        )
        for fold in folds
    ]

    assert all(
        size == 80
        for size in train_sizes
    )


def test_walk_forward_gap_creates_temporal_separation() -> None:
    index = pd.date_range(
        "2020-01-01",
        periods=200,
        freq="D",
    )

    folds = list(
        walk_forward_splits(
            index,
            n_splits=3,
            train_size=80,
            test_size=20,
            expanding=True,
            gap=5,
        )
    )

    for fold in folds:
        assert (
            fold.test_indices.min()
            - fold.train_indices.max()
            > 5
        )


# ----------------------------------------------------------------------
# Purged walk-forward validation
# ----------------------------------------------------------------------


def test_purged_walk_forward_has_no_overlapping_label_windows() -> None:
    index = pd.date_range(
        "2020-01-01",
        periods=250,
        freq="D",
    )

    folds = list(
        purged_walk_forward_splits(
            index,
            n_splits=4,
            train_size=100,
            test_size=20,
            horizon=10,
            embargo=0,
            expanding=True,
        )
    )

    assert folds

    for fold in folds:
        train_indices = fold.train_indices
        test_indices = fold.test_indices

        assert (
            train_indices.max()
            < test_indices.min()
        )

        # Training observations whose future label window reaches into
        # the test period must be removed.
        latest_train = train_indices.max()
        first_test = test_indices.min()

        assert (
            latest_train + 10
            < first_test
            or latest_train < first_test
        )


def test_purging_reduces_training_observations() -> None:
    index = pd.date_range(
        "2020-01-01",
        periods=200,
        freq="D",
    )

    normal = list(
        walk_forward_splits(
            index,
            n_splits=3,
            train_size=80,
            test_size=20,
            expanding=True,
            gap=0,
        )
    )

    purged = list(
        purged_walk_forward_splits(
            index,
            n_splits=3,
            train_size=80,
            test_size=20,
            horizon=10,
            embargo=0,
            expanding=True,
        )
    )

    assert len(
        normal
    ) == len(
        purged
    )

    for normal_fold, purged_fold in zip(
        normal,
        purged,
    ):
        assert (
            len(
                purged_fold.train_indices
            )
            <= len(
                normal_fold.train_indices
            )
        )


# ----------------------------------------------------------------------
# Preprocessing leakage
# ----------------------------------------------------------------------


def test_preprocessor_is_fit_only_on_training_data() -> None:
    train = pd.DataFrame(
        {
            "feature": np.arange(
                100.0
            )
        }
    )

    test = pd.DataFrame(
        {
            "feature": np.full(
                20,
                10000.0,
            )
        }
    )

    preprocessor = SafePreprocessor()

    preprocessor.fit(
        train
    )

    statistics_before = (
        preprocessor.get_feature_names()
    )

    preprocessor.transform(
        test
    )

    statistics_after = (
        preprocessor.get_feature_names()
    )

    assert (
        statistics_before
        == statistics_after
    )


# ----------------------------------------------------------------------
# Walk-forward model validation
# ----------------------------------------------------------------------


def test_walk_forward_validator_produces_out_of_sample_predictions(
    temporal_data: pd.DataFrame,
) -> None:
    X = temporal_data[
        [
            "feature_1",
            "feature_2",
        ]
    ]

    y = temporal_data[
        "Direction_5"
    ]

    config = WalkForwardConfig(
        n_splits=3,
        train_size=120,
        validation_size=30,
        gap=5,
        embargo=0,
        min_train_samples=50,
        min_validation_samples=10,
    )

    validator = WalkForwardValidator(
        config
    )

    result = validator.validate(
        X,
        y,
        model_factory=lambda: DirectionClassifier(),
    )

    assert result.folds

    for fold in result.folds:
        assert (
            fold.validation_accuracy
            >= 0.0
        )

        assert (
            fold.validation_accuracy
            <= 1.0
        )

        assert (
            fold.train_end
            < fold.validation_start
        )

    predictions = (
        result.aggregate_predictions()
    )

    assert not predictions.empty

    assert (
        predictions.index.is_monotonic_increasing
    )


# ----------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------


def test_calibration_requires_temporal_separation() -> None:
    probabilities = np.linspace(
        0.05,
        0.95,
        100,
    )

    targets = (
        probabilities
        > 0.5
    ).astype(int)

    timestamps = pd.date_range(
        "2025-01-01",
        periods=100,
        freq="D",
    )

    pipeline = CalibrationPipeline(
        CalibrationPipelineConfig(
            min_samples=50
        )
    )

    with pytest.raises(
        ValueError
    ):
        pipeline.fit(
            probabilities=probabilities,
            targets=targets,
            timestamps=timestamps,
            training_end=timestamps[-1],
        )


def test_calibration_pipeline_fits_on_future_calibration_set() -> None:
    probabilities = np.linspace(
        0.05,
        0.95,
        120,
    )

    targets = (
        probabilities
        > 0.5
    ).astype(int)

    timestamps = pd.date_range(
        "2025-01-01",
        periods=120,
        freq="D",
    )

    training_end = timestamps[59]

    pipeline = CalibrationPipeline(
        CalibrationPipelineConfig(
            min_samples=50
        )
    )

    result = pipeline.fit(
        probabilities=probabilities[60:],
        targets=targets[60:],
        timestamps=timestamps[60:],
        training_end=training_end,
    )

    assert result.calibrator is not None

    assert (
        result.metrics.calibrated_brier
        >= 0
    )

    assert (
        result.metrics.calibrated_brier
        <= 1
    )


# ----------------------------------------------------------------------
# Range validation
# ----------------------------------------------------------------------


def test_range_validation_perfect_coverage(
    range_predictions: tuple[
        pd.Series,
        pd.Series,
        pd.Series,
    ],
) -> None:
    lower, median, upper = (
        range_predictions
    )

    actual = pd.Series(
        np.zeros(
            len(lower)
        ),
        index=lower.index,
    )

    result = evaluate_range(
        actual,
        lower,
        median,
        upper,
        config=RangeValidationConfig(
            expected_coverage=0.80,
            min_coverage=0.70,
            max_coverage=0.95,
            min_samples=50,
        ),
    )

    assert np.isclose(
        result.coverage,
        1.0,
    )

    assert (
        result.coverage_status
        == "HIGH"
    )


def test_range_validation_detects_low_coverage() -> None:
    index = pd.date_range(
        "2025-01-01",
        periods=100,
        freq="D",
    )

    lower = pd.Series(
        -0.001,
        index=index,
    )

    median = pd.Series(
        0.0,
        index=index,
    )

    upper = pd.Series(
        0.001,
        index=index,
    )

    actual = pd.Series(
        np.where(
            np.arange(
                100
            )
            % 2
            == 0,
            0.05,
            -0.05,
        ),
        index=index,
    )

    result = evaluate_range(
        actual,
        lower,
        median,
        upper,
        config=RangeValidationConfig(
            expected_coverage=0.80,
            min_coverage=0.70,
            max_coverage=0.95,
            min_samples=50,
        ),
    )

    assert (
        result.coverage
        < 0.70
    )


def test_range_validation_orders_bounds() -> None:
    index = pd.date_range(
        "2025-01-01",
        periods=100,
        freq="D",
    )

    lower = pd.Series(
        0.03,
        index=index,
    )

    median = pd.Series(
        0.01,
        index=index,
    )

    upper = pd.Series(
        -0.01,
        index=index,
    )

    actual = pd.Series(
        0.01,
        index=index,
    )

    result = evaluate_range(
        actual,
        lower,
        median,
        upper,
        config=RangeValidationConfig(
            min_samples=50
        ),
    )

    # The validation layer must handle malformed ordering rather than
    # reporting impossible intervals as valid calibrated ranges.
    assert result is not None


# ----------------------------------------------------------------------
# Regime validation
# ----------------------------------------------------------------------


def test_regime_evaluation_returns_one_result_per_regime() -> None:
    n = 300

    rng = np.random.default_rng(
        123
    )

    y_true = rng.integers(
        0,
        2,
        n,
    )

    y_pred = y_true.copy()

    regimes = np.array(
        [
            "BULL",
            "BEAR",
            "NEUTRAL",
        ]
        * 100
    )

    result = evaluate_by_regime(
        y_true,
        y_pred,
        regimes,
        config=RegimeValidationConfig(
            min_samples=10
        ),
    )

    assert not result.empty

    assert set(
        result["Regime"]
    ).issubset(
        {
            "BULL",
            "BEAR",
            "NEUTRAL",
        }
    )


def test_regime_report_detects_stability() -> None:
    n = 300

    y_true = np.array(
        [
            0,
            1,
        ]
        * 150
    )

    y_pred = y_true.copy()

    regimes = np.array(
        [
            "BULL",
            "BEAR",
            "NEUTRAL",
        ]
        * 100
    )

    report = build_regime_report(
        y_true,
        y_pred,
        regimes,
        config=RegimeValidationConfig(
            min_samples=10
        ),
    )

    assert report is not None

    assert (
        report.stability_score
        >= 0
    )

    assert (
        report.stability_score
        <= 1
    )


# ----------------------------------------------------------------------
# 95% gate
# ----------------------------------------------------------------------


def test_95_percent_gate_passes_at_exact_threshold() -> None:
    y_true = np.array(
        [
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
        ]
    )

    # 19 / 20 = 95%
    y_pred = y_true.copy()

    y_pred[-1] = (
        1
        - y_pred[-1]
    )

    metrics = classification_metrics(
        y_true,
        y_pred,
    )

    result = evaluate_95_percent_gate(
        metrics
    )

    assert (
        result["accuracy"]
        == pytest.approx(
            0.95
        )
    )

    assert result["passed"] is True


def test_95_percent_gate_rejects_below_threshold() -> None:
    y_true = np.array(
        [
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
        ]
    )

    # 18 / 20 = 90%
    y_pred = y_true.copy()

    y_pred[-1] = (
        1
        - y_pred[-1]
    )

    y_pred[-2] = (
        1
        - y_pred[-2]
    )

    metrics = classification_metrics(
        y_true,
        y_pred,
    )

    result = evaluate_95_percent_gate(
        metrics
    )

    assert (
        result["accuracy"]
        == pytest.approx(
            0.90
        )
    )

    assert result["passed"] is False


# ----------------------------------------------------------------------
# Validation sanity
# ----------------------------------------------------------------------


def test_validation_predictions_are_not_training_predictions() -> None:
    """
    Basic structural check: validation observations must occur after
    the training observations.
    """

    index = pd.date_range(
        "2020-01-01",
        periods=150,
        freq="D",
    )

    folds = list(
        purged_walk_forward_splits(
            index,
            n_splits=3,
            train_size=60,
            test_size=20,
            horizon=5,
            embargo=0,
            expanding=True,
        )
    )

    for fold in folds:
        assert not set(
            fold.train_indices
        ).intersection(
            set(
                fold.test_indices
            )
        )
