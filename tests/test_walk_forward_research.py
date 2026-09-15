"""
Tests for the walk-forward research engine.

These tests focus on temporal integrity and leakage prevention rather
than trying to prove that a synthetic model is profitable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.selection import FeatureSelectionConfig
from src.models.classifier import ClassifierConfig
from src.models.preprocessing import PreprocessorConfig
from src.research.config import ResearchPipelineConfig
from src.research.walk_forward_research import (
    WalkForwardResearchEngine,
    WalkForwardResearchResult,
    run_walk_forward_research,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(42)

    index = pd.date_range(
        "2018-01-01",
        periods=800,
        freq="D",
    )

    f1 = rng.normal(
        0,
        1,
        len(index),
    )

    f2 = rng.normal(
        0,
        1,
        len(index),
    )

    f3 = (
        0.4 * f1
        + rng.normal(
            0,
            0.6,
            len(index),
        )
    )

    signal = (
        f1
        + 0.3 * f2
        + rng.normal(
            0,
            0.3,
            len(index),
        )
    )

    target = (
        signal > 0
    ).astype(int)

    return pd.DataFrame(
        {
            "Open": 100 + f1,
            "High": 101 + f1,
            "Low": 99 + f1,
            "Close": 100 + f1,
            "Volume": 100_000 + f2 * 1000,
            "Feature_1": f1,
            "Feature_2": f2,
            "Feature_3": f3,
            "Direction_5": target,
        },
        index=index,
    )


@pytest.fixture
def features() -> list[str]:
    return [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Feature_1",
        "Feature_2",
        "Feature_3",
    ]


@pytest.fixture
def config() -> ResearchPipelineConfig:
    return ResearchPipelineConfig()


@pytest.fixture
def engine(
    config,
) -> WalkForwardResearchEngine:

    return WalkForwardResearchEngine(
        config=config,
        preprocessor_config=PreprocessorConfig(),
        classifier_config=ClassifierConfig(
            model_type="logistic_regression"
        ),
        feature_selection_config=FeatureSelectionConfig(
            minimum_features=1,
            maximum_correlation=0.99,
        ),
    )


# ---------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------


def test_engine_can_be_created():
    engine = (
        WalkForwardResearchEngine()
    )

    assert engine is not None


def test_invalid_dataframe_fails(
    engine,
):
    invalid = pd.DataFrame()

    with pytest.raises(
        ValueError
    ):
        engine.run(
            invalid,
            ["Close"],
            "Direction_5",
            5,
        )


def test_non_datetime_index_fails(
    engine,
    dataframe,
    features,
):
    invalid = dataframe.copy()

    invalid.index = np.arange(
        len(invalid)
    )

    with pytest.raises(
        TypeError
    ):
        engine.run(
            invalid,
            features,
            "Direction_5",
            5,
        )


def test_duplicate_timestamps_fail(
    engine,
    dataframe,
    features,
):
    invalid = pd.concat(
        [
            dataframe,
            dataframe.iloc[:1],
        ]
    )

    with pytest.raises(
        ValueError
    ):
        engine.run(
            invalid,
            features,
            "Direction_5",
            5,
        )


def test_unsorted_data_fails(
    engine,
    dataframe,
    features,
):
    invalid = dataframe.iloc[
        ::-1
    ]

    with pytest.raises(
        ValueError
    ):
        engine.run(
            invalid,
            features,
            "Direction_5",
            5,
        )


# ---------------------------------------------------------------------
# Feature and target validation
# ---------------------------------------------------------------------


def test_missing_feature_fails(
    engine,
    dataframe,
):
    with pytest.raises(
        ValueError
    ):
        engine.run(
            dataframe,
            [
                "Close",
                "MissingFeature",
            ],
            "Direction_5",
            5,
        )


def test_missing_target_fails(
    engine,
    dataframe,
    features,
):
    with pytest.raises(
        ValueError
    ):
        engine.run(
            dataframe,
            features,
            "MissingTarget",
            5,
        )


def test_constant_target_fails(
    engine,
    dataframe,
    features,
):
    invalid = dataframe.copy()

    invalid[
        "ConstantTarget"
    ] = 1

    with pytest.raises(
        ValueError
    ):
        engine.run(
            invalid,
            features,
            "ConstantTarget",
            5,
        )


def test_invalid_horizon_fails(
    engine,
    dataframe,
    features,
):
    with pytest.raises(
        ValueError
    ):
        engine.run(
            dataframe,
            features,
            "Direction_5",
            0,
        )


# ---------------------------------------------------------------------
# Walk-forward result
# ---------------------------------------------------------------------


def test_run_returns_result(
    engine,
    dataframe,
    features,
):
    result = engine.run(
        dataframe,
        features,
        "Direction_5",
        horizon=5,
        model_name="logistic_regression",
    )

    assert isinstance(
        result,
        WalkForwardResearchResult,
    )


def test_folds_are_created(
    engine,
    dataframe,
    features,
):
    result = engine.run(
        dataframe,
        features,
        "Direction_5",
        horizon=5,
    )

    assert result.fold_count > 0


def test_result_has_exact_oos_predictions(
    engine,
    dataframe,
    features,
):
    result = engine.run(
        dataframe,
        features,
        "Direction_5",
        horizon=5,
    )

    predictions = (
        result.out_of_sample_predictions
    )

    assert not predictions.empty

    assert predictions.index.is_monotonic_increasing

    assert not predictions.index.has_duplicates

    assert set(
        [
            "Prediction",
            "Actual",
            "Fold",
        ]
    ).issubset(
        predictions.columns
    )


# ---------------------------------------------------------------------
# Temporal integrity
# ---------------------------------------------------------------------


def test_each_fold_training_precedes_validation(
    engine,
    dataframe,
    features,
):
    result = engine.run(
        dataframe,
        features,
        "Direction_5",
        horizon=5,
    )

    for fold in result.folds:
        assert (
            fold.train_end
            < fold.validation_start
        )


def test_each_fold_has_no_overlap(
    engine,
    dataframe,
    features,
):
    result = engine.run(
        dataframe,
        features,
        "Direction_5",
        horizon=5,
    )

    for fold in result.folds:
        assert set(
            fold.validation_index
        ).isdisjoint(
            set(
                pd.date_range(
                    fold.train_start,
                    fold.train_end,
                    freq="D",
                )
            )
        )


def test_validation_windows_do_not_overlap(
    engine,
    dataframe,
    features,
):
    result = engine.run(
        dataframe,
        features,
        "Direction_5",
        horizon=5,
    )

    all_validation_indices = []

    for fold in result.folds:
        all_validation_indices.extend(
            list(
                fold.validation_index
            )
        )

    assert len(
        all_validation_indices
    ) == len(
        set(
            all_validation_indices
        )
    )


def test_validation_windows_move_forward(
    engine,
    dataframe,
    features,
):
    result = engine.run(
        dataframe,
        features,
        "Direction_5",
        horizon=5,
    )

    starts = [
        fold.validation_start
        for fold in result.folds
    ]

    assert starts == sorted(
        starts
    )


# ---------------------------------------------------------------------
# Holdout isolation
# ---------------------------------------------------------------------


def test_final_holdout_is_not_used(
    engine,
    dataframe,
    features,
):
    result = engine.run(
        dataframe,
        features,
        "Direction_5",
        horizon=5,
    )

    assert (
        result.final_holdout_used
        is False
    )

    assert (
        result.metadata[
            "final_holdout_reserved"
        ]
        is True
    )


def test_oos_predictions_end_before_final_holdout(
    engine,
    dataframe,
    features,
):
    result = engine.run(
        dataframe,
        features,
        "Direction_5",
        horizon=5,
    )

    holdout = (
        engine.config.validation
        if engine.config is not None
        else None
    )

    assert (
        result.out_of_sample_predictions.index.max()
        < dataframe.index[-1]
    )


# ---------------------------------------------------------------------
# Fresh fitting guarantees
# ---------------------------------------------------------------------


def test_each_fold_uses_fresh_model(
    engine,
    dataframe,
    features,
):
    result = engine.run(
        dataframe,
        features,
        "Direction_5",
        horizon=5,
    )

    for fold in result.folds:
        assert (
            fold.metadata[
                "fresh_model"
            ]
            is True
        )


def test_each_fold_uses_fresh_preprocessor(
    engine,
    dataframe,
    features,
):
    result = engine.run(
        dataframe,
        features,
        "Direction_5",
        horizon=5,
    )

    for fold in result.folds:
        assert (
            fold.metadata[
                "fresh_preprocessor"
            ]
            is True
        )


def test_feature_selection_is_training_only(
    engine,
    dataframe,
    features,
):
    result = engine.run(
        dataframe,
        features,
        "Direction_5",
        horizon=5,
    )

    for fold in result.folds:
        assert (
            fold.metadata[
                "feature_selection_training_only"
            ]
            is True
        )


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------


def test_accuracy_is_in_valid_range(
    engine,
    dataframe,
    features,
):
    result = engine.run(
        dataframe,
        features,
        "Direction_5",
        horizon=5,
    )

    assert 0.0 <= (
        result.mean_accuracy
    ) <= 1.0

    assert 0.0 <= (
        result.minimum_accuracy
    ) <= 1.0


def test_accuracy_std_is_non_negative(
    engine,
    dataframe,
    features,
):
    result = engine.run(
        dataframe,
        features,
        "Direction_5",
        horizon=5,
    )

    assert (
        result.accuracy_std
        >= 0.0
    )


def test_fold_accuracy_matches_fold_metrics(
    engine,
    dataframe,
    features,
):
    result = engine.run(
        dataframe,
        features,
        "Direction_5",
        horizon=5,
    )

    for fold in result.folds:
        assert (
            fold.accuracy
            == fold.validation_metrics.accuracy
        )


# ---------------------------------------------------------------------
# 95% gate
# ---------------------------------------------------------------------


def test_95_percent_gate_is_strict(
    engine,
    dataframe,
    features,
):
    result = engine.run(
        dataframe,
        features,
        "Direction_5",
        horizon=5,
    )

    expected = (
        result.mean_accuracy >= 0.95
        and result.minimum_accuracy >= 0.95
    )

    assert (
        result.passed_95_percent_gate
        == expected
    )


def test_candidate_requires_stability(
    engine,
    dataframe,
    features,
):
    result = engine.run(
        dataframe,
        features,
        "Direction_5",
        horizon=5,
    )

    expected = (
        result.passed_95_percent_gate
        and result.accuracy_std <= 0.10
    )

    assert (
        result.research_candidate
        == expected
    )


# ---------------------------------------------------------------------
# Mutation attack
# ---------------------------------------------------------------------


def test_future_holdout_mutation_does_not_change_oos_predictions(
    engine,
    dataframe,
    features,
):
    """
    Modify only the final holdout.

    Walk-forward predictions must remain identical because the holdout
    is never part of candidate development.
    """

    original = dataframe.copy()

    first = engine.run(
        original,
        features,
        "Direction_5",
        horizon=5,
    )

    mutated = original.copy()

    # Identify final holdout using the same official splitter.
    from src.research.temporal_split import (
        TemporalSplitter,
    )

    splitter = TemporalSplitter(
        engine.config.validation
    )

    holdout = splitter.holdout_split(
        mutated
    )

    mutated.loc[
        holdout.holdout_index,
        features,
    ] = 999_999.0

    second = engine.run(
        mutated,
        features,
        "Direction_5",
        horizon=5,
    )

    pd.testing.assert_frame_equal(
        first.out_of_sample_predictions,
        second.out_of_sample_predictions,
    )


# ---------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------


def test_walk_forward_run_is_deterministic(
    engine,
    dataframe,
    features,
):
    first = engine.run(
        dataframe,
        features,
        "Direction_5",
        horizon=5,
    )

    second = engine.run(
        dataframe,
        features,
        "Direction_5",
        horizon=5,
    )

    assert (
        first.mean_accuracy
        == second.mean_accuracy
    )

    assert (
        first.minimum_accuracy
        == second.minimum_accuracy
    )

    pd.testing.assert_frame_equal(
        first.out_of_sample_predictions,
        second.out_of_sample_predictions,
    )


# ---------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------


def test_convenience_function(
    dataframe,
    features,
):
    result = run_walk_forward_research(
        data=dataframe,
        feature_columns=features,
        target_column="Direction_5",
        horizon=5,
        model_name="logistic_regression",
    )

    assert isinstance(
        result,
        WalkForwardResearchResult,
    )


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------


def test_summary_contains_research_safety_fields(
    engine,
    dataframe,
    features,
):
    result = engine.run(
        dataframe,
        features,
        "Direction_5",
        horizon=5,
    )

    summary = result.summary()

    assert (
        "mean_accuracy"
        in summary
    )

    assert (
        "minimum_accuracy"
        in summary
    )

    assert (
        "accuracy_std"
        in summary
    )

    assert (
        "final_holdout_used"
        in summary
    )

    assert (
        summary[
            "final_holdout_used"
        ]
        is False
    )


# ---------------------------------------------------------------------
# Research-only safety
# ---------------------------------------------------------------------


def test_result_is_research_only(
    engine,
    dataframe,
    features,
):
    result = engine.run(
        dataframe,
        features,
        "Direction_5",
        horizon=5,
    )

    assert (
        result.metadata[
            "research_only"
        ]
        is True
    )
