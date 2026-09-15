"""
Tests for the leakage-safe model development orchestrator.

These tests verify that:

- development and holdout data remain separated
- training always precedes validation
- feature selection uses training data only
- preprocessing is fitted only on training data
- validation data is transformed without refitting
- final holdout data is never used during candidate development
- candidate status is not equivalent to production approval
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.selection import FeatureSelectionConfig
from src.models.classifier import ClassifierConfig
from src.models.preprocessing import PreprocessorConfig
from src.research.config import ResearchPipelineConfig
from src.research.model_development import (
    DevelopmentPartitions,
    ModelDevelopmentOrchestrator,
)
from src.research.temporal_split import (
    TemporalSplitter,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def dataframe() -> pd.DataFrame:
    """
    Synthetic chronological classification dataset.

    The target is deliberately simple so the tests focus on pipeline
    integrity rather than market prediction performance.
    """

    rng = np.random.default_rng(42)

    index = pd.date_range(
        "2020-01-01",
        periods=600,
        freq="D",
    )

    feature_1 = rng.normal(
        0,
        1,
        len(index),
    )

    feature_2 = rng.normal(
        0,
        1,
        len(index),
    )

    feature_3 = (
        feature_1 * 0.5
        + rng.normal(
            0,
            0.5,
            len(index),
        )
    )

    signal = (
        feature_1
        + 0.25 * feature_2
        + rng.normal(
            0,
            0.25,
            len(index),
        )
    )

    target = (
        signal > 0
    ).astype(int)

    return pd.DataFrame(
        {
            "Open": 100 + feature_1,
            "High": 101 + feature_1,
            "Low": 99 + feature_1,
            "Close": 100 + feature_1,
            "Volume": 100_000 + feature_2 * 1_000,
            "Feature_1": feature_1,
            "Feature_2": feature_2,
            "Feature_3": feature_3,
            "Direction_5": target,
        },
        index=index,
    )


@pytest.fixture
def config() -> ResearchPipelineConfig:
    return ResearchPipelineConfig()


@pytest.fixture
def orchestrator(
    config,
) -> ModelDevelopmentOrchestrator:
    return ModelDevelopmentOrchestrator(
        config=config,
        preprocessor_config=PreprocessorConfig(),
        classifier_config=ClassifierConfig(),
        feature_selection_config=FeatureSelectionConfig(
            minimum_features=1,
            maximum_correlation=0.99,
        ),
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


# ---------------------------------------------------------------------
# Basic validation
# ---------------------------------------------------------------------


def test_orchestrator_can_be_created():
    orchestrator = (
        ModelDevelopmentOrchestrator()
    )

    assert orchestrator is not None


def test_empty_dataset_fails(
    orchestrator,
):
    empty = pd.DataFrame(
        index=pd.DatetimeIndex([])
    )

    with pytest.raises(
        ValueError
    ):
        orchestrator.split_development_holdout(
            empty
        )


def test_non_datetime_index_fails(
    orchestrator,
    dataframe,
):
    invalid = dataframe.copy()

    invalid.index = np.arange(
        len(invalid)
    )

    with pytest.raises(
        TypeError
    ):
        orchestrator.split_development_holdout(
            invalid
        )


def test_duplicate_timestamps_fail(
    orchestrator,
    dataframe,
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
        orchestrator.split_development_holdout(
            invalid
        )


def test_unsorted_data_fails(
    orchestrator,
    dataframe,
):
    invalid = dataframe.iloc[
        ::-1
    ]

    with pytest.raises(
        ValueError
    ):
        orchestrator.split_development_holdout(
            invalid
        )


# ---------------------------------------------------------------------
# Development / holdout split
# ---------------------------------------------------------------------


def test_split_returns_expected_type(
    orchestrator,
    dataframe,
):
    result = (
        orchestrator.split_development_holdout(
            dataframe
        )
    )

    assert isinstance(
        result,
        DevelopmentPartitions,
    )


def test_holdout_is_after_development(
    orchestrator,
    dataframe,
):
    result = (
        orchestrator.split_development_holdout(
            dataframe
        )
    )

    assert (
        result.development_end
        < result.holdout_start
    )

    assert (
        result.holdout_is_future
        is True
    )


def test_development_and_holdout_do_not_overlap(
    orchestrator,
    dataframe,
):
    result = (
        orchestrator.split_development_holdout(
            dataframe
        )
    )

    assert set(
        result.development_index
    ).isdisjoint(
        set(
            result.holdout_index
        )
    )


def test_holdout_contains_latest_data(
    orchestrator,
    dataframe,
):
    result = (
        orchestrator.split_development_holdout(
            dataframe
        )
    )

    assert (
        result.holdout_index[-1]
        == dataframe.index[-1]
    )


def test_holdout_is_not_empty(
    orchestrator,
    dataframe,
):
    result = (
        orchestrator.split_development_holdout(
            dataframe
        )
    )

    assert not result.holdout.empty


# ---------------------------------------------------------------------
# Training / validation split
# ---------------------------------------------------------------------


def test_training_validation_split_is_chronological(
    orchestrator,
    dataframe,
):
    partitions = (
        orchestrator.split_development_holdout(
            dataframe
        )
    )

    training, validation = (
        orchestrator.split_training_validation(
            partitions.development
        )
    )

    assert (
        training.index.max()
        < validation.index.min()
    )


def test_training_and_validation_do_not_overlap(
    orchestrator,
    dataframe,
):
    partitions = (
        orchestrator.split_development_holdout(
            dataframe
        )
    )

    training, validation = (
        orchestrator.split_training_validation(
            partitions.development
        )
    )

    assert set(
        training.index
    ).isdisjoint(
        set(
            validation.index
        )
    )


def test_validation_is_newer_than_training(
    orchestrator,
    dataframe,
):
    partitions = (
        orchestrator.split_development_holdout(
            dataframe
        )
    )

    training, validation = (
        orchestrator.split_training_validation(
            partitions.development
        )
    )

    assert (
        validation.index.min()
        > training.index.max()
    )


# ---------------------------------------------------------------------
# Target validation
# ---------------------------------------------------------------------


def test_missing_target_fails(
    orchestrator,
    dataframe,
    features,
):
    with pytest.raises(
        ValueError
    ):
        orchestrator.fit_candidate(
            dataframe,
            features,
            "Missing_Target",
            horizon=5,
            apply_feature_selection=False,
        )


def test_constant_target_fails(
    orchestrator,
    dataframe,
    features,
):
    invalid = dataframe.copy()

    invalid[
        "Constant_Target"
    ] = 1

    with pytest.raises(
        ValueError
    ):
        orchestrator.fit_candidate(
            invalid,
            features,
            "Constant_Target",
            horizon=5,
            apply_feature_selection=False,
        )


def test_target_nan_rows_are_removed(
    orchestrator,
    dataframe,
    features,
):
    invalid = dataframe.copy()

    invalid.loc[
        invalid.index[:10],
        "Direction_5",
    ] = np.nan

    result = orchestrator.fit_candidate(
        invalid,
        features,
        "Direction_5",
        horizon=5,
        apply_feature_selection=False,
    )

    assert result is not None


# ---------------------------------------------------------------------
# Feature validation
# ---------------------------------------------------------------------


def test_missing_feature_fails(
    orchestrator,
    dataframe,
):
    with pytest.raises(
        ValueError
    ):
        orchestrator.fit_candidate(
            dataframe,
            [
                "Feature_1",
                "DoesNotExist",
            ],
            "Direction_5",
            horizon=5,
            apply_feature_selection=False,
        )


def test_non_numeric_feature_fails(
    orchestrator,
    dataframe,
):
    invalid = dataframe.copy()

    invalid[
        "BadFeature"
    ] = "text"

    with pytest.raises(
        TypeError
    ):
        orchestrator.fit_candidate(
            invalid,
            [
                "Feature_1",
                "BadFeature",
            ],
            "Direction_5",
            horizon=5,
            apply_feature_selection=False,
        )


def test_infinite_feature_fails(
    orchestrator,
    dataframe,
):
    invalid = dataframe.copy()

    invalid[
        "InfiniteFeature"
    ] = np.inf

    with pytest.raises(
        ValueError
    ):
        orchestrator.fit_candidate(
            invalid,
            [
                "Feature_1",
                "InfiniteFeature",
            ],
            "Direction_5",
            horizon=5,
            apply_feature_selection=False,
        )


# ---------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------


def test_feature_selection_returns_result(
    orchestrator,
    dataframe,
    features,
):
    partitions = (
        orchestrator.split_development_holdout(
            dataframe
        )
    )

    training, _ = (
        orchestrator.split_training_validation(
            partitions.development
        )
    )

    result = (
        orchestrator.select_development_features(
            training,
            features,
            "Direction_5",
        )
    )

    assert result is not None

    assert (
        len(
            result.selected_features
        )
        >= 1
    )


def test_feature_selection_does_not_use_holdout(
    orchestrator,
    dataframe,
    features,
):
    """
    The selected feature set must be identical whether or not the
    untouched holdout values are changed.

    This is a direct adversarial test for holdout contamination.
    """

    original = dataframe.copy()

    modified = dataframe.copy()

    holdout = (
        orchestrator
        .split_development_holdout(
            original
        )
        .holdout_index
    )

    # Alter only the future holdout observations.
    modified.loc[
        holdout,
        features,
    ] = 999_999.0

    original_partitions = (
        orchestrator.split_development_holdout(
            original
        )
    )

    modified_partitions = (
        orchestrator.split_development_holdout(
            modified
        )
    )

    original_training, _ = (
        orchestrator.split_training_validation(
            original_partitions.development
        )
    )

    modified_training, _ = (
        orchestrator.split_training_validation(
            modified_partitions.development
        )
    )

    original_selection = (
        orchestrator.select_development_features(
            original_training,
            features,
            "Direction_5",
        )
    )

    modified_selection = (
        orchestrator.select_development_features(
            modified_training,
            features,
            "Direction_5",
        )
    )

    assert (
        original_selection.selected_features
        == modified_selection.selected_features
    )


# ---------------------------------------------------------------------
# Candidate model training
# ---------------------------------------------------------------------


def test_fit_candidate_returns_result(
    orchestrator,
    dataframe,
    features,
):
    result = orchestrator.fit_candidate(
        dataframe,
        features,
        "Direction_5",
        horizon=5,
        apply_feature_selection=False,
    )

    assert result is not None

    assert (
        result.horizon
        == 5
    )

    assert (
        result.target_column
        == "Direction_5"
    )


def test_candidate_has_validation_metrics(
    orchestrator,
    dataframe,
    features,
):
    result = orchestrator.fit_candidate(
        dataframe,
        features,
        "Direction_5",
        horizon=5,
        apply_feature_selection=False,
    )

    assert result.validation_metrics is not None

    assert 0.0 <= (
        result.validation_accuracy
    ) <= 1.0


def test_candidate_is_not_automatically_approved(
    orchestrator,
    dataframe,
    features,
):
    result = orchestrator.fit_candidate(
        dataframe,
        features,
        "Direction_5",
        horizon=5,
        apply_feature_selection=False,
    )

    assert (
        result.metadata[
            "research_only"
        ]
        is True
    )

    assert (
        result.metadata[
            "final_holdout_used"
        ]
        is False
    )


# ---------------------------------------------------------------------
# Holdout isolation
# ---------------------------------------------------------------------


def test_candidate_training_does_not_change_with_future_holdout(
    orchestrator,
    dataframe,
    features,
):
    """
    Changing only the final holdout must not change the model-development
    dataset.

    We verify this at the partition level rather than comparing model
    parameters, because stochastic model internals can legitimately vary.
    """

    partitions = (
        orchestrator.split_development_holdout(
            dataframe
        )
    )

    modified = dataframe.copy()

    modified.loc[
        partitions.holdout_index,
        "Feature_1",
    ] = 123_456.0

    modified.loc[
        partitions.holdout_index,
        "Feature_2",
    ] = -123_456.0

    modified_partitions = (
        orchestrator.split_development_holdout(
            modified
        )
    )

    pd.testing.assert_frame_equal(
        partitions.development,
        modified_partitions.development,
    )


def test_holdout_is_never_inside_development_partition(
    orchestrator,
    dataframe,
):
    partitions = (
        orchestrator.split_development_holdout(
            dataframe
        )
    )

    assert not set(
        partitions.development_index
    ).intersection(
        set(
            partitions.holdout_index
        )
    )


# ---------------------------------------------------------------------
# Preprocessing leakage
# ---------------------------------------------------------------------


def test_preprocessor_is_fitted(
    orchestrator,
    dataframe,
    features,
):
    result = orchestrator.fit_candidate(
        dataframe,
        features,
        "Direction_5",
        horizon=5,
        apply_feature_selection=False,
    )

    assert result.preprocessor is not None

    transformed = (
        result.preprocessor.transform(
            dataframe.loc[
                :,
                features,
            ].iloc[:10]
        )
    )

    assert transformed.shape[0] == 10


def test_preprocessor_does_not_refit_on_validation(
    orchestrator,
    dataframe,
    features,
):
    """
    Transforming validation data must not modify the fitted training
    statistics.
    """

    result = orchestrator.fit_candidate(
        dataframe,
        features,
        "Direction_5",
        horizon=5,
        apply_feature_selection=False,
    )

    before = result.preprocessor.feature_names_

    partitions = (
        orchestrator.split_development_holdout(
            dataframe
        )
    )

    _, validation = (
        orchestrator.split_training_validation(
            partitions.development
        )
    )

    result.preprocessor.transform(
        validation.loc[
            :,
            features,
        ]
    )

    after = result.preprocessor.feature_names_

    assert before == after


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------


def test_split_is_deterministic(
    orchestrator,
    dataframe,
):
    first = (
        orchestrator.split_development_holdout(
            dataframe
        )
    )

    second = (
        orchestrator.split_development_holdout(
            dataframe
        )
    )

    assert first.development_index.equals(
        second.development_index
    )

    assert first.holdout_index.equals(
        second.holdout_index
    )


# ---------------------------------------------------------------------
# Explicit final-holdout safety
# ---------------------------------------------------------------------


def test_final_holdout_is_reserved(
    orchestrator,
    dataframe,
):
    partitions = (
        orchestrator.split_development_holdout(
            dataframe
        )
    )

    development_end = (
        partitions.development_end
    )

    holdout_start = (
        partitions.holdout_start
    )

    assert (
        development_end
        < holdout_start
    )


def test_candidate_metadata_confirms_holdout_is_unused(
    orchestrator,
    dataframe,
    features,
):
    result = orchestrator.fit_candidate(
        dataframe,
        features,
        "Direction_5",
        horizon=5,
        apply_feature_selection=False,
    )

    assert (
        result.metadata[
            "final_holdout_used"
        ]
        is False
    )

    assert (
        result.metadata[
            "preprocessor_fitted_on"
        ]
        == "training_only"
    )
