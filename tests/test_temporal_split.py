"""
Tests for leakage-safe temporal splitting.

These tests verify that:

- data remains chronological
- final holdout is strictly later than development data
- walk-forward validation never trains on validation data
- folds move forward through time
- gap and embargo prevent adjacent contamination
- no random shuffling is introduced
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.config import ResearchValidationConfig
from src.research.temporal_split import (
    HoldoutSplit,
    TemporalSplit,
    TemporalSplitter,
    create_holdout_split,
    create_walk_forward_splits,
)


@pytest.fixture
def dataframe() -> pd.DataFrame:
    index = pd.date_range(
        "2020-01-01",
        periods=500,
        freq="D",
    )

    close = np.arange(
        100.0,
        600.0,
    )

    return pd.DataFrame(
        {
            "Open": close - 1.0,
            "High": close + 2.0,
            "Low": close - 2.0,
            "Close": close,
            "Volume": 100_000.0,
        },
        index=index,
    )


def make_config(
    **overrides,
) -> ResearchValidationConfig:
    """
    Build a validation configuration while keeping all defaults intact.
    """

    values = {
        "test_fraction": 0.20,
        "validation_fraction": 0.20,
        "minimum_accuracy": 0.95,
        "require_walk_forward": True,
        "require_out_of_sample": True,
        "require_no_data_leakage": True,
        "require_final_holdout": True,
    }

    values.update(overrides)

    return ResearchValidationConfig(
        **values
    )


# ---------------------------------------------------------------------
# Basic validation
# ---------------------------------------------------------------------


def test_splitter_accepts_valid_dataframe(
    dataframe,
):
    config = make_config()

    splitter = TemporalSplitter(
        config
    )

    result = splitter.holdout_split(
        dataframe
    )

    assert isinstance(
        result,
        HoldoutSplit,
    )


def test_empty_dataframe_fails():
    dataframe = pd.DataFrame(
        index=pd.DatetimeIndex([])
    )

    splitter = TemporalSplitter(
        make_config()
    )

    with pytest.raises(
        ValueError
    ):
        splitter.holdout_split(
            dataframe
        )


def test_non_datetime_index_fails(
    dataframe,
):
    invalid = dataframe.copy()

    invalid.index = np.arange(
        len(invalid)
    )

    splitter = TemporalSplitter(
        make_config()
    )

    with pytest.raises(
        TypeError
    ):
        splitter.holdout_split(
            invalid
        )


def test_duplicate_timestamps_fail(
    dataframe,
):
    invalid = pd.concat(
        [
            dataframe,
            dataframe.iloc[:1],
        ]
    )

    splitter = TemporalSplitter(
        make_config()
    )

    with pytest.raises(
        ValueError
    ):
        splitter.holdout_split(
            invalid
        )


def test_unsorted_timestamps_fail(
    dataframe,
):
    invalid = dataframe.iloc[
        ::-1
    ]

    splitter = TemporalSplitter(
        make_config()
    )

    with pytest.raises(
        ValueError
    ):
        splitter.holdout_split(
            invalid
        )


# ---------------------------------------------------------------------
# Final holdout
# ---------------------------------------------------------------------


def test_holdout_is_strictly_after_development(
    dataframe,
):
    splitter = TemporalSplitter(
        make_config()
    )

    result = splitter.holdout_split(
        dataframe
    )

    assert (
        result.development_end
        < result.holdout_start
    )


def test_holdout_contains_newest_observation(
    dataframe,
):
    splitter = TemporalSplitter(
        make_config()
    )

    result = splitter.holdout_split(
        dataframe
    )

    assert (
        result.holdout_index[-1]
        == dataframe.index[-1]
    )


def test_development_and_holdout_do_not_overlap(
    dataframe,
):
    splitter = TemporalSplitter(
        make_config()
    )

    result = splitter.holdout_split(
        dataframe
    )

    assert (
        set(result.development_index)
        .isdisjoint(
            set(result.holdout_index)
        )
    )


def test_holdout_size_matches_config(
    dataframe,
):
    config = make_config()

    splitter = TemporalSplitter(
        config
    )

    result = splitter.holdout_split(
        dataframe
    )

    expected = int(
        len(dataframe)
        * config.test_fraction
    )

    assert len(
        result.holdout_index
    ) == expected


# ---------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------


def test_walk_forward_returns_temporal_splits(
    dataframe,
):
    config = make_config()

    splitter = TemporalSplitter(
        config
    )

    folds = list(
        splitter.walk_forward(
            dataframe
        )
    )

    assert folds

    assert all(
        isinstance(
            fold,
            TemporalSplit,
        )
        for fold in folds
    )


def test_walk_forward_training_precedes_validation(
    dataframe,
):
    config = make_config()

    splitter = TemporalSplitter(
        config
    )

    folds = list(
        splitter.walk_forward(
            dataframe
        )
    )

    for fold in folds:
        assert (
            fold.train_end
            < fold.validation_start
        )


def test_walk_forward_has_no_index_overlap(
    dataframe,
):
    config = make_config()

    splitter = TemporalSplitter(
        config
    )

    folds = list(
        splitter.walk_forward(
            dataframe
        )
    )

    for fold in folds:
        assert (
            set(fold.train_index)
            .isdisjoint(
                set(
                    fold.validation_index
                )
            )
        )


def test_walk_forward_validation_is_chronological(
    dataframe,
):
    config = make_config()

    splitter = TemporalSplitter(
        config
    )

    folds = list(
        splitter.walk_forward(
            dataframe
        )
    )

    validation_starts = [
        fold.validation_start
        for fold in folds
    ]

    assert validation_starts == sorted(
        validation_starts
    )


def test_expanding_training_window_grows(
    dataframe,
):
    config = make_config(
        expanding=True
    )

    splitter = TemporalSplitter(
        config
    )

    folds = list(
        splitter.walk_forward(
            dataframe
        )
    )

    if len(folds) > 1:
        train_sizes = [
            len(fold.train_index)
            for fold in folds
        ]

        assert train_sizes == sorted(
            train_sizes
        )


# ---------------------------------------------------------------------
# Gap / embargo
# ---------------------------------------------------------------------


def test_gap_is_respected(
    dataframe,
):
    config = make_config(
        gap=5,
        embargo=0,
    )

    splitter = TemporalSplitter(
        config
    )

    folds = list(
        splitter.walk_forward(
            dataframe
        )
    )

    for fold in folds:
        position_train_end = (
            dataframe.index.get_loc(
                fold.train_end
            )
        )

        position_validation_start = (
            dataframe.index.get_loc(
                fold.validation_start
            )
        )

        separation = (
            position_validation_start
            - position_train_end
            - 1
        )

        assert separation >= 5


def test_embargo_is_respected(
    dataframe,
):
    config = make_config(
        gap=0,
        embargo=5,
    )

    splitter = TemporalSplitter(
        config
    )

    folds = list(
        splitter.walk_forward(
            dataframe
        )
    )

    for fold in folds:
        position_train_end = (
            dataframe.index.get_loc(
                fold.train_end
            )
        )

        position_validation_start = (
            dataframe.index.get_loc(
                fold.validation_start
            )
        )

        separation = (
            position_validation_start
            - position_train_end
            - 1
        )

        assert separation >= 5


# ---------------------------------------------------------------------
# Leakage protection
# ---------------------------------------------------------------------


def test_training_never_contains_validation_timestamps(
    dataframe,
):
    splitter = TemporalSplitter(
        make_config()
    )

    folds = list(
        splitter.walk_forward(
            dataframe
        )
    )

    for fold in folds:
        assert not set(
            fold.train_index
        ).intersection(
            set(
                fold.validation_index
            )
        )


def test_future_observations_never_enter_training(
    dataframe,
):
    splitter = TemporalSplitter(
        make_config()
    )

    folds = list(
        splitter.walk_forward(
            dataframe
        )
    )

    for fold in folds:
        assert (
            max(fold.train_index)
            < min(fold.validation_index)
        )


def test_validation_windows_move_forward(
    dataframe,
):
    splitter = TemporalSplitter(
        make_config()
    )

    folds = list(
        splitter.walk_forward(
            dataframe
        )
    )

    for previous, current in zip(
        folds,
        folds[1:],
    ):
        assert (
            current.validation_start
            > previous.validation_start
        )


# ---------------------------------------------------------------------
# Convenience APIs
# ---------------------------------------------------------------------


def test_create_holdout_split(
    dataframe,
):
    result = create_holdout_split(
        dataframe,
        make_config(),
    )

    assert isinstance(
        result,
        HoldoutSplit,
    )


def test_create_walk_forward_splits(
    dataframe,
):
    result = create_walk_forward_splits(
        dataframe,
        make_config(),
    )

    assert isinstance(
        result,
        list,
    )

    assert all(
        isinstance(
            fold,
            TemporalSplit,
        )
        for fold in result
    )


# ---------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------


def test_split_is_deterministic(
    dataframe,
):
    config = make_config()

    first = create_walk_forward_splits(
        dataframe,
        config,
    )

    second = create_walk_forward_splits(
        dataframe,
        config,
    )

    assert len(first) == len(second)

    for fold_a, fold_b in zip(
        first,
        second,
    ):
        assert fold_a.train_index.equals(
            fold_b.train_index
        )

        assert fold_a.validation_index.equals(
            fold_b.validation_index
        )


# ---------------------------------------------------------------------
# No shuffling
# ---------------------------------------------------------------------


def test_training_index_remains_chronological(
    dataframe,
):
    folds = create_walk_forward_splits(
        dataframe,
        make_config(),
    )

    for fold in folds:
        assert fold.train_index.is_monotonic_increasing
        assert fold.validation_index.is_monotonic_increasing
