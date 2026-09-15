"""
Leakage-safe time-series splitting utilities.

This module provides:

    1. Chronological train/validation/test splitting
    2. Walk-forward splits
    3. Purged splits
    4. Embargo periods
    5. Validation of temporal ordering

The goal is to prevent future observations from influencing
training observations.

This module does NOT randomly shuffle market data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    """
    A single chronological train/validation/test split.
    """

    train_index: np.ndarray
    validation_index: np.ndarray
    test_index: np.ndarray

    @property
    def train_size(self) -> int:
        return len(self.train_index)

    @property
    def validation_size(self) -> int:
        return len(self.validation_index)

    @property
    def test_size(self) -> int:
        return len(self.test_index)


@dataclass(frozen=True)
class WalkForwardSplit:
    """
    A single walk-forward training/test split.
    """

    train_index: np.ndarray
    test_index: np.ndarray

    train_start: int
    train_end: int
    test_start: int
    test_end: int


def _validate_index(
    index: pd.Index,
) -> None:
    """Validate an index used for splitting."""

    if len(index) < 10:
        raise ValueError(
            "At least 10 observations are required."
        )

    if index.has_duplicates:
        raise ValueError(
            "Index contains duplicate observations."
        )

    if isinstance(
        index,
        pd.DatetimeIndex,
    ):
        if not index.is_monotonic_increasing:
            raise ValueError(
                "DatetimeIndex must be chronological."
            )


def chronological_split(
    data: pd.DataFrame,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    test_fraction: float = 0.20,
) -> TimeSplit:
    """
    Split data chronologically.

    Example:

        first 60%  -> train
        next 20%   -> validation
        final 20%  -> test

    No shuffling occurs.
    """

    if not np.isclose(
        train_fraction
        + validation_fraction
        + test_fraction,
        1.0,
    ):
        raise ValueError(
            "Train, validation and test fractions must sum to 1."
        )

    if any(
        fraction <= 0
        for fraction in (
            train_fraction,
            validation_fraction,
            test_fraction,
        )
    ):
        raise ValueError(
            "All split fractions must be positive."
        )

    _validate_index(
        data.index
    )

    n = len(data)

    train_end = int(
        n * train_fraction
    )

    validation_end = (
        train_end
        + int(
            n * validation_fraction
        )
    )

    train_index = np.arange(
        0,
        train_end,
    )

    validation_index = np.arange(
        train_end,
        validation_end,
    )

    test_index = np.arange(
        validation_end,
        n,
    )

    if (
        len(train_index) == 0
        or len(validation_index) == 0
        or len(test_index) == 0
    ):
        raise ValueError(
            "One or more chronological splits are empty."
        )

    return TimeSplit(
        train_index=train_index,
        validation_index=validation_index,
        test_index=test_index,
    )


def split_by_dates(
    data: pd.DataFrame,
    train_end: str,
    validation_end: str,
) -> TimeSplit:
    """
    Split data using explicit calendar boundaries.

    Parameters
    ----------
    train_end:
        Last timestamp belonging to training.

    validation_end:
        Last timestamp belonging to validation.

    Everything after validation_end belongs to test.
    """

    _validate_index(
        data.index
    )

    if not isinstance(
        data.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "split_by_dates requires a DatetimeIndex."
        )

    train_end_ts = pd.Timestamp(
        train_end
    )

    validation_end_ts = pd.Timestamp(
        validation_end
    )

    if train_end_ts >= validation_end_ts:
        raise ValueError(
            "train_end must occur before validation_end."
        )

    train_index = np.flatnonzero(
        data.index <= train_end_ts
    )

    validation_index = np.flatnonzero(
        (
            data.index > train_end_ts
        )
        & (
            data.index <= validation_end_ts
        )
    )

    test_index = np.flatnonzero(
        data.index > validation_end_ts
    )

    if (
        len(train_index) == 0
        or len(validation_index) == 0
        or len(test_index) == 0
    ):
        raise ValueError(
            "One or more date-based splits are empty."
        )

    return TimeSplit(
        train_index=train_index,
        validation_index=validation_index,
        test_index=test_index,
    )


def validate_time_split(
    data: pd.DataFrame,
    split: TimeSplit,
) -> None:
    """
    Verify that a split is strictly chronological and non-overlapping.
    """

    train = data.index[
        split.train_index
    ]

    validation = data.index[
        split.validation_index
    ]

    test = data.index[
        split.test_index
    ]

    if train.max() >= validation.min():
        raise ValueError(
            "Training and validation periods overlap."
        )

    if validation.max() >= test.min():
        raise ValueError(
            "Validation and test periods overlap."
        )


def walk_forward_splits(
    data: pd.DataFrame,
    n_splits: int = 5,
    minimum_train_size: Optional[int] = None,
    test_size: Optional[int] = None,
    expanding_window: bool = True,
    gap: int = 0,
) -> Iterator[WalkForwardSplit]:
    """
    Generate chronological walk-forward splits.

    Expanding window example:

        Fold 1:
        TRAIN TRAIN TRAIN | TEST

        Fold 2:
        TRAIN TRAIN TRAIN TRAIN | TEST

        Fold 3:
        TRAIN TRAIN TRAIN TRAIN TRAIN | TEST

    A fixed rolling window can be requested with
    expanding_window=False.

    `gap` creates an observation gap between train and test.
    This is useful as an additional protection against
    temporal contamination.
    """

    _validate_index(
        data.index
    )

    n = len(data)

    if n_splits < 2:
        raise ValueError(
            "n_splits must be at least 2."
        )

    if minimum_train_size is None:
        minimum_train_size = max(
            100,
            n // (
                n_splits + 1
            ),
        )

    if minimum_train_size <= 0:
        raise ValueError(
            "minimum_train_size must be positive."
        )

    if test_size is None:
        remaining = (
            n
            - minimum_train_size
        )

        test_size = remaining // n_splits

    if test_size <= 0:
        raise ValueError(
            "test_size must be positive."
        )

    for fold in range(
        n_splits
    ):

        test_start = (
            minimum_train_size
            + fold * test_size
            + gap
        )

        test_end = (
            test_start
            + test_size
        )

        if test_end > n:
            break

        if expanding_window:

            train_start = 0

        else:

            train_start = (
                fold * test_size
            )

        train_end = (
            minimum_train_size
            + fold * test_size
        )

        train_index = np.arange(
            train_start,
            train_end,
        )

        test_index = np.arange(
            test_start,
            test_end,
        )

        if len(train_index) == 0:
            continue

        if len(test_index) == 0:
            continue

        yield WalkForwardSplit(
            train_index=train_index,
            test_index=test_index,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )


def purged_walk_forward_splits(
    data: pd.DataFrame,
    horizon: int,
    n_splits: int = 5,
    minimum_train_size: Optional[int] = None,
    test_size: Optional[int] = None,
    embargo: int = 0,
    expanding_window: bool = True,
) -> Iterator[WalkForwardSplit]:
    """
    Generate purged walk-forward splits.

    Why purge?

    Suppose a training observation at time T has a target based on
    the next 20 trading days.

    If the test period begins shortly after T, that training target
    can overlap the test period.

    Therefore training observations whose FUTURE LABEL WINDOW
    overlaps the test period are removed.

    Parameters
    ----------
    horizon:
        Number of future observations used to construct the target.

    embargo:
        Additional observations excluded after the training region.
    """

    if horizon <= 0:
        raise ValueError(
            "horizon must be positive."
        )

    for split in walk_forward_splits(
        data=data,
        n_splits=n_splits,
        minimum_train_size=minimum_train_size,
        test_size=test_size,
        expanding_window=expanding_window,
        gap=0,
    ):

        test_start = split.test_start

        purge_boundary = (
            test_start
            - horizon
        )

        train_index = (
            split.train_index[
                split.train_index
                < purge_boundary
            ]
        )

        if embargo > 0:

            embargo_boundary = (
                test_start
                + embargo
            )

            test_index = (
                split.test_index[
                    split.test_index
                    >= embargo_boundary
                ]
            )

        else:

            test_index = (
                split.test_index
            )

        if len(train_index) == 0:
            continue

        if len(test_index) == 0:
            continue

        yield WalkForwardSplit(
            train_index=train_index,
            test_index=test_index,
            train_start=int(
                train_index.min()
            ),
            train_end=int(
                train_index.max()
                + 1
            ),
            test_start=int(
                test_index.min()
            ),
            test_end=int(
                test_index.max()
                + 1
            ),
        )


def split_data(
    data: pd.DataFrame,
    split: TimeSplit,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Materialize a TimeSplit into three dataframes.
    """

    validate_time_split(
        data,
        split,
    )

    train = data.iloc[
        split.train_index
    ].copy()

    validation = data.iloc[
        split.validation_index
    ].copy()

    test = data.iloc[
        split.test_index
    ].copy()

    return (
        train,
        validation,
        test,
    )


def split_walk_forward_data(
    data: pd.DataFrame,
    split: WalkForwardSplit,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Materialize one walk-forward split.
    """

    train = data.iloc[
        split.train_index
    ].copy()

    test = data.iloc[
        split.test_index
    ].copy()

    if train.empty:
        raise ValueError(
            "Walk-forward training set is empty."
        )

    if test.empty:
        raise ValueError(
            "Walk-forward test set is empty."
        )

    if train.index.max() >= test.index.min():
        raise ValueError(
            "Walk-forward training data occurs after test data."
        )

    return (
        train,
        test,
    )


def purge_training_indices(
    train_indices: np.ndarray,
    test_start: int,
    horizon: int,
) -> np.ndarray:
    """
    Remove training observations whose future target window
    reaches into the test period.
    """

    if horizon <= 0:
        raise ValueError(
            "horizon must be positive."
        )

    boundary = (
        test_start
        - horizon
    )

    return train_indices[
        train_indices < boundary
    ]


def apply_embargo(
    test_indices: np.ndarray,
    embargo: int,
) -> np.ndarray:
    """
    Remove the first `embargo` observations from a test region.
    """

    if embargo < 0:
        raise ValueError(
            "embargo cannot be negative."
        )

    if embargo == 0:
        return test_indices

    return test_indices[
        embargo:
    ]


# Backward-compatible public name.
def date_split(
    data: pd.DataFrame,
    train_end: str,
    validation_end: str,
) -> TimeSplit:
    """Split a time-indexed dataframe by explicit date boundaries."""
    return split_by_dates(
        data,
        train_end=train_end,
        validation_end=validation_end,
    )
