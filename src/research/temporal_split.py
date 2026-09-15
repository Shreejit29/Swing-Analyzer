"""
AI Swing Analyser — Temporal Dataset Splitting.

Implements leakage-safe chronological splitting for financial time series.

Supported methods:

1. Train / Validation / Holdout chronological split.
2. Expanding Walk-Forward Validation.
3. Purged Walk-Forward Validation.
4. Embargo protection between train and validation.

No random shuffling is ever allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd

from .config import ResearchValidationConfig


# ---------------------------------------------------------------------
# Split containers
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalSplit:
    train_index: pd.Index
    validation_index: pd.Index
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


@dataclass(frozen=True)
class HoldoutSplit:
    development_index: pd.Index
    holdout_index: pd.Index
    development_end: pd.Timestamp
    holdout_start: pd.Timestamp


# ---------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------


class TemporalSplitter:
    """
    Leakage-safe chronological splitter.
    """

    def __init__(
        self,
        validation_config: ResearchValidationConfig,
    ) -> None:
        self.config = validation_config

    # --------------------------------------------------------------
    # Final Holdout
    # --------------------------------------------------------------

    def holdout_split(
        self,
        dataframe: pd.DataFrame,
    ) -> HoldoutSplit:

        self._validate_dataframe(dataframe)

        total = len(dataframe)

        development_size = int(
            total
            * (
                self.config.train_fraction
                + self.config.validation_fraction
            )
        )

        development = dataframe.iloc[:development_size]
        holdout = dataframe.iloc[development_size:]

        if len(holdout) < self.config.minimum_validation_samples:
            raise ValueError(
                "Holdout dataset is too small."
            )

        return HoldoutSplit(
            development_index=development.index,
            holdout_index=holdout.index,
            development_end=development.index.max(),
            holdout_start=holdout.index.min(),
        )

    # --------------------------------------------------------------
    # Walk Forward Generator
    # --------------------------------------------------------------

    def walk_forward(
        self,
        dataframe: pd.DataFrame,
    ) -> Iterator[TemporalSplit]:

        self._validate_dataframe(dataframe)

        development = dataframe.loc[
            self.holdout_split(dataframe).development_index
        ]

        total = len(development)

        folds = self.config.n_walk_forward_splits

        minimum_train = self.config.minimum_train_samples

        minimum_validation = self.config.minimum_validation_samples

        usable = total - minimum_train

        validation_window = usable // folds

        if validation_window < minimum_validation:
            raise ValueError(
                "Validation window is too small."
            )

        for fold in range(folds):

            validation_start = (
                minimum_train
                + fold * validation_window
            )

            validation_end = (
                validation_start
                + validation_window
            )

            validation_end = min(
                validation_end,
                total,
            )

            train_end = (
                validation_start
                - self.config.gap
                - self.config.embargo
            )

            if train_end < minimum_train:
                continue

            if self.config.expanding:
                train = development.iloc[:train_end]
            else:
                train = development.iloc[
                    fold * validation_window : train_end
                ]

            validation = development.iloc[
                validation_start:validation_end
            ]

            if (
                len(train)
                < minimum_train
                or len(validation)
                < minimum_validation
            ):
                continue

            yield TemporalSplit(
                train_index=train.index,
                validation_index=validation.index,
                train_start=train.index.min(),
                train_end=train.index.max(),
                validation_start=validation.index.min(),
                validation_end=validation.index.max(),
            )

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------

    @staticmethod
    def _validate_dataframe(
        dataframe: pd.DataFrame,
    ) -> None:

        if dataframe.empty:
            raise ValueError("Dataset is empty.")

        if not isinstance(
            dataframe.index,
            pd.DatetimeIndex,
        ):
            raise TypeError(
                "Dataset must use DatetimeIndex."
            )

        if dataframe.index.has_duplicates:
            raise ValueError(
                "Dataset contains duplicate timestamps."
            )

        if not dataframe.index.is_monotonic_increasing:
            raise ValueError(
                "Dataset is not chronologically sorted."
            )


# ---------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------


def create_holdout_split(
    dataframe: pd.DataFrame,
    validation_config: ResearchValidationConfig,
) -> HoldoutSplit:

    splitter = TemporalSplitter(validation_config)

    return splitter.holdout_split(dataframe)


def create_walk_forward_splits(
    dataframe: pd.DataFrame,
    validation_config: ResearchValidationConfig,
) -> list[TemporalSplit]:

    splitter = TemporalSplitter(validation_config)

    return list(
        splitter.walk_forward(dataframe)
    )


__all__ = [
    "TemporalSplit",
    "HoldoutSplit",
    "TemporalSplitter",
    "create_holdout_split",
    "create_walk_forward_splits",
]
