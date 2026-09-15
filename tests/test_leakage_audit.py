"""
Tests for the research leakage audit.

These tests intentionally create invalid situations to verify that the
research system refuses contaminated datasets.

No live market data is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.leakage_audit import (
    LeakageAuditConfig,
    LeakageAuditor,
    assert_leakage_free,
    audit_research_dataset,
    compare_future_mutation,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def dataset() -> pd.DataFrame:
    index = pd.date_range(
        "2020-01-01",
        periods=200,
        freq="D",
    )

    close = np.linspace(
        100.0,
        200.0,
        len(index),
    )

    return pd.DataFrame(
        {
            "Open": close - 1.0,
            "High": close + 2.0,
            "Low": close - 2.0,
            "Close": close,
            "Volume": 100_000.0,
            "RSI_14": 50.0,
            "Momentum_5": 0.01,
            "Future_Return_5": 0.05,
            "Direction_5": 1.0,
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
        "RSI_14",
        "Momentum_5",
    ]


@pytest.fixture
def targets() -> list[str]:
    return [
        "Future_Return_5",
        "Direction_5",
    ]


# ----------------------------------------------------------------------
# Basic audit tests
# ----------------------------------------------------------------------


def test_clean_dataset_passes(
    dataset,
    features,
    targets,
):
    report = audit_research_dataset(
        dataset,
        features,
        targets,
    )

    assert report.passed is True
    assert len(
        report.errors
    ) == 0


def test_report_contains_counts(
    dataset,
    features,
    targets,
):
    report = audit_research_dataset(
        dataset,
        features,
        targets,
    )

    assert report.checked_rows == len(
        dataset
    )

    assert (
        report.checked_features
        == len(features)
    )

    assert (
        report.checked_targets
        == len(targets)
    )


def test_summary_is_consistent(
    dataset,
    features,
    targets,
):
    report = audit_research_dataset(
        dataset,
        features,
        targets,
    )

    summary = report.summary()

    assert summary[
        "passed"
    ] is True

    assert summary[
        "rows"
    ] == len(dataset)

    assert summary[
        "features"
    ] == len(features)

    assert summary[
        "targets"
    ] == len(targets)


# ----------------------------------------------------------------------
# Feature / target leakage
# ----------------------------------------------------------------------


def test_feature_target_overlap_fails(
    dataset,
    targets,
):
    features = [
        "Close",
        "RSI_14",
        "Future_Return_5",
    ]

    report = audit_research_dataset(
        dataset,
        features,
        targets,
    )

    assert report.passed is False

    assert any(
        finding.check
        == "feature_target_overlap"
        for finding in report.errors
    )


def test_future_feature_name_fails(
    dataset,
    targets,
):
    features = [
        "Close",
        "RSI_14",
        "Future_Momentum",
    ]

    report = audit_research_dataset(
        dataset,
        features,
        targets,
    )

    assert report.passed is False

    assert any(
        finding.check
        == "future_feature_names"
        for finding in report.errors
    )


def test_forward_feature_name_fails(
    dataset,
    targets,
):
    features = [
        "Close",
        "Forward_Return",
    ]

    report = audit_research_dataset(
        dataset,
        features,
        targets,
    )

    assert report.passed is False


def test_next_day_feature_name_fails(
    dataset,
    targets,
):
    features = [
        "Close",
        "Next_Day_Return",
    ]

    report = audit_research_dataset(
        dataset,
        features,
        targets,
    )

    assert report.passed is False


# ----------------------------------------------------------------------
# Missing data structure
# ----------------------------------------------------------------------


def test_missing_feature_column_fails(
    dataset,
    targets,
):
    features = [
        "Close",
        "DoesNotExist",
    ]

    report = audit_research_dataset(
        dataset,
        features,
        targets,
    )

    assert report.passed is False

    assert any(
        finding.check
        == "feature_presence"
        for finding in report.errors
    )


def test_missing_target_column_fails(
    dataset,
    features,
):
    targets = [
        "Future_Return_100"
    ]

    report = audit_research_dataset(
        dataset,
        features,
        targets,
    )

    assert report.passed is False

    assert any(
        finding.check
        == "target_presence"
        for finding in report.errors
    )


def test_no_targets_fails(
    dataset,
    features,
):
    report = audit_research_dataset(
        dataset,
        features,
        [],
    )

    assert report.passed is False


def test_no_features_fails(
    dataset,
    targets,
):
    report = audit_research_dataset(
        dataset,
        [],
        targets,
    )

    assert report.passed is False


# ----------------------------------------------------------------------
# Chronological integrity
# ----------------------------------------------------------------------


def test_unsorted_dataset_fails(
    dataset,
    features,
    targets,
):
    unsorted = dataset.iloc[
        ::-1
    ]

    report = audit_research_dataset(
        unsorted,
        features,
        targets,
    )

    assert report.passed is False

    assert any(
        finding.check
        == "chronological_order"
        for finding in report.errors
    )


def test_duplicate_timestamps_fail(
    dataset,
    features,
    targets,
):
    duplicate = pd.concat(
        [
            dataset,
            dataset.iloc[
                :1
            ],
        ]
    )

    report = audit_research_dataset(
        duplicate,
        features,
        targets,
    )

    assert report.passed is False

    assert any(
        finding.check
        == "duplicate_timestamps"
        for finding in report.errors
    )


def test_non_datetime_index_fails(
    dataset,
    features,
    targets,
):
    invalid = dataset.copy()

    invalid.index = range(
        len(invalid)
    )

    report = audit_research_dataset(
        invalid,
        features,
        targets,
    )

    assert report.passed is False


# ----------------------------------------------------------------------
# Feature-value checks
# ----------------------------------------------------------------------


def test_non_numeric_feature_fails(
    dataset,
    targets,
):
    invalid = dataset.copy()

    invalid[
        "TextFeature"
    ] = "bad"

    features = [
        "Close",
        "TextFeature",
    ]

    report = audit_research_dataset(
        invalid,
        features,
        targets,
    )

    assert report.passed is False

    assert any(
        finding.check
        == "numeric_feature"
        for finding in report.errors
    )


def test_infinite_feature_fails(
    dataset,
    targets,
):
    invalid = dataset.copy()

    invalid[
        "InfiniteFeature"
    ] = np.inf

    features = [
        "Close",
        "InfiniteFeature",
    ]

    report = audit_research_dataset(
        invalid,
        features,
        targets,
    )

    assert report.passed is False

    assert any(
        finding.check
        == "infinite_feature"
        for finding in report.errors
    )


def test_nan_feature_is_not_automatically_leakage(
    dataset,
    features,
    targets,
):
    """
    NaN values can legitimately occur during indicator warm-up.

    Leakage auditing should not confuse missing values with future
    information.
    """

    modified = dataset.copy()

    modified[
        "RSI_14"
    ] = np.nan

    report = audit_research_dataset(
        modified,
        features,
        targets,
    )

    assert report.passed is True


# ----------------------------------------------------------------------
# Temporal holdout tests
# ----------------------------------------------------------------------


def test_development_and_holdout_are_separated():
    index = pd.date_range(
        "2020-01-01",
        periods=100,
        freq="D",
    )

    development = pd.DataFrame(
        {
            "Close": np.arange(
                70
            )
        },
        index=index[:70],
    )

    holdout = pd.DataFrame(
        {
            "Close": np.arange(
                30
            )
        },
        index=index[70:],
    )

    report = LeakageAuditor.audit_temporal_separation(
        development,
        holdout,
    )

    assert report.passed is True


def test_overlapping_development_and_holdout_fail():
    index = pd.date_range(
        "2020-01-01",
        periods=100,
        freq="D",
    )

    development = pd.DataFrame(
        {
            "Close": np.arange(
                80
            )
        },
        index=index[:80],
    )

    holdout = pd.DataFrame(
        {
            "Close": np.arange(
                30
            )
        },
        index=index[70:],
    )

    report = LeakageAuditor.audit_temporal_separation(
        development,
        holdout,
    )

    assert report.passed is False


def test_empty_holdout_fails():
    index = pd.date_range(
        "2020-01-01",
        periods=50,
        freq="D",
    )

    development = pd.DataFrame(
        {
            "Close": np.arange(
                50
            )
        },
        index=index,
    )

    holdout = pd.DataFrame(
        columns=[
            "Close"
        ],
            index=pd.DatetimeIndex(
                [],
                dtype="datetime64[ns]",
            ),
    )

    report = LeakageAuditor.audit_temporal_separation(
        development,
        holdout,
    )

    assert report.passed is False


# ----------------------------------------------------------------------
# Future mutation tests
# ----------------------------------------------------------------------


def test_future_mutation_does_not_change_past_features():
    index = pd.date_range(
        "2020-01-01",
        periods=100,
        freq="D",
    )

    original = pd.DataFrame(
        {
            "Close": np.arange(
                100,
                200,
            ),
            "RSI_14": np.linspace(
                40,
                60,
                100,
            ),
        },
        index=index,
    )

    mutated = original.copy()

    cutoff = index[69]

    mutated.loc[
        index[70:],
        "Close",
    ] = 10_000

    report = compare_future_mutation(
        original,
        mutated,
        [
            "RSI_14"
        ],
        cutoff=cutoff,
    )

    assert report.passed is True


def test_future_mutation_detects_leaky_feature():
    index = pd.date_range(
        "2020-01-01",
        periods=100,
        freq="D",
    )

    original = pd.DataFrame(
        {
            "Close": np.arange(
                100,
                200,
            ),
        },
        index=index,
    )

    mutated = original.copy()

    cutoff = index[69]

    # Simulate a feature that incorrectly depends on future values.
    original[
        "LeakyFeature"
    ] = original[
        "Close"
    ].shift(
        -10
    )

    mutated.loc[
        index[70:],
        "Close",
    ] = 10_000

    mutated[
        "LeakyFeature"
    ] = mutated[
        "Close"
    ].shift(
        -10
    )

    report = compare_future_mutation(
        original,
        mutated,
        [
            "LeakyFeature"
        ],
        cutoff=cutoff,
    )

    assert report.passed is False

    assert any(
        finding.check
        == "future_mutation"
        for finding in report.errors
    )


# ----------------------------------------------------------------------
# Configuration tests
# ----------------------------------------------------------------------


def test_custom_suspicious_token():
    config = LeakageAuditConfig(
        suspicious_tokens=(
            "future",
            "lookahead",
        )
    )

    auditor = LeakageAuditor(
        config
    )

    assert (
        "lookahead"
        in auditor.config.suspicious_tokens
    )


def test_feature_count_requirement_can_be_changed(
    dataset,
    targets,
):
    config = LeakageAuditConfig(
        minimum_feature_count=5
    )

    report = audit_research_dataset(
        dataset,
        [
            "Close"
        ],
        targets,
        config=config,
    )

    assert report.passed is False


# ----------------------------------------------------------------------
# Assert helper
# ----------------------------------------------------------------------


def test_assert_leakage_free_passes(
    dataset,
    features,
    targets,
):
    report = assert_leakage_free(
        dataset,
        features,
        targets,
    )

    assert report.passed is True


def test_assert_leakage_free_raises(
    dataset,
    targets,
):
    features = [
        "Close",
        "Future_Return_5",
    ]

    with pytest.raises(
        ValueError,
        match="failed leakage audit",
    ):
        assert_leakage_free(
            dataset,
            features,
            targets,
        )


# ----------------------------------------------------------------------
# Multiple findings
# ----------------------------------------------------------------------


def test_audit_can_report_multiple_errors(
    dataset,
    targets,
):
    invalid = dataset.copy()

    invalid.index = range(
        len(invalid)
    )

    features = [
        "Close",
        "Future_Return_5",
        "MissingFeature",
    ]

    report = audit_research_dataset(
        invalid,
        features,
        targets,
    )

    assert report.passed is False

    assert len(
        report.errors
    ) >= 2
