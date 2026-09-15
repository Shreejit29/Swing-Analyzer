"""
Tests for the research experiment runner.

The main purpose of these tests is to ensure that:

- candidate experiments are reproducible
- duplicate candidates are rejected
- invalid candidates are rejected
- candidates are selected using validation data
- final holdout data is never used for candidate selection
- failed candidates do not hide successful candidates
- no candidate is treated as production-approved
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.selection import FeatureSelectionConfig
from src.models.classifier import ClassifierConfig
from src.models.preprocessing import PreprocessorConfig
from src.research.experiment_runner import (
    ExperimentCandidate,
    ExperimentRunResult,
    ResearchExperimentRunner,
)
from src.research.model_development import (
    ModelDevelopmentOrchestrator,
)
from src.research.config import (
    ResearchPipelineConfig,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(123)

    index = pd.date_range(
        "2020-01-01",
        periods=600,
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
        f1 * 0.4
        + rng.normal(
            0,
            0.5,
            len(index),
        )
    )

    signal = (
        f1
        + 0.25 * f2
        + rng.normal(
            0,
            0.2,
            len(index),
        )
    )

    direction = (
        signal > 0
    ).astype(int)

    return pd.DataFrame(
        {
            "Open": 100 + f1,
            "High": 101 + f1,
            "Low": 99 + f1,
            "Close": 100 + f1,
            "Volume": 100_000 + f2 * 1_000,
            "Feature_1": f1,
            "Feature_2": f2,
            "Feature_3": f3,
            "Direction_5": direction,
            "Direction_10": direction,
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
def runner(
    config,
) -> ResearchExperimentRunner:

    orchestrator = (
        ModelDevelopmentOrchestrator(
            config=config,
            preprocessor_config=PreprocessorConfig(),
            classifier_config=ClassifierConfig(),
            feature_selection_config=FeatureSelectionConfig(
                minimum_features=1,
                maximum_correlation=0.99,
            ),
        )
    )

    return ResearchExperimentRunner(
        config=config,
        orchestrator=orchestrator,
    )


# ---------------------------------------------------------------------
# Candidate validation
# ---------------------------------------------------------------------


def test_candidate_can_be_created():
    candidate = ExperimentCandidate(
        model_name="random_forest",
        horizon=5,
        target_column="Direction_5",
    )

    assert (
        candidate.model_name
        == "random_forest"
    )

    assert candidate.horizon == 5


def test_empty_candidate_list_fails(
    runner,
    dataframe,
    features,
):
    with pytest.raises(
        ValueError
    ):
        runner.run(
            dataframe,
            features,
            [],
        )


def test_empty_model_name_fails(
    runner,
    dataframe,
    features,
):
    candidate = ExperimentCandidate(
        model_name="",
        horizon=5,
        target_column="Direction_5",
    )

    with pytest.raises(
        ValueError
    ):
        runner.run(
            dataframe,
            features,
            [candidate],
        )


def test_non_positive_horizon_fails(
    runner,
    dataframe,
    features,
):
    candidate = ExperimentCandidate(
        model_name="random_forest",
        horizon=0,
        target_column="Direction_5",
    )

    with pytest.raises(
        ValueError
    ):
        runner.run(
            dataframe,
            features,
            [candidate],
        )


def test_empty_target_name_fails(
    runner,
    dataframe,
    features,
):
    candidate = ExperimentCandidate(
        model_name="random_forest",
        horizon=5,
        target_column="",
    )

    with pytest.raises(
        ValueError
    ):
        runner.run(
            dataframe,
            features,
            [candidate],
        )


def test_duplicate_candidates_fail(
    runner,
    dataframe,
    features,
):
    candidate = ExperimentCandidate(
        model_name="random_forest",
        horizon=5,
        target_column="Direction_5",
    )

    with pytest.raises(
        ValueError
    ):
        runner.run(
            dataframe,
            features,
            [
                candidate,
                candidate,
            ],
        )


# ---------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------


def test_missing_target_fails(
    runner,
    dataframe,
    features,
):
    candidate = ExperimentCandidate(
        model_name="random_forest",
        horizon=5,
        target_column="Direction_999",
    )

    with pytest.raises(
        ValueError
    ):
        runner.run_candidate(
            dataframe,
            features,
            candidate,
            apply_feature_selection=False,
        )


# ---------------------------------------------------------------------
# Default candidates
# ---------------------------------------------------------------------


def test_default_candidates_are_generated():
    candidates = (
        ResearchExperimentRunner.default_candidates(
            horizons=[
                1,
                5,
            ]
        )
    )

    assert len(
        candidates
    ) == 6

    assert all(
        isinstance(
            candidate,
            ExperimentCandidate,
        )
        for candidate in candidates
    )


def test_default_candidate_targets_match_horizon():
    candidates = (
        ResearchExperimentRunner.default_candidates(
            horizons=[
                1,
                5,
                20,
            ]
        )
    )

    for candidate in candidates:
        assert (
            candidate.target_column
            == f"Direction_{candidate.horizon}"
        )


# ---------------------------------------------------------------------
# Candidate execution
# ---------------------------------------------------------------------


def test_run_candidate_returns_model_result(
    runner,
    dataframe,
    features,
):
    candidate = ExperimentCandidate(
        model_name="random_forest",
        horizon=5,
        target_column="Direction_5",
    )

    result = runner.run_candidate(
        dataframe,
        features,
        candidate,
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


def test_run_candidate_does_not_use_holdout(
    runner,
    dataframe,
    features,
):
    candidate = ExperimentCandidate(
        model_name="random_forest",
        horizon=5,
        target_column="Direction_5",
    )

    result = runner.run_candidate(
        dataframe,
        features,
        candidate,
        apply_feature_selection=False,
    )

    assert (
        result.metadata[
            "final_holdout_used"
        ]
        is False
    )


# ---------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------


def test_select_best_candidate():
    class FakeMetrics:
        def __init__(
            self,
            accuracy,
        ):
            self.accuracy = accuracy

    class FakeResult:
        def __init__(
            self,
            accuracy,
            feature_count,
        ):
            self.validation_metrics = FakeMetrics(
                accuracy
            )
            self.feature_columns = [
                f"f{i}"
                for i in range(
                    feature_count
                )
            ]

        @property
        def validation_accuracy(self):
            return (
                self.validation_metrics.accuracy
            )

    results = [
        FakeResult(
            0.70,
            10,
        ),
        FakeResult(
            0.91,
            10,
        ),
        FakeResult(
            0.82,
            10,
        ),
    ]

    best = (
        ResearchExperimentRunner.select_best_candidate(
            results
        )
    )

    assert (
        best.validation_accuracy
        == 0.91
    )


def test_select_best_candidate_requires_results():
    with pytest.raises(
        ValueError
    ):
        ResearchExperimentRunner.select_best_candidate(
            []
        )


# ---------------------------------------------------------------------
# Full experiment run
# ---------------------------------------------------------------------


def test_run_returns_result(
    runner,
    dataframe,
    features,
):
    candidates = [
        ExperimentCandidate(
            model_name="random_forest",
            horizon=5,
            target_column="Direction_5",
        ),
    ]

    result = runner.run(
        dataframe,
        features,
        candidates,
        apply_feature_selection=False,
    )

    assert isinstance(
        result,
        ExperimentRunResult,
    )

    assert (
        result.candidate_count
        == 1
    )

    assert result.successful


def test_selected_candidate_is_present(
    runner,
    dataframe,
    features,
):
    candidates = [
        ExperimentCandidate(
            model_name="random_forest",
            horizon=5,
            target_column="Direction_5",
        ),
        ExperimentCandidate(
            model_name="gradient_boosting",
            horizon=5,
            target_column="Direction_5",
        ),
    ]

    result = runner.run(
        dataframe,
        features,
        candidates,
        apply_feature_selection=False,
    )

    assert (
        result.selected_candidate
        is not None
    )


def test_holdout_is_not_used_for_selection(
    runner,
    dataframe,
    features,
):
    candidates = [
        ExperimentCandidate(
            model_name="random_forest",
            horizon=5,
            target_column="Direction_5",
        ),
        ExperimentCandidate(
            model_name="gradient_boosting",
            horizon=5,
            target_column="Direction_5",
        ),
    ]

    result = runner.run(
        dataframe,
        features,
        candidates,
        apply_feature_selection=False,
    )

    assert (
        result.holdout_used_for_selection
        is False
    )

    assert (
        result.metadata[
            "final_holdout_reserved"
        ]
        is True
    )


def test_candidate_selection_is_marked_research_only(
    runner,
    dataframe,
    features,
):
    candidates = [
        ExperimentCandidate(
            model_name="random_forest",
            horizon=5,
            target_column="Direction_5",
        ),
    ]

    result = runner.run(
        dataframe,
        features,
        candidates,
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
            "candidate_selection_frozen"
        ]
        is True
    )


# ---------------------------------------------------------------------
# Holdout mutation attack
# ---------------------------------------------------------------------


def test_changing_holdout_does_not_change_development_data(
    runner,
    dataframe,
    features,
):
    """
    Adversarial test:

    Modify only the final holdout and confirm that the development
    portion remains exactly unchanged.
    """

    original = dataframe.copy()

    partitions = (
        runner.orchestrator
        .split_development_holdout(
            original
        )
    )

    mutated = original.copy()

    mutated.loc[
        partitions.holdout_index,
        features,
    ] = 999_999.0

    mutated_partitions = (
        runner.orchestrator
        .split_development_holdout(
            mutated
        )
    )

    pd.testing.assert_frame_equal(
        partitions.development,
        mutated_partitions.development,
    )


# ---------------------------------------------------------------------
# Failed candidate handling
# ---------------------------------------------------------------------


def test_failed_candidate_does_not_prevent_valid_candidate(
    runner,
    dataframe,
    features,
):
    candidates = [
        ExperimentCandidate(
            model_name="invalid_model",
            horizon=5,
            target_column="Direction_5",
        ),
        ExperimentCandidate(
            model_name="random_forest",
            horizon=5,
            target_column="Direction_5",
        ),
    ]

    result = runner.run(
        dataframe,
        features,
        candidates,
        apply_feature_selection=False,
    )

    assert result.successful

    assert (
        result.candidate_count
        >= 1
    )

    assert result.warnings


def test_all_failed_candidates_raise(
    runner,
    dataframe,
    features,
):
    candidates = [
        ExperimentCandidate(
            model_name="invalid_model_a",
            horizon=5,
            target_column="Direction_5",
        ),
        ExperimentCandidate(
            model_name="invalid_model_b",
            horizon=5,
            target_column="Direction_5",
        ),
    ]

    with pytest.raises(
        RuntimeError
    ):
        runner.run(
            dataframe,
            features,
            candidates,
            apply_feature_selection=False,
        )


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------


def test_summary_contains_key_fields(
    runner,
    dataframe,
    features,
):
    candidates = [
        ExperimentCandidate(
            model_name="random_forest",
            horizon=5,
            target_column="Direction_5",
        ),
    ]

    result = runner.run(
        dataframe,
        features,
        candidates,
        apply_feature_selection=False,
    )

    summary = result.summary()

    assert (
        "candidate_count"
        in summary
    )

    assert (
        "successful"
        in summary
    )

    assert (
        "holdout_used_for_selection"
        in summary
    )

    assert (
        summary[
            "holdout_used_for_selection"
        ]
        is False
    )


# ---------------------------------------------------------------------
# Deterministic candidate generation
# ---------------------------------------------------------------------


def test_default_candidate_generation_is_deterministic():
    first = (
        ResearchExperimentRunner.default_candidates()
    )

    second = (
        ResearchExperimentRunner.default_candidates()
    )

    assert first == second
