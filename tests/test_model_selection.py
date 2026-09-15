"""
Tests for controlled development-only model selection.
"""

from __future__ import annotations

import pytest

from src.research.model_selection import (
    ControlledModelSelector,
    ModelCandidateScore,
    ModelSelectionConfig,
    select_best_model,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def candidate(
    candidate_id: str,
    model_name: str = "gradient_boosting",
    horizon: int = 5,
    validation_accuracy: float = 0.96,
    walk_forward_accuracy: float = 0.96,
    walk_forward_accuracy_std: float = 0.02,
    walk_forward_min_accuracy: float = 0.95,
    validation_brier: float | None = 0.10,
    feature_count: int = 20,
    training_samples: int = 500,
) -> ModelCandidateScore:

    return ModelCandidateScore(
        candidate_id=candidate_id,
        model_name=model_name,
        horizon=horizon,
        validation_accuracy=validation_accuracy,
        validation_brier=validation_brier,
        walk_forward_accuracy=walk_forward_accuracy,
        walk_forward_accuracy_std=walk_forward_accuracy_std,
        walk_forward_min_accuracy=walk_forward_min_accuracy,
        feature_count=feature_count,
        training_samples=training_samples,
    )


# ---------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------


def test_selector_constructs():

    selector = ControlledModelSelector()

    assert selector.config is not None


def test_default_selection_config_is_strict():

    config = ModelSelectionConfig()

    assert config.minimum_validation_accuracy == 0.95
    assert config.minimum_walk_forward_accuracy == 0.95
    assert config.maximum_walk_forward_accuracy_std == 0.10
    assert config.require_walk_forward is True


# ---------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------


def test_invalid_accuracy_threshold_is_rejected():

    with pytest.raises(ValueError):

        ControlledModelSelector(
            ModelSelectionConfig(
                minimum_validation_accuracy=1.1
            )
        )


def test_negative_accuracy_std_threshold_is_rejected():

    with pytest.raises(ValueError):

        ControlledModelSelector(
            ModelSelectionConfig(
                maximum_walk_forward_accuracy_std=-0.01
            )
        )


def test_invalid_brier_threshold_is_rejected():

    with pytest.raises(ValueError):

        ControlledModelSelector(
            ModelSelectionConfig(
                maximum_brier_score=1.1
            )
        )


def test_invalid_candidate_limit_is_rejected():

    with pytest.raises(ValueError):

        ControlledModelSelector(
            ModelSelectionConfig(
                maximum_candidates=0
            )
        )


# ---------------------------------------------------------------------
# Candidate validation
# ---------------------------------------------------------------------


def test_empty_candidate_id_is_rejected():

    selector = ControlledModelSelector()

    with pytest.raises(ValueError):

        selector.select(
            [
                candidate("")
            ]
        )


def test_duplicate_candidate_ids_are_rejected():

    selector = ControlledModelSelector()

    with pytest.raises(ValueError):

        selector.select(
            [
                candidate("MODEL-1"),
                candidate("MODEL-1"),
            ]
        )


def test_empty_model_name_is_rejected():

    selector = ControlledModelSelector()

    with pytest.raises(ValueError):

        selector.select(
            [
                candidate(
                    "MODEL-1",
                    model_name="",
                )
            ]
        )


def test_nonpositive_horizon_is_rejected():

    selector = ControlledModelSelector()

    with pytest.raises(ValueError):

        selector.select(
            [
                candidate(
                    "MODEL-1",
                    horizon=0,
                )
            ]
        )


def test_invalid_validation_accuracy_is_rejected():

    selector = ControlledModelSelector()

    with pytest.raises(ValueError):

        selector.select(
            [
                candidate(
                    "MODEL-1",
                    validation_accuracy=1.2,
                )
            ]
        )


# ---------------------------------------------------------------------
# Empty selection
# ---------------------------------------------------------------------


def test_empty_candidate_list_returns_no_selection():

    selector = ControlledModelSelector()

    result = selector.select([])

    assert result.selected is None
    assert not result.successful
    assert result.metadata[
        "final_holdout_used"
    ] is False


# ---------------------------------------------------------------------
# Candidate eligibility
# ---------------------------------------------------------------------


def test_candidate_below_validation_threshold_is_rejected():

    selector = ControlledModelSelector()

    result = selector.select(
        [
            candidate(
                "MODEL-1",
                validation_accuracy=0.94,
            )
        ]
    )

    assert result.selected is None
    assert len(
        result.rejected_candidates
    ) == 1


def test_candidate_below_walk_forward_threshold_is_rejected():

    selector = ControlledModelSelector()

    result = selector.select(
        [
            candidate(
                "MODEL-1",
                walk_forward_accuracy=0.94,
            )
        ]
    )

    assert result.selected is None


def test_candidate_without_walk_forward_is_rejected():

    selector = ControlledModelSelector()

    result = selector.select(
        [
            candidate(
                "MODEL-1",
                walk_forward_accuracy=None,
            )
        ]
    )

    assert result.selected is None


def test_unstable_walk_forward_candidate_is_rejected():

    selector = ControlledModelSelector()

    result = selector.select(
        [
            candidate(
                "MODEL-1",
                walk_forward_accuracy_std=0.20,
            )
        ]
    )

    assert result.selected is None


def test_lowest_walk_forward_fold_can_reject_candidate():

    selector = ControlledModelSelector()

    result = selector.select(
        [
            candidate(
                "MODEL-1",
                walk_forward_min_accuracy=0.90,
            )
        ]
    )

    assert result.selected is None


def test_brier_can_be_made_mandatory():

    selector = ControlledModelSelector(
        ModelSelectionConfig(
            require_brier=True
        )
    )

    result = selector.select(
        [
            candidate(
                "MODEL-1",
                validation_brier=None,
            )
        ]
    )

    assert result.selected is None


def test_excessive_brier_is_rejected():

    selector = ControlledModelSelector(
        ModelSelectionConfig(
            require_brier=True
        )
    )

    result = selector.select(
        [
            candidate(
                "MODEL-1",
                validation_brier=0.50,
            )
        ]
    )

    assert result.selected is None


# ---------------------------------------------------------------------
# Successful selection
# ---------------------------------------------------------------------


def test_passing_candidate_is_selected():

    selector = ControlledModelSelector()

    result = selector.select(
        [
            candidate("MODEL-1")
        ]
    )

    assert result.successful
    assert result.selected is not None
    assert (
        result.selected.candidate_id
        == "MODEL-1"
    )


def test_selected_candidate_is_first_ranked_candidate():

    selector = ControlledModelSelector()

    result = selector.select(
        [
            candidate("MODEL-1"),
            candidate(
                "MODEL-2",
                validation_accuracy=0.97,
                walk_forward_accuracy=0.97,
            ),
        ]
    )

    assert result.selected is not None

    assert (
        result.selected.candidate_id
        == result.ranked_candidates[0].candidate_id
    )


def test_higher_walk_forward_accuracy_is_preferred():

    selector = ControlledModelSelector()

    first = candidate(
        "MODEL-1",
        walk_forward_accuracy=0.955,
    )

    second = candidate(
        "MODEL-2",
        walk_forward_accuracy=0.975,
    )

    result = selector.select(
        [
            first,
            second,
        ]
    )

    assert result.selected is not None
    assert (
        result.selected.candidate_id
        == "MODEL-2"
    )


def test_more_stable_candidate_is_preferred_when_accuracy_is_similar():

    selector = ControlledModelSelector()

    first = candidate(
        "MODEL-1",
        walk_forward_accuracy=0.96,
        walk_forward_accuracy_std=0.08,
    )

    second = candidate(
        "MODEL-2",
        walk_forward_accuracy=0.96,
        walk_forward_accuracy_std=0.02,
    )

    result = selector.select(
        [
            first,
            second,
        ]
    )

    assert result.selected is not None
    assert (
        result.selected.candidate_id
        == "MODEL-2"
    )


# ---------------------------------------------------------------------
# Feature complexity
# ---------------------------------------------------------------------


def test_simpler_model_can_win_when_accuracy_is_equal():

    selector = ControlledModelSelector(
        ModelSelectionConfig(
            prefer_simpler_models=True,
            feature_count_penalty=0.001,
        )
    )

    complex_model = candidate(
        "COMPLEX",
        feature_count=100,
    )

    simple_model = candidate(
        "SIMPLE",
        feature_count=10,
    )

    result = selector.select(
        [
            complex_model,
            simple_model,
        ]
    )

    assert result.selected is not None

    assert (
        result.selected.candidate_id
        == "SIMPLE"
    )


def test_feature_penalty_can_be_disabled():

    selector = ControlledModelSelector(
        ModelSelectionConfig(
            prefer_simpler_models=False
        )
    )

    first = candidate(
        "MODEL-1",
        feature_count=100,
    )

    second = candidate(
        "MODEL-2",
        feature_count=10,
    )

    result = selector.select(
        [
            first,
            second,
        ]
    )

    assert result.selected is not None


# ---------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------


def test_ranked_candidates_are_ordered():

    selector = ControlledModelSelector()

    candidates = [
        candidate(
            "LOW",
            walk_forward_accuracy=0.955,
        ),
        candidate(
            "HIGH",
            walk_forward_accuracy=0.98,
        ),
        candidate(
            "MID",
            walk_forward_accuracy=0.97,
        ),
    ]

    result = selector.select(
        candidates
    )

    ranked_ids = [
        item.candidate_id
        for item in result.ranked_candidates
    ]

    assert ranked_ids[0] == "HIGH"


def test_rejected_candidates_are_separate():

    selector = ControlledModelSelector()

    result = selector.select(
        [
            candidate(
                "PASS",
                validation_accuracy=0.97,
            ),
            candidate(
                "FAIL",
                validation_accuracy=0.80,
            ),
        ]
    )

    assert (
        result.selected is not None
    )

    assert any(
        item.candidate_id == "FAIL"
        for item in result.rejected_candidates
    )


# ---------------------------------------------------------------------
# Final holdout protection
# ---------------------------------------------------------------------


def test_final_holdout_usage_is_rejected():

    selector = ControlledModelSelector()

    with pytest.raises(ValueError):

        selector.select(
            [
                candidate("MODEL-1")
            ],
            final_holdout_used=True,
        )


def test_selection_metadata_confirms_holdout_was_not_used():

    selector = ControlledModelSelector()

    result = selector.select(
        [
            candidate("MODEL-1")
        ]
    )

    assert (
        result.metadata[
            "final_holdout_used"
        ]
        is False
    )


def test_selection_is_research_only():

    selector = ControlledModelSelector()

    result = selector.select(
        [
            candidate("MODEL-1")
        ]
    )

    assert (
        result.metadata[
            "research_only"
        ]
        is True
    )


# ---------------------------------------------------------------------
# Candidate count protection
# ---------------------------------------------------------------------


def test_candidate_count_limit_is_enforced():

    selector = ControlledModelSelector(
        ModelSelectionConfig(
            maximum_candidates=2
        )
    )

    candidates = [
        candidate("MODEL-1"),
        candidate("MODEL-2"),
        candidate("MODEL-3"),
    ]

    with pytest.raises(ValueError):

        selector.select(
            candidates
        )


# ---------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------


def test_compare_returns_positive_for_better_candidate():

    first = candidate(
        "FIRST",
        walk_forward_accuracy=0.98,
    )

    second = candidate(
        "SECOND",
        walk_forward_accuracy=0.96,
    )

    assert (
        ControlledModelSelector.compare(
            first,
            second,
        )
        == 1
    )


def test_compare_returns_negative_for_worse_candidate():

    first = candidate(
        "FIRST",
        walk_forward_accuracy=0.94,
    )

    second = candidate(
        "SECOND",
        walk_forward_accuracy=0.96,
    )

    assert (
        ControlledModelSelector.compare(
            first,
            second,
        )
        == -1
    )


def test_compare_returns_zero_for_equal_candidates():

    first = candidate(
        "FIRST",
    )

    second = candidate(
        "SECOND",
    )

    assert (
        ControlledModelSelector.compare(
            first,
            second,
        )
        == 0
    )


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------


def test_summary_contains_selection_information():

    selector = ControlledModelSelector()

    result = selector.select(
        [
            candidate("MODEL-1")
        ]
    )

    summary = selector.summary(
        result
    )

    assert summary["successful"] is True

    assert (
        summary["selected_candidate_id"]
        == "MODEL-1"
    )

    assert (
        summary["final_holdout_used"]
        is False
    )

    assert (
        summary["research_only"]
        is True
    )


# ---------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------


def test_convenience_selection_function():

    result = select_best_model(
        [
            candidate("MODEL-1"),
            candidate(
                "MODEL-2",
                walk_forward_accuracy=0.98,
            ),
        ]
    )

    assert result.successful
    assert (
        result.selected_candidate_id
        == "MODEL-2"
    )


def test_convenience_function_rejects_holdout_selection():

    with pytest.raises(ValueError):

        select_best_model(
            [
                candidate("MODEL-1")
            ],
            final_holdout_used=True,
        )


# ---------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------


def test_selection_is_deterministic():

    candidates = [
        candidate(
            "MODEL-1",
            walk_forward_accuracy=0.96,
        ),
        candidate(
            "MODEL-2",
            walk_forward_accuracy=0.97,
        ),
        candidate(
            "MODEL-3",
            walk_forward_accuracy=0.955,
        ),
    ]

    selector = ControlledModelSelector()

    first = selector.select(
        candidates
    )

    second = selector.select(
        candidates
    )

    assert (
        first.selected_candidate_id
        == second.selected_candidate_id
    )

    assert [
        item.candidate_id
        for item in first.ranked_candidates
    ] == [
        item.candidate_id
        for item in second.ranked_candidates
    ]


# ---------------------------------------------------------------------
# No mutation
# ---------------------------------------------------------------------


def test_selection_does_not_mutate_input_candidates():

    candidates = [
        candidate("MODEL-1"),
        candidate("MODEL-2"),
    ]

    original_ids = [
        item.candidate_id
        for item in candidates
    ]

    selector = ControlledModelSelector()

    selector.select(
        candidates
    )

    assert [
        item.candidate_id
        for item in candidates
    ] == original_ids
