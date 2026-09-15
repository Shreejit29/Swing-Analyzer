"""
Tests for the controlled model-selection pipeline.
"""

from __future__ import annotations

import copy

import pytest

from src.research.model_selection import ModelSelectionConfig
from src.research.selection_pipeline import (
    SelectionCandidate,
    SelectionPipeline,
    SelectionPipelineResult,
    run_model_selection,
)


def candidate(
    model_id: str = "model_1",
    model_name: str = "GradientBoosting",
    horizon: int = 5,
    validation_accuracy: float = 0.97,
    walk_forward_mean_accuracy: float = 0.96,
    walk_forward_min_accuracy: float = 0.95,
    walk_forward_accuracy_std: float = 0.03,
    walk_forward_brier: float = 0.12,
    training_samples: int = 1000,
    feature_count: int = 20,
    complexity_score: float = 1.0,
) -> SelectionCandidate:
    return SelectionCandidate(
        model_id=model_id,
        model_name=model_name,
        horizon=horizon,
        validation_accuracy=validation_accuracy,
        walk_forward_mean_accuracy=walk_forward_mean_accuracy,
        walk_forward_min_accuracy=walk_forward_min_accuracy,
        walk_forward_accuracy_std=walk_forward_accuracy_std,
        walk_forward_brier=walk_forward_brier,
        training_samples=training_samples,
        feature_count=feature_count,
        complexity_score=complexity_score,
    )


def test_candidate_construction():
    item = candidate()

    assert item.model_id == "model_1"
    assert item.model_name == "GradientBoosting"
    assert item.horizon == 5
    assert item.final_holdout_used is False if hasattr(
        item, "final_holdout_used"
    ) else True


def test_pipeline_construction():
    pipeline = SelectionPipeline()

    assert pipeline.config is not None
    assert pipeline.selector is not None


def test_custom_config_is_used():
    config = ModelSelectionConfig(
        minimum_validation_accuracy=0.90,
        minimum_walk_forward_accuracy=0.90,
        max_walk_forward_accuracy_std=0.20,
    )

    pipeline = SelectionPipeline(config=config)

    assert pipeline.config == config


def test_invalid_empty_model_id_fails():
    pipeline = SelectionPipeline()

    bad = candidate(model_id="")

    with pytest.raises(ValueError, match="model_id"):
        pipeline.select([bad], horizon=5)


def test_invalid_empty_model_name_fails():
    pipeline = SelectionPipeline()

    bad = candidate(model_name="")

    with pytest.raises(ValueError, match="model_name"):
        pipeline.select([bad], horizon=5)


def test_invalid_horizon_fails():
    pipeline = SelectionPipeline()

    bad = candidate(horizon=0)

    with pytest.raises(ValueError, match="horizon"):
        pipeline.select([bad], horizon=5)


def test_negative_walk_forward_std_fails():
    pipeline = SelectionPipeline()

    bad = candidate(walk_forward_accuracy_std=-0.01)

    with pytest.raises(ValueError, match="std"):
        pipeline.select([bad], horizon=5)


def test_accuracy_outside_probability_range_fails():
    pipeline = SelectionPipeline()

    bad = candidate(validation_accuracy=1.10)

    with pytest.raises(ValueError, match="between 0 and 1"):
        pipeline.select([bad], horizon=5)


def test_negative_training_samples_fail():
    pipeline = SelectionPipeline()

    bad = candidate(training_samples=-1)

    with pytest.raises(ValueError, match="training_samples"):
        pipeline.select([bad], horizon=5)


def test_empty_candidate_list_fails():
    pipeline = SelectionPipeline()

    with pytest.raises(ValueError, match="candidate"):
        pipeline.select([], horizon=5)


def test_candidate_horizon_must_match_requested_horizon():
    pipeline = SelectionPipeline()

    bad = candidate(horizon=10)

    with pytest.raises(ValueError, match="horizon"):
        pipeline.select([bad], horizon=5)


def test_successful_selection_returns_result():
    pipeline = SelectionPipeline()

    candidates = [
        candidate(
            model_id="rf",
            model_name="RandomForest",
            validation_accuracy=0.96,
            walk_forward_mean_accuracy=0.95,
            walk_forward_min_accuracy=0.94,
            walk_forward_accuracy_std=0.04,
        ),
        candidate(
            model_id="gb",
            model_name="GradientBoosting",
            validation_accuracy=0.98,
            walk_forward_mean_accuracy=0.97,
            walk_forward_min_accuracy=0.96,
            walk_forward_accuracy_std=0.02,
        ),
    ]

    result = pipeline.select(candidates, horizon=5)

    assert isinstance(result, SelectionPipelineResult)
    assert result.horizon == 5
    assert len(result.candidates) == 2
    assert result.final_holdout_used is False
    assert result.production_approved is False


def test_selected_model_is_one_of_candidates():
    pipeline = SelectionPipeline()

    candidates = [
        candidate(model_id="rf", model_name="RandomForest"),
        candidate(model_id="gb", model_name="GradientBoosting"),
    ]

    result = pipeline.select(candidates, horizon=5)

    if result.selected_model_id is not None:
        assert result.selected_model_id in {"rf", "gb"}


def test_no_candidate_can_be_selected_if_all_fail():
    config = ModelSelectionConfig(
        minimum_validation_accuracy=0.99,
        minimum_walk_forward_accuracy=0.99,
        max_walk_forward_accuracy_std=0.01,
    )

    pipeline = SelectionPipeline(config=config)

    candidates = [
        candidate(
            validation_accuracy=0.80,
            walk_forward_mean_accuracy=0.80,
            walk_forward_min_accuracy=0.75,
            walk_forward_accuracy_std=0.20,
        )
    ]

    result = pipeline.select(candidates, horizon=5)

    assert result.selected_model_id is None
    assert result.successful is False
    assert result.final_holdout_used is False
    assert result.production_approved is False
    assert result.warnings


def test_final_holdout_is_never_used():
    pipeline = SelectionPipeline()

    candidates = [
        candidate(model_id="a"),
        candidate(model_id="b"),
    ]

    result = pipeline.select(candidates, horizon=5)

    assert result.final_holdout_used is False
    assert result.metadata["final_holdout_used"] is False
    assert result.metadata["research_only"] is True


def test_selection_does_not_mean_production_approval():
    pipeline = SelectionPipeline()

    result = pipeline.select(
        [candidate()],
        horizon=5,
    )

    assert result.selected_model_id is not None
    assert result.production_approved is False


def test_selected_candidate_property():
    pipeline = SelectionPipeline()

    candidates = [
        candidate(
            model_id="rf",
            model_name="RandomForest",
        ),
        candidate(
            model_id="gb",
            model_name="GradientBoosting",
            validation_accuracy=0.99,
            walk_forward_mean_accuracy=0.99,
            walk_forward_min_accuracy=0.98,
            walk_forward_accuracy_std=0.01,
        ),
    ]

    result = pipeline.select(candidates, horizon=5)

    selected = result.selected_candidate

    if result.selected_model_id is not None:
        assert selected is not None
        assert selected.model_id == result.selected_model_id


def test_summary_contains_core_information():
    pipeline = SelectionPipeline()

    result = pipeline.select(
        [candidate()],
        horizon=5,
    )

    summary = result.summary()

    assert summary["horizon"] == 5
    assert summary["candidate_count"] == 1
    assert "selected_model_id" in summary
    assert summary["final_holdout_used"] is False
    assert summary["production_approved"] is False


def test_multi_horizon_selection():
    pipeline = SelectionPipeline()

    candidates = [
        candidate(
            model_id="model_1d",
            model_name="GB_1D",
            horizon=1,
        ),
        candidate(
            model_id="model_5d",
            model_name="GB_5D",
            horizon=5,
        ),
        candidate(
            model_id="model_10d",
            model_name="RF_10D",
            horizon=10,
        ),
    ]

    results = pipeline.select_for_horizons(
        candidates=candidates,
        horizons=[1, 5, 10],
    )

    assert set(results.keys()) == {1, 5, 10}

    for horizon, result in results.items():
        assert result.horizon == horizon
        assert result.final_holdout_used is False


def test_missing_horizon_returns_safe_empty_result():
    pipeline = SelectionPipeline()

    candidates = [
        candidate(
            model_id="model_5d",
            horizon=5,
        )
    ]

    results = pipeline.select_for_horizons(
        candidates=candidates,
        horizons=[1, 5],
    )

    assert results[1].selected_model_id is None
    assert results[1].successful is False
    assert results[1].final_holdout_used is False

    assert results[5].horizon == 5


def test_duplicate_horizons_fail():
    pipeline = SelectionPipeline()

    with pytest.raises(ValueError, match="Duplicate"):
        pipeline.select_for_horizons(
            candidates=[candidate()],
            horizons=[5, 5],
        )


def test_cross_horizon_comparison():
    pipeline = SelectionPipeline()

    candidates = [
        candidate(model_id="m1", horizon=1),
        candidate(model_id="m5", horizon=5),
    ]

    results = pipeline.select_for_horizons(
        candidates=candidates,
        horizons=[1, 5],
    )

    rows = pipeline.compare_selected_models(results)

    assert len(rows) == 2
    assert rows[0]["horizon"] == 1
    assert rows[1]["horizon"] == 5


def test_convenience_api():
    result = run_model_selection(
        candidates=[candidate()],
        horizon=5,
    )

    assert isinstance(result, SelectionPipelineResult)
    assert result.horizon == 5
    assert result.final_holdout_used is False


def test_input_candidates_are_not_mutated():
    pipeline = SelectionPipeline()

    candidates = [
        candidate(model_id="a"),
        candidate(model_id="b"),
    ]

    original = copy.deepcopy(candidates)

    pipeline.select(candidates, horizon=5)

    assert candidates == original


def test_metadata_marks_pipeline_as_research_only():
    pipeline = SelectionPipeline()

    result = pipeline.select(
        [candidate()],
        horizon=5,
    )

    assert result.metadata["research_only"] is True
    assert result.metadata["selection_stage_only"] is True
    assert result.metadata["production_approval_required"] is True


def test_deterministic_selection():
    candidates = [
        candidate(
            model_id="rf",
            model_name="RandomForest",
            validation_accuracy=0.96,
            walk_forward_mean_accuracy=0.95,
            walk_forward_min_accuracy=0.94,
            walk_forward_accuracy_std=0.04,
        ),
        candidate(
            model_id="gb",
            model_name="GradientBoosting",
            validation_accuracy=0.98,
            walk_forward_mean_accuracy=0.97,
            walk_forward_min_accuracy=0.96,
            walk_forward_accuracy_std=0.02,
        ),
    ]

    pipeline_1 = SelectionPipeline()
    pipeline_2 = SelectionPipeline()

    result_1 = pipeline_1.select(candidates, horizon=5)
    result_2 = pipeline_2.select(candidates, horizon=5)

    assert result_1.selected_model_id == result_2.selected_model_id


def test_non_finite_metric_fails():
    pipeline = SelectionPipeline()

    bad = candidate(validation_accuracy=float("nan"))

    with pytest.raises(ValueError, match="finite"):
        pipeline.select([bad], horizon=5)


def test_non_numeric_metric_fails():
    pipeline = SelectionPipeline()

    bad = candidate()
    object.__setattr__(bad, "validation_accuracy", "0.97")

    with pytest.raises(TypeError, match="numeric"):
        pipeline.select([bad], horizon=5)
