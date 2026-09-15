"""
Tests for the multi-horizon model selection pipeline.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.research.model_selection import (
    ModelSelectionConfig,
)
from src.multi_horizon_selection_pipeline import (
    HorizonSelectionResult,
    MultiHorizonSelectionConfig,
    MultiHorizonSelectionPipeline,
    MultiHorizonSelectionResult,
    multi_horizon_selection_summary,
    select_multi_horizon_models,
)
from src.research.multi_horizon_training_pipeline import (
    HorizonTrainingResult,
    MultiHorizonTrainingResult,
)


def make_fold():
    model = object()
    preprocessor = object()

    return SimpleNamespace(
        model=model,
        preprocessor=preprocessor,
    )


def make_walk_forward_result(
    *,
    horizon: int,
    candidate_passed: bool = True,
    successful: bool = True,
):
    fold = make_fold()

    return SimpleNamespace(
        folds=[fold],
        feature_columns=[
            "RSI_14",
            "MACD",
            "EMA_20",
            "EMA_50",
        ],
        oos_predictions=SimpleNamespace(),
        candidate_passed=candidate_passed,
        metadata={
            "model_type": "gradient_boosting",
            "final_holdout_used": False,
        },
    )


def make_horizon_result(
    horizon: int,
    *,
    candidate_passed: bool = True,
    successful: bool = True,
):
    walk_forward = make_walk_forward_result(
        horizon=horizon,
        candidate_passed=candidate_passed,
        successful=successful,
    )

    return HorizonTrainingResult(
        horizon=horizon,
        target_column=f"Direction_{horizon}D",
        result=walk_forward,
        successful=successful,
        candidate_passed=candidate_passed,
        mean_accuracy=0.96 if candidate_passed else 0.72,
        minimum_accuracy=0.95 if candidate_passed else 0.65,
        accuracy_std=0.02 if candidate_passed else 0.15,
        warnings=[],
        errors=[],
        metadata={
            "final_holdout_used": False,
            "calibration_fitted": False,
            "research_only": True,
        },
    )


def make_training_result(
    horizons=(1, 3, 5),
    *,
    candidate_passed=True,
    successful=True,
):
    results = {}

    for horizon in horizons:
        results[horizon] = make_horizon_result(
            horizon,
            candidate_passed=candidate_passed,
            successful=successful,
        )

    return MultiHorizonTrainingResult(
        horizons=tuple(horizons),
        results=results,
        successful_horizons=tuple(
            horizons if successful else ()
        ),
        failed_horizons=tuple(
            () if successful else horizons
        ),
        candidate_passed=(
            candidate_passed and successful
        ),
        preferred_horizon=(
            5 if 5 in horizons else horizons[0]
        ),
        warnings=[],
        errors=[],
        metadata={
            "final_holdout_used": False,
            "calibration_fitted": False,
            "model_selected": False,
            "threshold_optimization_completed": False,
            "production_approved": False,
            "production_ready": False,
            "research_only": True,
        },
    )


def make_selection_config(
    horizons=(1, 3, 5),
    minimum_selected_horizons=1,
):
    return MultiHorizonSelectionConfig(
        selection=ModelSelectionConfig(
            minimum_validation_accuracy=0.50,
            minimum_walk_forward_accuracy=0.50,
            maximum_walk_forward_accuracy_std=1.0,
            max_candidates=20,
        ),
        minimum_selected_horizons=(
            minimum_selected_horizons
        ),
    )


def test_config_construction():
    config = make_selection_config()

    assert isinstance(
        config,
        MultiHorizonSelectionConfig,
    )

    assert (
        config.minimum_selected_horizons
        >= 1
    )


def test_default_config_is_safe():
    config = MultiHorizonSelectionConfig()

    assert (
        config.require_candidate_pass
        is True
    )

    assert (
        config.require_walk_forward_evidence
        is True
    )


def test_invalid_minimum_selected_horizons():
    with pytest.raises(ValueError):
        MultiHorizonSelectionConfig(
            minimum_selected_horizons=0
        )


def test_pipeline_construction():
    pipeline = MultiHorizonSelectionPipeline(
        config=make_selection_config()
    )

    assert isinstance(
        pipeline,
        MultiHorizonSelectionPipeline,
    )


def test_training_result_type_is_required():
    pipeline = MultiHorizonSelectionPipeline(
        config=make_selection_config()
    )

    with pytest.raises(TypeError):
        pipeline.select(None)


def test_empty_training_horizons_are_rejected():
    training = make_training_result(
        horizons=(1,)
    )

    training.horizons = ()

    pipeline = MultiHorizonSelectionPipeline(
        config=make_selection_config()
    )

    with pytest.raises(ValueError):
        pipeline.select(training)


def test_final_holdout_usage_is_rejected():
    training = make_training_result(
        horizons=(1, 5)
    )

    training.metadata[
        "final_holdout_used"
    ] = True

    pipeline = MultiHorizonSelectionPipeline(
        config=make_selection_config()
    )

    with pytest.raises(ValueError):
        pipeline.select(training)


def test_selection_returns_correct_result_type():
    training = make_training_result()

    result = select_multi_horizon_models(
        training,
        config=make_selection_config(),
    )

    assert isinstance(
        result,
        MultiHorizonSelectionResult,
    )


def test_requested_horizons_are_preserved():
    horizons = (1, 3, 5, 10, 20)

    training = make_training_result(
        horizons=horizons
    )

    result = select_multi_horizon_models(
        training,
        config=make_selection_config(),
    )

    assert (
        result.horizons
        == horizons
    )


def test_each_horizon_is_selected_independently():
    training = make_training_result(
        horizons=(1, 3, 5)
    )

    result = select_multi_horizon_models(
        training,
        config=make_selection_config(),
    )

    assert set(
        result.results.keys()
    ) == {1, 3, 5}

    for horizon in (1, 3, 5):
        assert (
            result.results[horizon].horizon
            == horizon
        )


def test_candidate_passing_horizon_can_be_selected():
    training = make_training_result(
        horizons=(5,),
        candidate_passed=True,
    )

    result = select_multi_horizon_models(
        training,
        config=make_selection_config(
            horizons=(5,)
        ),
    )

    selection = result.get(5)

    assert (
        selection.candidate_passed
        is True
    )


def test_failed_candidate_is_not_selected():
    training = make_training_result(
        horizons=(5,),
        candidate_passed=False,
    )

    result = select_multi_horizon_models(
        training,
        config=make_selection_config(
            horizons=(5,)
        ),
    )

    selection = result.get(5)

    assert (
        selection.selected
        is False
    )

    assert (
        selection.selected_model_id
        is None
    )


def test_failed_horizon_is_rejected():
    training = make_training_result(
        horizons=(1, 5),
        successful=True,
    )

    training.results[5] = make_horizon_result(
        5,
        candidate_passed=False,
        successful=False,
    )

    training.successful_horizons = (1,)
    training.failed_horizons = (5,)

    result = select_multi_horizon_models(
        training,
        config=make_selection_config(),
    )

    assert 5 in (
        result.rejected_horizons
    )


def test_selected_and_rejected_horizons_do_not_overlap():
    training = make_training_result(
        horizons=(1, 3, 5)
    )

    training.results[5] = make_horizon_result(
        5,
        candidate_passed=False,
        successful=True,
    )

    result = select_multi_horizon_models(
        training,
        config=make_selection_config(),
    )

    assert not (
        set(result.selected_horizons)
        & set(result.rejected_horizons)
    )


def test_selected_count_matches_selected_horizons():
    training = make_training_result()

    result = select_multi_horizon_models(
        training,
        config=make_selection_config(),
    )

    assert (
        result.selected_count
        == len(result.selected_horizons)
    )


def test_rejected_count_matches_rejected_horizons():
    training = make_training_result()

    result = select_multi_horizon_models(
        training,
        config=make_selection_config(),
    )

    assert (
        result.rejected_count
        == len(result.rejected_horizons)
    )


def test_get_returns_horizon_result():
    training = make_training_result(
        horizons=(1, 5)
    )

    result = select_multi_horizon_models(
        training,
        config=make_selection_config(),
    )

    horizon_result = result.get(5)

    assert isinstance(
        horizon_result,
        HorizonSelectionResult,
    )

    assert (
        horizon_result.horizon
        == 5
    )


def test_get_unknown_horizon_fails():
    training = make_training_result(
        horizons=(1,)
    )

    result = select_multi_horizon_models(
        training,
        config=make_selection_config(),
    )

    with pytest.raises(KeyError):
        result.get(20)


def test_selection_target_column_is_preserved():
    training = make_training_result(
        horizons=(5,)
    )

    result = select_multi_horizon_models(
        training,
        config=make_selection_config(),
    )

    assert (
        result.get(5).target_column
        == "Direction_5D"
    )


def test_final_holdout_is_never_used():
    training = make_training_result(
        horizons=(1, 3, 5)
    )

    result = select_multi_horizon_models(
        training,
        config=make_selection_config(),
    )

    assert (
        result.final_holdout_used
        is False
    )

    assert (
        result.metadata[
            "final_holdout_used"
        ]
        is False
    )

    for horizon_result in (
        result.results.values()
    ):
        assert (
            horizon_result.metadata[
                "final_holdout_used"
            ]
            is False
        )


def test_selection_does_not_calibrate():
    training = make_training_result(
        horizons=(1, 5)
    )

    result = select_multi_horizon_models(
        training,
        config=make_selection_config(),
    )

    assert (
        result.metadata[
            "calibration_fitted"
        ]
        is False
    )


def test_selection_does_not_optimize_thresholds():
    training = make_training_result(
        horizons=(1, 5)
    )

    result = select_multi_horizon_models(
        training,
        config=make_selection_config(),
    )

    assert (
        result.metadata[
            "threshold_optimization_completed"
        ]
        is False
    )


def test_selection_does_not_grant_production_approval():
    training = make_training_result(
        horizons=(1, 5)
    )

    result = select_multi_horizon_models(
        training,
        config=make_selection_config(),
    )

    assert (
        result.production_ready
        is False
    )

    assert (
        result.metadata[
            "production_approved"
        ]
        is False
    )


def test_selection_remains_research_only():
    training = make_training_result(
        horizons=(1, 3, 5)
    )

    result = select_multi_horizon_models(
        training,
        config=make_selection_config(),
    )

    assert (
        result.metadata[
            "research_only"
        ]
        is True
    )


def test_minimum_selected_horizons_gate():
    training = make_training_result(
        horizons=(1, 3, 5)
    )

    config = make_selection_config(
        minimum_selected_horizons=2
    )

    result = select_multi_horizon_models(
        training,
        config=config,
    )

    expected = (
        result.selected_count
        >= config.minimum_selected_horizons
    )

    assert (
        result.candidate_passed
        == expected
    )


def test_selection_summary_type():
    training = make_training_result(
        horizons=(1, 5)
    )

    result = select_multi_horizon_models(
        training,
        config=make_selection_config(),
    )

    summary = multi_horizon_selection_summary(
        result
    )

    assert isinstance(
        summary,
        dict,
    )


def test_selection_summary_contains_horizons():
    training = make_training_result(
        horizons=(1, 3, 5)
    )

    result = select_multi_horizon_models(
        training,
        config=make_selection_config(),
    )

    summary = multi_horizon_selection_summary(
        result
    )

    assert (
        summary["horizons"]
        == [1, 3, 5]
    )


def test_selection_summary_contains_results():
    training = make_training_result(
        horizons=(1, 5)
    )

    result = select_multi_horizon_models(
        training,
        config=make_selection_config(),
    )

    summary = multi_horizon_selection_summary(
        result
    )

    assert (
        "horizon_results"
        in summary
    )

    assert (
        "1"
        in summary["horizon_results"]
    )

    assert (
        "5"
        in summary["horizon_results"]
    )


def test_selection_summary_is_research_only():
    training = make_training_result(
        horizons=(1, 5)
    )

    result = select_multi_horizon_models(
        training,
        config=make_selection_config(),
    )

    summary = multi_horizon_selection_summary(
        result
    )

    assert (
        summary["research_only"]
        is True
    )

    assert (
        summary["production_ready"]
        is False
    )

    assert (
        summary["final_holdout_used"]
        is False
    )


def test_summary_rejects_wrong_type():
    with pytest.raises(TypeError):
        multi_horizon_selection_summary(
            None
        )


def test_training_input_is_not_mutated():
    training = make_training_result(
        horizons=(1, 3, 5)
    )

    original_horizons = (
        training.horizons
    )

    original_metadata = dict(
        training.metadata
    )

    select_multi_horizon_models(
        training,
        config=make_selection_config(),
    )

    assert (
        training.horizons
        == original_horizons
    )

    assert (
        training.metadata
        == original_metadata
    )


def test_selection_is_deterministic():
    training = make_training_result(
        horizons=(1, 3, 5)
    )

    config = make_selection_config()

    first = select_multi_horizon_models(
        training,
        config=config,
    )

    second = select_multi_horizon_models(
        training,
        config=config,
    )

    assert (
        first.selected_horizons
        == second.selected_horizons
    )

    assert (
        first.rejected_horizons
        == second.rejected_horizons
    )

    assert (
        first.candidate_passed
        == second.candidate_passed
    )


def test_missing_horizon_result_is_fail_closed():
    training = make_training_result(
        horizons=(1, 3)
    )

    del training.results[3]

    result = select_multi_horizon_models(
        training,
        config=make_selection_config(),
    )

    assert 3 in (
        result.rejected_horizons
    )

    assert any(
        "3D" in error
        for error in result.errors
    )


def test_no_candidate_can_still_be_research_only():
    training = make_training_result(
        horizons=(1, 3),
        candidate_passed=False,
    )

    result = select_multi_horizon_models(
        training,
        config=make_selection_config(),
    )

    assert (
        result.production_ready
        is False
    )

    assert (
        result.metadata[
            "research_only"
        ]
        is True
    )
