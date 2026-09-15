"""
Tests for the integrated model-selection research stage.
"""

from __future__ import annotations

import json

import pytest

from src.research.model_card import ModelCard
from src.research.selection_pipeline import SelectionCandidate
from src.research.selection_stage import (
    ModelSelectionStage,
    ModelSelectionStageResult,
    run_model_selection_stage,
)


def make_candidate(
    model_id: str = "gb_5d",
    model_name: str = "GradientBoosting",
    horizon: int = 5,
    validation_accuracy: float = 0.98,
    walk_forward_mean_accuracy: float = 0.97,
    walk_forward_min_accuracy: float = 0.95,
    walk_forward_accuracy_std: float = 0.03,
    walk_forward_brier: float = 0.12,
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
        training_samples=1000,
        feature_count=20,
        complexity_score=1.0,
    )


def make_candidates(horizon: int = 5):
    return [
        make_candidate(
            model_id=f"rf_{horizon}d",
            model_name="RandomForest",
            horizon=horizon,
            validation_accuracy=0.96,
            walk_forward_mean_accuracy=0.95,
            walk_forward_min_accuracy=0.93,
            walk_forward_accuracy_std=0.04,
        ),
        make_candidate(
            model_id=f"gb_{horizon}d",
            model_name="GradientBoosting",
            horizon=horizon,
            validation_accuracy=0.98,
            walk_forward_mean_accuracy=0.97,
            walk_forward_min_accuracy=0.96,
            walk_forward_accuracy_std=0.02,
        ),
    ]


def test_stage_construction(tmp_path):
    stage = ModelSelectionStage(
        selection_root=tmp_path / "selections",
        model_card_root=tmp_path / "cards",
    )

    assert stage.config is not None
    assert stage.pipeline is not None
    assert stage.selection_registry is not None


def test_successful_stage_execution(tmp_path):
    stage = ModelSelectionStage(
        selection_root=tmp_path / "selections",
        model_card_root=tmp_path / "cards",
    )

    result = stage.run(
        candidates=make_candidates(),
        symbol="RELIANCE",
        timeframe="1D",
        horizon=5,
        target="Direction_5",
    )

    assert isinstance(result, ModelSelectionStageResult)
    assert result.symbol == "RELIANCE"
    assert result.timeframe == "1D"
    assert result.horizon == 5
    assert result.target == "Direction_5"
    assert result.successful is True


def test_selected_model_is_recorded(tmp_path):
    stage = ModelSelectionStage(
        selection_root=tmp_path / "selections",
        model_card_root=tmp_path / "cards",
    )

    result = stage.run(
        candidates=make_candidates(),
        symbol="TCS",
        timeframe="1D",
        horizon=5,
        target="Direction_5",
    )

    assert result.selected_model_id is not None
    assert result.selected_model_name is not None


def test_selection_record_is_created(tmp_path):
    selection_root = tmp_path / "selections"

    stage = ModelSelectionStage(
        selection_root=selection_root,
        model_card_root=tmp_path / "cards",
    )

    result = stage.run(
        candidates=make_candidates(),
        symbol="TCS",
        timeframe="1D",
        horizon=5,
        target="Direction_5",
    )

    assert result.selection_record is not None
    assert result.selection_record.symbol == "TCS"

    files = list(selection_root.glob("SEL-*.json"))

    assert len(files) == 1


def test_model_card_is_created(tmp_path):
    card_root = tmp_path / "cards"

    stage = ModelSelectionStage(
        selection_root=tmp_path / "selections",
        model_card_root=card_root,
    )

    result = stage.run(
        candidates=make_candidates(),
        symbol="RELIANCE",
        timeframe="1D",
        horizon=5,
        target="Direction_5",
    )

    assert result.model_card is not None
    assert isinstance(result.model_card, ModelCard)
    assert result.model_card.model_id == result.selected_model_id

    card_files = list(card_root.glob("*.json"))

    assert len(card_files) == 1


def test_model_card_path_is_recorded(tmp_path):
    card_root = tmp_path / "cards"

    stage = ModelSelectionStage(
        selection_root=tmp_path / "selections",
        model_card_root=card_root,
    )

    result = stage.run(
        candidates=make_candidates(),
        symbol="RELIANCE",
        timeframe="1D",
        horizon=5,
        target="Direction_5",
    )

    assert result.model_card_path is not None
    assert result.model_card_path.endswith(".json")


def test_selection_is_research_only(tmp_path):
    stage = ModelSelectionStage(
        selection_root=tmp_path / "selections",
        model_card_root=tmp_path / "cards",
    )

    result = stage.run(
        candidates=make_candidates(),
        symbol="TCS",
        timeframe="1D",
        horizon=5,
        target="Direction_5",
    )

    assert result.metadata["research_only"] is True
    assert result.model_card is not None
    assert result.model_card.research_only is True


def test_selection_does_not_approve_model(tmp_path):
    stage = ModelSelectionStage(
        selection_root=tmp_path / "selections",
        model_card_root=tmp_path / "cards",
    )

    result = stage.run(
        candidates=make_candidates(),
        symbol="TCS",
        timeframe="1D",
        horizon=5,
        target="Direction_5",
    )

    assert result.production_approved is False
    assert result.metadata["production_approved"] is False

    assert result.selection_record is not None
    assert result.selection_record.production_approved is False

    assert result.model_card is not None
    assert result.model_card.production_approved is False


def test_final_holdout_is_not_used(tmp_path):
    stage = ModelSelectionStage(
        selection_root=tmp_path / "selections",
        model_card_root=tmp_path / "cards",
    )

    result = stage.run(
        candidates=make_candidates(),
        symbol="TCS",
        timeframe="1D",
        horizon=5,
        target="Direction_5",
    )

    assert result.final_holdout_used is False
    assert result.metadata["final_holdout_used"] is False

    assert result.selection_record is not None
    assert result.selection_record.final_holdout_used is False


def test_no_candidates_fails(tmp_path):
    stage = ModelSelectionStage(
        selection_root=tmp_path / "selections",
        model_card_root=tmp_path / "cards",
    )

    with pytest.raises(ValueError, match="candidate"):
        stage.run(
            candidates=[],
            symbol="TCS",
            timeframe="1D",
            horizon=5,
            target="Direction_5",
        )


def test_duplicate_model_ids_fail(tmp_path):
    stage = ModelSelectionStage(
        selection_root=tmp_path / "selections",
        model_card_root=tmp_path / "cards",
    )

    candidates = [
        make_candidate(model_id="duplicate"),
        make_candidate(model_id="duplicate"),
    ]

    with pytest.raises(ValueError, match="Duplicate"):
        stage.run(
            candidates=candidates,
            symbol="TCS",
            timeframe="1D",
            horizon=5,
            target="Direction_5",
        )


def test_empty_symbol_fails(tmp_path):
    stage = ModelSelectionStage(
        selection_root=tmp_path / "selections",
        model_card_root=tmp_path / "cards",
    )

    with pytest.raises(ValueError, match="symbol"):
        stage.run(
            candidates=make_candidates(),
            symbol="",
            timeframe="1D",
            horizon=5,
            target="Direction_5",
        )


def test_empty_timeframe_fails(tmp_path):
    stage = ModelSelectionStage(
        selection_root=tmp_path / "selections",
        model_card_root=tmp_path / "cards",
    )

    with pytest.raises(ValueError, match="timeframe"):
        stage.run(
            candidates=make_candidates(),
            symbol="TCS",
            timeframe="",
            horizon=5,
            target="Direction_5",
        )


def test_empty_target_fails(tmp_path):
    stage = ModelSelectionStage(
        selection_root=tmp_path / "selections",
        model_card_root=tmp_path / "cards",
    )

    with pytest.raises(ValueError, match="target"):
        stage.run(
            candidates=make_candidates(),
            symbol="TCS",
            timeframe="1D",
            horizon=5,
            target="",
        )


def test_invalid_horizon_fails(tmp_path):
    stage = ModelSelectionStage(
        selection_root=tmp_path / "selections",
        model_card_root=tmp_path / "cards",
    )

    with pytest.raises(ValueError, match="horizon"):
        stage.run(
            candidates=make_candidates(),
            symbol="TCS",
            timeframe="1D",
            horizon=0,
            target="Direction_0",
        )


def test_wrong_candidate_horizon_fails(tmp_path):
    stage = ModelSelectionStage(
        selection_root=tmp_path / "selections",
        model_card_root=tmp_path / "cards",
    )

    with pytest.raises(ValueError, match="horizon"):
        stage.run(
            candidates=make_candidates(horizon=5),
            symbol="TCS",
            timeframe="1D",
            horizon=10,
            target="Direction_10",
        )


def test_without_persistence(tmp_path):
    stage = ModelSelectionStage(
        selection_root=tmp_path / "selections",
        model_card_root=tmp_path / "cards",
    )

    result = stage.run_without_persistence(
        candidates=make_candidates(),
        symbol="TCS",
        timeframe="1D",
        horizon=5,
        target="Direction_5",
    )

    assert result.successful is True
    assert result.selection_record is None
    assert result.model_card is not None
    assert result.model_card_path is None

    assert not list(
        (tmp_path / "selections").glob("*.json")
    )


def test_disable_model_card_creation(tmp_path):
    stage = ModelSelectionStage(
        selection_root=tmp_path / "selections",
        model_card_root=tmp_path / "cards",
    )

    result = stage.run(
        candidates=make_candidates(),
        symbol="TCS",
        timeframe="1D",
        horizon=5,
        target="Direction_5",
        create_model_card=False,
    )

    assert result.successful is True
    assert result.model_card is None
    assert result.model_card_path is None


def test_disable_selection_persistence(tmp_path):
    stage = ModelSelectionStage(
        selection_root=tmp_path / "selections",
        model_card_root=tmp_path / "cards",
    )

    result = stage.run(
        candidates=make_candidates(),
        symbol="TCS",
        timeframe="1D",
        horizon=5,
        target="Direction_5",
        persist_selection=False,
        persist_model_card=False,
    )

    assert result.successful is True
    assert result.selection_record is None
    assert result.model_card is not None
    assert result.model_card_path is None


def test_failed_selection_does_not_create_model_card(tmp_path):
    from src.research.model_selection import ModelSelectionConfig

    config = ModelSelectionConfig(
        minimum_validation_accuracy=0.99,
        minimum_walk_forward_accuracy=0.99,
        max_walk_forward_accuracy_std=0.01,
    )

    stage = ModelSelectionStage(
        config=config,
        selection_root=tmp_path / "selections",
        model_card_root=tmp_path / "cards",
    )

    weak_candidates = [
        make_candidate(
            validation_accuracy=0.80,
            walk_forward_mean_accuracy=0.80,
            walk_forward_min_accuracy=0.75,
            walk_forward_accuracy_std=0.20,
        )
    ]

    result = stage.run(
        candidates=weak_candidates,
        symbol="TCS",
        timeframe="1D",
        horizon=5,
        target="Direction_5",
    )

    assert result.successful is False
    assert result.selected_model_id is None
    assert result.model_card is None


def test_multiple_horizons(tmp_path):
    stage = ModelSelectionStage(
        selection_root=tmp_path / "selections",
        model_card_root=tmp_path / "cards",
    )

    candidates = []

    for horizon in [1, 3, 5, 10, 20]:
        candidates.extend(
            make_candidates(horizon=horizon)
        )

    results = stage.select_multiple_horizons(
        candidates=candidates,
        symbol="RELIANCE",
        timeframe="1D",
        horizons=[1, 3, 5, 10, 20],
    )

    assert set(results.keys()) == {
        1,
        3,
        5,
        10,
        20,
    }

    for horizon, result in results.items():
        assert result.horizon == horizon
        assert result.successful is True
        assert result.final_holdout_used is False
        assert result.production_approved is False


def test_missing_horizon_is_skipped(tmp_path):
    stage = ModelSelectionStage(
        selection_root=tmp_path / "selections",
        model_card_root=tmp_path / "cards",
    )

    results = stage.select_multiple_horizons(
        candidates=make_candidates(horizon=5),
        symbol="TCS",
        timeframe="1D",
        horizons=[1, 5],
    )

    assert 1 not in results
    assert 5 in results


def test_duplicate_horizons_fail(tmp_path):
    stage = ModelSelectionStage(
        selection_root=tmp_path / "selections",
        model_card_root=tmp_path / "cards",
    )

    with pytest.raises(ValueError, match="Duplicate"):
        stage.select_multiple_horizons(
            candidates=make_candidates(),
            symbol="TCS",
            timeframe="1D",
            horizons=[5, 5],
        )


def test_selection_table(tmp_path):
    stage = ModelSelectionStage(
        selection_root=tmp_path / "selections",
        model_card_root=tmp_path / "cards",
    )

    results = stage.select_multiple_horizons(
        candidates=(
            make_candidates(1)
            + make_candidates(5)
        ),
        symbol="TCS",
        timeframe="1D",
        horizons=[1, 5],
        persist_selection=False,
        persist_model_cards=False,
    )

    table = stage.selection_table(results)

    assert len(table) == 2

    assert table[0]["horizon"] == 1
    assert table[1]["horizon"] == 5

    assert "selected_model_id" in table[0]
    assert "walk_forward_mean_accuracy" in table[0]
    assert "production_approved" in table[0]


def test_result_summary(tmp_path):
    stage = ModelSelectionStage(
        selection_root=tmp_path / "selections",
        model_card_root=tmp_path / "cards",
    )

    result = stage.run(
        candidates=make_candidates(),
        symbol="TCS",
        timeframe="1D",
        horizon=5,
        target="Direction_5",
    )

    summary = result.summary()

    assert summary["symbol"] == "TCS"
    assert summary["timeframe"] == "1D"
    assert summary["horizon"] == 5
    assert summary["successful"] is True
    assert summary["final_holdout_used"] is False
    assert summary["production_approved"] is False
    assert summary["model_card_created"] is True


def test_metadata_contains_stage_identity(tmp_path):
    stage = ModelSelectionStage(
        selection_root=tmp_path / "selections",
        model_card_root=tmp_path / "cards",
    )

    result = stage.run(
        candidates=make_candidates(),
        symbol="TCS",
        timeframe="1D",
        horizon=5,
        target="Direction_5",
    )

    assert result.metadata["stage"] == "model_selection"
    assert result.metadata["research_only"] is True
    assert result.metadata["production_approval_required"] is True


def test_persisted_selection_can_be_reloaded(tmp_path):
    selection_root = tmp_path / "selections"

    stage = ModelSelectionStage(
        selection_root=selection_root,
        model_card_root=tmp_path / "cards",
    )

    result = stage.run(
        candidates=make_candidates(),
        symbol="TCS",
        timeframe="1D",
        horizon=5,
        target="Direction_5",
    )

    assert result.selection_record is not None

    loaded = stage.selection_registry.load(
        result.selection_record.selection_id
    )

    assert loaded.selection_id == (
        result.selection_record.selection_id
    )
    assert loaded.selected_model_id == (
        result.selected_model_id
    )


def test_persisted_model_card_is_valid_json(tmp_path):
    card_root = tmp_path / "cards"

    stage = ModelSelectionStage(
        selection_root=tmp_path / "selections",
        model_card_root=card_root,
    )

    result = stage.run(
        candidates=make_candidates(),
        symbol="TCS",
        timeframe="1D",
        horizon=5,
        target="Direction_5",
    )

    assert result.model_card_path is not None

    with open(
        result.model_card_path,
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    assert payload["model_id"] == result.selected_model_id
    assert payload["research_only"] is True
    assert payload["production_approved"] is False


def test_convenience_api(tmp_path):
    result = run_model_selection_stage(
        candidates=make_candidates(),
        symbol="INFY",
        timeframe="1D",
        horizon=5,
        target="Direction_5",
        selection_root=tmp_path / "selections",
        model_card_root=tmp_path / "cards",
    )

    assert isinstance(result, ModelSelectionStageResult)
    assert result.symbol == "INFY"
    assert result.successful is True


def test_stage_does_not_mutate_candidates(tmp_path):
    candidates = make_candidates()

    before = [
        (
            candidate.model_id,
            candidate.validation_accuracy,
            candidate.walk_forward_mean_accuracy,
        )
        for candidate in candidates
    ]

    stage = ModelSelectionStage(
        selection_root=tmp_path / "selections",
        model_card_root=tmp_path / "cards",
    )

    stage.run(
        candidates=candidates,
        symbol="TCS",
        timeframe="1D",
        horizon=5,
        target="Direction_5",
    )

    after = [
        (
            candidate.model_id,
            candidate.validation_accuracy,
            candidate.walk_forward_mean_accuracy,
        )
        for candidate in candidates
    ]

    assert before == after
