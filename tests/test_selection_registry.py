"""
Tests for the persistent model-selection registry.
"""

from __future__ import annotations

import json

import pytest

from src.research.model_selection import ModelSelectionConfig
from src.research.selection_pipeline import (
    SelectionCandidate,
    SelectionPipeline,
)
from src.research.selection_registry import (
    SelectionRecord,
    SelectionRegistry,
    register_selection,
)


def make_candidate(
    model_id: str = "model_gb_5d",
    model_name: str = "GradientBoosting",
    horizon: int = 5,
    validation_accuracy: float = 0.97,
    walk_forward_mean_accuracy: float = 0.96,
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


def make_result(
    horizon: int = 5,
):
    pipeline = SelectionPipeline()

    return pipeline.select(
        candidates=[
            make_candidate(
                model_id="model_rf",
                model_name="RandomForest",
                horizon=horizon,
                validation_accuracy=0.96,
                walk_forward_mean_accuracy=0.95,
                walk_forward_min_accuracy=0.94,
                walk_forward_accuracy_std=0.04,
            ),
            make_candidate(
                model_id="model_gb",
                model_name="GradientBoosting",
                horizon=horizon,
                validation_accuracy=0.98,
                walk_forward_mean_accuracy=0.97,
                walk_forward_min_accuracy=0.96,
                walk_forward_accuracy_std=0.02,
            ),
        ],
        horizon=horizon,
    )


def test_registry_construction(tmp_path):
    registry = SelectionRegistry(tmp_path)

    assert registry.root == tmp_path
    assert registry.root.exists()


def test_record_creation(tmp_path):
    registry = SelectionRegistry(tmp_path)
    result = make_result()

    record = registry.create_record(
        result=result,
        symbol="RELIANCE",
        timeframe="1D",
    )

    assert isinstance(record, SelectionRecord)
    assert record.symbol == "RELIANCE"
    assert record.timeframe == "1D"
    assert record.horizon == 5
    assert record.candidate_count == 2
    assert record.research_only is True


def test_record_identity_is_generated(tmp_path):
    registry = SelectionRegistry(tmp_path)

    record = registry.create_record(
        result=make_result(),
        symbol="TCS",
        timeframe="1D",
    )

    assert record.selection_id.startswith("SEL-")
    assert len(record.selection_id) > 4
    assert record.created_at


def test_symbol_and_timeframe_are_normalized(tmp_path):
    registry = SelectionRegistry(tmp_path)

    record = registry.create_record(
        result=make_result(),
        symbol="reliance",
        timeframe="1d",
    )

    assert record.symbol == "RELIANCE"
    assert record.timeframe == "1D"


def test_selected_metrics_are_recorded(tmp_path):
    registry = SelectionRegistry(tmp_path)

    record = registry.create_record(
        result=make_result(),
        symbol="RELIANCE",
        timeframe="1D",
    )

    assert record.selected_model_id is not None
    assert record.validation_accuracy is not None
    assert record.walk_forward_mean_accuracy is not None
    assert record.walk_forward_min_accuracy is not None
    assert record.walk_forward_accuracy_std is not None


def test_save_creates_json_file(tmp_path):
    registry = SelectionRegistry(tmp_path)

    record = registry.create_record(
        result=make_result(),
        symbol="RELIANCE",
        timeframe="1D",
    )

    path = registry.save(record)

    assert path.exists()
    assert path.suffix == ".json"


def test_saved_json_is_valid(tmp_path):
    registry = SelectionRegistry(tmp_path)

    record = registry.create_record(
        result=make_result(),
        symbol="RELIANCE",
        timeframe="1D",
    )

    path = registry.save(record)

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["selection_id"] == record.selection_id
    assert payload["symbol"] == "RELIANCE"
    assert payload["horizon"] == 5


def test_save_does_not_overwrite_existing_record(tmp_path):
    registry = SelectionRegistry(tmp_path)

    record = registry.create_record(
        result=make_result(),
        symbol="RELIANCE",
        timeframe="1D",
    )

    registry.save(record)

    with pytest.raises(FileExistsError):
        registry.save(record)


def test_load_saved_record(tmp_path):
    registry = SelectionRegistry(tmp_path)

    original = registry.create_record(
        result=make_result(),
        symbol="RELIANCE",
        timeframe="1D",
    )

    registry.save(original)

    loaded = registry.load(original.selection_id)

    assert loaded == original


def test_missing_record_fails(tmp_path):
    registry = SelectionRegistry(tmp_path)

    with pytest.raises(FileNotFoundError):
        registry.load("SEL-does-not-exist")


def test_register_creates_and_saves_record(tmp_path):
    registry = SelectionRegistry(tmp_path)

    record = registry.register(
        result=make_result(),
        symbol="TCS",
        timeframe="1D",
    )

    assert isinstance(record, SelectionRecord)
    assert (tmp_path / f"{record.selection_id}.json").exists()


def test_list_records(tmp_path):
    registry = SelectionRegistry(tmp_path)

    record_1 = registry.register(
        result=make_result(horizon=1),
        symbol="TCS",
        timeframe="1D",
    )

    record_2 = registry.register(
        result=make_result(horizon=5),
        symbol="TCS",
        timeframe="1D",
    )

    records = registry.list_records()

    ids = {record.selection_id for record in records}

    assert record_1.selection_id in ids
    assert record_2.selection_id in ids
    assert len(records) == 2


def test_find_by_symbol(tmp_path):
    registry = SelectionRegistry(tmp_path)

    registry.register(
        result=make_result(),
        symbol="TCS",
        timeframe="1D",
    )

    registry.register(
        result=make_result(),
        symbol="RELIANCE",
        timeframe="1D",
    )

    records = registry.find(symbol="TCS")

    assert len(records) == 1
    assert records[0].symbol == "TCS"


def test_find_by_timeframe(tmp_path):
    registry = SelectionRegistry(tmp_path)

    registry.register(
        result=make_result(),
        symbol="TCS",
        timeframe="1D",
    )

    registry.register(
        result=make_result(),
        symbol="TCS",
        timeframe="4H",
    )

    records = registry.find(timeframe="4H")

    assert len(records) == 1
    assert records[0].timeframe == "4H"


def test_find_by_horizon(tmp_path):
    registry = SelectionRegistry(tmp_path)

    registry.register(
        result=make_result(horizon=1),
        symbol="TCS",
        timeframe="1D",
    )

    registry.register(
        result=make_result(horizon=5),
        symbol="TCS",
        timeframe="1D",
    )

    records = registry.find(horizon=5)

    assert len(records) == 1
    assert records[0].horizon == 5


def test_find_by_selected_model(tmp_path):
    registry = SelectionRegistry(tmp_path)

    record = registry.register(
        result=make_result(),
        symbol="TCS",
        timeframe="1D",
    )

    records = registry.find(
        selected_model_id=record.selected_model_id,
    )

    assert len(records) == 1
    assert records[0].selection_id == record.selection_id


def test_latest_returns_matching_record(tmp_path):
    registry = SelectionRegistry(tmp_path)

    record = registry.register(
        result=make_result(horizon=5),
        symbol="TCS",
        timeframe="1D",
    )

    latest = registry.latest(
        symbol="TCS",
        timeframe="1D",
        horizon=5,
    )

    assert latest is not None
    assert latest.selection_id == record.selection_id


def test_latest_returns_none_when_missing(tmp_path):
    registry = SelectionRegistry(tmp_path)

    latest = registry.latest(
        symbol="TCS",
        timeframe="1D",
        horizon=20,
    )

    assert latest is None


def test_selection_never_marks_production_approved(tmp_path):
    registry = SelectionRegistry(tmp_path)

    record = registry.register(
        result=make_result(),
        symbol="RELIANCE",
        timeframe="1D",
    )

    assert record.production_approved is False


def test_selection_never_uses_final_holdout(tmp_path):
    registry = SelectionRegistry(tmp_path)

    record = registry.register(
        result=make_result(),
        symbol="RELIANCE",
        timeframe="1D",
    )

    assert record.final_holdout_used is False


def test_tampered_holdout_record_is_rejected(tmp_path):
    registry = SelectionRegistry(tmp_path)

    record = registry.create_record(
        result=make_result(),
        symbol="RELIANCE",
        timeframe="1D",
    )

    path = registry.save(record)

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    payload["final_holdout_used"] = True

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    with pytest.raises(ValueError, match="final holdout"):
        registry.load(record.selection_id)


def test_tampered_approval_record_is_rejected(tmp_path):
    registry = SelectionRegistry(tmp_path)

    record = registry.create_record(
        result=make_result(),
        symbol="RELIANCE",
        timeframe="1D",
    )

    path = registry.save(record)

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    payload["production_approved"] = True

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    with pytest.raises(ValueError, match="production-approved"):
        registry.load(record.selection_id)


def test_invalid_symbol_fails(tmp_path):
    registry = SelectionRegistry(tmp_path)

    with pytest.raises(ValueError, match="symbol"):
        registry.create_record(
            result=make_result(),
            symbol="",
            timeframe="1D",
        )


def test_invalid_timeframe_fails(tmp_path):
    registry = SelectionRegistry(tmp_path)

    with pytest.raises(ValueError, match="timeframe"):
        registry.create_record(
            result=make_result(),
            symbol="TCS",
            timeframe="",
        )


def test_delete_record(tmp_path):
    registry = SelectionRegistry(tmp_path)

    record = registry.register(
        result=make_result(),
        symbol="TCS",
        timeframe="1D",
    )

    path = tmp_path / f"{record.selection_id}.json"

    assert path.exists()

    registry.delete(record.selection_id)

    assert not path.exists()


def test_delete_missing_record_fails(tmp_path):
    registry = SelectionRegistry(tmp_path)

    with pytest.raises(FileNotFoundError):
        registry.delete("SEL-missing")


def test_registry_summary(tmp_path):
    registry = SelectionRegistry(tmp_path)

    registry.register(
        result=make_result(),
        symbol="TCS",
        timeframe="1D",
    )

    registry.register(
        result=make_result(horizon=10),
        symbol="RELIANCE",
        timeframe="1D",
    )

    summary = registry.summary()

    assert summary["total_records"] == 2
    assert summary["successful_selections"] == 2
    assert summary["unsuccessful_selections"] == 0
    assert summary["final_holdout_used"] is False
    assert summary["production_approved_records"] == 0


def test_convenience_registration_api(tmp_path):
    result = make_result()

    record = register_selection(
        result=result,
        symbol="TCS",
        timeframe="1D",
        root=tmp_path,
    )

    assert isinstance(record, SelectionRecord)
    assert (tmp_path / f"{record.selection_id}.json").exists()


def test_research_only_metadata_is_preserved(tmp_path):
    registry = SelectionRegistry(tmp_path)

    record = registry.register(
        result=make_result(),
        symbol="TCS",
        timeframe="1D",
    )

    assert record.research_only is True
    assert record.metadata["research_only"] is True


def test_registry_does_not_modify_result(tmp_path):
    registry = SelectionRegistry(tmp_path)

    result = make_result()

    before = result.summary().copy()

    registry.register(
        result=result,
        symbol="TCS",
        timeframe="1D",
    )

    after = result.summary().copy()

    assert before == after


def test_multiple_horizons_are_kept_separate(tmp_path):
    registry = SelectionRegistry(tmp_path)

    records = []

    for horizon in [1, 3, 5, 10, 20]:
        records.append(
            registry.register(
                result=make_result(horizon=horizon),
                symbol="TCS",
                timeframe="1D",
            )
        )

    stored = registry.find(
        symbol="TCS",
        timeframe="1D",
    )

    assert len(stored) == 5
    assert {record.horizon for record in stored} == {
        1,
        3,
        5,
        10,
        20,
    }


def test_invalid_record_payload_missing_required_field(tmp_path):
    registry = SelectionRegistry(tmp_path)

    path = tmp_path / "SEL-invalid.json"

    payload = {
        "selection_id": "SEL-invalid",
        "created_at": "2026-01-01T00:00:00+00:00",
    }

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    with pytest.raises(ValueError, match="missing fields"):
        registry.load("SEL-invalid")
