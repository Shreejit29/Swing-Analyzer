"""
Tests for the research model-card layer.
"""

from __future__ import annotations

import json

import pytest

from src.research.model_card import (
    ModelCard,
    ModelCardBuilder,
    load_model_card,
    model_card_from_selection,
    model_card_summary,
    save_model_card,
)
from src.research.selection_pipeline import SelectionPipeline


def make_selection_result():
    pipeline = SelectionPipeline()

    return pipeline.select(
        candidates=[
            __import__(
                "src.research.selection_pipeline",
                fromlist=["SelectionCandidate"],
            ).SelectionCandidate(
                model_id="model_rf",
                model_name="RandomForest",
                horizon=5,
                validation_accuracy=0.96,
                walk_forward_mean_accuracy=0.95,
                walk_forward_min_accuracy=0.94,
                walk_forward_accuracy_std=0.04,
                walk_forward_brier=0.15,
                training_samples=1000,
                feature_count=20,
                complexity_score=1.0,
            ),
            __import__(
                "src.research.selection_pipeline",
                fromlist=["SelectionCandidate"],
            ).SelectionCandidate(
                model_id="model_gb",
                model_name="GradientBoosting",
                horizon=5,
                validation_accuracy=0.98,
                walk_forward_mean_accuracy=0.97,
                walk_forward_min_accuracy=0.96,
                walk_forward_accuracy_std=0.02,
                walk_forward_brier=0.12,
                training_samples=1000,
                feature_count=18,
                complexity_score=1.0,
            ),
        ],
        horizon=5,
    )


def make_builder():
    return ModelCardBuilder(
        model_id="model_gb_5d",
        model_name="GradientBoosting",
        symbol="RELIANCE",
        timeframe="1D",
        horizon=5,
        target="Direction_5",
        created_at="2026-09-09T12:00:00+00:00",
    )


def test_model_card_construction():
    builder = make_builder()

    card = builder.build()

    assert isinstance(card, ModelCard)
    assert card.model_id == "model_gb_5d"
    assert card.model_name == "GradientBoosting"
    assert card.symbol == "RELIANCE"
    assert card.timeframe == "1D"
    assert card.horizon == 5
    assert card.target == "Direction_5"


def test_model_card_defaults_to_research_only():
    card = make_builder().build()

    assert card.research_only is True
    assert card.production_approved is False


def test_identity_validation():
    with pytest.raises(ValueError, match="model_id"):
        ModelCardBuilder(
            model_id="",
            model_name="GB",
            symbol="TCS",
            timeframe="1D",
            horizon=5,
            target="Direction_5",
            created_at="2026-01-01",
        )


def test_model_name_validation():
    with pytest.raises(ValueError, match="model_name"):
        ModelCardBuilder(
            model_id="model",
            model_name="",
            symbol="TCS",
            timeframe="1D",
            horizon=5,
            target="Direction_5",
            created_at="2026-01-01",
        )


def test_symbol_validation():
    with pytest.raises(ValueError, match="symbol"):
        ModelCardBuilder(
            model_id="model",
            model_name="GB",
            symbol="",
            timeframe="1D",
            horizon=5,
            target="Direction_5",
            created_at="2026-01-01",
        )


def test_timeframe_validation():
    with pytest.raises(ValueError, match="timeframe"):
        ModelCardBuilder(
            model_id="model",
            model_name="GB",
            symbol="TCS",
            timeframe="",
            horizon=5,
            target="Direction_5",
            created_at="2026-01-01",
        )


def test_horizon_validation():
    with pytest.raises(ValueError, match="horizon"):
        ModelCardBuilder(
            model_id="model",
            model_name="GB",
            symbol="TCS",
            timeframe="1D",
            horizon=0,
            target="Direction_5",
            created_at="2026-01-01",
        )


def test_target_validation():
    with pytest.raises(ValueError, match="target"):
        ModelCardBuilder(
            model_id="model",
            model_name="GB",
            symbol="TCS",
            timeframe="1D",
            horizon=5,
            target="",
            created_at="2026-01-01",
        )


def test_validation_evidence():
    card = (
        make_builder()
        .add_validation(
            accuracy=0.97,
            passed=True,
        )
        .build()
    )

    assert card.validation_accuracy == 0.97
    assert card.validation_passed is True


def test_holdout_evidence():
    card = (
        make_builder()
        .add_holdout(
            accuracy=0.96,
            passed=True,
        )
        .build()
    )

    assert card.holdout_accuracy == 0.96
    assert card.holdout_passed is True


def test_walk_forward_evidence():
    card = (
        make_builder()
        .add_walk_forward(
            mean_accuracy=0.96,
            min_accuracy=0.94,
            accuracy_std=0.03,
        )
        .build()
    )

    assert card.walk_forward_mean_accuracy == 0.96
    assert card.walk_forward_min_accuracy == 0.94
    assert card.walk_forward_accuracy_std == 0.03


def test_calibration_evidence():
    card = (
        make_builder()
        .add_calibration(
            brier=0.12,
            ece=0.08,
            passed=True,
        )
        .build()
    )

    assert card.calibration_brier == 0.12
    assert card.calibration_ece == 0.08
    assert card.calibration_passed is True


def test_range_evidence():
    card = (
        make_builder()
        .add_range_validation(
            coverage=0.81,
            passed=True,
        )
        .build()
    )

    assert card.range_coverage == 0.81
    assert card.range_validation_passed is True


def test_regime_evidence():
    card = (
        make_builder()
        .add_regime_validation(
            stability=0.74,
            passed=True,
        )
        .build()
    )

    assert card.regime_stability == 0.74
    assert card.regime_validation_passed is True


def test_backtest_evidence():
    card = (
        make_builder()
        .add_backtest(
            profit_factor=1.65,
            sharpe_ratio=1.10,
            max_drawdown=0.18,
            trade_count=75,
            passed=True,
        )
        .build()
    )

    assert card.profit_factor == 1.65
    assert card.sharpe_ratio == 1.10
    assert card.max_drawdown == 0.18
    assert card.trade_count == 75
    assert card.backtest_passed is True


def test_robustness_evidence():
    card = (
        make_builder()
        .add_robustness(
            score=0.78,
            passed=True,
        )
        .build()
    )

    assert card.robustness_score == 0.78
    assert card.robustness_passed is True


def test_leakage_status():
    card = (
        make_builder()
        .add_leakage_status(True)
        .build()
    )

    assert card.leakage_free is True


def test_model_size():
    card = (
        make_builder()
        .add_model_size(
            feature_count=25,
            training_samples=5000,
        )
        .build()
    )

    assert card.feature_count == 25
    assert card.training_samples == 5000


def test_limitations_are_recorded():
    card = (
        make_builder()
        .add_limitation("Market regime may change.")
        .add_limitation("Historical performance is not guaranteed.")
        .build()
    )

    assert len(card.limitations) == 2
    assert "Market regime may change." in card.limitations


def test_warnings_are_recorded():
    card = (
        make_builder()
        .add_warning("Final holdout not evaluated.")
        .build()
    )

    assert len(card.warnings) == 1
    assert "Final holdout not evaluated." in card.warnings


def test_metadata_is_recorded():
    card = (
        make_builder()
        .add_metadata(
            source="walk_forward_research",
            experiment_id="EXP-001",
        )
        .build()
    )

    assert card.metadata["source"] == "walk_forward_research"
    assert card.metadata["experiment_id"] == "EXP-001"


def test_approval_defaults_to_false():
    card = make_builder().build()

    assert card.production_approved is False


def test_explicit_approval_can_be_documented():
    card = (
        make_builder()
        .mark_production_approved(True)
        .build()
    )

    assert card.production_approved is True
    assert card.research_only is False


def test_selection_record_creates_research_model_card():
    result = make_selection_result()

    selected = result.selected_model_id

    assert selected is not None

    record = type(
        "SelectionRecord",
        (),
        {
            "selected_model_id": selected,
            "selected_model_name": result.selected_model_name,
            "symbol": "RELIANCE",
            "timeframe": "1D",
            "horizon": 5,
            "created_at": "2026-09-09T12:00:00+00:00",
            "validation_accuracy": (
                result.selected_candidate.validation_accuracy
            ),
            "walk_forward_mean_accuracy": (
                result.selected_candidate.walk_forward_mean_accuracy
            ),
            "walk_forward_min_accuracy": (
                result.selected_candidate.walk_forward_min_accuracy
            ),
            "walk_forward_accuracy_std": (
                result.selected_candidate.walk_forward_accuracy_std
            ),
            "walk_forward_brier": (
                result.selected_candidate.walk_forward_brier
            ),
            "final_holdout_used": False,
            "selection_id": "SEL-test",
        },
    )()

    card = model_card_from_selection(
        selection_record=record,
        target="Direction_5",
    )

    assert card.model_id == selected
    assert card.symbol == "RELIANCE"
    assert card.timeframe == "1D"
    assert card.horizon == 5
    assert card.research_only is True
    assert card.production_approved is False
    assert card.metadata["selection_id"] == "SEL-test"


def test_selection_model_card_explicitly_warns_about_holdout():
    result = make_selection_result()

    selected = result.selected_candidate

    record = type(
        "SelectionRecord",
        (),
        {
            "selected_model_id": result.selected_model_id,
            "selected_model_name": result.selected_model_name,
            "symbol": "TCS",
            "timeframe": "1D",
            "horizon": 5,
            "created_at": "2026-09-09T12:00:00+00:00",
            "validation_accuracy": selected.validation_accuracy,
            "walk_forward_mean_accuracy": (
                selected.walk_forward_mean_accuracy
            ),
            "walk_forward_min_accuracy": (
                selected.walk_forward_min_accuracy
            ),
            "walk_forward_accuracy_std": (
                selected.walk_forward_accuracy_std
            ),
            "walk_forward_brier": selected.walk_forward_brier,
            "final_holdout_used": False,
            "selection_id": "SEL-test-2",
        },
    )()

    card = model_card_from_selection(
        selection_record=record,
        target="Direction_5",
    )

    assert any(
        "holdout" in warning.lower()
        for warning in card.warnings
    )


def test_selection_card_never_claims_production_approval():
    result = make_selection_result()

    selected = result.selected_candidate

    record = type(
        "SelectionRecord",
        (),
        {
            "selected_model_id": result.selected_model_id,
            "selected_model_name": result.selected_model_name,
            "symbol": "TCS",
            "timeframe": "1D",
            "horizon": 5,
            "created_at": "2026-09-09T12:00:00+00:00",
            "validation_accuracy": selected.validation_accuracy,
            "walk_forward_mean_accuracy": (
                selected.walk_forward_mean_accuracy
            ),
            "walk_forward_min_accuracy": (
                selected.walk_forward_min_accuracy
            ),
            "walk_forward_accuracy_std": (
                selected.walk_forward_accuracy_std
            ),
            "walk_forward_brier": selected.walk_forward_brier,
            "final_holdout_used": False,
            "selection_id": "SEL-test-3",
        },
    )()

    card = model_card_from_selection(
        selection_record=record,
        target="Direction_5",
    )

    assert card.production_approved is False
    assert card.research_only is True


def test_save_model_card(tmp_path):
    card = (
        make_builder()
        .add_validation(0.97, True)
        .build()
    )

    path = save_model_card(
        card,
        root=tmp_path,
    )

    assert path.exists()
    assert path.name == "model_gb_5d.json"


def test_saved_model_card_is_valid_json(tmp_path):
    card = make_builder().build()

    path = save_model_card(
        card,
        root=tmp_path,
    )

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["model_id"] == "model_gb_5d"
    assert payload["symbol"] == "RELIANCE"
    assert payload["research_only"] is True


def test_duplicate_model_card_is_not_overwritten(tmp_path):
    card = make_builder().build()

    save_model_card(card, root=tmp_path)

    with pytest.raises(FileExistsError):
        save_model_card(card, root=tmp_path)


def test_load_model_card(tmp_path):
    card = (
        make_builder()
        .add_validation(0.97, True)
        .add_limitation("Regime sensitivity.")
        .add_warning("Research stage.")
        .build()
    )

    save_model_card(card, root=tmp_path)

    loaded = load_model_card(
        "model_gb_5d",
        root=tmp_path,
    )

    assert loaded.model_id == card.model_id
    assert loaded.validation_accuracy == 0.97
    assert loaded.limitations == card.limitations
    assert loaded.warnings == card.warnings


def test_missing_model_card_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model_card(
            "does_not_exist",
            root=tmp_path,
        )


def test_model_card_summary():
    card = (
        make_builder()
        .add_validation(0.97, True)
        .add_holdout(0.96, True)
        .add_walk_forward(0.96, 0.94, 0.03)
        .add_range_validation(0.81, True)
        .add_regime_validation(0.75, True)
        .add_backtest(1.5, 1.0, 0.20, 50, True)
        .add_robustness(0.80, True)
        .build()
    )

    summary = model_card_summary(card)

    assert summary["model_id"] == "model_gb_5d"
    assert summary["validation_accuracy"] == 0.97
    assert summary["holdout_accuracy"] == 0.96
    assert summary["walk_forward_mean_accuracy"] == 0.96
    assert summary["range_coverage"] == 0.81
    assert summary["regime_stability"] == 0.75
    assert summary["profit_factor"] == 1.5
    assert summary["trade_count"] == 50
    assert summary["production_approved"] is False


def test_summary_counts_warnings_and_limitations():
    card = (
        make_builder()
        .add_warning("Warning 1")
        .add_warning("Warning 2")
        .add_limitation("Limitation 1")
        .build()
    )

    summary = model_card_summary(card)

    assert summary["warning_count"] == 2
    assert summary["limitation_count"] == 1


def test_none_selection_record_fails():
    with pytest.raises(ValueError, match="selection_record"):
        model_card_from_selection(
            selection_record=None,
            target="Direction_5",
        )


def test_model_card_is_immutable():
    card = make_builder().build()

    with pytest.raises(
        AttributeError
    ):
        card.model_id = "changed"


def test_card_research_status_survives_save_load(tmp_path):
    card = (
        make_builder()
        .add_validation(0.98, True)
        .build()
    )

    save_model_card(card, root=tmp_path)

    loaded = load_model_card(
        "model_gb_5d",
        root=tmp_path,
    )

    assert loaded.research_only is True
    assert loaded.production_approved is False
