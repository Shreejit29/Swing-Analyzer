"""
Tests for the model-card registry.
"""

from __future__ import annotations

import json

import pytest

from src.research.model_card import ModelCard, ModelCardBuilder
from src.research.model_card_registry import (
    ModelCardRegistry,
    register_model_card,
)


def make_card(
    model_id: str = "model_gb_5d",
    symbol: str = "RELIANCE",
    timeframe: str = "1D",
    horizon: int = 5,
    approved: bool = False,
) -> ModelCard:
    builder = ModelCardBuilder(
        model_id=model_id,
        model_name="GradientBoosting",
        symbol=symbol,
        timeframe=timeframe,
        horizon=horizon,
        target=f"Direction_{horizon}",
        created_at="2026-09-09T12:00:00+00:00",
    )

    builder.add_validation(
        accuracy=0.97,
        passed=True,
    )

    builder.add_walk_forward(
        mean_accuracy=0.96,
        min_accuracy=0.94,
        accuracy_std=0.03,
    )

    builder.add_leakage_status(True)

    if approved:
        builder.mark_production_approved(True)

    return builder.build()


def test_registry_construction(tmp_path):
    registry = ModelCardRegistry(tmp_path)

    assert registry.root == tmp_path
    assert registry.root.exists()


def test_save_card(tmp_path):
    registry = ModelCardRegistry(tmp_path)

    card = make_card()

    path = registry.save(card)

    assert path.exists()
    assert path.name == "model_gb_5d.json"


def test_saved_card_contains_expected_fields(tmp_path):
    registry = ModelCardRegistry(tmp_path)

    card = make_card()

    path = registry.save(card)

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["model_id"] == "model_gb_5d"
    assert payload["symbol"] == "RELIANCE"
    assert payload["timeframe"] == "1D"
    assert payload["horizon"] == 5
    assert payload["research_only"] is True
    assert payload["production_approved"] is False


def test_duplicate_card_is_not_overwritten(tmp_path):
    registry = ModelCardRegistry(tmp_path)

    card = make_card()

    registry.save(card)

    with pytest.raises(FileExistsError):
        registry.save(card)


def test_load_card(tmp_path):
    registry = ModelCardRegistry(tmp_path)

    card = make_card()

    registry.save(card)

    loaded = registry.load(card.model_id)

    assert isinstance(loaded, ModelCard)
    assert loaded == card


def test_missing_card_fails(tmp_path):
    registry = ModelCardRegistry(tmp_path)

    with pytest.raises(FileNotFoundError):
        registry.load("missing_model")


def test_empty_model_id_fails(tmp_path):
    registry = ModelCardRegistry(tmp_path)

    with pytest.raises(ValueError, match="model_id"):
        registry.load("")


def test_list_cards(tmp_path):
    registry = ModelCardRegistry(tmp_path)

    card_1 = make_card(
        model_id="model_1",
        symbol="TCS",
        horizon=1,
    )

    card_2 = make_card(
        model_id="model_5",
        symbol="TCS",
        horizon=5,
    )

    registry.save(card_1)
    registry.save(card_2)

    cards = registry.list_cards()

    assert len(cards) == 2

    ids = {card.model_id for card in cards}

    assert ids == {"model_1", "model_5"}


def test_find_by_symbol(tmp_path):
    registry = ModelCardRegistry(tmp_path)

    registry.save(
        make_card(
            model_id="tcs_model",
            symbol="TCS",
        )
    )

    registry.save(
        make_card(
            model_id="rel_model",
            symbol="RELIANCE",
        )
    )

    cards = registry.find(symbol="TCS")

    assert len(cards) == 1
    assert cards[0].symbol == "TCS"


def test_find_symbol_is_case_insensitive(tmp_path):
    registry = ModelCardRegistry(tmp_path)

    registry.save(
        make_card(
            model_id="tcs_model",
            symbol="TCS",
        )
    )

    cards = registry.find(symbol="tcs")

    assert len(cards) == 1
    assert cards[0].symbol == "TCS"


def test_find_by_timeframe(tmp_path):
    registry = ModelCardRegistry(tmp_path)

    registry.save(
        make_card(
            model_id="daily",
            timeframe="1D",
        )
    )

    registry.save(
        make_card(
            model_id="four_hour",
            timeframe="4H",
        )
    )

    cards = registry.find(timeframe="4H")

    assert len(cards) == 1
    assert cards[0].timeframe == "4H"


def test_find_by_horizon(tmp_path):
    registry = ModelCardRegistry(tmp_path)

    registry.save(
        make_card(
            model_id="h1",
            horizon=1,
        )
    )

    registry.save(
        make_card(
            model_id="h5",
            horizon=5,
        )
    )

    cards = registry.find(horizon=5)

    assert len(cards) == 1
    assert cards[0].horizon == 5


def test_find_by_approval_status(tmp_path):
    registry = ModelCardRegistry(tmp_path)

    registry.save(
        make_card(
            model_id="research",
            approved=False,
        )
    )

    registry.save(
        make_card(
            model_id="approved",
            approved=True,
        )
    )

    research_cards = registry.find(
        production_approved=False
    )

    approved_cards = registry.find(
        production_approved=True
    )

    assert len(research_cards) == 1
    assert research_cards[0].model_id == "research"

    assert len(approved_cards) == 1
    assert approved_cards[0].model_id == "approved"


def test_latest_returns_matching_card(tmp_path):
    registry = ModelCardRegistry(tmp_path)

    first = make_card(
        model_id="first",
        symbol="TCS",
        horizon=5,
    )

    second = make_card(
        model_id="second",
        symbol="TCS",
        horizon=5,
    )

    registry.save(first)
    registry.save(second)

    latest = registry.latest(
        symbol="TCS",
        timeframe="1D",
        horizon=5,
    )

    assert latest is not None
    assert latest.model_id in {"first", "second"}


def test_latest_returns_none_when_no_match(tmp_path):
    registry = ModelCardRegistry(tmp_path)

    registry.save(
        make_card(
            model_id="tcs_5d",
            symbol="TCS",
            horizon=5,
        )
    )

    latest = registry.latest(
        symbol="RELIANCE",
        timeframe="1D",
        horizon=5,
    )

    assert latest is None


def test_approved_cards(tmp_path):
    registry = ModelCardRegistry(tmp_path)

    registry.save(
        make_card(
            model_id="research",
            approved=False,
        )
    )

    registry.save(
        make_card(
            model_id="approved",
            approved=True,
        )
    )

    cards = registry.approved_cards()

    assert len(cards) == 1
    assert cards[0].model_id == "approved"


def test_research_cards(tmp_path):
    registry = ModelCardRegistry(tmp_path)

    registry.save(
        make_card(
            model_id="research_1",
            approved=False,
        )
    )

    registry.save(
        make_card(
            model_id="research_2",
            approved=False,
        )
    )

    registry.save(
        make_card(
            model_id="approved",
            approved=True,
        )
    )

    cards = registry.research_cards()

    assert len(cards) == 2
    assert all(
        card.production_approved is False
        for card in cards
    )


def test_registry_does_not_grant_approval(tmp_path):
    registry = ModelCardRegistry(tmp_path)

    card = make_card()

    registry.save(card)

    loaded = registry.load(card.model_id)

    assert loaded.production_approved is False
    assert loaded.research_only is True


def test_approved_card_can_be_recorded_but_not_granted_by_registry(
    tmp_path,
):
    registry = ModelCardRegistry(tmp_path)

    card = make_card(
        model_id="approved_model",
        approved=True,
    )

    registry.save(card)

    loaded = registry.load("approved_model")

    assert loaded.production_approved is True
    assert loaded.research_only is False


def test_delete_card(tmp_path):
    registry = ModelCardRegistry(tmp_path)

    card = make_card()

    path = registry.save(card)

    assert path.exists()

    registry.delete(card.model_id)

    assert not path.exists()


def test_delete_missing_card_fails(tmp_path):
    registry = ModelCardRegistry(tmp_path)

    with pytest.raises(FileNotFoundError):
        registry.delete("missing_model")


def test_list_is_sorted_by_created_at(tmp_path):
    registry = ModelCardRegistry(tmp_path)

    card_old = ModelCardBuilder(
        model_id="old",
        model_name="RandomForest",
        symbol="TCS",
        timeframe="1D",
        horizon=5,
        target="Direction_5",
        created_at="2026-01-01T00:00:00+00:00",
    ).build()

    card_new = ModelCardBuilder(
        model_id="new",
        model_name="GradientBoosting",
        symbol="TCS",
        timeframe="1D",
        horizon=5,
        target="Direction_5",
        created_at="2026-09-01T00:00:00+00:00",
    ).build()

    registry.save(card_old)
    registry.save(card_new)

    cards = registry.list_cards()

    assert cards[0].model_id == "new"
    assert cards[1].model_id == "old"


def test_summary(tmp_path):
    registry = ModelCardRegistry(tmp_path)

    registry.save(
        make_card(
            model_id="research_1",
            approved=False,
        )
    )

    registry.save(
        make_card(
            model_id="approved_1",
            approved=True,
        )
    )

    summary = registry.summary()

    assert summary["total_cards"] == 2
    assert summary["research_cards"] == 1
    assert summary["approved_cards"] == 1
    assert summary["leakage_free_cards"] == 2


def test_summary_counts_holdout_results(tmp_path):
    registry = ModelCardRegistry(tmp_path)

    card = (
        ModelCardBuilder(
            model_id="model_holdout",
            model_name="GradientBoosting",
            symbol="TCS",
            timeframe="1D",
            horizon=5,
            target="Direction_5",
            created_at="2026-09-09T12:00:00+00:00",
        )
        .add_holdout(
            accuracy=0.96,
            passed=True,
        )
        .build()
    )

    registry.save(card)

    summary = registry.summary()

    assert summary["cards_with_holdout_results"] == 1


def test_summary_counts_backtest_results(tmp_path):
    registry = ModelCardRegistry(tmp_path)

    card = (
        ModelCardBuilder(
            model_id="model_backtest",
            model_name="GradientBoosting",
            symbol="TCS",
            timeframe="1D",
            horizon=5,
            target="Direction_5",
            created_at="2026-09-09T12:00:00+00:00",
        )
        .add_backtest(
            profit_factor=1.5,
            sharpe_ratio=1.0,
            max_drawdown=0.2,
            trade_count=50,
            passed=True,
        )
        .build()
    )

    registry.save(card)

    summary = registry.summary()

    assert summary["cards_with_backtest_results"] == 1


def test_convenience_registration(tmp_path):
    card = make_card()

    path = register_model_card(
        card,
        root=tmp_path,
    )

    assert path.exists()

    loaded = ModelCardRegistry(tmp_path).load(
        card.model_id
    )

    assert loaded == card


def test_invalid_card_type_fails(tmp_path):
    registry = ModelCardRegistry(tmp_path)

    with pytest.raises(TypeError):
        registry.save("not a model card")


def test_corrupted_card_json_fails(tmp_path):
    registry = ModelCardRegistry(tmp_path)

    path = tmp_path / "corrupted.json"

    path.write_text(
        "{ invalid json",
        encoding="utf-8",
    )

    with pytest.raises(json.JSONDecodeError):
        registry.list_cards()


def test_card_contents_are_preserved(tmp_path):
    registry = ModelCardRegistry(tmp_path)

    card = (
        ModelCardBuilder(
            model_id="detailed_model",
            model_name="GradientBoosting",
            symbol="INFY",
            timeframe="4H",
            horizon=10,
            target="Direction_10",
            created_at="2026-09-09T12:00:00+00:00",
        )
        .add_validation(0.98, True)
        .add_holdout(0.96, True)
        .add_walk_forward(0.97, 0.94, 0.03)
        .add_calibration(0.12, 0.07, True)
        .add_range_validation(0.82, True)
        .add_regime_validation(0.75, True)
        .add_backtest(1.6, 1.1, 0.18, 60, True)
        .add_robustness(0.81, True)
        .add_leakage_status(True)
        .add_model_size(25, 5000)
        .add_limitation("Future regimes may differ.")
        .add_warning("Research evidence only.")
        .add_metadata(experiment_id="EXP-001")
        .build()
    )

    registry.save(card)

    loaded = registry.load(card.model_id)

    assert loaded.validation_accuracy == 0.98
    assert loaded.holdout_accuracy == 0.96
    assert loaded.walk_forward_mean_accuracy == 0.97
    assert loaded.calibration_brier == 0.12
    assert loaded.calibration_ece == 0.07
    assert loaded.range_coverage == 0.82
    assert loaded.regime_stability == 0.75
    assert loaded.profit_factor == 1.6
    assert loaded.sharpe_ratio == 1.1
    assert loaded.max_drawdown == 0.18
    assert loaded.trade_count == 60
    assert loaded.robustness_score == 0.81
    assert loaded.feature_count == 25
    assert loaded.training_samples == 5000
    assert loaded.metadata["experiment_id"] == "EXP-001"
    assert "Future regimes may differ." in loaded.limitations
    assert "Research evidence only." in loaded.warnings
