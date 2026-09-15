"""
AI Swing Analyser — Model Card Registry.

Persistent registry for research model cards.

Design principles
-----------------
1. Model cards are documentation, not approval authority.
2. Production approval must come from the dedicated approval layer.
3. Final holdout usage is explicitly tracked.
4. Existing cards are never silently overwritten.
5. Loading a card does not make a model production-ready.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .model_card import ModelCard


class ModelCardRegistry:
    """JSON-backed registry for model cards."""

    def __init__(
        self,
        root: str | Path = "experiments/model_cards",
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, card: ModelCard) -> Path:
        """Persist a model card without overwriting an existing card."""

        if not isinstance(card, ModelCard):
            raise TypeError("card must be a ModelCard.")

        path = self.root / f"{card.model_id}.json"

        if path.exists():
            raise FileExistsError(
                f"Model card already exists: {path}"
            )

        payload = asdict(card)
        payload["limitations"] = list(card.limitations)
        payload["warnings"] = list(card.warnings)

        temporary_path = path.with_suffix(".tmp")

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
            )

        temporary_path.replace(path)

        return path

    def load(self, model_id: str) -> ModelCard:
        """Load a model card by model ID."""

        if not model_id:
            raise ValueError("model_id cannot be empty.")

        path = self.root / f"{model_id}.json"

        if not path.exists():
            raise FileNotFoundError(
                f"Model card not found: {model_id}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            payload = json.load(handle)

        payload["limitations"] = tuple(
            payload.get("limitations", [])
        )
        payload["warnings"] = tuple(
            payload.get("warnings", [])
        )

        return ModelCard(**payload)

    def list_cards(self) -> list[ModelCard]:
        """Return all registered model cards."""

        cards: list[ModelCard] = []

        for path in sorted(self.root.glob("*.json")):
            with path.open(
                "r",
                encoding="utf-8",
            ) as handle:
                payload = json.load(handle)

            payload["limitations"] = tuple(
                payload.get("limitations", [])
            )
            payload["warnings"] = tuple(
                payload.get("warnings", [])
            )

            cards.append(
                ModelCard(**payload)
            )

        cards.sort(
            key=lambda card: card.created_at,
            reverse=True,
        )

        return cards

    def find(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        horizon: Optional[int] = None,
        production_approved: Optional[bool] = None,
    ) -> list[ModelCard]:
        """Find model cards using optional filters."""

        cards = self.list_cards()

        if symbol is not None:
            normalized_symbol = symbol.upper()

            cards = [
                card
                for card in cards
                if card.symbol.upper() == normalized_symbol
            ]

        if timeframe is not None:
            normalized_timeframe = timeframe.upper()

            cards = [
                card
                for card in cards
                if card.timeframe.upper() == normalized_timeframe
            ]

        if horizon is not None:
            cards = [
                card
                for card in cards
                if card.horizon == horizon
            ]

        if production_approved is not None:
            cards = [
                card
                for card in cards
                if card.production_approved == production_approved
            ]

        return cards

    def latest(
        self,
        symbol: str,
        timeframe: str,
        horizon: int,
    ) -> Optional[ModelCard]:
        """Return the newest card for a symbol/timeframe/horizon."""

        cards = self.find(
            symbol=symbol,
            timeframe=timeframe,
            horizon=horizon,
        )

        return cards[0] if cards else None

    def approved_cards(self) -> list[ModelCard]:
        """
        Return cards documenting approved models.

        This does NOT grant approval. It only reads the recorded state.
        """

        return self.find(production_approved=True)

    def research_cards(self) -> list[ModelCard]:
        """Return cards still marked as research-only."""

        return self.find(production_approved=False)

    def delete(self, model_id: str) -> None:
        """Delete a model card from the local research registry."""

        if not model_id:
            raise ValueError("model_id cannot be empty.")

        path = self.root / f"{model_id}.json"

        if not path.exists():
            raise FileNotFoundError(
                f"Model card not found: {model_id}"
            )

        path.unlink()

    def summary(self) -> dict:
        """Return registry-level statistics."""

        cards = self.list_cards()

        return {
            "root": str(self.root),
            "total_cards": len(cards),
            "research_cards": sum(
                card.research_only
                for card in cards
            ),
            "approved_cards": sum(
                card.production_approved
                for card in cards
            ),
            "leakage_free_cards": sum(
                card.leakage_free
                for card in cards
            ),
            "cards_with_holdout_results": sum(
                card.holdout_accuracy is not None
                for card in cards
            ),
            "cards_with_backtest_results": sum(
                card.trade_count is not None
                for card in cards
            ),
        }


def register_model_card(
    card: ModelCard,
    root: str | Path = "experiments/model_cards",
) -> Path:
    """Convenience API for registering a model card."""

    registry = ModelCardRegistry(root=root)

    return registry.save(card)


__all__ = [
    "ModelCardRegistry",
    "register_model_card",
]
