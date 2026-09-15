"""
AI Swing Analyser — Research Model Card.

A model card is the human-readable research record for a model.

It does not approve a model for production. Production approval remains
the responsibility of the dedicated approval and model-registry layers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class ModelCard:
    """Standardized research documentation for one model."""

    model_id: str
    model_name: str

    symbol: str
    timeframe: str
    horizon: int
    target: str

    created_at: str

    validation_accuracy: Optional[float] = None
    holdout_accuracy: Optional[float] = None

    walk_forward_mean_accuracy: Optional[float] = None
    walk_forward_min_accuracy: Optional[float] = None
    walk_forward_accuracy_std: Optional[float] = None

    calibration_brier: Optional[float] = None
    calibration_ece: Optional[float] = None

    range_coverage: Optional[float] = None
    regime_stability: Optional[float] = None

    profit_factor: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    trade_count: Optional[int] = None

    robustness_score: Optional[float] = None

    feature_count: Optional[int] = None
    training_samples: Optional[int] = None

    validation_passed: bool = False
    holdout_passed: bool = False
    leakage_free: bool = False
    calibration_passed: bool = False
    range_validation_passed: bool = False
    regime_validation_passed: bool = False
    backtest_passed: bool = False
    robustness_passed: bool = False

    production_approved: bool = False
    research_only: bool = True

    intended_use: str = (
        "Research and swing-trading model evaluation."
    )

    limitations: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelCardBuilder:
    """Build model cards from research evidence."""

    def __init__(
        self,
        model_id: str,
        model_name: str,
        symbol: str,
        timeframe: str,
        horizon: int,
        target: str,
        created_at: str,
    ) -> None:
        if not model_id:
            raise ValueError("model_id cannot be empty.")

        if not model_name:
            raise ValueError("model_name cannot be empty.")

        if not symbol:
            raise ValueError("symbol cannot be empty.")

        if not timeframe:
            raise ValueError("timeframe cannot be empty.")

        if horizon <= 0:
            raise ValueError("horizon must be positive.")

        if not target:
            raise ValueError("target cannot be empty.")

        if not created_at:
            raise ValueError("created_at cannot be empty.")

        self._data: dict[str, Any] = {
            "model_id": model_id,
            "model_name": model_name,
            "symbol": symbol.upper(),
            "timeframe": timeframe.upper(),
            "horizon": horizon,
            "target": target,
            "created_at": created_at,
            "research_only": True,
            "production_approved": False,
        }

    def add_validation(
        self,
        accuracy: Optional[float],
        passed: bool,
    ) -> "ModelCardBuilder":
        self._data["validation_accuracy"] = accuracy
        self._data["validation_passed"] = bool(passed)
        return self

    def add_holdout(
        self,
        accuracy: Optional[float],
        passed: bool,
    ) -> "ModelCardBuilder":
        self._data["holdout_accuracy"] = accuracy
        self._data["holdout_passed"] = bool(passed)
        return self

    def add_walk_forward(
        self,
        mean_accuracy: Optional[float],
        min_accuracy: Optional[float],
        accuracy_std: Optional[float],
    ) -> "ModelCardBuilder":
        self._data["walk_forward_mean_accuracy"] = mean_accuracy
        self._data["walk_forward_min_accuracy"] = min_accuracy
        self._data["walk_forward_accuracy_std"] = accuracy_std
        return self

    def add_calibration(
        self,
        brier: Optional[float],
        ece: Optional[float],
        passed: bool,
    ) -> "ModelCardBuilder":
        self._data["calibration_brier"] = brier
        self._data["calibration_ece"] = ece
        self._data["calibration_passed"] = bool(passed)
        return self

    def add_range_validation(
        self,
        coverage: Optional[float],
        passed: bool,
    ) -> "ModelCardBuilder":
        self._data["range_coverage"] = coverage
        self._data["range_validation_passed"] = bool(passed)
        return self

    def add_regime_validation(
        self,
        stability: Optional[float],
        passed: bool,
    ) -> "ModelCardBuilder":
        self._data["regime_stability"] = stability
        self._data["regime_validation_passed"] = bool(passed)
        return self

    def add_backtest(
        self,
        profit_factor: Optional[float],
        sharpe_ratio: Optional[float],
        max_drawdown: Optional[float],
        trade_count: Optional[int],
        passed: bool,
    ) -> "ModelCardBuilder":
        self._data["profit_factor"] = profit_factor
        self._data["sharpe_ratio"] = sharpe_ratio
        self._data["max_drawdown"] = max_drawdown
        self._data["trade_count"] = trade_count
        self._data["backtest_passed"] = bool(passed)
        return self

    def add_robustness(
        self,
        score: Optional[float],
        passed: bool,
    ) -> "ModelCardBuilder":
        self._data["robustness_score"] = score
        self._data["robustness_passed"] = bool(passed)
        return self

    def add_leakage_status(
        self,
        passed: bool,
    ) -> "ModelCardBuilder":
        self._data["leakage_free"] = bool(passed)
        return self

    def add_model_size(
        self,
        feature_count: Optional[int],
        training_samples: Optional[int],
    ) -> "ModelCardBuilder":
        self._data["feature_count"] = feature_count
        self._data["training_samples"] = training_samples
        return self

    def add_limitation(
        self,
        limitation: str,
    ) -> "ModelCardBuilder":
        if limitation:
            current = list(self._data.get("limitations", ()))
            current.append(limitation)
            self._data["limitations"] = tuple(current)
        return self

    def add_warning(
        self,
        warning: str,
    ) -> "ModelCardBuilder":
        if warning:
            current = list(self._data.get("warnings", ()))
            current.append(warning)
            self._data["warnings"] = tuple(current)
        return self

    def add_metadata(
        self,
        **metadata: Any,
    ) -> "ModelCardBuilder":
        current = dict(self._data.get("metadata", {}))
        current.update(metadata)
        self._data["metadata"] = current
        return self

    def mark_production_approved(
        self,
        approved: bool,
    ) -> "ModelCardBuilder":
        """
        Record approval only when explicitly supplied by the approval layer.

        The default remains False.
        """

        self._data["production_approved"] = bool(approved)

        if approved:
            self._data["research_only"] = False

        return self

    def build(self) -> ModelCard:
        """
        Build an immutable model card.

        A model card can never become production-approved merely because
        its numerical metrics are strong.
        """

        self._data.setdefault("limitations", ())
        self._data.setdefault("warnings", ())
        self._data.setdefault("metadata", {})

        return ModelCard(
            **self._data
        )


def model_card_from_selection(
    selection_record: Any,
    target: str,
) -> ModelCard:
    """
    Create a model card from a SelectionRecord-like object.

    Selection records are research evidence only, so the resulting card
    remains unapproved.
    """

    if selection_record is None:
        raise ValueError("selection_record cannot be None.")

    builder = ModelCardBuilder(
        model_id=selection_record.selected_model_id
        or "UNSELECTED",
        model_name=selection_record.selected_model_name
        or "No model selected",
        symbol=selection_record.symbol,
        timeframe=selection_record.timeframe,
        horizon=selection_record.horizon,
        target=target,
        created_at=selection_record.created_at,
    )

    builder.add_validation(
        accuracy=selection_record.validation_accuracy,
        passed=(
            selection_record.validation_accuracy is not None
            and selection_record.validation_accuracy >= 0.95
        ),
    )

    builder.add_walk_forward(
        mean_accuracy=selection_record.walk_forward_mean_accuracy,
        min_accuracy=selection_record.walk_forward_min_accuracy,
        accuracy_std=selection_record.walk_forward_accuracy_std,
    )

    if selection_record.walk_forward_brier is not None:
        builder.add_calibration(
            brier=selection_record.walk_forward_brier,
            ece=None,
            passed=False,
        )
        builder.add_warning(
            "Walk-forward Brier score is not a substitute for "
            "dedicated probability calibration."
        )

    builder.add_leakage_status(
        passed=not selection_record.final_holdout_used
    )

    builder.add_warning(
        "Final holdout performance has not been evaluated in this "
        "selection record."
    )

    builder.add_warning(
        "Selection does not constitute production approval."
    )

    builder.add_limitation(
        "Model selection is based on development and walk-forward evidence."
    )

    builder.add_limitation(
        "Performance may change under future market regimes."
    )

    builder.add_metadata(
        selection_id=selection_record.selection_id,
        source="selection_registry",
        final_holdout_used=False,
        production_approval_required=True,
    )

    return builder.build()


def save_model_card(
    card: ModelCard,
    root: str | Path = "experiments/model_cards",
) -> Path:
    """Persist a model card as JSON."""

    if not isinstance(card, ModelCard):
        raise TypeError("card must be a ModelCard.")

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    path = root_path / f"{card.model_id}.json"

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


def load_model_card(
    model_id: str,
    root: str | Path = "experiments/model_cards",
) -> ModelCard:
    """Load one model card."""

    if not model_id:
        raise ValueError("model_id cannot be empty.")

    path = Path(root) / f"{model_id}.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Model card not found: {model_id}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    if payload.get("production_approved") is True:
        # Approval may be documented, but the model card itself does
        # not grant approval.
        payload["production_approved"] = True

    payload["limitations"] = tuple(
        payload.get("limitations", [])
    )
    payload["warnings"] = tuple(
        payload.get("warnings", [])
    )

    return ModelCard(**payload)


def model_card_summary(
    card: ModelCard,
) -> dict[str, Any]:
    """Return a compact dashboard-friendly summary."""

    return {
        "model_id": card.model_id,
        "model_name": card.model_name,
        "symbol": card.symbol,
        "timeframe": card.timeframe,
        "horizon": card.horizon,
        "validation_accuracy": card.validation_accuracy,
        "holdout_accuracy": card.holdout_accuracy,
        "walk_forward_mean_accuracy": (
            card.walk_forward_mean_accuracy
        ),
        "walk_forward_min_accuracy": (
            card.walk_forward_min_accuracy
        ),
        "walk_forward_accuracy_std": (
            card.walk_forward_accuracy_std
        ),
        "range_coverage": card.range_coverage,
        "regime_stability": card.regime_stability,
        "profit_factor": card.profit_factor,
        "sharpe_ratio": card.sharpe_ratio,
        "max_drawdown": card.max_drawdown,
        "trade_count": card.trade_count,
        "robustness_score": card.robustness_score,
        "leakage_free": card.leakage_free,
        "production_approved": card.production_approved,
        "research_only": card.research_only,
        "warning_count": len(card.warnings),
        "limitation_count": len(card.limitations),
    }


__all__ = [
    "ModelCard",
    "ModelCardBuilder",
    "model_card_from_selection",
    "save_model_card",
    "load_model_card",
    "model_card_summary",
]
