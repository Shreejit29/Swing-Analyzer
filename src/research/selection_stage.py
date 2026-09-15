"""
AI Swing Analyser — Model Selection Research Stage.

Integrates:
    candidate experiment evidence
        ↓
    controlled model selection
        ↓
    selection registry
        ↓
    model card

This stage remains research-only.

Final holdout evaluation, calibration, backtesting, robustness,
and production approval happen in later stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .model_card import (
    ModelCard,
    model_card_from_selection,
    save_model_card,
)
from .model_selection import ModelSelectionConfig
from .selection_pipeline import (
    SelectionCandidate,
    SelectionPipeline,
    SelectionPipelineResult,
)
from .selection_registry import (
    SelectionRecord,
    SelectionRegistry,
)


@dataclass
class ModelSelectionStageResult:
    """Complete output of the controlled selection stage."""

    symbol: str
    timeframe: str
    horizon: int
    target: str

    selection: SelectionPipelineResult

    selection_record: Optional[SelectionRecord] = None
    model_card: Optional[ModelCard] = None
    model_card_path: Optional[str] = None

    successful: bool = False
    final_holdout_used: bool = False
    production_approved: bool = False

    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def selected_model_id(self) -> Optional[str]:
        return self.selection.selected_model_id

    @property
    def selected_model_name(self) -> Optional[str]:
        return self.selection.selected_model_name

    def summary(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "horizon": self.horizon,
            "target": self.target,
            "selected_model_id": self.selected_model_id,
            "selected_model_name": self.selected_model_name,
            "candidate_count": len(self.selection.candidates),
            "successful": self.successful,
            "final_holdout_used": self.final_holdout_used,
            "production_approved": self.production_approved,
            "model_card_created": self.model_card is not None,
            "model_card_path": self.model_card_path,
            "warning_count": len(self.warnings),
            "warnings": list(self.warnings),
        }


class ModelSelectionStage:
    """
    Research-stage orchestration for controlled model selection.

    Responsibilities
    ----------------
    - Validate candidate evidence.
    - Select the strongest eligible model.
    - Persist the selection record.
    - Create a research model card.
    - Keep the result explicitly research-only.

    Not responsible for
    -------------------
    - final holdout evaluation,
    - probability calibration,
    - range validation,
    - regime validation,
    - backtesting,
    - robustness,
    - production approval.
    """

    def __init__(
        self,
        config: Optional[ModelSelectionConfig] = None,
        selection_root: str | Path = "experiments/selections",
        model_card_root: str | Path = "experiments/model_cards",
    ) -> None:
        self.config = config or ModelSelectionConfig()

        self.pipeline = SelectionPipeline(
            config=self.config,
        )

        self.selection_registry = SelectionRegistry(
            root=selection_root,
        )

        self.model_card_root = Path(model_card_root)

    @staticmethod
    def _validate_identity(
        symbol: str,
        timeframe: str,
        horizon: int,
        target: str,
    ) -> None:
        if not symbol or not symbol.strip():
            raise ValueError("symbol cannot be empty.")

        if not timeframe or not timeframe.strip():
            raise ValueError("timeframe cannot be empty.")

        if horizon <= 0:
            raise ValueError("horizon must be positive.")

        if not target or not target.strip():
            raise ValueError("target cannot be empty.")

    @staticmethod
    def _validate_candidates(
        candidates: Iterable[SelectionCandidate],
    ) -> list[SelectionCandidate]:
        candidate_list = list(candidates)

        if not candidate_list:
            raise ValueError(
                "At least one selection candidate is required."
            )

        model_ids = [
            candidate.model_id
            for candidate in candidate_list
        ]

        if len(model_ids) != len(set(model_ids)):
            raise ValueError(
                "Duplicate candidate model_id values are not allowed."
            )

        return candidate_list

    def run(
        self,
        candidates: Iterable[SelectionCandidate],
        symbol: str,
        timeframe: str,
        horizon: int,
        target: str,
        create_model_card: bool = True,
        persist_selection: bool = True,
        persist_model_card: bool = True,
    ) -> ModelSelectionStageResult:
        """
        Execute the complete controlled model-selection stage.
        """

        self._validate_identity(
            symbol=symbol,
            timeframe=timeframe,
            horizon=horizon,
            target=target,
        )

        candidate_list = self._validate_candidates(
            candidates
        )

        selection = self.pipeline.select(
            candidates=candidate_list,
            horizon=horizon,
        )

        warnings = list(selection.warnings)

        selection_record: Optional[SelectionRecord] = None
        model_card: Optional[ModelCard] = None
        model_card_path: Optional[str] = None

        if persist_selection:
            selection_record = self.selection_registry.register(
                result=selection,
                symbol=symbol,
                timeframe=timeframe,
            )

        if create_model_card and selection.successful:
            if selection_record is not None:
                model_card = model_card_from_selection(
                    selection_record=selection_record,
                    target=target,
                )
            else:
                # A temporary selection record is required because the
                # model-card builder uses the selection identity and
                # evidence metadata.
                temporary_record = self.selection_registry.create_record(
                    result=selection,
                    symbol=symbol,
                    timeframe=timeframe,
                )

                model_card = model_card_from_selection(
                    selection_record=temporary_record,
                    target=target,
                )

            if persist_model_card:
                path = save_model_card(
                    model_card,
                    root=self.model_card_root,
                )
                model_card_path = str(path)

        if not selection.successful:
            warnings.append(
                "No model passed controlled selection; "
                "downstream production stages must not proceed."
            )

        return ModelSelectionStageResult(
            symbol=symbol.upper(),
            timeframe=timeframe.upper(),
            horizon=horizon,
            target=target,
            selection=selection,
            selection_record=selection_record,
            model_card=model_card,
            model_card_path=model_card_path,
            successful=selection.successful,
            final_holdout_used=False,
            production_approved=False,
            warnings=warnings,
            metadata={
                "stage": "model_selection",
                "research_only": True,
                "final_holdout_used": False,
                "production_approved": False,
                "selection_persisted": (
                    selection_record is not None
                ),
                "model_card_created": (
                    model_card is not None
                ),
                "production_approval_required": True,
            },
        )

    def run_without_persistence(
        self,
        candidates: Iterable[SelectionCandidate],
        symbol: str,
        timeframe: str,
        horizon: int,
        target: str,
    ) -> ModelSelectionStageResult:
        """
        Run selection without writing files.

        Useful for research experiments and unit tests.
        """

        return self.run(
            candidates=candidates,
            symbol=symbol,
            timeframe=timeframe,
            horizon=horizon,
            target=target,
            create_model_card=True,
            persist_selection=False,
            persist_model_card=False,
        )

    def select_multiple_horizons(
        self,
        candidates: Iterable[SelectionCandidate],
        symbol: str,
        timeframe: str,
        horizons: Iterable[int],
        target_prefix: str = "Direction_",
        create_model_cards: bool = True,
        persist_selection: bool = True,
        persist_model_cards: bool = True,
    ) -> dict[int, ModelSelectionStageResult]:
        """
        Run independent selection for multiple prediction horizons.

        Each horizon receives its own model-selection decision.
        """

        candidate_list = list(candidates)
        horizon_list = list(horizons)

        if not horizon_list:
            raise ValueError(
                "At least one horizon is required."
            )

        if len(set(horizon_list)) != len(horizon_list):
            raise ValueError(
                "Duplicate horizons are not allowed."
            )

        results: dict[int, ModelSelectionStageResult] = {}

        for horizon in horizon_list:
            horizon_candidates = [
                candidate
                for candidate in candidate_list
                if candidate.horizon == horizon
            ]

            if not horizon_candidates:
                continue

            results[horizon] = self.run(
                candidates=horizon_candidates,
                symbol=symbol,
                timeframe=timeframe,
                horizon=horizon,
                target=f"{target_prefix}{horizon}",
                create_model_card=create_model_cards,
                persist_selection=persist_selection,
                persist_model_card=persist_model_cards,
            )

        return results

    @staticmethod
    def selection_table(
        results: dict[int, ModelSelectionStageResult],
    ) -> list[dict[str, Any]]:
        """Create a compact cross-horizon selection table."""

        rows: list[dict[str, Any]] = []

        for horizon in sorted(results):
            result = results[horizon]
            selected = result.selection.selected_candidate

            rows.append(
                {
                    "horizon": horizon,
                    "symbol": result.symbol,
                    "timeframe": result.timeframe,
                    "selected_model_id": (
                        result.selected_model_id
                    ),
                    "selected_model_name": (
                        result.selected_model_name
                    ),
                    "validation_accuracy": (
                        selected.validation_accuracy
                        if selected is not None
                        else None
                    ),
                    "walk_forward_mean_accuracy": (
                        selected.walk_forward_mean_accuracy
                        if selected is not None
                        else None
                    ),
                    "walk_forward_min_accuracy": (
                        selected.walk_forward_min_accuracy
                        if selected is not None
                        else None
                    ),
                    "walk_forward_accuracy_std": (
                        selected.walk_forward_accuracy_std
                        if selected is not None
                        else None
                    ),
                    "successful": result.successful,
                    "final_holdout_used": (
                        result.final_holdout_used
                    ),
                    "production_approved": (
                        result.production_approved
                    ),
                }
            )

        return rows


def run_model_selection_stage(
    candidates: Iterable[SelectionCandidate],
    symbol: str,
    timeframe: str,
    horizon: int,
    target: str,
    config: Optional[ModelSelectionConfig] = None,
    selection_root: str | Path = "experiments/selections",
    model_card_root: str | Path = "experiments/model_cards",
) -> ModelSelectionStageResult:
    """Convenience API for the integrated selection stage."""

    stage = ModelSelectionStage(
        config=config,
        selection_root=selection_root,
        model_card_root=model_card_root,
    )

    return stage.run(
        candidates=candidates,
        symbol=symbol,
        timeframe=timeframe,
        horizon=horizon,
        target=target,
    )


__all__ = [
    "ModelSelectionStageResult",
    "ModelSelectionStage",
    "run_model_selection_stage",
]
