"""
AI Swing Analyser — Controlled Model Selection Pipeline.

Connects candidate experiment results with the controlled model selector.

Design principles
-----------------
1. Final holdout data is never used for model selection.
2. Only development/validation/walk-forward evidence is considered.
3. Selection is deterministic.
4. A model is selected only when it satisfies the configured research gates.
5. Selection does not imply production approval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from .model_selection import (
    ControlledModelSelector,
    ModelCandidateScore,
    ModelSelectionConfig,
    ModelSelectionResult,
)


@dataclass(frozen=True)
class SelectionCandidate:
    """
    Research evidence for one candidate model.

    All metrics must come from development/validation/walk-forward
    evaluation. Final holdout metrics are deliberately excluded.
    """

    model_id: str
    model_name: str
    horizon: int

    validation_accuracy: float
    walk_forward_mean_accuracy: float
    walk_forward_min_accuracy: float
    walk_forward_accuracy_std: float

    walk_forward_brier: Optional[float] = None
    training_samples: int = 0

    feature_count: Optional[int] = None
    complexity_score: Optional[float] = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectionPipelineResult:
    """Result of controlled model selection."""

    horizon: int
    candidates: list[SelectionCandidate]
    selection_result: ModelSelectionResult

    selected_model_id: Optional[str] = None
    selected_model_name: Optional[str] = None

    final_holdout_used: bool = False
    production_approved: bool = False

    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def successful(self) -> bool:
        return self.selected_model_id is not None

    @property
    def selected_candidate(self) -> Optional[SelectionCandidate]:
        if self.selected_model_id is None:
            return None

        for candidate in self.candidates:
            if candidate.model_id == self.selected_model_id:
                return candidate

        return None

    def summary(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "candidate_count": len(self.candidates),
            "selected_model_id": self.selected_model_id,
            "selected_model_name": self.selected_model_name,
            "successful": self.successful,
            "final_holdout_used": self.final_holdout_used,
            "production_approved": self.production_approved,
            "warning_count": len(self.warnings),
            "warnings": list(self.warnings),
        }


class SelectionPipeline:
    """
    Controlled model-selection orchestration layer.

    This class performs selection only.

    It does NOT:
        - train models,
        - evaluate the final holdout,
        - calibrate probabilities,
        - approve production models,
        - generate trading signals.
    """

    def __init__(
        self,
        config: Optional[ModelSelectionConfig] = None,
    ) -> None:
        self.config = config or ModelSelectionConfig()
        self.selector = ControlledModelSelector(self.config)

    @staticmethod
    def _validate_candidate(candidate: SelectionCandidate) -> None:
        if not candidate.model_id:
            raise ValueError("Candidate model_id cannot be empty.")

        if not candidate.model_name:
            raise ValueError("Candidate model_name cannot be empty.")

        if candidate.horizon <= 0:
            raise ValueError("Candidate horizon must be positive.")

        numeric_values = {
            "validation_accuracy": candidate.validation_accuracy,
            "walk_forward_mean_accuracy": candidate.walk_forward_mean_accuracy,
            "walk_forward_min_accuracy": candidate.walk_forward_min_accuracy,
            "walk_forward_accuracy_std": candidate.walk_forward_accuracy_std,
        }

        if candidate.walk_forward_brier is not None:
            numeric_values["walk_forward_brier"] = candidate.walk_forward_brier

        for name, value in numeric_values.items():
            if not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric.")

            if not float("-inf") < float(value) < float("inf"):
                raise ValueError(f"{name} must be finite.")

        for name in (
            "validation_accuracy",
            "walk_forward_mean_accuracy",
            "walk_forward_min_accuracy",
        ):
            value = float(getattr(candidate, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")

        if candidate.walk_forward_accuracy_std < 0:
            raise ValueError(
                "walk_forward_accuracy_std cannot be negative."
            )

        if candidate.walk_forward_brier is not None:
            if not 0.0 <= candidate.walk_forward_brier <= 1.0:
                raise ValueError(
                    "walk_forward_brier must be between 0 and 1."
                )

        if candidate.training_samples < 0:
            raise ValueError("training_samples cannot be negative.")

        if candidate.feature_count is not None and candidate.feature_count < 0:
            raise ValueError("feature_count cannot be negative.")

    @staticmethod
    def _to_selector_candidate(
        candidate: SelectionCandidate,
    ) -> ModelCandidateScore:
        """
        Convert pipeline evidence to the selector's candidate representation.
        """

        return ModelCandidateScore(
            model_id=candidate.model_id,
            model_name=candidate.model_name,
            validation_accuracy=candidate.validation_accuracy,
            walk_forward_mean_accuracy=candidate.walk_forward_mean_accuracy,
            walk_forward_min_accuracy=candidate.walk_forward_min_accuracy,
            walk_forward_accuracy_std=candidate.walk_forward_accuracy_std,
            walk_forward_brier=candidate.walk_forward_brier,
            training_samples=candidate.training_samples,
            feature_count=candidate.feature_count,
            complexity_score=candidate.complexity_score,
        )

    def select(
        self,
        candidates: Sequence[SelectionCandidate],
        horizon: int,
    ) -> SelectionPipelineResult:
        """
        Select the best research candidate.

        Final holdout information is not accepted as an input.
        """

        if horizon <= 0:
            raise ValueError("horizon must be positive.")

        candidate_list = list(candidates)

        if not candidate_list:
            raise ValueError("At least one candidate is required.")

        for candidate in candidate_list:
            self._validate_candidate(candidate)

            if candidate.horizon != horizon:
                raise ValueError(
                    f"Candidate {candidate.model_id!r} has horizon "
                    f"{candidate.horizon}, expected {horizon}."
                )

        model_candidates = [
            self._to_selector_candidate(candidate)
            for candidate in candidate_list
        ]

        selection_result = self.selector.select(model_candidates)

        selected_model_id = getattr(
            selection_result,
            "selected_model_id",
            None,
        )

        selected_model_name = None

        if selected_model_id is not None:
            for candidate in candidate_list:
                if candidate.model_id == selected_model_id:
                    selected_model_name = candidate.model_name
                    break

        warnings: list[str] = []

        if selected_model_id is None:
            warnings.append(
                "No candidate passed the controlled model-selection gates."
            )

        return SelectionPipelineResult(
            horizon=horizon,
            candidates=candidate_list,
            selection_result=selection_result,
            selected_model_id=selected_model_id,
            selected_model_name=selected_model_name,
            final_holdout_used=False,
            production_approved=False,
            warnings=warnings,
            metadata={
                "research_only": True,
                "selection_stage_only": True,
                "final_holdout_used": False,
                "production_approval_required": True,
            },
        )

    def select_for_horizons(
        self,
        candidates: Iterable[SelectionCandidate],
        horizons: Sequence[int],
    ) -> dict[int, SelectionPipelineResult]:
        """
        Select independently for each prediction horizon.
        """

        candidate_list = list(candidates)
        horizon_list = list(horizons)

        if not horizon_list:
            raise ValueError("At least one horizon is required.")

        if len(set(horizon_list)) != len(horizon_list):
            raise ValueError("Duplicate horizons are not allowed.")

        results: dict[int, SelectionPipelineResult] = {}

        for horizon in horizon_list:
            horizon_candidates = [
                candidate
                for candidate in candidate_list
                if candidate.horizon == horizon
            ]

            if not horizon_candidates:
                results[horizon] = SelectionPipelineResult(
                    horizon=horizon,
                    candidates=[],
                    selection_result=ModelSelectionResult(
                        selected_model=None,
                        ranked_candidates=[],
                        eligible_candidates=[],
                        rejected_candidates=[],
                        warnings=[
                            f"No candidates available for horizon {horizon}."
                        ],
                        final_holdout_used=False,
                    ),
                    warnings=[
                        f"No candidates available for horizon {horizon}."
                    ],
                    metadata={
                        "research_only": True,
                        "final_holdout_used": False,
                    },
                )
                continue

            results[horizon] = self.select(
                candidates=horizon_candidates,
                horizon=horizon,
            )

        return results

    @staticmethod
    def compare_selected_models(
        results: dict[int, SelectionPipelineResult],
    ) -> list[dict[str, Any]]:
        """
        Produce a compact cross-horizon selection table.
        """

        rows: list[dict[str, Any]] = []

        for horizon in sorted(results):
            result = results[horizon]

            selected = result.selected_candidate

            rows.append(
                {
                    "horizon": horizon,
                    "selected_model_id": result.selected_model_id,
                    "selected_model_name": result.selected_model_name,
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
                    "final_holdout_used": result.final_holdout_used,
                    "production_approved": result.production_approved,
                    "successful": result.successful,
                }
            )

        return rows


def run_model_selection(
    candidates: Sequence[SelectionCandidate],
    horizon: int,
    config: Optional[ModelSelectionConfig] = None,
) -> SelectionPipelineResult:
    """
    Convenience API for controlled model selection.
    """

    pipeline = SelectionPipeline(config=config)

    return pipeline.select(
        candidates=candidates,
        horizon=horizon,
    )


__all__ = [
    "SelectionCandidate",
    "SelectionPipelineResult",
    "SelectionPipeline",
    "run_model_selection",
]
