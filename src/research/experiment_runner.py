"""
AI Swing Analyser — Research Experiment Runner.

Coordinates reproducible candidate-model experiments.

Important:
    The final holdout is NOT used to select the best candidate.

Candidate selection is based only on development/validation evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import pandas as pd

from src.models.classifier import ClassifierConfig
from src.models.preprocessing import PreprocessorConfig
from src.features.selection import FeatureSelectionConfig
from src.research.config import ResearchPipelineConfig
from src.research.model_development import (
    ModelDevelopmentResult,
    ModelDevelopmentOrchestrator,
)


@dataclass(frozen=True)
class ExperimentCandidate:
    """
    Description of a candidate experiment.
    """

    model_name: str
    horizon: int
    target_column: str


@dataclass
class ExperimentRunResult:
    """
    Collection of candidate experiment results.
    """

    candidates: list[ModelDevelopmentResult] = field(
        default_factory=list
    )

    selected_candidate: ModelDevelopmentResult | None = None

    selection_metric: str = "validation_accuracy"

    holdout_used_for_selection: bool = False

    warnings: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def successful(self) -> bool:
        return self.selected_candidate is not None

    @property
    def best_validation_accuracy(self) -> float | None:
        if not self.candidates:
            return None

        return max(
            candidate.validation_accuracy
            for candidate in self.candidates
        )

    def summary(self) -> dict[str, Any]:
        return {
            "candidate_count": self.candidate_count,
            "successful": self.successful,
            "selection_metric": self.selection_metric,
            "holdout_used_for_selection": (
                self.holdout_used_for_selection
            ),
            "best_validation_accuracy": (
                self.best_validation_accuracy
            ),
            "selected_candidate": (
                self.selected_candidate.summary()
                if self.selected_candidate is not None
                else None
            ),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


class ResearchExperimentRunner:
    """
    Run and compare candidate models.

    The runner deliberately treats 95% accuracy as a research gate,
    not as evidence of production readiness.
    """

    def __init__(
        self,
        config: ResearchPipelineConfig | None = None,
        orchestrator: ModelDevelopmentOrchestrator | None = None,
    ) -> None:

        self.config = config

        if orchestrator is not None:
            self.orchestrator = orchestrator
        else:
            self.orchestrator = (
                ModelDevelopmentOrchestrator(
                    config=config,
                    preprocessor_config=PreprocessorConfig(),
                    classifier_config=ClassifierConfig(),
                    feature_selection_config=(
                        FeatureSelectionConfig()
                    ),
                )
            )

    # -----------------------------------------------------------------
    # Candidate validation
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_candidates(
        candidates: Sequence[ExperimentCandidate],
    ) -> None:

        if not candidates:
            raise ValueError(
                "At least one experiment candidate is required."
            )

        seen: set[tuple[str, int, str]] = set()

        for candidate in candidates:

            if not candidate.model_name:
                raise ValueError(
                    "Candidate model name must not be empty."
                )

            if candidate.horizon <= 0:
                raise ValueError(
                    "Candidate horizon must be positive."
                )

            if not candidate.target_column:
                raise ValueError(
                    "Candidate target column must not be empty."
                )

            key = (
                candidate.model_name,
                candidate.horizon,
                candidate.target_column,
            )

            if key in seen:
                raise ValueError(
                    "Duplicate experiment candidate: "
                    f"{key}"
                )

            seen.add(key)

    # -----------------------------------------------------------------
    # Target resolution
    # -----------------------------------------------------------------

    @staticmethod
    def _resolve_target(
        data: pd.DataFrame,
        candidate: ExperimentCandidate,
    ) -> str:

        target = candidate.target_column

        if target in data.columns:
            return target

        raise ValueError(
            f"Target '{target}' does not exist in dataset."
        )

    # -----------------------------------------------------------------
    # Candidate execution
    # -----------------------------------------------------------------

    def run_candidate(
        self,
        data: pd.DataFrame,
        feature_columns: Sequence[str],
        candidate: ExperimentCandidate,
        apply_feature_selection: bool = True,
    ) -> ModelDevelopmentResult:

        target = self._resolve_target(
            data,
            candidate,
        )

        return self.orchestrator.fit_candidate(
            data=data,
            feature_columns=feature_columns,
            target_column=target,
            horizon=candidate.horizon,
            model_name=candidate.model_name,
            apply_feature_selection=(
                apply_feature_selection
            ),
        )

    # -----------------------------------------------------------------
    # Candidate selection
    # -----------------------------------------------------------------

    @staticmethod
    def select_best_candidate(
        results: Sequence[ModelDevelopmentResult],
    ) -> ModelDevelopmentResult:

        if not results:
            raise ValueError(
                "Cannot select from zero candidate results."
            )

        # IMPORTANT:
        # Selection uses validation accuracy only.
        # Final holdout performance is intentionally unavailable here.
        ranked = sorted(
            results,
            key=lambda result: (
                result.validation_accuracy,
                -len(result.feature_columns),
            ),
            reverse=True,
        )

        return ranked[0]

    # -----------------------------------------------------------------
    # Full experiment
    # -----------------------------------------------------------------

    def run(
        self,
        data: pd.DataFrame,
        feature_columns: Sequence[str],
        candidates: Sequence[ExperimentCandidate],
        apply_feature_selection: bool = True,
    ) -> ExperimentRunResult:

        self._validate_candidates(
            candidates
        )

        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                "Cannot run experiments on an empty dataset."
            )

        results: list[
            ModelDevelopmentResult
        ] = []

        errors: list[str] = []

        for candidate in candidates:

            try:
                result = self.run_candidate(
                    data=data,
                    feature_columns=feature_columns,
                    candidate=candidate,
                    apply_feature_selection=(
                        apply_feature_selection
                    ),
                )

                results.append(
                    result
                )

            except Exception as exc:
                errors.append(
                    f"{candidate.model_name}/"
                    f"{candidate.horizon}/"
                    f"{candidate.target_column}: "
                    f"{exc}"
                )

        if not results:
            raise RuntimeError(
                "All candidate experiments failed.\n"
                + "\n".join(errors)
            )

        selected = (
            self.select_best_candidate(
                results
            )
        )

        warnings = list(
            errors
        )

        if selected.validation_accuracy < 0.95:
            warnings.append(
                "No candidate achieved the strict "
                "95% validation accuracy research gate."
            )

        warnings.append(
            "Candidate selection uses development/validation "
            "data only. Final holdout remains untouched."
        )

        return ExperimentRunResult(
            candidates=results,
            selected_candidate=selected,
            selection_metric=(
                "validation_accuracy"
            ),
            holdout_used_for_selection=False,
            warnings=warnings,
            metadata={
                "research_only": True,
                "candidate_selection_frozen": True,
                "final_holdout_reserved": True,
                "candidate_count": len(results),
            },
        )

    # -----------------------------------------------------------------
    # Standard candidate generation
    # -----------------------------------------------------------------

    @staticmethod
    def default_candidates(
        horizons: Sequence[int] = (
            1,
            3,
            5,
            10,
            20,
        ),
    ) -> list[ExperimentCandidate]:

        candidates: list[
            ExperimentCandidate
        ] = []

        models = [
            "logistic_regression",
            "random_forest",
            "gradient_boosting",
        ]

        for horizon in horizons:
            for model_name in models:
                candidates.append(
                    ExperimentCandidate(
                        model_name=model_name,
                        horizon=int(horizon),
                        target_column=(
                            f"Direction_{int(horizon)}"
                        ),
                    )
                )

        return candidates


__all__ = [
    "ExperimentCandidate",
    "ExperimentRunResult",
    "ResearchExperimentRunner",
]
