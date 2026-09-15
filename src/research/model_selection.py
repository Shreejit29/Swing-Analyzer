"""
AI Swing Analyser — Controlled Model Selection.

Selects the best research candidate using development / validation
evidence only.

The final holdout is deliberately excluded from selection.

Important:
    A model with the highest score is not automatically production
    approved. Selection only identifies the candidate that should move
    to final holdout evaluation and downstream research gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


# ---------------------------------------------------------------------
# Candidate representation
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ModelCandidateScore:
    """
    Development-time score for one model/horizon candidate.
    """

    candidate_id: str

    model_name: str

    horizon: int

    validation_accuracy: float

    validation_brier: float | None = None

    walk_forward_accuracy: float | None = None

    walk_forward_accuracy_std: float | None = None

    walk_forward_min_accuracy: float | None = None

    feature_count: int | None = None

    training_samples: int | None = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    @property
    def selection_accuracy(self) -> float:

        if self.walk_forward_accuracy is not None:
            return float(
                self.walk_forward_accuracy
            )

        return float(
            self.validation_accuracy
        )


@dataclass
class ModelSelectionResult:
    """
    Result of controlled candidate selection.
    """

    selected: ModelCandidateScore | None

    ranked_candidates: list[
        ModelCandidateScore
    ] = field(
        default_factory=list
    )

    rejected_candidates: list[
        ModelCandidateScore
    ] = field(
        default_factory=list
    )

    reasons: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def successful(self) -> bool:
        return self.selected is not None

    @property
    def selected_candidate_id(self) -> str | None:

        if self.selected is None:
            return None

        return self.selected.candidate_id


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSelectionConfig:
    """
    Conservative development-time model-selection rules.
    """

    minimum_validation_accuracy: float = 0.95

    minimum_walk_forward_accuracy: float = 0.95

    maximum_walk_forward_accuracy_std: float = 0.10

    minimum_walk_forward_accuracy: float = 0.95

    maximum_brier_score: float = 0.25

    require_walk_forward: bool = True

    require_brier: bool = False

    maximum_candidates: int = 100

    prefer_simpler_models: bool = True

    feature_count_penalty: float = 0.0001

    training_sample_bonus: float = 0.0


# ---------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------


class ControlledModelSelector:
    """
    Selects one candidate using development evidence only.

    Final holdout metrics must never be supplied to this selector.
    """

    def __init__(
        self,
        config: ModelSelectionConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else ModelSelectionConfig()
        )

        self._validate_config()

    # -----------------------------------------------------------------
    # Configuration validation
    # -----------------------------------------------------------------

    def _validate_config(self) -> None:

        probability_values = (
            self.config.minimum_validation_accuracy,
            self.config.minimum_walk_forward_accuracy,
        )

        for value in probability_values:

            if not (
                0.0
                <= value
                <= 1.0
            ):
                raise ValueError(
                    "Accuracy thresholds must "
                    "be between 0 and 1."
                )

        if (
            self.config.maximum_walk_forward_accuracy_std
            < 0
        ):
            raise ValueError(
                "Maximum walk-forward accuracy "
                "standard deviation cannot be negative."
            )

        if not (
            0.0
            <= self.config.maximum_brier_score
            <= 1.0
        ):
            raise ValueError(
                "maximum_brier_score must "
                "be between 0 and 1."
            )

        if (
            self.config.maximum_candidates
            < 1
        ):
            raise ValueError(
                "maximum_candidates must be positive."
            )

        if (
            self.config.feature_count_penalty
            < 0
        ):
            raise ValueError(
                "feature_count_penalty cannot be negative."
            )

        if (
            self.config.training_sample_bonus
            < 0
        ):
            raise ValueError(
                "training_sample_bonus cannot be negative."
            )

    # -----------------------------------------------------------------
    # Selection
    # -----------------------------------------------------------------

    def select(
        self,
        candidates: Sequence[
            ModelCandidateScore
        ],
        *,
        final_holdout_used: bool = False,
    ) -> ModelSelectionResult:

        if final_holdout_used:
            raise ValueError(
                "Final holdout data cannot be used "
                "for model selection."
            )

        candidates = list(
            candidates
        )

        if not candidates:
            return ModelSelectionResult(
                selected=None,
                reasons=[
                    "No model candidates were supplied."
                ],
                metadata={
                    "final_holdout_used": False,
                    "research_only": True,
                },
            )

        if (
            len(candidates)
            > self.config.maximum_candidates
        ):
            raise ValueError(
                "Candidate count exceeds configured "
                "maximum."
            )

        self._validate_candidates(
            candidates
        )

        eligible: list[
            ModelCandidateScore
        ] = []

        rejected: list[
            ModelCandidateScore
        ] = []

        rejection_reasons: dict[
            str,
            str,
        ] = {}

        for candidate in candidates:

            reason = self._eligibility_reason(
                candidate
            )

            if reason is None:
                eligible.append(
                    candidate
                )
            else:
                rejected.append(
                    candidate
                )

                rejection_reasons[
                    candidate.candidate_id
                ] = reason

        if not eligible:

            return ModelSelectionResult(
                selected=None,
                ranked_candidates=[],
                rejected_candidates=rejected,
                reasons=[
                    "No candidate passed the "
                    "development selection gates."
                ],
                metadata={
                    "final_holdout_used": False,
                    "research_only": True,
                    "rejection_reasons": (
                        rejection_reasons
                    ),
                },
            )

        ranked = sorted(
            eligible,
            key=self._ranking_key,
            reverse=True,
        )

        selected = ranked[0]

        return ModelSelectionResult(
            selected=selected,
            ranked_candidates=ranked,
            rejected_candidates=rejected,
            reasons=[
                "Selected using development and "
                "walk-forward evidence only.",
                "Final holdout was not used.",
            ],
            metadata={
                "final_holdout_used": False,
                "research_only": True,
                "selection_method": (
                    "development_walk_forward"
                ),
                "candidate_count": len(
                    candidates
                ),
                "eligible_count": len(
                    eligible
                ),
                "rejection_reasons": (
                    rejection_reasons
                ),
            },
        )

    # -----------------------------------------------------------------
    # Candidate validation
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_candidates(
        candidates: Sequence[
            ModelCandidateScore
        ],
    ) -> None:

        seen: set[str] = set()

        for candidate in candidates:

            if not candidate.candidate_id:
                raise ValueError(
                    "candidate_id cannot be empty."
                )

            if (
                candidate.candidate_id
                in seen
            ):
                raise ValueError(
                    "Candidate IDs must be unique."
                )

            seen.add(
                candidate.candidate_id
            )

            if not candidate.model_name:
                raise ValueError(
                    "model_name cannot be empty."
                )

            if candidate.horizon <= 0:
                raise ValueError(
                    "horizon must be positive."
                )

            if not (
                0.0
                <= candidate.validation_accuracy
                <= 1.0
            ):
                raise ValueError(
                    "validation_accuracy must "
                    "be between 0 and 1."
                )

            optional_probabilities = (
                candidate.walk_forward_accuracy,
            )

            for value in optional_probabilities:

                if value is None:
                    continue

                if not (
                    0.0
                    <= value
                    <= 1.0
                ):
                    raise ValueError(
                        "Walk-forward accuracy must "
                        "be between 0 and 1."
                    )

            if (
                candidate.walk_forward_accuracy_std
                is not None
                and candidate.walk_forward_accuracy_std
                < 0
            ):
                raise ValueError(
                    "Walk-forward accuracy standard "
                    "deviation cannot be negative."
                )

            if (
                candidate.walk_forward_min_accuracy
                is not None
                and not (
                    0.0
                    <= candidate.walk_forward_min_accuracy
                    <= 1.0
                )
            ):
                raise ValueError(
                    "Minimum walk-forward accuracy "
                    "must be between 0 and 1."
                )

            if (
                candidate.validation_brier
                is not None
                and not (
                    0.0
                    <= candidate.validation_brier
                    <= 1.0
                )
            ):
                raise ValueError(
                    "validation_brier must "
                    "be between 0 and 1."
                )

            if (
                candidate.feature_count
                is not None
                and candidate.feature_count
                < 1
            ):
                raise ValueError(
                    "feature_count must be positive."
                )

            if (
                candidate.training_samples
                is not None
                and candidate.training_samples
                < 1
            ):
                raise ValueError(
                    "training_samples must be positive."
                )

    # -----------------------------------------------------------------
    # Eligibility
    # -----------------------------------------------------------------

    def _eligibility_reason(
        self,
        candidate: ModelCandidateScore,
    ) -> str | None:

        if (
            candidate.validation_accuracy
            < self.config.minimum_validation_accuracy
        ):
            return (
                "Validation accuracy is below "
                "the configured threshold."
            )

        if self.config.require_walk_forward:

            if (
                candidate.walk_forward_accuracy
                is None
            ):
                return (
                    "Walk-forward accuracy is missing."
                )

            if (
                candidate.walk_forward_accuracy
                < self.config.minimum_walk_forward_accuracy
            ):
                return (
                    "Walk-forward accuracy is below "
                    "the configured threshold."
                )

            if (
                candidate.walk_forward_accuracy_std
                is None
            ):
                return (
                    "Walk-forward accuracy stability "
                    "is missing."
                )

            if (
                candidate.walk_forward_accuracy_std
                > self.config.maximum_walk_forward_accuracy_std
            ):
                return (
                    "Walk-forward accuracy is unstable."
                )

            if (
                candidate.walk_forward_min_accuracy
                is not None
                and candidate.walk_forward_min_accuracy
                < self.config.minimum_walk_forward_accuracy
            ):
                return (
                    "At least one walk-forward fold "
                    "is below the minimum accuracy."
                )

        if self.config.require_brier:

            if (
                candidate.validation_brier
                is None
            ):
                return (
                    "Brier score is required but missing."
                )

            if (
                candidate.validation_brier
                > self.config.maximum_brier_score
            ):
                return (
                    "Brier score exceeds the configured maximum."
                )

        return None

    # -----------------------------------------------------------------
    # Ranking
    # -----------------------------------------------------------------

    def _ranking_key(
        self,
        candidate: ModelCandidateScore,
    ) -> float:

        score = (
            candidate.selection_accuracy
        )

        if (
            candidate.validation_brier
            is not None
        ):
            score -= (
                candidate.validation_brier
                * 0.01
            )

        if (
            candidate.walk_forward_accuracy_std
            is not None
        ):
            score -= (
                candidate.walk_forward_accuracy_std
                * 0.10
            )

        if (
            self.config.prefer_simpler_models
            and candidate.feature_count is not None
        ):
            score -= (
                candidate.feature_count
                * self.config.feature_count_penalty
            )

        if (
            candidate.training_samples
            is not None
            and self.config.training_sample_bonus
            > 0
        ):
            score += (
                np.log1p(
                    candidate.training_samples
                )
                * self.config.training_sample_bonus
            )

        return float(score)

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------

    @staticmethod
    def compare(
        first: ModelCandidateScore,
        second: ModelCandidateScore,
    ) -> int:

        first_accuracy = (
            first.selection_accuracy
        )

        second_accuracy = (
            second.selection_accuracy
        )

        if first_accuracy > second_accuracy:
            return 1

        if first_accuracy < second_accuracy:
            return -1

        first_brier = (
            first.validation_brier
            if first.validation_brier
            is not None
            else np.inf
        )

        second_brier = (
            second.validation_brier
            if second.validation_brier
            is not None
            else np.inf
        )

        if first_brier < second_brier:
            return 1

        if first_brier > second_brier:
            return -1

        return 0

    @staticmethod
    def summary(
        result: ModelSelectionResult,
    ) -> dict[str, Any]:

        return {
            "successful": result.successful,
            "selected_candidate_id": (
                result.selected_candidate_id
            ),
            "ranked_candidate_count": len(
                result.ranked_candidates
            ),
            "rejected_candidate_count": len(
                result.rejected_candidates
            ),
            "final_holdout_used": result.metadata.get(
                "final_holdout_used",
                False,
            ),
            "research_only": result.metadata.get(
                "research_only",
                True,
            ),
            "reasons": list(
                result.reasons
            ),
        }


# ---------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------


def select_best_model(
    candidates: Iterable[
        ModelCandidateScore
    ],
    *,
    config: ModelSelectionConfig | None = None,
    final_holdout_used: bool = False,
) -> ModelSelectionResult:

    selector = ControlledModelSelector(
        config=config
    )

    return selector.select(
        list(candidates),
        final_holdout_used=final_holdout_used,
    )


__all__ = [
    "ModelCandidateScore",
    "ModelSelectionResult",
    "ModelSelectionConfig",
    "ControlledModelSelector",
    "select_best_model",
]
