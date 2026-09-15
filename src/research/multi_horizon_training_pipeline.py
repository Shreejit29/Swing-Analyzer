"""
AI Swing Analyser — Multi-Horizon Research Training Pipeline.

Trains independent directional models for multiple prediction horizons.

Default horizons:
    1D
    3D
    5D
    10D
    20D

Research-only boundary:
    - no final holdout usage
    - no production approval
    - no calibration
    - no threshold optimization
    - no forced trading decision

Each horizon is trained independently so that the prediction problem
for a short-term move is not incorrectly treated as identical to the
prediction problem for a longer swing horizon.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .research_model_pipeline import (
    ResearchModelDataset,
)
from .walk_forward_training_pipeline import (
    WalkForwardTrainingConfig,
    WalkForwardTrainingResult,
    train_walk_forward,
)


DEFAULT_HORIZONS = (
    1,
    3,
    5,
    10,
    20,
)


@dataclass(frozen=True)
class MultiHorizonTrainingConfig:
    """Configuration for independent horizon models."""

    horizons: tuple[int, ...] = DEFAULT_HORIZONS

    walk_forward: WalkForwardTrainingConfig = (
        WalkForwardTrainingConfig()
    )

    minimum_successful_horizons: int = 1

    preferred_horizon: int = 5

    def __post_init__(self) -> None:
        normalized = tuple(
            int(horizon)
            for horizon in self.horizons
        )

        if not normalized:
            raise ValueError(
                "At least one horizon is required."
            )

        if any(
            horizon <= 0
            for horizon in normalized
        ):
            raise ValueError(
                "All horizons must be positive."
            )

        if len(set(normalized)) != len(
            normalized
        ):
            raise ValueError(
                "Duplicate horizons are not allowed."
            )

        if (
            self.minimum_successful_horizons
            < 1
        ):
            raise ValueError(
                "minimum_successful_horizons must be positive."
            )

        if (
            self.minimum_successful_horizons
            > len(normalized)
        ):
            raise ValueError(
                "minimum_successful_horizons cannot exceed "
                "the number of horizons."
            )

        if (
            self.preferred_horizon
            not in normalized
        ):
            raise ValueError(
                "preferred_horizon must be one of horizons."
            )


@dataclass
class HorizonTrainingResult:
    """Training result for one prediction horizon."""

    horizon: int

    target_column: str

    result: WalkForwardTrainingResult

    successful: bool = True

    warnings: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, object] = field(
        default_factory=dict
    )

    @property
    def mean_accuracy(self) -> float:
        return self.result.mean_accuracy

    @property
    def minimum_accuracy(self) -> float:
        return self.result.minimum_accuracy

    @property
    def accuracy_std(self) -> float:
        return self.result.accuracy_std

    @property
    def candidate_passed(self) -> bool:
        return self.result.candidate_passed


@dataclass
class MultiHorizonTrainingResult:
    """Aggregated multi-horizon research result."""

    horizons: tuple[int, ...]

    results: dict[
        int,
        HorizonTrainingResult,
    ]

    successful_horizons: tuple[int, ...]

    failed_horizons: tuple[int, ...]

    preferred_horizon: int

    candidate_passed: bool

    warnings: list[str] = field(
        default_factory=list
    )

    errors: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, object] = field(
        default_factory=dict
    )

    @property
    def successful_count(self) -> int:
        return len(
            self.successful_horizons
        )

    @property
    def failed_count(self) -> int:
        return len(
            self.failed_horizons
        )

    @property
    def production_ready(self) -> bool:
        """
        Multi-horizon research training never grants production
        approval.
        """

        return False

    def get(
        self,
        horizon: int,
    ) -> HorizonTrainingResult:
        if horizon not in self.results:
            raise KeyError(
                f"Horizon {horizon}D was not trained."
            )

        return self.results[horizon]

    def summary(self) -> dict[str, object]:
        horizon_summary = {}

        for horizon, result in (
            self.results.items()
        ):
            horizon_summary[str(horizon)] = {
                "target_column": result.target_column,
                "successful": result.successful,
                "mean_accuracy": result.mean_accuracy,
                "minimum_accuracy": (
                    result.minimum_accuracy
                ),
                "accuracy_std": result.accuracy_std,
                "candidate_passed": (
                    result.candidate_passed
                ),
            }

        return {
            "horizons": list(
                self.horizons
            ),
            "successful_horizons": list(
                self.successful_horizons
            ),
            "failed_horizons": list(
                self.failed_horizons
            ),
            "preferred_horizon": (
                self.preferred_horizon
            ),
            "candidate_passed": (
                self.candidate_passed
            ),
            "successful_count": (
                self.successful_count
            ),
            "failed_count": (
                self.failed_count
            ),
            "horizon_results": horizon_summary,
            "research_only": True,
            "production_ready": False,
            "final_holdout_used": False,
        }


class MultiHorizonTrainingPipeline:
    """
    Train independent walk-forward models for multiple horizons.
    """

    def __init__(
        self,
        *,
        config: MultiHorizonTrainingConfig
        | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else MultiHorizonTrainingConfig()
        )

    @staticmethod
    def _find_direction_target(
        dataset: ResearchModelDataset,
        horizon: int,
    ) -> str:
        """
        Find the directional target corresponding to a horizon.

        Target names are resolved from the dataset rather than being
        fabricated so that the model pipeline remains coupled to the
        actual target-generation layer.
        """

        candidates = []

        for column in (
            dataset.direction_columns
        ):
            lowered = column.lower()

            if str(horizon) in lowered:
                candidates.append(
                    column
                )

        if not candidates:
            raise ValueError(
                "No directional target found for "
                f"horizon {horizon}D."
            )

        # Deterministic selection.
        return sorted(
            candidates
        )[0]

    def _validate_dataset(
        self,
        dataset: ResearchModelDataset,
    ) -> None:
        if not isinstance(
            dataset,
            ResearchModelDataset,
        ):
            raise TypeError(
                "dataset must be ResearchModelDataset."
            )

        if dataset.rows == 0:
            raise ValueError(
                "Research model dataset is empty."
            )

        if not dataset.feature_columns:
            raise ValueError(
                "No feature columns are available."
            )

        if not dataset.direction_columns:
            raise ValueError(
                "No directional targets are available."
            )

    def train(
        self,
        dataset: ResearchModelDataset,
    ) -> MultiHorizonTrainingResult:
        """
        Train every configured horizon independently.

        A failure in one horizon does not automatically destroy
        successful results from other horizons. The final result
        records the failure explicitly.
        """

        self._validate_dataset(
            dataset
        )

        horizon_results: dict[
            int,
            HorizonTrainingResult,
        ] = {}

        successful: list[int] = []
        failed: list[int] = []
        warnings: list[str] = []
        errors: list[str] = []

        for horizon in self.config.horizons:
            try:
                target_column = (
                    self._find_direction_target(
                        dataset,
                        horizon,
                    )
                )

                result = train_walk_forward(
                    dataset,
                    target_column=target_column,
                    config=self.config.walk_forward,
                )

                horizon_result = (
                    HorizonTrainingResult(
                        horizon=horizon,
                        target_column=target_column,
                        result=result,
                        successful=True,
                        warnings=list(
                            result.warnings
                        ),
                        metadata={
                            "research_only": True,
                            "final_holdout_used": False,
                            "calibration_fitted": False,
                            "threshold_optimization_completed": False,
                            "model_selected": False,
                            "production_approved": False,
                        },
                    )
                )

                horizon_results[
                    horizon
                ] = horizon_result

                successful.append(
                    horizon
                )

                if not result.candidate_passed:
                    warnings.append(
                        f"{horizon}D candidate failed the "
                        "strict walk-forward research gate."
                    )

            except Exception as exc:
                failed.append(
                    horizon
                )

                message = (
                    f"{horizon}D training failed: "
                    f"{type(exc).__name__}: {exc}"
                )

                errors.append(
                    message
                )

        successful_tuple = tuple(
            successful
        )

        failed_tuple = tuple(
            failed
        )

        candidate_count = sum(
            result.candidate_passed
            for result in horizon_results.values()
        )

        candidate_passed = (
            len(successful)
            >= self.config.minimum_successful_horizons
            and candidate_count
            >= self.config.minimum_successful_horizons
        )

        warnings.extend(
            [
                "Each horizon is treated as an independent research problem.",
                "Final holdout data was not used.",
                "Walk-forward success does not guarantee future performance.",
                "Multi-horizon agreement must be evaluated separately before trading.",
            ]
        )

        metadata = {
            "research_only": True,
            "production_ready": False,
            "production_approved": False,
            "final_holdout_used": False,
            "calibration_fitted": False,
            "threshold_optimization_completed": False,
            "model_selected": False,
            "horizons_requested": list(
                self.config.horizons
            ),
            "successful_horizons": list(
                successful_tuple
            ),
            "failed_horizons": list(
                failed_tuple
            ),
            "minimum_successful_horizons": (
                self.config.minimum_successful_horizons
            ),
            "preferred_horizon": (
                self.config.preferred_horizon
            ),
        }

        return MultiHorizonTrainingResult(
            horizons=self.config.horizons,
            results=horizon_results,
            successful_horizons=successful_tuple,
            failed_horizons=failed_tuple,
            preferred_horizon=(
                self.config.preferred_horizon
            ),
            candidate_passed=candidate_passed,
            warnings=warnings,
            errors=errors,
            metadata=metadata,
        )


def train_multi_horizon(
    dataset: ResearchModelDataset,
    *,
    horizons: Iterable[int] | None = None,
    config: MultiHorizonTrainingConfig
    | None = None,
) -> MultiHorizonTrainingResult:
    """
    Convenience function for multi-horizon research training.
    """

    if config is None:
        if horizons is None:
            config = (
                MultiHorizonTrainingConfig()
            )
        else:
            normalized = tuple(
                int(horizon)
                for horizon in horizons
            )

            preferred = (
                5
                if 5 in normalized
                else normalized[0]
            )

            config = (
                MultiHorizonTrainingConfig(
                    horizons=normalized,
                    preferred_horizon=preferred,
                )
            )
    elif horizons is not None:
        raise ValueError(
            "Provide either config or horizons, not both."
        )

    pipeline = MultiHorizonTrainingPipeline(
        config=config
    )

    return pipeline.train(
        dataset
    )


def multi_horizon_training_summary(
    result: MultiHorizonTrainingResult,
) -> dict[str, object]:
    """Return a compact multi-horizon training summary."""

    if not isinstance(
        result,
        MultiHorizonTrainingResult,
    ):
        raise TypeError(
            "result must be MultiHorizonTrainingResult."
        )

    return result.summary()


__all__ = [
    "DEFAULT_HORIZONS",
    "MultiHorizonTrainingConfig",
    "HorizonTrainingResult",
    "MultiHorizonTrainingResult",
    "MultiHorizonTrainingPipeline",
    "train_multi_horizon",
    "multi_horizon_training_summary",
]
