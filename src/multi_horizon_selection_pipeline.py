"""
AI Swing Analyser — Multi-Horizon Model Selection Pipeline.

Connects multi-horizon walk-forward training results to the
development-only model-selection boundary.

Important rules
---------------
1. Final holdout is never used.
2. Selection is based only on development evidence.
3. A model failing walk-forward stability cannot be selected.
4. Selection does not grant production approval.
5. Calibration does not happen here.
6. Trading thresholds are not optimized here.
7. Each horizon remains an independent prediction problem.

Pipeline:

    Multi-Horizon Training
             ↓
    Walk-Forward Evidence
             ↓
    Controlled Selection
             ↓
    Selected Research Candidate
             ↓
    Calibration / Range / Regime / Backtest
             ↓
    Final Holdout
             ↓
    Production Approval
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.research.model_selection import (
    ControlledModelSelector,
    ModelSelectionConfig,
    ModelSelectionResult,
)
from src.research.multi_horizon_training_pipeline import (
    HorizonTrainingResult,
    MultiHorizonTrainingResult,
)


@dataclass(frozen=True)
class MultiHorizonSelectionConfig:
    """Configuration for development-only model selection."""

    selection: ModelSelectionConfig = field(
        default_factory=ModelSelectionConfig
    )

    require_candidate_pass: bool = True

    require_walk_forward_evidence: bool = True

    allow_failed_horizons: bool = True

    minimum_selected_horizons: int = 1

    def __post_init__(self) -> None:
        if not isinstance(
            self.selection,
            ModelSelectionConfig,
        ):
            raise TypeError(
                "selection must be ModelSelectionConfig."
            )

        if (
            self.minimum_selected_horizons
            < 1
        ):
            raise ValueError(
                "minimum_selected_horizons must be positive."
            )


@dataclass
class HorizonSelectionResult:
    """Selection result for one horizon."""

    horizon: int

    target_column: str

    selected_model_id: str | None

    selection_result: Any

    selected: bool

    candidate_passed: bool

    warnings: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, object] = field(
        default_factory=dict
    )


@dataclass
class MultiHorizonSelectionResult:
    """Aggregated multi-horizon selection result."""

    horizons: tuple[int, ...]

    results: dict[
        int,
        HorizonSelectionResult,
    ]

    selected_horizons: tuple[int, ...]

    rejected_horizons: tuple[int, ...]

    candidate_passed: bool

    production_ready: bool

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
    def selected_count(self) -> int:
        return len(
            self.selected_horizons
        )

    @property
    def rejected_count(self) -> int:
        return len(
            self.rejected_horizons
        )

    @property
    def final_holdout_used(self) -> bool:
        return bool(
            self.metadata.get(
                "final_holdout_used",
                False,
            )
        )

    def get(
        self,
        horizon: int,
    ) -> HorizonSelectionResult:
        if horizon not in self.results:
            raise KeyError(
                f"Horizon {horizon}D was not evaluated."
            )

        return self.results[horizon]

    def summary(self) -> dict[str, object]:
        horizon_summary: dict[
            str,
            dict[str, object],
        ] = {}

        for horizon, result in (
            self.results.items()
        ):
            horizon_summary[
                str(horizon)
            ] = {
                "target_column": result.target_column,
                "selected_model_id": (
                    result.selected_model_id
                ),
                "selected": result.selected,
                "candidate_passed": (
                    result.candidate_passed
                ),
            }

        return {
            "horizons": list(
                self.horizons
            ),
            "selected_horizons": list(
                self.selected_horizons
            ),
            "rejected_horizons": list(
                self.rejected_horizons
            ),
            "selected_count": (
                self.selected_count
            ),
            "rejected_count": (
                self.rejected_count
            ),
            "candidate_passed": (
                self.candidate_passed
            ),
            "production_ready": False,
            "final_holdout_used": (
                self.final_holdout_used
            ),
            "research_only": True,
            "horizon_results": horizon_summary,
        }


class MultiHorizonSelectionPipeline:
    """
    Select the strongest development-only candidate for each horizon.

    This class intentionally does not perform model fitting.
    It consumes already-generated research evidence.
    """

    def __init__(
        self,
        *,
        config: MultiHorizonSelectionConfig
        | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else MultiHorizonSelectionConfig()
        )

    @staticmethod
    def _validate_training_result(
        result: MultiHorizonTrainingResult,
    ) -> None:
        if not isinstance(
            result,
            MultiHorizonTrainingResult,
        ):
            raise TypeError(
                "result must be MultiHorizonTrainingResult."
            )

        if not result.horizons:
            raise ValueError(
                "Training result contains no horizons."
            )

        if result.metadata.get(
            "final_holdout_used",
            False,
        ):
            raise ValueError(
                "Model selection cannot use a result that "
                "claims final holdout usage."
            )

    @staticmethod
    def _candidate_from_training(
        horizon_result: HorizonTrainingResult,
        horizon: int,
    ) -> Any:
        """
        Convert a trained horizon result into the candidate shape
        expected by the controlled selector.

        A lightweight object is deliberately used here so this layer
        does not duplicate the selector's candidate schema.
        """

        class Candidate:
            pass

        candidate = Candidate()

        candidate.model_id = (
            f"research_{horizon}d_"
            f"{horizon_result.result.metadata.get(
                'model_type',
                'model',
            )}"
        )

        candidate.horizon = horizon

        candidate.target_column = (
            horizon_result.target_column
        )

        candidate.validation_accuracy = (
            horizon_result.mean_accuracy
        )

        candidate.walk_forward_accuracy = (
            horizon_result.mean_accuracy
        )

        candidate.minimum_walk_forward_accuracy = (
            horizon_result.minimum_accuracy
        )

        candidate.walk_forward_accuracy_std = (
            horizon_result.accuracy_std
        )

        candidate.brier_score = (
            None
        )

        candidate.final_holdout_used = False

        candidate.production_approved = False

        candidate.model = (
            horizon_result.result.folds[-1].model
            if horizon_result.result.folds
            else None
        )

        candidate.preprocessor = (
            horizon_result.result.folds[-1].preprocessor
            if horizon_result.result.folds
            else None
        )

        candidate.feature_columns = (
            horizon_result.result.feature_columns
        )

        candidate.research_only = True

        candidate.metadata = {
            "horizon": horizon,
            "target_column": (
                horizon_result.target_column
            ),
            "research_only": True,
            "final_holdout_used": False,
            "production_approved": False,
            "walk_forward_validated": True,
        }

        return candidate

    @staticmethod
    def _selection_model_id(
        selection_result: Any,
    ) -> str | None:
        """
        Extract the selected model identifier without assuming
        one exact historical implementation of ModelSelectionResult.
        """

        for attribute in (
            "selected_model_id",
            "selected_candidate_id",
            "model_id",
        ):
            value = getattr(
                selection_result,
                attribute,
                None,
            )

            if value is not None:
                return str(value)

        selected = getattr(
            selection_result,
            "selected",
            None,
        )

        if selected is not None:
            value = getattr(
                selected,
                "model_id",
                None,
            )

            if value is not None:
                return str(value)

        return None

    def _select_one(
        self,
        horizon_result: HorizonTrainingResult,
        horizon: int,
    ) -> HorizonSelectionResult:
        """Select the best candidate for one horizon."""

        candidate_passed = (
            horizon_result.candidate_passed
        )

        warnings: list[str] = []

        if (
            self.config.require_candidate_pass
            and not candidate_passed
        ):
            warnings.append(
                f"{horizon}D candidate failed the "
                "walk-forward candidate gate."
            )

            return HorizonSelectionResult(
                horizon=horizon,
                target_column=(
                    horizon_result.target_column
                ),
                selected_model_id=None,
                selection_result=None,
                selected=False,
                candidate_passed=False,
                warnings=warnings,
                metadata={
                    "research_only": True,
                    "final_holdout_used": False,
                    "selection_performed": False,
                },
            )

        if (
            self.config.require_walk_forward_evidence
            and not horizon_result.result.folds
        ):
            warnings.append(
                f"{horizon}D has no walk-forward evidence."
            )

            return HorizonSelectionResult(
                horizon=horizon,
                target_column=(
                    horizon_result.target_column
                ),
                selected_model_id=None,
                selection_result=None,
                selected=False,
                candidate_passed=False,
                warnings=warnings,
                metadata={
                    "research_only": True,
                    "final_holdout_used": False,
                    "selection_performed": False,
                },
            )

        candidate = (
            self._candidate_from_training(
                horizon_result,
                horizon,
            )
        )

        selector = ControlledModelSelector(
            config=self.config.selection
        )

        try:
            selection_result = selector.select(
                [candidate]
            )
        except AttributeError:
            # Compatibility path for selector implementations that
            # expose a differently named public selection method.
            selection_result = selector.select_best(
                [candidate]
            )

        selected_model_id = (
            self._selection_model_id(
                selection_result
            )
        )

        selected = (
            selected_model_id is not None
        )

        if not selected:
            warnings.append(
                f"{horizon}D produced no selected model."
            )

        return HorizonSelectionResult(
            horizon=horizon,
            target_column=(
                horizon_result.target_column
            ),
            selected_model_id=(
                selected_model_id
            ),
            selection_result=selection_result,
            selected=selected,
            candidate_passed=candidate_passed,
            warnings=warnings,
            metadata={
                "research_only": True,
                "final_holdout_used": False,
                "selection_performed": True,
                "production_approved": False,
            },
        )

    def select(
        self,
        training_result: MultiHorizonTrainingResult,
    ) -> MultiHorizonSelectionResult:
        """
        Select development candidates independently by horizon.
        """

        self._validate_training_result(
            training_result
        )

        results: dict[
            int,
            HorizonSelectionResult,
        ] = {}

        selected: list[int] = []
        rejected: list[int] = []

        warnings: list[str] = []
        errors: list[str] = []

        for horizon in training_result.horizons:

            horizon_result = (
                training_result.results.get(
                    horizon
                )
            )

            if horizon_result is None:
                rejected.append(
                    horizon
                )

                errors.append(
                    f"{horizon}D has no training result."
                )

                continue

            if not horizon_result.successful:
                rejected.append(
                    horizon
                )

                warnings.append(
                    f"{horizon}D training was unsuccessful."
                )

                continue

            try:
                selection = self._select_one(
                    horizon_result,
                    horizon,
                )

                results[
                    horizon
                ] = selection

                if selection.selected:
                    selected.append(
                        horizon
                    )
                else:
                    rejected.append(
                        horizon
                    )

                warnings.extend(
                    selection.warnings
                )

            except Exception as exc:
                rejected.append(
                    horizon
                )

                errors.append(
                    f"{horizon}D selection failed: "
                    f"{type(exc).__name__}: {exc}"
                )

        candidate_passed = (
            len(selected)
            >= self.config.minimum_selected_horizons
        )

        warnings.extend(
            [
                "Model selection uses development evidence only.",
                "Final holdout data was not used.",
                "Selection does not grant production approval.",
                "Probability calibration is performed in a later research stage.",
            ]
        )

        metadata = {
            "research_only": True,
            "production_ready": False,
            "production_approved": False,
            "final_holdout_used": False,
            "selection_completed": True,
            "calibration_fitted": False,
            "threshold_optimization_completed": False,
            "selected_horizons": list(
                selected
            ),
            "rejected_horizons": list(
                rejected
            ),
        }

        return MultiHorizonSelectionResult(
            horizons=training_result.horizons,
            results=results,
            selected_horizons=tuple(
                selected
            ),
            rejected_horizons=tuple(
                rejected
            ),
            candidate_passed=(
                candidate_passed
            ),
            production_ready=False,
            warnings=warnings,
            errors=errors,
            metadata=metadata,
        )


def select_multi_horizon_models(
    training_result: MultiHorizonTrainingResult,
    *,
    config: MultiHorizonSelectionConfig
    | None = None,
) -> MultiHorizonSelectionResult:
    """Convenience API for multi-horizon model selection."""

    pipeline = MultiHorizonSelectionPipeline(
        config=config
    )

    return pipeline.select(
        training_result
    )


def multi_horizon_selection_summary(
    result: MultiHorizonSelectionResult,
) -> dict[str, object]:
    """Return a compact selection summary."""

    if not isinstance(
        result,
        MultiHorizonSelectionResult,
    ):
        raise TypeError(
            "result must be MultiHorizonSelectionResult."
        )

    return result.summary()


__all__ = [
    "MultiHorizonSelectionConfig",
    "HorizonSelectionResult",
    "MultiHorizonSelectionResult",
    "MultiHorizonSelectionPipeline",
    "select_multi_horizon_models",
    "multi_horizon_selection_summary",
]
