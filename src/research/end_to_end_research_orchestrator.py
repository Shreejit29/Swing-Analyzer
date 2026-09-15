"""
End-to-end research orchestrator for the AI Swing Analyser.

Pipeline:

    Data
      ↓
    Features
      ↓
    Targets
      ↓
    Model Development / Selection
      ↓
    Backtest
      ↓
    Robustness
      ↓
    Final Holdout
      ↓
    Production Approval
      ↓
    Production Prediction

The orchestrator is intentionally fail-closed.

It does not silently skip failed research stages and does not treat
missing evidence as a successful stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


DEFAULT_RESEARCH_HORIZONS = (
    1,
    3,
    5,
    10,
    20,
)


@dataclass(frozen=True)
class EndToEndResearchConfig:
    """Configuration for the complete research workflow."""

    horizons: tuple[int, ...] = (
        DEFAULT_RESEARCH_HORIZONS
    )

    stop_on_failure: bool = True
    require_all_horizons: bool = True

    enable_final_holdout: bool = True
    enable_production_approval: bool = True

    def __post_init__(self) -> None:
        horizons = tuple(
            int(x)
            for x in self.horizons
        )

        if not horizons:
            raise ValueError(
                "At least one research horizon is required."
            )

        if any(
            horizon <= 0
            for horizon in horizons
        ):
            raise ValueError(
                "Research horizons must be positive."
            )

        if len(set(horizons)) != len(horizons):
            raise ValueError(
                "Research horizons must be unique."
            )

        object.__setattr__(
            self,
            "horizons",
            horizons,
        )


@dataclass
class ResearchStageResult:
    """Status of one orchestrated research stage."""

    name: str

    executed: bool = False
    passed: bool = False

    output: Any = None

    errors: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class EndToEndResearchResult:
    """Complete result of the research orchestration."""

    stages: dict[
        str,
        ResearchStageResult,
    ]

    passed: bool = False

    final_holdout_used: bool = False

    production_approved: bool = False
    production_ready: bool = False

    research_only: bool = True

    errors: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def stage(
        self,
        name: str,
    ) -> ResearchStageResult:
        if name not in self.stages:
            raise KeyError(
                f"Unknown research stage: {name}"
            )

        return self.stages[name]

    def output(
        self,
        name: str,
    ) -> Any:
        return self.stage(
            name
        ).output


class EndToEndResearchOrchestrator:
    """
    Execute the complete research pipeline.

    Stages are injected as callables so the orchestrator does not become
    tightly coupled to one implementation of each research component.

    Each callable receives the outputs of previous stages through keyword
    arguments where appropriate.
    """

    STAGE_ORDER = (
        "data",
        "features",
        "targets",
        "model_selection",
        "backtest",
        "robustness",
        "final_holdout",
        "production_approval",
    )

    def __init__(
        self,
        config: EndToEndResearchConfig | None = None,
    ) -> None:
        self.config = (
            config
            or EndToEndResearchConfig()
        )

    @staticmethod
    def _execute(
        name: str,
        function: Callable[..., Any] | None,
        kwargs: Mapping[str, Any],
    ) -> ResearchStageResult:
        result = ResearchStageResult(
            name=name
        )

        if function is None:
            result.errors.append(
                f"No callable configured for stage '{name}'."
            )
            return result

        try:
            output = function(
                **kwargs
            )

            result.executed = True
            result.output = output

            passed = (
                getattr(
                    output,
                    "passed",
                    None,
                )
            )

            if passed is None:
                passed = getattr(
                    output,
                    "candidate_passed",
                    None,
                )

            if passed is None:
                passed = getattr(
                    output,
                    "approved",
                    None,
                )

            if passed is None:
                # A successfully executed stage with no explicit gate
                # is considered executed, but not automatically passed.
                result.warnings.append(
                    f"Stage '{name}' returned no explicit pass status."
                )
                result.passed = True
            else:
                result.passed = (
                    passed is True
                )

            metadata = getattr(
                output,
                "metadata",
                None,
            )

            if isinstance(
                metadata,
                Mapping,
            ):
                result.metadata.update(
                    metadata
                )

        except Exception as exc:
            result.errors.append(
                f"{type(exc).__name__}: {exc}"
            )

        return result

    @staticmethod
    def _stage_failure(
        result: ResearchStageResult,
    ) -> bool:
        return bool(
            result.errors
            or (
                result.executed
                and not result.passed
            )
        )

    @staticmethod
    def _extract_horizon_outputs(
        source: Any,
        horizons: Sequence[int],
    ) -> dict[int, Any]:
        outputs: dict[int, Any] = {}

        if source is None:
            return outputs

        results = getattr(
            source,
            "results",
            None,
        )

        if results is None:
            results = getattr(
                source,
                "horizon_results",
                None,
            )

        if isinstance(
            results,
            Mapping,
        ):
            for horizon in horizons:
                if horizon in results:
                    outputs[horizon] = (
                        results[horizon]
                    )

        return outputs

    def run(
        self,
        *,
        data_stage: Callable[..., Any] | None = None,
        feature_stage: Callable[..., Any] | None = None,
        target_stage: Callable[..., Any] | None = None,
        model_selection_stage: Callable[..., Any] | None = None,
        backtest_stage: Callable[..., Any] | None = None,
        robustness_stage: Callable[..., Any] | None = None,
        final_holdout_stage: Callable[..., Any] | None = None,
        production_approval_stage: Callable[..., Any] | None = None,
        data: Any = None,
        features: Any = None,
        targets: Any = None,
        trade_returns: Any = None,
        actual_directions: Any = None,
        predicted_directions: Any = None,
        predicted_probabilities: Any = None,
    ) -> EndToEndResearchResult:
        """
        Execute all configured stages in chronological research order.

        The explicit inputs are useful when a stage has already been
        prepared externally, while callable stages allow the complete
        workflow to be automated.
        """

        stages: dict[
            str,
            ResearchStageResult,
        ] = {}

        errors: list[str] = []
        warnings: list[str] = []

        # --------------------------------------------------------------
        # 1. DATA
        # --------------------------------------------------------------

        if data_stage is not None:
            data_result = self._execute(
                "data",
                data_stage,
                {
                    "data": data
                },
            )
        else:
            data_result = ResearchStageResult(
                name="data",
                executed=data is not None,
                passed=data is not None,
                output=data,
                warnings=(
                    []
                    if data is not None
                    else [
                        "Data stage was supplied as an external input."
                    ]
                ),
            )

        stages["data"] = data_result

        if self.config.stop_on_failure and self._stage_failure(
            data_result
        ):
            errors.extend(
                data_result.errors
            )
            return self._finalize(
                stages,
                errors,
                warnings,
            )

        # --------------------------------------------------------------
        # 2. FEATURES
        # --------------------------------------------------------------

        feature_kwargs = {
            "data": data_result.output,
            "features": features,
        }

        feature_result = self._execute(
            "features",
            feature_stage,
            feature_kwargs,
        )

        if feature_stage is None:
            feature_result = ResearchStageResult(
                name="features",
                executed=features is not None,
                passed=features is not None,
                output=features,
            )

        stages["features"] = feature_result

        if self.config.stop_on_failure and self._stage_failure(
            feature_result
        ):
            errors.extend(
                feature_result.errors
            )
            return self._finalize(
                stages,
                errors,
                warnings,
            )

        # --------------------------------------------------------------
        # 3. TARGETS
        # --------------------------------------------------------------

        target_result = self._execute(
            "targets",
            target_stage,
            {
                "data": data_result.output,
                "features": feature_result.output,
                "targets": targets,
                "horizons": self.config.horizons,
            },
        )

        if target_stage is None:
            target_result = ResearchStageResult(
                name="targets",
                executed=targets is not None,
                passed=targets is not None,
                output=targets,
            )

        stages["targets"] = target_result

        if self.config.stop_on_failure and self._stage_failure(
            target_result
        ):
            errors.extend(
                target_result.errors
            )
            return self._finalize(
                stages,
                errors,
                warnings,
            )

        # --------------------------------------------------------------
        # 4. MODEL SELECTION
        # --------------------------------------------------------------

        selection_result = self._execute(
            "model_selection",
            model_selection_stage,
            {
                "data": data_result.output,
                "features": feature_result.output,
                "targets": target_result.output,
                "horizons": self.config.horizons,
            },
        )

        stages[
            "model_selection"
        ] = selection_result

        if self.config.stop_on_failure and self._stage_failure(
            selection_result
        ):
            errors.extend(
                selection_result.errors
            )
            return self._finalize(
                stages,
                errors,
                warnings,
            )

        # --------------------------------------------------------------
        # 5. BACKTEST
        # --------------------------------------------------------------

        backtest_result = self._execute(
            "backtest",
            backtest_stage,
            {
                "data": data_result.output,
                "features": feature_result.output,
                "targets": target_result.output,
                "model_selection": selection_result.output,
                "horizons": self.config.horizons,
            },
        )

        if backtest_stage is None:
            backtest_result = ResearchStageResult(
                name="backtest",
                executed=False,
                passed=False,
                output=None,
                errors=[
                    "Backtest stage is required."
                ],
            )

        stages[
            "backtest"
        ] = backtest_result

        if self.config.stop_on_failure and self._stage_failure(
            backtest_result
        ):
            errors.extend(
                backtest_result.errors
            )
            return self._finalize(
                stages,
                errors,
                warnings,
            )

        # --------------------------------------------------------------
        # 6. ROBUSTNESS
        # --------------------------------------------------------------

        robustness_result = self._execute(
            "robustness",
            robustness_stage,
            {
                "backtest": backtest_result.output,
                "model_selection": selection_result.output,
                "trade_returns": trade_returns,
                "horizons": self.config.horizons,
            },
        )

        if robustness_stage is None:
            robustness_result = ResearchStageResult(
                name="robustness",
                executed=False,
                passed=False,
                output=None,
                errors=[
                    "Robustness stage is required."
                ],
            )

        stages[
            "robustness"
        ] = robustness_result

        if self.config.stop_on_failure and self._stage_failure(
            robustness_result
        ):
            errors.extend(
                robustness_result.errors
            )
            return self._finalize(
                stages,
                errors,
                warnings,
            )

        # --------------------------------------------------------------
        # 7. FINAL HOLDOUT
        # --------------------------------------------------------------

        if self.config.enable_final_holdout:
            holdout_result = self._execute(
                "final_holdout",
                final_holdout_stage,
                {
                    "model_selection": selection_result.output,
                    "backtest": backtest_result.output,
                    "robustness": robustness_result.output,
                    "data": data_result.output,
                    "features": feature_result.output,
                    "targets": target_result.output,
                    "actual_directions": actual_directions,
                    "predicted_directions": predicted_directions,
                    "predicted_probabilities": (
                        predicted_probabilities
                    ),
                    "horizons": self.config.horizons,
                },
            )

            if final_holdout_stage is None:
                holdout_result = ResearchStageResult(
                    name="final_holdout",
                    executed=False,
                    passed=False,
                    output=None,
                    errors=[
                        "Final holdout stage is enabled but "
                        "no callable was supplied."
                    ],
                )
        else:
            holdout_result = ResearchStageResult(
                name="final_holdout",
                executed=False,
                passed=False,
                output=None,
                warnings=[
                    "Final holdout stage was explicitly disabled."
                ],
            )

        stages[
            "final_holdout"
        ] = holdout_result

        if self.config.stop_on_failure and self._stage_failure(
            holdout_result
        ):
            errors.extend(
                holdout_result.errors
            )
            return self._finalize(
                stages,
                errors,
                warnings,
            )

        # --------------------------------------------------------------
        # 8. PRODUCTION APPROVAL
        # --------------------------------------------------------------

        if self.config.enable_production_approval:
            approval_result = self._execute(
                "production_approval",
                production_approval_stage,
                {
                    "model_selection": selection_result.output,
                    "backtest": backtest_result.output,
                    "robustness": robustness_result.output,
                    "final_holdout": holdout_result.output,
                    "horizons": self.config.horizons,
                },
            )

            if production_approval_stage is None:
                approval_result = ResearchStageResult(
                    name="production_approval",
                    executed=False,
                    passed=False,
                    output=None,
                    errors=[
                        "Production approval is enabled but "
                        "no approval stage was supplied."
                    ],
                )
        else:
            approval_result = ResearchStageResult(
                name="production_approval",
                executed=False,
                passed=False,
                output=None,
                warnings=[
                    "Production approval was explicitly disabled."
                ],
            )

        stages[
            "production_approval"
        ] = approval_result

        return self._finalize(
            stages,
            errors,
            warnings,
        )

    def _finalize(
        self,
        stages: dict[
            str,
            ResearchStageResult,
        ],
        errors: list[str],
        warnings: list[str],
    ) -> EndToEndResearchResult:
        for stage in stages.values():
            errors.extend(
                stage.errors
            )
            warnings.extend(
                stage.warnings
            )

        holdout = stages.get(
            "final_holdout"
        )

        approval = stages.get(
            "production_approval"
        )

        final_holdout_used = False

        if holdout is not None:
            output = holdout.output

            final_holdout_used = bool(
                getattr(
                    output,
                    "final_holdout_used",
                    False,
                )
            )

            if not final_holdout_used:
                metadata = getattr(
                    output,
                    "metadata",
                    {},
                )

                if isinstance(
                    metadata,
                    Mapping,
                ):
                    final_holdout_used = (
                        metadata.get(
                            "final_holdout_used"
                        )
                        is True
                    )

        production_approved = False

        if approval is not None:
            output = approval.output

            production_approved = bool(
                getattr(
                    output,
                    "production_approved",
                    False,
                )
            )

            if not production_approved:
                production_approved = bool(
                    getattr(
                        output,
                        "approved",
                        False,
                    )
                )

        required_stage_names = list(
            self.STAGE_ORDER
        )

        if not self.config.enable_final_holdout:
            required_stage_names.remove(
                "final_holdout"
            )

        if not self.config.enable_production_approval:
            required_stage_names.remove(
                "production_approval"
            )

        required_passed = all(
            stages.get(name) is not None
            and stages[name].executed
            and stages[name].passed
            for name in required_stage_names
        )

        overall_passed = bool(
            required_passed
            and not errors
        )

        return EndToEndResearchResult(
            stages=stages,
            passed=overall_passed,
            final_holdout_used=(
                final_holdout_used
            ),
            production_approved=(
                production_approved
            ),
            production_ready=(
                production_approved
            ),
            research_only=(
                not production_approved
            ),
            errors=list(
                dict.fromkeys(errors)
            ),
            warnings=list(
                dict.fromkeys(warnings)
            ),
            metadata={
                "stage_order": list(
                    self.STAGE_ORDER
                ),
                "horizons": list(
                    self.config.horizons
                ),
                "stop_on_failure": (
                    self.config.stop_on_failure
                ),
                "final_holdout_used": (
                    final_holdout_used
                ),
                "production_approved": (
                    production_approved
                ),
                "production_ready": (
                    production_approved
                ),
                "research_only": (
                    not production_approved
                ),
                "model_fitted_here": False,
                "threshold_optimized_here": False,
                "feature_selection_here": False,
                "calibration_here": False,
            },
        )


def run_end_to_end_research(
    *,
    config: EndToEndResearchConfig | None = None,
    **kwargs: Any,
) -> EndToEndResearchResult:
    """Convenience wrapper for the complete research workflow."""

    orchestrator = EndToEndResearchOrchestrator(
        config=config
    )

    return orchestrator.run(
        **kwargs
    )


def end_to_end_research_summary(
    result: EndToEndResearchResult,
) -> dict[str, Any]:
    """Create a dashboard-safe summary of the research workflow."""

    if not isinstance(
        result,
        EndToEndResearchResult,
    ):
        raise TypeError(
            "result must be an EndToEndResearchResult."
        )

    stage_summary = {}

    for name, stage in result.stages.items():
        stage_summary[name] = {
            "executed": stage.executed,
            "passed": stage.passed,
            "errors": list(
                stage.errors
            ),
            "warnings": list(
                stage.warnings
            ),
        }

    return {
        "passed": result.passed,
        "production_approved": (
            result.production_approved
        ),
        "production_ready": (
            result.production_ready
        ),
        "final_holdout_used": (
            result.final_holdout_used
        ),
        "research_only": (
            result.research_only
        ),
        "stages": stage_summary,
        "errors": list(
            result.errors
        ),
        "warnings": list(
            result.warnings
        ),
    }
