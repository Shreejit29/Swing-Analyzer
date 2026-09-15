"""
Production approval integration for the AI Swing Analyser.

This module is the final research-governance gate before production
inference is allowed.

Approval requires evidence from:
    1. Model selection
    2. Backtesting
    3. Robustness research
    4. Final holdout evaluation

The module does NOT:
    - train a model
    - tune hyperparameters
    - optimize thresholds
    - select features
    - modify predictions
    - rerun research

A failed gate is always fail-closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ProductionApprovalConfig:
    """Configuration for the final production approval gate."""

    required_horizons: tuple[int, ...] = (
        1,
        3,
        5,
        10,
        20,
    )

    minimum_approved_horizons: int = 1

    require_model_selection: bool = True
    require_backtest: bool = True
    require_robustness: bool = True
    require_final_holdout: bool = True

    require_95_percent_gate: bool = True

    allow_research_only: bool = False

    def __post_init__(self) -> None:
        horizons = tuple(
            int(x)
            for x in self.required_horizons
        )

        if not horizons:
            raise ValueError(
                "At least one required horizon is necessary."
            )

        if any(x <= 0 for x in horizons):
            raise ValueError(
                "Required horizons must be positive."
            )

        if len(set(horizons)) != len(horizons):
            raise ValueError(
                "Required horizons must be unique."
            )

        if (
            self.minimum_approved_horizons < 1
            or self.minimum_approved_horizons
            > len(horizons)
        ):
            raise ValueError(
                "minimum_approved_horizons must be between "
                "1 and the number of required horizons."
            )

        object.__setattr__(
            self,
            "required_horizons",
            horizons,
        )


@dataclass
class HorizonApprovalResult:
    """Approval evidence for one forecast horizon."""

    horizon: int

    model_selection_passed: bool = False
    backtest_passed: bool = False
    robustness_passed: bool = False
    final_holdout_passed: bool = False

    holdout_accuracy: float | None = None

    approved: bool = False

    reasons: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class ProductionApprovalResult:
    """Complete production approval decision."""

    approved: bool

    required_horizons: tuple[int, ...]

    horizon_results: dict[
        int,
        HorizonApprovalResult,
    ]

    approved_horizons: tuple[int, ...] = ()

    rejected_horizons: tuple[int, ...] = ()

    model_selection_passed: bool = False
    backtest_passed: bool = False
    robustness_passed: bool = False
    final_holdout_passed: bool = False

    final_holdout_used: bool = False

    research_only: bool = False
    production_ready: bool = False
    production_approved: bool = False

    reasons: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def get(
        self,
        horizon: int,
    ) -> HorizonApprovalResult:
        if horizon not in self.horizon_results:
            raise KeyError(
                f"No approval result for horizon={horizon}."
            )

        return self.horizon_results[horizon]


class ProductionApprovalIntegration:
    """
    Final governance gate before production inference.

    The gate is intentionally conservative. Missing evidence is treated
    as failure rather than being inferred as success.
    """

    def __init__(
        self,
        config: ProductionApprovalConfig | None = None,
    ) -> None:
        self.config = (
            config
            or ProductionApprovalConfig()
        )

    @staticmethod
    def _extract(
        source: Any,
        names: tuple[str, ...],
    ) -> Any:
        if source is None:
            return None

        if isinstance(
            source,
            Mapping,
        ):
            for name in names:
                if name in source:
                    return source[name]

        for name in names:
            if hasattr(
                source,
                name,
            ):
                return getattr(
                    source,
                    name,
                )

        return None

    @classmethod
    def _horizon_result(
        cls,
        source: Any,
        horizon: int,
    ) -> Any:
        if source is None:
            return None

        if isinstance(
            source,
            Mapping,
        ):
            return source.get(horizon)

        results = cls._extract(
            source,
            (
                "results",
                "horizon_results",
            ),
        )

        if isinstance(
            results,
            Mapping,
        ):
            return results.get(horizon)

        getter = getattr(
            source,
            "get",
            None,
        )

        if callable(getter):
            try:
                return getter(horizon)
            except (
                KeyError,
                TypeError,
            ):
                return None

        return None

    @classmethod
    def _passed(
        cls,
        source: Any,
        horizon: int,
    ) -> bool:
        item = cls._horizon_result(
            source,
            horizon,
        )

        value = cls._extract(
            item,
            (
                "passed",
                "approved",
                "candidate_passed",
            ),
        )

        return value is True

    @classmethod
    def _accuracy(
        cls,
        source: Any,
        horizon: int,
    ) -> float | None:
        item = cls._horizon_result(
            source,
            horizon,
        )

        value = cls._extract(
            item,
            (
                "accuracy",
                "holdout_accuracy",
            ),
        )

        if value is None:
            return None

        try:
            return float(value)
        except (
            TypeError,
            ValueError,
        ):
            return None

    @classmethod
    def _final_holdout_used(
        cls,
        source: Any,
    ) -> bool:
        value = cls._extract(
            source,
            (
                "final_holdout_used",
            ),
        )

        if value is True:
            return True

        metadata = cls._extract(
            source,
            (
                "metadata",
            ),
        )

        if isinstance(
            metadata,
            Mapping,
        ):
            return (
                metadata.get(
                    "final_holdout_used"
                )
                is True
            )

        return False

    @classmethod
    def _production_approved(
        cls,
        source: Any,
    ) -> bool:
        value = cls._extract(
            source,
            (
                "production_approved",
            ),
        )

        if value is True:
            return True

        metadata = cls._extract(
            source,
            (
                "metadata",
            ),
        )

        if isinstance(
            metadata,
            Mapping,
        ):
            return (
                metadata.get(
                    "production_approved"
                )
                is True
            )

        return False

    @classmethod
    def _validate_research_provenance(
        cls,
        source_name: str,
        source: Any,
    ) -> list[str]:
        problems: list[str] = []

        if cls._production_approved(source):
            problems.append(
                f"{source_name} already reports "
                "production approval; approval must "
                "be granted only by this final gate."
            )

        return problems

    def _evaluate_horizon(
        self,
        horizon: int,
        model_selection: Any,
        backtest: Any,
        robustness: Any,
        final_holdout: Any,
    ) -> HorizonApprovalResult:
        result = HorizonApprovalResult(
            horizon=horizon,
            metadata={
                "research_only": False,
                "final_holdout_used": True,
                "production_approved": False,
            },
        )

        result.model_selection_passed = (
            self._passed(
                model_selection,
                horizon,
            )
        )

        result.backtest_passed = (
            self._passed(
                backtest,
                horizon,
            )
        )

        result.robustness_passed = (
            self._passed(
                robustness,
                horizon,
            )
        )

        result.final_holdout_passed = (
            self._passed(
                final_holdout,
                horizon,
            )
        )

        result.holdout_accuracy = (
            self._accuracy(
                final_holdout,
                horizon,
            )
        )

        if self.config.require_model_selection:
            if not result.model_selection_passed:
                result.reasons.append(
                    "Model-selection gate failed."
                )

        if self.config.require_backtest:
            if not result.backtest_passed:
                result.reasons.append(
                    "Backtest gate failed."
                )

        if self.config.require_robustness:
            if not result.robustness_passed:
                result.reasons.append(
                    "Robustness gate failed."
                )

        if self.config.require_final_holdout:
            if not result.final_holdout_passed:
                result.reasons.append(
                    "Final holdout gate failed."
                )

        if self.config.require_95_percent_gate:
            accuracy = (
                result.holdout_accuracy
            )

            if accuracy is None:
                result.reasons.append(
                    "Final holdout accuracy is unavailable."
                )
            elif accuracy < 0.95:
                result.reasons.append(
                    "The strict 95% research accuracy "
                    "gate was not achieved."
                )

        required_checks = []

        if self.config.require_model_selection:
            required_checks.append(
                result.model_selection_passed
            )

        if self.config.require_backtest:
            required_checks.append(
                result.backtest_passed
            )

        if self.config.require_robustness:
            required_checks.append(
                result.robustness_passed
            )

        if self.config.require_final_holdout:
            required_checks.append(
                result.final_holdout_passed
            )

        result.approved = bool(
            required_checks
            and all(required_checks)
            and (
                not self.config.require_95_percent_gate
                or (
                    result.holdout_accuracy
                    is not None
                    and result.holdout_accuracy
                    >= 0.95
                )
            )
        )

        if not result.approved:
            result.warnings.append(
                "This horizon must not be used "
                "for production inference."
            )

        return result

    def run(
        self,
        model_selection: Any,
        backtest: Any,
        robustness: Any,
        final_holdout: Any,
    ) -> ProductionApprovalResult:
        """
        Evaluate all configured horizons.

        This method only evaluates evidence. It never modifies or retrains
        the candidate model.
        """

        sources = {
            "model_selection": model_selection,
            "backtest": backtest,
            "robustness": robustness,
            "final_holdout": final_holdout,
        }

        errors: list[str] = []

        for name, source in sources.items():
            errors.extend(
                self._validate_research_provenance(
                    name,
                    source,
                )
            )

        final_holdout_used = (
            self._final_holdout_used(
                final_holdout
            )
        )

        if (
            self.config.require_final_holdout
            and not final_holdout_used
        ):
            errors.append(
                "Final holdout provenance is missing. "
                "Production approval is blocked."
            )

        horizon_results: dict[
            int,
            HorizonApprovalResult,
        ] = {}

        approved = []
        rejected = []

        for horizon in self.config.required_horizons:
            item = self._evaluate_horizon(
                horizon=horizon,
                model_selection=model_selection,
                backtest=backtest,
                robustness=robustness,
                final_holdout=final_holdout,
            )

            horizon_results[horizon] = item

            if item.approved:
                approved.append(horizon)
            else:
                rejected.append(horizon)

        if errors:
            for item in horizon_results.values():
                item.approved = False
                item.reasons.extend(
                    errors
                )

            approved.clear()
            rejected = list(
                self.config.required_horizons
            )

        candidate_approved = bool(
            len(approved)
            >= self.config.minimum_approved_horizons
            and final_holdout_used
            and not errors
        )

        model_selection_passed = bool(
            approved
            and all(
                horizon_results[h].model_selection_passed
                for h in approved
            )
        )

        backtest_passed = bool(
            approved
            and all(
                horizon_results[h].backtest_passed
                for h in approved
            )
        )

        robustness_passed = bool(
            approved
            and all(
                horizon_results[h].robustness_passed
                for h in approved
            )
        )

        final_holdout_passed = bool(
            approved
            and all(
                horizon_results[h].final_holdout_passed
                for h in approved
            )
        )

        reasons = list(errors)

        if not candidate_approved:
            reasons.append(
                "Production approval was not granted."
            )

        return ProductionApprovalResult(
            approved=candidate_approved,
            required_horizons=(
                self.config.required_horizons
            ),
            horizon_results=horizon_results,
            approved_horizons=tuple(
                approved
            ),
            rejected_horizons=tuple(
                rejected
            ),
            model_selection_passed=(
                model_selection_passed
            ),
            backtest_passed=(
                backtest_passed
            ),
            robustness_passed=(
                robustness_passed
            ),
            final_holdout_passed=(
                final_holdout_passed
            ),
            final_holdout_used=(
                final_holdout_used
            ),
            research_only=False,
            production_ready=(
                candidate_approved
            ),
            production_approved=(
                candidate_approved
            ),
            reasons=reasons,
            warnings=[
                warning
                for item in horizon_results.values()
                for warning in item.warnings
            ],
            metadata={
                "approval_gate": True,
                "final_holdout_used": (
                    final_holdout_used
                ),
                "research_only": False,
                "production_ready": (
                    candidate_approved
                ),
                "production_approved": (
                    candidate_approved
                ),
                "model_fitted_here": False,
                "threshold_optimized_here": False,
                "feature_selection_here": False,
                "calibration_here": False,
            },
        )


def approve_for_production(
    model_selection: Any,
    backtest: Any,
    robustness: Any,
    final_holdout: Any,
    config: ProductionApprovalConfig | None = None,
) -> ProductionApprovalResult:
    """Convenience wrapper for the production approval gate."""

    gate = ProductionApprovalIntegration(
        config=config
    )

    return gate.run(
        model_selection=model_selection,
        backtest=backtest,
        robustness=robustness,
        final_holdout=final_holdout,
    )


def production_approval_summary(
    result: ProductionApprovalResult,
) -> dict[str, Any]:
    """Return a dashboard-safe approval summary."""

    if not isinstance(
        result,
        ProductionApprovalResult,
    ):
        raise TypeError(
            "result must be a ProductionApprovalResult."
        )

    horizon_summary = {}

    for horizon, item in (
        result.horizon_results.items()
    ):
        horizon_summary[str(horizon)] = {
            "approved": item.approved,
            "model_selection_passed": (
                item.model_selection_passed
            ),
            "backtest_passed": (
                item.backtest_passed
            ),
            "robustness_passed": (
                item.robustness_passed
            ),
            "final_holdout_passed": (
                item.final_holdout_passed
            ),
            "holdout_accuracy": (
                item.holdout_accuracy
            ),
            "reasons": list(
                item.reasons
            ),
            "warnings": list(
                item.warnings
            ),
        }

    return {
        "approved": result.approved,
        "production_ready": (
            result.production_ready
        ),
        "production_approved": (
            result.production_approved
        ),
        "final_holdout_used": (
            result.final_holdout_used
        ),
        "approved_horizons": list(
            result.approved_horizons
        ),
        "rejected_horizons": list(
            result.rejected_horizons
        ),
        "horizon_results": horizon_summary,
        "reasons": list(
            result.reasons
        ),
        "warnings": list(
            result.warnings
        ),
    }
