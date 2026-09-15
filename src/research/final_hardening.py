"""
Final production hardening and governance checks.

This module provides a single fail-closed gate for the AI Swing Analyser.

Production inference is allowed only when:

    1. Required research stages passed.
    2. Required horizons are available.
    3. Final holdout evaluation was performed.
    4. Final holdout was not used for training/tuning/selection.
    5. Production approval is explicit.
    6. Research-only metadata does not indicate forbidden operations.
    7. No unresolved errors remain.

This module does NOT train models, tune thresholds, select features,
calibrate probabilities, or modify research results.

It is deliberately conservative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from src.research.api_contracts import (
    extract_approved,
    extract_errors,
    extract_final_holdout_used,
    extract_horizons,
    extract_production_ready,
    extract_passed,
    extract_results_mapping,
    extract_value,
    validate_horizon_consistency,
    validate_research_provenance,
)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class HardeningConfig:
    """
    Immutable configuration for the final production gate.
    """

    required_horizons: tuple[int, ...] = (
        1,
        3,
        5,
        10,
        20,
    )

    require_all_horizons: bool = True

    require_final_holdout: bool = True

    require_explicit_approval: bool = True

    require_production_ready: bool = True

    block_on_errors: bool = True

    block_on_warnings: bool = False

    minimum_approval_count: int = 1


# ---------------------------------------------------------------------
# RESULT TYPES
# ---------------------------------------------------------------------


@dataclass
class HardeningIssue:
    """
    A single hardening failure or warning.
    """

    code: str
    message: str
    severity: str = "ERROR"
    stage: str | None = None


@dataclass
class HardeningResult:
    """
    Final result of production hardening.
    """

    passed: bool

    production_ready: bool

    required_horizons: tuple[int, ...]

    approved_horizons: tuple[int, ...]

    issues: list[HardeningIssue] = field(
        default_factory=list
    )

    warnings: list[HardeningIssue] = field(
        default_factory=list
    )

    checks: dict[str, bool] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def blocked(self) -> bool:
        return not self.production_ready

    @property
    def error_count(self) -> int:
        return len(self.issues)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)


# ---------------------------------------------------------------------
# HARDENING ENGINE
# ---------------------------------------------------------------------


class FinalHardening:
    """
    Fail-closed production governance engine.

    This class only evaluates evidence already produced by the research
    pipeline. It never changes or retrains a model.
    """

    FORBIDDEN_RESEARCH_OPERATIONS = (
        "model_fitted_here",
        "threshold_optimized_here",
        "feature_selection_here",
        "calibration_here",
        "hyperparameter_tuned_here",
    )

    HOLDOUT_LEAKAGE_FLAGS = (
        "used_for_training",
        "used_for_tuning",
        "used_for_model_selection",
        "used_for_feature_selection",
        "used_for_calibration",
    )

    def __init__(
        self,
        config: HardeningConfig | None = None,
    ):
        self.config = (
            config
            if config is not None
            else HardeningConfig()
        )

    # -----------------------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------------------

    def run(
        self,
        *,
        stages: Mapping[str, Any],
        final_holdout: Any,
        production_approval: Any,
    ) -> HardeningResult:
        """
        Run every final production safety check.

        Parameters
        ----------
        stages:
            Mapping containing research-stage results.

        final_holdout:
            Final holdout evaluation result.

        production_approval:
            Explicit production approval result.

        Returns
        -------
        HardeningResult
            Production-ready only when every mandatory gate passes.
        """

        issues: list[HardeningIssue] = []
        warnings: list[HardeningIssue] = []
        checks: dict[str, bool] = {}

        # -------------------------------------------------------------
        # 1. Stage presence
        # -------------------------------------------------------------

        required_stages = (
            "data",
            "features",
            "targets",
            "model_selection",
            "backtest",
            "robustness",
        )

        for stage_name in required_stages:
            exists = (
                stage_name in stages
                and stages[stage_name] is not None
            )

            checks[
                f"stage_present:{stage_name}"
            ] = exists

            if not exists:
                issues.append(
                    HardeningIssue(
                        code="MISSING_STAGE",
                        message=(
                            f"Required research stage "
                            f"'{stage_name}' is missing."
                        ),
                        stage=stage_name,
                    )
                )

        # -------------------------------------------------------------
        # 2. Stage pass checks
        # -------------------------------------------------------------

        for stage_name, stage in stages.items():
            if stage is None:
                continue

            passed = extract_passed(
                stage
            )

            checks[
                f"stage_passed:{stage_name}"
            ] = passed

            if not passed:
                issues.append(
                    HardeningIssue(
                        code="STAGE_FAILED",
                        message=(
                            f"Research stage "
                            f"'{stage_name}' did not pass."
                        ),
                        stage=stage_name,
                    )
                )

        # -------------------------------------------------------------
        # 3. Horizon consistency
        # -------------------------------------------------------------

        try:
            validate_horizon_consistency(
                stages,
                self.config.required_horizons,
            )

            checks[
                "horizon_consistency"
            ] = True

        except Exception as exc:
            checks[
                "horizon_consistency"
            ] = False

            issues.append(
                HardeningIssue(
                    code="HORIZON_INCONSISTENCY",
                    message=str(exc),
                )
            )

        # -------------------------------------------------------------
        # 4. Final holdout presence
        # -------------------------------------------------------------

        holdout_present = (
            final_holdout is not None
        )

        checks[
            "final_holdout_present"
        ] = holdout_present

        if (
            self.config.require_final_holdout
            and not holdout_present
        ):
            issues.append(
                HardeningIssue(
                    code="MISSING_FINAL_HOLDOUT",
                    message=(
                        "Final holdout evaluation "
                        "is required."
                    ),
                    stage="final_holdout",
                )
            )

        # -------------------------------------------------------------
        # 5. Final holdout usage
        # -------------------------------------------------------------

        holdout_used = False

        if holdout_present:
            holdout_used = (
                extract_final_holdout_used(
                    final_holdout
                )
            )

        checks[
            "final_holdout_evaluated"
        ] = holdout_used

        if (
            self.config.require_final_holdout
            and not holdout_used
        ):
            issues.append(
                HardeningIssue(
                    code="HOLDOUT_NOT_EVALUATED",
                    message=(
                        "The final holdout is present "
                        "but explicit holdout evaluation "
                        "evidence is missing."
                    ),
                    stage="final_holdout",
                )
            )

        # -------------------------------------------------------------
        # 6. Holdout leakage
        # -------------------------------------------------------------

        leakage_free = (
            self._check_holdout_leakage(
                final_holdout,
                issues,
            )
        )

        checks[
            "final_holdout_leakage_free"
        ] = leakage_free

        # -------------------------------------------------------------
        # 7. Research provenance
        # -------------------------------------------------------------

        provenance_ok = True

        for stage_name, stage in stages.items():
            if stage is None:
                continue

            try:
                validate_research_provenance(
                    stage
                )
            except Exception as exc:
                provenance_ok = False

                issues.append(
                    HardeningIssue(
                        code="INVALID_RESEARCH_PROVENANCE",
                        message=str(exc),
                        stage=stage_name,
                    )
                )

        checks[
            "research_provenance"
        ] = provenance_ok

        # -------------------------------------------------------------
        # 8. Explicit production approval
        # -------------------------------------------------------------

        approval_present = (
            production_approval is not None
        )

        checks[
            "approval_present"
        ] = approval_present

        if (
            self.config.require_explicit_approval
            and not approval_present
        ):
            issues.append(
                HardeningIssue(
                    code="MISSING_APPROVAL",
                    message=(
                        "Explicit production approval "
                        "is required."
                    ),
                    stage="production_approval",
                )
            )

        approved = False

        if approval_present:
            approved = extract_approved(
                production_approval
            )

        checks[
            "explicit_approval"
        ] = approved

        if (
            self.config.require_explicit_approval
            and not approved
        ):
            issues.append(
                HardeningIssue(
                    code="NOT_APPROVED",
                    message=(
                        "Production approval has not "
                        "been explicitly granted."
                    ),
                    stage="production_approval",
                )
            )

        # -------------------------------------------------------------
        # 9. Production-ready flag
        # -------------------------------------------------------------

        production_ready = False

        if approval_present:
            production_ready = (
                extract_production_ready(
                    production_approval
                )
            )

        checks[
            "production_ready_flag"
        ] = production_ready

        if (
            self.config.require_production_ready
            and not production_ready
        ):
            issues.append(
                HardeningIssue(
                    code="NOT_PRODUCTION_READY",
                    message=(
                        "Production-ready status "
                        "has not been explicitly established."
                    ),
                    stage="production_approval",
                )
            )

        # -------------------------------------------------------------
        # 10. Approved horizon verification
        # -------------------------------------------------------------

        approved_horizons = (
            self._approved_horizons(
                production_approval
            )
        )

        checks[
            "approved_horizons_available"
        ] = bool(approved_horizons)

        missing_horizons = [
            horizon
            for horizon
            in self.config.required_horizons
            if horizon not in approved_horizons
        ]

        if (
            self.config.require_all_horizons
            and missing_horizons
        ):
            issues.append(
                HardeningIssue(
                    code="MISSING_APPROVED_HORIZONS",
                    message=(
                        "Required horizons are not "
                        f"explicitly approved: "
                        f"{missing_horizons}"
                    ),
                    stage="production_approval",
                )
            )

        # -------------------------------------------------------------
        # 11. Error inspection
        # -------------------------------------------------------------

        if self.config.block_on_errors:
            for stage_name, stage in stages.items():
                if stage is None:
                    continue

                errors = extract_errors(
                    stage
                )

                if errors:
                    issues.append(
                        HardeningIssue(
                            code="UNRESOLVED_STAGE_ERRORS",
                            message=(
                                f"Stage '{stage_name}' "
                                f"contains unresolved errors."
                            ),
                            stage=stage_name,
                        )
                    )

        if (
            self.config.block_on_errors
            and approval_present
        ):
            errors = extract_errors(
                production_approval
            )

            if errors:
                issues.append(
                    HardeningIssue(
                        code="UNRESOLVED_APPROVAL_ERRORS",
                        message=(
                            "Production approval contains "
                            "unresolved errors."
                        ),
                        stage="production_approval",
                    )
                )

        # -------------------------------------------------------------
        # 12. Warnings
        # -------------------------------------------------------------

        if self.config.block_on_warnings:
            for stage_name, stage in stages.items():
                warnings_found = self._extract_warnings(
                    stage
                )

                for warning in warnings_found:
                    warnings.append(
                        HardeningIssue(
                            code="STAGE_WARNING",
                            message=str(warning),
                            severity="WARNING",
                            stage=stage_name,
                        )
                    )

        # -------------------------------------------------------------
        # FINAL DECISION
        # -------------------------------------------------------------

        passed = len(issues) == 0

        final_ready = (
            passed
            and approved
            and (
                production_ready
                if self.config.require_production_ready
                else True
            )
        )

        return HardeningResult(
            passed=passed,
            production_ready=final_ready,
            required_horizons=(
                self.config.required_horizons
            ),
            approved_horizons=tuple(
                approved_horizons
            ),
            issues=issues,
            warnings=warnings,
            checks=checks,
            metadata={
                "fail_closed": True,
                "research_only": False,
                "model_training_performed": False,
                "threshold_optimization_performed": False,
                "feature_selection_performed": False,
                "calibration_performed": False,
                "final_holdout_required": (
                    self.config.require_final_holdout
                ),
                "explicit_approval_required": (
                    self.config.require_explicit_approval
                ),
            },
        )

    # -----------------------------------------------------------------
    # HOLDOUT LEAKAGE
    # -----------------------------------------------------------------

    def _check_holdout_leakage(
        self,
        holdout: Any,
        issues: list[HardeningIssue],
    ) -> bool:
        if holdout is None:
            return False

        leakage_free = True

        for flag in self.HOLDOUT_LEAKAGE_FLAGS:
            value = self._extract_flag(
                holdout,
                flag,
            )

            if value is True:
                leakage_free = False

                issues.append(
                    HardeningIssue(
                        code="HOLDOUT_LEAKAGE",
                        message=(
                            f"Final holdout was marked "
                            f"as '{flag}'."
                        ),
                        stage="final_holdout",
                    )
                )

        return leakage_free

    # -----------------------------------------------------------------
    # APPROVED HORIZONS
    # -----------------------------------------------------------------

    def _approved_horizons(
        self,
        approval: Any,
    ) -> tuple[int, ...]:
        if approval is None:
            return ()

        result_mapping = (
            extract_results_mapping(
                approval
            )
        )

        approved = []

        for key, result in result_mapping.items():
            try:
                horizon = int(key)
            except (
                TypeError,
                ValueError,
            ):
                continue

            if extract_approved(
                result
            ):
                approved.append(
                    horizon
                )

        if approved:
            return tuple(
                sorted(
                    set(approved)
                )
            )

        # Some approval objects expose horizons
        # directly rather than through results.
        horizons = extract_horizons(
            approval
        )

        return tuple(
            sorted(
                set(
                    int(h)
                    for h in horizons
                    if self._horizon_is_approved(
                        approval,
                        h,
                    )
                )
            )
        )

    def _horizon_is_approved(
        self,
        approval: Any,
        horizon: int,
    ) -> bool:
        mapping = extract_results_mapping(
            approval
        )

        result = mapping.get(
            horizon
        )

        if result is None:
            result = mapping.get(
                str(horizon)
            )

        if result is None:
            return False

        return extract_approved(
            result
        )

    # -----------------------------------------------------------------
    # GENERIC EXTRACTION
    # -----------------------------------------------------------------

    @staticmethod
    def _extract_flag(
        obj: Any,
        name: str,
    ) -> bool:
        if obj is None:
            return False

        if isinstance(
            obj,
            Mapping,
        ):
            value = obj.get(
                name,
                False,
            )
        else:
            value = getattr(
                obj,
                name,
                False,
            )

        return bool(value)

    @staticmethod
    def _extract_warnings(
        obj: Any,
    ) -> Iterable[Any]:
        if obj is None:
            return ()

        if isinstance(
            obj,
            Mapping,
        ):
            return obj.get(
                "warnings",
                (),
            )

        return getattr(
            obj,
            "warnings",
            (),
        )


# ---------------------------------------------------------------------
# SIMPLE PUBLIC HELPER
# ---------------------------------------------------------------------


def harden_for_production(
    *,
    stages: Mapping[str, Any],
    final_holdout: Any,
    production_approval: Any,
    config: HardeningConfig | None = None,
) -> HardeningResult:
    """
    Convenience wrapper around FinalHardening.

    This function is intentionally fail-closed.

    A caller must receive:

        result.production_ready is True

    before enabling live model inference.
    """

    engine = FinalHardening(
        config=config
    )

    return engine.run(
        stages=stages,
        final_holdout=final_holdout,
        production_approval=production_approval,
    )


# ---------------------------------------------------------------------
# PRODUCTION SAFETY HELPER
# ---------------------------------------------------------------------


def assert_production_ready(
    result: HardeningResult,
) -> None:
    """
    Raise an exception unless the complete research system
    has passed final hardening.
    """

    if not isinstance(
        result,
        HardeningResult,
    ):
        raise TypeError(
            "Expected HardeningResult."
        )

    if not result.production_ready:
        messages = [
            issue.message
            for issue in result.issues
        ]

        detail = (
            "; ".join(messages)
            if messages
            else "No production approval."
        )

        raise RuntimeError(
            "PRODUCTION BLOCKED: "
            + detail
        )
