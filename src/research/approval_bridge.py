"""
AI Swing Analyser — Research Approval Bridge.

Connects the integrated research evidence framework to the existing
ProductionApprovalEngine.

This module does not create research evidence and does not train models.
It only translates already-produced evidence into the final approval
decision.

Approval is always fail-closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .integration import (
    EvidenceStatus,
    ResearchEvidence,
    assert_research_ready,
)
from .research_config import (
    IntegratedApprovalConfig,
)


# ---------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------


@dataclass
class ApprovalBridgeResult:
    """
    Final bridge result.

    `approved` is true only when every mandatory production gate has
    been explicitly evaluated and passed.
    """

    approved: bool

    status: str

    model_id: str | None = None

    experiment_id: str | None = None

    passed_gates: list[str] = field(
        default_factory=list
    )

    failed_gates: list[str] = field(
        default_factory=list
    )

    missing_gates: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    reasons: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def production_ready(self) -> bool:
        return self.approved


# ---------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------


class ResearchApprovalBridge:
    """
    Converts ResearchEvidence into a conservative approval decision.

    The bridge intentionally performs explicit checks instead of relying
    solely on a numerical evidence score.

    A high aggregate score can never compensate for a failed critical
    gate.
    """

    def __init__(
        self,
        config: IntegratedApprovalConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else IntegratedApprovalConfig()
        )

    # -----------------------------------------------------------------
    # Public evaluation
    # -----------------------------------------------------------------

    def evaluate(
        self,
        evidence: ResearchEvidence,
    ) -> ApprovalBridgeResult:

        passed: list[str] = []
        failed: list[str] = []
        missing: list[str] = []
        warnings: list[str] = []
        reasons: list[str] = []

        checks = self._build_checks(
            evidence
        )

        for name, status, reason, critical in checks:

            if status == EvidenceStatus.PASS:

                passed.append(name)

            elif status == EvidenceStatus.FAIL:

                failed.append(name)

                if reason:
                    reasons.append(
                        f"{name}: {reason}"
                    )

            elif status == EvidenceStatus.WARNING:

                warnings.append(name)

                if critical:
                    missing.append(name)

                    reasons.append(
                        f"{name}: warning is insufficient "
                        "for a mandatory production gate."
                    )

            else:

                missing.append(name)

                reasons.append(
                    f"{name}: evidence was not evaluated."
                )

        # -------------------------------------------------------------
        # Explicit fail-closed decision
        # -------------------------------------------------------------

        if failed:

            status = "REJECTED"

        elif missing:

            status = "HOLD"

        elif len(passed) != len(checks):

            status = "HOLD"

        else:

            status = "APPROVED"

        approved = status == "APPROVED"

        metadata = {
            "research_only": False,
            "approval_decision": status,
            "approved": approved,
            "evidence_score": (
                evidence.evidence_score()
            ),
            "all_required_evaluated": (
                evidence.all_required_evaluated()
            ),
            "all_critical_pass": (
                evidence.all_critical_pass()
            ),
        }

        return ApprovalBridgeResult(
            approved=approved,
            status=status,
            model_id=evidence.model_id,
            experiment_id=evidence.experiment_id,
            passed_gates=passed,
            failed_gates=failed,
            missing_gates=missing,
            warnings=warnings,
            reasons=reasons,
            metadata=metadata,
        )

    # -----------------------------------------------------------------
    # Gate construction
    # -----------------------------------------------------------------

    def _build_checks(
        self,
        evidence: ResearchEvidence,
    ) -> list[
        tuple[
            str,
            EvidenceStatus,
            str | None,
            bool,
        ]
    ]:

        checks = []

        self._append_if_required(
            checks,
            "walk_forward_validation",
            evidence.validation,
            self.config.require_validation,
        )

        self._append_if_required(
            checks,
            "final_holdout",
            evidence.holdout,
            self.config.require_holdout,
        )

        self._append_if_required(
            checks,
            "calibration",
            evidence.calibration,
            self.config.require_calibration,
        )

        self._append_if_required(
            checks,
            "range_validation",
            evidence.range_validation,
            self.config.require_range_validation,
        )

        self._append_if_required(
            checks,
            "regime_validation",
            evidence.regime,
            self.config.require_regime_validation,
        )

        self._append_if_required(
            checks,
            "backtest",
            evidence.backtest,
            self.config.require_backtest,
        )

        self._append_if_required(
            checks,
            "robustness",
            evidence.robustness,
            self.config.require_robustness,
        )

        self._append_if_required(
            checks,
            "leakage_audit",
            evidence.leakage,
            self.config.require_leakage_audit,
        )

        return checks

    @staticmethod
    def _append_if_required(
        checks: list,
        name: str,
        item,
        required: bool,
    ) -> None:

        if not required:
            return

        if item is None:

            checks.append(
                (
                    name,
                    EvidenceStatus.NOT_EVALUATED,
                    "Required evidence is missing.",
                    True,
                )
            )

            return

        reason = ResearchApprovalBridge._failure_reason(
            item
        )

        checks.append(
            (
                name,
                item.status,
                reason,
                bool(item.critical),
            )
        )

    # -----------------------------------------------------------------
    # Failure explanation
    # -----------------------------------------------------------------

    @staticmethod
    def _failure_reason(
        item,
    ) -> str | None:

        if item.status == EvidenceStatus.PASS:
            return None

        if item.status == EvidenceStatus.FAIL:

            if item.details:
                return str(
                    item.details
                )

            if item.metrics:
                return (
                    "Evidence metrics did not satisfy "
                    "the configured gate."
                )

            return "Evidence gate failed."

        if item.status == EvidenceStatus.WARNING:

            if item.details:
                return str(
                    item.details
                )

            return "Evidence is only a warning."

        return "Evidence was not evaluated."

    # -----------------------------------------------------------------
    # Safety assertions
    # -----------------------------------------------------------------

    def assert_approved(
        self,
        evidence: ResearchEvidence,
    ) -> ApprovalBridgeResult:

        result = self.evaluate(
            evidence
        )

        if not result.approved:

            raise RuntimeError(
                "Research evidence is not approved "
                "for production. "
                f"Status={result.status}; "
                f"failed={result.failed_gates}; "
                f"missing={result.missing_gates}"
            )

        return result

    def can_proceed(
        self,
        evidence: ResearchEvidence,
    ) -> bool:

        return self.evaluate(
            evidence
        ).approved

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    @staticmethod
    def summary(
        result: ApprovalBridgeResult,
    ) -> dict[str, Any]:

        return {
            "approved": result.approved,
            "production_ready": (
                result.production_ready
            ),
            "status": result.status,
            "model_id": result.model_id,
            "experiment_id": result.experiment_id,
            "passed_gate_count": len(
                result.passed_gates
            ),
            "failed_gate_count": len(
                result.failed_gates
            ),
            "missing_gate_count": len(
                result.missing_gates
            ),
            "warning_count": len(
                result.warnings
            ),
            "passed_gates": list(
                result.passed_gates
            ),
            "failed_gates": list(
                result.failed_gates
            ),
            "missing_gates": list(
                result.missing_gates
            ),
            "reasons": list(
                result.reasons
            ),
        }


# ---------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------


def evaluate_research_approval(
    evidence: ResearchEvidence,
    config: IntegratedApprovalConfig | None = None,
) -> ApprovalBridgeResult:

    bridge = ResearchApprovalBridge(
        config=config
    )

    return bridge.evaluate(
        evidence
    )


def assert_research_approved(
    evidence: ResearchEvidence,
    config: IntegratedApprovalConfig | None = None,
) -> ApprovalBridgeResult:

    bridge = ResearchApprovalBridge(
        config=config
    )

    return bridge.assert_approved(
        evidence
    )


__all__ = [
    "ApprovalBridgeResult",
    "ResearchApprovalBridge",
    "evaluate_research_approval",
    "assert_research_approved",
]
