"""
AI Swing Analyser — Research Evidence Integration.

This module connects the independent research stages into one
auditable evidence object.

It does NOT train models and does NOT create trading signals.

Its job is to answer:

    "Do we have enough independent evidence to allow
     the model to proceed toward production approval?"

Required evidence can include:

    - walk-forward validation
    - final holdout
    - probability calibration
    - target-range validation
    - regime validation
    - backtest
    - robustness
    - leakage audit

The module intentionally fails closed:

    missing evidence -> not approved
    failed critical evidence -> not approved

This keeps the research process conservative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------
# Evidence status
# ---------------------------------------------------------------------


class EvidenceStatus(str, Enum):
    """
    Status of one research evidence category.
    """

    NOT_EVALUATED = "NOT_EVALUATED"
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"


# ---------------------------------------------------------------------
# Evidence item
# ---------------------------------------------------------------------


@dataclass
class EvidenceItem:
    """
    Represents one independent research gate.
    """

    name: str

    status: EvidenceStatus = (
        EvidenceStatus.NOT_EVALUATED
    )

    score: float | None = None

    threshold: float | None = None

    metrics: dict[str, Any] = field(
        default_factory=dict
    )

    details: str = ""

    critical: bool = True

    source: str | None = None

    def passed(self) -> bool:
        return (
            self.status
            == EvidenceStatus.PASS
        )

    def failed(self) -> bool:
        return (
            self.status
            == EvidenceStatus.FAIL
        )

    def evaluated(self) -> bool:
        return (
            self.status
            != EvidenceStatus.NOT_EVALUATED
        )


# ---------------------------------------------------------------------
# Complete evidence package
# ---------------------------------------------------------------------


@dataclass
class ResearchEvidence:
    """
    Complete evidence collected for a research candidate.
    """

    model_id: str | None = None

    experiment_id: str | None = None

    symbol: str | None = None

    timeframe: str | None = None

    horizon: int | None = None

    validation: EvidenceItem = field(
        default_factory=lambda: EvidenceItem(
            name="walk_forward_validation"
        )
    )

    holdout: EvidenceItem = field(
        default_factory=lambda: EvidenceItem(
            name="final_holdout"
        )
    )

    calibration: EvidenceItem = field(
        default_factory=lambda: EvidenceItem(
            name="probability_calibration"
        )
    )

    range_validation: EvidenceItem = field(
        default_factory=lambda: EvidenceItem(
            name="target_range_validation"
        )
    )

    regime: EvidenceItem = field(
        default_factory=lambda: EvidenceItem(
            name="regime_validation"
        )
    )

    backtest: EvidenceItem = field(
        default_factory=lambda: EvidenceItem(
            name="backtest"
        )
    )

    robustness: EvidenceItem = field(
        default_factory=lambda: EvidenceItem(
            name="robustness"
        )
    )

    leakage: EvidenceItem = field(
        default_factory=lambda: EvidenceItem(
            name="leakage_audit"
        )
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    warnings: list[str] = field(
        default_factory=list
    )

    notes: list[str] = field(
        default_factory=list
    )

    # -----------------------------------------------------------------
    # Evidence collection
    # -----------------------------------------------------------------

    def items(self) -> list[EvidenceItem]:
        return [
            self.validation,
            self.holdout,
            self.calibration,
            self.range_validation,
            self.regime,
            self.backtest,
            self.robustness,
            self.leakage,
        ]

    def evaluated_items(
        self,
    ) -> list[EvidenceItem]:

        return [
            item
            for item in self.items()
            if item is not None and item.evaluated()
        ]

    def failed_items(
        self,
    ) -> list[EvidenceItem]:

        return [
            item
            for item in self.items()
            if item is not None and item.failed()
        ]

    def missing_items(
        self,
    ) -> list[EvidenceItem]:

        return [
            item
            for item in self.items()
            if item is None or not item.evaluated()
        ]

    # -----------------------------------------------------------------
    # Gate evaluation
    # -----------------------------------------------------------------

    def critical_failures(
        self,
    ) -> list[EvidenceItem]:

        return [
            item
            for item in self.items()
            if item is not None
            and item.critical
            and item.failed()
        ]

    def critical_missing(
        self,
    ) -> list[EvidenceItem]:

        return [
            item
            for item in self.items()
            if item is None
            or (item.critical and not item.evaluated())
        ]

    def all_required_evaluated(
        self,
    ) -> bool:

        return len(
            self.critical_missing()
        ) == 0

    def all_critical_pass(
        self,
    ) -> bool:

        return (
            self.all_required_evaluated()
            and len(
                self.critical_failures()
            )
            == 0
        )

    # -----------------------------------------------------------------
    # Score
    # -----------------------------------------------------------------

    def evidence_score(
        self,
    ) -> float:

        evaluated = (
            self.evaluated_items()
        )

        if not evaluated:
            return 0.0

        passed = sum(
            item.passed()
            for item in evaluated
        )

        return float(
            passed / len(evaluated)
        )

    # -----------------------------------------------------------------
    # Production eligibility
    # -----------------------------------------------------------------

    def production_eligible(
        self,
    ) -> bool:

        """
        Conservative production eligibility check.

        This does NOT replace the final ProductionApprovalEngine.

        It only answers whether the evidence package is sufficiently
        complete and free from critical failures to proceed to that
        final approval stage.
        """

        return self.all_critical_pass()

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    def summary(
        self,
    ) -> dict[str, Any]:

        return {
            "model_id": self.model_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "horizon": self.horizon,
            "evidence_score": (
                self.evidence_score()
            ),
            "production_eligible": (
                self.production_eligible()
            ),
            "all_required_evaluated": (
                self.all_required_evaluated()
            ),
            "critical_failures": [
                item.name
                for item in self.critical_failures()
            ],
            "critical_missing": [
                item.name
                for item in self.critical_missing()
            ],
            "warnings": list(
                self.warnings
            ),
            "notes": list(
                self.notes
            ),
        }


# ---------------------------------------------------------------------
# Integration builder
# ---------------------------------------------------------------------


class ResearchEvidenceBuilder:
    """
    Incrementally builds a ResearchEvidence package.

    Each stage can be recorded independently.

    The builder does not modify the underlying research result.
    """

    def __init__(
        self,
        model_id: str | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        horizon: int | None = None,
    ) -> None:

        self.evidence = ResearchEvidence(
            model_id=model_id,
            symbol=symbol,
            timeframe=timeframe,
            horizon=horizon,
        )

    # -----------------------------------------------------------------
    # Generic setter
    # -----------------------------------------------------------------

    @staticmethod
    def _make_item(
        name: str,
        status: EvidenceStatus,
        score: float | None = None,
        threshold: float | None = None,
        metrics: dict[str, Any] | None = None,
        details: str = "",
        critical: bool = True,
        source: str | None = None,
    ) -> EvidenceItem:

        return EvidenceItem(
            name=name,
            status=status,
            score=score,
            threshold=threshold,
            metrics=(
                dict(metrics)
                if metrics is not None
                else {}
            ),
            details=details,
            critical=critical,
            source=source,
        )

    def set_evidence(
        self,
        name: str,
        status: EvidenceStatus,
        score: float | None = None,
        threshold: float | None = None,
        metrics: dict[str, Any] | None = None,
        details: str = "",
        critical: bool = True,
        source: str | None = None,
    ) -> "ResearchEvidenceBuilder":

        valid_names = {
            item.name
            for item in self.evidence.items()
        }

        if name not in valid_names:
            raise ValueError(
                f"Unknown evidence category: {name}"
            )

        item = self._make_item(
            name=name,
            status=status,
            score=score,
            threshold=threshold,
            metrics=metrics,
            details=details,
            critical=critical,
            source=source,
        )

        setattr(
            self.evidence,
            name,
            item,
        )

        return self

    # -----------------------------------------------------------------
    # Stage-specific helpers
    # -----------------------------------------------------------------

    def set_validation(
        self,
        passed: bool,
        accuracy: float | None = None,
        threshold: float = 0.95,
        metrics: dict[str, Any] | None = None,
        details: str = "",
    ) -> "ResearchEvidenceBuilder":

        return self.set_evidence(
            name="walk_forward_validation",
            status=(
                EvidenceStatus.PASS
                if passed
                else EvidenceStatus.FAIL
            ),
            score=accuracy,
            threshold=threshold,
            metrics=metrics,
            details=details,
            critical=True,
            source="walk_forward_research",
        )

    def set_holdout(
        self,
        passed: bool,
        accuracy: float | None = None,
        threshold: float = 0.95,
        metrics: dict[str, Any] | None = None,
        details: str = "",
    ) -> "ResearchEvidenceBuilder":

        return self.set_evidence(
            name="final_holdout",
            status=(
                EvidenceStatus.PASS
                if passed
                else EvidenceStatus.FAIL
            ),
            score=accuracy,
            threshold=threshold,
            metrics=metrics,
            details=details,
            critical=True,
            source="final_holdout",
        )

    def set_calibration(
        self,
        passed: bool,
        brier_score: float | None = None,
        ece: float | None = None,
        metrics: dict[str, Any] | None = None,
        details: str = "",
    ) -> "ResearchEvidenceBuilder":

        return self.set_evidence(
            name="probability_calibration",
            status=(
                EvidenceStatus.PASS
                if passed
                else EvidenceStatus.FAIL
            ),
            score=brier_score,
            metrics={
                **(
                    metrics
                    if metrics is not None
                    else {}
                ),
                "brier_score": brier_score,
                "ece": ece,
            },
            details=details,
            critical=True,
            source="calibration_research",
        )

    def set_range_validation(
        self,
        passed: bool,
        coverage: float | None = None,
        minimum_coverage: float = 0.70,
        metrics: dict[str, Any] | None = None,
        details: str = "",
    ) -> "ResearchEvidenceBuilder":

        return self.set_evidence(
            name="target_range_validation",
            status=(
                EvidenceStatus.PASS
                if passed
                else EvidenceStatus.FAIL
            ),
            score=coverage,
            threshold=minimum_coverage,
            metrics=metrics,
            details=details,
            critical=True,
            source="range_research",
        )

    def set_regime(
        self,
        passed: bool,
        stability_score: float | None = None,
        minimum_stability: float = 0.60,
        metrics: dict[str, Any] | None = None,
        details: str = "",
    ) -> "ResearchEvidenceBuilder":

        return self.set_evidence(
            name="regime_validation",
            status=(
                EvidenceStatus.PASS
                if passed
                else EvidenceStatus.FAIL
            ),
            score=stability_score,
            threshold=minimum_stability,
            metrics=metrics,
            details=details,
            critical=True,
            source="regime_research",
        )

    def set_backtest(
        self,
        passed: bool,
        profit_factor: float | None = None,
        sharpe: float | None = None,
        max_drawdown: float | None = None,
        trades: int | None = None,
        metrics: dict[str, Any] | None = None,
        details: str = "",
    ) -> "ResearchEvidenceBuilder":

        return self.set_evidence(
            name="backtest",
            status=(
                EvidenceStatus.PASS
                if passed
                else EvidenceStatus.FAIL
            ),
            score=profit_factor,
            metrics={
                **(
                    metrics
                    if metrics is not None
                    else {}
                ),
                "profit_factor": profit_factor,
                "sharpe": sharpe,
                "max_drawdown": max_drawdown,
                "trades": trades,
            },
            details=details,
            critical=True,
            source="backtest_research",
        )

    def set_robustness(
        self,
        passed: bool,
        robustness_score: float | None = None,
        minimum_score: float = 0.60,
        metrics: dict[str, Any] | None = None,
        details: str = "",
    ) -> "ResearchEvidenceBuilder":

        return self.set_evidence(
            name="robustness",
            status=(
                EvidenceStatus.PASS
                if passed
                else EvidenceStatus.FAIL
            ),
            score=robustness_score,
            threshold=minimum_score,
            metrics=metrics,
            details=details,
            critical=True,
            source="robustness_research",
        )

    def set_leakage(
        self,
        passed: bool,
        metrics: dict[str, Any] | None = None,
        details: str = "",
    ) -> "ResearchEvidenceBuilder":

        return self.set_evidence(
            name="leakage_audit",
            status=(
                EvidenceStatus.PASS
                if passed
                else EvidenceStatus.FAIL
            ),
            metrics=metrics,
            details=details,
            critical=True,
            source="leakage_audit",
        )

    # -----------------------------------------------------------------
    # Warnings and notes
    # -----------------------------------------------------------------

    def add_warning(
        self,
        message: str,
    ) -> "ResearchEvidenceBuilder":

        if message:
            self.evidence.warnings.append(
                str(message)
            )

        return self

    def add_note(
        self,
        message: str,
    ) -> "ResearchEvidenceBuilder":

        if message:
            self.evidence.notes.append(
                str(message)
            )

        return self

    # -----------------------------------------------------------------
    # Finalize
    # -----------------------------------------------------------------

    def build(
        self,
    ) -> ResearchEvidence:

        return self.evidence


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------


def create_research_evidence(
    model_id: str | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    horizon: int | None = None,
) -> ResearchEvidenceBuilder:

    return ResearchEvidenceBuilder(
        model_id=model_id,
        symbol=symbol,
        timeframe=timeframe,
        horizon=horizon,
    )


# ---------------------------------------------------------------------
# Production safety helper
# ---------------------------------------------------------------------


def assert_research_ready(
    evidence: ResearchEvidence,
) -> None:

    if not isinstance(
        evidence,
        ResearchEvidence,
    ):
        raise TypeError(
            "evidence must be a ResearchEvidence instance."
        )

    failures = evidence.critical_failures()

    if failures:
        names = ", ".join(
            item.name
            for item in failures
        )

        raise RuntimeError(
            "Research evidence contains critical "
            f"failures: {names}"
        )

    missing = evidence.critical_missing()

    if missing:
        names = ", ".join(
            item.name
            for item in missing
        )

        raise RuntimeError(
            "Research evidence is incomplete. "
            f"Missing required evidence: {names}"
        )


__all__ = [
    "EvidenceStatus",
    "EvidenceItem",
    "ResearchEvidence",
    "ResearchEvidenceBuilder",
    "create_research_evidence",
    "assert_research_ready",
]
