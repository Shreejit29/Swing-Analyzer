"""
AI Swing Analyser — Master Research Pipeline Integration.

Integrates all major research stages into a single fail-closed
evidence and production-readiness boundary.

Research flow:

    Data
      ↓
    Features
      ↓
    Targets
      ↓
    Model Development
      ↓
    Walk-Forward Validation
      ↓
    Calibration
      ↓
    Range Validation
      ↓
    Regime Validation
      ↓
    Backtest
      ↓
    Robustness
      ↓
    Final Holdout
      ↓
    Leakage Audit
      ↓
    Evidence
      ↓
    Production Approval

This module orchestrates evidence only.
It does not train models or perform research calculations itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

from .evidence_collector import (
    collect_final_holdout_evidence,
)

from .integration import (
    EvidenceItem,
    EvidenceStatus,
    ResearchEvidence,
)

from .pipeline_holdout import (
    HoldoutPipelineResult,
)


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

FINAL_HOLDOUT_STAGE = "final_holdout"


DEFAULT_INTEGRATION_STAGES = (
    "data",
    "features",
    "targets",
    "model_development",
    "walk_forward_validation",
    "calibration",
    "range_validation",
    "regime_validation",
    "backtest",
    "robustness",
    "final_holdout",
    "leakage_audit",
)


# ----------------------------------------------------------------------
# Integration status
# ----------------------------------------------------------------------


class IntegrationStatus(str, Enum):
    """
    Overall lifecycle status of the integrated research pipeline.

    This enum provides the public status API expected by
    src.research.__init__ and other integration modules.
    """

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


# ----------------------------------------------------------------------
# Integration stage
# ----------------------------------------------------------------------


@dataclass
class IntegrationStage:
    """
    Lifecycle information for one research stage.
    """

    name: str

    status: str = "NOT_STARTED"

    started_at: str | None = None

    completed_at: str | None = None

    score: float | None = None

    metrics: dict[str, Any] = field(
        default_factory=dict
    )

    details: str = ""

    error: str | None = None

    def start(self) -> None:
        """
        Mark this stage as running.
        """

        self.status = "RUNNING"

        self.started_at = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

    def complete(
        self,
        *,
        score: float | None = None,
        metrics: Mapping[str, Any] | None = None,
        details: str = "",
    ) -> None:
        """
        Mark this stage as complete.
        """

        self.status = "COMPLETE"

        self.completed_at = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        self.score = score

        if metrics:
            self.metrics.update(
                dict(metrics)
            )

        self.details = details

    def fail(
        self,
        error: str,
    ) -> None:
        """
        Mark this stage as failed.
        """

        self.status = "FAILED"

        self.completed_at = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        self.error = str(error)


# ----------------------------------------------------------------------
# Integrated research result
# ----------------------------------------------------------------------


@dataclass
class IntegratedResearchResult:
    """
    Complete integrated research result.
    """

    model_id: str

    status: str = "CREATED"

    stages: dict[
        str,
        IntegrationStage,
    ] = field(
        default_factory=dict
    )

    evidence: ResearchEvidence | None = None

    production_eligible: bool = False

    errors: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def failed_stages(
        self,
    ) -> list[IntegrationStage]:
        """
        Return all failed integration stages.
        """

        return [
            stage
            for stage in self.stages.values()
            if stage.status == "FAILED"
        ]

    def incomplete_stages(
        self,
    ) -> list[IntegrationStage]:
        """
        Return all stages that are not complete.
        """

        return [
            stage
            for stage in self.stages.values()
            if stage.status
            not in {
                "COMPLETE",
            }
        ]

    def summary(
        self,
    ) -> dict[str, Any]:
        """
        Return a compact integration summary.
        """

        return {
            "model_id": self.model_id,
            "status": self.status,
            "stage_count": len(
                self.stages
            ),
            "completed_stages": sum(
                stage.status == "COMPLETE"
                for stage in self.stages.values()
            ),
            "failed_stages": len(
                self.failed_stages()
            ),
            "incomplete_stages": len(
                self.incomplete_stages()
            ),
            "production_eligible": (
                self.production_eligible
            ),
            "error_count": len(
                self.errors
            ),
            "warning_count": len(
                self.warnings
            ),
        }


# ----------------------------------------------------------------------
# Research pipeline integrator
# ----------------------------------------------------------------------


class ResearchPipelineIntegrator:
    """
    Master evidence integration boundary.

    The integrator is deliberately fail-closed.
    """

    def __init__(
        self,
        model_id: str,
        stage_names: tuple[str, ...] | None = None,
    ) -> None:

        if not isinstance(
            model_id,
            str,
        ):
            raise TypeError(
                "model_id must be a string."
            )

        if not model_id.strip():
            raise ValueError(
                "model_id cannot be empty."
            )

        names = (
            stage_names
            if stage_names is not None
            else DEFAULT_INTEGRATION_STAGES
        )

        if not names:
            raise ValueError(
                "At least one integration stage is required."
            )

        if len(names) != len(set(names)):
            raise ValueError(
                "Integration stage names must be unique."
            )

        self.result = IntegratedResearchResult(
            model_id=model_id,
            stages={
                name: IntegrationStage(
                    name=name
                )
                for name in names
            },
            metadata={
                "research_only": True,
                "production_approved": False,
            },
        )

    @property
    def model_id(
        self,
    ) -> str:
        """
        Return the integration model ID.
        """

        return self.result.model_id

    def start(
        self,
    ) -> IntegratedResearchResult:
        """
        Start the integrated research process.
        """

        self.result.status = (
            IntegrationStatus.RUNNING.value
        )

        self.result.metadata[
            "started_at"
        ] = datetime.now(
            timezone.utc
        ).isoformat()

        return self.result

    def start_stage(
        self,
        stage_name: str,
    ) -> IntegrationStage:
        """
        Start an individual integration stage.
        """

        stage = self._get_stage(
            stage_name
        )

        stage.start()

        return stage

    def complete_stage(
        self,
        stage_name: str,
        *,
        score: float | None = None,
        metrics: Mapping[str, Any] | None = None,
        details: str = "",
    ) -> IntegrationStage:
        """
        Complete an individual integration stage.
        """

        stage = self._get_stage(
            stage_name
        )

        stage.complete(
            score=score,
            metrics=metrics,
            details=details,
        )

        return stage

    def fail_stage(
        self,
        stage_name: str,
        error: str,
    ) -> IntegrationStage:
        """
        Fail an individual integration stage.
        """

        stage = self._get_stage(
            stage_name
        )

        stage.fail(
            error
        )

        self.result.errors.append(
            f"{stage_name}: {error}"
        )

        self.result.status = (
            IntegrationStatus.FAILED.value
        )

        return stage

    def attach_evidence(
        self,
        evidence: ResearchEvidence,
    ) -> IntegratedResearchResult:
        """
        Attach research evidence to the integration result.
        """

        if not isinstance(
            evidence,
            ResearchEvidence,
        ):
            raise TypeError(
                "evidence must be a ResearchEvidence object."
            )

        self.result.evidence = evidence

        return self.result

    def attach_final_holdout(
        self,
        holdout_result: HoldoutPipelineResult | None,
    ) -> EvidenceItem:
        """
        Attach final holdout evidence to the integrated evidence object.

        If no evidence object exists yet, one is created.
        """

        if self.result.evidence is None:
            self.result.evidence = ResearchEvidence(
                model_id=self.model_id
            )

        if (
            self.result.evidence.model_id
            != self.model_id
        ):
            raise ValueError(
                "ResearchEvidence model_id does not match "
                "the integration model_id."
            )

        item = collect_final_holdout_evidence(
            holdout_result,
            critical=True,
        )

        setter = getattr(
            self.result.evidence,
            "set_evidence",
            None,
        )

        if callable(setter):
            setter(
                FINAL_HOLDOUT_STAGE,
                item,
            )

        else:
            category = getattr(
                self.result.evidence,
                "final_holdout",
                None,
            )

            if isinstance(
                category,
                dict,
            ):
                category[
                    FINAL_HOLDOUT_STAGE
                ] = item

            else:
                raise AttributeError(
                    "ResearchEvidence does not expose a "
                    "compatible evidence insertion interface."
                )

        self._record_final_holdout_stage(
            item
        )

        return item

    def _record_final_holdout_stage(
        self,
        item: EvidenceItem,
    ) -> None:
        """
        Convert final-holdout evidence into integration-stage status.
        """

        stage = self._get_stage(
            FINAL_HOLDOUT_STAGE
        )

        if item.status == EvidenceStatus.PASS:

            stage.complete(
                score=item.score,
                metrics=item.metrics,
                details=item.details,
            )

        elif item.status == EvidenceStatus.FAIL:

            stage.fail(
                item.details
            )

            self.result.errors.append(
                f"{FINAL_HOLDOUT_STAGE}: "
                f"{item.details}"
            )

        else:

            stage.status = "NOT_EVALUATED"

            stage.details = item.details

            self.result.warnings.append(
                f"{FINAL_HOLDOUT_STAGE}: "
                f"{item.details}"
            )

    def evaluate_production_eligibility(
        self,
    ) -> bool:
        """
        Evaluate whether integrated evidence is sufficient for
        the production approval boundary.

        This method does not grant production approval.
        """

        if self.result.evidence is None:

            self.result.production_eligible = False

            return False

        failed_stages = (
            self.failed_stages()
        )

        if failed_stages:

            self.result.production_eligible = False

            return False

        incomplete = (
            self.incomplete_stages()
        )

        if incomplete:

            self.result.production_eligible = False

            return False

        evidence = self.result.evidence

        try:

            eligible = bool(
                evidence.production_eligible()
            )

        except Exception:

            eligible = False

        self.result.production_eligible = (
            eligible
        )

        return eligible

    def can_proceed(
        self,
        next_stage: str,
    ) -> bool:
        """
        Determine whether the named stage may proceed.

        Final holdout has an additional strict requirement:
        all preceding development stages must already be complete.
        """

        self._get_stage(
            next_stage
        )

        ordered = list(
            self.result.stages.keys()
        )

        position = ordered.index(
            next_stage
        )

        if position == 0:
            return True

        previous = ordered[
            :position
        ]

        return all(
            self.result.stages[name].status
            == "COMPLETE"
            for name in previous
        )

    def assert_can_proceed(
        self,
        next_stage: str,
    ) -> None:
        """
        Raise an error when the next stage is not allowed to proceed.
        """

        if not self.can_proceed(
            next_stage
        ):
            raise RuntimeError(
                f"Research pipeline cannot proceed to "
                f"'{next_stage}'. One or more preceding "
                f"stages are incomplete or failed."
            )

    def complete(
        self,
    ) -> IntegratedResearchResult:
        """
        Complete the integration only when every stage and required
        evidence item has passed.

        Production approval remains false.
        """

        if self.result.errors:

            self.result.status = (
                IntegrationStatus.FAILED.value
            )

            self.result.production_eligible = False

            return self.result

        incomplete = (
            self.incomplete_stages()
        )

        if incomplete:

            self.result.status = (
                IntegrationStatus.BLOCKED.value
            )

            self.result.production_eligible = False

            return self.result

        eligible = (
            self.evaluate_production_eligibility()
        )

        if not eligible:

            self.result.status = (
                IntegrationStatus.BLOCKED.value
            )

            self.result.production_eligible = False

            return self.result

        self.result.status = (
            IntegrationStatus.COMPLETE.value
        )

        # This integration layer never grants production approval.
        self.result.metadata[
            "production_approved"
        ] = False

        self.result.metadata[
            "completed_at"
        ] = datetime.now(
            timezone.utc
        ).isoformat()

        return self.result

    def _get_stage(
        self,
        stage_name: str,
    ) -> IntegrationStage:
        """
        Retrieve an integration stage by name.
        """

        if stage_name not in self.result.stages:

            raise KeyError(
                f"Unknown research stage: "
                f"{stage_name}"
            )

        return self.result.stages[
            stage_name
        ]

    def failed_stages(
        self,
    ) -> list[IntegrationStage]:
        """
        Return failed stages.
        """

        return self.result.failed_stages()

    def incomplete_stages(
        self,
    ) -> list[IntegrationStage]:
        """
        Return incomplete stages.
        """

        return self.result.incomplete_stages()


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


def create_research_integrator(
    model_id: str,
    stage_names: tuple[str, ...] | None = None,
) -> ResearchPipelineIntegrator:
    """
    Create a fresh master research integrator.
    """

    return ResearchPipelineIntegrator(
        model_id=model_id,
        stage_names=stage_names,
    )


# ----------------------------------------------------------------------
# Final holdout convenience function
# ----------------------------------------------------------------------


def integrate_final_holdout(
    *,
    model_id: str,
    holdout_result: HoldoutPipelineResult | None,
) -> IntegratedResearchResult:
    """
    Convenience helper for attaching final holdout evidence to a
    fresh integration result.

    The resulting pipeline remains research-only.
    """

    integrator = ResearchPipelineIntegrator(
        model_id=model_id
    )

    integrator.start()

    integrator.attach_final_holdout(
        holdout_result
    )

    return integrator.result


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


__all__ = [
    "FINAL_HOLDOUT_STAGE",
    "DEFAULT_INTEGRATION_STAGES",
    "IntegrationStatus",
    "IntegrationStage",
    "IntegratedResearchResult",
    "ResearchPipelineIntegrator",
    "create_research_integrator",
    "integrate_final_holdout",
]
