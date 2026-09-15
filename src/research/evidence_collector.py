"""
AI Swing Analyser — Research Evidence Collector.

Centralizes research-stage outputs into the common ResearchEvidence
structure.

Supported evidence:
- Walk-forward validation
- Final holdout
- Calibration
- Range validation
- Regime validation
- Backtesting
- Robustness
- Leakage audit

Design principles:
- Fail closed.
- Missing evidence is never treated as PASS.
- Final holdout remains isolated.
- This module does not train models.
- This module does not approve production models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .final_holdout_evidence import (
    adapt_final_holdout_result,
)
from .integration import (
    EvidenceItem,
    EvidenceStatus,
    ResearchEvidence,
    ResearchEvidenceBuilder,
)
from .stage_adapters import (
    adapt_backtest_result,
    adapt_calibration_result,
    adapt_leakage_result,
    adapt_range_result,
    adapt_regime_result,
    adapt_robustness_result,
    adapt_stage_result,
    adapt_walk_forward_result,
)


# ---------------------------------------------------------------------
# Evidence stage names
# ---------------------------------------------------------------------

WALK_FORWARD_STAGE = "walk_forward_validation"
FINAL_HOLDOUT_STAGE = "final_holdout"
CALIBRATION_STAGE = "calibration"
RANGE_STAGE = "range_validation"
REGIME_STAGE = "regime_validation"
BACKTEST_STAGE = "backtest"
ROBUSTNESS_STAGE = "robustness"
LEAKAGE_STAGE = "leakage_audit"


REQUIRED_EVIDENCE_STAGES = (
    WALK_FORWARD_STAGE,
    FINAL_HOLDOUT_STAGE,
    CALIBRATION_STAGE,
    RANGE_STAGE,
    REGIME_STAGE,
    BACKTEST_STAGE,
    ROBUSTNESS_STAGE,
    LEAKAGE_STAGE,
)


# ---------------------------------------------------------------------
# Collection result
# ---------------------------------------------------------------------


@dataclass
class EvidenceCollectionResult:
    """Result returned by the centralized evidence collector."""

    evidence: ResearchEvidence

    collected_stages: list[str] = field(
        default_factory=list
    )

    missing_stages: list[str] = field(
        default_factory=list
    )

    failed_stages: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    errors: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def complete(self) -> bool:
        """True only when all required evidence is present."""

        return not self.missing_stages and not self.errors

    @property
    def production_eligible(self) -> bool:
        """
        Evidence alone never grants approval.

        This property is deliberately conservative.
        """

        if not self.complete:
            return False

        if self.failed_stages:
            return False

        try:
            return bool(
                self.evidence.production_eligible()
            )
        except Exception:
            return False

    def summary(self) -> dict[str, Any]:
        return {
            "model_id": getattr(
                self.evidence,
                "model_id",
                None,
            ),
            "complete": self.complete,
            "production_eligible": (
                self.production_eligible
            ),
            "collected_stages": list(
                self.collected_stages
            ),
            "missing_stages": list(
                self.missing_stages
            ),
            "failed_stages": list(
                self.failed_stages
            ),
            "warning_count": len(
                self.warnings
            ),
            "error_count": len(
                self.errors
            ),
            "research_only": True,
        }


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------


def _get_model_id(
    result: Any,
) -> str | None:
    """Extract model identity from a stage result."""

    if result is None:
        return None

    value = getattr(
        result,
        "model_id",
        None,
    )

    if value is None:
        metadata = getattr(
            result,
            "metadata",
            None,
        )

        if isinstance(
            metadata,
            Mapping,
        ):
            value = metadata.get(
                "model_id"
            )

    if value is None:
        return None

    return str(value)


def _normalise_stage_name(
    name: str,
) -> str:
    """Normalize stage names used by the collector."""

    aliases = {
        "walk_forward": WALK_FORWARD_STAGE,
        "walk_forward_validation": WALK_FORWARD_STAGE,
        "holdout": FINAL_HOLDOUT_STAGE,
        "final_holdout": FINAL_HOLDOUT_STAGE,
        "calibration": CALIBRATION_STAGE,
        "range": RANGE_STAGE,
        "range_validation": RANGE_STAGE,
        "regime": REGIME_STAGE,
        "regime_validation": REGIME_STAGE,
        "backtest": BACKTEST_STAGE,
        "robustness": ROBUSTNESS_STAGE,
        "leakage": LEAKAGE_STAGE,
        "leakage_audit": LEAKAGE_STAGE,
    }

    key = str(
        name
    ).strip().lower()

    return aliases.get(
        key,
        key,
    )


def _attach(
    evidence: ResearchEvidence,
    item: EvidenceItem,
    stage_name: str,
) -> None:
    """
    Attach an EvidenceItem using the public ResearchEvidence API.

    The primary path uses set_evidence(). A dictionary fallback is
    retained for compatibility with older ResearchEvidence versions.
    """

    if hasattr(
        evidence,
        "set_evidence",
    ):
        evidence.set_evidence(
            stage_name,
            item,
        )
        return

    for attribute in (
        "evidence",
        "categories",
        "items",
    ):
        container = getattr(
            evidence,
            attribute,
            None,
        )

        if isinstance(
            container,
            dict,
        ):
            container[
                stage_name
            ] = item
            return

    raise TypeError(
        "ResearchEvidence does not expose a supported "
        "evidence attachment interface."
    )


def _new_evidence(
    model_id: str,
) -> ResearchEvidence:
    """
    Construct ResearchEvidence through its public constructor.

    This keeps construction in one place and makes compatibility
    issues explicit.
    """

    try:
        return ResearchEvidence(
            model_id=model_id
        )
    except TypeError:
        try:
            return ResearchEvidence(
                model_id
            )
        except TypeError as exc:
            raise TypeError(
                "Unable to construct ResearchEvidence "
                "with the available API."
            ) from exc


# ---------------------------------------------------------------------
# Individual stage collection
# ---------------------------------------------------------------------


def _adapt_stage(
    stage_name: str,
    result: Any,
) -> EvidenceItem:
    """Convert one research-stage result into EvidenceItem."""

    stage = _normalise_stage_name(
        stage_name
    )

    if stage == WALK_FORWARD_STAGE:
        return adapt_walk_forward_result(
            result
        )

    if stage == FINAL_HOLDOUT_STAGE:
        return adapt_final_holdout_result(
            result
        )

    if stage == CALIBRATION_STAGE:
        return adapt_calibration_result(
            result
        )

    if stage == RANGE_STAGE:
        return adapt_range_result(
            result
        )

    if stage == REGIME_STAGE:
        return adapt_regime_result(
            result
        )

    if stage == BACKTEST_STAGE:
        return adapt_backtest_result(
            result
        )

    if stage == ROBUSTNESS_STAGE:
        return adapt_robustness_result(
            result
        )

    if stage == LEAKAGE_STAGE:
        return adapt_leakage_result(
            result
        )

    return adapt_stage_result(
        stage,
        result,
    )


# ---------------------------------------------------------------------
# Main collector
# ---------------------------------------------------------------------


class ResearchEvidenceCollector:
    """
    Collect independent research-stage outputs into one evidence object.
    """

    def __init__(
        self,
        *,
        required_stages: tuple[
            str,
            ...
        ] = REQUIRED_EVIDENCE_STAGES,
    ) -> None:
        self.required_stages = tuple(
            _normalise_stage_name(
                stage
            )
            for stage in required_stages
        )

    def collect(
        self,
        *,
        model_id: str,
        stages: Mapping[str, Any] | None = None,
    ) -> EvidenceCollectionResult:
        """
        Collect stage results.

        Missing stages remain NOT_EVALUATED and therefore cannot
        satisfy production eligibility.
        """

        if not isinstance(
            model_id,
            str,
        ) or not model_id.strip():
            raise ValueError(
                "model_id must be a non-empty string."
            )

        stage_map = (
            {}
            if stages is None
            else dict(stages)
        )

        evidence = _new_evidence(
            model_id
        )

        collected: list[str] = []
        missing: list[str] = []
        failed: list[str] = []
        warnings: list[str] = []
        errors: list[str] = []

        for required_stage in self.required_stages:
            result = None

            for supplied_name, supplied_result in (
                stage_map.items()
            ):
                if (
                    _normalise_stage_name(
                        supplied_name
                    )
                    == required_stage
                ):
                    result = supplied_result
                    break

            if result is None:
                missing.append(
                    required_stage
                )

                item = EvidenceItem(
                    name=required_stage,
                    status=EvidenceStatus.NOT_EVALUATED,
                    score=None,
                    threshold=None,
                    metrics={},
                    details=(
                        "Required evidence was not supplied."
                    ),
                    critical=True,
                    source=(
                        "ResearchEvidenceCollector"
                    ),
                )

                _attach(
                    evidence,
                    item,
                    required_stage,
                )

                continue

            try:
                item = _adapt_stage(
                    required_stage,
                    result,
                )

                _attach(
                    evidence,
                    item,
                    required_stage,
                )

                collected.append(
                    required_stage
                )

                if (
                    item.status
                    == EvidenceStatus.FAIL
                ):
                    failed.append(
                        required_stage
                    )

                if (
                    item.status
                    == EvidenceStatus.WARNING
                ):
                    warnings.append(
                        f"{required_stage}: "
                        f"{item.details}"
                    )

                if (
                    item.status
                    == EvidenceStatus.NOT_EVALUATED
                ):
                    missing.append(
                        required_stage
                    )

            except Exception as exc:
                errors.append(
                    f"{required_stage}: {exc}"
                )

                failed.append(
                    required_stage
                )

                item = EvidenceItem(
                    name=required_stage,
                    status=EvidenceStatus.FAIL,
                    score=None,
                    threshold=None,
                    metrics={},
                    details=(
                        "Evidence adaptation failed: "
                        f"{exc}"
                    ),
                    critical=True,
                    source=(
                        "ResearchEvidenceCollector"
                    ),
                )

                _attach(
                    evidence,
                    item,
                    required_stage,
                )

        return EvidenceCollectionResult(
            evidence=evidence,
            collected_stages=collected,
            missing_stages=missing,
            failed_stages=failed,
            warnings=warnings,
            errors=errors,
            metadata={
                "research_only": True,
                "production_approved": False,
                "final_holdout_used_for_collection": (
                    FINAL_HOLDOUT_STAGE
                    in collected
                ),
            },
        )


# ---------------------------------------------------------------------
# Convenience collector
# ---------------------------------------------------------------------


def collect_research_evidence(
    *,
    model_id: str,
    stages: Mapping[str, Any] | None = None,
) -> EvidenceCollectionResult:
    """Convenience wrapper around ResearchEvidenceCollector."""

    collector = ResearchEvidenceCollector()

    return collector.collect(
        model_id=model_id,
        stages=stages,
    )


# ---------------------------------------------------------------------
# Final holdout evidence API
# ---------------------------------------------------------------------


def collect_final_holdout_evidence(
    result: Any,
) -> EvidenceItem:
    """
    Convert a protected final-holdout result to EvidenceItem.

    None means that the holdout has not been evaluated.
    """

    if result is None:
        return EvidenceItem(
            name=FINAL_HOLDOUT_STAGE,
            status=EvidenceStatus.NOT_EVALUATED,
            score=None,
            threshold=0.95,
            metrics={},
            details=(
                "Final holdout has not been evaluated."
            ),
            critical=True,
            source=(
                "final_holdout_evidence"
            ),
        )

    return adapt_final_holdout_result(
        result
    )


def attach_final_holdout_to_evidence(
    evidence: ResearchEvidence,
    result: Any,
) -> ResearchEvidence:
    """Attach final-holdout evidence to an existing evidence object."""

    if not isinstance(
        evidence,
        ResearchEvidence,
    ):
        raise TypeError(
            "evidence must be ResearchEvidence."
        )

    item = collect_final_holdout_evidence(
        result
    )

    _attach(
        evidence,
        item,
        FINAL_HOLDOUT_STAGE,
    )

    return evidence


def build_final_holdout_evidence(
    model_id: str,
    result: Any,
) -> ResearchEvidence:
    """Create ResearchEvidence containing final-holdout evidence."""

    if not isinstance(
        model_id,
        str,
    ) or not model_id.strip():
        raise ValueError(
            "model_id must be a non-empty string."
        )

    evidence = _new_evidence(
        model_id
    )

    attach_final_holdout_to_evidence(
        evidence,
        result,
    )

    return evidence


def final_holdout_passed(
    result: Any,
) -> bool:
    """Return True only when final-holdout evidence explicitly passes."""

    item = collect_final_holdout_evidence(
        result
    )

    return (
        item.status
        == EvidenceStatus.PASS
    )


def final_holdout_evaluated(
    result: Any,
) -> bool:
    """Return whether final-holdout evaluation actually occurred."""

    item = collect_final_holdout_evidence(
        result
    )

    return (
        item.status
        != EvidenceStatus.NOT_EVALUATED
    )


def final_holdout_status(
    result: Any,
) -> EvidenceStatus:
    """Return final-holdout evidence status."""

    return collect_final_holdout_evidence(
        result
    ).status


def final_holdout_summary(
    result: Any,
) -> dict[str, Any]:
    """Return a compact final-holdout evidence summary."""

    item = collect_final_holdout_evidence(
        result
    )

    return {
        "name": item.name,
        "status": item.status.value,
        "score": item.score,
        "threshold": item.threshold,
        "metrics": dict(
            item.metrics
        ),
        "critical": item.critical,
        "source": item.source,
    }


def validate_final_holdout_identity(
    result: Any,
    model_id: str,
) -> bool:
    """
    Verify that final-holdout evidence belongs to the expected model.
    """

    if result is None:
        return False

    result_model_id = _get_model_id(
        result
    )

    if result_model_id is None:
        return False

    return (
        result_model_id
        == str(model_id)
    )


def collect_holdout_evidence_map(
    results: Mapping[str, Any],
) -> dict[str, EvidenceItem]:
    """
    Convert multiple model-specific holdout results into evidence items.

    Every result must identify its model.
    """

    if not isinstance(
        results,
        Mapping,
    ):
        raise TypeError(
            "results must be a mapping."
        )

    output: dict[
        str,
        EvidenceItem,
    ] = {}

    for model_id, result in results.items():
        if not isinstance(
            model_id,
            str,
        ) or not model_id.strip():
            raise ValueError(
                "Holdout model IDs must be non-empty strings."
            )

        if result is None:
            output[
                model_id
            ] = collect_final_holdout_evidence(
                None
            )
            continue

        result_model_id = _get_model_id(
            result
        )

        if (
            result_model_id is not None
            and result_model_id
            != model_id
        ):
            raise ValueError(
                "Holdout result model identity does not "
                f"match key '{model_id}'."
            )

        output[
            model_id
        ] = collect_final_holdout_evidence(
            result
        )

    return output


__all__ = [
    "WALK_FORWARD_STAGE",
    "FINAL_HOLDOUT_STAGE",
    "CALIBRATION_STAGE",
    "RANGE_STAGE",
    "REGIME_STAGE",
    "BACKTEST_STAGE",
    "ROBUSTNESS_STAGE",
    "LEAKAGE_STAGE",
    "REQUIRED_EVIDENCE_STAGES",
    "EvidenceCollectionResult",
    "ResearchEvidenceCollector",
    "collect_research_evidence",
    "collect_final_holdout_evidence",
    "attach_final_holdout_to_evidence",
    "build_final_holdout_evidence",
    "final_holdout_passed",
    "final_holdout_evaluated",
    "final_holdout_status",
    "final_holdout_summary",
    "validate_final_holdout_identity",
    "collect_holdout_evidence_map",
]
