"""
AI Swing Analyser — Final Holdout Evidence Adapter.

Converts a protected final-holdout pipeline result into the common
ResearchEvidence representation.

Important:
    A successful holdout evaluation is evidence only.
    It does NOT approve a model for production.
"""

from __future__ import annotations

from typing import Any

from .integration import (
    EvidenceItem,
    EvidenceStatus,
    ResearchEvidence,
)
from .pipeline_holdout import HoldoutPipelineResult


FINAL_HOLDOUT_EVIDENCE_NAME = "final_holdout"


def _safe_float(
    value: Any,
) -> float | None:
    """Safely convert a value to float."""

    if value is None:
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if result != result:
        return None

    return result


def _build_metrics(
    result: HoldoutPipelineResult,
) -> dict[str, Any]:
    """Extract serializable holdout metrics."""

    metrics: dict[str, Any] = {}

    if result.accuracy is not None:
        metrics["accuracy"] = float(
            result.accuracy
        )

    metrics["accuracy_threshold"] = (
        result.stage.gate.metadata.get(
            "accuracy_threshold",
            0.95,
        )
    )

    if result.holdout_result is not None:
        evaluation = result.holdout_result

        for attribute in (
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "brier_score",
            "log_loss",
        ):
            value = getattr(
                evaluation,
                attribute,
                None,
            )

            value = _safe_float(value)

            if value is not None:
                metrics[attribute] = value

        classification = getattr(
            evaluation,
            "classification",
            None,
        )

        if classification is not None:
            for attribute in (
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "brier_score",
                "log_loss",
            ):
                value = getattr(
                    classification,
                    attribute,
                    None,
                )

                value = _safe_float(value)

                if value is not None:
                    metrics[attribute] = value

    return metrics


def adapt_final_holdout_result(
    result: HoldoutPipelineResult,
    *,
    critical: bool = True,
) -> EvidenceItem:
    """
    Convert a protected holdout result into EvidenceItem.

    Fail-closed behavior:
        - no evaluation → NOT_EVALUATED
        - unsuccessful evaluation → FAIL
        - evaluated below threshold → FAIL
        - evaluated above threshold → PASS

    This function never changes production approval state.
    """

    if not isinstance(
        result,
        HoldoutPipelineResult,
    ):
        raise TypeError(
            "result must be a HoldoutPipelineResult."
        )

    metrics = _build_metrics(
        result
    )

    threshold = 0.95

    if result.accuracy is not None:
        threshold = max(
            0.0,
            min(
                1.0,
                threshold,
            ),
        )

    if not result.evaluated:
        return EvidenceItem(
            name=FINAL_HOLDOUT_EVIDENCE_NAME,
            status=EvidenceStatus.NOT_EVALUATED,
            score=None,
            threshold=threshold,
            metrics=metrics,
            details=(
                "Final holdout evaluation was not performed "
                "because the holdout protection gate did not "
                "permit evaluation."
            ),
            critical=critical,
            source="ProtectedHoldoutPipeline",
        )

    if not result.completed:
        return EvidenceItem(
            name=FINAL_HOLDOUT_EVIDENCE_NAME,
            status=EvidenceStatus.FAIL,
            score=_safe_float(
                result.accuracy
            ),
            threshold=threshold,
            metrics=metrics,
            details=(
                "Final holdout stage did not complete successfully."
            ),
            critical=critical,
            source="ProtectedHoldoutPipeline",
        )

    accuracy = _safe_float(
        result.accuracy
    )

    if accuracy is None:
        return EvidenceItem(
            name=FINAL_HOLDOUT_EVIDENCE_NAME,
            status=EvidenceStatus.FAIL,
            score=None,
            threshold=threshold,
            metrics=metrics,
            details=(
                "Final holdout evaluation completed, but no "
                "valid accuracy value was available."
            ),
            critical=critical,
            source="ProtectedHoldoutPipeline",
        )

    if accuracy < threshold:
        return EvidenceItem(
            name=FINAL_HOLDOUT_EVIDENCE_NAME,
            status=EvidenceStatus.FAIL,
            score=accuracy,
            threshold=threshold,
            metrics=metrics,
            details=(
                f"Final holdout accuracy "
                f"{accuracy:.4f} is below the required "
                f"{threshold:.4f} threshold."
            ),
            critical=critical,
            source="ProtectedHoldoutPipeline",
        )

    return EvidenceItem(
        name=FINAL_HOLDOUT_EVIDENCE_NAME,
        status=EvidenceStatus.PASS,
        score=accuracy,
        threshold=threshold,
        metrics=metrics,
        details=(
            f"Final holdout accuracy "
            f"{accuracy:.4f} meets the required "
            f"{threshold:.4f} threshold."
        ),
        critical=critical,
        source="ProtectedHoldoutPipeline",
    )


def attach_final_holdout_evidence(
    evidence: ResearchEvidence,
    result: HoldoutPipelineResult,
    *,
    critical: bool = True,
) -> ResearchEvidence:
    """
    Attach final holdout evidence to an existing evidence object.

    The supplied ResearchEvidence object is updated through its existing
    evidence-builder interface where available.
    """

    if not isinstance(
        evidence,
        ResearchEvidence,
    ):
        raise TypeError(
            "evidence must be a ResearchEvidence object."
        )

    item = adapt_final_holdout_result(
        result,
        critical=critical,
    )

    setter = getattr(
        evidence,
        "set_evidence",
        None,
    )

    if callable(setter):
        setter(
            item.name,
            item,
        )
        return evidence

    final_holdout = getattr(
        evidence,
        "final_holdout",
        None,
    )

    if isinstance(
        final_holdout,
        dict,
    ):
        final_holdout[item.name] = item
        return evidence

    raise AttributeError(
        "ResearchEvidence does not expose a compatible "
        "evidence insertion interface."
    )


def final_holdout_evidence_summary(
    result: HoldoutPipelineResult,
) -> dict[str, Any]:
    """Return a compact summary suitable for dashboards/logs."""

    item = adapt_final_holdout_result(
        result
    )

    return {
        "name": item.name,
        "status": item.status.value,
        "score": item.score,
        "threshold": item.threshold,
        "critical": item.critical,
        "metrics": dict(
            item.metrics
        ),
        "details": item.details,
        "source": item.source,
    }


__all__ = [
    "FINAL_HOLDOUT_EVIDENCE_NAME",
    "adapt_final_holdout_result",
    "attach_final_holdout_evidence",
    "final_holdout_evidence_summary",
]
