"""
Tests for final holdout evidence adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.research.final_holdout_evidence import (
    FINAL_HOLDOUT_EVIDENCE_NAME,
    adapt_final_holdout_result,
    attach_final_holdout_evidence,
    final_holdout_evidence_summary,
)
from src.research.integration import (
    EvidenceItem,
    EvidenceStatus,
    ResearchEvidence,
)
from src.research.pipeline_holdout import (
    HoldoutPipelineResult,
)


@dataclass
class FakeGate:
    status: object
    metadata: dict


@dataclass
class FakeStage:
    gate: FakeGate
    evaluation: object | None
    successful: bool
    final_holdout_used: bool
    holdout_accuracy: float | None
    warnings: tuple[str, ...] = ()

    @property
    def evaluated(self) -> bool:
        return self.evaluation is not None

    @property
    def eligible_for_evaluation(self) -> bool:
        return self.gate.status.value == "READY"


class FakeHoldoutResult:
    """
    Lightweight result object used to isolate evidence-adapter
    behavior from model-training implementation details.
    """

    def __init__(
        self,
        *,
        model_id: str = "model_test",
        accuracy: float | None = 0.97,
        completed: bool = True,
        evaluated: bool = True,
        holdout_used: bool = True,
        passed: bool = True,
    ):
        class Status:
            value = "READY"

        gate = FakeGate(
            status=Status(),
            metadata={
                "accuracy_threshold": 0.95,
            },
        )

        evaluation = (
            object()
            if evaluated
            else None
        )

        self.model_id = model_id
        self.stage = FakeStage(
            gate=gate,
            evaluation=evaluation,
            successful=completed,
            final_holdout_used=holdout_used,
            holdout_accuracy=accuracy,
        )
        self.completed = completed
        self.accuracy = accuracy
        self.passed_accuracy_gate = passed
        self.production_approved = False
        self.metadata = {
            "research_only": True,
            "production_approved": False,
        }


def make_result(
    *,
    accuracy: float | None = 0.97,
    completed: bool = True,
    evaluated: bool = True,
    passed: bool = True,
):
    return FakeHoldoutResult(
        accuracy=accuracy,
        completed=completed,
        evaluated=evaluated,
        passed=passed,
    )


def test_evidence_name():
    assert (
        FINAL_HOLDOUT_EVIDENCE_NAME
        == "final_holdout"
    )


def test_successful_holdout_becomes_pass():
    result = make_result(
        accuracy=0.97,
        completed=True,
        evaluated=True,
        passed=True,
    )

    item = adapt_final_holdout_result(
        result
    )

    assert isinstance(
        item,
        EvidenceItem,
    )

    assert item.name == (
        FINAL_HOLDOUT_EVIDENCE_NAME
    )

    assert item.status == EvidenceStatus.PASS
    assert item.score == pytest.approx(0.97)
    assert item.critical is True


def test_accuracy_below_threshold_becomes_fail():
    result = make_result(
        accuracy=0.94,
        completed=True,
        evaluated=True,
        passed=False,
    )

    item = adapt_final_holdout_result(
        result
    )

    assert item.status == EvidenceStatus.FAIL
    assert item.score == pytest.approx(0.94)
    assert item.threshold == pytest.approx(
        0.95
    )


def test_not_evaluated_becomes_not_evaluated():
    result = make_result(
        accuracy=None,
        completed=False,
        evaluated=False,
        passed=False,
    )

    item = adapt_final_holdout_result(
        result
    )

    assert (
        item.status
        == EvidenceStatus.NOT_EVALUATED
    )

    assert item.score is None


def test_incomplete_evaluation_becomes_fail():
    result = make_result(
        accuracy=0.96,
        completed=False,
        evaluated=True,
        passed=False,
    )

    item = adapt_final_holdout_result(
        result
    )

    assert item.status == EvidenceStatus.FAIL


def test_missing_accuracy_becomes_fail():
    result = make_result(
        accuracy=None,
        completed=True,
        evaluated=True,
        passed=False,
    )

    item = adapt_final_holdout_result(
        result
    )

    assert item.status == EvidenceStatus.FAIL
    assert item.score is None


def test_custom_critical_flag():
    result = make_result()

    item = adapt_final_holdout_result(
        result,
        critical=False,
    )

    assert item.critical is False


def test_source_is_protected_holdout_pipeline():
    result = make_result()

    item = adapt_final_holdout_result(
        result
    )

    assert (
        item.source
        == "ProtectedHoldoutPipeline"
    )


def test_pass_details_contain_threshold():
    result = make_result(
        accuracy=0.98
    )

    item = adapt_final_holdout_result(
        result
    )

    assert "0.9800" in item.details
    assert "0.9500" in item.details


def test_fail_details_contain_accuracy():
    result = make_result(
        accuracy=0.92,
        passed=False,
    )

    item = adapt_final_holdout_result(
        result
    )

    assert "0.9200" in item.details


def test_not_evaluated_details_explain_protection():
    result = make_result(
        accuracy=None,
        completed=False,
        evaluated=False,
        passed=False,
    )

    item = adapt_final_holdout_result(
        result
    )

    assert (
        "gate" in item.details.lower()
        or "permit" in item.details.lower()
    )


def test_invalid_result_type_fails():
    with pytest.raises(TypeError):
        adapt_final_holdout_result(
            object()
        )


def test_metrics_contain_accuracy():
    result = make_result(
        accuracy=0.975
    )

    item = adapt_final_holdout_result(
        result
    )

    assert "accuracy" in item.metrics
    assert item.metrics[
        "accuracy"
    ] == pytest.approx(0.975)


def test_metrics_are_serializable():
    result = make_result()

    item = adapt_final_holdout_result(
        result
    )

    assert isinstance(
        item.metrics,
        dict,
    )

    for value in item.metrics.values():
        assert isinstance(
            value,
            (int, float),
        )


def test_evidence_score_matches_accuracy():
    result = make_result(
        accuracy=0.965
    )

    item = adapt_final_holdout_result(
        result
    )

    assert item.score == pytest.approx(
        0.965
    )


def test_attach_requires_research_evidence():
    result = make_result()

    with pytest.raises(TypeError):
        attach_final_holdout_evidence(
            object(),
            result,
        )


def test_attach_requires_valid_result():
    evidence = ResearchEvidence(
        model_id="model_test"
    )

    with pytest.raises(TypeError):
        attach_final_holdout_evidence(
            evidence,
            object(),
        )


def test_final_holdout_summary():
    result = make_result(
        accuracy=0.97
    )

    summary = final_holdout_evidence_summary(
        result
    )

    assert summary["name"] == (
        FINAL_HOLDOUT_EVIDENCE_NAME
    )

    assert summary["status"] == "PASS"
    assert summary["score"] == pytest.approx(
        0.97
    )

    assert summary["threshold"] == pytest.approx(
        0.95
    )

    assert summary["critical"] is True


def test_summary_contains_metrics():
    result = make_result(
        accuracy=0.96
    )

    summary = final_holdout_evidence_summary(
        result
    )

    assert "metrics" in summary
    assert (
        summary["metrics"]["accuracy"]
        == pytest.approx(0.96)
    )


def test_summary_contains_details():
    result = make_result()

    summary = final_holdout_evidence_summary(
        result
    )

    assert summary["details"]
    assert summary["source"] == (
        "ProtectedHoldoutPipeline"
    )


def test_evidence_never_grants_production_approval():
    result = make_result(
        accuracy=1.0
    )

    item = adapt_final_holdout_result(
        result
    )

    assert item.status == EvidenceStatus.PASS

    # Evidence status is not production approval.
    assert result.production_approved is False


def test_holdout_pass_does_not_override_research_only_state():
    result = make_result(
        accuracy=1.0
    )

    item = adapt_final_holdout_result(
        result
    )

    assert item.status == EvidenceStatus.PASS
    assert result.metadata[
        "research_only"
    ] is True


def test_low_accuracy_cannot_become_warning():
    result = make_result(
        accuracy=0.50,
        passed=False,
    )

    item = adapt_final_holdout_result(
        result
    )

    assert item.status == EvidenceStatus.FAIL


def test_evidence_threshold_is_strict_95_percent():
    result = make_result(
        accuracy=0.949999,
        passed=False,
    )

    item = adapt_final_holdout_result(
        result
    )

    assert item.status == EvidenceStatus.FAIL


def test_exact_95_percent_passes():
    result = make_result(
        accuracy=0.95,
        passed=True,
    )

    item = adapt_final_holdout_result(
        result
    )

    assert item.status == EvidenceStatus.PASS


def test_above_95_percent_passes():
    result = make_result(
        accuracy=0.951,
        passed=True,
    )

    item = adapt_final_holdout_result(
        result
    )

    assert item.status == EvidenceStatus.PASS


def test_adapter_is_deterministic():
    result = make_result(
        accuracy=0.973
    )

    item_1 = adapt_final_holdout_result(
        result
    )

    item_2 = adapt_final_holdout_result(
        result
    )

    assert item_1 == item_2


def test_adapter_does_not_modify_result():
    result = make_result(
        accuracy=0.97
    )

    original_accuracy = (
        result.accuracy
    )

    original_approval = (
        result.production_approved
    )

    adapt_final_holdout_result(
        result
    )

    assert (
        result.accuracy
        == original_accuracy
    )

    assert (
        result.production_approved
        == original_approval
    )


def test_fail_closed_when_result_not_completed():
    result = make_result(
        accuracy=0.99,
        completed=False,
        evaluated=True,
        passed=True,
    )

    item = adapt_final_holdout_result(
        result
    )

    assert item.status == EvidenceStatus.FAIL


def test_not_evaluated_is_not_a_pass():
    result = make_result(
        accuracy=None,
        completed=False,
        evaluated=False,
        passed=False,
    )

    item = adapt_final_holdout_result(
        result
    )

    assert item.status != EvidenceStatus.PASS


def test_holdout_evidence_is_critical_by_default():
    result = make_result()

    item = adapt_final_holdout_result(
        result
    )

    assert item.critical is True
