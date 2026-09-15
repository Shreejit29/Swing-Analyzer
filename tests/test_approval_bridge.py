"""
Tests for the fail-closed research approval bridge.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.research.approval_bridge import (
    ResearchApprovalBridge,
    assert_research_approved,
    evaluate_research_approval,
)
from src.research.integration import (
    EvidenceItem,
    EvidenceStatus,
    ResearchEvidence,
)
from src.research.research_config import (
    IntegratedApprovalConfig,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def passing_item(
    name: str,
    *,
    critical: bool = True,
) -> EvidenceItem:

    return EvidenceItem(
        name=name,
        status=EvidenceStatus.PASS,
        score=1.0,
        threshold=1.0,
        metrics={
            "accuracy": 0.96,
        },
        details="Gate passed.",
        critical=critical,
        source="test",
    )


def failing_item(
    name: str,
    *,
    critical: bool = True,
) -> EvidenceItem:

    return EvidenceItem(
        name=name,
        status=EvidenceStatus.FAIL,
        score=0.0,
        threshold=1.0,
        metrics={
            "accuracy": 0.80,
        },
        details="Gate failed.",
        critical=critical,
        source="test",
    )


def warning_item(
    name: str,
    *,
    critical: bool = True,
) -> EvidenceItem:

    return EvidenceItem(
        name=name,
        status=EvidenceStatus.WARNING,
        score=0.5,
        threshold=1.0,
        metrics={},
        details="Gate produced a warning.",
        critical=critical,
        source="test",
    )


def missing_item(
    name: str,
    *,
    critical: bool = True,
) -> EvidenceItem:

    return EvidenceItem(
        name=name,
        status=EvidenceStatus.NOT_EVALUATED,
        score=None,
        threshold=1.0,
        metrics={},
        details="Evidence was not evaluated.",
        critical=critical,
        source="test",
    )


def complete_passing_evidence() -> ResearchEvidence:

    return ResearchEvidence(
        model_id="MODEL-001",
        experiment_id="EXP-001",
        validation=passing_item(
            "walk_forward_validation"
        ),
        holdout=passing_item(
            "final_holdout"
        ),
        calibration=passing_item(
            "calibration"
        ),
        range_validation=passing_item(
            "range_validation"
        ),
        regime=passing_item(
            "regime_validation"
        ),
        backtest=passing_item(
            "backtest"
        ),
        robustness=passing_item(
            "robustness"
        ),
        leakage=passing_item(
            "leakage_audit"
        ),
        metadata={
            "research_only": True,
        },
    )


# ---------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------


def test_bridge_constructs_with_defaults():

    bridge = ResearchApprovalBridge()

    assert bridge.config is not None


def test_custom_approval_configuration_is_preserved():

    config = IntegratedApprovalConfig(
        minimum_validation_accuracy=0.90
    )

    bridge = ResearchApprovalBridge(
        config=config
    )

    assert (
        bridge.config.minimum_validation_accuracy
        == 0.90
    )


# ---------------------------------------------------------------------
# Complete evidence
# ---------------------------------------------------------------------


def test_complete_passing_evidence_is_approved():

    bridge = ResearchApprovalBridge()

    result = bridge.evaluate(
        complete_passing_evidence()
    )

    assert result.approved
    assert result.production_ready
    assert result.status == "APPROVED"


def test_all_mandatory_gates_are_passed():

    bridge = ResearchApprovalBridge()

    result = bridge.evaluate(
        complete_passing_evidence()
    )

    expected = {
        "walk_forward_validation",
        "final_holdout",
        "calibration",
        "range_validation",
        "regime_validation",
        "backtest",
        "robustness",
        "leakage_audit",
    }

    assert set(
        result.passed_gates
    ) == expected


def test_identity_is_preserved():

    bridge = ResearchApprovalBridge()

    result = bridge.evaluate(
        complete_passing_evidence()
    )

    assert result.model_id == "MODEL-001"
    assert result.experiment_id == "EXP-001"


# ---------------------------------------------------------------------
# Fail-closed behavior
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "field_name",
    [
        "validation",
        "holdout",
        "calibration",
        "range_validation",
        "regime",
        "backtest",
        "robustness",
        "leakage",
    ],
)
def test_missing_required_evidence_causes_hold(
    field_name,
):

    evidence = complete_passing_evidence()

    setattr(
        evidence,
        field_name,
        None,
    )

    bridge = ResearchApprovalBridge()

    result = bridge.evaluate(
        evidence
    )

    assert not result.approved
    assert result.status == "HOLD"


def test_failed_gate_causes_rejection():

    evidence = complete_passing_evidence()

    evidence.backtest = failing_item(
        "backtest"
    )

    bridge = ResearchApprovalBridge()

    result = bridge.evaluate(
        evidence
    )

    assert not result.approved
    assert result.status == "REJECTED"
    assert (
        "backtest"
        in result.failed_gates
    )


def test_failed_validation_gate_causes_rejection():

    evidence = complete_passing_evidence()

    evidence.validation = failing_item(
        "walk_forward_validation"
    )

    bridge = ResearchApprovalBridge()

    result = bridge.evaluate(
        evidence
    )

    assert not result.approved
    assert result.status == "REJECTED"


def test_failed_holdout_gate_causes_rejection():

    evidence = complete_passing_evidence()

    evidence.holdout = failing_item(
        "final_holdout"
    )

    bridge = ResearchApprovalBridge()

    result = bridge.evaluate(
        evidence
    )

    assert not result.approved
    assert result.status == "REJECTED"


def test_failed_leakage_gate_causes_rejection():

    evidence = complete_passing_evidence()

    evidence.leakage = failing_item(
        "leakage_audit"
    )

    bridge = ResearchApprovalBridge()

    result = bridge.evaluate(
        evidence
    )

    assert not result.approved
    assert result.status == "REJECTED"

    assert (
        "leakage_audit"
        in result.failed_gates
    )


def test_warning_on_critical_gate_cannot_approve():

    evidence = complete_passing_evidence()

    evidence.calibration = warning_item(
        "calibration",
        critical=True,
    )

    bridge = ResearchApprovalBridge()

    result = bridge.evaluate(
        evidence
    )

    assert not result.approved
    assert result.status == "HOLD"


def test_not_evaluated_gate_cannot_approve():

    evidence = complete_passing_evidence()

    evidence.range_validation = missing_item(
        "range_validation"
    )

    bridge = ResearchApprovalBridge()

    result = bridge.evaluate(
        evidence
    )

    assert not result.approved
    assert result.status == "HOLD"


# ---------------------------------------------------------------------
# Multiple failures
# ---------------------------------------------------------------------


def test_multiple_failures_are_all_reported():

    evidence = complete_passing_evidence()

    evidence.validation = failing_item(
        "walk_forward_validation"
    )

    evidence.backtest = failing_item(
        "backtest"
    )

    evidence.robustness = failing_item(
        "robustness"
    )

    bridge = ResearchApprovalBridge()

    result = bridge.evaluate(
        evidence
    )

    assert not result.approved

    assert (
        "walk_forward_validation"
        in result.failed_gates
    )

    assert (
        "backtest"
        in result.failed_gates
    )

    assert (
        "robustness"
        in result.failed_gates
    )


def test_failure_takes_precedence_over_missing():

    evidence = complete_passing_evidence()

    evidence.backtest = failing_item(
        "backtest"
    )

    evidence.holdout = None

    bridge = ResearchApprovalBridge()

    result = bridge.evaluate(
        evidence
    )

    assert result.status == "REJECTED"
    assert not result.approved


# ---------------------------------------------------------------------
# Optional gates
# ---------------------------------------------------------------------


def test_disabled_required_gate_is_not_checked():

    config = IntegratedApprovalConfig(
        require_robustness=False
    )

    evidence = complete_passing_evidence()

    evidence.robustness = None

    bridge = ResearchApprovalBridge(
        config=config
    )

    result = bridge.evaluate(
        evidence
    )

    assert result.approved


def test_disabled_multiple_gates_are_not_checked():

    config = IntegratedApprovalConfig(
        require_robustness=False,
        require_range_validation=False,
    )

    evidence = complete_passing_evidence()

    evidence.robustness = None
    evidence.range_validation = None

    bridge = ResearchApprovalBridge(
        config=config
    )

    result = bridge.evaluate(
        evidence
    )

    assert result.approved


# ---------------------------------------------------------------------
# Safety assertion
# ---------------------------------------------------------------------


def test_assert_approved_returns_result_when_valid():

    bridge = ResearchApprovalBridge()

    result = bridge.assert_approved(
        complete_passing_evidence()
    )

    assert result.approved


def test_assert_approved_raises_when_rejected():

    evidence = complete_passing_evidence()

    evidence.backtest = failing_item(
        "backtest"
    )

    bridge = ResearchApprovalBridge()

    with pytest.raises(RuntimeError):

        bridge.assert_approved(
            evidence
        )


def test_assert_approved_raises_when_hold():

    evidence = complete_passing_evidence()

    evidence.holdout = None

    bridge = ResearchApprovalBridge()

    with pytest.raises(RuntimeError):

        bridge.assert_approved(
            evidence
        )


def test_can_proceed_matches_approval():

    bridge = ResearchApprovalBridge()

    passing = complete_passing_evidence()

    assert bridge.can_proceed(
        passing
    )

    passing.backtest = failing_item(
        "backtest"
    )

    assert not bridge.can_proceed(
        passing
    )


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------


def test_summary_contains_core_fields():

    bridge = ResearchApprovalBridge()

    result = bridge.evaluate(
        complete_passing_evidence()
    )

    summary = bridge.summary(
        result
    )

    assert summary["approved"] is True
    assert summary["production_ready"] is True
    assert summary["status"] == "APPROVED"

    assert (
        summary["passed_gate_count"]
        == 8
    )

    assert (
        summary["failed_gate_count"]
        == 0
    )

    assert (
        summary["missing_gate_count"]
        == 0
    )


def test_rejection_summary_contains_failed_gate():

    evidence = complete_passing_evidence()

    evidence.backtest = failing_item(
        "backtest"
    )

    bridge = ResearchApprovalBridge()

    result = bridge.evaluate(
        evidence
    )

    summary = bridge.summary(
        result
    )

    assert summary["approved"] is False
    assert "backtest" in (
        summary["failed_gates"]
    )


# ---------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------


def test_evaluate_convenience_function():

    result = evaluate_research_approval(
        complete_passing_evidence()
    )

    assert result.approved


def test_assert_convenience_function():

    result = assert_research_approved(
        complete_passing_evidence()
    )

    assert result.approved


def test_assert_convenience_function_rejects_invalid_evidence():

    evidence = complete_passing_evidence()

    evidence.leakage = failing_item(
        "leakage_audit"
    )

    with pytest.raises(RuntimeError):

        assert_research_approved(
            evidence
        )


# ---------------------------------------------------------------------
# Evidence score cannot override failures
# ---------------------------------------------------------------------


def test_high_score_cannot_override_failed_gate():

    evidence = complete_passing_evidence()

    evidence.backtest = EvidenceItem(
        name="backtest",
        status=EvidenceStatus.FAIL,
        score=1.0,
        threshold=0.5,
        metrics={
            "profit_factor": 5.0,
            "sharpe": 5.0,
        },
        details="Explicitly failed.",
        critical=True,
        source="test",
    )

    bridge = ResearchApprovalBridge()

    result = bridge.evaluate(
        evidence
    )

    assert not result.approved
    assert result.status == "REJECTED"


# ---------------------------------------------------------------------
# Research-only evidence cannot self-approve
# ---------------------------------------------------------------------


def test_research_only_metadata_does_not_auto_approve():

    evidence = complete_passing_evidence()

    evidence.metadata[
        "production_ready"
    ] = True

    # The bridge must use actual evidence statuses,
    # not a metadata claim.
    evidence.backtest = None

    bridge = ResearchApprovalBridge()

    result = bridge.evaluate(
        evidence
    )

    assert not result.approved
    assert result.status == "HOLD"
