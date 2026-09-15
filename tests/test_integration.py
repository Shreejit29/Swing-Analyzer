"""
Tests for the integrated research evidence framework.

The tests verify:

- evidence item behavior
- evidence collection
- pass/fail/missing detection
- evidence scoring
- production eligibility
- builder helpers
- holdout and leakage gates
- incomplete evidence rejection
- critical failure rejection
- metadata and summary behavior
"""

from __future__ import annotations

import pytest

from src.research.integration import (
    EvidenceItem,
    EvidenceStatus,
    ResearchEvidence,
    ResearchEvidenceBuilder,
    assert_research_ready,
    create_research_evidence,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def complete_builder():
    builder = ResearchEvidenceBuilder(
        model_id="TEST_MODEL",
        symbol="RELIANCE",
        timeframe="1D",
        horizon=5,
    )

    builder.set_validation(
        passed=True,
        accuracy=0.96,
    )

    builder.set_holdout(
        passed=True,
        accuracy=0.95,
    )

    builder.set_calibration(
        passed=True,
        brier_score=0.12,
        ece=0.08,
    )

    builder.set_range_validation(
        passed=True,
        coverage=0.81,
    )

    builder.set_regime(
        passed=True,
        stability_score=0.75,
    )

    builder.set_backtest(
        passed=True,
        profit_factor=1.8,
        sharpe=1.1,
        max_drawdown=-0.18,
        trades=100,
    )

    builder.set_robustness(
        passed=True,
        robustness_score=0.78,
    )

    builder.set_leakage(
        passed=True,
    )

    return builder


@pytest.fixture
def complete_evidence(
    complete_builder,
):
    return complete_builder.build()


# ---------------------------------------------------------------------
# EvidenceStatus
# ---------------------------------------------------------------------


def test_evidence_status_values():
    assert (
        EvidenceStatus.NOT_EVALUATED.value
        == "NOT_EVALUATED"
    )

    assert (
        EvidenceStatus.PASS.value
        == "PASS"
    )

    assert (
        EvidenceStatus.FAIL.value
        == "FAIL"
    )

    assert (
        EvidenceStatus.WARNING.value
        == "WARNING"
    )


# ---------------------------------------------------------------------
# EvidenceItem
# ---------------------------------------------------------------------


def test_evidence_item_defaults():
    item = EvidenceItem(
        name="test"
    )

    assert item.name == "test"
    assert (
        item.status
        == EvidenceStatus.NOT_EVALUATED
    )
    assert item.score is None
    assert item.threshold is None
    assert item.metrics == {}
    assert item.critical is True


def test_evidence_item_passed():
    item = EvidenceItem(
        name="test",
        status=EvidenceStatus.PASS,
    )

    assert item.passed() is True
    assert item.failed() is False
    assert item.evaluated() is True


def test_evidence_item_failed():
    item = EvidenceItem(
        name="test",
        status=EvidenceStatus.FAIL,
    )

    assert item.passed() is False
    assert item.failed() is True
    assert item.evaluated() is True


def test_not_evaluated_item():
    item = EvidenceItem(
        name="test"
    )

    assert item.passed() is False
    assert item.failed() is False
    assert item.evaluated() is False


def test_warning_is_evaluated():
    item = EvidenceItem(
        name="test",
        status=EvidenceStatus.WARNING,
    )

    assert item.evaluated() is True
    assert item.failed() is False
    assert item.passed() is False


# ---------------------------------------------------------------------
# ResearchEvidence construction
# ---------------------------------------------------------------------


def test_research_evidence_defaults():
    evidence = ResearchEvidence()

    assert evidence.model_id is None
    assert evidence.symbol is None
    assert evidence.timeframe is None
    assert evidence.horizon is None

    assert len(
        evidence.items()
    ) == 8


def test_evidence_categories_are_present():
    evidence = ResearchEvidence()

    names = {
        item.name
        for item in evidence.items()
    }

    expected = {
        "walk_forward_validation",
        "final_holdout",
        "probability_calibration",
        "target_range_validation",
        "regime_validation",
        "backtest",
        "robustness",
        "leakage_audit",
    }

    assert names == expected


def test_missing_items_initially_present():
    evidence = ResearchEvidence()

    assert len(
        evidence.missing_items()
    ) == 8

    assert len(
        evidence.critical_missing()
    ) == 8


# ---------------------------------------------------------------------
# Evidence state detection
# ---------------------------------------------------------------------


def test_evaluated_items():
    evidence = ResearchEvidence()

    evidence.validation = EvidenceItem(
        name="walk_forward_validation",
        status=EvidenceStatus.PASS,
    )

    assert len(
        evidence.evaluated_items()
    ) == 1


def test_failed_items():
    evidence = ResearchEvidence()

    evidence.validation = EvidenceItem(
        name="walk_forward_validation",
        status=EvidenceStatus.FAIL,
    )

    failed = evidence.failed_items()

    assert len(failed) == 1
    assert (
        failed[0].name
        == "walk_forward_validation"
    )


def test_critical_failures():
    evidence = ResearchEvidence()

    evidence.validation = EvidenceItem(
        name="walk_forward_validation",
        status=EvidenceStatus.FAIL,
        critical=True,
    )

    assert len(
        evidence.critical_failures()
    ) == 1


def test_noncritical_failure_is_not_critical():
    evidence = ResearchEvidence()

    evidence.validation = EvidenceItem(
        name="walk_forward_validation",
        status=EvidenceStatus.FAIL,
        critical=False,
    )

    assert (
        len(
            evidence.critical_failures()
        )
        == 0
    )


# ---------------------------------------------------------------------
# Required-evidence gates
# ---------------------------------------------------------------------


def test_incomplete_evidence_is_not_ready():
    evidence = ResearchEvidence()

    assert (
        evidence.all_required_evaluated()
        is False
    )

    assert (
        evidence.all_critical_pass()
        is False
    )

    assert (
        evidence.production_eligible()
        is False
    )


def test_one_missing_critical_gate_blocks_readiness():
    evidence = ResearchEvidence()

    for item in evidence.items():
        item.status = EvidenceStatus.PASS

    evidence.holdout.status = (
        EvidenceStatus.NOT_EVALUATED
    )

    assert (
        evidence.production_eligible()
        is False
    )


def test_one_failed_critical_gate_blocks_readiness():
    evidence = ResearchEvidence()

    for item in evidence.items():
        item.status = EvidenceStatus.PASS

    evidence.backtest.status = (
        EvidenceStatus.FAIL
    )

    assert (
        evidence.production_eligible()
        is False
    )


def test_all_critical_gates_pass():
    evidence = ResearchEvidence()

    for item in evidence.items():
        item.status = EvidenceStatus.PASS

    assert (
        evidence.all_required_evaluated()
        is True
    )

    assert (
        evidence.all_critical_pass()
        is True
    )

    assert (
        evidence.production_eligible()
        is True
    )


# ---------------------------------------------------------------------
# Evidence score
# ---------------------------------------------------------------------


def test_empty_evidence_score_is_zero():
    evidence = ResearchEvidence()

    assert (
        evidence.evidence_score()
        == 0.0
    )


def test_all_pass_score_is_one():
    evidence = ResearchEvidence()

    for item in evidence.items():
        item.status = EvidenceStatus.PASS

    assert (
        evidence.evidence_score()
        == 1.0
    )


def test_half_pass_score():
    evidence = ResearchEvidence()

    items = evidence.items()

    for index, item in enumerate(items):
        item.status = (
            EvidenceStatus.PASS
            if index < 4
            else EvidenceStatus.FAIL
        )

    assert (
        evidence.evidence_score()
        == 0.5
    )


def test_missing_items_are_not_counted_in_score():
    evidence = ResearchEvidence()

    evidence.validation.status = (
        EvidenceStatus.PASS
    )

    evidence.holdout.status = (
        EvidenceStatus.FAIL
    )

    score = evidence.evidence_score()

    assert (
        score == 0.5
    )


# ---------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------


def test_builder_metadata():
    builder = ResearchEvidenceBuilder(
        model_id="MODEL_001",
        symbol="TCS",
        timeframe="1D",
        horizon=10,
    )

    evidence = builder.build()

    assert (
        evidence.model_id
        == "MODEL_001"
    )

    assert (
        evidence.symbol
        == "TCS"
    )

    assert (
        evidence.timeframe
        == "1D"
    )

    assert (
        evidence.horizon
        == 10
    )


def test_builder_returns_itself():
    builder = ResearchEvidenceBuilder()

    returned = builder.set_validation(
        passed=True,
        accuracy=0.96,
    )

    assert returned is builder


def test_validation_helper():
    builder = ResearchEvidenceBuilder()

    builder.set_validation(
        passed=True,
        accuracy=0.97,
    )

    item = (
        builder.build().validation
    )

    assert (
        item.status
        == EvidenceStatus.PASS
    )

    assert (
        item.score
        == 0.97
    )

    assert (
        item.threshold
        == 0.95
    )


def test_failed_validation_helper():
    builder = ResearchEvidenceBuilder()

    builder.set_validation(
        passed=False,
        accuracy=0.82,
    )

    item = (
        builder.build().validation
    )

    assert (
        item.status
        == EvidenceStatus.FAIL
    )

    assert (
        item.score
        == 0.82
    )


def test_holdout_helper():
    builder = ResearchEvidenceBuilder()

    builder.set_holdout(
        passed=True,
        accuracy=0.96,
    )

    item = (
        builder.build().holdout
    )

    assert (
        item.status
        == EvidenceStatus.PASS
    )

    assert (
        item.source
        == "final_holdout"
    )


def test_calibration_helper():
    builder = ResearchEvidenceBuilder()

    builder.set_calibration(
        passed=True,
        brier_score=0.10,
        ece=0.05,
    )

    item = (
        builder.build().calibration
    )

    assert (
        item.status
        == EvidenceStatus.PASS
    )

    assert (
        item.metrics[
            "brier_score"
        ]
        == 0.10
    )

    assert (
        item.metrics["ece"]
        == 0.05
    )


def test_range_helper():
    builder = ResearchEvidenceBuilder()

    builder.set_range_validation(
        passed=True,
        coverage=0.82,
    )

    item = (
        builder.build()
        .range_validation
    )

    assert (
        item.status
        == EvidenceStatus.PASS
    )

    assert (
        item.score
        == 0.82
    )


def test_regime_helper():
    builder = ResearchEvidenceBuilder()

    builder.set_regime(
        passed=True,
        stability_score=0.74,
    )

    item = (
        builder.build().regime
    )

    assert (
        item.status
        == EvidenceStatus.PASS
    )

    assert (
        item.score
        == 0.74
    )


def test_backtest_helper():
    builder = ResearchEvidenceBuilder()

    builder.set_backtest(
        passed=True,
        profit_factor=1.8,
        sharpe=1.1,
        max_drawdown=-0.20,
        trades=100,
    )

    item = (
        builder.build().backtest
    )

    assert (
        item.status
        == EvidenceStatus.PASS
    )

    assert (
        item.metrics[
            "profit_factor"
        ]
        == 1.8
    )

    assert (
        item.metrics["sharpe"]
        == 1.1
    )

    assert (
        item.metrics["max_drawdown"]
        == -0.20
    )

    assert (
        item.metrics["trades"]
        == 100
    )


def test_robustness_helper():
    builder = ResearchEvidenceBuilder()

    builder.set_robustness(
        passed=True,
        robustness_score=0.80,
    )

    item = (
        builder.build()
        .robustness
    )

    assert (
        item.status
        == EvidenceStatus.PASS
    )

    assert (
        item.score
        == 0.80
    )


def test_leakage_helper():
    builder = ResearchEvidenceBuilder()

    builder.set_leakage(
        passed=True
    )

    item = (
        builder.build().leakage
    )

    assert (
        item.status
        == EvidenceStatus.PASS
    )

    assert (
        item.source
        == "leakage_audit"
    )


# ---------------------------------------------------------------------
# Generic setter
# ---------------------------------------------------------------------


def test_generic_setter():
    builder = ResearchEvidenceBuilder()

    builder.set_evidence(
        name="backtest",
        status=EvidenceStatus.WARNING,
        score=1.1,
        metrics={
            "profit_factor": 1.1
        },
        critical=False,
    )

    item = (
        builder.build().backtest
    )

    assert (
        item.status
        == EvidenceStatus.WARNING
    )

    assert (
        item.score
        == 1.1
    )

    assert (
        item.critical is False
    )


def test_unknown_evidence_name_fails():
    builder = ResearchEvidenceBuilder()

    with pytest.raises(ValueError):
        builder.set_evidence(
            name="unknown_gate",
            status=EvidenceStatus.PASS,
        )


# ---------------------------------------------------------------------
# Warnings and notes
# ---------------------------------------------------------------------


def test_warning_is_recorded():
    builder = ResearchEvidenceBuilder()

    builder.add_warning(
        "Insufficient trade count."
    )

    evidence = builder.build()

    assert (
        evidence.warnings
        == ["Insufficient trade count."]
    )


def test_note_is_recorded():
    builder = ResearchEvidenceBuilder()

    builder.add_note(
        "Research only."
    )

    evidence = builder.build()

    assert (
        evidence.notes
        == ["Research only."]
    )


def test_empty_warning_is_ignored():
    builder = ResearchEvidenceBuilder()

    builder.add_warning("")

    assert (
        builder.build().warnings
        == []
    )


# ---------------------------------------------------------------------
# Complete evidence
# ---------------------------------------------------------------------


def test_complete_evidence_is_ready(
    complete_evidence,
):
    assert (
        complete_evidence.production_eligible()
        is True
    )


def test_complete_evidence_has_all_evaluated(
    complete_evidence,
):
    assert (
        complete_evidence.all_required_evaluated()
        is True
    )


def test_complete_evidence_has_no_critical_failures(
    complete_evidence,
):
    assert (
        complete_evidence.critical_failures()
        == []
    )


def test_complete_evidence_score_is_one(
    complete_evidence,
):
    assert (
        complete_evidence.evidence_score()
        == 1.0
    )


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------


def test_summary_contains_identity(
    complete_evidence,
):
    summary = (
        complete_evidence.summary()
    )

    assert (
        summary["model_id"]
        == "TEST_MODEL"
    )

    assert (
        summary["symbol"]
        == "RELIANCE"
    )

    assert (
        summary["timeframe"]
        == "1D"
    )

    assert (
        summary["horizon"]
        == 5
    )


def test_summary_contains_gate_state(
    complete_evidence,
):
    summary = (
        complete_evidence.summary()
    )

    assert (
        summary[
            "production_eligible"
        ]
        is True
    )

    assert (
        summary[
            "all_required_evaluated"
        ]
        is True
    )

    assert (
        summary[
            "critical_failures"
        ]
        == []
    )

    assert (
        summary[
            "critical_missing"
        ]
        == []
    )


# ---------------------------------------------------------------------
# Fail-closed assertion
# ---------------------------------------------------------------------


def test_assert_research_ready_accepts_complete(
    complete_evidence,
):
    assert_research_ready(
        complete_evidence
    )


def test_assert_research_ready_rejects_incomplete():
    evidence = ResearchEvidence()

    with pytest.raises(RuntimeError):
        assert_research_ready(
            evidence
        )


def test_assert_research_ready_rejects_failed_gate(
    complete_evidence,
):
    complete_evidence.backtest.status = (
        EvidenceStatus.FAIL
    )

    with pytest.raises(RuntimeError):
        assert_research_ready(
            complete_evidence
        )


def test_assert_research_ready_rejects_wrong_type():
    with pytest.raises(TypeError):
        assert_research_ready(
            None
        )


# ---------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------


def test_create_research_evidence():
    builder = create_research_evidence(
        model_id="MODEL_X",
        symbol="INFY",
        timeframe="1D",
        horizon=3,
    )

    evidence = builder.build()

    assert (
        evidence.model_id
        == "MODEL_X"
    )

    assert (
        evidence.symbol
        == "INFY"
    )

    assert (
        evidence.timeframe
        == "1D"
    )

    assert (
        evidence.horizon
        == 3
    )


# ---------------------------------------------------------------------
# Holdout and leakage are mandatory
# ---------------------------------------------------------------------


def test_holdout_is_critical():
    evidence = ResearchEvidence()

    assert (
        evidence.holdout.critical
        is True
    )


def test_leakage_is_critical():
    evidence = ResearchEvidence()

    assert (
        evidence.leakage.critical
        is True
    )


def test_holdout_failure_blocks_complete_package(
    complete_evidence,
):
    complete_evidence.holdout.status = (
        EvidenceStatus.FAIL
    )

    assert (
        complete_evidence.production_eligible()
        is False
    )


def test_leakage_failure_blocks_complete_package(
    complete_evidence,
):
    complete_evidence.leakage.status = (
        EvidenceStatus.FAIL
    )

    assert (
        complete_evidence.production_eligible()
        is False
    )
