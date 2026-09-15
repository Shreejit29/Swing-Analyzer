"""
Tests for the master research pipeline integration layer.

These tests verify:
- stage lifecycle
- chronological stage ordering
- stage failure handling
- evidence attachment
- final holdout integration
- fail-closed production eligibility
- production approval separation
"""

from __future__ import annotations

import pytest

from src.research.integration import (
    EvidenceItem,
    EvidenceStatus,
    ResearchEvidence,
)
from src.research.pipeline_integration import (
    DEFAULT_INTEGRATION_STAGES,
    FINAL_HOLDOUT_STAGE,
    IntegratedResearchResult,
    IntegrationStage,
    ResearchPipelineIntegrator,
    create_research_integrator,
    integrate_final_holdout,
)


class FakeHoldoutResult:
    """Minimal successful holdout result for integration tests."""

    def __init__(
        self,
        model_id: str = "model_test",
        accuracy: float = 0.97,
        passed: bool = True,
    ):
        self.model_id = model_id
        self.accuracy = accuracy
        self.holdout_accuracy = accuracy
        self.passed_accuracy_gate = passed
        self.completed = True
        self.evaluated = True
        self.final_holdout_used = True
        self.production_approved = False

        self.metadata = {
            "research_only": True,
            "production_approved": False,
        }


def complete_all_stages(
    integrator: ResearchPipelineIntegrator,
) -> None:
    """Complete every configured stage."""

    for stage_name in integrator.result.stages:
        integrator.start_stage(stage_name)
        integrator.complete_stage(
            stage_name,
            score=0.95,
        )


# ---------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------


def test_default_stage_order():
    assert DEFAULT_INTEGRATION_STAGES[-2:] == (
        "final_holdout",
        "leakage_audit",
    )


def test_final_holdout_stage_constant():
    assert FINAL_HOLDOUT_STAGE == "final_holdout"


def test_integrator_construction():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    assert isinstance(
        integrator.result,
        IntegratedResearchResult,
    )

    assert (
        integrator.model_id
        == "model_test"
    )

    assert (
        integrator.result.status
        == "CREATED"
    )

    assert (
        integrator.result.production_eligible
        is False
    )


def test_default_stages_are_created():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    assert tuple(
        integrator.result.stages.keys()
    ) == DEFAULT_INTEGRATION_STAGES


def test_custom_stages():
    stages = (
        "data",
        "features",
        "final_holdout",
    )

    integrator = ResearchPipelineIntegrator(
        model_id="model_test",
        stage_names=stages,
    )

    assert tuple(
        integrator.result.stages.keys()
    ) == stages


def test_empty_model_id_fails():
    with pytest.raises(ValueError):
        ResearchPipelineIntegrator(
            model_id=""
        )


def test_non_string_model_id_fails():
    with pytest.raises(TypeError):
        ResearchPipelineIntegrator(
            model_id=123
        )


def test_empty_stage_list_fails():
    with pytest.raises(ValueError):
        ResearchPipelineIntegrator(
            model_id="model_test",
            stage_names=(),
        )


def test_duplicate_stage_names_fail():
    with pytest.raises(ValueError):
        ResearchPipelineIntegrator(
            model_id="model_test",
            stage_names=(
                "data",
                "data",
            ),
        )


# ---------------------------------------------------------------------
# IntegrationStage
# ---------------------------------------------------------------------


def test_integration_stage_initial_state():
    stage = IntegrationStage(
        name="data"
    )

    assert stage.name == "data"
    assert stage.status == "NOT_STARTED"
    assert stage.score is None
    assert stage.error is None


def test_integration_stage_start():
    stage = IntegrationStage(
        name="data"
    )

    stage.start()

    assert stage.status == "RUNNING"
    assert stage.started_at is not None


def test_integration_stage_complete():
    stage = IntegrationStage(
        name="data"
    )

    stage.start()

    stage.complete(
        score=0.95,
        metrics={
            "rows": 1000,
        },
        details="Data passed.",
    )

    assert stage.status == "COMPLETE"
    assert stage.score == pytest.approx(
        0.95
    )
    assert stage.metrics["rows"] == 1000
    assert stage.details == "Data passed."
    assert stage.completed_at is not None


def test_integration_stage_failure():
    stage = IntegrationStage(
        name="data"
    )

    stage.start()
    stage.fail(
        "Data quality failure."
    )

    assert stage.status == "FAILED"
    assert (
        stage.error
        == "Data quality failure."
    )
    assert stage.completed_at is not None


# ---------------------------------------------------------------------
# Pipeline lifecycle
# ---------------------------------------------------------------------


def test_start_pipeline():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    result = integrator.start()

    assert result.status == "RUNNING"
    assert result.metadata[
        "started_at"
    ] is not None


def test_start_stage():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    stage = integrator.start_stage(
        "data"
    )

    assert stage.status == "RUNNING"


def test_complete_stage():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    integrator.start_stage("data")

    stage = integrator.complete_stage(
        "data",
        score=0.98,
        metrics={
            "rows": 5000,
        },
    )

    assert stage.status == "COMPLETE"
    assert stage.score == pytest.approx(
        0.98
    )


def test_unknown_stage_fails():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    with pytest.raises(KeyError):
        integrator.start_stage(
            "unknown_stage"
        )


def test_failed_stage_sets_pipeline_failed():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    stage = integrator.fail_stage(
        "data",
        "Invalid data.",
    )

    assert stage.status == "FAILED"
    assert (
        integrator.result.status
        == "FAILED"
    )

    assert len(
        integrator.result.errors
    ) == 1


# ---------------------------------------------------------------------
# Stage ordering
# ---------------------------------------------------------------------


def test_first_stage_can_proceed():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    assert (
        integrator.can_proceed("data")
        is True
    )


def test_second_stage_requires_first_stage():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    assert (
        integrator.can_proceed(
            "features"
        )
        is False
    )


def test_second_stage_allowed_after_first():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    integrator.start_stage("data")
    integrator.complete_stage("data")

    assert (
        integrator.can_proceed(
            "features"
        )
        is True
    )


def test_final_holdout_requires_previous_stages():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    assert (
        integrator.can_proceed(
            FINAL_HOLDOUT_STAGE
        )
        is False
    )


def test_final_holdout_allowed_after_previous_stages():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    for stage_name in DEFAULT_INTEGRATION_STAGES:
        if stage_name == FINAL_HOLDOUT_STAGE:
            break

        integrator.start_stage(stage_name)
        integrator.complete_stage(
            stage_name
        )

    assert (
        integrator.can_proceed(
            FINAL_HOLDOUT_STAGE
        )
        is True
    )


def test_failed_previous_stage_blocks_next():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    integrator.fail_stage(
        "data",
        "Failure.",
    )

    assert (
        integrator.can_proceed(
            "features"
        )
        is False
    )


def test_assert_can_proceed_raises_when_blocked():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    with pytest.raises(RuntimeError):
        integrator.assert_can_proceed(
            "features"
        )


# ---------------------------------------------------------------------
# Evidence attachment
# ---------------------------------------------------------------------


def test_attach_evidence():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    evidence = ResearchEvidence(
        model_id="model_test"
    )

    result = integrator.attach_evidence(
        evidence
    )

    assert result.evidence is evidence


def test_attach_wrong_evidence_type_fails():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    with pytest.raises(TypeError):
        integrator.attach_evidence(
            object()
        )


def test_attach_evidence_wrong_model_id_fails_on_holdout():
    integrator = ResearchPipelineIntegrator(
        model_id="model_A"
    )

    evidence = ResearchEvidence(
        model_id="model_B"
    )

    integrator.attach_evidence(
        evidence
    )

    with pytest.raises(ValueError):
        integrator.attach_final_holdout(
            FakeHoldoutResult(
                model_id="model_A"
            )
        )


# ---------------------------------------------------------------------
# Final holdout integration
# ---------------------------------------------------------------------


def test_attach_final_holdout_creates_evidence():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    result = FakeHoldoutResult(
        accuracy=0.97
    )

    item = integrator.attach_final_holdout(
        result
    )

    assert isinstance(
        item,
        EvidenceItem,
    )

    assert (
        item.status
        == EvidenceStatus.PASS
    )

    assert (
        integrator.result.evidence
        is not None
    )


def test_final_holdout_pass_updates_stage():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    integrator.attach_final_holdout(
        FakeHoldoutResult(
            accuracy=0.97
        )
    )

    stage = integrator.result.stages[
        FINAL_HOLDOUT_STAGE
    ]

    assert stage.status == "COMPLETE"
    assert stage.score == pytest.approx(
        0.97
    )


def test_final_holdout_failure_updates_stage():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    integrator.attach_final_holdout(
        FakeHoldoutResult(
            accuracy=0.80,
            passed=False,
        )
    )

    stage = integrator.result.stages[
        FINAL_HOLDOUT_STAGE
    ]

    assert stage.status == "FAILED"
    assert (
        integrator.result.status
        == "FAILED"
    )


def test_missing_final_holdout_is_not_evaluated():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    item = integrator.attach_final_holdout(
        None
    )

    assert (
        item.status
        == EvidenceStatus.NOT_EVALUATED
    )

    stage = integrator.result.stages[
        FINAL_HOLDOUT_STAGE
    ]

    assert (
        stage.status
        == "NOT_EVALUATED"
    )


def test_final_holdout_exact_95_percent_passes():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    item = integrator.attach_final_holdout(
        FakeHoldoutResult(
            accuracy=0.95
        )
    )

    assert (
        item.status
        == EvidenceStatus.PASS
    )


def test_final_holdout_below_95_fails():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    item = integrator.attach_final_holdout(
        FakeHoldoutResult(
            accuracy=0.949
            ,
            passed=False,
        )
    )

    assert (
        item.status
        == EvidenceStatus.FAIL
    )


def test_final_holdout_cannot_grant_approval():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    integrator.attach_final_holdout(
        FakeHoldoutResult(
            accuracy=1.0
        )
    )

    assert (
        integrator.result.metadata[
            "production_approved"
        ]
        is False
    )

    assert (
        integrator.result.production_eligible
        is False
    )


# ---------------------------------------------------------------------
# Production eligibility
# ---------------------------------------------------------------------


def test_no_evidence_is_not_eligible():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    assert (
        integrator.evaluate_production_eligibility()
        is False
    )


def test_failed_stage_blocks_eligibility():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    integrator.fail_stage(
        "data",
        "Failure.",
    )

    assert (
        integrator.evaluate_production_eligibility()
        is False
    )


def test_incomplete_stages_block_eligibility():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    integrator.attach_final_holdout(
        FakeHoldoutResult()
    )

    assert (
        integrator.evaluate_production_eligibility()
        is False
    )


def test_complete_pipeline_without_evidence_is_blocked():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    complete_all_stages(
        integrator
    )

    result = integrator.complete()

    assert result.status == "BLOCKED"
    assert (
        result.production_eligible
        is False
    )


# ---------------------------------------------------------------------
# Completion behavior
# ---------------------------------------------------------------------


def test_complete_with_failed_stage():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    integrator.fail_stage(
        "data",
        "Data failure.",
    )

    result = integrator.complete()

    assert result.status == "FAILED"
    assert (
        result.production_eligible
        is False
    )


def test_complete_with_incomplete_stages():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    result = integrator.complete()

    assert result.status == "BLOCKED"
    assert (
        result.production_eligible
        is False
    )


def test_successful_lifecycle_requires_all_evidence():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    complete_all_stages(
        integrator
    )

    # Evidence itself is still empty, so the
    # integration must remain blocked.
    result = integrator.complete()

    assert result.status == "BLOCKED"
    assert (
        result.production_eligible
        is False
    )


# ---------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------


def test_failed_stages():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    integrator.fail_stage(
        "data",
        "Failure.",
    )

    failed = (
        integrator.result.failed_stages()
    )

    assert len(failed) == 1
    assert failed[0].name == "data"


def test_incomplete_stages_initially_all():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    incomplete = (
        integrator.result.incomplete_stages()
    )

    assert len(incomplete) == len(
        DEFAULT_INTEGRATION_STAGES
    )


def test_summary():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    summary = (
        integrator.result.summary()
    )

    assert summary["model_id"] == (
        "model_test"
    )

    assert (
        summary["stage_count"]
        == len(DEFAULT_INTEGRATION_STAGES)
    )

    assert (
        summary["production_eligible"]
        is False
    )


# ---------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------


def test_create_research_integrator():
    integrator = (
        create_research_integrator(
            "model_test"
        )
    )

    assert isinstance(
        integrator,
        ResearchPipelineIntegrator,
    )

    assert (
        integrator.model_id
        == "model_test"
    )


def test_integrate_final_holdout():
    result = integrate_final_holdout(
        model_id="model_test",
        holdout_result=FakeHoldoutResult(
            accuracy=0.97
        ),
    )

    assert isinstance(
        result,
        IntegratedResearchResult,
    )

    assert (
        result.evidence
        is not None
    )

    assert (
        result.stages[
            FINAL_HOLDOUT_STAGE
        ].status
        == "COMPLETE"
    )


def test_integrate_final_holdout_missing_result():
    result = integrate_final_holdout(
        model_id="model_test",
        holdout_result=None,
    )

    assert (
        result.stages[
            FINAL_HOLDOUT_STAGE
        ].status
        == "NOT_EVALUATED"
    )

    assert (
        result.production_eligible
        is False
    )


# ---------------------------------------------------------------------
# Research-only boundary
# ---------------------------------------------------------------------


def test_integration_metadata_is_research_only():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    assert (
        integrator.result.metadata[
            "research_only"
        ]
        is True
    )

    assert (
        integrator.result.metadata[
            "production_approved"
        ]
        is False
    )


def test_completion_never_grants_production_approval():
    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    complete_all_stages(
        integrator
    )

    result = integrator.complete()

    assert (
        result.metadata[
            "production_approved"
        ]
        is False
    )


def test_holdout_result_approval_flag_is_not_changed():
    holdout = FakeHoldoutResult()

    integrator = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    integrator.attach_final_holdout(
        holdout
    )

    assert (
        holdout.production_approved
        is False
    )


# ---------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------


def test_stage_order_is_deterministic():
    first = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    second = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    assert tuple(
        first.result.stages.keys()
    ) == tuple(
        second.result.stages.keys()
    )


def test_final_holdout_attachment_is_deterministic():
    first = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    second = ResearchPipelineIntegrator(
        model_id="model_test"
    )

    holdout_1 = FakeHoldoutResult(
        accuracy=0.97
    )

    holdout_2 = FakeHoldoutResult(
        accuracy=0.97
    )

    item_1 = first.attach_final_holdout(
        holdout_1
    )

    item_2 = second.attach_final_holdout(
        holdout_2
    )

    assert (
        item_1.status
        == item_2.status
    )

    assert (
        item_1.score
        == item_2.score
    )
