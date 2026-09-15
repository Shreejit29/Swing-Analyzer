"""
Tests for the centralized research evidence collector.

No model training or live market-data access is required.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.research.evidence_collector import (
    BACKTEST_STAGE,
    CALIBRATION_STAGE,
    FINAL_HOLDOUT_STAGE,
    LEAKAGE_STAGE,
    RANGE_STAGE,
    REGIME_STAGE,
    REQUIRED_EVIDENCE_STAGES,
    ROBUSTNESS_STAGE,
    WALK_FORWARD_STAGE,
    EvidenceCollectionResult,
    ResearchEvidenceCollector,
    attach_final_holdout_to_evidence,
    build_final_holdout_evidence,
    collect_final_holdout_evidence,
    collect_holdout_evidence_map,
    collect_research_evidence,
    final_holdout_evaluated,
    final_holdout_passed,
    final_holdout_status,
    final_holdout_summary,
    validate_final_holdout_identity,
)
from src.research.integration import (
    EvidenceItem,
    EvidenceStatus,
    ResearchEvidence,
)


# ---------------------------------------------------------------------
# Fake stage results
# ---------------------------------------------------------------------


def passing_result(
    model_id: str = "model_test",
    **kwargs,
):
    values = {
        "model_id": model_id,
        "passed": True,
        "accuracy": 0.97,
        "brier_score": 0.10,
        "ece": 0.08,
        "coverage": 0.82,
        "stability_score": 0.80,
        "profit_factor": 1.50,
        "sharpe": 1.10,
        "max_drawdown": 0.15,
        "robustness_score": 0.80,
        "leakage_free": True,
        "final_holdout_used": False,
    }

    values.update(kwargs)

    return SimpleNamespace(
        **values
    )


def failing_result(
    model_id: str = "model_test",
    **kwargs,
):
    values = {
        "model_id": model_id,
        "passed": False,
        "accuracy": 0.70,
        "brier_score": 0.40,
        "ece": 0.30,
        "coverage": 0.50,
        "stability_score": 0.30,
        "profit_factor": 0.80,
        "sharpe": 0.10,
        "max_drawdown": 0.40,
        "robustness_score": 0.20,
        "leakage_free": False,
        "final_holdout_used": False,
    }

    values.update(kwargs)

    return SimpleNamespace(
        **values
    )


def holdout_result(
    model_id: str = "model_test",
    accuracy: float = 0.97,
    evaluated: bool = True,
    completed: bool = True,
    **kwargs,
):
    values = {
        "model_id": model_id,
        "accuracy": accuracy,
        "evaluated": evaluated,
        "completed": completed,
        "final_holdout_used": evaluated,
    }

    values.update(kwargs)

    return SimpleNamespace(
        **values
    )


def complete_stage_map(
    model_id: str = "model_test",
):
    return {
        WALK_FORWARD_STAGE: passing_result(
            model_id
        ),
        FINAL_HOLDOUT_STAGE: holdout_result(
            model_id
        ),
        CALIBRATION_STAGE: passing_result(
            model_id
        ),
        RANGE_STAGE: passing_result(
            model_id
        ),
        REGIME_STAGE: passing_result(
            model_id
        ),
        BACKTEST_STAGE: passing_result(
            model_id
        ),
        ROBUSTNESS_STAGE: passing_result(
            model_id
        ),
        LEAKAGE_STAGE: passing_result(
            model_id,
            leakage_free=True,
        ),
    }


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------


def test_required_evidence_stages_are_defined():
    assert (
        WALK_FORWARD_STAGE
        in REQUIRED_EVIDENCE_STAGES
    )

    assert (
        FINAL_HOLDOUT_STAGE
        in REQUIRED_EVIDENCE_STAGES
    )

    assert (
        CALIBRATION_STAGE
        in REQUIRED_EVIDENCE_STAGES
    )

    assert (
        RANGE_STAGE
        in REQUIRED_EVIDENCE_STAGES
    )

    assert (
        REGIME_STAGE
        in REQUIRED_EVIDENCE_STAGES
    )

    assert (
        BACKTEST_STAGE
        in REQUIRED_EVIDENCE_STAGES
    )

    assert (
        ROBUSTNESS_STAGE
        in REQUIRED_EVIDENCE_STAGES
    )

    assert (
        LEAKAGE_STAGE
        in REQUIRED_EVIDENCE_STAGES
    )


# ---------------------------------------------------------------------
# Collector construction
# ---------------------------------------------------------------------


def test_collector_construction():
    collector = ResearchEvidenceCollector()

    assert tuple(
        collector.required_stages
    ) == tuple(
        REQUIRED_EVIDENCE_STAGES
    )


def test_custom_required_stages():
    collector = ResearchEvidenceCollector(
        required_stages=(
            WALK_FORWARD_STAGE,
            FINAL_HOLDOUT_STAGE,
        )
    )

    assert collector.required_stages == (
        WALK_FORWARD_STAGE,
        FINAL_HOLDOUT_STAGE,
    )


def test_stage_aliases_are_normalized():
    collector = ResearchEvidenceCollector(
        required_stages=(
            "walk_forward",
            "holdout",
            "calibration",
        )
    )

    assert collector.required_stages == (
        WALK_FORWARD_STAGE,
        FINAL_HOLDOUT_STAGE,
        CALIBRATION_STAGE,
    )


# ---------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------


def test_model_id_is_required():
    collector = ResearchEvidenceCollector()

    with pytest.raises(ValueError):
        collector.collect(
            model_id=""
        )


def test_model_id_must_be_string():
    collector = ResearchEvidenceCollector()

    with pytest.raises(ValueError):
        collector.collect(
            model_id=123
        )


# ---------------------------------------------------------------------
# Missing evidence
# ---------------------------------------------------------------------


def test_missing_evidence_is_not_evaluated():
    collector = ResearchEvidenceCollector(
        required_stages=(
            WALK_FORWARD_STAGE,
        )
    )

    result = collector.collect(
        model_id="model_test",
        stages={},
    )

    assert isinstance(
        result,
        EvidenceCollectionResult,
    )

    assert (
        WALK_FORWARD_STAGE
        in result.missing_stages
    )

    assert not result.complete

    assert not result.production_eligible


def test_missing_evidence_does_not_pass():
    collector = ResearchEvidenceCollector(
        required_stages=(
            WALK_FORWARD_STAGE,
        )
    )

    result = collector.collect(
        model_id="model_test",
        stages={},
    )

    assert (
        len(result.evidence.failed_items())
        == 0
    )

    assert (
        len(result.evidence.missing_items())
        >= 1
    )


def test_empty_stage_mapping_is_safe():
    collector = ResearchEvidenceCollector(
        required_stages=(
            WALK_FORWARD_STAGE,
            FINAL_HOLDOUT_STAGE,
        )
    )

    result = collector.collect(
        model_id="model_test",
        stages=None,
    )

    assert (
        len(result.missing_stages)
        == 2
    )

    assert not result.production_eligible


# ---------------------------------------------------------------------
# Complete collection
# ---------------------------------------------------------------------


def test_complete_collection():
    collector = ResearchEvidenceCollector()

    result = collector.collect(
        model_id="model_test",
        stages=complete_stage_map(),
    )

    assert result.complete

    assert not result.missing_stages

    assert (
        len(result.collected_stages)
        == len(REQUIRED_EVIDENCE_STAGES)
    )


def test_complete_collection_preserves_model_identity():
    collector = ResearchEvidenceCollector()

    result = collector.collect(
        model_id="model_test",
        stages=complete_stage_map(
            "model_test"
        ),
    )

    assert (
        result.evidence.model_id
        == "model_test"
    )


def test_complete_collection_is_research_only():
    collector = ResearchEvidenceCollector()

    result = collector.collect(
        model_id="model_test",
        stages=complete_stage_map(),
    )

    assert (
        result.metadata[
            "research_only"
        ]
        is True
    )

    assert (
        result.metadata[
            "production_approved"
        ]
        is False
    )


# ---------------------------------------------------------------------
# Failed evidence
# ---------------------------------------------------------------------


def test_failed_stage_is_recorded():
    collector = ResearchEvidenceCollector(
        required_stages=(
            WALK_FORWARD_STAGE,
        )
    )

    result = collector.collect(
        model_id="model_test",
        stages={
            WALK_FORWARD_STAGE:
                failing_result()
        },
    )

    assert (
        WALK_FORWARD_STAGE
        in result.failed_stages
    )

    assert not result.production_eligible


def test_failed_evidence_is_not_converted_to_pass():
    collector = ResearchEvidenceCollector(
        required_stages=(
            WALK_FORWARD_STAGE,
        )
    )

    result = collector.collect(
        model_id="model_test",
        stages={
            WALK_FORWARD_STAGE:
                failing_result()
        },
    )

    item = result.evidence.validation

    assert (
        item.status
        == EvidenceStatus.FAIL
    )


# ---------------------------------------------------------------------
# Adapter failure handling
# ---------------------------------------------------------------------


def test_invalid_stage_result_fails_closed():
    collector = ResearchEvidenceCollector(
        required_stages=(
            WALK_FORWARD_STAGE,
        )
    )

    result = collector.collect(
        model_id="model_test",
        stages={
            WALK_FORWARD_STAGE:
                object()
        },
    )

    assert (
        WALK_FORWARD_STAGE
        in result.failed_stages
    )

    assert len(
        result.errors
    ) >= 1

    assert not result.production_eligible


# ---------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------


def test_collect_research_evidence():
    result = collect_research_evidence(
        model_id="model_test",
        stages=complete_stage_map(),
    )

    assert isinstance(
        result,
        EvidenceCollectionResult,
    )

    assert result.complete


# ---------------------------------------------------------------------
# Final holdout evidence
# ---------------------------------------------------------------------


def test_none_holdout_is_not_evaluated():
    item = collect_final_holdout_evidence(
        None
    )

    assert isinstance(
        item,
        EvidenceItem,
    )

    assert (
        item.status
        == EvidenceStatus.NOT_EVALUATED
    )

    assert not final_holdout_evaluated(
        None
    )

    assert not final_holdout_passed(
        None
    )


def test_passing_holdout():
    result = holdout_result(
        accuracy=0.97
    )

    item = collect_final_holdout_evidence(
        result
    )

    assert (
        item.status
        == EvidenceStatus.PASS
    )

    assert final_holdout_evaluated(
        result
    )

    assert final_holdout_passed(
        result
    )


def test_failing_holdout():
    result = holdout_result(
        accuracy=0.80
    )

    item = collect_final_holdout_evidence(
        result
    )

    assert (
        item.status
        == EvidenceStatus.FAIL
    )

    assert final_holdout_evaluated(
        result
    )

    assert not final_holdout_passed(
        result
    )


def test_incomplete_holdout_fails():
    result = holdout_result(
        accuracy=0.99,
        evaluated=True,
        completed=False,
    )

    item = collect_final_holdout_evidence(
        result
    )

    assert (
        item.status
        != EvidenceStatus.PASS
    )


def test_unevaluated_holdout_is_not_evaluated():
    result = holdout_result(
        accuracy=0.99,
        evaluated=False,
        completed=False,
    )

    item = collect_final_holdout_evidence(
        result
    )

    assert (
        item.status
        == EvidenceStatus.NOT_EVALUATED
    )


def test_holdout_threshold_is_95_percent():
    passing = collect_final_holdout_evidence(
        holdout_result(
            accuracy=0.95
        )
    )

    failing = collect_final_holdout_evidence(
        holdout_result(
            accuracy=0.9499
        )
    )

    assert (
        passing.status
        == EvidenceStatus.PASS
    )

    assert (
        failing.status
        == EvidenceStatus.FAIL
    )


def test_holdout_summary():
    result = holdout_result(
        accuracy=0.97
    )

    summary = final_holdout_summary(
        result
    )

    assert (
        summary["name"]
        == FINAL_HOLDOUT_STAGE
    )

    assert (
        summary["status"]
        == EvidenceStatus.PASS.value
    )

    assert (
        summary["score"]
        == pytest.approx(0.97)
    )

    assert (
        summary["threshold"]
        == pytest.approx(0.95)
    )


def test_holdout_status():
    result = holdout_result(
        accuracy=0.97
    )

    assert (
        final_holdout_status(
            result
        )
        == EvidenceStatus.PASS
    )


# ---------------------------------------------------------------------
# Final holdout identity
# ---------------------------------------------------------------------


def test_holdout_identity_matches():
    result = holdout_result(
        model_id="model_test"
    )

    assert validate_final_holdout_identity(
        result,
        "model_test",
    )


def test_holdout_identity_mismatch():
    result = holdout_result(
        model_id="other_model"
    )

    assert not validate_final_holdout_identity(
        result,
        "model_test",
    )


def test_missing_holdout_identity_is_invalid():
    result = SimpleNamespace(
        accuracy=0.99,
        evaluated=True,
        completed=True,
    )

    assert not validate_final_holdout_identity(
        result,
        "model_test",
    )


def test_none_holdout_identity_is_invalid():
    assert not validate_final_holdout_identity(
        None,
        "model_test",
    )


# ---------------------------------------------------------------------
# Final holdout evidence construction
# ---------------------------------------------------------------------


def test_build_final_holdout_evidence():
    result = holdout_result(
        model_id="model_test",
        accuracy=0.97,
    )

    evidence = build_final_holdout_evidence(
        "model_test",
        result,
    )

    assert isinstance(
        evidence,
        ResearchEvidence,
    )

    assert (
        evidence.model_id
        == "model_test"
    )


def test_build_final_holdout_evidence_requires_model_id():
    with pytest.raises(ValueError):
        build_final_holdout_evidence(
            "",
            holdout_result(),
        )


def test_attach_final_holdout_to_evidence():
    evidence = ResearchEvidence(
        model_id="model_test"
    )

    result = holdout_result(
        model_id="model_test",
        accuracy=0.97,
    )

    returned = attach_final_holdout_to_evidence(
        evidence,
        result,
    )

    assert returned is evidence

    assert (
        evidence.final_holdout.status
        == EvidenceStatus.PASS
    )


def test_attach_none_holdout_is_not_evaluated():
    evidence = ResearchEvidence(
        model_id="model_test"
    )

    attach_final_holdout_to_evidence(
        evidence,
        None,
    )

    assert (
        evidence.final_holdout.status
        == EvidenceStatus.NOT_EVALUATED
    )


# ---------------------------------------------------------------------
# Multiple holdout results
# ---------------------------------------------------------------------


def test_collect_holdout_evidence_map():
    results = {
        "model_a": holdout_result(
            model_id="model_a",
            accuracy=0.97,
        ),
        "model_b": holdout_result(
            model_id="model_b",
            accuracy=0.80,
        ),
    }

    evidence_map = (
        collect_holdout_evidence_map(
            results
        )
    )

    assert set(
        evidence_map.keys()
    ) == {
        "model_a",
        "model_b",
    }

    assert (
        evidence_map[
            "model_a"
        ].status
        == EvidenceStatus.PASS
    )

    assert (
        evidence_map[
            "model_b"
        ].status
        == EvidenceStatus.FAIL
    )


def test_collect_holdout_evidence_map_identity_mismatch():
    results = {
        "model_a": holdout_result(
            model_id="model_b"
        )
    }

    with pytest.raises(ValueError):
        collect_holdout_evidence_map(
            results
        )


def test_collect_holdout_evidence_map_none():
    results = {
        "model_a": None
    }

    evidence_map = (
        collect_holdout_evidence_map(
            results
        )
    )

    assert (
        evidence_map[
            "model_a"
        ].status
        == EvidenceStatus.NOT_EVALUATED
    )


def test_collect_holdout_evidence_map_requires_mapping():
    with pytest.raises(TypeError):
        collect_holdout_evidence_map(
            []
        )


# ---------------------------------------------------------------------
# Collector + holdout integration
# ---------------------------------------------------------------------


def test_collector_includes_final_holdout():
    collector = ResearchEvidenceCollector()

    stages = complete_stage_map()

    result = collector.collect(
        model_id="model_test",
        stages=stages,
    )

    assert (
        FINAL_HOLDOUT_STAGE
        in result.collected_stages
    )

    assert (
        result.evidence.final_holdout.status
        == EvidenceStatus.PASS
    )


def test_missing_final_holdout_blocks_complete_collection():
    collector = ResearchEvidenceCollector()

    stages = complete_stage_map()

    del stages[
        FINAL_HOLDOUT_STAGE
    ]

    result = collector.collect(
        model_id="model_test",
        stages=stages,
    )

    assert (
        FINAL_HOLDOUT_STAGE
        in result.missing_stages
    )

    assert not result.complete

    assert not result.production_eligible


# ---------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------


def test_collection_is_deterministic():
    stages_1 = complete_stage_map()
    stages_2 = complete_stage_map()

    collector_1 = ResearchEvidenceCollector()
    collector_2 = ResearchEvidenceCollector()

    result_1 = collector_1.collect(
        model_id="model_test",
        stages=stages_1,
    )

    result_2 = collector_2.collect(
        model_id="model_test",
        stages=stages_2,
    )

    assert (
        result_1.collected_stages
        == result_2.collected_stages
    )

    assert (
        result_1.missing_stages
        == result_2.missing_stages
    )

    assert (
        result_1.failed_stages
        == result_2.failed_stages
    )


# ---------------------------------------------------------------------
# Research-only boundary
# ---------------------------------------------------------------------


def test_collector_never_auto_approves():
    collector = ResearchEvidenceCollector()

    result = collector.collect(
        model_id="model_test",
        stages=complete_stage_map(),
    )

    # Evidence collection itself must not be treated
    # as the production approval operation.
    assert (
        result.metadata[
            "production_approved"
        ]
        is False
    )


def test_final_holdout_does_not_auto_approve():
    evidence = build_final_holdout_evidence(
        "model_test",
        holdout_result(
            model_id="model_test",
            accuracy=1.0,
        ),
    )

    assert (
        evidence.model_id
        == "model_test"
    )

    assert (
        evidence.final_holdout.status
        == EvidenceStatus.PASS
    )

    # A PASS evidence item is not itself a production approval.
    assert not getattr(
        evidence,
        "production_approved",
        False,
    )
