"""
Tests for the final holdout protection gate.
"""

from __future__ import annotations

import pytest

from src.research.holdout_gate import (
    FinalHoldoutGate,
    HoldoutGateInput,
    HoldoutGateResult,
    HoldoutGateStatus,
    create_holdout_gate,
    evaluate_holdout_gate,
)


def valid_input(**overrides) -> HoldoutGateInput:
    values = {
        "model_id": "model_gb_5d",
        "selection_completed": True,
        "walk_forward_completed": True,
        "feature_selection_frozen": True,
        "hyperparameters_frozen": True,
        "preprocessing_frozen": True,
        "calibration_fitted": False,
        "threshold_optimization_completed": False,
        "final_holdout_used_previously": False,
    }

    values.update(overrides)

    return HoldoutGateInput(**values)


def test_gate_construction():
    gate = FinalHoldoutGate()

    assert gate is not None


def test_input_construction():
    evidence = valid_input()

    assert evidence.model_id == "model_gb_5d"
    assert evidence.selection_completed is True
    assert evidence.walk_forward_completed is True


def test_valid_evidence_is_ready():
    gate = FinalHoldoutGate()

    result = gate.evaluate(valid_input())

    assert isinstance(result, HoldoutGateResult)
    assert result.status == HoldoutGateStatus.READY
    assert result.eligible is True
    assert result.safe_to_evaluate is True


def test_valid_evidence_has_no_failures():
    result = evaluate_holdout_gate(valid_input())

    assert result.failures == ()


def test_valid_evidence_has_warning():
    result = evaluate_holdout_gate(valid_input())

    assert result.warnings
    assert any(
        "holdout" in warning.lower()
        for warning in result.warnings
    )


def test_empty_model_id_fails():
    gate = FinalHoldoutGate()

    evidence = valid_input(model_id="")

    with pytest.raises(ValueError, match="model_id"):
        gate.evaluate(evidence)


def test_missing_selection_blocks_gate():
    result = evaluate_holdout_gate(
        valid_input(
            selection_completed=False,
        )
    )

    assert result.status == HoldoutGateStatus.BLOCKED
    assert result.eligible is False
    assert any(
        "selection" in failure.lower()
        for failure in result.failures
    )


def test_missing_walk_forward_blocks_gate():
    result = evaluate_holdout_gate(
        valid_input(
            walk_forward_completed=False,
        )
    )

    assert result.status == HoldoutGateStatus.BLOCKED
    assert result.eligible is False
    assert any(
        "walk-forward" in failure.lower()
        for failure in result.failures
    )


def test_unfrozen_feature_selection_blocks_gate():
    result = evaluate_holdout_gate(
        valid_input(
            feature_selection_frozen=False,
        )
    )

    assert result.status == HoldoutGateStatus.BLOCKED
    assert result.eligible is False
    assert any(
        "feature selection" in failure.lower()
        for failure in result.failures
    )


def test_unfrozen_hyperparameters_block_gate():
    result = evaluate_holdout_gate(
        valid_input(
            hyperparameters_frozen=False,
        )
    )

    assert result.status == HoldoutGateStatus.BLOCKED
    assert result.eligible is False
    assert any(
        "hyperparameter" in failure.lower()
        for failure in result.failures
    )


def test_unfrozen_preprocessing_blocks_gate():
    result = evaluate_holdout_gate(
        valid_input(
            preprocessing_frozen=False,
        )
    )

    assert result.status == HoldoutGateStatus.BLOCKED
    assert result.eligible is False
    assert any(
        "preprocessing" in failure.lower()
        for failure in result.failures
    )


def test_calibration_before_holdout_blocks_gate():
    result = evaluate_holdout_gate(
        valid_input(
            calibration_fitted=True,
        )
    )

    assert result.status == HoldoutGateStatus.BLOCKED
    assert result.eligible is False
    assert any(
        "calibration" in failure.lower()
        for failure in result.failures
    )


def test_threshold_optimization_before_holdout_blocks_gate():
    result = evaluate_holdout_gate(
        valid_input(
            threshold_optimization_completed=True,
        )
    )

    assert result.status == HoldoutGateStatus.BLOCKED
    assert result.eligible is False
    assert any(
        "threshold" in failure.lower()
        for failure in result.failures
    )


def test_previous_holdout_usage_blocks_gate():
    result = evaluate_holdout_gate(
        valid_input(
            final_holdout_used_previously=True,
        )
    )

    assert result.status == HoldoutGateStatus.BLOCKED
    assert result.eligible is False
    assert any(
        "already been used" in failure.lower()
        for failure in result.failures
    )


def test_multiple_failures_are_reported():
    result = evaluate_holdout_gate(
        valid_input(
            selection_completed=False,
            walk_forward_completed=False,
            feature_selection_frozen=False,
            hyperparameters_frozen=False,
            preprocessing_frozen=False,
            calibration_fitted=True,
            threshold_optimization_completed=True,
            final_holdout_used_previously=True,
        )
    )

    assert result.status == HoldoutGateStatus.BLOCKED
    assert result.eligible is False
    assert len(result.failures) >= 7


def test_assert_ready_passes_for_valid_input():
    gate = FinalHoldoutGate()

    result = gate.assert_ready(
        valid_input()
    )

    assert result.eligible is True
    assert result.status == HoldoutGateStatus.READY


def test_assert_ready_raises_for_invalid_input():
    gate = FinalHoldoutGate()

    with pytest.raises(
        RuntimeError,
        match="holdout gate blocked",
    ):
        gate.assert_ready(
            valid_input(
                selection_completed=False,
            )
        )


def test_gate_requires_correct_input_type():
    gate = FinalHoldoutGate()

    with pytest.raises(TypeError):
        gate.evaluate("invalid")


def test_safe_to_evaluate_matches_eligibility():
    ready = evaluate_holdout_gate(
        valid_input()
    )

    blocked = evaluate_holdout_gate(
        valid_input(
            walk_forward_completed=False,
        )
    )

    assert ready.safe_to_evaluate is True
    assert blocked.safe_to_evaluate is False


def test_metadata_marks_holdout_as_protected():
    result = evaluate_holdout_gate(
        valid_input()
    )

    assert result.metadata["final_holdout_protected"] is True
    assert result.metadata["holdout_used_for_selection"] is False
    assert result.metadata[
        "holdout_used_for_feature_selection"
    ] is False
    assert result.metadata[
        "holdout_used_for_hyperparameter_tuning"
    ] is False
    assert result.metadata[
        "holdout_used_for_calibration"
    ] is False


def test_metadata_marks_stage_research_only():
    result = evaluate_holdout_gate(
        valid_input()
    )

    assert result.metadata["research_only"] is True


def test_custom_metadata_is_preserved():
    result = evaluate_holdout_gate(
        valid_input(
            metadata={
                "experiment_id": "EXP-001",
                "research_run": "RUN-42",
            }
        )
    )

    assert result.metadata["experiment_id"] == "EXP-001"
    assert result.metadata["research_run"] == "RUN-42"


def test_summary_for_ready_gate():
    result = evaluate_holdout_gate(
        valid_input()
    )

    summary = result.summary()

    assert summary["status"] == "READY"
    assert summary["eligible"] is True
    assert summary["safe_to_evaluate"] is True
    assert summary["selection_completed"] is True
    assert summary["walk_forward_completed"] is True
    assert summary["final_holdout_used_previously"] is False


def test_summary_for_blocked_gate():
    result = evaluate_holdout_gate(
        valid_input(
            preprocessing_frozen=False,
        )
    )

    summary = result.summary()

    assert summary["status"] == "BLOCKED"
    assert summary["eligible"] is False
    assert summary["safe_to_evaluate"] is False
    assert summary["failure_count"] >= 1


def test_convenience_constructor():
    evidence = create_holdout_gate(
        model_id="model_rf_10d",
        selection_completed=True,
        walk_forward_completed=True,
        feature_selection_frozen=True,
        hyperparameters_frozen=True,
        preprocessing_frozen=True,
    )

    assert isinstance(evidence, HoldoutGateInput)
    assert evidence.model_id == "model_rf_10d"
    assert evidence.final_holdout_used_previously is False


def test_convenience_constructor_preserves_metadata():
    evidence = create_holdout_gate(
        model_id="model_rf_10d",
        selection_completed=True,
        walk_forward_completed=True,
        feature_selection_frozen=True,
        hyperparameters_frozen=True,
        preprocessing_frozen=True,
        metadata={"source": "research_pipeline"},
    )

    assert evidence.metadata["source"] == "research_pipeline"


def test_holdout_gate_is_deterministic():
    evidence = valid_input()

    gate = FinalHoldoutGate()

    result_1 = gate.evaluate(evidence)
    result_2 = gate.evaluate(evidence)

    assert result_1 == result_2


def test_gate_does_not_modify_input_metadata():
    metadata = {
        "experiment_id": "EXP-001",
    }

    evidence = valid_input(
        metadata=metadata
    )

    evaluate_holdout_gate(evidence)

    assert evidence.metadata == {
        "experiment_id": "EXP-001",
    }


def test_final_holdout_usage_has_priority_over_other_status():
    result = evaluate_holdout_gate(
        valid_input(
            final_holdout_used_previously=True,
        )
    )

    assert result.eligible is False
    assert result.status == HoldoutGateStatus.BLOCKED


def test_holdout_gate_never_returns_approved_status():
    statuses = {
        HoldoutGateStatus.LOCKED,
        HoldoutGateStatus.READY,
        HoldoutGateStatus.EVALUATED,
        HoldoutGateStatus.BLOCKED,
    }

    result = evaluate_holdout_gate(
        valid_input()
    )

    assert result.status in statuses
    assert result.status != HoldoutGateStatus.EVALUATED


def test_gate_does_not_evaluate_holdout_metrics():
    result = evaluate_holdout_gate(
        valid_input()
    )

    summary = result.summary()

    assert "accuracy" not in summary
    assert "brier" not in summary
    assert "sharpe" not in summary


def test_gate_does_not_claim_production_approval():
    result = evaluate_holdout_gate(
        valid_input()
    )

    assert result.metadata["research_only"] is True
    assert result.eligible is True

    # Eligibility means only that evaluation may begin.
    assert "production_approved" not in result.metadata


def test_all_required_freeze_conditions_are_explicit():
    evidence = valid_input()

    assert evidence.selection_completed
    assert evidence.walk_forward_completed
    assert evidence.feature_selection_frozen
    assert evidence.hyperparameters_frozen
    assert evidence.preprocessing_frozen

    result = evaluate_holdout_gate(evidence)

    assert result.eligible is True
