"""
Full integration and leakage tests for the AI Swing Analyser.

These tests focus on research integrity and production safety.

Core invariants:

1. Final holdout must remain isolated.
2. Production approval cannot be inferred from prediction confidence.
3. Missing approval must fail closed.
4. Missing holdout evidence must fail closed.
5. Required horizons must remain consistent.
6. Research components must not silently become production-approved.
7. Integration layers must not fit or tune models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from src.research.api_contracts import (
    extract_approved,
    extract_final_holdout_used,
    normalize_horizon_contract,
    normalize_stage_contract,
    production_pipeline_approved,
    require_final_holdout,
    require_production_approval,
    require_stage_passed,
    validate_horizon_consistency,
    validate_research_provenance,
)
from src.research.end_to_end_research_orchestrator import (
    EndToEndResearchConfig,
    EndToEndResearchOrchestrator,
)
from src.research.final_holdout_integration import (
    FinalHoldoutIntegration,
    FinalHoldoutIntegrationConfig,
)
from src.research.production_approval_integration import (
    ProductionApprovalConfig,
    ProductionApprovalIntegration,
)
from src.research.production_prediction_integration import (
    ProductionPredictionConfig,
    ProductionPredictionIntegration,
)


# ---------------------------------------------------------------------
# TEST FIXTURES
# ---------------------------------------------------------------------


@dataclass
class FakeHorizonResult:
    horizon: int
    passed: bool = True
    approved: bool = False
    evaluated: bool = True
    accuracy: float | None = None
    confidence: float | None = None
    metadata: dict = field(
        default_factory=dict
    )


@dataclass
class FakeStage:
    passed: bool = True
    approved: bool = False
    production_ready: bool = False
    production_approved: bool = False
    final_holdout_used: bool = False

    results: dict = field(
        default_factory=dict
    )

    metadata: dict = field(
        default_factory=dict
    )

    errors: list = field(
        default_factory=list
    )

    warnings: list = field(
        default_factory=list
    )


def make_stage(
    horizons=(1, 3, 5),
    *,
    passed=True,
    approved=False,
    final_holdout_used=False,
    accuracy=None,
):
    results = {}

    for horizon in horizons:
        results[horizon] = FakeHorizonResult(
            horizon=horizon,
            passed=passed,
            approved=approved,
            evaluated=True,
            accuracy=accuracy,
            confidence=(
                accuracy
                if accuracy is not None
                else None
            ),
            metadata={
                "research_only": not approved,
                "final_holdout_used": (
                    final_holdout_used
                ),
                "production_approved": (
                    approved
                ),
            },
        )

    return FakeStage(
        passed=passed,
        approved=approved,
        production_ready=approved,
        production_approved=approved,
        final_holdout_used=(
            final_holdout_used
        ),
        results=results,
        metadata={
            "research_only": not approved,
            "final_holdout_used": (
                final_holdout_used
            ),
            "production_approved": (
                approved
            ),
            "production_ready": approved,
            "model_fitted_here": False,
            "threshold_optimized_here": False,
            "feature_selection_here": False,
            "calibration_here": False,
        },
    )


def make_approved_pipeline_evidence(
    horizons=(1, 3, 5),
):
    selection = make_stage(
        horizons=horizons,
        passed=True,
    )

    backtest = make_stage(
        horizons=horizons,
        passed=True,
    )

    robustness = make_stage(
        horizons=horizons,
        passed=True,
    )

    holdout = make_stage(
        horizons=horizons,
        passed=True,
        final_holdout_used=True,
        accuracy=0.96,
    )

    approval = make_stage(
        horizons=horizons,
        passed=True,
        approved=True,
        final_holdout_used=True,
        accuracy=0.96,
    )

    return (
        selection,
        backtest,
        robustness,
        holdout,
        approval,
    )


def make_directions(
    horizons=(1, 3, 5),
    n=50,
):
    actual = {}
    predicted = {}
    probabilities = {}

    for horizon in horizons:
        values = np.array(
            [0, 1] * (n // 2)
        )

        if len(values) < n:
            values = np.append(
                values,
                0,
            )

        values = values[:n]

        actual[horizon] = pd.Series(
            values
        )

        predicted[horizon] = pd.Series(
            values
        )

        probabilities[horizon] = pd.Series(
            [0.96 if x == 1 else 0.04 for x in values]
        )

    return (
        actual,
        predicted,
        probabilities,
    )


# ---------------------------------------------------------------------
# API CONTRACT TESTS
# ---------------------------------------------------------------------


def test_missing_approval_is_not_approved():
    stage = make_stage(
        horizons=(5,)
    )

    assert (
        extract_approved(stage)
        is False
    )


def test_explicit_approval_is_detected():
    stage = make_stage(
        horizons=(5,),
        approved=True,
        final_holdout_used=True,
    )

    assert (
        extract_approved(stage)
        is True
    )


def test_holdout_usage_is_detected():
    stage = make_stage(
        horizons=(5,),
        final_holdout_used=True,
    )

    assert (
        extract_final_holdout_used(stage)
        is True
    )


def test_missing_holdout_is_not_detected():
    stage = make_stage(
        horizons=(5,)
    )

    assert (
        extract_final_holdout_used(stage)
        is False
    )


def test_normalized_stage_contract():
    stage = make_stage(
        horizons=(1, 5),
        passed=True,
    )

    contract = normalize_stage_contract(
        "backtest",
        stage,
        horizons=(1, 5),
    )

    assert contract.stage_name == "backtest"
    assert contract.executed is True
    assert contract.passed is True
    assert contract.horizons == (
        1,
        5,
    )


def test_missing_horizon_does_not_pass():
    stage = make_stage(
        horizons=(1,)
    )

    contract = normalize_horizon_contract(
        stage,
        5,
    )

    assert contract.passed is False
    assert contract.approved is False
    assert contract.evaluated is False


def test_require_stage_passed_rejects_failure():
    stage = make_stage(
        horizons=(5,),
        passed=False,
    )

    with pytest.raises(ValueError):
        require_stage_passed(
            "backtest",
            stage,
        )


def test_require_stage_passed_accepts_success():
    stage = make_stage(
        horizons=(5,),
        passed=True,
    )

    require_stage_passed(
        "backtest",
        stage,
    )


def test_require_final_holdout_rejects_missing_evidence():
    stage = make_stage(
        horizons=(5,)
    )

    with pytest.raises(ValueError):
        require_final_holdout(
            stage
        )


def test_require_final_holdout_accepts_evidence():
    stage = make_stage(
        horizons=(5,),
        final_holdout_used=True,
    )

    require_final_holdout(
        stage
    )


def test_require_production_approval_rejects_missing_approval():
    stage = make_stage(
        horizons=(5,)
    )

    with pytest.raises(ValueError):
        require_production_approval(
            stage
        )


def test_require_production_approval_accepts_approval():
    stage = make_stage(
        horizons=(5,),
        approved=True,
        final_holdout_used=True,
    )

    require_production_approval(
        stage
    )


# ---------------------------------------------------------------------
# HORIZON CONSISTENCY
# ---------------------------------------------------------------------


def test_horizon_consistency_passes():
    sources = {
        "selection": make_stage(
            horizons=(1, 3, 5)
        ),
        "backtest": make_stage(
            horizons=(1, 3, 5)
        ),
        "robustness": make_stage(
            horizons=(1, 3, 5)
        ),
    }

    validate_horizon_consistency(
        sources,
        (1, 3, 5),
    )


def test_horizon_consistency_rejects_missing_horizon():
    sources = {
        "selection": make_stage(
            horizons=(1, 3, 5)
        ),
        "backtest": make_stage(
            horizons=(1, 3)
        ),
    }

    with pytest.raises(ValueError):
        validate_horizon_consistency(
            sources,
            (1, 3, 5),
        )


# ---------------------------------------------------------------------
# PROVENANCE / LEAKAGE
# ---------------------------------------------------------------------


def test_research_result_without_forbidden_flags_is_valid():
    stage = make_stage(
        horizons=(5,)
    )

    validate_research_provenance(
        stage
    )


def test_production_approved_research_result_is_rejected():
    stage = make_stage(
        horizons=(5,),
        approved=True,
    )

    with pytest.raises(ValueError):
        validate_research_provenance(
            stage
        )


def test_model_fitting_inside_integration_is_rejected():
    stage = make_stage(
        horizons=(5,)
    )

    stage.metadata[
        "model_fitted_here"
    ] = True

    with pytest.raises(ValueError):
        validate_research_provenance(
            stage
        )


def test_threshold_optimization_inside_integration_is_rejected():
    stage = make_stage(
        horizons=(5,)
    )

    stage.metadata[
        "threshold_optimized_here"
    ] = True

    with pytest.raises(ValueError):
        validate_research_provenance(
            stage
        )


def test_feature_selection_inside_integration_is_rejected():
    stage = make_stage(
        horizons=(5,)
    )

    stage.metadata[
        "feature_selection_here"
    ] = True

    with pytest.raises(ValueError):
        validate_research_provenance(
            stage
        )


# ---------------------------------------------------------------------
# FINAL HOLDOUT
# ---------------------------------------------------------------------


def test_final_holdout_requires_sufficient_samples():
    config = FinalHoldoutIntegrationConfig(
        horizons=(5,),
        minimum_direction_samples=30,
        minimum_accuracy=0.95,
    )

    engine = FinalHoldoutIntegration(
        config
    )

    actual = {
        5: pd.Series(
            [0, 1] * 10
        )
    }

    predicted = {
        5: pd.Series(
            [0, 1] * 10
        )
    }

    probabilities = {
        5: pd.Series(
            [0.96, 0.04] * 10
        )
    }

    result = engine.run(
        holdout_data={
            "used_for_training": False,
            "used_for_tuning": False,
            "used_for_model_selection": False,
        },
        actual_directions=actual,
        predicted_directions=predicted,
        predicted_probabilities=probabilities,
    )

    assert result.candidate_passed is False


def test_final_holdout_can_pass_at_95_percent():
    config = FinalHoldoutIntegrationConfig(
        horizons=(5,),
        minimum_direction_samples=30,
        minimum_accuracy=0.95,
    )

    engine = FinalHoldoutIntegration(
        config
    )

    actual = pd.Series(
        [0, 1] * 25
    )

    predicted = actual.copy()

    result = engine.run(
        holdout_data={
            "used_for_training": False,
            "used_for_tuning": False,
            "used_for_model_selection": False,
        },
        actual_directions={
            5: actual
        },
        predicted_directions={
            5: predicted
        },
        predicted_probabilities={
            5: pd.Series(
                [0.96, 0.96] * 25
            )
        },
    )

    assert result.get(
        5
    ).accuracy == 1.0

    assert result.get(
        5
    ).passed is True


def test_holdout_training_leakage_is_rejected():
    engine = FinalHoldoutIntegration(
        FinalHoldoutIntegrationConfig(
            horizons=(5,)
        )
    )

    actual, predicted, probabilities = (
        make_directions(
            horizons=(5,),
            n=50,
        )
    )

    with pytest.raises(ValueError):
        engine.run(
            holdout_data={
                "used_for_training": True
            },
            actual_directions=actual,
            predicted_directions=predicted,
            predicted_probabilities=probabilities,
        )


def test_holdout_tuning_leakage_is_rejected():
    engine = FinalHoldoutIntegration(
        FinalHoldoutIntegrationConfig(
            horizons=(5,)
        )
    )

    actual, predicted, probabilities = (
        make_directions(
            horizons=(5,),
            n=50,
        )
    )

    with pytest.raises(ValueError):
        engine.run(
            holdout_data={
                "used_for_tuning": True
            },
            actual_directions=actual,
            predicted_directions=predicted,
            predicted_probabilities=probabilities,
        )


def test_holdout_feature_selection_leakage_is_rejected():
    engine = FinalHoldoutIntegration(
        FinalHoldoutIntegrationConfig(
            horizons=(5,)
        )
    )

    actual, predicted, probabilities = (
        make_directions(
            horizons=(5,),
            n=50,
        )
    )

    with pytest.raises(ValueError):
        engine.run(
            holdout_data={
                "used_for_feature_selection": True
            },
            actual_directions=actual,
            predicted_directions=predicted,
            predicted_probabilities=probabilities,
        )


def test_holdout_calibration_leakage_is_rejected():
    engine = FinalHoldoutIntegration(
        FinalHoldoutIntegrationConfig(
            horizons=(5,)
        )
    )

    actual, predicted, probabilities = (
        make_directions(
            horizons=(5,),
            n=50,
        )
    )

    with pytest.raises(ValueError):
        engine.run(
            holdout_data={
                "used_for_calibration": True
            },
            actual_directions=actual,
            predicted_directions=predicted,
            predicted_probabilities=probabilities,
        )


# ---------------------------------------------------------------------
# PRODUCTION APPROVAL
# ---------------------------------------------------------------------


def test_production_approval_requires_holdout():
    selection = make_stage(
        horizons=(5,),
        passed=True,
    )

    backtest = make_stage(
        horizons=(5,),
        passed=True,
    )

    robustness = make_stage(
        horizons=(5,),
        passed=True,
    )

    holdout = make_stage(
        horizons=(5,),
        passed=True,
        final_holdout_used=False,
        accuracy=0.99,
    )

    engine = ProductionApprovalIntegration(
        ProductionApprovalConfig(
            required_horizons=(5,),
            minimum_approved_horizons=1,
        )
    )

    result = engine.run(
        model_selection=selection,
        backtest=backtest,
        robustness=robustness,
        final_holdout=holdout,
    )

    assert result.approved is False
    assert result.production_approved is False


def test_production_approval_requires_all_research_gates():
    selection = make_stage(
        horizons=(5,),
        passed=True,
    )

    backtest = make_stage(
        horizons=(5,),
        passed=False,
    )

    robustness = make_stage(
        horizons=(5,),
        passed=True,
    )

    holdout = make_stage(
        horizons=(5,),
        passed=True,
        final_holdout_used=True,
        accuracy=1.0,
    )

    engine = ProductionApprovalIntegration(
        ProductionApprovalConfig(
            required_horizons=(5,),
        )
    )

    result = engine.run(
        model_selection=selection,
        backtest=backtest,
        robustness=robustness,
        final_holdout=holdout,
    )

    assert result.approved is False
    assert result.production_ready is False
    assert result.production_approved is False


def test_95_percent_holdout_gate_is_enforced():
    selection = make_stage(
        horizons=(5,),
        passed=True,
    )

    backtest = make_stage(
        horizons=(5,),
        passed=True,
    )

    robustness = make_stage(
        horizons=(5,),
        passed=True,
    )

    holdout = make_stage(
        horizons=(5,),
        passed=True,
        final_holdout_used=True,
        accuracy=0.94,
    )

    engine = ProductionApprovalIntegration(
        ProductionApprovalConfig(
            required_horizons=(5,),
            require_95_percent_gate=True,
        )
    )

    result = engine.run(
        model_selection=selection,
        backtest=backtest,
        robustness=robustness,
        final_holdout=holdout,
    )

    assert result.approved is False
    assert 5 in result.rejected_horizons


def test_95_percent_gate_can_be_disabled_explicitly():
    selection = make_stage(
        horizons=(5,),
        passed=True,
    )

    backtest = make_stage(
        horizons=(5,),
        passed=True,
    )

    robustness = make_stage(
        horizons=(5,),
        passed=True,
    )

    holdout = make_stage(
        horizons=(5,),
        passed=True,
        final_holdout_used=True,
        accuracy=0.94,
    )

    engine = ProductionApprovalIntegration(
        ProductionApprovalConfig(
            required_horizons=(5,),
            require_95_percent_gate=False,
        )
    )

    result = engine.run(
        model_selection=selection,
        backtest=backtest,
        robustness=robustness,
        final_holdout=holdout,
    )

    assert result.approved is True


def test_missing_horizon_cannot_be_approved():
    (
        selection,
        backtest,
        robustness,
        holdout,
        _,
    ) = make_approved_pipeline_evidence(
        horizons=(1,)
    )

    engine = ProductionApprovalIntegration(
        ProductionApprovalConfig(
            required_horizons=(1, 5),
            minimum_approved_horizons=1,
        )
    )

    result = engine.run(
        model_selection=selection,
        backtest=backtest,
        robustness=robustness,
        final_holdout=holdout,
    )

    assert result.approved is False


def test_approval_result_contains_explicit_governance_flags():
    (
        selection,
        backtest,
        robustness,
        holdout,
        _,
    ) = make_approved_pipeline_evidence(
        horizons=(5,)
    )

    engine = ProductionApprovalIntegration(
        ProductionApprovalConfig(
            required_horizons=(5,),
        )
    )

    result = engine.run(
        model_selection=selection,
        backtest=backtest,
        robustness=robustness,
        final_holdout=holdout,
    )

    assert (
        result.production_approved
        is False
    )


# ---------------------------------------------------------------------
# PRODUCTION PREDICTION
# ---------------------------------------------------------------------


def test_unapproved_model_returns_wait():
    approval = make_stage(
        horizons=(5,),
        passed=True,
        approved=False,
        final_holdout_used=True,
    )

    engine = ProductionPredictionIntegration(
        ProductionPredictionConfig(
            allowed_horizons=(5,),
            minimum_probability=0.95,
        )
    )

    prediction = engine.predict(
        symbol="RELIANCE",
        horizon=5,
        probability_up=0.99,
        approval_result=approval,
    )

    assert (
        prediction.predicted_direction
        == "WAIT"
    )

    assert (
        prediction.executable
        is False
    )

    assert (
        prediction.production_approved
        is False
    )


def test_missing_holdout_returns_wait():
    approval = make_stage(
        horizons=(5,),
        passed=True,
        approved=True,
        final_holdout_used=False,
    )

    engine = ProductionPredictionIntegration(
        ProductionPredictionConfig(
            allowed_horizons=(5,),
        )
    )

    prediction = engine.predict(
        symbol="RELIANCE",
        horizon=5,
        probability_up=0.99,
        approval_result=approval,
    )

    assert (
        prediction.predicted_direction
        == "WAIT"
    )

    assert (
        prediction.executable
        is False
    )


def test_unapproved_horizon_returns_wait():
    approval = make_stage(
        horizons=(1,),
        passed=True,
        approved=True,
        final_holdout_used=True,
    )

    engine = ProductionPredictionIntegration(
        ProductionPredictionConfig(
            allowed_horizons=(1, 5),
        )
    )

    prediction = engine.predict(
        symbol="RELIANCE",
        horizon=5,
        probability_up=0.99,
        approval_result=approval,
    )

    assert (
        prediction.predicted_direction
        == "WAIT"
    )

    assert (
        prediction.executable
        is False
    )


def test_high_probability_cannot_bypass_approval():
    approval = make_stage(
        horizons=(5,),
        passed=False,
        approved=False,
        final_holdout_used=False,
    )

    engine = ProductionPredictionIntegration(
        ProductionPredictionConfig(
            allowed_horizons=(5,),
            minimum_probability=0.95,
        )
    )

    prediction = engine.predict(
        symbol="TCS",
        horizon=5,
        probability_up=0.9999,
        approval_result=approval,
    )

    assert prediction.executable is False
    assert prediction.predicted_direction == "WAIT"


def test_probability_range_is_validated():
    approval = make_stage(
        horizons=(5,),
        approved=True,
        final_holdout_used=True,
    )

    engine = ProductionPredictionIntegration(
        ProductionPredictionConfig(
            allowed_horizons=(5,),
        )
    )

    with pytest.raises(ValueError):
        engine.predict(
            symbol="INFY",
            horizon=5,
            probability_up=1.5,
            approval_result=approval,
        )


def test_negative_probability_is_rejected():
    approval = make_stage(
        horizons=(5,),
        approved=True,
        final_holdout_used=True,
    )

    engine = ProductionPredictionIntegration(
        ProductionPredictionConfig(
            allowed_horizons=(5,),
        )
    )

    with pytest.raises(ValueError):
        engine.predict(
            symbol="INFY",
            horizon=5,
            probability_up=-0.1,
            approval_result=approval,
        )


def test_invalid_target_range_blocks_execution():
    approval = make_stage(
        horizons=(5,),
        approved=True,
        final_holdout_used=True,
    )

    engine = ProductionPredictionIntegration(
        ProductionPredictionConfig(
            allowed_horizons=(5,),
            minimum_probability=0.95,
        )
    )

    prediction = engine.predict(
        symbol="INFY",
        horizon=5,
        probability_up=0.99,
        approval_result=approval,
        current_price=100,
        target_low=120,
        target_mid=110,
        target_high=130,
    )

    assert prediction.executable is False


# ---------------------------------------------------------------------
# END-TO-END ORCHESTRATOR SAFETY
# ---------------------------------------------------------------------


def test_orchestrator_does_not_skip_failed_backtest():
    config = EndToEndResearchConfig(
        horizons=(5,),
        stop_on_failure=True,
    )

    orchestrator = (
        EndToEndResearchOrchestrator(
            config
        )
    )

    calls = []

    def success(name):
        def stage(**kwargs):
            calls.append(name)
            return make_stage(
                horizons=(5,),
                passed=True,
            )

        return stage

    def failed_backtest(**kwargs):
        calls.append("backtest")
        return make_stage(
            horizons=(5,),
            passed=False,
        )

    result = orchestrator.run(
        data_stage=success("data"),
        feature_stage=success("features"),
        target_stage=success("targets"),
        model_selection_stage=success(
            "selection"
        ),
        backtest_stage=failed_backtest,
        robustness_stage=success(
            "robustness"
        ),
        final_holdout_stage=success(
            "holdout"
        ),
        production_approval_stage=success(
            "approval"
        ),
    )

    assert result.production_approved is False

    assert calls == [
        "data",
        "features",
        "targets",
        "selection",
        "backtest",
    ]


def test_orchestrator_cannot_create_approval_without_approval_stage():
    config = EndToEndResearchConfig(
        horizons=(5,),
    )

    orchestrator = (
        EndToEndResearchOrchestrator(
            config
        )
    )

    def success(**kwargs):
        return make_stage(
            horizons=(5,),
            passed=True,
        )

    def holdout(**kwargs):
        return make_stage(
            horizons=(5,),
            passed=True,
            final_holdout_used=True,
        )

    result = orchestrator.run(
        data_stage=success,
        feature_stage=success,
        target_stage=success,
        model_selection_stage=success,
        backtest_stage=success,
        robustness_stage=success,
        final_holdout_stage=holdout,
        production_approval_stage=None,
    )

    assert result.production_approved is False
    assert result.production_ready is False


# ---------------------------------------------------------------------
# FINAL PRODUCTION PREDICATE
# ---------------------------------------------------------------------


def test_production_pipeline_requires_all_horizons():
    approval = make_stage(
        horizons=(1,),
        passed=True,
        approved=True,
        final_holdout_used=True,
    )

    holdout = make_stage(
        horizons=(1,),
        passed=True,
        final_holdout_used=True,
    )

    assert (
        production_pipeline_approved(
            approval_result=approval,
            final_holdout_result=holdout,
            required_horizons=(1, 5),
        )
        is False
    )


def test_production_pipeline_requires_holdout():
    approval = make_stage(
        horizons=(5,),
        passed=True,
        approved=True,
        final_holdout_used=True,
    )

    holdout = make_stage(
        horizons=(5,),
        passed=True,
        final_holdout_used=False,
    )

    assert (
        production_pipeline_approved(
            approval_result=approval,
            final_holdout_result=holdout,
            required_horizons=(5,),
        )
        is False
    )


def test_production_pipeline_requires_approval():
    approval = make_stage(
        horizons=(5,),
        passed=True,
        approved=False,
    )

    holdout = make_stage(
        horizons=(5,),
        passed=True,
        final_holdout_used=True,
    )

    assert (
        production_pipeline_approved(
            approval_result=approval,
            final_holdout_result=holdout,
            required_horizons=(5,),
        )
        is False
    )


def test_production_pipeline_requires_explicit_horizon_approval():
    approval = make_stage(
        horizons=(5,),
        passed=True,
        approved=False,
        final_holdout_used=True,
    )

    holdout = make_stage(
        horizons=(5,),
        passed=True,
        final_holdout_used=True,
    )

    assert (
        production_pipeline_approved(
            approval_result=approval,
            final_holdout_result=holdout,
            required_horizons=(5,),
        )
        is False
    )


# ---------------------------------------------------------------------
# IMMUTABILITY / RESEARCH SAFETY
# ---------------------------------------------------------------------


def test_final_holdout_input_is_not_modified():
    holdout_data = {
        "used_for_training": False,
        "used_for_tuning": False,
        "used_for_model_selection": False,
    }

    original = dict(
        holdout_data
    )

    actual, predicted, probabilities = (
        make_directions(
            horizons=(5,),
            n=50,
        )
    )

    engine = FinalHoldoutIntegration(
        FinalHoldoutIntegrationConfig(
            horizons=(5,),
        )
    )

    engine.run(
        holdout_data=holdout_data,
        actual_directions=actual,
        predicted_directions=predicted,
        predicted_probabilities=probabilities,
    )

    assert (
        holdout_data
        == original
    )


def test_prediction_does_not_modify_approval_metadata():
    approval = make_stage(
        horizons=(5,),
        approved=False,
        final_holdout_used=False,
    )

    original = {
        key: value
        for key, value
        in approval.metadata.items()
    }

    engine = ProductionPredictionIntegration(
        ProductionPredictionConfig(
            allowed_horizons=(5,),
        )
    )

    engine.predict(
        symbol="RELIANCE",
        horizon=5,
        probability_up=0.99,
        approval_result=approval,
    )

    assert (
        approval.metadata
        == original
    )


# ---------------------------------------------------------------------
# ABSENCE OF FALSE CERTAINTY
# ---------------------------------------------------------------------


def test_probability_does_not_equal_approval():
    """
    A 99.9% model probability is not itself evidence of validation.
    """

    approval = make_stage(
        horizons=(5,),
        approved=False,
        final_holdout_used=False,
    )

    engine = ProductionPredictionIntegration(
        ProductionPredictionConfig(
            allowed_horizons=(5,),
            minimum_probability=0.95,
        )
    )

    prediction = engine.predict(
        symbol="HDFCBANK",
        horizon=5,
        probability_up=0.999,
        approval_result=approval,
    )

    assert prediction.confidence >= 0.95
    assert prediction.production_approved is False
    assert prediction.executable is False
    assert prediction.predicted_direction == "WAIT"


def test_missing_evidence_never_becomes_success():
    stage = make_stage(
        horizons=(5,)
    )

    assert extract_approved(stage) is False
    assert extract_final_holdout_used(stage) is False

    contract = normalize_horizon_contract(
        stage,
        5,
    )

    assert contract.approved is False


def test_full_approval_predicate_is_strict():
    (
        selection,
        backtest,
        robustness,
        holdout,
        approval,
    ) = make_approved_pipeline_evidence(
        horizons=(5,)
    )

    # Explicitly approve the horizon.
    approval.results[5].approved = True
    approval.approved = True
    approval.production_approved = True
    approval.final_holdout_used = True
    approval.metadata[
        "production_approved"
    ] = True

    assert (
        production_pipeline_approved(
            approval_result=approval,
            final_holdout_result=holdout,
            required_horizons=(5,),
        )
        is True
    )
