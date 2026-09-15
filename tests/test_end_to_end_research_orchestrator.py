"""
Tests for the end-to-end research orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.research.end_to_end_research_orchestrator import (
    DEFAULT_RESEARCH_HORIZONS,
    EndToEndResearchConfig,
    EndToEndResearchOrchestrator,
    EndToEndResearchResult,
    ResearchStageResult,
    end_to_end_research_summary,
)


@dataclass
class FakeStageResult:
    passed: bool = True
    candidate_passed: bool = False
    approved: bool = False
    production_approved: bool = False
    production_ready: bool = False
    final_holdout_used: bool = False
    metadata: dict | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def successful_stage(
    **kwargs,
):
    return FakeStageResult(
        passed=True,
        metadata={
            "research_only": True,
            **kwargs,
        },
    )


def failed_stage():
    return FakeStageResult(
        passed=False,
        metadata={
            "research_only": True,
        },
    )


def approved_stage():
    return FakeStageResult(
        passed=True,
        approved=True,
        production_approved=True,
        production_ready=True,
        final_holdout_used=True,
        metadata={
            "final_holdout_used": True,
            "production_approved": True,
            "production_ready": True,
        },
    )


def make_basic_orchestrator(
    horizons=(1, 3, 5),
):
    return EndToEndResearchOrchestrator(
        EndToEndResearchConfig(
            horizons=horizons,
        )
    )


def test_default_horizons_are_defined():
    assert DEFAULT_RESEARCH_HORIZONS == (
        1,
        3,
        5,
        10,
        20,
    )


def test_config_accepts_valid_horizons():
    config = EndToEndResearchConfig(
        horizons=(1, 3, 5)
    )

    assert config.horizons == (
        1,
        3,
        5,
    )


def test_config_rejects_empty_horizons():
    with pytest.raises(ValueError):
        EndToEndResearchConfig(
            horizons=()
        )


def test_config_rejects_non_positive_horizons():
    with pytest.raises(ValueError):
        EndToEndResearchConfig(
            horizons=(1, 0, 5)
        )


def test_config_rejects_duplicate_horizons():
    with pytest.raises(ValueError):
        EndToEndResearchConfig(
            horizons=(1, 3, 3)
        )


def test_orchestrator_has_fixed_stage_order():
    assert (
        EndToEndResearchOrchestrator.STAGE_ORDER
        == (
            "data",
            "features",
            "targets",
            "model_selection",
            "backtest",
            "robustness",
            "final_holdout",
            "production_approval",
        )
    )


def test_external_data_can_be_used():
    orchestrator = make_basic_orchestrator()

    result = orchestrator.run(
        data={"rows": 100}
    )

    assert isinstance(
        result,
        EndToEndResearchResult,
    )

    assert (
        result.stage("data").executed
        is True
    )

    assert (
        result.stage("data").output
        == {"rows": 100}
    )


def test_missing_external_data_fails_closed():
    orchestrator = make_basic_orchestrator()

    result = orchestrator.run()

    assert result.passed is False
    assert result.research_only is True
    assert result.production_approved is False


def test_all_stages_execute_in_order():
    orchestrator = make_basic_orchestrator()

    calls = []

    def data_stage(**kwargs):
        calls.append("data")
        return successful_stage()

    def feature_stage(**kwargs):
        calls.append("features")
        return successful_stage()

    def target_stage(**kwargs):
        calls.append("targets")
        return successful_stage()

    def selection_stage(**kwargs):
        calls.append("model_selection")
        return successful_stage()

    def backtest_stage(**kwargs):
        calls.append("backtest")
        return successful_stage()

    def robustness_stage(**kwargs):
        calls.append("robustness")
        return successful_stage()

    def holdout_stage(**kwargs):
        calls.append("final_holdout")
        return successful_stage(
            final_holdout_used=True
        )

    def approval_stage(**kwargs):
        calls.append("production_approval")
        return approved_stage()

    result = orchestrator.run(
        data_stage=data_stage,
        feature_stage=feature_stage,
        target_stage=target_stage,
        model_selection_stage=selection_stage,
        backtest_stage=backtest_stage,
        robustness_stage=robustness_stage,
        final_holdout_stage=holdout_stage,
        production_approval_stage=approval_stage,
    )

    assert calls == [
        "data",
        "features",
        "targets",
        "model_selection",
        "backtest",
        "robustness",
        "final_holdout",
        "production_approval",
    ]

    assert result.production_approved is True


def test_stage_outputs_are_forwarded():
    orchestrator = make_basic_orchestrator()

    captured = {}

    def data_stage(**kwargs):
        return "DATA"

    def feature_stage(**kwargs):
        captured["data"] = kwargs["data"]
        return "FEATURES"

    def target_stage(**kwargs):
        captured["features"] = kwargs["features"]
        return "TARGETS"

    def selection_stage(**kwargs):
        captured["targets"] = kwargs["targets"]
        return "SELECTION"

    def backtest_stage(**kwargs):
        captured["selection"] = (
            kwargs["model_selection"]
        )
        return "BACKTEST"

    def robustness_stage(**kwargs):
        captured["backtest"] = (
            kwargs["backtest"]
        )
        return "ROBUSTNESS"

    def holdout_stage(**kwargs):
        captured["robustness"] = (
            kwargs["robustness"]
        )
        return successful_stage(
            final_holdout_used=True
        )

    def approval_stage(**kwargs):
        captured["holdout"] = (
            kwargs["final_holdout"]
        )
        return approved_stage()

    result = orchestrator.run(
        data_stage=data_stage,
        feature_stage=feature_stage,
        target_stage=target_stage,
        model_selection_stage=selection_stage,
        backtest_stage=backtest_stage,
        robustness_stage=robustness_stage,
        final_holdout_stage=holdout_stage,
        production_approval_stage=approval_stage,
    )

    assert result.production_approved is True

    assert captured["data"] == "DATA"
    assert captured["features"] == "FEATURES"
    assert captured["targets"] == "TARGETS"
    assert captured["selection"] == "SELECTION"
    assert captured["backtest"] == "BACKTEST"
    assert captured["robustness"] == "ROBUSTNESS"


def test_data_failure_stops_pipeline():
    orchestrator = make_basic_orchestrator()

    calls = []

    def data_stage(**kwargs):
        calls.append("data")
        return failed_stage()

    def feature_stage(**kwargs):
        calls.append("features")
        return successful_stage()

    result = orchestrator.run(
        data_stage=data_stage,
        feature_stage=feature_stage,
    )

    assert calls == ["data"]

    assert (
        result.stage("data").passed
        is False
    )

    assert (
        result.stage("features").executed
        is False
    )

    assert result.passed is False


def test_feature_failure_stops_pipeline():
    orchestrator = make_basic_orchestrator()

    calls = []

    def data_stage(**kwargs):
        calls.append("data")
        return successful_stage()

    def feature_stage(**kwargs):
        calls.append("features")
        return failed_stage()

    def target_stage(**kwargs):
        calls.append("targets")
        return successful_stage()

    result = orchestrator.run(
        data_stage=data_stage,
        feature_stage=feature_stage,
        target_stage=target_stage,
    )

    assert calls == [
        "data",
        "features",
    ]

    assert (
        result.stage("targets").executed
        is False
    )


def test_target_failure_stops_pipeline():
    orchestrator = make_basic_orchestrator()

    calls = []

    def data_stage(**kwargs):
        calls.append("data")
        return successful_stage()

    def feature_stage(**kwargs):
        calls.append("features")
        return successful_stage()

    def target_stage(**kwargs):
        calls.append("targets")
        return failed_stage()

    def selection_stage(**kwargs):
        calls.append("model_selection")
        return successful_stage()

    result = orchestrator.run(
        data_stage=data_stage,
        feature_stage=feature_stage,
        target_stage=target_stage,
        model_selection_stage=selection_stage,
    )

    assert calls == [
        "data",
        "features",
        "targets",
    ]


def test_model_selection_failure_stops_pipeline():
    orchestrator = make_basic_orchestrator()

    calls = []

    def data_stage(**kwargs):
        calls.append("data")
        return successful_stage()

    def feature_stage(**kwargs):
        calls.append("features")
        return successful_stage()

    def target_stage(**kwargs):
        calls.append("targets")
        return successful_stage()

    def selection_stage(**kwargs):
        calls.append("model_selection")
        return failed_stage()

    def backtest_stage(**kwargs):
        calls.append("backtest")
        return successful_stage()

    result = orchestrator.run(
        data_stage=data_stage,
        feature_stage=feature_stage,
        target_stage=target_stage,
        model_selection_stage=selection_stage,
        backtest_stage=backtest_stage,
    )

    assert calls == [
        "data",
        "features",
        "targets",
        "model_selection",
    ]

    assert (
        result.stage("backtest").executed
        is False
    )


def test_backtest_failure_stops_pipeline():
    orchestrator = make_basic_orchestrator()

    calls = []

    def data_stage(**kwargs):
        calls.append("data")
        return successful_stage()

    def feature_stage(**kwargs):
        calls.append("features")
        return successful_stage()

    def target_stage(**kwargs):
        calls.append("targets")
        return successful_stage()

    def selection_stage(**kwargs):
        calls.append("model_selection")
        return successful_stage()

    def backtest_stage(**kwargs):
        calls.append("backtest")
        return failed_stage()

    def robustness_stage(**kwargs):
        calls.append("robustness")
        return successful_stage()

    result = orchestrator.run(
        data_stage=data_stage,
        feature_stage=feature_stage,
        target_stage=target_stage,
        model_selection_stage=selection_stage,
        backtest_stage=backtest_stage,
        robustness_stage=robustness_stage,
    )

    assert calls == [
        "data",
        "features",
        "targets",
        "model_selection",
        "backtest",
    ]

    assert (
        result.stage("robustness").executed
        is False
    )


def test_robustness_failure_stops_pipeline():
    orchestrator = make_basic_orchestrator()

    calls = []

    def data_stage(**kwargs):
        calls.append("data")
        return successful_stage()

    def feature_stage(**kwargs):
        calls.append("features")
        return successful_stage()

    def target_stage(**kwargs):
        calls.append("targets")
        return successful_stage()

    def selection_stage(**kwargs):
        calls.append("model_selection")
        return successful_stage()

    def backtest_stage(**kwargs):
        calls.append("backtest")
        return successful_stage()

    def robustness_stage(**kwargs):
        calls.append("robustness")
        return failed_stage()

    def holdout_stage(**kwargs):
        calls.append("final_holdout")
        return successful_stage(
            final_holdout_used=True
        )

    result = orchestrator.run(
        data_stage=data_stage,
        feature_stage=feature_stage,
        target_stage=target_stage,
        model_selection_stage=selection_stage,
        backtest_stage=backtest_stage,
        robustness_stage=robustness_stage,
        final_holdout_stage=holdout_stage,
    )

    assert calls == [
        "data",
        "features",
        "targets",
        "model_selection",
        "backtest",
        "robustness",
    ]

    assert (
        result.stage("final_holdout").executed
        is False
    )


def test_holdout_failure_stops_approval():
    orchestrator = make_basic_orchestrator()

    calls = []

    def successful(name):
        def fn(**kwargs):
            calls.append(name)
            return successful_stage()

        return fn

    def holdout(**kwargs):
        calls.append("final_holdout")
        return failed_stage()

    def approval(**kwargs):
        calls.append("production_approval")
        return approved_stage()

    result = orchestrator.run(
        data_stage=successful("data"),
        feature_stage=successful("features"),
        target_stage=successful("targets"),
        model_selection_stage=successful(
            "model_selection"
        ),
        backtest_stage=successful("backtest"),
        robustness_stage=successful("robustness"),
        final_holdout_stage=holdout,
        production_approval_stage=approval,
    )

    assert calls[-1] == "final_holdout"

    assert (
        "production_approval"
        not in calls
    )

    assert result.production_approved is False


def test_production_approval_failure_is_not_approved():
    orchestrator = make_basic_orchestrator()

    def successful_stage_fn(**kwargs):
        return successful_stage()

    def holdout(**kwargs):
        return successful_stage(
            final_holdout_used=True
        )

    def approval(**kwargs):
        return failed_stage()

    result = orchestrator.run(
        data_stage=successful_stage_fn,
        feature_stage=successful_stage_fn,
        target_stage=successful_stage_fn,
        model_selection_stage=successful_stage_fn,
        backtest_stage=successful_stage_fn,
        robustness_stage=successful_stage_fn,
        final_holdout_stage=holdout,
        production_approval_stage=approval,
    )

    assert result.passed is False
    assert result.production_approved is False
    assert result.production_ready is False


def test_successful_complete_pipeline():
    orchestrator = make_basic_orchestrator()

    def research_stage(**kwargs):
        return successful_stage()

    def holdout_stage(**kwargs):
        return successful_stage(
            final_holdout_used=True
        )

    def approval_stage(**kwargs):
        return approved_stage()

    result = orchestrator.run(
        data_stage=research_stage,
        feature_stage=research_stage,
        target_stage=research_stage,
        model_selection_stage=research_stage,
        backtest_stage=research_stage,
        robustness_stage=research_stage,
        final_holdout_stage=holdout_stage,
        production_approval_stage=approval_stage,
    )

    assert result.passed is True
    assert result.final_holdout_used is True
    assert result.production_approved is True
    assert result.production_ready is True


def test_complete_pipeline_is_not_research_only_after_approval():
    orchestrator = make_basic_orchestrator()

    def research_stage(**kwargs):
        return successful_stage()

    def holdout_stage(**kwargs):
        return successful_stage(
            final_holdout_used=True
        )

    def approval_stage(**kwargs):
        return approved_stage()

    result = orchestrator.run(
        data_stage=research_stage,
        feature_stage=research_stage,
        target_stage=research_stage,
        model_selection_stage=research_stage,
        backtest_stage=research_stage,
        robustness_stage=research_stage,
        final_holdout_stage=holdout_stage,
        production_approval_stage=approval_stage,
    )

    assert result.research_only is False


def test_missing_production_approval_blocks_completion():
    orchestrator = make_basic_orchestrator()

    def research_stage(**kwargs):
        return successful_stage()

    def holdout_stage(**kwargs):
        return successful_stage(
            final_holdout_used=True
        )

    result = orchestrator.run(
        data_stage=research_stage,
        feature_stage=research_stage,
        target_stage=research_stage,
        model_selection_stage=research_stage,
        backtest_stage=research_stage,
        robustness_stage=research_stage,
        final_holdout_stage=holdout_stage,
        production_approval_stage=None,
    )

    assert result.production_approved is False
    assert result.passed is False


def test_missing_holdout_blocks_completion():
    orchestrator = make_basic_orchestrator()

    def research_stage(**kwargs):
        return successful_stage()

    result = orchestrator.run(
        data_stage=research_stage,
        feature_stage=research_stage,
        target_stage=research_stage,
        model_selection_stage=research_stage,
        backtest_stage=research_stage,
        robustness_stage=research_stage,
        final_holdout_stage=None,
        production_approval_stage=research_stage,
    )

    assert result.passed is False
    assert result.production_approved is False


def test_holdout_can_be_explicitly_disabled():
    config = EndToEndResearchConfig(
        horizons=(1, 3, 5),
        enable_final_holdout=False,
        enable_production_approval=False,
    )

    orchestrator = EndToEndResearchOrchestrator(
        config
    )

    def research_stage(**kwargs):
        return successful_stage()

    result = orchestrator.run(
        data_stage=research_stage,
        feature_stage=research_stage,
        target_stage=research_stage,
        model_selection_stage=research_stage,
        backtest_stage=research_stage,
        robustness_stage=research_stage,
    )

    assert result.passed is True
    assert result.final_holdout_used is False
    assert result.production_approved is False


def test_disabled_approval_never_creates_production_approval():
    config = EndToEndResearchConfig(
        horizons=(5,),
        enable_final_holdout=False,
        enable_production_approval=False,
    )

    orchestrator = EndToEndResearchOrchestrator(
        config
    )

    def research_stage(**kwargs):
        return approved_stage()

    result = orchestrator.run(
        data_stage=research_stage,
        feature_stage=research_stage,
        target_stage=research_stage,
        model_selection_stage=research_stage,
        backtest_stage=research_stage,
        robustness_stage=research_stage,
    )

    assert result.production_approved is False
    assert result.production_ready is False


def test_exception_in_stage_is_captured():
    orchestrator = make_basic_orchestrator()

    def data_stage(**kwargs):
        raise RuntimeError(
            "synthetic failure"
        )

    result = orchestrator.run(
        data_stage=data_stage
    )

    assert result.passed is False
    assert result.errors

    assert any(
        "synthetic failure" in error
        for error in result.errors
    )


def test_stage_result_with_no_explicit_pass_status():
    orchestrator = make_basic_orchestrator()

    def data_stage(**kwargs):
        return {"data": 123}

    result = orchestrator.run(
        data_stage=data_stage
    )

    assert (
        result.stage("data").executed
        is True
    )

    assert (
        result.stage("data").passed
        is True
    )

    assert result.stage(
        "data"
    ).warnings


def test_stage_output_can_be_retrieved():
    stage = ResearchStageResult(
        name="data",
        executed=True,
        passed=True,
        output={"x": 1},
    )

    result = EndToEndResearchResult(
        stages={
            "data": stage
        }
    )

    assert (
        result.output("data")
        == {"x": 1}
    )


def test_unknown_stage_raises():
    result = EndToEndResearchResult(
        stages={}
    )

    with pytest.raises(KeyError):
        result.stage("unknown")


def test_unknown_output_stage_raises():
    result = EndToEndResearchResult(
        stages={}
    )

    with pytest.raises(KeyError):
        result.output("unknown")


def test_final_holdout_metadata_is_detected():
    orchestrator = make_basic_orchestrator()

    def research_stage(**kwargs):
        return successful_stage()

    def holdout_stage(**kwargs):
        return FakeStageResult(
            passed=True,
            metadata={
                "final_holdout_used": True
            },
        )

    def approval_stage(**kwargs):
        return approved_stage()

    result = orchestrator.run(
        data_stage=research_stage,
        feature_stage=research_stage,
        target_stage=research_stage,
        model_selection_stage=research_stage,
        backtest_stage=research_stage,
        robustness_stage=research_stage,
        final_holdout_stage=holdout_stage,
        production_approval_stage=approval_stage,
    )

    assert result.final_holdout_used is True


def test_production_approval_can_be_detected_from_approved_flag():
    orchestrator = make_basic_orchestrator()

    def research_stage(**kwargs):
        return successful_stage()

    def holdout_stage(**kwargs):
        return successful_stage(
            final_holdout_used=True
        )

    def approval_stage(**kwargs):
        return FakeStageResult(
            passed=True,
            approved=True,
            final_holdout_used=True,
        )

    result = orchestrator.run(
        data_stage=research_stage,
        feature_stage=research_stage,
        target_stage=research_stage,
        model_selection_stage=research_stage,
        backtest_stage=research_stage,
        robustness_stage=research_stage,
        final_holdout_stage=holdout_stage,
        production_approval_stage=approval_stage,
    )

    assert result.production_approved is True


def test_summary_returns_dictionary():
    result = EndToEndResearchResult(
        stages={
            "data": ResearchStageResult(
                name="data",
                executed=True,
                passed=True,
            )
        }
    )

    summary = end_to_end_research_summary(
        result
    )

    assert isinstance(
        summary,
        dict,
    )


def test_summary_contains_stage_information():
    result = EndToEndResearchResult(
        stages={
            "data": ResearchStageResult(
                name="data",
                executed=True,
                passed=True,
            )
        }
    )

    summary = end_to_end_research_summary(
        result
    )

    assert "stages" in summary
    assert "data" in summary["stages"]


def test_summary_contains_governance_flags():
    result = EndToEndResearchResult(
        stages={}
    )

    summary = end_to_end_research_summary(
        result
    )

    assert (
        "production_approved"
        in summary
    )

    assert (
        "production_ready"
        in summary
    )

    assert (
        "final_holdout_used"
        in summary
    )

    assert (
        "research_only"
        in summary
    )


def test_summary_rejects_wrong_type():
    with pytest.raises(TypeError):
        end_to_end_research_summary(
            None
        )


def test_horizons_are_forwarded_to_stages():
    orchestrator = make_basic_orchestrator(
        horizons=(3, 10)
    )

    captured = {}

    def data_stage(**kwargs):
        captured["data_horizons"] = kwargs.get(
            "horizons"
        )
        return successful_stage()

    def feature_stage(**kwargs):
        return successful_stage()

    def target_stage(**kwargs):
        captured["target_horizons"] = kwargs[
            "horizons"
        ]
        return successful_stage()

    def selection_stage(**kwargs):
        captured[
            "selection_horizons"
        ] = kwargs["horizons"]
        return successful_stage()

    def backtest_stage(**kwargs):
        captured[
            "backtest_horizons"
        ] = kwargs["horizons"]
        return successful_stage()

    def robustness_stage(**kwargs):
        captured[
            "robustness_horizons"
        ] = kwargs["horizons"]
        return successful_stage()

    def holdout_stage(**kwargs):
        captured[
            "holdout_horizons"
        ] = kwargs["horizons"]
        return successful_stage(
            final_holdout_used=True
        )

    def approval_stage(**kwargs):
        captured[
            "approval_horizons"
        ] = kwargs["horizons"]
        return approved_stage()

    result = orchestrator.run(
        data_stage=data_stage,
        feature_stage=feature_stage,
        target_stage=target_stage,
        model_selection_stage=selection_stage,
        backtest_stage=backtest_stage,
        robustness_stage=robustness_stage,
        final_holdout_stage=holdout_stage,
        production_approval_stage=approval_stage,
    )

    assert result.production_approved is True

    assert captured["target_horizons"] == (
        3,
        10,
    )

    assert captured["selection_horizons"] == (
        3,
        10,
    )

    assert captured["backtest_horizons"] == (
        3,
        10,
    )

    assert captured["robustness_horizons"] == (
        3,
        10,
    )

    assert captured["holdout_horizons"] == (
        3,
        10,
    )

    assert captured["approval_horizons"] == (
        3,
        10,
    )


def test_production_approval_cannot_run_after_earlier_failure():
    orchestrator = make_basic_orchestrator()

    calls = []

    def data_stage(**kwargs):
        calls.append("data")
        return successful_stage()

    def feature_stage(**kwargs):
        calls.append("features")
        return successful_stage()

    def target_stage(**kwargs):
        calls.append("targets")
        return successful_stage()

    def selection_stage(**kwargs):
        calls.append("selection")
        return successful_stage()

    def backtest_stage(**kwargs):
        calls.append("backtest")
        return failed_stage()

    def robustness_stage(**kwargs):
        calls.append("robustness")
        return approved_stage()

    def holdout_stage(**kwargs):
        calls.append("holdout")
        return approved_stage()

    def approval_stage(**kwargs):
        calls.append("approval")
        return approved_stage()

    result = orchestrator.run(
        data_stage=data_stage,
        feature_stage=feature_stage,
        target_stage=target_stage,
        model_selection_stage=selection_stage,
        backtest_stage=backtest_stage,
        robustness_stage=robustness_stage,
        final_holdout_stage=holdout_stage,
        production_approval_stage=approval_stage,
    )

    assert result.production_approved is False

    assert calls == [
        "data",
        "features",
        "targets",
        "selection",
        "backtest",
    ]


def test_production_approval_is_never_inferred_from_backtest():
    orchestrator = make_basic_orchestrator()

    def successful_backtest(**kwargs):
        return approved_stage()

    def successful_research(**kwargs):
        return approved_stage()

    result = orchestrator.run(
        data_stage=successful_research,
        feature_stage=successful_research,
        target_stage=successful_research,
        model_selection_stage=successful_research,
        backtest_stage=successful_backtest,
        robustness_stage=successful_research,
        final_holdout_stage=None,
        production_approval_stage=None,
    )

    assert result.production_approved is False
    assert result.production_ready is False


def test_research_result_defaults_to_research_only():
    result = EndToEndResearchResult(
        stages={}
    )

    assert result.research_only is True
    assert result.production_approved is False
    assert result.production_ready is False


def test_stage_errors_are_propagated():
    orchestrator = make_basic_orchestrator()

    def data_stage(**kwargs):
        result = ResearchStageResult(
            name="data",
            executed=True,
            passed=False,
            errors=["data corruption"],
        )
        return result

    result = orchestrator.run(
        data_stage=data_stage
    )

    assert any(
        "data corruption" in error
        for error in result.errors
    )


def test_research_workflow_remains_fail_closed():
    orchestrator = make_basic_orchestrator()

    def successful(**kwargs):
        return successful_stage()

    def holdout(**kwargs):
        return successful_stage(
            final_holdout_used=True
        )

    def approval(**kwargs):
        return FakeStageResult(
            passed=True,
            approved=False,
            production_approved=False,
            production_ready=False,
            final_holdout_used=True,
        )

    result = orchestrator.run(
        data_stage=successful,
        feature_stage=successful,
        target_stage=successful,
        model_selection_stage=successful,
        backtest_stage=successful,
        robustness_stage=successful,
        final_holdout_stage=holdout,
        production_approval_stage=approval,
    )

    assert result.production_approved is False
    assert result.production_ready is False
    assert result.research_only is True
