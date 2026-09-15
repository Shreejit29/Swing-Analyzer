"""
Tests for research stage adapters.

The adapters are intentionally conservative:

- known passing evidence -> PASS
- known failing evidence -> FAIL
- missing/ambiguous evidence -> NOT_EVALUATED
- final holdout contamination -> FAIL
- unsupported stage -> error
"""

from __future__ import annotations

import pytest

from src.research.integration import EvidenceStatus
from src.research.stage_adapters import (
    adapt_backtest_result,
    adapt_calibration_result,
    adapt_holdout_result,
    adapt_leakage_result,
    adapt_range_result,
    adapt_regime_result,
    adapt_robustness_result,
    adapt_stage_result,
    adapt_walk_forward_result,
)


# ---------------------------------------------------------------------
# Simple mock result
# ---------------------------------------------------------------------


class MockResult:
    def __init__(
        self,
        **kwargs,
    ):
        for key, value in kwargs.items():
            setattr(
                self,
                key,
                value,
            )

    def summary(self):
        return {
            key: value
            for key, value in self.__dict__.items()
        }


# ---------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------


def test_walk_forward_pass():
    result = MockResult(
        mean_accuracy=0.97,
        minimum_accuracy=0.95,
        passed_accuracy_gate=True,
    )

    evidence = adapt_walk_forward_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.PASS
    )

    assert (
        evidence.score
        == 0.97
    )

    assert (
        evidence.threshold
        == 0.95
    )


def test_walk_forward_failure():
    result = MockResult(
        mean_accuracy=0.90,
        minimum_accuracy=0.88,
        passed_accuracy_gate=False,
    )

    evidence = adapt_walk_forward_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.FAIL
    )


def test_walk_forward_can_derive_pass():
    result = MockResult(
        mean_accuracy=0.96,
        minimum_accuracy=0.95,
    )

    evidence = adapt_walk_forward_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.PASS
    )


def test_walk_forward_missing_is_not_evaluated():
    evidence = adapt_walk_forward_result(
        MockResult()
    )

    assert (
        evidence.status
        == EvidenceStatus.NOT_EVALUATED
    )


def test_walk_forward_none_is_not_evaluated():
    evidence = adapt_walk_forward_result(
        None
    )

    assert (
        evidence.status
        == EvidenceStatus.NOT_EVALUATED
    )


# ---------------------------------------------------------------------
# Final holdout
# ---------------------------------------------------------------------


def test_holdout_pass():
    result = MockResult(
        accuracy=0.96,
        passed_accuracy_gate=True,
    )

    evidence = adapt_holdout_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.PASS
    )

    assert (
        evidence.score
        == 0.96
    )


def test_holdout_can_derive_pass():
    result = MockResult(
        accuracy=0.96,
    )

    evidence = adapt_holdout_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.PASS
    )


def test_holdout_failure():
    result = MockResult(
        accuracy=0.90,
        passed_accuracy_gate=False,
    )

    evidence = adapt_holdout_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.FAIL
    )


def test_holdout_missing():
    evidence = adapt_holdout_result(
        MockResult()
    )

    assert (
        evidence.status
        == EvidenceStatus.NOT_EVALUATED
    )


def test_holdout_none():
    evidence = adapt_holdout_result(
        None
    )

    assert (
        evidence.status
        == EvidenceStatus.NOT_EVALUATED
    )


def test_holdout_training_contamination_fails():
    result = MockResult(
        accuracy=0.99,
        holdout_used_for_training=True,
    )

    evidence = adapt_holdout_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.FAIL
    )


def test_holdout_selection_contamination_fails():
    result = MockResult(
        accuracy=0.99,
        holdout_used_for_selection=True,
    )

    evidence = adapt_holdout_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.FAIL
    )


# ---------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------


def test_calibration_pass():
    result = MockResult(
        calibrated_brier=0.12,
        calibrated_ece=0.08,
        passed=True,
    )

    evidence = adapt_calibration_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.PASS
    )

    assert (
        evidence.metrics[
            "brier_score"
        ]
        == 0.12
    )

    assert (
        evidence.metrics["ece"]
        == 0.08
    )


def test_calibration_can_derive_pass():
    result = MockResult(
        calibrated_brier=0.10,
        calibrated_ece=0.05,
    )

    evidence = adapt_calibration_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.PASS
    )


def test_calibration_failure():
    result = MockResult(
        calibrated_brier=0.30,
        calibrated_ece=0.20,
    )

    evidence = adapt_calibration_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.FAIL
    )


def test_calibration_missing():
    evidence = adapt_calibration_result(
        MockResult()
    )

    assert (
        evidence.status
        == EvidenceStatus.NOT_EVALUATED
    )


def test_calibration_holdout_use_fails():
    result = MockResult(
        calibrated_brier=0.10,
        calibrated_ece=0.05,
        final_holdout_used=True,
    )

    evidence = adapt_calibration_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.FAIL
    )


# ---------------------------------------------------------------------
# Range validation
# ---------------------------------------------------------------------


def test_range_pass():
    result = MockResult(
        coverage=0.82,
        passed=True,
    )

    evidence = adapt_range_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.PASS
    )

    assert (
        evidence.score
        == 0.82
    )


def test_range_can_derive_pass():
    result = MockResult(
        coverage=0.80,
    )

    evidence = adapt_range_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.PASS
    )


def test_range_failure():
    result = MockResult(
        coverage=0.55,
    )

    evidence = adapt_range_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.FAIL
    )


def test_range_missing():
    evidence = adapt_range_result(
        MockResult()
    )

    assert (
        evidence.status
        == EvidenceStatus.NOT_EVALUATED
    )


def test_range_holdout_use_fails():
    result = MockResult(
        coverage=0.85,
        final_holdout_used=True,
    )

    evidence = adapt_range_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.FAIL
    )


# ---------------------------------------------------------------------
# Regime
# ---------------------------------------------------------------------


def test_regime_pass():
    result = MockResult(
        stability_score=0.75,
        passed=True,
    )

    evidence = adapt_regime_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.PASS
    )

    assert (
        evidence.score
        == 0.75
    )


def test_regime_can_derive_pass():
    result = MockResult(
        stability_score=0.70,
    )

    evidence = adapt_regime_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.PASS
    )


def test_regime_failure():
    result = MockResult(
        stability_score=0.40,
    )

    evidence = adapt_regime_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.FAIL
    )


def test_regime_missing():
    evidence = adapt_regime_result(
        MockResult()
    )

    assert (
        evidence.status
        == EvidenceStatus.NOT_EVALUATED
    )


def test_regime_holdout_use_fails():
    result = MockResult(
        stability_score=0.80,
        final_holdout_used=True,
    )

    evidence = adapt_regime_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.FAIL
    )


# ---------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------


def test_backtest_pass():
    result = MockResult(
        profit_factor=1.8,
        sharpe=1.2,
        max_drawdown=-0.18,
        trades=100,
    )

    evidence = adapt_backtest_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.PASS
    )


def test_backtest_failure():
    result = MockResult(
        profit_factor=1.0,
        sharpe=0.5,
        max_drawdown=-0.40,
        trades=20,
    )

    evidence = adapt_backtest_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.FAIL
    )


def test_backtest_missing():
    evidence = adapt_backtest_result(
        MockResult()
    )

    assert (
        evidence.status
        == EvidenceStatus.NOT_EVALUATED
    )


def test_backtest_none():
    evidence = adapt_backtest_result(
        None
    )

    assert (
        evidence.status
        == EvidenceStatus.NOT_EVALUATED
    )


def test_backtest_trade_count_is_checked():
    result = MockResult(
        profit_factor=2.0,
        sharpe=1.5,
        max_drawdown=-0.10,
        trades=10,
    )

    evidence = adapt_backtest_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.FAIL
    )


# ---------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------


def test_robustness_pass():
    result = MockResult(
        robustness_score=0.80,
        passed=True,
    )

    evidence = adapt_robustness_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.PASS
    )

    assert (
        evidence.score
        == 0.80
    )


def test_robustness_can_derive_pass():
    result = MockResult(
        robustness_score=0.75,
    )

    evidence = adapt_robustness_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.PASS
    )


def test_robustness_failure():
    result = MockResult(
        robustness_score=0.40,
    )

    evidence = adapt_robustness_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.FAIL
    )


def test_robustness_missing():
    evidence = adapt_robustness_result(
        MockResult()
    )

    assert (
        evidence.status
        == EvidenceStatus.NOT_EVALUATED
    )


def test_robustness_holdout_use_fails():
    result = MockResult(
        robustness_score=0.90,
        final_holdout_used=True,
    )

    evidence = adapt_robustness_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.FAIL
    )


# ---------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------


def test_leakage_pass():
    result = MockResult(
        passed=True,
    )

    evidence = adapt_leakage_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.PASS
    )


def test_leakage_failure():
    result = MockResult(
        passed=False,
    )

    evidence = adapt_leakage_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.FAIL
    )


def test_leakage_free_alias():
    result = MockResult(
        leakage_free=True,
    )

    evidence = adapt_leakage_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.PASS
    )


def test_leakage_missing():
    evidence = adapt_leakage_result(
        MockResult()
    )

    assert (
        evidence.status
        == EvidenceStatus.NOT_EVALUATED
    )


# ---------------------------------------------------------------------
# Generic adapter
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "stage_name",
    [
        "walk_forward_validation",
        "final_holdout",
        "calibration",
        "range_validation",
        "regime_validation",
        "backtest",
        "robustness",
        "leakage_audit",
    ],
)
def test_generic_adapter_accepts_known_stages(
    stage_name,
):
    evidence = adapt_stage_result(
        stage_name,
        None,
    )

    assert evidence is not None


def test_generic_adapter_rejects_unknown_stage():
    with pytest.raises(ValueError):
        adapt_stage_result(
            "unknown_stage",
            None,
        )


# ---------------------------------------------------------------------
# Conservative behavior
# ---------------------------------------------------------------------


def test_ambiguous_validation_does_not_pass():
    result = MockResult(
        mean_accuracy=0.94,
    )

    evidence = adapt_walk_forward_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.NOT_EVALUATED
    )


def test_ambiguous_calibration_does_not_pass():
    result = MockResult(
        calibrated_brier=0.10,
    )

    evidence = adapt_calibration_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.NOT_EVALUATED
    )


def test_ambiguous_range_does_not_pass():
    result = MockResult(
        coverage=None,
    )

    evidence = adapt_range_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.NOT_EVALUATED
    )


def test_ambiguous_regime_does_not_pass():
    result = MockResult(
        stability_score=None,
    )

    evidence = adapt_regime_result(
        result
    )

    assert (
        evidence.status
        == EvidenceStatus.NOT_EVALUATED
    )


# ---------------------------------------------------------------------
# Critical evidence
# ---------------------------------------------------------------------


def test_all_adapted_evidence_is_critical():
    results = [
        adapt_walk_forward_result(
            None
        ),
        adapt_holdout_result(
            None
        ),
        adapt_calibration_result(
            None
        ),
        adapt_range_result(
            None
        ),
        adapt_regime_result(
            None
        ),
        adapt_backtest_result(
            None
        ),
        adapt_robustness_result(
            None
        ),
        adapt_leakage_result(
            None
        ),
    ]

    assert all(
        item.critical
        for item in results
    )
