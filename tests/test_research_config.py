"""
Tests for the integrated research configuration.

These tests protect the research system from accidentally weakening
or corrupting its validation and production-safety thresholds.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.research.research_config import (
    IntegratedApprovalConfig,
    IntegratedBacktestConfig,
    IntegratedCalibrationConfig,
    IntegratedHoldoutConfig,
    IntegratedRangeConfig,
    IntegratedRegimeConfig,
    IntegratedResearchConfig,
    IntegratedRobustnessConfig,
    IntegratedTradingConfig,
    IntegratedValidationConfig,
    default_integrated_research_config,
)


# ---------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------


def test_default_configuration_is_valid():

    config = default_integrated_research_config()

    config.validate()

    assert isinstance(
        config,
        IntegratedResearchConfig,
    )


def test_default_configuration_is_strict():

    config = default_integrated_research_config()

    assert config.validation.minimum_accuracy == 0.95
    assert config.holdout.minimum_accuracy == 0.95

    assert (
        config.validation.maximum_generalization_gap
        == 0.10
    )

    assert (
        config.regime.minimum_stability_score
        == 0.60
    )

    assert (
        config.backtest.minimum_profit_factor
        == 1.20
    )

    assert config.backtest.minimum_sharpe == 0.80
    assert config.backtest.maximum_drawdown == 0.30

    assert (
        config.robustness.minimum_score
        == 0.60
    )


def test_default_timeframes():

    config = default_integrated_research_config()

    assert config.primary_timeframe == "1D"

    assert config.supported_timeframes == (
        "4H",
        "1D",
        "1W",
        "1M",
    )


def test_default_prediction_horizons():

    config = default_integrated_research_config()

    assert config.prediction_horizons == (
        1,
        3,
        5,
        10,
        20,
    )


# ---------------------------------------------------------------------
# Validation configuration
# ---------------------------------------------------------------------


def test_validation_accuracy_must_be_valid():

    config = IntegratedResearchConfig(
        validation=IntegratedValidationConfig(
            minimum_accuracy=1.1
        )
    )

    with pytest.raises(ValueError):
        config.validate()


def test_negative_generalization_gap_is_rejected():

    validation = IntegratedValidationConfig(
        maximum_generalization_gap=-0.01
    )

    config = IntegratedResearchConfig(
        validation=validation
    )

    with pytest.raises(ValueError):
        config.validate()


def test_walk_forward_requires_multiple_folds():

    validation = IntegratedValidationConfig(
        minimum_walk_forward_folds=1
    )

    config = IntegratedResearchConfig(
        validation=validation
    )

    with pytest.raises(ValueError):
        config.validate()


def test_training_sample_requirement_must_be_positive():

    validation = IntegratedValidationConfig(
        minimum_train_samples=0
    )

    config = IntegratedResearchConfig(
        validation=validation
    )

    with pytest.raises(ValueError):
        config.validate()


# ---------------------------------------------------------------------
# Calibration configuration
# ---------------------------------------------------------------------


def test_calibration_method_is_restricted():

    calibration = IntegratedCalibrationConfig(
        method="random"
    )

    config = IntegratedResearchConfig(
        calibration=calibration
    )

    with pytest.raises(ValueError):
        config.validate()


def test_calibration_thresholds_are_validated():

    calibration = IntegratedCalibrationConfig(
        maximum_brier=1.5
    )

    config = IntegratedResearchConfig(
        calibration=calibration
    )

    with pytest.raises(ValueError):
        config.validate()


def test_high_confidence_threshold_must_be_between_zero_and_one():

    calibration = IntegratedCalibrationConfig(
        high_confidence_threshold=1.0
    )

    config = IntegratedResearchConfig(
        calibration=calibration
    )

    with pytest.raises(ValueError):
        config.validate()


# ---------------------------------------------------------------------
# Range configuration
# ---------------------------------------------------------------------


def test_range_quantiles_must_be_ordered():

    range_config = IntegratedRangeConfig(
        lower_quantile=0.50,
        median_quantile=0.20,
        upper_quantile=0.90,
    )

    config = IntegratedResearchConfig(
        range_model=range_config
    )

    with pytest.raises(ValueError):
        config.validate()


def test_range_coverage_bounds_must_be_ordered():

    range_config = IntegratedRangeConfig(
        minimum_coverage=0.90,
        maximum_coverage=0.70,
    )

    config = IntegratedResearchConfig(
        range_model=range_config
    )

    with pytest.raises(ValueError):
        config.validate()


# ---------------------------------------------------------------------
# Regime configuration
# ---------------------------------------------------------------------


def test_regime_stability_must_be_between_zero_and_one():

    regime = IntegratedRegimeConfig(
        minimum_stability_score=1.5
    )

    config = IntegratedResearchConfig(
        regime=regime
    )

    with pytest.raises(ValueError):
        config.validate()


# ---------------------------------------------------------------------
# Backtest configuration
# ---------------------------------------------------------------------


def test_backtest_capital_must_be_positive():

    backtest = IntegratedBacktestConfig(
        initial_capital=0
    )

    config = IntegratedResearchConfig(
        backtest=backtest
    )

    with pytest.raises(ValueError):
        config.validate()


def test_backtest_probability_threshold_must_be_valid():

    backtest = IntegratedBacktestConfig(
        probability_threshold=-0.1
    )

    config = IntegratedResearchConfig(
        backtest=backtest
    )

    with pytest.raises(ValueError):
        config.validate()


def test_backtest_trade_count_must_be_positive():

    backtest = IntegratedBacktestConfig(
        minimum_trades=0
    )

    config = IntegratedResearchConfig(
        backtest=backtest
    )

    with pytest.raises(ValueError):
        config.validate()


# ---------------------------------------------------------------------
# Robustness configuration
# ---------------------------------------------------------------------


def test_robustness_requires_sufficient_simulations():

    robustness = IntegratedRobustnessConfig(
        simulations=50
    )

    config = IntegratedResearchConfig(
        robustness=robustness
    )

    with pytest.raises(ValueError):
        config.validate()


def test_robustness_score_must_be_valid():

    robustness = IntegratedRobustnessConfig(
        minimum_score=1.2
    )

    config = IntegratedResearchConfig(
        robustness=robustness
    )

    with pytest.raises(ValueError):
        config.validate()


# ---------------------------------------------------------------------
# Holdout configuration
# ---------------------------------------------------------------------


def test_holdout_fraction_must_be_between_zero_and_one():

    holdout = IntegratedHoldoutConfig(
        test_fraction=1.0
    )

    config = IntegratedResearchConfig(
        holdout=holdout
    )

    with pytest.raises(ValueError):
        config.validate()


def test_holdout_accuracy_must_be_valid():

    holdout = IntegratedHoldoutConfig(
        minimum_accuracy=-0.1
    )

    config = IntegratedResearchConfig(
        holdout=holdout
    )

    with pytest.raises(ValueError):
        config.validate()


def test_holdout_protection_defaults_are_enabled():

    config = default_integrated_research_config()

    assert config.holdout.must_be_untouched
    assert config.holdout.forbid_model_selection
    assert config.holdout.forbid_hyperparameter_tuning
    assert config.holdout.forbid_feature_selection
    assert config.holdout.forbid_calibration


# ---------------------------------------------------------------------
# Approval configuration
# ---------------------------------------------------------------------


def test_approval_accuracy_thresholds_are_strict():

    approval = IntegratedApprovalConfig(
        minimum_validation_accuracy=1.1
    )

    config = IntegratedResearchConfig(
        approval=approval
    )

    with pytest.raises(ValueError):
        config.validate()


def test_approval_trade_count_must_be_positive():

    approval = IntegratedApprovalConfig(
        minimum_trades=0
    )

    config = IntegratedResearchConfig(
        approval=approval
    )

    with pytest.raises(ValueError):
        config.validate()


def test_approval_required_gates_are_enabled():

    config = default_integrated_research_config()

    assert config.approval.require_validation
    assert config.approval.require_holdout
    assert config.approval.require_calibration
    assert config.approval.require_range_validation
    assert config.approval.require_regime_validation
    assert config.approval.require_backtest
    assert config.approval.require_robustness
    assert config.approval.require_leakage_audit


# ---------------------------------------------------------------------
# Trading configuration
# ---------------------------------------------------------------------


def test_trading_probability_thresholds_are_ordered():

    trading = IntegratedTradingConfig(
        minimum_probability=0.80,
        high_confidence_probability=0.70,
    )

    config = IntegratedResearchConfig(
        trading=trading
    )

    with pytest.raises(ValueError):
        config.validate()


def test_trading_risk_reward_must_be_positive():

    trading = IntegratedTradingConfig(
        minimum_risk_reward=0
    )

    config = IntegratedResearchConfig(
        trading=trading
    )

    with pytest.raises(ValueError):
        config.validate()


def test_trading_risk_must_be_valid():

    trading = IntegratedTradingConfig(
        maximum_risk_per_trade=1.5
    )

    config = IntegratedResearchConfig(
        trading=trading
    )

    with pytest.raises(ValueError):
        config.validate()


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------


def test_to_dict_contains_all_major_sections():

    config = default_integrated_research_config()

    data = config.to_dict()

    expected_sections = {
        "validation",
        "calibration",
        "range_model",
        "regime",
        "backtest",
        "robustness",
        "holdout",
        "approval",
        "trading",
    }

    assert expected_sections.issubset(
        data.keys()
    )


def test_to_dict_does_not_mutate_configuration():

    config = default_integrated_research_config()

    before = config.to_dict()

    result = config.to_dict()

    after = config.to_dict()

    assert result == before
    assert after == before


def test_summary_contains_core_gates():

    config = default_integrated_research_config()

    summary = config.summary()

    assert (
        summary["minimum_validation_accuracy"]
        == 0.95
    )

    assert (
        summary["minimum_holdout_accuracy"]
        == 0.95
    )

    assert (
        summary["minimum_profit_factor"]
        == 1.20
    )

    assert summary["research_only"] is True


# ---------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------


def test_configuration_is_frozen():

    config = default_integrated_research_config()

    with pytest.raises(
        AttributeError
    ):
        config.primary_timeframe = "4H"


def test_nested_configuration_is_frozen():

    config = default_integrated_research_config()

    with pytest.raises(
        AttributeError
    ):
        config.validation.minimum_accuracy = 0.90


# ---------------------------------------------------------------------
# Safe modification pattern
# ---------------------------------------------------------------------


def test_replace_can_create_explicit_research_variant():

    config = default_integrated_research_config()

    modified_validation = replace(
        config.validation,
        minimum_accuracy=0.90,
    )

    modified = replace(
        config,
        validation=modified_validation,
    )

    modified.validate()

    assert (
        modified.validation.minimum_accuracy
        == 0.90
    )

    assert (
        config.validation.minimum_accuracy
        == 0.95
    )


# ---------------------------------------------------------------------
# Default factory
# ---------------------------------------------------------------------


def test_factory_returns_independent_configurations():

    first = default_integrated_research_config()
    second = default_integrated_research_config()

    assert first == second
    assert first is not second


# ---------------------------------------------------------------------
# Safety principle
# ---------------------------------------------------------------------


def test_configuration_does_not_claim_production_readiness():

    config = default_integrated_research_config()

    summary = config.summary()

    assert summary["research_only"] is True
