"""
AI Swing Analyser — Integrated Research Configuration.

Central configuration for the complete research workflow.

The purpose of this module is to ensure that all research stages
use one consistent set of safety thresholds.

Important:

    Configuration defines gates.
    Configuration does NOT prove that a model passes those gates.

Actual evidence must still be produced by independent evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


# ---------------------------------------------------------------------
# Validation configuration
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class IntegratedValidationConfig:
    """
    Configuration for temporal model validation.
    """

    minimum_accuracy: float = 0.95

    maximum_accuracy_std: float = 0.10

    maximum_generalization_gap: float = 0.10

    require_walk_forward: bool = True

    minimum_walk_forward_folds: int = 5

    gap: int = 0

    embargo: int = 0

    expanding: bool = True

    minimum_train_samples: int = 100

    minimum_validation_samples: int = 30


# ---------------------------------------------------------------------
# Calibration configuration
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class IntegratedCalibrationConfig:
    """
    Configuration for probability calibration.
    """

    method: str = "sigmoid"

    minimum_samples: int = 50

    maximum_brier: float = 0.25

    maximum_ece: float = 0.15

    high_confidence_threshold: float = 0.70


# ---------------------------------------------------------------------
# Range configuration
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class IntegratedRangeConfig:
    """
    Configuration for probabilistic target ranges.
    """

    lower_quantile: float = 0.10

    median_quantile: float = 0.50

    upper_quantile: float = 0.90

    expected_coverage: float = 0.80

    minimum_coverage: float = 0.70

    maximum_coverage: float = 0.95

    minimum_samples: int = 100


# ---------------------------------------------------------------------
# Regime configuration
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class IntegratedRegimeConfig:
    """
    Configuration for regime stability.
    """

    minimum_stability_score: float = 0.60

    minimum_regime_samples: int = 30

    maximum_accuracy_std: float = 0.15

    require_all_regimes: bool = False


# ---------------------------------------------------------------------
# Backtest configuration
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class IntegratedBacktestConfig:
    """
    Configuration for research backtesting.
    """

    minimum_profit_factor: float = 1.20

    minimum_sharpe: float = 0.80

    maximum_drawdown: float = 0.30

    minimum_trades: int = 30

    probability_threshold: float = 0.60

    transaction_cost: float = 0.001

    slippage: float = 0.0005

    maximum_holding_period: int = 10

    initial_capital: float = 100000.0


# ---------------------------------------------------------------------
# Robustness configuration
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class IntegratedRobustnessConfig:
    """
    Configuration for robustness and stress testing.
    """

    simulations: int = 5000

    minimum_profit_probability: float = 0.50

    maximum_drawdown: float = 0.30

    minimum_trades: int = 30

    minimum_score: float = 0.60

    transaction_cost: float = 0.001

    slippage: float = 0.0005

    cost_multipliers: tuple[float, ...] = (
        1.5,
        2.0,
        3.0,
    )

    slippage_multipliers: tuple[float, ...] = (
        1.5,
        2.0,
        3.0,
    )

    trade_removal_fraction: float = 0.10

    random_state: int = 42


# ---------------------------------------------------------------------
# Holdout configuration
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class IntegratedHoldoutConfig:
    """
    Configuration for the untouched final holdout.
    """

    minimum_accuracy: float = 0.95

    test_fraction: float = 0.20

    must_be_untouched: bool = True

    forbid_model_selection: bool = True

    forbid_hyperparameter_tuning: bool = True

    forbid_feature_selection: bool = True

    forbid_calibration: bool = True


# ---------------------------------------------------------------------
# Approval configuration
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class IntegratedApprovalConfig:
    """
    Final production approval thresholds.

    These are deliberately conservative.
    """

    minimum_validation_accuracy: float = 0.95

    minimum_holdout_accuracy: float = 0.95

    maximum_generalization_gap: float = 0.10

    maximum_brier: float = 0.25

    maximum_ece: float = 0.15

    minimum_range_coverage: float = 0.70

    maximum_range_coverage: float = 0.95

    minimum_regime_stability: float = 0.60

    minimum_profit_factor: float = 1.20

    minimum_sharpe: float = 0.80

    maximum_drawdown: float = 0.30

    minimum_trades: int = 30

    minimum_robustness_score: float = 0.60

    require_validation: bool = True

    require_holdout: bool = True

    require_calibration: bool = True

    require_range_validation: bool = True

    require_regime_validation: bool = True

    require_backtest: bool = True

    require_robustness: bool = True

    require_leakage_audit: bool = True


# ---------------------------------------------------------------------
# Trading configuration
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class IntegratedTradingConfig:
    """
    Configuration for the eventual swing-trading output.
    """

    minimum_probability: float = 0.60

    high_confidence_probability: float = 0.70

    maximum_model_disagreement: float = 0.25

    minimum_model_agreement: float = 0.60

    minimum_mtf_alignment: float = 0.60

    minimum_risk_reward: float = 2.0

    maximum_risk_per_trade: float = 0.01

    maximum_capital_allocation: float = 0.25

    long_enabled: bool = True

    short_enabled: bool = False


# ---------------------------------------------------------------------
# Master configuration
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class IntegratedResearchConfig:
    """
    Master configuration for the complete research system.
    """

    project_name: str = (
        "AI Swing Analyser"
    )

    version: str = "0.1.0"

    country: str = "India"

    timezone: str = "Asia/Kolkata"

    primary_timeframe: str = "1D"

    supported_timeframes: tuple[str, ...] = (
        "4H",
        "1D",
        "1W",
        "1M",
    )

    prediction_horizons: tuple[int, ...] = (
        1,
        3,
        5,
        10,
        20,
    )

    validation: IntegratedValidationConfig = (
        field(
            default_factory=(
                IntegratedValidationConfig
            )
        )
    )

    calibration: IntegratedCalibrationConfig = (
        field(
            default_factory=(
                IntegratedCalibrationConfig
            )
        )
    )

    range_model: IntegratedRangeConfig = (
        field(
            default_factory=(
                IntegratedRangeConfig
            )
        )
    )

    regime: IntegratedRegimeConfig = (
        field(
            default_factory=(
                IntegratedRegimeConfig
            )
        )
    )

    backtest: IntegratedBacktestConfig = (
        field(
            default_factory=(
                IntegratedBacktestConfig
            )
        )
    )

    robustness: IntegratedRobustnessConfig = (
        field(
            default_factory=(
                IntegratedRobustnessConfig
            )
        )
    )

    holdout: IntegratedHoldoutConfig = (
        field(
            default_factory=(
                IntegratedHoldoutConfig
            )
        )
    )

    approval: IntegratedApprovalConfig = (
        field(
            default_factory=(
                IntegratedApprovalConfig
            )
        )
    )

    trading: IntegratedTradingConfig = (
        field(
            default_factory=(
                IntegratedTradingConfig
            )
        )
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    def validate(self) -> None:

        if not self.project_name.strip():
            raise ValueError(
                "project_name cannot be empty."
            )

        if not self.timezone.strip():
            raise ValueError(
                "timezone cannot be empty."
            )

        if (
            self.primary_timeframe
            not in self.supported_timeframes
        ):
            raise ValueError(
                "primary_timeframe must be one "
                "of supported_timeframes."
            )

        if not self.prediction_horizons:
            raise ValueError(
                "At least one prediction horizon "
                "is required."
            )

        if any(
            horizon <= 0
            for horizon
            in self.prediction_horizons
        ):
            raise ValueError(
                "Prediction horizons must be positive."
            )

        if (
            len(
                set(
                    self.prediction_horizons
                )
            )
            != len(
                self.prediction_horizons
            )
        ):
            raise ValueError(
                "Prediction horizons must be unique."
            )

        if not (
            0.0
            < self.validation.minimum_accuracy
            <= 1.0
        ):
            raise ValueError(
                "minimum_accuracy must be between 0 and 1."
            )

        if (
            self.validation.maximum_accuracy_std
            < 0
        ):
            raise ValueError(
                "maximum_accuracy_std cannot be negative."
            )

        if (
            self.validation.maximum_generalization_gap
            < 0
        ):
            raise ValueError(
                "maximum_generalization_gap cannot be negative."
            )

        if (
            self.validation.minimum_walk_forward_folds
            < 2
        ):
            raise ValueError(
                "At least two walk-forward folds are required."
            )

        if (
            self.validation.minimum_train_samples
            < 1
        ):
            raise ValueError(
                "minimum_train_samples must be positive."
            )

        if (
            self.validation.minimum_validation_samples
            < 1
        ):
            raise ValueError(
                "minimum_validation_samples must be positive."
            )

        if self.calibration.method not in {
            "sigmoid",
            "isotonic",
        }:
            raise ValueError(
                "Calibration method must be "
                "'sigmoid' or 'isotonic'."
            )

        if (
            self.calibration.minimum_samples
            < 1
        ):
            raise ValueError(
                "Calibration minimum_samples must be positive."
            )

        if not (
            0.0
            <= self.calibration.maximum_brier
            <= 1.0
        ):
            raise ValueError(
                "maximum_brier must be between 0 and 1."
            )

        if not (
            0.0
            <= self.calibration.maximum_ece
            <= 1.0
        ):
            raise ValueError(
                "maximum_ece must be between 0 and 1."
            )

        if not (
            0.0
            < self.calibration.high_confidence_threshold
            < 1.0
        ):
            raise ValueError(
                "high_confidence_threshold must be "
                "between 0 and 1."
            )

        quantiles = (
            self.range_model.lower_quantile,
            self.range_model.median_quantile,
            self.range_model.upper_quantile,
        )

        if not (
            0.0
            < quantiles[0]
            < quantiles[1]
            < quantiles[2]
            < 1.0
        ):
            raise ValueError(
                "Range quantiles must satisfy "
                "0 < lower < median < upper < 1."
            )

        if not (
            0.0
            < self.range_model.minimum_coverage
            <= self.range_model.maximum_coverage
            < 1.0
        ):
            raise ValueError(
                "Range coverage thresholds are invalid."
            )

        if (
            self.range_model.minimum_samples
            < 1
        ):
            raise ValueError(
                "Range minimum_samples must be positive."
            )

        if (
            self.regime.minimum_stability_score
            < 0
            or self.regime.minimum_stability_score
            > 1
        ):
            raise ValueError(
                "Regime stability must be between 0 and 1."
            )

        if (
            self.regime.minimum_regime_samples
            < 1
        ):
            raise ValueError(
                "minimum_regime_samples must be positive."
            )

        if (
            self.backtest.minimum_profit_factor
            < 0
        ):
            raise ValueError(
                "minimum_profit_factor cannot be negative."
            )

        if (
            self.backtest.minimum_trades
            < 1
        ):
            raise ValueError(
                "minimum_trades must be positive."
            )

        if (
            self.backtest.initial_capital
            <= 0
        ):
            raise ValueError(
                "initial_capital must be positive."
            )

        if not (
            0.0
            <= self.backtest.probability_threshold
            <= 1.0
        ):
            raise ValueError(
                "backtest probability threshold "
                "must be between 0 and 1."
            )

        if (
            self.robustness.simulations
            < 100
        ):
            raise ValueError(
                "At least 100 robustness simulations are required."
            )

        if (
            self.robustness.minimum_score
            < 0
            or self.robustness.minimum_score
            > 1
        ):
            raise ValueError(
                "minimum robustness score must "
                "be between 0 and 1."
            )

        if (
            self.robustness.minimum_trades
            < 1
        ):
            raise ValueError(
                "Robustness minimum_trades must be positive."
            )

        if not (
            0.0
            < self.holdout.test_fraction
            < 1.0
        ):
            raise ValueError(
                "Holdout test_fraction must be "
                "between 0 and 1."
            )

        if (
            self.holdout.minimum_accuracy
            < 0
            or self.holdout.minimum_accuracy
            > 1
        ):
            raise ValueError(
                "Holdout minimum_accuracy must "
                "be between 0 and 1."
            )

        if (
            self.approval.minimum_validation_accuracy
            < 0
            or self.approval.minimum_validation_accuracy
            > 1
        ):
            raise ValueError(
                "Approval validation accuracy must "
                "be between 0 and 1."
            )

        if (
            self.approval.minimum_holdout_accuracy
            < 0
            or self.approval.minimum_holdout_accuracy
            > 1
        ):
            raise ValueError(
                "Approval holdout accuracy must "
                "be between 0 and 1."
            )

        if (
            self.approval.minimum_trades
            < 1
        ):
            raise ValueError(
                "Approval minimum_trades must be positive."
            )

        if (
            self.approval.minimum_robustness_score
            < 0
            or self.approval.minimum_robustness_score
            > 1
        ):
            raise ValueError(
                "Approval robustness score must "
                "be between 0 and 1."
            )

        if not (
            0.0
            < self.trading.minimum_probability
            <= 1.0
        ):
            raise ValueError(
                "minimum_probability must be "
                "between 0 and 1."
            )

        if not (
            self.trading.minimum_probability
            <= self.trading.high_confidence_probability
            <= 1.0
        ):
            raise ValueError(
                "High confidence probability must "
                "be >= minimum probability."
            )

        if not (
            0.0
            <= self.trading.maximum_model_disagreement
            <= 1.0
        ):
            raise ValueError(
                "maximum_model_disagreement must "
                "be between 0 and 1."
            )

        if not (
            0.0
            <= self.trading.minimum_model_agreement
            <= 1.0
        ):
            raise ValueError(
                "minimum_model_agreement must "
                "be between 0 and 1."
            )

        if not (
            0.0
            <= self.trading.minimum_mtf_alignment
            <= 1.0
        ):
            raise ValueError(
                "minimum_mtf_alignment must "
                "be between 0 and 1."
            )

        if (
            self.trading.minimum_risk_reward
            <= 0
        ):
            raise ValueError(
                "minimum_risk_reward must be positive."
            )

        if not (
            0.0
            < self.trading.maximum_risk_per_trade
            <= 1.0
        ):
            raise ValueError(
                "maximum_risk_per_trade must "
                "be between 0 and 1."
            )

        if not (
            0.0
            < self.trading.maximum_capital_allocation
            <= 1.0
        ):
            raise ValueError(
                "maximum_capital_allocation must "
                "be between 0 and 1."
            )

    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:

        self.validate()

        return asdict(
            self
        )

    def summary(self) -> dict[str, Any]:

        self.validate()

        return {
            "project": self.project_name,
            "version": self.version,
            "primary_timeframe": (
                self.primary_timeframe
            ),
            "supported_timeframes": list(
                self.supported_timeframes
            ),
            "prediction_horizons": list(
                self.prediction_horizons
            ),
            "minimum_validation_accuracy": (
                self.validation.minimum_accuracy
            ),
            "minimum_holdout_accuracy": (
                self.holdout.minimum_accuracy
            ),
            "minimum_regime_stability": (
                self.regime.minimum_stability_score
            ),
            "minimum_profit_factor": (
                self.backtest.minimum_profit_factor
            ),
            "minimum_sharpe": (
                self.backtest.minimum_sharpe
            ),
            "maximum_drawdown": (
                self.backtest.maximum_drawdown
            ),
            "minimum_robustness_score": (
                self.robustness.minimum_score
            ),
            "minimum_trade_probability": (
                self.trading.minimum_probability
            ),
            "research_only": True,
        }


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------


def default_integrated_research_config(
) -> IntegratedResearchConfig:

    config = IntegratedResearchConfig()

    config.validate()

    return config


__all__ = [
    "IntegratedValidationConfig",
    "IntegratedCalibrationConfig",
    "IntegratedRangeConfig",
    "IntegratedRegimeConfig",
    "IntegratedBacktestConfig",
    "IntegratedRobustnessConfig",
    "IntegratedHoldoutConfig",
    "IntegratedApprovalConfig",
    "IntegratedTradingConfig",
    "IntegratedResearchConfig",
    "default_integrated_research_config",
]
