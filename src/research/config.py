"""
AI Swing Analyser - Research Pipeline Configuration.

Central configuration for reproducible model research.

The configuration controls:
    - symbols
    - timeframes
    - prediction horizons
    - validation
    - feature selection
    - hyperparameter search
    - calibration
    - range modelling
    - backtesting
    - robustness
    - production approval

Important:
    Configuration values do not guarantee model performance.
    They define the research protocol used to evaluate performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


# ----------------------------------------------------------------------
# Default constants
# ----------------------------------------------------------------------

DEFAULT_TIMEFRAMES = (
    "4H",
    "1D",
    "1W",
    "1M",
)

DEFAULT_HORIZONS = (
    1,
    3,
    5,
    10,
    20,
)

DEFAULT_MARKET_SYMBOLS = {
    "NIFTY50": "^NSEI",
    "SENSEX": "^BSESN",
    "NIFTYBANK": "^NSEBANK",
}


# ----------------------------------------------------------------------
# Validation configuration
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ResearchValidationConfig:
    """
    Temporal validation policy.

    All validation must preserve chronological ordering.
    """

    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    test_fraction: float = 0.20

    n_walk_forward_splits: int = 5

    gap: int = 0
    embargo: int = 0

    expanding: bool = True

    minimum_train_samples: int = 100
    minimum_validation_samples: int = 30

    minimum_accuracy: float = 0.95

    maximum_generalization_gap: float = 0.10

    require_walk_forward: bool = True
    require_final_holdout: bool = True
    require_no_leakage: bool = True

    def __post_init__(self) -> None:
        fractions = (
            self.train_fraction,
            self.validation_fraction,
            self.test_fraction,
        )

        if any(
            fraction <= 0
            for fraction in fractions
        ):
            raise ValueError(
                "Validation fractions must be greater than zero."
            )

        total = sum(
            fractions
        )

        if abs(
            total - 1.0
        ) > 1e-9:
            raise ValueError(
                "Train, validation and test fractions must sum to 1.0."
            )

        if self.n_walk_forward_splits < 2:
            raise ValueError(
                "At least two walk-forward splits are required."
            )

        if self.gap < 0:
            raise ValueError(
                "gap cannot be negative."
            )

        if self.embargo < 0:
            raise ValueError(
                "embargo cannot be negative."
            )

        if self.minimum_train_samples < 1:
            raise ValueError(
                "minimum_train_samples must be positive."
            )

        if self.minimum_validation_samples < 1:
            raise ValueError(
                "minimum_validation_samples must be positive."
            )

        if not (
            0.5
            <= self.minimum_accuracy
            <= 1.0
        ):
            raise ValueError(
                "minimum_accuracy must be between 0.5 and 1.0."
            )

        if not (
            0.0
            <= self.maximum_generalization_gap
            <= 1.0
        ):
            raise ValueError(
                "maximum_generalization_gap must be between 0 and 1."
            )


# ----------------------------------------------------------------------
# Feature selection configuration
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ResearchFeatureConfig:
    """
    Feature-selection policy.

    Feature selection must be fitted only on development data.
    """

    enabled: bool = True

    max_missing_fraction: float = 0.40

    minimum_variance: float = 1e-12

    maximum_correlation: float = 0.95

    correlation_method: str = "spearman"

    minimum_features: int = 10

    maximum_features: int | None = None

    minimum_fold_frequency: float = 0.60

    minimum_importance: float = 0.0

    include_features: tuple[str, ...] = field(
        default_factory=tuple
    )

    exclude_features: tuple[str, ...] = field(
        default_factory=tuple
    )

    def __post_init__(self) -> None:
        if not (
            0.0
            <= self.max_missing_fraction
            < 1.0
        ):
            raise ValueError(
                "max_missing_fraction must be in [0, 1)."
            )

        if self.minimum_variance < 0:
            raise ValueError(
                "minimum_variance cannot be negative."
            )

        if not (
            0.0
            < self.maximum_correlation
            <= 1.0
        ):
            raise ValueError(
                "maximum_correlation must be in (0, 1]."
            )

        if self.minimum_features < 1:
            raise ValueError(
                "minimum_features must be positive."
            )

        if (
            self.maximum_features is not None
            and self.maximum_features
            < self.minimum_features
        ):
            raise ValueError(
                "maximum_features cannot be less than minimum_features."
            )

        if not (
            0.0
            <= self.minimum_fold_frequency
            <= 1.0
        ):
            raise ValueError(
                "minimum_fold_frequency must be between 0 and 1."
            )

        if self.minimum_importance < 0:
            raise ValueError(
                "minimum_importance cannot be negative."
            )

        allowed_methods = {
            "pearson",
            "spearman",
            "kendall",
        }

        if (
            self.correlation_method.lower()
            not in allowed_methods
        ):
            raise ValueError(
                "Unsupported correlation_method."
            )


# ----------------------------------------------------------------------
# Hyperparameter configuration
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ResearchHyperparameterConfig:
    """
    Controlled hyperparameter-search policy.

    The final holdout must never be used during this search.
    """

    enabled: bool = True

    max_trials: int = 50

    n_splits: int = 5

    gap: int = 0
    embargo: int = 0

    expanding: bool = True

    minimum_train_samples: int = 100
    minimum_validation_samples: int = 30

    minimum_accuracy: float = 0.95
    maximum_accuracy_std: float = 0.10

    accuracy_weight: float = 0.70
    stability_weight: float = 0.15
    brier_weight: float = 0.15

    def __post_init__(self) -> None:
        if self.max_trials < 1:
            raise ValueError(
                "max_trials must be positive."
            )

        if self.n_splits < 2:
            raise ValueError(
                "n_splits must be at least 2."
            )

        if self.gap < 0 or self.embargo < 0:
            raise ValueError(
                "gap and embargo cannot be negative."
            )

        if self.minimum_train_samples < 1:
            raise ValueError(
                "minimum_train_samples must be positive."
            )

        if self.minimum_validation_samples < 1:
            raise ValueError(
                "minimum_validation_samples must be positive."
            )

        if not (
            0.5
            <= self.minimum_accuracy
            <= 1.0
        ):
            raise ValueError(
                "minimum_accuracy must be between 0.5 and 1.0."
            )

        if not (
            0.0
            <= self.maximum_accuracy_std
            <= 1.0
        ):
            raise ValueError(
                "maximum_accuracy_std must be between 0 and 1."
            )

        weights = (
            self.accuracy_weight,
            self.stability_weight,
            self.brier_weight,
        )

        if any(
            weight < 0
            for weight in weights
        ):
            raise ValueError(
                "Hyperparameter objective weights cannot be negative."
            )

        if sum(
            weights
        ) <= 0:
            raise ValueError(
                "At least one hyperparameter objective weight is required."
            )


# ----------------------------------------------------------------------
# Calibration configuration
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ResearchCalibrationConfig:
    """
    Probability calibration policy.
    """

    enabled: bool = True

    method: str = "sigmoid"

    minimum_samples: int = 50

    maximum_brier: float = 0.25

    maximum_ece: float = 0.15

    high_confidence_threshold: float = 0.70

    n_bins: int = 10

    def __post_init__(self) -> None:
        if self.method.lower() not in {
            "sigmoid",
            "isotonic",
        }:
            raise ValueError(
                "Calibration method must be 'sigmoid' or 'isotonic'."
            )

        if self.minimum_samples < 10:
            raise ValueError(
                "minimum_samples must be at least 10."
            )

        if self.maximum_brier < 0:
            raise ValueError(
                "maximum_brier cannot be negative."
            )

        if not (
            0.0
            <= self.maximum_ece
            <= 1.0
        ):
            raise ValueError(
                "maximum_ece must be between 0 and 1."
            )

        if not (
            0.5
            <= self.high_confidence_threshold
            <= 1.0
        ):
            raise ValueError(
                "high_confidence_threshold must be between 0.5 and 1."
            )

        if self.n_bins < 2:
            raise ValueError(
                "n_bins must be at least 2."
            )


# ----------------------------------------------------------------------
# Range-model configuration
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ResearchRangeConfig:
    """
    Target-price interval configuration.
    """

    enabled: bool = True

    lower_quantile: float = 0.10
    median_quantile: float = 0.50
    upper_quantile: float = 0.90

    expected_coverage: float = 0.80

    minimum_coverage: float = 0.70
    maximum_coverage: float = 0.95

    minimum_samples: int = 100

    n_walk_forward_splits: int = 5

    gap: int = 0
    embargo: int = 0

    def __post_init__(self) -> None:
        quantiles = (
            self.lower_quantile,
            self.median_quantile,
            self.upper_quantile,
        )

        if not all(
            0.0 < value < 1.0
            for value in quantiles
        ):
            raise ValueError(
                "Range quantiles must be between 0 and 1."
            )

        if not (
            self.lower_quantile
            < self.median_quantile
            < self.upper_quantile
        ):
            raise ValueError(
                "Range quantiles must be ordered."
            )

        if not (
            0.0
            < self.minimum_coverage
            <= self.expected_coverage
            <= self.maximum_coverage
            <= 1.0
        ):
            raise ValueError(
                "Invalid range coverage thresholds."
            )

        if self.minimum_samples < 1:
            raise ValueError(
                "minimum_samples must be positive."
            )

        if self.n_walk_forward_splits < 2:
            raise ValueError(
                "At least two range-validation folds are required."
            )

        if self.gap < 0 or self.embargo < 0:
            raise ValueError(
                "gap and embargo cannot be negative."
            )


# ----------------------------------------------------------------------
# Backtest configuration
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ResearchBacktestConfig:
    """
    Walk-forward trading simulation configuration.
    """

    enabled: bool = True

    initial_capital: float = 100_000.0

    probability_threshold: float = 0.60

    long_enabled: bool = True
    short_enabled: bool = False

    risk_per_trade: float = 0.01

    transaction_cost: float = 0.001

    slippage: float = 0.0005

    stop_loss_fraction: float = 0.03
    take_profit_fraction: float = 0.06

    max_holding_period: int = 10

    max_concurrent_positions: int = 1

    minimum_risk_reward: float = 2.0

    n_walk_forward_splits: int = 5

    gap: int = 0
    embargo: int = 0

    expanding: bool = True

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError(
                "initial_capital must be positive."
            )

        if not (
            0.5
            <= self.probability_threshold
            <= 1.0
        ):
            raise ValueError(
                "probability_threshold must be between 0.5 and 1."
            )

        if not (
            0.0
            < self.risk_per_trade
            <= 0.05
        ):
            raise ValueError(
                "risk_per_trade must be in (0, 0.05]."
            )

        if self.transaction_cost < 0:
            raise ValueError(
                "transaction_cost cannot be negative."
            )

        if self.slippage < 0:
            raise ValueError(
                "slippage cannot be negative."
            )

        if not (
            0.0
            < self.stop_loss_fraction
            < 1.0
        ):
            raise ValueError(
                "stop_loss_fraction must be between 0 and 1."
            )

        if not (
            0.0
            < self.take_profit_fraction
            < 2.0
        ):
            raise ValueError(
                "take_profit_fraction must be between 0 and 2."
            )

        if self.max_holding_period < 1:
            raise ValueError(
                "max_holding_period must be positive."
            )

        if self.max_concurrent_positions < 1:
            raise ValueError(
                "max_concurrent_positions must be positive."
            )

        if self.minimum_risk_reward < 1.0:
            raise ValueError(
                "minimum_risk_reward should be at least 1.0."
            )

        if self.n_walk_forward_splits < 2:
            raise ValueError(
                "At least two backtest folds are required."
            )

        if self.gap < 0 or self.embargo < 0:
            raise ValueError(
                "gap and embargo cannot be negative."
            )


# ----------------------------------------------------------------------
# Robustness configuration
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ResearchRobustnessConfig:
    """
    Robustness and stress-testing policy.
    """

    enabled: bool = True

    simulations: int = 5_000

    cost_multipliers: tuple[float, ...] = (
        1.0,
        1.5,
        2.0,
    )

    slippage_multipliers: tuple[float, ...] = (
        1.0,
        1.5,
        2.0,
    )

    remove_best_fraction: float = 0.10
    remove_worst_fraction: float = 0.10

    minimum_profit_probability: float = 0.50
    maximum_drawdown: float = 0.30
    minimum_trades: int = 30

    minimum_robustness_score: float = 0.60

    def __post_init__(self) -> None:
        if self.simulations < 100:
            raise ValueError(
                "At least 100 robustness simulations are required."
            )

        if any(
            multiplier <= 0
            for multiplier in self.cost_multipliers
        ):
            raise ValueError(
                "Cost multipliers must be positive."
            )

        if any(
            multiplier <= 0
            for multiplier in self.slippage_multipliers
        ):
            raise ValueError(
                "Slippage multipliers must be positive."
            )

        for fraction in (
            self.remove_best_fraction,
            self.remove_worst_fraction,
        ):
            if not (
                0.0
                <= fraction
                < 1.0
            ):
                raise ValueError(
                    "Trade-removal fractions must be in [0, 1)."
                )

        if not (
            0.0
            <= self.minimum_profit_probability
            <= 1.0
        ):
            raise ValueError(
                "minimum_profit_probability must be between 0 and 1."
            )

        if not (
            0.0
            < self.maximum_drawdown
            < 1.0
        ):
            raise ValueError(
                "maximum_drawdown must be between 0 and 1."
            )

        if self.minimum_trades < 1:
            raise ValueError(
                "minimum_trades must be positive."
            )

        if not (
            0.0
            <= self.minimum_robustness_score
            <= 1.0
        ):
            raise ValueError(
                "minimum_robustness_score must be between 0 and 1."
            )


# ----------------------------------------------------------------------
# Approval configuration
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ResearchApprovalConfig:
    """
    Production approval policy.

    These are deliberately strict.

    A model cannot become production-approved merely because it has
    high in-sample accuracy.
    """

    enabled: bool = True

    minimum_validation_accuracy: float = 0.95

    minimum_final_holdout_accuracy: float = 0.95

    maximum_generalization_gap: float = 0.10

    maximum_calibration_brier: float = 0.25

    maximum_calibration_ece: float = 0.15

    minimum_range_coverage: float = 0.70
    maximum_range_coverage: float = 0.95

    minimum_regime_stability: float = 0.60

    minimum_profit_factor: float = 1.20
    minimum_sharpe: float = 0.80
    maximum_drawdown: float = 0.30
    minimum_trades: int = 30

    minimum_robustness_score: float = 0.60

    require_validation: bool = True
    require_final_holdout: bool = True
    require_no_leakage: bool = True
    require_calibration: bool = True
    require_range_validation: bool = True
    require_regime_validation: bool = True
    require_backtest: bool = True
    require_robustness: bool = True

    def __post_init__(self) -> None:
        if not (
            0.5
            <= self.minimum_validation_accuracy
            <= 1.0
        ):
            raise ValueError(
                "minimum_validation_accuracy must be between 0.5 and 1."
            )

        if not (
            0.5
            <= self.minimum_final_holdout_accuracy
            <= 1.0
        ):
            raise ValueError(
                "minimum_final_holdout_accuracy must be between 0.5 and 1."
            )

        if not (
            0.0
            <= self.maximum_generalization_gap
            <= 1.0
        ):
            raise ValueError(
                "maximum_generalization_gap must be between 0 and 1."
            )

        if self.maximum_calibration_brier < 0:
            raise ValueError(
                "maximum_calibration_brier cannot be negative."
            )

        if not (
            0.0
            <= self.maximum_calibration_ece
            <= 1.0
        ):
            raise ValueError(
                "maximum_calibration_ece must be between 0 and 1."
            )

        if not (
            0.0
            <= self.minimum_range_coverage
            <= self.maximum_range_coverage
            <= 1.0
        ):
            raise ValueError(
                "Invalid range coverage limits."
            )

        if not (
            0.0
            <= self.minimum_regime_stability
            <= 1.0
        ):
            raise ValueError(
                "minimum_regime_stability must be between 0 and 1."
            )

        if self.minimum_profit_factor < 0:
            raise ValueError(
                "minimum_profit_factor cannot be negative."
            )

        if self.minimum_sharpe < 0:
            raise ValueError(
                "minimum_sharpe cannot be negative."
            )

        if not (
            0.0
            < self.maximum_drawdown
            < 1.0
        ):
            raise ValueError(
                "maximum_drawdown must be between 0 and 1."
            )

        if self.minimum_trades < 1:
            raise ValueError(
                "minimum_trades must be positive."
            )

        if not (
            0.0
            <= self.minimum_robustness_score
            <= 1.0
        ):
            raise ValueError(
                "minimum_robustness_score must be between 0 and 1."
            )


# ----------------------------------------------------------------------
# Master research configuration
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ResearchPipelineConfig:
    """
    Master configuration for one reproducible research run.
    """

    symbol: str

    timeframe: str = "1D"

    horizons: tuple[int, ...] = DEFAULT_HORIZONS

    market_symbols: Mapping[
        str,
        str,
    ] = field(
        default_factory=lambda: dict(
            DEFAULT_MARKET_SYMBOLS
        )
    )

    random_state: int = 42

    validation: ResearchValidationConfig = field(
        default_factory=ResearchValidationConfig
    )

    features: ResearchFeatureConfig = field(
        default_factory=ResearchFeatureConfig
    )

    hyperparameter: ResearchHyperparameterConfig = field(
        default_factory=ResearchHyperparameterConfig
    )

    calibration: ResearchCalibrationConfig = field(
        default_factory=ResearchCalibrationConfig
    )

    range_model: ResearchRangeConfig = field(
        default_factory=ResearchRangeConfig
    )

    backtest: ResearchBacktestConfig = field(
        default_factory=ResearchBacktestConfig
    )

    robustness: ResearchRobustnessConfig = field(
        default_factory=ResearchRobustnessConfig
    )

    approval: ResearchApprovalConfig = field(
        default_factory=ResearchApprovalConfig
    )

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper()

        if not symbol:
            raise ValueError(
                "symbol cannot be empty."
            )

        if self.timeframe.upper() not in DEFAULT_TIMEFRAMES:
            raise ValueError(
                f"Unsupported timeframe: {self.timeframe}. "
                f"Supported: {DEFAULT_TIMEFRAMES}"
            )

        if not self.horizons:
            raise ValueError(
                "At least one prediction horizon is required."
            )

        normalized_horizons = tuple(
            sorted(
                set(
                    int(
                        horizon
                    )
                    for horizon in self.horizons
                )
            )
        )

        if any(
            horizon <= 0
            for horizon in normalized_horizons
        ):
            raise ValueError(
                "Prediction horizons must be positive."
            )

        object.__setattr__(
            self,
            "symbol",
            symbol,
        )

        object.__setattr__(
            self,
            "timeframe",
            self.timeframe.upper(),
        )

        object.__setattr__(
            self,
            "horizons",
            normalized_horizons,
        )

        if self.random_state < 0:
            raise ValueError(
                "random_state cannot be negative."
            )

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_mapping(
        cls,
        symbol: str,
        config: Mapping[str, Any] | None = None,
        *,
        timeframe: str | None = None,
        horizons: Sequence[int] | None = None,
    ) -> "ResearchPipelineConfig":
        """
        Construct configuration from a YAML-style mapping.

        Missing values fall back to conservative defaults.
        """

        config = config or {}

        prediction = config.get(
            "prediction",
            {},
        )

        validation = config.get(
            "validation",
            {},
        )

        risk = config.get(
            "risk",
            {},
        )

        return cls(
            symbol=symbol,
            timeframe=(
                timeframe
                or config.get(
                    "timeframe",
                    "1D",
                )
            ),
            horizons=tuple(
                horizons
                if horizons is not None
                else prediction.get(
                    "horizons",
                    DEFAULT_HORIZONS,
                )
            ),
            market_symbols=config.get(
                "market_symbols",
                DEFAULT_MARKET_SYMBOLS,
            ),
            random_state=int(
                config.get(
                    "random_state",
                    42,
                )
            ),
            validation=ResearchValidationConfig(
                train_fraction=float(
                    validation.get(
                        "train_fraction",
                        0.60,
                    )
                ),
                validation_fraction=float(
                    validation.get(
                        "validation_fraction",
                        0.20,
                    )
                ),
                test_fraction=float(
                    validation.get(
                        "test_fraction",
                        0.20,
                    )
                ),
                n_walk_forward_splits=int(
                    validation.get(
                        "n_walk_forward_splits",
                        5,
                    )
                ),
                gap=int(
                    validation.get(
                        "gap",
                        0,
                    )
                ),
                embargo=int(
                    validation.get(
                        "embargo",
                        0,
                    )
                ),
                expanding=bool(
                    validation.get(
                        "expanding",
                        True,
                    )
                ),
                minimum_train_samples=int(
                    validation.get(
                        "minimum_train_samples",
                        100,
                    )
                ),
                minimum_validation_samples=int(
                    validation.get(
                        "minimum_validation_samples",
                        30,
                    )
                ),
                minimum_accuracy=float(
                    validation.get(
                        "minimum_accuracy",
                        0.95,
                    )
                ),
                maximum_generalization_gap=float(
                    validation.get(
                        "maximum_generalization_gap",
                        0.10,
                    )
                ),
                require_walk_forward=bool(
                    validation.get(
                        "require_walk_forward",
                        True,
                    )
                ),
                require_final_holdout=bool(
                    validation.get(
                        "require_final_holdout",
                        True,
                    )
                ),
                require_no_leakage=bool(
                    validation.get(
                        "require_no_data_leakage",
                    validation.get(
                        "require_no_leakage",
                        True,
                    ),
                    )
                ),
            ),
            backtest=ResearchBacktestConfig(
                initial_capital=float(
                    config.get(
                        "initial_capital",
                        100_000.0,
                    )
                ),
                probability_threshold=float(
                    config.get(
                        "probability_threshold",
                        0.60,
                    )
                ),
                long_enabled=bool(
                    config.get(
                        "long_enabled",
                        True,
                    )
                ),
                short_enabled=bool(
                    config.get(
                        "short_enabled",
                        False,
                    )
                ),
                risk_per_trade=float(
                    risk.get(
                        "max_risk_per_trade",
                        0.01,
                    )
                ),
                minimum_risk_reward=float(
                    risk.get(
                        "minimum_risk_reward",
                        2.0,
                    )
                ),
            ),
            approval=ResearchApprovalConfig(
                minimum_validation_accuracy=float(
                    validation.get(
                        "minimum_accuracy",
                        0.95,
                    )
                ),
                minimum_final_holdout_accuracy=float(
                    validation.get(
                        "minimum_accuracy",
                        0.95,
                    )
                ),
                maximum_generalization_gap=float(
                    validation.get(
                        "maximum_generalization_gap",
                        0.10,
                    )
                ),
            ),
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON/YAML-friendly representation.
        """

        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "horizons": list(
                self.horizons
            ),
            "market_symbols": dict(
                self.market_symbols
            ),
            "random_state": self.random_state,
            "validation": {
                key: value
                for key, value in vars(
                    self.validation
                ).items()
            },
            "features": {
                key: value
                for key, value in vars(
                    self.features
                ).items()
            },
            "hyperparameter": {
                key: value
                for key, value in vars(
                    self.hyperparameter
                ).items()
            },
            "calibration": {
                key: value
                for key, value in vars(
                    self.calibration
                ).items()
            },
            "range_model": {
                key: value
                for key, value in vars(
                    self.range_model
                ).items()
            },
            "backtest": {
                key: value
                for key, value in vars(
                    self.backtest
                ).items()
            },
            "robustness": {
                key: value
                for key, value in vars(
                    self.robustness
                ).items()
            },
            "approval": {
                key: value
                for key, value in vars(
                    self.approval
                ).items()
            },
        }

    def summary(self) -> dict[str, Any]:
        """
        Compact human-readable configuration summary.
        """

        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "horizons": self.horizons,
            "random_state": self.random_state,
            "minimum_accuracy": (
                self.validation.minimum_accuracy
            ),
            "final_holdout_required": (
                self.validation.require_final_holdout
            ),
            "walk_forward_required": (
                self.validation.require_walk_forward
            ),
            "feature_selection_enabled": (
                self.features.enabled
            ),
            "hyperparameter_search_enabled": (
                self.hyperparameter.enabled
            ),
            "calibration_enabled": (
                self.calibration.enabled
            ),
            "range_model_enabled": (
                self.range_model.enabled
            ),
            "backtest_enabled": (
                self.backtest.enabled
            ),
            "robustness_enabled": (
                self.robustness.enabled
            ),
            "production_approval_enabled": (
                self.approval.enabled
            ),
        }


__all__ = [
    "DEFAULT_TIMEFRAMES",
    "DEFAULT_HORIZONS",
    "DEFAULT_MARKET_SYMBOLS",
    "ResearchValidationConfig",
    "ResearchFeatureConfig",
    "ResearchHyperparameterConfig",
    "ResearchCalibrationConfig",
    "ResearchRangeConfig",
    "ResearchBacktestConfig",
    "ResearchRobustnessConfig",
    "ResearchApprovalConfig",
    "ResearchPipelineConfig",
]
