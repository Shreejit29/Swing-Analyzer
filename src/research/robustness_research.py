"""
AI Swing Analyser — Robustness Research Engine.

Tests whether trading performance remains credible when assumptions
are made less favorable.

Stress dimensions:

    1. Monte Carlo trade-return resampling
    2. Transaction-cost stress
    3. Slippage stress
    4. Best-trade removal
    5. Worst-trade removal
    6. Drawdown / profitability robustness

The engine does not train models.

The final holdout must never be used for robustness-driven model
selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.models.robustness import (
    RobustnessConfig,
)


# ---------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------


@dataclass
class MonteCarloResult:
    """
    Monte Carlo robustness result.
    """

    simulations: int

    probability_of_profit: float

    probability_within_max_drawdown: float

    median_final_return: float

    mean_final_return: float

    worst_final_return: float

    best_final_return: float

    median_max_drawdown: float

    worst_max_drawdown: float

    percentile_05_return: float

    percentile_95_return: float


@dataclass
class StressResult:
    """
    Result of a cost/slippage stress test.
    """

    scenario: str

    adjustment_per_trade: float

    original_total_return: float

    stressed_total_return: float

    original_mean_trade: float

    stressed_mean_trade: float

    profitable_trades: int

    total_trades: int

    passed: bool


@dataclass
class TradeRemovalResult:
    """
    Result after removing a subset of trades.
    """

    scenario: str

    removed_trades: int

    remaining_trades: int

    total_return: float

    mean_trade_return: float

    max_drawdown: float

    profitable: bool


@dataclass
class RobustnessResearchResult:
    """
    Complete robustness research result.
    """

    monte_carlo: MonteCarloResult

    cost_stress: list[StressResult]

    slippage_stress: list[StressResult]

    trade_removal: list[TradeRemovalResult]

    robustness_score: float

    passed: bool

    final_holdout_used: bool

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    warnings: list[str] = field(
        default_factory=list
    )

    def summary(self) -> dict[str, Any]:
        return {
            "robustness_score": (
                self.robustness_score
            ),
            "passed": self.passed,
            "final_holdout_used": (
                self.final_holdout_used
            ),
            "monte_carlo": {
                "simulations": (
                    self.monte_carlo.simulations
                ),
                "probability_of_profit": (
                    self.monte_carlo.probability_of_profit
                ),
                "probability_within_max_drawdown": (
                    self.monte_carlo.probability_within_max_drawdown
                ),
                "median_final_return": (
                    self.monte_carlo.median_final_return
                ),
                "worst_final_return": (
                    self.monte_carlo.worst_final_return
                ),
                "median_max_drawdown": (
                    self.monte_carlo.median_max_drawdown
                ),
                "worst_max_drawdown": (
                    self.monte_carlo.worst_max_drawdown
                ),
            },
            "cost_stress_scenarios": len(
                self.cost_stress
            ),
            "slippage_stress_scenarios": len(
                self.slippage_stress
            ),
            "trade_removal_scenarios": len(
                self.trade_removal
            ),
            "metadata": dict(
                self.metadata
            ),
            "warnings": list(
                self.warnings
            ),
        }


# ---------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------


class RobustnessResearchEngine:
    """
    Research-only robustness analyzer.

    No model training occurs in this class.
    """

    def __init__(
        self,
        config: RobustnessConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else RobustnessConfig()
        )

        self._validate_config()

    # -----------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------

    def _validate_config(self) -> None:

        simulations = getattr(
            self.config,
            "simulations",
            5000,
        )

        if simulations < 100:
            raise ValueError(
                "At least 100 Monte Carlo simulations are required."
            )

        probability_threshold = getattr(
            self.config,
            "minimum_profit_probability",
            0.50,
        )

        if not (
            0.0
            <= probability_threshold
            <= 1.0
        ):
            raise ValueError(
                "minimum_profit_probability must be between 0 and 1."
            )

        maximum_drawdown = getattr(
            self.config,
            "maximum_drawdown",
            0.30,
        )

        if maximum_drawdown < 0:
            raise ValueError(
                "maximum_drawdown cannot be negative."
            )

        minimum_trades = getattr(
            self.config,
            "minimum_trades",
            30,
        )

        if minimum_trades < 1:
            raise ValueError(
                "minimum_trades must be positive."
            )

    # -----------------------------------------------------------------
    # Input validation
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_trade_returns(
        trade_returns: np.ndarray,
    ) -> np.ndarray:

        returns = np.asarray(
            trade_returns,
            dtype=float,
        ).reshape(-1)

        if len(returns) == 0:
            raise ValueError(
                "trade_returns must not be empty."
            )

        if not np.isfinite(
            returns
        ).all():
            raise ValueError(
                "trade_returns contains non-finite values."
            )

        return returns

    # -----------------------------------------------------------------
    # Basic metrics
    # -----------------------------------------------------------------

    @staticmethod
    def _total_return(
        trade_returns: np.ndarray,
    ) -> float:

        if len(trade_returns) == 0:
            return 0.0

        return float(
            np.prod(
                1.0 + trade_returns
            )
            - 1.0
        )

    @staticmethod
    def _mean_return(
        trade_returns: np.ndarray,
    ) -> float:

        if len(trade_returns) == 0:
            return 0.0

        return float(
            np.mean(
                trade_returns
            )
        )

    @staticmethod
    def _max_drawdown(
        trade_returns: np.ndarray,
    ) -> float:

        if len(trade_returns) == 0:
            return 0.0

        equity = np.cumprod(
            1.0 + trade_returns
        )

        running_peak = np.maximum.accumulate(
            equity
        )

        drawdown = (
            equity
            / running_peak
            - 1.0
        )

        return float(
            np.min(drawdown)
        )

    # -----------------------------------------------------------------
    # Monte Carlo
    # -----------------------------------------------------------------

    def _monte_carlo(
        self,
        trade_returns: np.ndarray,
    ) -> MonteCarloResult:

        simulations = int(
            getattr(
                self.config,
                "simulations",
                5000,
            )
        )

        maximum_drawdown = float(
            getattr(
                self.config,
                "maximum_drawdown",
                0.30,
            )
        )

        rng = np.random.default_rng(
            getattr(
                self.config,
                "random_state",
                42,
            )
        )

        n = len(
            trade_returns
        )

        final_returns = np.empty(
            simulations,
            dtype=float,
        )

        max_drawdowns = np.empty(
            simulations,
            dtype=float,
        )

        for i in range(
            simulations
        ):
            sample = rng.choice(
                trade_returns,
                size=n,
                replace=True,
            )

            final_returns[i] = (
                self._total_return(
                    sample
                )
            )

            max_drawdowns[i] = (
                self._max_drawdown(
                    sample
                )
            )

        probability_profit = float(
            np.mean(
                final_returns > 0
            )
        )

        probability_drawdown = float(
            np.mean(
                np.abs(
                    max_drawdowns
                )
                <= maximum_drawdown
            )
        )

        return MonteCarloResult(
            simulations=simulations,
            probability_of_profit=(
                probability_profit
            ),
            probability_within_max_drawdown=(
                probability_drawdown
            ),
            median_final_return=float(
                np.median(
                    final_returns
                )
            ),
            mean_final_return=float(
                np.mean(
                    final_returns
                )
            ),
            worst_final_return=float(
                np.min(
                    final_returns
                )
            ),
            best_final_return=float(
                np.max(
                    final_returns
                )
            ),
            median_max_drawdown=float(
                np.median(
                    max_drawdowns
                )
            ),
            worst_max_drawdown=float(
                np.min(
                    max_drawdowns
                )
            ),
            percentile_05_return=float(
                np.percentile(
                    final_returns,
                    5,
                )
            ),
            percentile_95_return=float(
                np.percentile(
                    final_returns,
                    95,
                )
            ),
        )

    # -----------------------------------------------------------------
    # Cost / slippage stress
    # -----------------------------------------------------------------

    def _stress(
        self,
        trade_returns: np.ndarray,
        adjustment: float,
        scenario: str,
    ) -> StressResult:

        if adjustment < 0:
            raise ValueError(
                "Stress adjustment cannot be negative."
            )

        stressed = (
            trade_returns
            - adjustment
        )

        minimum_trades = int(
            getattr(
                self.config,
                "minimum_trades",
                30,
            )
        )

        stressed_total = (
            self._total_return(
                stressed
            )
        )

        original_total = (
            self._total_return(
                trade_returns
            )
        )

        minimum_mean_trade = float(
            getattr(
                self.config,
                "minimum_mean_trade",
                0.0,
            )
        )

        passed = (
            len(stressed)
            >= minimum_trades
            and stressed_total
            > 0.0
            and self._mean_return(
                stressed
            )
            >= minimum_mean_trade
        )

        return StressResult(
            scenario=scenario,
            adjustment_per_trade=float(
                adjustment
            ),
            original_total_return=(
                original_total
            ),
            stressed_total_return=(
                stressed_total
            ),
            original_mean_trade=(
                self._mean_return(
                    trade_returns
                )
            ),
            stressed_mean_trade=(
                self._mean_return(
                    stressed
                )
            ),
            profitable_trades=int(
                np.sum(
                    stressed > 0
                )
            ),
            total_trades=len(
                stressed
            ),
            passed=passed,
        )

    # -----------------------------------------------------------------
    # Trade-removal stress
    # -----------------------------------------------------------------

    def _remove_extreme_trades(
        self,
        trade_returns: np.ndarray,
        fraction: float,
        remove_best: bool,
    ) -> TradeRemovalResult:

        if not (
            0.0
            <= fraction
            < 1.0
        ):
            raise ValueError(
                "Trade-removal fraction must be "
                "between 0 and 1."
            )

        n = len(
            trade_returns
        )

        remove_count = int(
            np.floor(
                n * fraction
            )
        )

        if remove_count == 0:
            remaining = trade_returns.copy()
        else:
            order = np.argsort(
                trade_returns
            )

            if remove_best:
                remove_indices = (
                    order[-remove_count:]
                )
            else:
                remove_indices = (
                    order[:remove_count]
                )

            keep_mask = np.ones(
                n,
                dtype=bool,
            )

            keep_mask[
                remove_indices
            ] = False

            # Preserve original chronological order.
            remaining = trade_returns[
                keep_mask
            ]

        total = self._total_return(
            remaining
        )

        mean_trade = self._mean_return(
            remaining
        )

        drawdown = self._max_drawdown(
            remaining
        )

        return TradeRemovalResult(
            scenario=(
                "remove_best"
                if remove_best
                else "remove_worst"
            ),
            removed_trades=(
                remove_count
            ),
            remaining_trades=len(
                remaining
            ),
            total_return=total,
            mean_trade_return=mean_trade,
            max_drawdown=drawdown,
            profitable=(
                total > 0.0
            ),
        )

    # -----------------------------------------------------------------
    # Main analysis
    # -----------------------------------------------------------------

    def analyze(
        self,
        trade_returns: np.ndarray,
        final_holdout_used: bool = False,
    ) -> RobustnessResearchResult:

        if final_holdout_used:
            raise ValueError(
                "Final holdout data must not be used "
                "for robustness research."
            )

        returns = self._validate_trade_returns(
            trade_returns
        )

        minimum_trades = int(
            getattr(
                self.config,
                "minimum_trades",
                30,
            )
        )

        warnings: list[str] = []

        if len(returns) < minimum_trades:
            warnings.append(
                "Trade count is below the configured minimum."
            )

        monte_carlo = self._monte_carlo(
            returns
        )

        cost_multipliers = getattr(
            self.config,
            "cost_multipliers",
            [1.5, 2.0, 3.0],
        )

        slippage_multipliers = getattr(
            self.config,
            "slippage_multipliers",
            [1.5, 2.0, 3.0],
        )

        base_cost = float(
            getattr(
                self.config,
                "transaction_cost",
                0.001,
            )
        )

        base_slippage = float(
            getattr(
                self.config,
                "slippage",
                0.0005,
            )
        )

        cost_stress = [
            self._stress(
                returns,
                base_cost * float(
                    multiplier
                ),
                f"transaction_cost_{multiplier:.1f}x",
            )
            for multiplier in cost_multipliers
        ]

        slippage_stress = [
            self._stress(
                returns,
                base_slippage * float(
                    multiplier
                ),
                f"slippage_{multiplier:.1f}x",
            )
            for multiplier in slippage_multipliers
        ]

        removal_fraction = float(
            getattr(
                self.config,
                "trade_removal_fraction",
                0.10,
            )
        )

        trade_removal = [
            self._remove_extreme_trades(
                returns,
                removal_fraction,
                remove_best=True,
            ),
            self._remove_extreme_trades(
                returns,
                removal_fraction,
                remove_best=False,
            ),
        ]

        minimum_profit_probability = float(
            getattr(
                self.config,
                "minimum_profit_probability",
                0.50,
            )
        )

        maximum_drawdown = float(
            getattr(
                self.config,
                "maximum_drawdown",
                0.30,
            )
        )

        mc_profit_score = float(
            np.clip(
                monte_carlo.probability_of_profit,
                0.0,
                1.0,
            )
        )

        mc_drawdown_score = float(
            np.clip(
                monte_carlo.probability_within_max_drawdown,
                0.0,
                1.0,
            )
        )

        cost_pass_score = (
            np.mean(
                [
                    float(
                        item.passed
                    )
                    for item in cost_stress
                ]
            )
            if cost_stress
            else 0.0
        )

        slippage_pass_score = (
            np.mean(
                [
                    float(
                        item.passed
                    )
                    for item in slippage_stress
                ]
            )
            if slippage_stress
            else 0.0
        )

        removal_score = (
            np.mean(
                [
                    float(
                        item.profitable
                    )
                    for item in trade_removal
                ]
            )
            if trade_removal
            else 0.0
        )

        robustness_score = float(
            np.mean(
                [
                    mc_profit_score,
                    mc_drawdown_score,
                    cost_pass_score,
                    slippage_pass_score,
                    removal_score,
                ]
            )
        )

        passed = (
            len(returns)
            >= minimum_trades
            and monte_carlo.probability_of_profit
            >= minimum_profit_probability
            and np.abs(
                monte_carlo.worst_max_drawdown
            )
            <= maximum_drawdown
            and robustness_score
            >= 0.60
        )

        if (
            monte_carlo.probability_of_profit
            < minimum_profit_probability
        ):
            warnings.append(
                "Monte Carlo probability of profit "
                "is below the configured threshold."
            )

        if (
            np.abs(
                monte_carlo.worst_max_drawdown
            )
            > maximum_drawdown
        ):
            warnings.append(
                "Monte Carlo worst drawdown exceeds "
                "the configured maximum."
            )

        return RobustnessResearchResult(
            monte_carlo=monte_carlo,
            cost_stress=cost_stress,
            slippage_stress=slippage_stress,
            trade_removal=trade_removal,
            robustness_score=robustness_score,
            passed=passed,
            final_holdout_used=False,
            metadata={
                "evaluation_only": True,
                "model_fitted": False,
                "final_holdout_used": False,
                "trade_order_preserved_after_removal": True,
                "research_only": True,
            },
            warnings=warnings,
        )


# ---------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------


def run_robustness_research(
    trade_returns: np.ndarray,
    config: RobustnessConfig | None = None,
) -> RobustnessResearchResult:

    engine = RobustnessResearchEngine(
        config=config
    )

    return engine.analyze(
        trade_returns=trade_returns,
        final_holdout_used=False,
    )


__all__ = [
    "MonteCarloResult",
    "StressResult",
    "TradeRemovalResult",
    "RobustnessResearchResult",
    "RobustnessResearchEngine",
    "run_robustness_research",
]
