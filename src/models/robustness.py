"""
Robustness and stress-testing utilities for AI Swing Analyser.

Purpose
-------
Evaluate whether a trading strategy's historical performance is robust
rather than dependent on a small number of lucky trades.

Tests include:
    - bootstrap trade-return simulation
    - trade-order permutation
    - transaction-cost stress
    - slippage stress
    - removal of best trades
    - removal of worst trades
    - drawdown stress
    - Monte Carlo percentile analysis

IMPORTANT
---------
These tests are diagnostic.

They do not prove that future performance will match historical
performance.

Robustness analysis must be performed on out-of-sample trades whenever
possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .monte_carlo import (
    bootstrap_trade_returns,
    shuffle_trade_sequence,
    monte_carlo_summary,
)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass
class RobustnessConfig:
    """
    Configuration for strategy robustness testing.
    """

    simulations: int = 5000

    random_state: int = 42

    cost_multipliers: tuple[float, ...] = (
        1.0,
        1.5,
        2.0,
        3.0,
    )

    slippage_multipliers: tuple[float, ...] = (
        1.0,
        1.5,
        2.0,
        3.0,
    )

    remove_best_trade_fraction: float = 0.10

    remove_worst_trade_fraction: float = 0.10

    minimum_profit_probability: float = 0.50

    maximum_drawdown: float = 0.30

    minimum_trade_count: int = 30

    def __post_init__(self) -> None:
        if self.simulations < 100:
            raise ValueError(
                "At least 100 simulations are required."
            )

        if self.minimum_trade_count < 1:
            raise ValueError(
                "minimum_trade_count must be positive."
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

        if not (
            0.0
            <= self.remove_best_trade_fraction
            < 1.0
        ):
            raise ValueError(
                "remove_best_trade_fraction must be in [0, 1)."
            )

        if not (
            0.0
            <= self.remove_worst_trade_fraction
            < 1.0
        ):
            raise ValueError(
                "remove_worst_trade_fraction must be in [0, 1)."
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


# ----------------------------------------------------------------------
# Result containers
# ----------------------------------------------------------------------


@dataclass
class MonteCarloRobustnessResult:
    """
    Monte Carlo robustness result.
    """

    simulations: int

    mean_return: float

    median_return: float

    return_5th_percentile: float

    return_25th_percentile: float

    return_75th_percentile: float

    return_95th_percentile: float

    probability_of_profit: float

    mean_max_drawdown: float

    median_max_drawdown: float

    worst_max_drawdown: float

    probability_within_drawdown_limit: float

    passed: bool

    notes: list[str] = field(
        default_factory=list
    )


@dataclass
class StressScenarioResult:
    """
    Result from one stress scenario.
    """

    scenario: str

    multiplier: float

    original_return: float

    stressed_return: float

    original_profit_factor: float

    stressed_profit_factor: float

    original_max_drawdown: float

    stressed_max_drawdown: float

    passed: bool


@dataclass
class TradeRemovalResult:
    """
    Result after removing unusually strong or weak trades.
    """

    scenario: str

    trades_removed: int

    original_trade_count: int

    remaining_trade_count: int

    original_total_return: float

    stressed_total_return: float

    original_max_drawdown: float

    stressed_max_drawdown: float

    passed: bool


@dataclass
class RobustnessReport:
    """
    Complete robustness report.
    """

    trade_count: int

    monte_carlo: Optional[
        MonteCarloRobustnessResult
    ]

    cost_stress: pd.DataFrame

    slippage_stress: pd.DataFrame

    trade_removal: pd.DataFrame

    passed: bool

    robustness_score: float

    notes: list[str] = field(
        default_factory=list
    )

    def summary(self) -> Dict[str, Any]:
        return {
            "trade_count": self.trade_count,
            "passed": self.passed,
            "robustness_score": (
                self.robustness_score
            ),
            "monte_carlo": (
                self.monte_carlo.__dict__
                if self.monte_carlo is not None
                else None
            ),
            "cost_stress_scenarios": len(
                self.cost_stress
            ),
            "slippage_stress_scenarios": len(
                self.slippage_stress
            ),
            "trade_removal_scenarios": len(
                self.trade_removal
            ),
        }


# ----------------------------------------------------------------------
# Main analyzer
# ----------------------------------------------------------------------


class RobustnessAnalyzer:
    """
    Strategy robustness analyzer.

    The analyzer works primarily from trade-level returns.

    Trade returns should already include the normal transaction costs and
    slippage used by the base backtest.
    """

    def __init__(
        self,
        config: Optional[
            RobustnessConfig
        ] = None,
    ) -> None:
        self.config = (
            config
            or RobustnessConfig()
        )

    # ------------------------------------------------------------------
    # Complete analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        trade_returns: Sequence[float],
        *,
        base_cost_fraction: float = 0.001,
        base_slippage_fraction: float = 0.0005,
    ) -> RobustnessReport:
        """
        Run the complete robustness analysis.
        """

        returns = self._validate_returns(
            trade_returns
        )

        if len(returns) < (
            self.config.minimum_trade_count
        ):
            raise ValueError(
                "Insufficient trades for robustness analysis. "
                f"Required at least "
                f"{self.config.minimum_trade_count}, "
                f"received {len(returns)}."
            )

        monte_carlo = (
            self._monte_carlo_analysis(
                returns
            )
        )

        cost_stress = (
            self._cost_stress(
                returns,
                base_cost_fraction,
            )
        )

        slippage_stress = (
            self._slippage_stress(
                returns,
                base_slippage_fraction,
            )
        )

        trade_removal = (
            self._trade_removal_stress(
                returns
            )
        )

        passed = self._overall_pass(
            monte_carlo=monte_carlo,
            cost_stress=cost_stress,
            slippage_stress=slippage_stress,
            trade_removal=trade_removal,
        )

        score = self._robustness_score(
            monte_carlo=monte_carlo,
            cost_stress=cost_stress,
            slippage_stress=slippage_stress,
            trade_removal=trade_removal,
        )

        notes = [
            (
                "Robustness analysis should use out-of-sample "
                "trade returns whenever possible."
            ),
            (
                "Monte Carlo results are stress diagnostics, "
                "not guarantees of future performance."
            ),
        ]

        if passed:
            notes.append(
                "Strategy passed the configured robustness tests."
            )
        else:
            notes.append(
                "Strategy failed one or more robustness tests."
            )

        return RobustnessReport(
            trade_count=len(returns),
            monte_carlo=monte_carlo,
            cost_stress=cost_stress,
            slippage_stress=slippage_stress,
            trade_removal=trade_removal,
            passed=passed,
            robustness_score=score,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------

    def _monte_carlo_analysis(
        self,
        returns: np.ndarray,
    ) -> MonteCarloRobustnessResult:
        rng = np.random.default_rng(
            self.config.random_state
        )

        final_returns = []
        max_drawdowns = []

        for _ in range(
            self.config.simulations
        ):
            sample = rng.choice(
                returns,
                size=len(returns),
                replace=True,
            )

            equity = np.cumprod(
                1.0 + sample
            )

            final_return = (
                equity[-1] - 1.0
            )

            peak = np.maximum.accumulate(
                equity
            )

            drawdown = (
                equity / peak
            ) - 1.0

            max_drawdown = float(
                drawdown.min()
            )

            final_returns.append(
                final_return
            )

            max_drawdowns.append(
                max_drawdown
            )

        final_returns = np.asarray(
            final_returns
        )

        max_drawdowns = np.asarray(
            max_drawdowns
        )

        probability_profit = float(
            np.mean(
                final_returns > 0
            )
        )

        probability_drawdown_limit = float(
            np.mean(
                np.abs(
                    max_drawdowns
                )
                <= self.config.maximum_drawdown
            )
        )

        passed = (
            probability_profit
            >= self.config
            .minimum_profit_probability
            and probability_drawdown_limit
            >= self.config
            .minimum_profit_probability
        )

        notes = []

        if probability_profit >= (
            self.config.minimum_profit_probability
        ):
            notes.append(
                "Monte Carlo probability of profit passed."
            )
        else:
            notes.append(
                "Monte Carlo probability of profit failed."
            )

        return MonteCarloRobustnessResult(
            simulations=self.config.simulations,
            mean_return=float(
                final_returns.mean()
            ),
            median_return=float(
                np.median(
                    final_returns
                )
            ),
            return_5th_percentile=float(
                np.percentile(
                    final_returns,
                    5,
                )
            ),
            return_25th_percentile=float(
                np.percentile(
                    final_returns,
                    25,
                )
            ),
            return_75th_percentile=float(
                np.percentile(
                    final_returns,
                    75,
                )
            ),
            return_95th_percentile=float(
                np.percentile(
                    final_returns,
                    95,
                )
            ),
            probability_of_profit=(
                probability_profit
            ),
            mean_max_drawdown=float(
                max_drawdowns.mean()
            ),
            median_max_drawdown=float(
                np.median(
                    max_drawdowns
                )
            ),
            worst_max_drawdown=float(
                max_drawdowns.min()
            ),
            probability_within_drawdown_limit=(
                probability_drawdown_limit
            ),
            passed=passed,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Cost stress
    # ------------------------------------------------------------------

    def _cost_stress(
        self,
        returns: np.ndarray,
        base_cost_fraction: float,
    ) -> pd.DataFrame:
        rows = []

        original_return = (
            self._compound_return(
                returns
            )
        )

        original_pf = (
            self._profit_factor(
                returns
            )
        )

        original_dd = (
            self._max_drawdown(
                returns
            )
        )

        for multiplier in (
            self.config.cost_multipliers
        ):
            additional_cost = (
                base_cost_fraction
                * (
                    multiplier
                    - 1.0
                )
            )

            stressed = (
                returns
                - additional_cost
            )

            stressed_return = (
                self._compound_return(
                    stressed
                )
            )

            stressed_pf = (
                self._profit_factor(
                    stressed
                )
            )

            stressed_dd = (
                self._max_drawdown(
                    stressed
                )
            )

            passed = (
                stressed_return > 0
                and stressed_dd
                >= -self.config.maximum_drawdown
            )

            rows.append(
                {
                    "scenario": (
                        "transaction_cost"
                    ),
                    "multiplier": multiplier,
                    "original_return": (
                        original_return
                    ),
                    "stressed_return": (
                        stressed_return
                    ),
                    "original_profit_factor": (
                        original_pf
                    ),
                    "stressed_profit_factor": (
                        stressed_pf
                    ),
                    "original_max_drawdown": (
                        original_dd
                    ),
                    "stressed_max_drawdown": (
                        stressed_dd
                    ),
                    "passed": passed,
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Slippage stress
    # ------------------------------------------------------------------

    def _slippage_stress(
        self,
        returns: np.ndarray,
        base_slippage_fraction: float,
    ) -> pd.DataFrame:
        rows = []

        original_return = (
            self._compound_return(
                returns
            )
        )

        original_pf = (
            self._profit_factor(
                returns
            )
        )

        original_dd = (
            self._max_drawdown(
                returns
            )
        )

        for multiplier in (
            self.config.slippage_multipliers
        ):
            additional_slippage = (
                base_slippage_fraction
                * (
                    multiplier
                    - 1.0
                )
            )

            stressed = (
                returns
                - additional_slippage
            )

            stressed_return = (
                self._compound_return(
                    stressed
                )
            )

            stressed_pf = (
                self._profit_factor(
                    stressed
                )
            )

            stressed_dd = (
                self._max_drawdown(
                    stressed
                )
            )

            passed = (
                stressed_return > 0
                and stressed_dd
                >= -self.config.maximum_drawdown
            )

            rows.append(
                {
                    "scenario": (
                        "slippage"
                    ),
                    "multiplier": multiplier,
                    "original_return": (
                        original_return
                    ),
                    "stressed_return": (
                        stressed_return
                    ),
                    "original_profit_factor": (
                        original_pf
                    ),
                    "stressed_profit_factor": (
                        stressed_pf
                    ),
                    "original_max_drawdown": (
                        original_dd
                    ),
                    "stressed_max_drawdown": (
                        stressed_dd
                    ),
                    "passed": passed,
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Trade-removal stress
    # ------------------------------------------------------------------

    def _trade_removal_stress(
        self,
        returns: np.ndarray,
    ) -> pd.DataFrame:
        rows = []

        original_return = (
            self._compound_return(
                returns
            )
        )

        original_dd = (
            self._max_drawdown(
                returns
            )
        )

        count = len(returns)

        best_remove = int(
            np.floor(
                count
                * self.config
                .remove_best_trade_fraction
            )
        )

        worst_remove = int(
            np.floor(
                count
                * self.config
                .remove_worst_trade_fraction
            )
        )

        # --------------------------------------------------------------
        # Remove best trades
        # --------------------------------------------------------------

        if best_remove > 0:
            ordered = np.sort(
                returns
            )

            stressed = ordered[
                : count - best_remove
            ]

            stressed_return = (
                self._compound_return(
                    stressed
                )
            )

            stressed_dd = (
                self._max_drawdown(
                    stressed
                )
            )

            passed = (
                stressed_return > 0
                and stressed_dd
                >= -self.config.maximum_drawdown
            )

            rows.append(
                {
                    "scenario": (
                        "remove_best_trades"
                    ),
                    "trades_removed": best_remove,
                    "original_trade_count": count,
                    "remaining_trade_count": len(
                        stressed
                    ),
                    "original_total_return": (
                        original_return
                    ),
                    "stressed_total_return": (
                        stressed_return
                    ),
                    "original_max_drawdown": (
                        original_dd
                    ),
                    "stressed_max_drawdown": (
                        stressed_dd
                    ),
                    "passed": passed,
                }
            )

        # --------------------------------------------------------------
        # Remove worst trades
        # --------------------------------------------------------------

        if worst_remove > 0:
            ordered = np.sort(
                returns
            )

            stressed = ordered[
                worst_remove:
            ]

            stressed_return = (
                self._compound_return(
                    stressed
                )
            )

            stressed_dd = (
                self._max_drawdown(
                    stressed
                )
            )

            passed = (
                stressed_return > 0
                and stressed_dd
                >= -self.config.maximum_drawdown
            )

            rows.append(
                {
                    "scenario": (
                        "remove_worst_trades"
                    ),
                    "trades_removed": worst_remove,
                    "original_trade_count": count,
                    "remaining_trade_count": len(
                        stressed
                    ),
                    "original_total_return": (
                        original_return
                    ),
                    "stressed_total_return": (
                        stressed_return
                    ),
                    "original_max_drawdown": (
                        original_dd
                    ),
                    "stressed_max_drawdown": (
                        stressed_dd
                    ),
                    "passed": passed,
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Overall decision
    # ------------------------------------------------------------------

    @staticmethod
    def _overall_pass(
        *,
        monte_carlo: Optional[
            MonteCarloRobustnessResult
        ],
        cost_stress: pd.DataFrame,
        slippage_stress: pd.DataFrame,
        trade_removal: pd.DataFrame,
    ) -> bool:
        if monte_carlo is None:
            return False

        if not monte_carlo.passed:
            return False

        for frame in (
            cost_stress,
            slippage_stress,
            trade_removal,
        ):
            if frame.empty:
                return False

            if not bool(
                frame["passed"].all()
            ):
                return False

        return True

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    @staticmethod
    def _robustness_score(
        *,
        monte_carlo: Optional[
            MonteCarloRobustnessResult
        ],
        cost_stress: pd.DataFrame,
        slippage_stress: pd.DataFrame,
        trade_removal: pd.DataFrame,
    ) -> float:
        components = []

        if monte_carlo is not None:
            components.append(
                monte_carlo
                .probability_of_profit
            )

            components.append(
                monte_carlo
                .probability_within_drawdown_limit
            )

        for frame in (
            cost_stress,
            slippage_stress,
            trade_removal,
        ):
            if not frame.empty:
                components.append(
                    float(
                        frame[
                            "passed"
                        ].mean()
                    )
                )

        if not components:
            return 0.0

        return float(
            np.mean(
                components
            )
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _compound_return(
        returns: Sequence[float],
    ) -> float:
        values = np.asarray(
            returns,
            dtype=float,
        )

        if len(values) == 0:
            return 0.0

        return float(
            np.prod(
                1.0 + values
            )
            - 1.0
        )

    @staticmethod
    def _max_drawdown(
        returns: Sequence[float],
    ) -> float:
        values = np.asarray(
            returns,
            dtype=float,
        )

        if len(values) == 0:
            return 0.0

        equity = np.cumprod(
            1.0 + values
        )

        peak = np.maximum.accumulate(
            equity
        )

        drawdown = (
            equity / peak
        ) - 1.0

        return float(
            drawdown.min()
        )

    @staticmethod
    def _profit_factor(
        returns: Sequence[float],
    ) -> float:
        values = np.asarray(
            returns,
            dtype=float,
        )

        gains = values[
            values > 0
        ].sum()

        losses = np.abs(
            values[
                values < 0
            ].sum()
        )

        if losses == 0:
            if gains > 0:
                return float("inf")

            return 0.0

        return float(
            gains / losses
        )

    @staticmethod
    def _validate_returns(
        trade_returns: Sequence[float],
    ) -> np.ndarray:
        values = np.asarray(
            trade_returns,
            dtype=float,
        ).reshape(-1)

        if len(values) == 0:
            raise ValueError(
                "No trade returns supplied."
            )

        if not np.isfinite(values).all():
            raise ValueError(
                "Trade returns contain NaN or infinite values."
            )

        if (
            values <= -1.0
        ).any():
            raise ValueError(
                "Trade returns cannot be <= -100%."
            )

        return values


# ----------------------------------------------------------------------
# Convenience functions
# ----------------------------------------------------------------------


def analyze_strategy_robustness(
    trade_returns: Sequence[float],
    *,
    config: Optional[
        RobustnessConfig
    ] = None,
    base_cost_fraction: float = 0.001,
    base_slippage_fraction: float = 0.0005,
) -> RobustnessReport:
    """
    Convenience wrapper for robustness analysis.
    """

    analyzer = RobustnessAnalyzer(
        config=config
    )

    return analyzer.analyze(
        trade_returns,
        base_cost_fraction=(
            base_cost_fraction
        ),
        base_slippage_fraction=(
            base_slippage_fraction
        ),
    )


def robustness_summary(
    report: RobustnessReport,
) -> Dict[str, Any]:
    """Return compact robustness information."""

    return report.summary()


__all__ = [
    "RobustnessConfig",
    "MonteCarloRobustnessResult",
    "StressScenarioResult",
    "TradeRemovalResult",
    "RobustnessReport",
    "RobustnessAnalyzer",
    "analyze_strategy_robustness",
    "robustness_summary",
]
