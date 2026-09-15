"""
Monte Carlo robustness analysis for AI Swing Analyser.

Tests the robustness of historical trading results by repeatedly
resampling the observed trade returns.

This is NOT a replacement for walk-forward validation.

It is an additional robustness diagnostic.

Uses:
    - Trade-return bootstrap
    - Random trade-order simulation
    - Final-return distribution
    - Drawdown distribution
    - Probability of loss
    - Percentile analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MonteCarloConfig:
    """Monte Carlo simulation configuration."""

    simulations: int = 5000

    initial_capital: float = 100_000.0

    random_state: int = 42

    periods_per_year: int = 252


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation results."""

    simulations: int

    median_final_capital: float

    mean_final_capital: float

    percentile_5_final_capital: float
    percentile_25_final_capital: float
    percentile_75_final_capital: float
    percentile_95_final_capital: float

    probability_of_loss: float

    median_return: float

    median_max_drawdown: float

    percentile_95_max_drawdown: float

    worst_simulated_max_drawdown: float

    return_distribution: np.ndarray

    final_capital_distribution: np.ndarray

    drawdown_distribution: np.ndarray


def _validate_returns(
    trade_returns: np.ndarray,
) -> np.ndarray:
    """Validate trade returns."""

    returns = np.asarray(
        trade_returns,
        dtype=float,
    ).reshape(-1)

    if len(returns) < 10:
        raise ValueError(
            "At least 10 trades are recommended for Monte Carlo analysis."
        )

    if not np.isfinite(
        returns
    ).all():
        raise ValueError(
            "Trade returns contain non-finite values."
        )

    if np.any(
        returns <= -1.0
    ):
        raise ValueError(
            "Trade return cannot be <= -100%."
        )

    return returns


def _maximum_drawdown(
    equity: np.ndarray,
) -> float:
    """Calculate maximum drawdown."""

    running_max = np.maximum.accumulate(
        equity
    )

    drawdown = (
        equity
        / running_max
        - 1.0
    )

    return float(
        drawdown.min()
    )


def bootstrap_trade_returns(
    trade_returns: np.ndarray,
    simulations: int = 5000,
    random_state: int = 42,
    initial_capital: float = 100_000.0,
) -> MonteCarloResult:
    """
    Bootstrap observed trade returns.

    Each simulation contains the same number of trades as the
    original strategy, but the trades are sampled with replacement.

    This answers:

        "If the observed trade distribution is representative,
         how much could the outcome vary simply because the
         sequence of trades was different?"
    """

    returns = _validate_returns(
        trade_returns
    )

    if simulations < 100:
        raise ValueError(
            "At least 100 simulations are recommended."
        )

    if initial_capital <= 0:
        raise ValueError(
            "initial_capital must be positive."
        )

    rng = np.random.default_rng(
        random_state
    )

    n_trades = len(
        returns
    )

    final_capitals = np.empty(
        simulations,
        dtype=float,
    )

    total_returns = np.empty(
        simulations,
        dtype=float,
    )

    max_drawdowns = np.empty(
        simulations,
        dtype=float,
    )

    for simulation in range(
        simulations
    ):

        sampled = rng.choice(
            returns,
            size=n_trades,
            replace=True,
        )

        equity = (
            initial_capital
            * np.cumprod(
                1.0 + sampled
            )
        )

        final_capital = float(
            equity[-1]
        )

        final_capitals[
            simulation
        ] = final_capital

        total_returns[
            simulation
        ] = (
            final_capital
            / initial_capital
            - 1.0
        )

        max_drawdowns[
            simulation
        ] = _maximum_drawdown(
            equity
        )

    probability_of_loss = float(
        np.mean(
            final_capitals
            < initial_capital
        )
    )

    return MonteCarloResult(
        simulations=simulations,

        median_final_capital=float(
            np.median(
                final_capitals
            )
        ),

        mean_final_capital=float(
            np.mean(
                final_capitals
            )
        ),

        percentile_5_final_capital=float(
            np.percentile(
                final_capitals,
                5,
            )
        ),

        percentile_25_final_capital=float(
            np.percentile(
                final_capitals,
                25,
            )
        ),

        percentile_75_final_capital=float(
            np.percentile(
                final_capitals,
                75,
            )
        ),

        percentile_95_final_capital=float(
            np.percentile(
                final_capitals,
                95,
            )
        ),

        probability_of_loss=(
            probability_of_loss
        ),

        median_return=float(
            np.median(
                total_returns
            )
        ),

        median_max_drawdown=float(
            np.median(
                max_drawdowns
            )
        ),

        percentile_95_max_drawdown=float(
            np.percentile(
                max_drawdowns,
                95,
            )
        ),

        worst_simulated_max_drawdown=float(
            np.min(
                max_drawdowns
            )
        ),

        return_distribution=(
            total_returns
        ),

        final_capital_distribution=(
            final_capitals
        ),

        drawdown_distribution=(
            max_drawdowns
        ),
    )


def shuffle_trade_sequence(
    trade_returns: np.ndarray,
    simulations: int = 5000,
    random_state: int = 42,
    initial_capital: float = 100_000.0,
) -> MonteCarloResult:
    """
    Test sensitivity to trade ordering.

    Unlike bootstrap_trade_returns(), this keeps every observed
    trade exactly once and only changes its order.

    Useful for estimating sequence risk and drawdown sensitivity.
    """

    returns = _validate_returns(
        trade_returns
    )

    if simulations < 100:
        raise ValueError(
            "At least 100 simulations are recommended."
        )

    rng = np.random.default_rng(
        random_state
    )

    n_trades = len(
        returns
    )

    final_capitals = np.empty(
        simulations,
        dtype=float,
    )

    total_returns = np.empty(
        simulations,
        dtype=float,
    )

    max_drawdowns = np.empty(
        simulations,
        dtype=float,
    )

    for simulation in range(
        simulations
    ):

        shuffled = rng.permutation(
            returns
        )

        equity = (
            initial_capital
            * np.cumprod(
                1.0 + shuffled
            )
        )

        final_capital = float(
            equity[-1]
        )

        final_capitals[
            simulation
        ] = final_capital

        total_returns[
            simulation
        ] = (
            final_capital
            / initial_capital
            - 1.0
        )

        max_drawdowns[
            simulation
        ] = _maximum_drawdown(
            equity
        )

    return MonteCarloResult(
        simulations=simulations,

        median_final_capital=float(
            np.median(
                final_capitals
            )
        ),

        mean_final_capital=float(
            np.mean(
                final_capitals
            )
        ),

        percentile_5_final_capital=float(
            np.percentile(
                final_capitals,
                5,
            )
        ),

        percentile_25_final_capital=float(
            np.percentile(
                final_capitals,
                25,
            )
        ),

        percentile_75_final_capital=float(
            np.percentile(
                final_capitals,
                75,
            )
        ),

        percentile_95_final_capital=float(
            np.percentile(
                final_capitals,
                95,
            )
        ),

        probability_of_loss=float(
            np.mean(
                final_capitals
                < initial_capital
            )
        ),

        median_return=float(
            np.median(
                total_returns
            )
        ),

        median_max_drawdown=float(
            np.median(
                max_drawdowns
            )
        ),

        percentile_95_max_drawdown=float(
            np.percentile(
                max_drawdowns,
                95,
            )
        ),

        worst_simulated_max_drawdown=float(
            np.min(
                max_drawdowns
            )
        ),

        return_distribution=(
            total_returns
        ),

        final_capital_distribution=(
            final_capitals
        ),

        drawdown_distribution=(
            max_drawdowns
        ),
    )


def monte_carlo_summary(
    result: MonteCarloResult,
) -> pd.DataFrame:
    """
    Convert Monte Carlo result into dashboard-friendly table.
    """

    return pd.DataFrame(
        [
            {
                "Metric": "Simulations",
                "Value": result.simulations,
            },
            {
                "Metric": "Mean Final Capital",
                "Value": result.mean_final_capital,
            },
            {
                "Metric": "Median Final Capital",
                "Value": result.median_final_capital,
            },
            {
                "Metric": "5th Percentile Capital",
                "Value": (
                    result.percentile_5_final_capital
                ),
            },
            {
                "Metric": "25th Percentile Capital",
                "Value": (
                    result.percentile_25_final_capital
                ),
            },
            {
                "Metric": "75th Percentile Capital",
                "Value": (
                    result.percentile_75_final_capital
                ),
            },
            {
                "Metric": "95th Percentile Capital",
                "Value": (
                    result.percentile_95_final_capital
                ),
            },
            {
                "Metric": "Probability of Loss",
                "Value": (
                    result.probability_of_loss
                ),
            },
            {
                "Metric": "Median Return",
                "Value": result.median_return,
            },
            {
                "Metric": "Median Max Drawdown",
                "Value": (
                    result.median_max_drawdown
                ),
            },
            {
                "Metric": "95th Percentile Drawdown",
                "Value": (
                    result.percentile_95_max_drawdown
                ),
            },
        ]
    )


def robustness_flags(
    result: MonteCarloResult,
    maximum_probability_of_loss: float = 0.20,
    maximum_drawdown: float = -0.30,
) -> dict:
    """
    Generate conservative robustness diagnostics.

    These are diagnostics rather than automatic production gates.
    """

    loss_risk_ok = (
        result.probability_of_loss
        <= maximum_probability_of_loss
    )

    drawdown_ok = (
        result.percentile_95_max_drawdown
        >= maximum_drawdown
    )

    return {
        "probability_of_loss": (
            result.probability_of_loss
        ),
        "probability_of_loss_ok": bool(
            loss_risk_ok
        ),
        "95_percentile_drawdown": (
            result.percentile_95_max_drawdown
        ),
        "drawdown_ok": bool(
            drawdown_ok
        ),
        "robustness_pass": bool(
            loss_risk_ok
            and drawdown_ok
        ),
        "warning": (
            "Monte Carlo analysis does not prove future profitability."
        ),
    }


def compare_monte_carlo_results(
    bootstrap: MonteCarloResult,
    shuffled: MonteCarloResult,
) -> pd.DataFrame:
    """
    Compare bootstrap and sequence-shuffle robustness.
    """

    return pd.DataFrame(
        [
            {
                "Analysis": "Bootstrap",
                "Median Return": (
                    bootstrap.median_return
                ),
                "Probability of Loss": (
                    bootstrap.probability_of_loss
                ),
                "Median Max Drawdown": (
                    bootstrap.median_max_drawdown
                ),
                "95% Max Drawdown": (
                    bootstrap.percentile_95_max_drawdown
                ),
            },
            {
                "Analysis": "Sequence Shuffle",
                "Median Return": (
                    shuffled.median_return
                ),
                "Probability of Loss": (
                    shuffled.probability_of_loss
                ),
                "Median Max Drawdown": (
                    shuffled.median_max_drawdown
                ),
                "95% Max Drawdown": (
                    shuffled.percentile_95_max_drawdown
                ),
            },
        ]
    )
