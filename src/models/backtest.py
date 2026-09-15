"""
Backtesting engine for AI Swing Analyser.

This module converts model predictions into a simple swing-trading
strategy and evaluates historical performance.

The backtester is deliberately conservative.

It supports:

    - Long signals
    - Short signals
    - Confidence thresholds
    - Stop-loss
    - Take-profit
    - Maximum holding period
    - Slippage
    - Transaction costs
    - Equity curve
    - Drawdown
    - Trade statistics

IMPORTANT:

This is a research backtester, not a live execution engine.

Predictions must be generated without using future information.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    """
    Configuration for swing-trading backtest.
    """

    initial_capital: float = 100_000.0

    probability_threshold: float = 0.60

    stop_loss_pct: float = 0.03

    take_profit_pct: float = 0.06

    max_holding_period: int = 10

    @property
    def max_holding_periods(self) -> int:
        """Backward-compatible plural form used internally."""
        return self.max_holding_period

    transaction_cost_pct: float = 0.001

    slippage_pct: float = 0.0005

    # Backward-compatible aliases used by research callers.
    transaction_cost: float | None = None

    slippage: float | None = None

    allow_long: bool = True

    allow_short: bool = False

    risk_per_trade: float = 0.01

    max_concurrent_positions: int = 1

    def __post_init__(self) -> None:
        if self.transaction_cost is not None:
            self.transaction_cost_pct = float(self.transaction_cost)
        if self.slippage is not None:
            self.slippage_pct = float(self.slippage)

        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")
        if not 0.50 <= self.probability_threshold < 1.0:
            raise ValueError("probability_threshold must be between 0.50 and 1.0.")
        if self.stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct must be positive.")
        if self.take_profit_pct <= 0:
            raise ValueError("take_profit_pct must be positive.")
        if self.max_holding_period < 1:
            raise ValueError("max_holding_period must be >= 1.")
        if self.transaction_cost_pct < 0:
            raise ValueError("transaction_cost_pct cannot be negative.")
        if self.slippage_pct < 0:
            raise ValueError("slippage_pct cannot be negative.")
        if not 0.0 < self.risk_per_trade <= 1.0:
            raise ValueError("risk_per_trade must be in (0, 1].")
        if self.max_concurrent_positions < 1:
            raise ValueError("max_concurrent_positions must be >= 1.")


@dataclass
class Trade:
    """
    Individual completed trade.
    """

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp

    direction: str

    entry_price: float
    exit_price: float

    gross_return: float
    transaction_cost: float
    slippage_cost: float

    net_return: float
    pnl: float

    holding_periods: int

    exit_reason: str


@dataclass
class BacktestResult:
    """
    Complete backtest output.
    """

    initial_capital: float

    final_capital: float

    total_return: float

    annualized_return: float

    volatility: float

    sharpe_ratio: float

    maximum_drawdown: float

    win_rate: float

    profit_factor: float

    average_trade_return: float

    median_trade_return: float

    trades: int

    winning_trades: int

    losing_trades: int

    average_holding_period: float

    trades_dataframe: pd.DataFrame

    equity_curve: pd.DataFrame

    @property
    def metrics(self) -> dict[str, float | int]:
        """Compatibility view of the core backtest metrics."""
        return {
            "final_capital": self.final_capital,
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "maximum_drawdown": self.maximum_drawdown,
            "max_drawdown": self.maximum_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "average_trade_return": self.average_trade_return,
            "median_trade_return": self.median_trade_return,
            "trades": self.trades,
            "trade_count": self.trades,
        }


def _validate_price_data(
    data: pd.DataFrame,
) -> None:
    """Validate OHLCV price dataframe."""

    required = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing price columns: {missing}"
        )

    if not isinstance(
        data.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "Price data must use a DatetimeIndex."
        )

    if data.index.has_duplicates:
        raise ValueError(
            "Price data contains duplicate timestamps."
        )

    if not data.index.is_monotonic_increasing:
        raise ValueError(
            "Price data must be chronologically sorted."
        )

    prices = data[
        required
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if prices.isna().any().any():
        raise ValueError(
            "Price data contains missing OHLC values."
        )

    if (
        prices["High"]
        < prices["Low"]
    ).any():
        raise ValueError(
            "High price is below Low price."
        )


def _validate_predictions(
    data: pd.DataFrame,
    probability: pd.Series,
) -> None:
    """Validate prediction series."""

    if not isinstance(
        probability.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "Prediction index must be DatetimeIndex."
        )

    if not probability.index.is_monotonic_increasing:
        raise ValueError(
            "Prediction timestamps must be sorted."
        )

    if not probability.index.is_unique:
        raise ValueError(
            "Prediction timestamps must be unique."
        )

    aligned = probability.reindex(
        data.index
    )

    if aligned.isna().all():
        raise ValueError(
            "No predictions overlap price data."
        )

    if (
        aligned.dropna()
        .lt(0)
        .any()
        or
        aligned.dropna()
        .gt(1)
        .any()
    ):
        raise ValueError(
            "Prediction probabilities must be between 0 and 1."
        )


def _calculate_long_return(
    entry_price: float,
    exit_price: float,
) -> float:
    """Calculate long gross return."""

    return (
        exit_price
        / entry_price
        - 1.0
    )


def _calculate_short_return(
    entry_price: float,
    exit_price: float,
) -> float:
    """Calculate short gross return."""

    return (
        entry_price
        / exit_price
        - 1.0
    )


def _apply_costs(
    gross_return: float,
    transaction_cost_pct: float,
    slippage_pct: float,
) -> tuple[float, float, float]:
    """
    Apply transaction costs and slippage.

    Costs are treated conservatively as a reduction in return.
    """

    transaction_cost = (
        2.0
        * transaction_cost_pct
    )

    slippage_cost = (
        2.0
        * slippage_pct
    )

    net_return = (
        gross_return
        - transaction_cost
        - slippage_cost
    )

    return (
        net_return,
        transaction_cost,
        slippage_cost,
    )


def _position_size(
    capital: float,
    entry_price: float,
    stop_price: float,
    risk_fraction: float,
) -> float:
    """
    Calculate quantity from fixed fractional risk.

    Risk per share:

        |entry - stop|

    Position value is limited by available capital.
    """

    # This public helper defines long-position sizing. A stop must be
    # strictly below entry; short-position sizing is handled by the
    # backtest trade simulation separately.
    if entry_price <= 0 or stop_price >= entry_price:
        return 0.0

    risk_per_share = entry_price - stop_price

    if risk_per_share <= 0:
        return 0.0

    risk_amount = (
        capital
        * risk_fraction
    )

    quantity = (
        risk_amount
        / risk_per_share
    )

    max_quantity = (
        capital
        / entry_price
    )

    return float(
        min(
            quantity,
            max_quantity,
        )
    )


def _exit_trade(
    direction: str,
    entry_price: float,
    exit_price: float,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    holding_periods: int,
    exit_reason: str,
    capital: float,
    config: BacktestConfig,
) -> Trade:
    """Create completed trade."""

    if direction == "LONG":

        gross_return = (
            _calculate_long_return(
                entry_price,
                exit_price,
            )
        )

    elif direction == "SHORT":

        gross_return = (
            _calculate_short_return(
                entry_price,
                exit_price,
            )
        )

    else:
        raise ValueError(
            f"Unknown direction: {direction}"
        )

    (
        net_return,
        transaction_cost,
        slippage_cost,
    ) = _apply_costs(
        gross_return,
        config.transaction_cost_pct,
        config.slippage_pct,
    )

    stop_price = (
        entry_price
        * (
            1.0
            - config.stop_loss_pct
        )
        if direction == "LONG"
        else
        entry_price
        * (
            1.0
            + config.stop_loss_pct
        )
    )

    quantity = _position_size(
        capital=capital,
        entry_price=entry_price,
        stop_price=stop_price,
        risk_fraction=config.risk_per_trade,
    )

    pnl = (
        quantity
        * entry_price
        * net_return
    )

    return Trade(
        entry_time=entry_time,
        exit_time=exit_time,
        direction=direction,
        entry_price=float(entry_price),
        exit_price=float(exit_price),
        gross_return=float(gross_return),
        transaction_cost=float(
            transaction_cost
        ),
        slippage_cost=float(
            slippage_cost
        ),
        net_return=float(net_return),
        pnl=float(pnl),
        holding_periods=int(
            holding_periods
        ),
        exit_reason=exit_reason,
    )


def _simulate_long_trade(
    data: pd.DataFrame,
    entry_position: int,
    config: BacktestConfig,
    capital: float,
) -> Trade:
    """Simulate one long trade."""

    entry_row = data.iloc[
        entry_position
    ]

    entry_time = data.index[
        entry_position
    ]

    entry_price = float(
        entry_row["Open"]
    )

    stop_price = (
        entry_price
        * (
            1.0
            - config.stop_loss_pct
        )
    )

    target_price = (
        entry_price
        * (
            1.0
            + config.take_profit_pct
        )
    )

    last_position = min(
        entry_position
        + config.max_holding_periods,
        len(data) - 1,
    )

    for position in range(
        entry_position + 1,
        last_position + 1,
    ):

        row = data.iloc[
            position
        ]

        high = float(
            row["High"]
        )

        low = float(
            row["Low"]
        )

        timestamp = data.index[
            position
        ]

        # Conservative assumption:
        # if stop and target are both touched in the same
        # candle, assume the stop was hit first.
        if low <= stop_price:

            return _exit_trade(
                direction="LONG",
                entry_price=entry_price,
                exit_price=stop_price,
                entry_time=entry_time,
                exit_time=timestamp,
                holding_periods=(
                    position
                    - entry_position
                ),
                exit_reason="STOP_LOSS",
                capital=capital,
                config=config,
            )

        if high >= target_price:

            return _exit_trade(
                direction="LONG",
                entry_price=entry_price,
                exit_price=target_price,
                entry_time=entry_time,
                exit_time=timestamp,
                holding_periods=(
                    position
                    - entry_position
                ),
                exit_reason="TAKE_PROFIT",
                capital=capital,
                config=config,
            )

    exit_position = last_position

    exit_row = data.iloc[
        exit_position
    ]

    return _exit_trade(
        direction="LONG",
        entry_price=entry_price,
        exit_price=float(
            exit_row["Close"]
        ),
        entry_time=entry_time,
        exit_time=data.index[
            exit_position
        ],
        holding_periods=(
            exit_position
            - entry_position
        ),
        exit_reason="TIME_EXIT",
        capital=capital,
        config=config,
    )


def _simulate_short_trade(
    data: pd.DataFrame,
    entry_position: int,
    config: BacktestConfig,
    capital: float,
) -> Trade:
    """Simulate one short trade."""

    entry_row = data.iloc[
        entry_position
    ]

    entry_time = data.index[
        entry_position
    ]

    entry_price = float(
        entry_row["Open"]
    )

    stop_price = (
        entry_price
        * (
            1.0
            + config.stop_loss_pct
        )
    )

    target_price = (
        entry_price
        * (
            1.0
            - config.take_profit_pct
        )
    )

    last_position = min(
        entry_position
        + config.max_holding_periods,
        len(data) - 1,
    )

    for position in range(
        entry_position + 1,
        last_position + 1,
    ):

        row = data.iloc[
            position
        ]

        high = float(
            row["High"]
        )

        low = float(
            row["Low"]
        )

        timestamp = data.index[
            position
        ]

        if high >= stop_price:

            return _exit_trade(
                direction="SHORT",
                entry_price=entry_price,
                exit_price=stop_price,
                entry_time=entry_time,
                exit_time=timestamp,
                holding_periods=(
                    position
                    - entry_position
                ),
                exit_reason="STOP_LOSS",
                capital=capital,
                config=config,
            )

        if low <= target_price:

            return _exit_trade(
                direction="SHORT",
                entry_price=entry_price,
                exit_price=target_price,
                entry_time=entry_time,
                exit_time=timestamp,
                holding_periods=(
                    position
                    - entry_position
                ),
                exit_reason="TAKE_PROFIT",
                capital=capital,
                config=config,
            )

    exit_position = last_position

    exit_row = data.iloc[
        exit_position
    ]

    return _exit_trade(
        direction="SHORT",
        entry_price=entry_price,
        exit_price=float(
            exit_row["Close"]
        ),
        entry_time=entry_time,
        exit_time=data.index[
            exit_position
        ],
        holding_periods=(
            exit_position
            - entry_position
        ),
        exit_reason="TIME_EXIT",
        capital=capital,
        config=config,
    )


def _equity_curve(
    data: pd.DataFrame,
    trades: list[Trade],
    initial_capital: float,
) -> pd.DataFrame:
    """Construct equity curve from completed trades."""

    equity = pd.Series(
        initial_capital,
        index=data.index,
        dtype=float,
    )

    current = (
        initial_capital
    )

    trade_by_exit = {
        trade.exit_time: trade
        for trade in trades
    }

    for timestamp in data.index:

        if timestamp in trade_by_exit:

            current += (
                trade_by_exit[
                    timestamp
                ].pnl
            )

        equity.loc[
            timestamp
        ] = current

    running_max = (
        equity.cummax()
    )

    drawdown = (
        equity
        / running_max
        - 1.0
    )

    return pd.DataFrame(
        {
            "Equity": equity,
            "Drawdown": drawdown,
        },
        index=data.index,
    )


def _performance_metrics(
    trades: list[Trade],
    equity_curve: pd.DataFrame,
    initial_capital: float,
    periods_per_year: int,
) -> dict:
    """Calculate backtest performance metrics."""

    final_capital = float(
        equity_curve["Equity"].iloc[-1]
    )

    total_return = (
        final_capital
        / initial_capital
        - 1.0
    )

    periods = len(
        equity_curve
    )

    years = (
        periods
        / periods_per_year
    )

    if years > 0 and final_capital > 0:

        annualized_return = (
            (
                final_capital
                / initial_capital
            )
            ** (
                1.0
                / years
            )
            - 1.0
        )

    else:

        annualized_return = -1.0

    equity_returns = (
        equity_curve["Equity"]
        .pct_change()
        .fillna(0.0)
    )

    volatility = float(
        equity_returns.std(
            ddof=1
        )
        * np.sqrt(
            periods_per_year
        )
    )

    if (
        equity_returns.std(
            ddof=1
        )
        > 0
    ):

        sharpe = float(
            equity_returns.mean()
            / equity_returns.std(
                ddof=1
            )
            * np.sqrt(
                periods_per_year
            )
        )

    else:

        sharpe = 0.0

    maximum_drawdown = float(
        equity_curve["Drawdown"].min()
    )

    trade_returns = np.asarray(
        [
            trade.net_return
            for trade in trades
        ],
        dtype=float,
    )

    winning = trade_returns[
        trade_returns > 0
    ]

    losing = trade_returns[
        trade_returns < 0
    ]

    if len(trade_returns) > 0:

        win_rate = float(
            len(winning)
            / len(trade_returns)
        )

        average_trade = float(
            np.mean(
                trade_returns
            )
        )

        median_trade = float(
            np.median(
                trade_returns
            )
        )

    else:

        win_rate = 0.0
        average_trade = 0.0
        median_trade = 0.0

    gross_profit = (
        winning.sum()
    )

    gross_loss = abs(
        losing.sum()
    )

    if gross_loss > 0:

        profit_factor = float(
            gross_profit
            / gross_loss
        )

    elif gross_profit > 0:

        profit_factor = float(
            "inf"
        )

    else:

        profit_factor = 0.0

    average_holding = (
        float(
            np.mean(
                [
                    trade.holding_periods
                    for trade in trades
                ]
            )
        )
        if trades
        else 0.0
    )

    return {
        "final_capital": final_capital,
        "total_return": float(
            total_return
        ),
        "annualized_return": float(
            annualized_return
        ),
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "maximum_drawdown": (
            maximum_drawdown
        ),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "average_trade_return": (
            average_trade
        ),
        "median_trade_return": (
            median_trade
        ),
        "trades": len(trades),
        "winning_trades": len(
            winning
        ),
        "losing_trades": len(
            losing
        ),
        "average_holding_period": (
            average_holding
        ),
    }


def run_backtest(
    data: pd.DataFrame,
    probability: Optional[pd.Series] = None,
    config: Optional[
        BacktestConfig
    ] = None,
    periods_per_year: int = 252,
) -> BacktestResult:
    """
    Run a historical swing-trading backtest.

    Signal timing:

        Prediction at candle t
              ↓
        Entry at candle t+1 Open

    This avoids entering at the same close used to generate
    the prediction.

    Long:
        probability >= threshold

    Short:
        probability <= 1 - threshold

    Only one position is allowed at a time by default.
    """

    config = (
        config
        if config is not None
        else BacktestConfig()
    )

    if probability is None:
        if "Probability" not in data.columns:
            raise ValueError(
                "probability must be supplied when data does not contain "
                "a 'Probability' column."
            )
        probability = data["Probability"]

    _validate_price_data(
        data
    )

    _validate_predictions(
        data,
        probability,
    )

    if not 0.50 <= (
        config.probability_threshold
    ) < 1.0:

        raise ValueError(
            "probability_threshold must be between 0.50 and 1.0."
        )

    if config.stop_loss_pct <= 0:
        raise ValueError(
            "stop_loss_pct must be positive."
        )

    if config.take_profit_pct <= 0:
        raise ValueError(
            "take_profit_pct must be positive."
        )

    if config.max_holding_periods < 1:
        raise ValueError(
            "max_holding_periods must be >= 1."
        )

    if config.initial_capital <= 0:
        raise ValueError(
            "initial_capital must be positive."
        )

    probability = (
        probability
        .reindex(data.index)
    )

    trades: list[Trade] = []

    capital = (
        config.initial_capital
    )

    position = 0

    while position < len(data) - 1:

        current_probability = (
            probability.iloc[position]
        )

        if pd.isna(
            current_probability
        ):
            position += 1
            continue

        direction = None

        if (
            config.allow_long
            and current_probability
            >= config.probability_threshold
        ):

            direction = "LONG"

        elif (
            config.allow_short
            and current_probability
            <= (
                1.0
                - config.probability_threshold
            )
        ):

            direction = "SHORT"

        if direction is None:

            position += 1
            continue

        entry_position = (
            position + 1
        )

        if entry_position >= len(data):
            break

        if direction == "LONG":

            trade = _simulate_long_trade(
                data=data,
                entry_position=entry_position,
                config=config,
                capital=capital,
            )

        else:

            trade = _simulate_short_trade(
                data=data,
                entry_position=entry_position,
                config=config,
                capital=capital,
            )

        trades.append(
            trade
        )

        capital += trade.pnl

        exit_position = (
            data.index.get_loc(
                trade.exit_time
            )
        )

        position = (
            exit_position + 1
        )

    equity_curve = _equity_curve(
        data=data,
        trades=trades,
        initial_capital=(
            config.initial_capital
        ),
    )

    metrics = _performance_metrics(
        trades=trades,
        equity_curve=equity_curve,
        initial_capital=(
            config.initial_capital
        ),
        periods_per_year=(
            periods_per_year
        ),
    )

    if trades:

        trades_dataframe = pd.DataFrame(
            [
                asdict(trade)
                for trade in trades
            ]
        )

    else:

        trades_dataframe = pd.DataFrame(
            columns=[
                field
                for field in Trade.__dataclass_fields__
            ]
        )

    return BacktestResult(
        initial_capital=(
            config.initial_capital
        ),
        final_capital=(
            metrics["final_capital"]
        ),
        total_return=(
            metrics["total_return"]
        ),
        annualized_return=(
            metrics["annualized_return"]
        ),
        volatility=(
            metrics["volatility"]
        ),
        sharpe_ratio=(
            metrics["sharpe_ratio"]
        ),
        maximum_drawdown=(
            metrics["maximum_drawdown"]
        ),
        win_rate=(
            metrics["win_rate"]
        ),
        profit_factor=(
            metrics["profit_factor"]
        ),
        average_trade_return=(
            metrics[
                "average_trade_return"
            ]
        ),
        median_trade_return=(
            metrics[
                "median_trade_return"
            ]
        ),
        trades=(
            metrics["trades"]
        ),
        winning_trades=(
            metrics["winning_trades"]
        ),
        losing_trades=(
            metrics["losing_trades"]
        ),
        average_holding_period=(
            metrics[
                "average_holding_period"
            ]
        ),
        trades_dataframe=(
            trades_dataframe
        ),
        equity_curve=(
            equity_curve
        ),
    )


def calculate_position_size(
    capital: float,
    entry_price: float,
    stop_price: float,
    risk_fraction: float,
) -> float:
    """Public compatibility wrapper for fixed-risk position sizing."""
    return _position_size(
        capital=capital,
        entry_price=entry_price,
        stop_price=stop_price,
        risk_fraction=risk_fraction,
    )
