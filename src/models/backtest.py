"""
Backtesting engine for AI Swing Analyser.

This module converts model predictions into a historical swing-trading
strategy and evaluates both statistical and economic performance.

Design principles
-----------------
1. Prediction at candle t can only enter at candle t+1.
2. No future information is used to generate the signal.
3. Stop-loss and take-profit are evaluated conservatively.
4. Long and short position sizing are handled separately.
5. Transaction costs and slippage are explicitly included.
6. Risk-based position sizing is supported.
7. Equity, drawdown and trade-level statistics are reported.
8. Economic metrics are exposed for production-model approval.

IMPORTANT
---------
This is a research backtester, not a live execution engine.

Backtest results are only meaningful when the prediction series itself
was generated using a leakage-free walk-forward process.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass
class BacktestConfig:
    """
    Configuration for the swing-trading backtest.
    """

    initial_capital: float = 100_000.0

    probability_threshold: float = 0.60

    stop_loss_pct: float = 0.03

    take_profit_pct: float = 0.06

    max_holding_period: int = 10

    @property
    def max_holding_periods(self) -> int:
        """Backward-compatible plural form."""
        return self.max_holding_period

    transaction_cost_pct: float = 0.001

    slippage_pct: float = 0.0005

    # Backward-compatible aliases.
    transaction_cost: float | None = None

    slippage: float | None = None

    allow_long: bool = True

    allow_short: bool = False

    risk_per_trade: float = 0.01

    max_concurrent_positions: int = 1

    def __post_init__(self) -> None:

        if self.transaction_cost is not None:
            self.transaction_cost_pct = float(
                self.transaction_cost
            )

        if self.slippage is not None:
            self.slippage_pct = float(
                self.slippage
            )

        if self.initial_capital <= 0:
            raise ValueError(
                "initial_capital must be positive."
            )

        if not (
            0.50
            <= self.probability_threshold
            < 1.0
        ):
            raise ValueError(
                "probability_threshold must be between 0.50 and 1.0."
            )

        if self.stop_loss_pct <= 0:
            raise ValueError(
                "stop_loss_pct must be positive."
            )

        if self.take_profit_pct <= 0:
            raise ValueError(
                "take_profit_pct must be positive."
            )

        if self.max_holding_period < 1:
            raise ValueError(
                "max_holding_period must be >= 1."
            )

        if self.transaction_cost_pct < 0:
            raise ValueError(
                "transaction_cost_pct cannot be negative."
            )

        if self.slippage_pct < 0:
            raise ValueError(
                "slippage_pct cannot be negative."
            )

        if not (
            0.0
            < self.risk_per_trade
            <= 1.0
        ):
            raise ValueError(
                "risk_per_trade must be in (0, 1]."
            )

        if self.max_concurrent_positions < 1:
            raise ValueError(
                "max_concurrent_positions must be >= 1."
            )


# ----------------------------------------------------------------------
# Trade
# ----------------------------------------------------------------------


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

    quantity: float = 0.0

    position_value: float = 0.0

    risk_amount: float = 0.0


# ----------------------------------------------------------------------
# Backtest result
# ----------------------------------------------------------------------


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

    # Additional research metrics.
    expectancy: float = 0.0

    downside_deviation: float = 0.0

    sortino_ratio: float = 0.0

    calmar_ratio: float = 0.0

    exposure: float = 0.0

    total_transaction_cost: float = 0.0

    total_slippage_cost: float = 0.0

    best_trade: float = 0.0

    worst_trade: float = 0.0

    @property
    def metrics(
        self,
    ) -> dict[str, float | int]:
        """
        Compatibility view of the backtest metrics.
        """

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
            "average_trade_return": (
                self.average_trade_return
            ),
            "median_trade_return": (
                self.median_trade_return
            ),
            "trades": self.trades,
            "trade_count": self.trades,
            "expectancy": self.expectancy,
            "downside_deviation": (
                self.downside_deviation
            ),
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "exposure": self.exposure,
            "total_transaction_cost": (
                self.total_transaction_cost
            ),
            "total_slippage_cost": (
                self.total_slippage_cost
            ),
            "best_trade": self.best_trade,
            "worst_trade": self.worst_trade,
        }


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def _validate_price_data(
    data: pd.DataFrame,
) -> None:
    """
    Validate OHLC price data.
    """

    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise TypeError(
            "Price data must be a pandas DataFrame."
        )

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

    if len(data) < 2:
        raise ValueError(
            "At least two price rows are required."
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

    if not np.isfinite(
        prices.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Price data contains NaN or infinite values."
        )

    if (
        prices[
            "High"
        ]
        < prices[
            "Low"
        ]
    ).any():
        raise ValueError(
            "High price is below Low price."
        )

    if (
        prices[
            "Open"
        ]
        <= 0
    ).any():
        raise ValueError(
            "Open prices must be positive."
        )

    if (
        prices[
            "High"
        ]
        <= 0
    ).any():
        raise ValueError(
            "High prices must be positive."
        )

    if (
        prices[
            "Low"
        ]
        <= 0
    ).any():
        raise ValueError(
            "Low prices must be positive."
        )

    if (
        prices[
            "Close"
        ]
        <= 0
    ).any():
        raise ValueError(
            "Close prices must be positive."
        )


def _validate_predictions(
    data: pd.DataFrame,
    probability: pd.Series,
) -> None:
    """
    Validate prediction probabilities.
    """

    if not isinstance(
        probability,
        pd.Series,
    ):
        probability = pd.Series(
            probability
        )

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

    numeric = pd.to_numeric(
        aligned,
        errors="coerce",
    )

    if (
        numeric.dropna()
        .lt(0.0)
        .any()
    ):
        raise ValueError(
            "Prediction probabilities must be >= 0."
        )

    if (
        numeric.dropna()
        .gt(1.0)
        .any()
    ):
        raise ValueError(
            "Prediction probabilities must be <= 1."
        )


# ----------------------------------------------------------------------
# Return calculations
# ----------------------------------------------------------------------


def _calculate_long_return(
    entry_price: float,
    exit_price: float,
) -> float:
    """
    Calculate gross long return.
    """

    if entry_price <= 0:
        raise ValueError(
            "entry_price must be positive."
        )

    return (
        exit_price
        / entry_price
        - 1.0
    )


def _calculate_short_return(
    entry_price: float,
    exit_price: float,
) -> float:
    """
    Calculate gross short return.

    Example:

        Entry = 100
        Exit  = 95

        Gross return = 5.26%
    """

    if entry_price <= 0:
        raise ValueError(
            "entry_price must be positive."
        )

    if exit_price <= 0:
        raise ValueError(
            "exit_price must be positive."
        )

    return (
        entry_price
        / exit_price
        - 1.0
    )


def _apply_costs(
    gross_return: float,
    transaction_cost_pct: float,
    slippage_pct: float,
) -> tuple[
    float,
    float,
    float,
]:
    """
    Apply round-trip transaction costs and slippage.

    The percentages are treated as fractions of traded notional.

    Example:

        transaction_cost_pct = 0.001
        slippage_pct = 0.0005

    Total round-trip cost:

        2 * 0.001 + 2 * 0.0005
        = 0.003
        = 0.30%
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
        float(net_return),
        float(transaction_cost),
        float(slippage_cost),
    )


# ----------------------------------------------------------------------
# Position sizing
# ----------------------------------------------------------------------


def _position_size(
    capital: float,
    entry_price: float,
    stop_price: float,
    risk_fraction: float,
) -> float:
    """
    Calculate quantity using fixed fractional risk.

    This helper preserves the original long-position behavior.

    For a long:

        risk/share = entry - stop

    Position size is also capped by available capital.
    """

    if capital <= 0:
        return 0.0

    if entry_price <= 0:
        return 0.0

    if stop_price >= entry_price:
        return 0.0

    if not (
        0.0
        < risk_fraction
        <= 1.0
    ):
        return 0.0

    risk_per_share = (
        entry_price
        - stop_price
    )

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


def _position_size_directional(
    *,
    capital: float,
    entry_price: float,
    stop_price: float,
    risk_fraction: float,
    direction: str,
) -> float:
    """
    Direction-aware position sizing.

    Long:

        risk/share = stop - entry in absolute terms.

    Short:

        risk/share = stop - entry in absolute terms.

    Both are converted to an absolute risk per share.

    Position value is capped at available capital.
    """

    if capital <= 0:
        return 0.0

    if entry_price <= 0:
        return 0.0

    if stop_price <= 0:
        return 0.0

    if not (
        0.0
        < risk_fraction
        <= 1.0
    ):
        return 0.0

    direction = str(
        direction
    ).upper()

    if direction == "LONG":

        if stop_price >= entry_price:
            return 0.0

    elif direction == "SHORT":

        if stop_price <= entry_price:
            return 0.0

    else:
        raise ValueError(
            f"Unknown direction: {direction}"
        )

    risk_per_share = abs(
        entry_price
        - stop_price
    )

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


# ----------------------------------------------------------------------
# Trade construction
# ----------------------------------------------------------------------


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
    """
    Create a completed trade.

    Position sizing is direction-aware.
    """

    direction = str(
        direction
    ).upper()

    if direction == "LONG":

        gross_return = (
            _calculate_long_return(
                entry_price,
                exit_price,
            )
        )

        stop_price = (
            entry_price
            * (
                1.0
                - config.stop_loss_pct
            )
        )

    elif direction == "SHORT":

        gross_return = (
            _calculate_short_return(
                entry_price,
                exit_price,
            )
        )

        stop_price = (
            entry_price
            * (
                1.0
                + config.stop_loss_pct
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
        gross_return=gross_return,
        transaction_cost_pct=(
            config.transaction_cost_pct
        ),
        slippage_pct=(
            config.slippage_pct
        ),
    )

    quantity = (
        _position_size_directional(
            capital=capital,
            entry_price=entry_price,
            stop_price=stop_price,
            risk_fraction=(
                config.risk_per_trade
            ),
            direction=direction,
        )
    )

    position_value = (
        quantity
        * entry_price
    )

    risk_amount = (
        capital
        * config.risk_per_trade
    )

    pnl = (
        position_value
        * net_return
    )

    return Trade(
        entry_time=entry_time,
        exit_time=exit_time,
        direction=direction,
        entry_price=float(
            entry_price
        ),
        exit_price=float(
            exit_price
        ),
        gross_return=float(
            gross_return
        ),
        transaction_cost=float(
            transaction_cost
        ),
        slippage_cost=float(
            slippage_cost
        ),
        net_return=float(
            net_return
        ),
        pnl=float(
            pnl
        ),
        holding_periods=int(
            holding_periods
        ),
        exit_reason=str(
            exit_reason
        ),
        quantity=float(
            quantity
        ),
        position_value=float(
            position_value
        ),
        risk_amount=float(
            risk_amount
        ),
    )


# ----------------------------------------------------------------------
# Long trade simulation
# ----------------------------------------------------------------------


def _simulate_long_trade(
    data: pd.DataFrame,
    entry_position: int,
    config: BacktestConfig,
    capital: float,
) -> Trade:
    """
    Simulate one long trade.

    If stop and target are both touched in the same candle,
    stop-loss is assumed to occur first.
    """

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

        # Conservative intrabar ordering.
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


# ----------------------------------------------------------------------
# Short trade simulation
# ----------------------------------------------------------------------


def _simulate_short_trade(
    data: pd.DataFrame,
    entry_position: int,
    config: BacktestConfig,
    capital: float,
) -> Trade:
    """
    Simulate one short trade.

    If stop and target are both touched in the same candle,
    stop-loss is assumed to occur first.
    """

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


# ----------------------------------------------------------------------
# Equity curve
# ----------------------------------------------------------------------


def _equity_curve(
    data: pd.DataFrame,
    trades: list[Trade],
    initial_capital: float,
) -> pd.DataFrame:
    """
    Construct an equity curve from completed trades.

    Since this is a trade-level research engine, unrealized P&L is not
    marked continuously. Equity changes when a trade is closed.
    """

    equity = pd.Series(
        initial_capital,
        index=data.index,
        dtype=float,
    )

    current = float(
        initial_capital
    )

    trade_by_exit: dict[
        pd.Timestamp,
        list[Trade],
    ] = {}

    for trade in trades:

        trade_by_exit.setdefault(
            trade.exit_time,
            [],
        ).append(
            trade
        )

    for timestamp in data.index:

        if timestamp in trade_by_exit:

            for trade in trade_by_exit[
                timestamp
            ]:

                current += trade.pnl

        equity.loc[
            timestamp
        ] = current

    running_max = equity.cummax()

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


# ----------------------------------------------------------------------
# Performance metrics
# ----------------------------------------------------------------------


def _performance_metrics(
    trades: list[Trade],
    equity_curve: pd.DataFrame,
    initial_capital: float,
    periods_per_year: int,
) -> dict:
    """
    Calculate statistical and economic backtest metrics.
    """

    if periods_per_year <= 0:
        raise ValueError(
            "periods_per_year must be positive."
        )

    final_capital = float(
        equity_curve[
            "Equity"
        ].iloc[-1]
    )

    total_return = (
        final_capital
        / initial_capital
        - 1.0
    )

    periods = max(
        len(equity_curve),
        1,
    )

    years = (
        periods
        / periods_per_year
    )

    if (
        years > 0
        and final_capital > 0
    ):

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
        equity_curve[
            "Equity"
        ]
        .pct_change()
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(0.0)
    )

    return_std = float(
        equity_returns.std(
            ddof=1
        )
    )

    if (
        not np.isfinite(
            return_std
        )
    ):
        return_std = 0.0

    volatility = float(
        return_std
        * np.sqrt(
            periods_per_year
        )
    )

    if return_std > 0:

        sharpe = float(
            equity_returns.mean()
            / return_std
            * np.sqrt(
                periods_per_year
            )
        )

    else:

        sharpe = 0.0

    # --------------------------------------------------------------
    # Drawdown
    # --------------------------------------------------------------

    maximum_drawdown = float(
        equity_curve[
            "Drawdown"
        ].min()
    )

    # --------------------------------------------------------------
    # Trade returns
    # --------------------------------------------------------------

    trade_returns = np.asarray(
        [
            trade.net_return
            for trade in trades
        ],
        dtype=float,
    )

    if len(trade_returns) > 0:

        winning = trade_returns[
            trade_returns > 0
        ]

        losing = trade_returns[
            trade_returns < 0
        ]

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

        best_trade = float(
            np.max(
                trade_returns
            )
        )

        worst_trade = float(
            np.min(
                trade_returns
            )
        )

        expectancy = float(
            average_trade
        )

    else:

        winning = np.asarray(
            [],
            dtype=float,
        )

        losing = np.asarray(
            [],
            dtype=float,
        )

        win_rate = 0.0
        average_trade = 0.0
        median_trade = 0.0
        best_trade = 0.0
        worst_trade = 0.0
        expectancy = 0.0

    # --------------------------------------------------------------
    # Profit factor
    # --------------------------------------------------------------

    gross_profit = float(
        winning.sum()
    )

    gross_loss = float(
        abs(
            losing.sum()
        )
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

    # --------------------------------------------------------------
    # Downside risk
    # --------------------------------------------------------------

    negative_returns = (
        equity_returns[
            equity_returns < 0
        ]
    )

    if len(
        negative_returns
    ) > 1:

        downside_deviation = float(
            negative_returns.std(
                ddof=1
            )
            * np.sqrt(
                periods_per_year
            )
        )

    else:

        downside_deviation = 0.0

    if downside_deviation > 0:

        sortino_ratio = float(
            equity_returns.mean()
            * periods_per_year
            / downside_deviation
        )

    else:

        sortino_ratio = 0.0

    # --------------------------------------------------------------
    # Calmar ratio
    # --------------------------------------------------------------

    if (
        maximum_drawdown < 0
        and np.isfinite(
            annualized_return
        )
    ):

        calmar_ratio = float(
            annualized_return
            / abs(
                maximum_drawdown
            )
        )

    else:

        calmar_ratio = 0.0

    # --------------------------------------------------------------
    # Holding period
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # Exposure
    # --------------------------------------------------------------

    total_periods = len(
        equity_curve
    )

    occupied_periods = sum(
        max(
            0,
            trade.holding_periods,
        )
        for trade in trades
    )

    exposure = (
        float(
            occupied_periods
            / total_periods
        )
        if total_periods > 0
        else 0.0
    )

    exposure = min(
        max(
            exposure,
            0.0,
        ),
        1.0,
    )

    # --------------------------------------------------------------
    # Costs
    # --------------------------------------------------------------

    total_transaction_cost = float(
        sum(
            trade.transaction_cost
            * trade.position_value
            for trade in trades
        )
    )

    total_slippage_cost = float(
        sum(
            trade.slippage_cost
            * trade.position_value
            for trade in trades
        )
    )

    return {
        "final_capital": final_capital,
        "total_return": float(
            total_return
        ),
        "annualized_return": float(
            annualized_return
        ),
        "volatility": float(
            volatility
        ),
        "sharpe_ratio": float(
            sharpe
        ),
        "maximum_drawdown": float(
            maximum_drawdown
        ),
        "win_rate": float(
            win_rate
        ),
        "profit_factor": float(
            profit_factor
        ),
        "average_trade_return": float(
            average_trade
        ),
        "median_trade_return": float(
            median_trade
        ),
        "trades": len(
            trades
        ),
        "winning_trades": len(
            winning
        ),
        "losing_trades": len(
            losing
        ),
        "average_holding_period": float(
            average_holding
        ),
        "expectancy": float(
            expectancy
        ),
        "downside_deviation": float(
            downside_deviation
        ),
        "sortino_ratio": float(
            sortino_ratio
        ),
        "calmar_ratio": float(
            calmar_ratio
        ),
        "exposure": float(
            exposure
        ),
        "total_transaction_cost": float(
            total_transaction_cost
        ),
        "total_slippage_cost": float(
            total_slippage_cost
        ),
        "best_trade": float(
            best_trade
        ),
        "worst_trade": float(
            worst_trade
        ),
    }


# ----------------------------------------------------------------------
# Main backtest
# ----------------------------------------------------------------------


def run_backtest(
    data: pd.DataFrame,
    probability: Optional[
        pd.Series
    ] = None,
    config: Optional[
        BacktestConfig
    ] = None,
    periods_per_year: int = 252,
) -> BacktestResult:
    """
    Run a historical swing-trading backtest.

    Signal timing
    -------------
        Prediction at candle t
                    ↓
        Entry at candle t+1 Open

    This prevents same-candle look-ahead from the prediction close.

    Long
    ----
        probability >= threshold

    Short
    -----
        probability <= 1 - threshold

    Position policy
    ---------------
    The current implementation is sequential by design and therefore
    permits at most one active trade at a time.

    ``max_concurrent_positions`` is retained for compatibility and is
    validated, but multi-position portfolio accounting is intentionally
    left for a later portfolio-engine layer.
    """

    config = (
        config
        if config is not None
        else BacktestConfig()
    )

    _validate_price_data(
        data
    )

    if probability is None:

        if "Probability" not in data.columns:

            raise ValueError(
                "probability must be supplied when data does not contain "
                "a 'Probability' column."
            )

        probability = data[
            "Probability"
        ]

    if not isinstance(
        probability,
        pd.Series,
    ):

        probability = pd.Series(
            probability,
            index=data.index,
        )

    _validate_predictions(
        data,
        probability,
    )

    probability = (
        pd.to_numeric(
            probability,
            errors="coerce",
        )
        .reindex(
            data.index
        )
    )

    trades: list[Trade] = []

    capital = float(
        config.initial_capital
    )

    position = 0

    while position < (
        len(data) - 1
    ):

        current_probability = (
            probability.iloc[
                position
            ]
        )

        if pd.isna(
            current_probability
        ):

            position += 1
            continue

        direction: Optional[
            str
        ] = None

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

        # ----------------------------------------------------------
        # Entry is ALWAYS the next candle.
        # ----------------------------------------------------------

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

        # Find the next available signal after the trade exits.
        exit_position = (
            data.index.get_loc(
                trade.exit_time
            )
        )

        position = (
            exit_position + 1
        )

    # ------------------------------------------------------------------
    # Equity curve
    # ------------------------------------------------------------------

    equity_curve = _equity_curve(
        data=data,
        trades=trades,
        initial_capital=(
            config.initial_capital
        ),
    )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Trade dataframe
    # ------------------------------------------------------------------

    if trades:

        trades_dataframe = pd.DataFrame(
            [
                asdict(
                    trade
                )
                for trade in trades
            ]
        )

    else:

        trades_dataframe = pd.DataFrame(
            columns=[
                field
                for field
                in Trade.__dataclass_fields__
            ]
        )

    return BacktestResult(
        initial_capital=(
            config.initial_capital
        ),
        final_capital=(
            metrics[
                "final_capital"
            ]
        ),
        total_return=(
            metrics[
                "total_return"
            ]
        ),
        annualized_return=(
            metrics[
                "annualized_return"
            ]
        ),
        volatility=(
            metrics[
                "volatility"
            ]
        ),
        sharpe_ratio=(
            metrics[
                "sharpe_ratio"
            ]
        ),
        maximum_drawdown=(
            metrics[
                "maximum_drawdown"
            ]
        ),
        win_rate=(
            metrics[
                "win_rate"
            ]
        ),
        profit_factor=(
            metrics[
                "profit_factor"
            ]
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
            metrics[
                "trades"
            ]
        ),
        winning_trades=(
            metrics[
                "winning_trades"
            ]
        ),
        losing_trades=(
            metrics[
                "losing_trades"
            ]
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
        expectancy=(
            metrics[
                "expectancy"
            ]
        ),
        downside_deviation=(
            metrics[
                "downside_deviation"
            ]
        ),
        sortino_ratio=(
            metrics[
                "sortino_ratio"
            ]
        ),
        calmar_ratio=(
            metrics[
                "calmar_ratio"
            ]
        ),
        exposure=(
            metrics[
                "exposure"
            ]
        ),
        total_transaction_cost=(
            metrics[
                "total_transaction_cost"
            ]
        ),
        total_slippage_cost=(
            metrics[
                "total_slippage_cost"
            ]
        ),
        best_trade=(
            metrics[
                "best_trade"
            ]
        ),
        worst_trade=(
            metrics[
                "worst_trade"
            ]
        ),
    )


# ----------------------------------------------------------------------
# Public compatibility helper
# ----------------------------------------------------------------------


def calculate_position_size(
    capital: float,
    entry_price: float,
    stop_price: float,
    risk_fraction: float,
) -> float:
    """
    Public compatibility wrapper for fixed-risk long position sizing.

    For short positions, the internal directional sizing function is
    used by the backtester.
    """

    return _position_size(
        capital=capital,
        entry_price=entry_price,
        stop_price=stop_price,
        risk_fraction=risk_fraction,
    )


__all__ = [
    "BacktestConfig",
    "Trade",
    "BacktestResult",
    "run_backtest",
    "calculate_position_size",
]
