"""
Walk-forward backtesting engine for the Swing Analyzer.

Convention:
    OHLCV columns are lowercase:
        open, high, low, close, volume

The backtester:
    - avoids look-ahead bias
    - enters on the next day's open
    - uses fixed fractional risk
    - accounts for transaction costs and slippage
    - applies stop-loss / target
    - force-closes open positions at the end
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class Trade:
    entry_date: Any
    exit_date: Any
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    return_pct: float
    reason: str


def _get_column(df: pd.DataFrame, name: str) -> pd.Series:
    """Case-insensitive column lookup."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")

    lookup = {str(c).lower(): c for c in df.columns}

    if name.lower() in lookup:
        return pd.to_numeric(df[lookup[name.lower()]], errors="coerce")

    raise ValueError(f"Required column '{name}' not found.")


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean lowercase OHLCV DataFrame."""
    out = pd.DataFrame(index=df.index)

    for column in ["open", "high", "low", "close", "volume"]:
        out[column] = _get_column(df, column)

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["open", "high", "low", "close"])

    return out


def _extract_signal(row: pd.Series) -> int:
    """
    Convert common signal formats into:

        1  = long
        0  = no position
       -1  = short

    The current Swing Analyzer is primarily long-oriented, but the
    function supports common signal column names.
    """
    for column in ["signal", "Signal", "prediction", "Prediction"]:
        if column in row.index:
            value = row[column]

            if isinstance(value, str):
                text = value.upper().strip()

                if text in {"BUY", "LONG", "1", "STRONG BUY"}:
                    return 1

                if text in {"SELL", "SHORT", "-1", "STRONG SELL"}:
                    return -1

                return 0

            try:
                numeric = float(value)

                if numeric > 0:
                    return 1

                if numeric < 0:
                    return -1

            except (TypeError, ValueError):
                pass

    return 0


def _position_size(
    capital: float,
    risk_per_trade: float,
    entry_price: float,
    stop_price: float,
) -> float:
    """
    Calculate position size from the amount of capital at risk.

    Example:
        capital = ₹100,000
        risk = 1%
        entry = ₹100
        stop = ₹97

        risk amount = ₹1,000
        risk/share = ₹3
        shares ≈ 333
    """
    risk_amount = capital * risk_per_trade
    risk_per_share = abs(entry_price - stop_price)

    if risk_amount <= 0 or risk_per_share <= 0:
        return 0.0

    return max(0.0, risk_amount / risk_per_share)


def _apply_slippage(price: float, side: int, is_entry: bool, slippage: float) -> float:
    """Apply percentage slippage."""
    if side == 1:
        # Long:
        # entry becomes more expensive, exit becomes cheaper.
        if is_entry:
            return price * (1 + slippage)
        return price * (1 - slippage)

    # Short:
    # entry becomes cheaper, exit becomes more expensive.
    if is_entry:
        return price * (1 - slippage)

    return price * (1 + slippage)


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = 100000.0,
    risk_per_trade: float = 0.01,
    stop_loss_pct: float = 0.03,
    target_pct: float = 0.06,
    transaction_cost_pct: float = 0.001,
    slippage_pct: float = 0.0005,
) -> Dict[str, Any]:
    """
    Run a signal-based backtest.

    Parameters
    ----------
    df:
        OHLCV DataFrame. Must contain a signal column.

    initial_capital:
        Starting capital.

    risk_per_trade:
        Fraction of current equity risked on each trade.

    stop_loss_pct:
        Stop distance from entry.

    target_pct:
        Target distance from entry.

    transaction_cost_pct:
        Round-trip transaction-cost approximation.

    slippage_pct:
        Slippage applied to executions.

    Returns
    -------
    dict
        Contains trades, equity curve and summary statistics.
    """
    if df is None or len(df) < 3:
        raise ValueError("At least 3 rows of data are required for backtesting.")

    data = _normalize_ohlcv(df)

    # Preserve signal columns from original dataframe.
    for column in df.columns:
        if column not in data.columns:
            data[column] = df.loc[data.index, column]

    data = data.sort_index()

    capital = float(initial_capital)
    equity = capital

    trades: List[Trade] = []
    equity_records: List[Dict[str, Any]] = []

    position: Optional[Dict[str, Any]] = None

    for i in range(len(data)):
        row = data.iloc[i]
        date = data.index[i]

        open_price = float(row["open"])
        high_price = float(row["high"])
        low_price = float(row["low"])
        close_price = float(row["close"])

        # -----------------------------------------------------
        # Manage an existing position.
        # -----------------------------------------------------
        if position is not None:
            side = position["side"]
            stop_price = position["stop_price"]
            target_price = position["target_price"]

            exit_price = None
            reason = None

            if side == 1:
                stop_hit = low_price <= stop_price
                target_hit = high_price >= target_price

                # Conservative assumption:
                # if both are touched on the same candle,
                # assume stop was hit first.
                if stop_hit:
                    exit_price = stop_price
                    reason = "STOP"

                elif target_hit:
                    exit_price = target_price
                    reason = "TARGET"

            else:
                stop_hit = high_price >= stop_price
                target_hit = low_price <= target_price

                if stop_hit:
                    exit_price = stop_price
                    reason = "STOP"

                elif target_hit:
                    exit_price = target_price
                    reason = "TARGET"

            # Force close on final candle.
            if i == len(data) - 1 and exit_price is None:
                exit_price = close_price
                reason = "END"

            if exit_price is not None:
                executed_exit = _apply_slippage(
                    exit_price,
                    side,
                    False,
                    slippage_pct,
                )

                entry_price = position["entry_price"]
                shares = position["shares"]

                gross_pnl = (
                    executed_exit - entry_price
                ) * shares * side

                entry_value = abs(entry_price * shares)
                exit_value = abs(executed_exit * shares)

                transaction_cost = (
                    entry_value + exit_value
                ) * transaction_cost_pct

                pnl = gross_pnl - transaction_cost

                equity += pnl

                return_pct = (
                    pnl / entry_value
                    if entry_value > 0
                    else 0.0
                )

                trades.append(
                    Trade(
                        entry_date=position["entry_date"],
                        exit_date=date,
                        entry_price=entry_price,
                        exit_price=executed_exit,
                        shares=shares,
                        pnl=float(pnl),
                        return_pct=float(return_pct),
                        reason=reason,
                    )
                )

                position = None

        # -----------------------------------------------------
        # Generate a new entry.
        #
        # Important:
        # Signal on today's candle is executed at the NEXT
        # candle's open. This prevents look-ahead bias.
        # -----------------------------------------------------
        if position is None and i < len(data) - 1:
            signal = _extract_signal(row)

            if signal != 0:
                next_row = data.iloc[i + 1]
                next_open = float(next_row["open"])

                executed_entry = _apply_slippage(
                    next_open,
                    signal,
                    True,
                    slippage_pct,
                )

                if signal == 1:
                    stop_price = executed_entry * (1 - stop_loss_pct)
                    target_price = executed_entry * (1 + target_pct)
                else:
                    stop_price = executed_entry * (1 + stop_loss_pct)
                    target_price = executed_entry * (1 - target_pct)

                shares = _position_size(
                    capital=equity,
                    risk_per_trade=risk_per_trade,
                    entry_price=executed_entry,
                    stop_price=stop_price,
                )

                # Never allocate more than available capital.
                max_shares = (
                    equity / executed_entry
                    if executed_entry > 0
                    else 0.0
                )

                shares = min(shares, max_shares)

                if shares > 0:
                    position = {
                        "side": signal,
                        "entry_date": data.index[i + 1],
                        "entry_price": executed_entry,
                        "shares": shares,
                        "stop_price": stop_price,
                        "target_price": target_price,
                    }

        equity_records.append(
            {
                "date": date,
                "equity": equity,
            }
        )

    equity_curve = pd.DataFrame(equity_records)

    # ---------------------------------------------------------
    # Performance statistics
    # ---------------------------------------------------------
    trade_df = pd.DataFrame(
        [
            {
                "entry_date": trade.entry_date,
                "exit_date": trade.exit_date,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "shares": trade.shares,
                "pnl": trade.pnl,
                "return_pct": trade.return_pct,
                "reason": trade.reason,
            }
            for trade in trades
        ]
    )

    if trades:
        wins = [trade for trade in trades if trade.pnl > 0]
        losses = [trade for trade in trades if trade.pnl < 0]

        winning_pnl = sum(trade.pnl for trade in wins)
        losing_pnl = abs(sum(trade.pnl for trade in losses))

        win_rate = len(wins) / len(trades)

        if losing_pnl > 0:
            profit_factor = winning_pnl / losing_pnl
        else:
            profit_factor = np.inf if winning_pnl > 0 else 0.0

        avg_trade = np.mean([trade.pnl for trade in trades])

    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade = 0.0

    if not equity_curve.empty:
        equity_values = equity_curve["equity"]

        running_max = equity_values.cummax()

        drawdown = (
            equity_values - running_max
        ) / running_max.replace(0, np.nan)

        max_drawdown = float(drawdown.min())

        final_equity = float(equity_values.iloc[-1])

    else:
        max_drawdown = 0.0
        final_equity = capital

    total_return = (
        final_equity / initial_capital - 1
        if initial_capital > 0
        else 0.0
    )

    summary = {
        "initial_capital": float(initial_capital),
        "final_equity": final_equity,
        "total_return": float(total_return),
        "total_return_pct": float(total_return * 100),
        "total_trades": len(trades),
        "winning_trades": sum(trade.pnl > 0 for trade in trades),
        "losing_trades": sum(trade.pnl < 0 for trade in trades),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "average_trade_pnl": float(avg_trade),
        "max_drawdown": float(max_drawdown),
        "max_drawdown_pct": float(max_drawdown * 100),
    }

    return {
        "summary": summary,
        "trades": trade_df,
        "equity_curve": equity_curve,
    }


def backtest(
    df: pd.DataFrame,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Compatibility wrapper.

    Allows existing code to call:

        backtest(df)

    while using the new engine internally.
    """
    return run_backtest(df, **kwargs)


__all__ = [
    "Trade",
    "run_backtest",
    "backtest",
]
