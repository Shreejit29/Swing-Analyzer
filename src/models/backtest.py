from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.engine import build_features
from .classifier import SwingClassifier


@dataclass
class BacktestConfig:
    initial_capital: float = 100000.0
    horizon: int = 5

    probability_threshold: float = 0.60
    stop_loss_pct: float = 0.03
    target_pct: float = 0.06

    risk_per_trade: float = 0.01
    transaction_cost: float = 0.001
    slippage: float = 0.0005

    retrain_every: int = 20
    min_train_rows: int = 180


def _validate_config(cfg: BacktestConfig):
    if cfg.initial_capital <= 0:
        raise ValueError("Initial capital must be greater than zero.")

    if cfg.horizon < 1:
        raise ValueError("Prediction horizon must be at least 1 day.")

    if not 0.50 <= cfg.probability_threshold <= 0.95:
        raise ValueError(
            "Probability threshold must be between 0.50 and 0.95."
        )

    if cfg.stop_loss_pct <= 0:
        raise ValueError("Stop loss percentage must be greater than zero.")

    if cfg.target_pct <= 0:
        raise ValueError("Target percentage must be greater than zero.")

    if cfg.risk_per_trade <= 0:
        raise ValueError("Risk per trade must be greater than zero.")

    if cfg.retrain_every < 1:
        raise ValueError("Retraining interval must be at least 1.")

    if cfg.min_train_rows < 30:
        raise ValueError("Minimum training rows must be at least 30.")


def _execution_cost(cfg: BacktestConfig) -> float:
    return (
        cfg.transaction_cost * 2.0
        + cfg.slippage * 2.0
    )


def _open_position(
    side: str,
    next_open: float,
    capital: float,
    cfg: BacktestConfig,
    entry_date,
):
    if side == "BUY":
        entry = next_open * (1.0 + cfg.slippage)
        stop = entry * (1.0 - cfg.stop_loss_pct)
        target = entry * (1.0 + cfg.target_pct)
    else:
        entry = next_open * (1.0 - cfg.slippage)
        stop = entry * (1.0 + cfg.stop_loss_pct)
        target = entry * (1.0 - cfg.target_pct)

    risk_amount = capital * cfg.risk_per_trade
    risk_per_share = abs(entry - stop)

    if risk_per_share <= 0:
        return None

    quantity = risk_amount / risk_per_share

    return {
        "side": side,
        "entry": float(entry),
        "stop": float(stop),
        "target": float(target),
        "quantity": float(quantity),
        "date": entry_date,
        "bars": 0,
    }


def _close_position(
    position,
    exit_price,
    date,
    reason,
    cfg: BacktestConfig,
):
    entry = position["entry"]
    quantity = position["quantity"]

    if position["side"] == "BUY":
        gross_pnl = (
            exit_price - entry
        ) * quantity
    else:
        gross_pnl = (
            entry - exit_price
        ) * quantity

    trading_cost = (
        entry
        * quantity
        * _execution_cost(cfg)
    )

    net_pnl = gross_pnl - trading_cost

    invested = entry * quantity

    trade_return = (
        net_pnl / invested
        if invested > 0
        else 0.0
    )

    return {
        "entry_date": position["date"],
        "exit_date": date,
        "side": position["side"],
        "entry": float(entry),
        "exit": float(exit_price),
        "quantity": float(quantity),
        "return": float(trade_return),
        "pnl": float(net_pnl),
        "reason": reason,
        "bars": position["bars"],
    }


def run_backtest(
    df: pd.DataFrame,
    config: BacktestConfig | None = None,
):
    """
    Walk-forward backtest.

    Signals are generated using information available at the
    current candle and executed at the following day's open.
    """

    cfg = config or BacktestConfig()
    _validate_config(cfg)

    if df is None or df.empty:
        raise ValueError("No market data supplied for backtesting.")

    # Central feature pipeline also normalizes OHLCV.
    data = build_features(df).copy()

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        col for col in required
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            "Backtest data is missing: "
            + ", ".join(missing)
        )

    minimum_rows = (
        cfg.min_train_rows
        + cfg.horizon
        + 10
    )

    if len(data) < minimum_rows:
        raise ValueError(
            f"Need at least {minimum_rows} rows for backtesting. "
            f"Only {len(data)} rows are available."
        )

    capital = float(cfg.initial_capital)

    equity_records = []
    trades = []

    model = None
    last_train_index = -10**9
    position = None

    last_processed_index = (
        len(data)
        - cfg.horizon
        - 1
    )

    for i in range(
        cfg.min_train_rows,
        last_processed_index + 1,
    ):
        row = data.iloc[i]
        date = data.index[i]

        # ---------------------------------------------------------
        # Manage open position
        # ---------------------------------------------------------
        if position is not None:
            position["bars"] += 1

            high = float(row["high"])
            low = float(row["low"])

            exit_price = None
            reason = None

            if position["side"] == "BUY":

                # Conservative assumption:
                # stop is considered hit first if both occur.
                if low <= position["stop"]:
                    exit_price = position["stop"]
                    reason = "STOP"

                elif high >= position["target"]:
                    exit_price = position["target"]
                    reason = "TARGET"

            else:

                if high >= position["stop"]:
                    exit_price = position["stop"]
                    reason = "STOP"

                elif low <= position["target"]:
                    exit_price = position["target"]
                    reason = "TARGET"

            # Time-based exit.
            if (
                exit_price is None
                and position["bars"] >= cfg.horizon
            ):
                exit_price = float(row["close"])
                reason = "TIME"

            if exit_price is not None:

                trade = _close_position(
                    position,
                    exit_price,
                    date,
                    reason,
                    cfg,
                )

                capital += trade["pnl"]
                trades.append(trade)

                position = None

        # ---------------------------------------------------------
        # Retrain model
        # ---------------------------------------------------------
        if (
            position is None
            and (
                model is None
                or i - last_train_index
                >= cfg.retrain_every
            )
        ):

            try:
                training_data = data.iloc[: i + 1][
                    required
                ]

                model = SwingClassifier(
                    horizon=cfg.horizon,
                    probability_threshold=(
                        cfg.probability_threshold
                    ),
                    min_samples=cfg.min_train_rows,
                )

                model.fit(training_data)

                last_train_index = i

            except Exception:
                model = None

        # ---------------------------------------------------------
        # Generate signal
        # ---------------------------------------------------------
        if position is None and model is not None:

            current_data = data.iloc[: i + 1][
                required
            ]

            try:
                p_up = model.predict_proba(
                    current_data
                )
            except Exception:
                p_up = None

            # Handle either scalar or array probability.
            if p_up is not None:

                if isinstance(p_up, (list, tuple, np.ndarray)):
                    arr = np.asarray(p_up)

                    if arr.ndim == 2:
                        p_up = float(arr[-1, 1])
                    else:
                        p_up = float(arr[-1])

                else:
                    p_up = float(p_up)

                if p_up >= cfg.probability_threshold:
                    side = "BUY"

                elif (
                    1.0 - p_up
                    >= cfg.probability_threshold
                ):
                    side = "SELL"

                else:
                    side = None

                # -------------------------------------------------
                # Enter next day
                # -------------------------------------------------
                if side is not None:

                    next_row = data.iloc[i + 1]
                    next_open = float(
                        next_row["open"]
                    )

                    if (
                        np.isfinite(next_open)
                        and next_open > 0
                    ):
                        position = _open_position(
                            side,
                            next_open,
                            capital,
                            cfg,
                            data.index[i + 1],
                        )

        # ---------------------------------------------------------
        # Mark-to-market equity
        # ---------------------------------------------------------
        equity = capital

        if position is not None:

            current_close = float(
                row["close"]
            )

            if position["side"] == "BUY":
                unrealized = (
                    current_close
                    - position["entry"]
                ) * position["quantity"]

            else:
                unrealized = (
                    position["entry"]
                    - current_close
                ) * position["quantity"]

            equity += unrealized

        equity_records.append(
            {
                "date": date,
                "equity": float(equity),
            }
        )

    # -------------------------------------------------------------
    # Force close remaining position
    # -------------------------------------------------------------
    if position is not None:

        final_date = data.index[-1]
        final_close = float(
            data["close"].iloc[-1]
        )

        trade = _close_position(
            position,
            final_close,
            final_date,
            "END",
            cfg,
        )

        capital += trade["pnl"]
        trades.append(trade)

        if equity_records:
            equity_records[-1]["equity"] = capital

    # -------------------------------------------------------------
    # Equity curve
    # -------------------------------------------------------------
    equity_df = pd.DataFrame(
        equity_records
    )

    if not equity_df.empty:

        equity_df = equity_df.set_index(
            "date"
        )

        peak = equity_df[
            "equity"
        ].cummax()

        drawdown = (
            equity_df["equity"] / peak
        ) - 1.0

        max_drawdown = float(
            drawdown.min()
        )

        total_return = (
            capital
            / cfg.initial_capital
        ) - 1.0

    else:

        equity_df = pd.DataFrame(
            columns=["equity"]
        )

        max_drawdown = 0.0
        total_return = 0.0

    # -------------------------------------------------------------
    # Trade statistics
    # -------------------------------------------------------------
    trades_df = pd.DataFrame(trades)

    if not trades_df.empty:

        wins = trades_df[
            trades_df["pnl"] > 0
        ]

        losses = trades_df[
            trades_df["pnl"] < 0
        ]

        win_rate = (
            len(wins)
            / len(trades_df)
        )

        gross_profit = float(
            wins["pnl"].sum()
        )

        gross_loss = abs(
            float(losses["pnl"].sum())
        )

        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else np.inf
        )

        average_trade = float(
            trades_df["return"].mean()
        )

    else:

        win_rate = 0.0
        profit_factor = 0.0
        average_trade = 0.0

    return {
        "initial_capital": float(
            cfg.initial_capital
        ),
        "final_capital": float(
            capital
        ),
        "total_return": float(
            total_return
        ),
        "max_drawdown": float(
            max_drawdown
        ),
        "trades": int(
            len(trades_df)
        ),
        "win_rate": float(
            win_rate
        ),
        "profit_factor": float(
            profit_factor
        ),
        "average_trade": float(
            average_trade
        ),
        "equity_curve": equity_df,
        "trades_df": trades_df,
    }


def backtest(
    df: pd.DataFrame,
    **kwargs,
):
    """
    Compatibility wrapper.
    """
    return run_backtest(
        df,
        BacktestConfig(**kwargs),
    )


__all__ = [
    "BacktestConfig",
    "run_backtest",
    "backtest",
]
