"""
Walk-forward backtesting engine.

Uses the same feature pipeline and Gradient Boosting classifier
as the Stock Analyzer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.features.engine import build_features
from src.models.classifier import SwingClassifier


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

    def validate(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError(
                "initial_capital must be greater than zero."
            )

        if self.horizon < 1:
            raise ValueError(
                "horizon must be at least 1."
            )

        if not 0.5 <= self.probability_threshold <= 1:
            raise ValueError(
                "probability_threshold must be between 0.5 and 1."
            )

        if self.stop_loss_pct <= 0:
            raise ValueError(
                "stop_loss_pct must be greater than zero."
            )

        if self.target_pct <= 0:
            raise ValueError(
                "target_pct must be greater than zero."
            )

        if not 0 < self.risk_per_trade <= 1:
            raise ValueError(
                "risk_per_trade must be between 0 and 1."
            )

        if self.transaction_cost < 0:
            raise ValueError(
                "transaction_cost cannot be negative."
            )

        if self.slippage < 0:
            raise ValueError(
                "slippage cannot be negative."
            )

        if self.retrain_every < 1:
            raise ValueError(
                "retrain_every must be at least 1."
            )

        if self.min_train_rows < 30:
            raise ValueError(
                "min_train_rows must be at least 30."
            )


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize OHLCV column names to lowercase.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    out = df.copy()

    # Flatten MultiIndex columns.
    if isinstance(out.columns, pd.MultiIndex):
        flattened = []

        for col in out.columns:
            parts = [
                str(x).strip()
                for x in col
                if str(x).strip()
            ]

            flattened.append(
                "_".join(parts)
            )

        out.columns = flattened

    rename_map = {}

    for column in out.columns:

        clean = str(column).strip().lower()

        if clean in {
            "open",
            "open_price",
        }:
            rename_map[column] = "open"

        elif clean in {
            "high",
            "high_price",
        }:
            rename_map[column] = "high"

        elif clean in {
            "low",
            "low_price",
        }:
            rename_map[column] = "low"

        elif clean in {
            "close",
            "close_price",
            "adj close",
            "adj_close",
        }:
            rename_map[column] = "close"

        elif clean in {
            "volume",
            "vol",
        }:
            rename_map[column] = "volume"

    out = out.rename(
        columns=rename_map
    )

    # Remove duplicate columns.
    out = out.loc[
        :,
        ~out.columns.duplicated()
    ]

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        col
        for col in required
        if col not in out.columns
    ]

    if missing:
        raise ValueError(
            "Missing OHLCV columns: "
            + ", ".join(missing)
        )

    for col in required:
        out[col] = pd.to_numeric(
            out[col],
            errors="coerce",
        )

    out = out.dropna(
        subset=required
    )

    return out


def _extract_probability(
    prediction: Any,
) -> float:
    """
    Convert classifier probability output into
    a single probability for the bullish class.
    """

    if prediction is None:
        return 0.5

    if np.isscalar(prediction):
        try:
            value = float(prediction)

            if 0 <= value <= 1:
                return value

        except Exception:
            return 0.5

    try:
        array = np.asarray(
            prediction,
            dtype=float,
        )

        if array.ndim == 0:
            value = float(array)

        elif array.ndim == 1:
            if len(array) == 0:
                return 0.5

            if len(array) == 1:
                value = float(array[0])

            else:
                value = float(array[-1])

        else:
            row = array[-1]

            if len(row) == 1:
                value = float(row[0])

            else:
                value = float(row[1])

        return float(
            np.clip(
                value,
                0.0,
                1.0,
            )
        )

    except Exception:
        return 0.5


def _calculate_trade_result(
    data: pd.DataFrame,
    entry_index: int,
    entry_price: float,
    horizon: int,
    stop_loss_pct: float,
    target_pct: float,
) -> Dict[str, Any]:
    """
    Simulate a long trade after entry.

    Entry is assumed to occur at the next available open.
    """

    last_index = min(
        entry_index + horizon,
        len(data) - 1,
    )

    stop_price = (
        entry_price
        * (1.0 - stop_loss_pct)
    )

    target_price = (
        entry_price
        * (1.0 + target_pct)
    )

    exit_index = last_index
    exit_price = float(
        data["close"].iloc[last_index]
    )

    exit_reason = "HORIZON"

    for i in range(
        entry_index,
        last_index + 1,
    ):

        high = float(
            data["high"].iloc[i]
        )

        low = float(
            data["low"].iloc[i]
        )

        # Conservative assumption:
        # if both levels are touched on the
        # same candle, stop loss is assumed first.
        if low <= stop_price:

            exit_index = i
            exit_price = stop_price
            exit_reason = "STOP_LOSS"
            break

        if high >= target_price:

            exit_index = i
            exit_price = target_price
            exit_reason = "TARGET"
            break

    return {
        "exit_index": exit_index,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
    }


def run_backtest(
    df: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> Dict[str, Any]:
    """
    Run a walk-forward long-only backtest.

    The model is retrained periodically using only data
    available before the prediction point.
    """

    if config is None:
        config = BacktestConfig()

    config.validate()

    data = _normalize_ohlcv(df)

    if len(data) < config.min_train_rows + config.horizon + 10:
        raise ValueError(
            f"Not enough data for backtesting. "
            f"Need at least "
            f"{config.min_train_rows + config.horizon + 10} "
            f"rows, received {len(data)}."
        )

    # ---------------------------------------------------------
    # Build the exact same feature pipeline used by the model.
    # ---------------------------------------------------------

    features = build_features(
        data
    )

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Target:
    # future return after the requested horizon.
    features["future_return"] = (
        features["close"]
        .shift(-config.horizon)
        / features["close"]
        - 1.0
    )

    features["target"] = (
        features["future_return"] > 0
    ).astype(int)

    feature_rows = features.dropna(
        subset=["future_return"]
    ).copy()

    if len(feature_rows) < config.min_train_rows:
        raise ValueError(
            "Insufficient usable rows after feature construction."
        )

    capital = float(
        config.initial_capital
    )

    equity = []
    trades: List[Dict[str, Any]] = []

    model = None
    last_training_index = -config.retrain_every

    # ---------------------------------------------------------
    # Walk forward.
    # ---------------------------------------------------------

    start_index = config.min_train_rows

    i = start_index

    while i < len(feature_rows) - 1:

        # -----------------------------------------------------
        # Retrain model periodically.
        # -----------------------------------------------------

        if (
            model is None
            or i - last_training_index
            >= config.retrain_every
        ):

            training_data = feature_rows.iloc[
                :i
            ].copy()

            if len(training_data) < config.min_train_rows:
                i += 1
                continue

            model = SwingClassifier(
                horizon=config.horizon,
                probability_threshold=(
                    config.probability_threshold
                ),
                random_state=42,
                min_samples=config.min_train_rows,
            )

            try:
                model.fit(
                    training_data
                )

                last_training_index = i

            except Exception:
                model = None
                i += 1
                continue

        # -----------------------------------------------------
        # Current row.
        # -----------------------------------------------------

        current_row = feature_rows.iloc[
            i:i + 1
        ].copy()

        try:
            prediction = model.predict_proba(
                current_row
            )

            probability = _extract_probability(
                prediction
            )

        except Exception:
            probability = 0.5

        # -----------------------------------------------------
        # No trade.
        # -----------------------------------------------------

        if (
            probability
            < config.probability_threshold
        ):

            equity.append(
                {
                    "index": feature_rows.index[i],
                    "equity": capital,
                }
            )

            i += 1
            continue

        # -----------------------------------------------------
        # Enter on next day's open.
        # -----------------------------------------------------

        entry_index = i + 1

        if entry_index >= len(feature_rows):
            break

        entry_date = feature_rows.index[
            entry_index
        ]

        entry_price = float(
            feature_rows[
                "open"
            ].iloc[entry_index]
        )

        if not np.isfinite(entry_price) or entry_price <= 0:
            i += 1
            continue

        # Account for entry costs and slippage.
        effective_entry = (
            entry_price
            * (1.0 + config.slippage)
        )

        # -----------------------------------------------------
        # Position sizing.
        #
        # Risk amount / stop distance
        # -----------------------------------------------------

        risk_amount = (
            capital
            * config.risk_per_trade
        )

        stop_distance = (
            effective_entry
            * config.stop_loss_pct
        )

        if stop_distance <= 0:
            i += 1
            continue

        shares = (
            risk_amount
            / stop_distance
        )

        if shares <= 0:
            i += 1
            continue

        trade = _calculate_trade_result(
            feature_rows,
            entry_index,
            effective_entry,
            config.horizon,
            config.stop_loss_pct,
            config.target_pct,
        )

        exit_price = float(
            trade["exit_price"]
        )

        effective_exit = (
            exit_price
            * (1.0 - config.slippage)
        )

        gross_pnl = (
            effective_exit
            - effective_entry
        ) * shares

        transaction_cost = (
            (
                effective_entry
                + effective_exit
            )
            * shares
            * config.transaction_cost
        )

        net_pnl = (
            gross_pnl
            - transaction_cost
        )

        capital += net_pnl

        trade_record = {
            "entry_date": entry_date,
            "exit_date": feature_rows.index[
                trade["exit_index"]
            ],
            "entry_price": effective_entry,
            "exit_price": effective_exit,
            "shares": shares,
            "probability": probability,
            "pnl": net_pnl,
            "return_pct": (
                net_pnl
                / (
                    effective_entry
                    * shares
                )
                * 100.0
            ),
            "exit_reason": trade[
                "exit_reason"
            ],
            "capital_after": capital,
        }

        trades.append(
            trade_record
        )

        # Record equity.
        equity.append(
            {
                "index": feature_rows.index[
                    trade["exit_index"]
                ],
                "equity": capital,
            }
        )

        # Move beyond the completed trade.
        i = (
            trade["exit_index"]
            + 1
        )

    # ---------------------------------------------------------
    # Equity curve.
    # ---------------------------------------------------------

    if equity:

        equity_df = pd.DataFrame(
            equity
        )

        equity_df = (
            equity_df
            .drop_duplicates(
                subset=["index"],
                keep="last",
            )
            .set_index("index")
        )

    else:

        equity_df = pd.DataFrame(
            {
                "equity": [
                    config.initial_capital
                ]
            }
        )

    # ---------------------------------------------------------
    # Performance metrics.
    # ---------------------------------------------------------

    final_capital = float(
        capital
    )

    total_return = (
        final_capital
        / config.initial_capital
        - 1.0
    )

    equity_values = equity_df[
        "equity"
    ].astype(float)

    running_max = (
        equity_values
        .cummax()
    )

    drawdown = (
        equity_values
        / running_max
        - 1.0
    )

    max_drawdown = (
        float(drawdown.min())
        if len(drawdown)
        else 0.0
    )

    trades_df = pd.DataFrame(
        trades
    )

    if not trades_df.empty:

        wins = trades_df[
            "pnl"
        ] > 0

        win_rate = float(
            wins.mean()
        )

        gross_profit = float(
            trades_df.loc[
                trades_df["pnl"] > 0,
                "pnl",
            ].sum()
        )

        gross_loss = float(
            -trades_df.loc[
                trades_df["pnl"] < 0,
                "pnl",
            ].sum()
        )

        if gross_loss > 0:
            profit_factor = (
                gross_profit
                / gross_loss
            )
        else:
            profit_factor = float(
                "inf"
            )

        average_trade = float(
            trades_df["pnl"].mean()
        )

    else:

        win_rate = 0.0
        profit_factor = 0.0
        average_trade = 0.0

    return {
        "initial_capital": (
            config.initial_capital
        ),
        "final_capital": (
            final_capital
        ),
        "total_return": (
            total_return
        ),
        "max_drawdown": (
            max_drawdown
        ),
        "trades": len(trades),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "average_trade": average_trade,
        "equity_curve": equity_df,
        "trades_df": trades_df,
    }


def backtest(
    df: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> Dict[str, Any]:
    """
    Backward-compatible wrapper around run_backtest().
    """

    return run_backtest(
        df,
        config=config,
    )


__all__ = [
    "BacktestConfig",
    "run_backtest",
    "backtest",
]
