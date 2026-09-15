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


def run_backtest(df: pd.DataFrame, config: BacktestConfig | None = None):
    cfg = config or BacktestConfig()
    data = build_features(df).copy()
    if len(data) < cfg.min_train_rows + cfg.horizon + 5:
        raise ValueError(f"Need at least {cfg.min_train_rows + cfg.horizon + 5} rows for backtesting.")

    capital = float(cfg.initial_capital)
    equity = []
    trades = []
    model = None
    last_train = -10**9
    position = None

    for i in range(cfg.min_train_rows, len(data) - cfg.horizon - 1):
        row = data.iloc[i]
        date = data.index[i]

        if position is not None:
            position["bars"] += 1
            high, low = float(row.high), float(row.low)
            exit_price = None
            reason = None
            if position["side"] == "BUY":
                if low <= position["stop"]:
                    exit_price, reason = position["stop"], "STOP"
                elif high >= position["target"]:
                    exit_price, reason = position["target"], "TARGET"
            else:
                if high >= position["stop"]:
                    exit_price, reason = position["stop"], "STOP"
                elif low <= position["target"]:
                    exit_price, reason = position["target"], "TARGET"
            if exit_price is None and position["bars"] >= cfg.horizon:
                exit_price, reason = float(row.close), "TIME"
            if exit_price is not None:
                gross = ((exit_price - position["entry"]) / position["entry"] if position["side"] == "BUY" else (position["entry"] - exit_price) / position["entry"])
                cost = cfg.transaction_cost * 2 + cfg.slippage * 2
                net = gross - cost
                pnl = position["capital_risk"] * net / cfg.stop_loss_pct if cfg.stop_loss_pct > 0 else position["capital_risk"] * net
                capital += pnl
                trades.append({"entry_date": position["date"], "exit_date": date, "side": position["side"], "entry": position["entry"], "exit": exit_price, "return": net, "pnl": pnl, "reason": reason, "bars": position["bars"]})
                position = None

        if position is None and (i - last_train >= cfg.retrain_every or model is None):
            try:
                model = SwingClassifier(horizon=cfg.horizon, probability_threshold=cfg.probability_threshold, min_samples=cfg.min_train_rows)
                model.fit(data.iloc[: i + 1][["open", "high", "low", "close", "volume"]])
                last_train = i
            except Exception:
                model = None

        if position is None and model is not None:
            p_up = model.predict_proba(data.iloc[: i + 1][["open", "high", "low", "close", "volume"]])
            side = "BUY" if p_up >= cfg.probability_threshold else "SELL" if (1 - p_up) >= cfg.probability_threshold else None
            if side:
                next_open = float(data.iloc[i + 1].open)
                if side == "BUY":
                    entry = next_open * (1 + cfg.slippage)
                    stop = entry * (1 - cfg.stop_loss_pct)
                    target = entry * (1 + cfg.target_pct)
                else:
                    entry = next_open * (1 - cfg.slippage)
                    stop = entry * (1 + cfg.stop_loss_pct)
                    target = entry * (1 - cfg.target_pct)
                position = {"side": side, "entry": entry, "stop": stop, "target": target, "date": data.index[i + 1], "bars": 0, "capital_risk": capital * cfg.risk_per_trade}

        equity.append({"date": date, "equity": capital})

    equity_df = pd.DataFrame(equity).set_index("date") if equity else pd.DataFrame(columns=["equity"])
    trades_df = pd.DataFrame(trades)
    if not equity_df.empty:
        peak = equity_df["equity"].cummax()
        drawdown = equity_df["equity"] / peak - 1
        max_dd = float(drawdown.min())
        total_return = equity_df["equity"].iloc[-1] / cfg.initial_capital - 1
    else:
        max_dd, total_return = 0.0, 0.0
    if not trades_df.empty:
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] < 0]
        win_rate = len(wins) / len(trades_df)
        gross_profit = wins["pnl"].sum()
        gross_loss = abs(losses["pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss else np.inf
        avg_trade = float(trades_df["return"].mean())
    else:
        win_rate, profit_factor, avg_trade = 0.0, 0.0, 0.0
    result = {
        "initial_capital": cfg.initial_capital,
        "final_capital": capital,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "trades": len(trades_df),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "average_trade": avg_trade,
        "equity_curve": equity_df,
        "trades_df": trades_df,
    }
    return result
