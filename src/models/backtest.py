"""
Walk-Forward Swing Trading Backtester

Features:
- Chronological walk-forward training
- Ensemble ML model
- Model agreement filter
- Next-day-open entry
- Stop-loss / target / horizon exits
- Transaction costs
- Slippage
- Risk-based position sizing
- Equity curve
- Drawdown
- Sharpe
- CAGR
- Win rate
- Profit factor

Historical sentiment is intentionally not used here yet because
using today's news against historical trades would create
look-ahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.features.engine import (
    build_features as engine_build_features
)

from src.models.classifier import (
    SwingClassifier
)


# =====================================================================
# CONFIGURATION
# =====================================================================


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

    min_model_agreement: float = 0.60

    max_positions: int = 1


# =====================================================================
# HELPERS
# =====================================================================


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:

        result = float(value)

        if np.isfinite(result):
            return result

    except Exception:
        pass

    return float(default)


# =====================================================================
# OHLCV NORMALIZATION
# =====================================================================


def _normalize_ohlcv(
    df: pd.DataFrame,
) -> pd.DataFrame:

    data = df.copy()

    if isinstance(
        data.columns,
        pd.MultiIndex,
    ):

        data.columns = [
            "_".join(
                str(x)
                for x in column
                if str(x).lower() != "nan"
            ).strip("_")
            for column in data.columns
        ]

    data.columns = [
        str(column)
        .strip()
        .lower()
        for column in data.columns
    ]

    rename_map = {}

    for column in data.columns:

        if column in {
            "adj_close",
            "adjusted_close",
        }:

            rename_map[column] = "close"

    data = data.rename(
        columns=rename_map
    )

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:

        raise ValueError(
            "Missing OHLCV columns: "
            + ", ".join(missing)
        )

    for column in required:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    data = data.dropna(
        subset=required
    )

    data = data.sort_index()

    data = data[
        ~data.index.duplicated(
            keep="last"
        )
    ]

    if data.empty:

        raise ValueError(
            "No valid OHLCV data remains."
        )

    return data


# =====================================================================
# EQUITY METRICS
# =====================================================================


def _calculate_equity_metrics(
    equity_curve: pd.Series,
    initial_capital: float,
) -> Dict[str, float]:

    if (
        equity_curve is None
        or equity_curve.empty
    ):

        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "volatility": 0.0,
            "sharpe": 0.0,
        }

    equity = pd.to_numeric(
        equity_curve,
        errors="coerce",
    ).dropna()

    if equity.empty:

        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "volatility": 0.0,
            "sharpe": 0.0,
        }

    final_equity = float(
        equity.iloc[-1]
    )

    total_return = (
        final_equity
        / initial_capital
        - 1.0
    )

    running_max = equity.cummax()

    drawdown = (
        equity
        / running_max
        - 1.0
    )

    max_drawdown_pct = float(
        drawdown.min()
    )

    max_drawdown = float(
        initial_capital
        * abs(max_drawdown_pct)
    )

    returns = (
        equity
        .pct_change()
        .dropna()
    )

    if len(returns) >= 2:

        volatility = float(
            returns.std()
            * np.sqrt(252)
        )

        std_return = float(
            returns.std()
        )

        if std_return > 0:

            sharpe = float(
                returns.mean()
                / std_return
                * np.sqrt(252)
            )

        else:

            sharpe = 0.0

    else:

        volatility = 0.0
        sharpe = 0.0

    try:

        days = (
            equity.index[-1]
            - equity.index[0]
        ).days

        years = (
            days / 365.25
        )

        if (
            years > 0
            and final_equity > 0
        ):

            cagr = float(
                (
                    final_equity
                    / initial_capital
                )
                ** (1.0 / years)
                - 1.0
            )

        else:

            cagr = 0.0

    except Exception:

        cagr = 0.0

    return {
        "total_return": float(
            total_return
        ),
        "cagr": float(
            cagr
        ),
        "max_drawdown": float(
            max_drawdown
        ),
        "max_drawdown_pct": float(
            max_drawdown_pct
        ),
        "volatility": float(
            volatility
        ),
        "sharpe": float(
            sharpe
        ),
    }


# =====================================================================
# EXIT SIMULATION
# =====================================================================


def _simulate_exit(
    data: pd.DataFrame,
    entry_position: int,
    entry_price: float,
    signal: str,
    stop_loss_pct: float,
    target_pct: float,
    horizon: int,
) -> Dict[str, Any]:

    last_position = len(data) - 1

    max_exit_position = min(
        entry_position + horizon,
        last_position,
    )

    for position in range(
        entry_position,
        max_exit_position + 1,
    ):

        row = data.iloc[position]

        high = float(
            row["high"]
        )

        low = float(
            row["low"]
        )

        # ------------------------------------------------------------
        # LONG
        # ------------------------------------------------------------

        if signal == "BUY":

            stop_price = (
                entry_price
                * (1.0 - stop_loss_pct)
            )

            target_price = (
                entry_price
                * (1.0 + target_pct)
            )

            # Conservative assumption:
            # if both stop and target are hit in the same candle,
            # assume the stop was hit first.
            if low <= stop_price:

                return {
                    "exit_position": position,
                    "exit_price": stop_price,
                    "exit_reason": "STOP LOSS",
                }

            if high >= target_price:

                return {
                    "exit_position": position,
                    "exit_price": target_price,
                    "exit_reason": "TARGET",
                }

        # ------------------------------------------------------------
        # SHORT
        # ------------------------------------------------------------

        elif signal == "SELL":

            stop_price = (
                entry_price
                * (1.0 + stop_loss_pct)
            )

            target_price = (
                entry_price
                * (1.0 - target_pct)
            )

            if high >= stop_price:

                return {
                    "exit_position": position,
                    "exit_price": stop_price,
                    "exit_reason": "STOP LOSS",
                }

            if low <= target_price:

                return {
                    "exit_position": position,
                    "exit_price": target_price,
                    "exit_reason": "TARGET",
                }

    exit_position = max_exit_position

    exit_price = float(
        data.iloc[
            exit_position
        ]["close"]
    )

    if exit_position >= last_position:

        exit_reason = "END OF DATA"

    else:

        exit_reason = "HORIZON"

    return {
        "exit_position": exit_position,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
    }


# =====================================================================
# TRADE RETURN
# =====================================================================


def _calculate_trade_return(
    signal: str,
    entry_price: float,
    exit_price: float,
    transaction_cost: float,
    slippage: float,
) -> Dict[str, float]:

    entry_price = float(
        entry_price
    )

    exit_price = float(
        exit_price
    )

    if signal == "BUY":

        effective_entry = (
            entry_price
            * (1.0 + slippage)
        )

        effective_exit = (
            exit_price
            * (1.0 - slippage)
        )

        gross_return = (
            effective_exit
            / effective_entry
            - 1.0
        )

    else:

        effective_entry = (
            entry_price
            * (1.0 - slippage)
        )

        effective_exit = (
            exit_price
            * (1.0 + slippage)
        )

        gross_return = (
            effective_entry
            / effective_exit
            - 1.0
        )

    net_return = (
        gross_return
        - 2.0 * transaction_cost
    )

    return {
        "gross_return": float(
            gross_return
        ),
        "net_return": float(
            net_return
        ),
    }


# =====================================================================
# WALK-FORWARD BACKTEST
# =====================================================================


def run_backtest(
    df: pd.DataFrame,
    config: Optional[
        BacktestConfig
    ] = None,
) -> Dict[str, Any]:

    if config is None:

        config = BacktestConfig()

    # ---------------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------------

    if config.horizon < 1:

        raise ValueError(
            "Horizon must be at least 1."
        )

    if config.retrain_every < 1:

        raise ValueError(
            "Retrain interval must be at least 1."
        )

    if not (
        0.50
        <= config.probability_threshold
        < 1.0
    ):

        raise ValueError(
            "Probability threshold must be "
            "between 0.50 and 0.99."
        )

    # ---------------------------------------------------------------
    # DATA
    # ---------------------------------------------------------------

    data = _normalize_ohlcv(
        df
    )

    minimum_required = (
        config.min_train_rows
        + config.horizon
        + 10
    )

    if len(data) < minimum_required:

        raise ValueError(
            "Not enough historical data for "
            "walk-forward backtesting. "
            f"Need at least {minimum_required} "
            f"rows, got {len(data)}."
        )

    # ---------------------------------------------------------------
    # FEATURES
    # ---------------------------------------------------------------

    features = engine_build_features(
        data
    )

    if (
        features is None
        or features.empty
    ):

        raise ValueError(
            "Feature engineering returned no data."
        )

    common_index = (
        data.index.intersection(
            features.index
        )
    )

    data = data.loc[
        common_index
    ].copy()

    features = features.loc[
        common_index
    ].copy()

    data = data.sort_index()

    features = features.sort_index()

    # ---------------------------------------------------------------
    # PORTFOLIO
    # ---------------------------------------------------------------

    capital = float(
        config.initial_capital
    )

    trades = []

    equity_dates = []

    equity_values = []

    cached_model = None

    last_retrain_position = -10**9

    last_model_position = -1

    position_open_until = -1

    # ---------------------------------------------------------------
    # WALK FORWARD
    # ---------------------------------------------------------------

    start_position = (
        config.min_train_rows
    )

    for signal_position in range(
        start_position,
        len(data) - 1,
    ):

        current_date = data.index[
            signal_position
        ]

        # -----------------------------------------------------------
        # Record equity at current decision date.
        # -----------------------------------------------------------

        equity_dates.append(
            current_date
        )

        equity_values.append(
            capital
        )

        # -----------------------------------------------------------
        # Existing position.
        # -----------------------------------------------------------

        if (
            signal_position
            <= position_open_until
        ):

            continue

        # -----------------------------------------------------------
        # Retraining.
        # -----------------------------------------------------------

        should_retrain = (
            cached_model is None
            or (
                signal_position
                - last_retrain_position
                >= config.retrain_every
            )
        )

        if should_retrain:

            # Training data ends at current decision date.
            #
            # SwingClassifier removes the final `horizon` rows
            # because their future outcome is not yet known.
            train_features = features.iloc[
                : signal_position + 1
            ].copy()

            try:

                model = SwingClassifier(
                    horizon=config.horizon,
                    probability_threshold=(
                        config.probability_threshold
                    ),
                    min_samples=80,
                )

                model.fit(
                    train_features
                )

                cached_model = model

                last_retrain_position = (
                    signal_position
                )

                last_model_position = (
                    signal_position
                )

            except Exception:

                continue

        if cached_model is None:

            continue

        # -----------------------------------------------------------
        # Current prediction.
        # -----------------------------------------------------------

        current_features = features.iloc[
            [signal_position]
        ].copy()

        try:

            probability = float(
                cached_model
                .predict_proba(
                    current_features
                )[0, 1]
            )

        except Exception:

            continue

        probability = float(
            np.clip(
                probability,
                0.0,
                1.0,
            )
        )

        # -----------------------------------------------------------
        # Agreement.
        # -----------------------------------------------------------

        try:

            agreement_info = (
                cached_model
                .model_agreement(
                    current_features
                )
            )

            agreement = _safe_float(
                agreement_info.get(
                    "agreement",
                    0.0,
                ),
                0.0,
            )

        except Exception:

            agreement = 0.0

        # -----------------------------------------------------------
        # Direction.
        # -----------------------------------------------------------

        if probability >= (
            config.probability_threshold
        ):

            signal = "BUY"

        elif probability <= (
            1.0
            - config.probability_threshold
        ):

            signal = "SELL"

        else:

            signal = "WAIT"

        # -----------------------------------------------------------
        # Agreement filter.
        # -----------------------------------------------------------

        if (
            signal != "WAIT"
            and agreement
            < config.min_model_agreement
        ):

            signal = "WAIT"

        if signal == "WAIT":

            continue

        # -----------------------------------------------------------
        # NEXT DAY ENTRY
        # -----------------------------------------------------------

        entry_position = (
            signal_position + 1
        )

        if entry_position >= len(data):

            break

        entry_date = data.index[
            entry_position
        ]

        entry_price = float(
            data.iloc[
                entry_position
            ]["open"]
        )

        if (
            not np.isfinite(
                entry_price
            )
            or entry_price <= 0
        ):

            continue

        # -----------------------------------------------------------
        # RISK-BASED POSITION SIZE
        # -----------------------------------------------------------

        risk_amount = (
            capital
            * config.risk_per_trade
        )

        stop_distance = (
            entry_price
            * config.stop_loss_pct
        )

        if (
            stop_distance <= 0
            or not np.isfinite(
                stop_distance
            )
        ):

            continue

        quantity = (
            risk_amount
            / stop_distance
        )

        if (
            quantity <= 0
            or not np.isfinite(
                quantity
            )
        ):

            continue

        # -----------------------------------------------------------
        # EXIT
        # -----------------------------------------------------------

        exit_info = _simulate_exit(
            data=data,
            entry_position=entry_position,
            entry_price=entry_price,
            signal=signal,
            stop_loss_pct=(
                config.stop_loss_pct
            ),
            target_pct=(
                config.target_pct
            ),
            horizon=config.horizon,
        )

        exit_position = int(
            exit_info[
                "exit_position"
            ]
        )

        exit_price = float(
            exit_info[
                "exit_price"
            ]
        )

        exit_date = data.index[
            exit_position
        ]

        exit_reason = str(
            exit_info[
                "exit_reason"
            ]
        )

        # -----------------------------------------------------------
        # PNL
        # -----------------------------------------------------------

        return_info = (
            _calculate_trade_return(
                signal=signal,
                entry_price=entry_price,
                exit_price=exit_price,
                transaction_cost=(
                    config.transaction_cost
                ),
                slippage=config.slippage,
            )
        )

        net_return = float(
            return_info[
                "net_return"
            ]
        )

        pnl = (
            capital
            * net_return
        )

        capital_before = capital

        capital = max(
            0.0,
            capital + pnl,
        )

        # -----------------------------------------------------------
        # RECORD
        # -----------------------------------------------------------

        trades.append(
            {
                "signal_date": current_date,

                "entry_date": entry_date,

                "exit_date": exit_date,

                "signal": signal,

                "probability_up": probability,

                "model_agreement": agreement,

                "entry_price": entry_price,

                "exit_price": exit_price,

                "quantity": quantity,

                "gross_return": return_info[
                    "gross_return"
                ],

                "net_return": net_return,

                "pnl": pnl,

                "capital_before": (
                    capital_before
                ),

                "capital_after": capital,

                "exit_reason": exit_reason,

                "model_retrained": (
                    last_model_position
                    == signal_position
                ),
            }
        )

        # -----------------------------------------------------------
        # Block overlapping positions.
        # -----------------------------------------------------------

        position_open_until = (
            exit_position
        )

    # =================================================================
    # FINAL EQUITY
    # =================================================================

    if not equity_dates:

        equity_curve = pd.Series(
            [config.initial_capital],
            index=[
                data.index[0]
            ],
            name="equity",
        )

    else:

        equity_dates = list(
            equity_dates
        )

        equity_values = list(
            equity_values
        )

        equity_dates.append(
            data.index[-1]
        )

        equity_values.append(
            capital
        )

        equity_curve = pd.Series(
            equity_values,
            index=equity_dates,
            name="equity",
        )

        equity_curve = (
            equity_curve[
                ~equity_curve.index.duplicated(
                    keep="last"
                )
            ]
            .sort_index()
        )

    # =================================================================
    # TRADES
    # =================================================================

    trades_df = pd.DataFrame(
        trades
    )

    # =================================================================
    # EQUITY METRICS
    # =================================================================

    equity_metrics = (
        _calculate_equity_metrics(
            equity_curve,
            config.initial_capital,
        )
    )

    # =================================================================
    # TRADE METRICS
    # =================================================================

    if trades_df.empty:

        winning_trades = 0
        losing_trades = 0
        win_rate = 0.0
        profit_factor = 0.0
        average_trade = 0.0
        best_trade = 0.0
        worst_trade = 0.0
        average_probability = 0.0
        average_agreement = 0.0

    else:

        trade_returns = pd.to_numeric(
            trades_df[
                "net_return"
            ],
            errors="coerce",
        ).fillna(0.0)

        winning = (
            trade_returns > 0
        )

        losing = (
            trade_returns < 0
        )

        winning_trades = int(
            winning.sum()
        )

        losing_trades = int(
            losing.sum()
        )

        win_rate = float(
            winning.mean()
        )

        gross_profit = float(
            trade_returns[
                winning
            ].sum()
        )

        gross_loss = abs(
            float(
                trade_returns[
                    losing
                ].sum()
            )
        )

        if gross_loss > 0:

            profit_factor = (
                gross_profit
                / gross_loss
            )

        elif gross_profit > 0:

            profit_factor = float(
                "inf"
            )

        else:

            profit_factor = 0.0

        average_trade = float(
            trade_returns.mean()
        )

        best_trade = float(
            trade_returns.max()
        )

        worst_trade = float(
            trade_returns.min()
        )

        average_probability = float(
            trades_df[
                "probability_up"
            ].mean()
        )

        average_agreement = float(
            trades_df[
                "model_agreement"
            ].mean()
        )

    # =================================================================
    # METRICS
    # =================================================================

    metrics = {
        **equity_metrics,

        "initial_capital": float(
            config.initial_capital
        ),

        "final_capital": float(
            capital
        ),

        "total_trades": int(
            len(trades_df)
        ),

        "winning_trades": int(
            winning_trades
        ),

        "losing_trades": int(
            losing_trades
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

        "best_trade": float(
            best_trade
        ),

        "worst_trade": float(
            worst_trade
        ),

        "average_probability": float(
            average_probability
        ),

        "average_model_agreement": float(
            average_agreement
        ),
    }

    # =================================================================
    # RESULT
    # =================================================================

    return {
        "metrics": metrics,

        "trades": trades_df,

        "equity_curve": equity_curve,

        "features": features,

        "config": config,

        "final_capital": float(
            capital
        ),

        "total_return": float(
            equity_metrics[
                "total_return"
            ]
        ),

        "max_drawdown": float(
            equity_metrics[
                "max_drawdown_pct"
            ]
        ),
    }


# =====================================================================
# BACKWARD-COMPATIBLE FUNCTION
# =====================================================================


def backtest(
    df: pd.DataFrame,
    horizon: int = 5,
    probability_threshold: float = 0.60,
    stop_loss_pct: float = 0.03,
    target_pct: float = 0.06,
    risk_per_trade: float = 0.01,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
    retrain_every: int = 20,
    min_train_rows: int = 180,
    **kwargs: Any,
) -> Dict[str, Any]:

    config = BacktestConfig(
        horizon=int(
            horizon
        ),
        probability_threshold=float(
            probability_threshold
        ),
        stop_loss_pct=float(
            stop_loss_pct
        ),
        target_pct=float(
            target_pct
        ),
        risk_per_trade=float(
            risk_per_trade
        ),
        transaction_cost=float(
            transaction_cost
        ),
        slippage=float(
            slippage
        ),
        retrain_every=int(
            retrain_every
        ),
        min_train_rows=int(
            min_train_rows
        ),
    )

    return run_backtest(
        df,
        config=config,
    )


# =====================================================================
# SAFE FEATURE FUNCTION
# =====================================================================


def build_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compatibility function.

    This directly calls the central feature engine.
    No recursive reference.
    """

    return engine_build_features(
        df
    )


__all__ = [
    "BacktestConfig",
    "run_backtest",
    "backtest",
    "build_features",
]
