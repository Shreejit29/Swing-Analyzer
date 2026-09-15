"""
AI Swing Analyser - Backtesting Dashboard.

This page provides an interactive research backtesting interface.

Important:
    - A backtest is not proof of future profitability.
    - Final holdout data must remain protected.
    - Production approval is handled separately.
    - No random train/test split is used.
    - Costs and slippage are explicitly configurable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.data.downloader import download_daily
from src.data.quality import (
    clean_ohlcv,
    validate_ohlcv,
)
from src.models.backtest import (
    BacktestConfig,
    run_backtest,
)
from src.evaluation.plots import (
    calculate_drawdown,
    plot_drawdown,
    plot_equity_curve,
    plot_return_distribution,
)


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

ROOT_DIR = Path(
    __file__
).resolve().parents[1]


# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="AI Swing Analyser — Backtesting",
    page_icon="🧪",
    layout="wide",
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def normalize_symbol(
    symbol: str,
) -> str:
    symbol = (
        symbol.strip()
        .upper()
    )

    if not symbol:
        return ""

    if (
        symbol.endswith(
            ".NS"
        )
        or symbol.endswith(
            ".BO"
        )
    ):
        return symbol

    return (
        f"{symbol}.NS"
    )


def get_metric(
    metrics: Any,
    name: str,
    default: Any = None,
) -> Any:
    """
    Safely retrieve a metric from either a dataclass/object or dict.
    """

    if metrics is None:
        return default

    if isinstance(
        metrics,
        dict,
    ):
        return metrics.get(
            name,
            default,
        )

    return getattr(
        metrics,
        name,
        default,
    )


def metric_percent(
    value: Any,
) -> str:
    try:
        return (
            f"{float(value) * 100:.2f}%"
        )
    except (
        TypeError,
        ValueError,
    ):
        return "N/A"


def metric_number(
    value: Any,
    decimals: int = 2,
) -> str:
    try:
        return (
            f"{float(value):.{decimals}f}"
        )
    except (
        TypeError,
        ValueError,
    ):
        return "N/A"


def extract_trade_returns(
    result: Any,
) -> pd.Series:
    """
    Extract trade returns from the backtest result.

    The backtest API is still intentionally defensive here because the
    research engine is evolving.
    """

    values: list[
        float
    ] = []

    trades = getattr(
        result,
        "trades",
        [],
    )

    for trade in trades:
        value = getattr(
            trade,
            "return_pct",
            None,
        )

        if value is None:
            value = getattr(
                trade,
                "return_fraction",
                None,
            )

        if value is None:
            value = getattr(
                trade,
                "pnl_pct",
                None,
            )

        if value is None:
            continue

        try:
            values.append(
                float(
                    value
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

    return pd.Series(
        values,
        dtype=float,
    )


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------

st.title(
    "🧪 Backtesting"
)

st.caption(
    "Historical trading simulation with explicit costs, slippage and risk controls."
)


# ----------------------------------------------------------------------
# Research warning
# ----------------------------------------------------------------------

st.warning(
    "Backtesting is a research tool. Strong historical results do not "
    "guarantee future performance. The production approval engine "
    "requires additional validation, robustness and holdout gates."
)


# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------

st.sidebar.header(
    "Backtest Configuration"
)

symbol_input = st.sidebar.text_input(
    "NSE Symbol",
    value="RELIANCE",
)

symbol = normalize_symbol(
    symbol_input
)

period = st.sidebar.selectbox(
    "Historical Period",
    [
        "1y",
        "2y",
        "5y",
        "10y",
    ],
    index=1,
)

initial_capital = st.sidebar.number_input(
    "Initial Capital (₹)",
    min_value=10_000.0,
    value=100_000.0,
    step=10_000.0,
)

probability_threshold = st.sidebar.slider(
    "Signal Probability Threshold",
    min_value=0.50,
    max_value=0.95,
    value=0.60,
    step=0.01,
)

stop_loss_pct = st.sidebar.slider(
    "Stop Loss (%)",
    min_value=0.50,
    max_value=10.0,
    value=3.0,
    step=0.25,
)

take_profit_pct = st.sidebar.slider(
    "Take Profit (%)",
    min_value=1.0,
    max_value=30.0,
    value=6.0,
    step=0.50,
)

max_holding_period = st.sidebar.number_input(
    "Maximum Holding Period (bars)",
    min_value=1,
    max_value=100,
    value=10,
    step=1,
)

risk_per_trade = st.sidebar.slider(
    "Risk Per Trade (%)",
    min_value=0.10,
    max_value=5.0,
    value=1.0,
    step=0.10,
)

transaction_cost_pct = st.sidebar.number_input(
    "Transaction Cost (%)",
    min_value=0.0,
    max_value=2.0,
    value=0.10,
    step=0.01,
)

slippage_pct = st.sidebar.number_input(
    "Slippage (%)",
    min_value=0.0,
    max_value=2.0,
    value=0.05,
    step=0.01,
)

run_button = st.sidebar.button(
    "▶ Run Backtest",
    type="primary",
    use_container_width=True,
)


# ----------------------------------------------------------------------
# Validation of controls
# ----------------------------------------------------------------------

if stop_loss_pct <= 0:
    st.error(
        "Stop loss must be greater than zero."
    )
    st.stop()

if take_profit_pct <= 0:
    st.error(
        "Take profit must be greater than zero."
    )
    st.stop()

if (
    take_profit_pct
    / stop_loss_pct
    < 2.0
):
    st.warning(
        "Configured reward/risk is below the project's default "
        "2.0 minimum. This does not automatically invalidate a "
        "research backtest, but it is unlikely to pass production "
        "approval."
    )


# ----------------------------------------------------------------------
# Empty state
# ----------------------------------------------------------------------

if not symbol:
    st.info(
        "Enter an NSE symbol."
    )
    st.stop()

if not run_button:
    st.info(
        "Configure the parameters and click **Run Backtest**."
    )
    st.stop()


# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------

with st.spinner(
    f"Downloading historical data for {symbol}..."
):
    try:
        raw_data = download_daily(
            symbol,
            period=period,
        )
    except Exception as exc:
        st.error(
            "Historical data download failed."
        )
        st.exception(
            exc
        )
        st.stop()


if raw_data.empty:
    st.error(
        "No historical data was returned."
    )
    st.stop()


# ----------------------------------------------------------------------
# Data quality
# ----------------------------------------------------------------------

try:
    quality = validate_ohlcv(
        raw_data
    )
except Exception as exc:
    st.error(
        "Data quality validation failed."
    )
    st.exception(
        exc
    )
    st.stop()


if not quality.passed:
    st.error(
        "The historical dataset failed quality checks."
    )

    if quality.errors:
        for error in quality.errors:
            st.write(
                f"• {error}"
            )

    st.stop()


data = clean_ohlcv(
    raw_data
)

if len(data) < 100:
    st.error(
        "At least 100 usable observations are required for this "
        "research backtest."
    )
    st.stop()


# ----------------------------------------------------------------------
# Signal construction
# ----------------------------------------------------------------------

st.subheader(
    "Signal Source"
)

st.info(
    "The current research backtester requires a `Probability` column. "
    "Until an approved model is connected, this page cannot honestly "
    "claim that the probability represents an AI forecast."
)


# ----------------------------------------------------------------------
# Existing model probability
# ----------------------------------------------------------------------

if (
    "Probability"
    not in data.columns
):
    st.warning(
        "No model probability series is available for this dataset."
    )

    st.markdown(
        """
        **Current status**

        ```text
        Historical OHLCV
              ↓
        Data quality
              ↓
        Backtest engine
              ↓
        ❌ Model probability unavailable
        ```

        The backtester will not create artificial probabilities merely
        to produce an attractive performance curve.
        """
    )

    st.stop()


# ----------------------------------------------------------------------
# Backtest configuration
# ----------------------------------------------------------------------

config = BacktestConfig(
    initial_capital=float(
        initial_capital
    ),
    probability_threshold=float(
        probability_threshold
    ),
    stop_loss_pct=float(
        stop_loss_pct
        / 100.0
    ),
    take_profit_pct=float(
        take_profit_pct
        / 100.0
    ),
    max_holding_period=int(
        max_holding_period
    ),
    transaction_cost_pct=float(
        transaction_cost_pct
        / 100.0
    ),
    slippage_pct=float(
        slippage_pct
        / 100.0
    ),
    risk_per_trade=float(
        risk_per_trade
        / 100.0
    ),
)


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------

with st.spinner(
    "Running historical trading simulation..."
):
    try:
        result = run_backtest(
            data,
            config=config,
        )

    except Exception as exc:
        st.error(
            "Backtest execution failed."
        )
        st.exception(
            exc
        )
        st.stop()


# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------

st.success(
    "Backtest completed."
)

metrics = getattr(
    result,
    "metrics",
    None,
)


# ----------------------------------------------------------------------
# Key metrics
# ----------------------------------------------------------------------

st.subheader(
    "Performance Summary"
)

col1, col2, col3, col4, col5, col6 = st.columns(
    6
)

with col1:
    st.metric(
        "Total Return",
        metric_percent(
            get_metric(
                metrics,
                "total_return",
            )
        ),
    )

with col2:
    st.metric(
        "Annualized Return",
        metric_percent(
            get_metric(
                metrics,
                "annualized_return",
            )
        ),
    )

with col3:
    st.metric(
        "Sharpe Ratio",
        metric_number(
            get_metric(
                metrics,
                "sharpe_ratio",
            )
        ),
    )

with col4:
    st.metric(
        "Max Drawdown",
        metric_percent(
            get_metric(
                metrics,
                "max_drawdown",
            )
        ),
    )

with col5:
    st.metric(
        "Win Rate",
        metric_percent(
            get_metric(
                metrics,
                "win_rate",
            )
        ),
    )

with col6:
    trades = get_metric(
        metrics,
        "number_of_trades",
        get_metric(
            metrics,
            "trades",
            0,
        ),
    )

    st.metric(
        "Trades",
        metric_number(
            trades,
            0,
        ),
    )


# ----------------------------------------------------------------------
# Profit factor / risk-reward
# ----------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(
    4
)

with col1:
    st.metric(
        "Profit Factor",
        metric_number(
            get_metric(
                metrics,
                "profit_factor",
            )
        ),
    )

with col2:
    st.metric(
        "Average Trade",
        metric_percent(
            get_metric(
                metrics,
                "average_trade_return",
            )
        ),
    )

with col3:
    st.metric(
        "Volatility",
        metric_percent(
            get_metric(
                metrics,
                "annualized_volatility",
            )
        ),
    )

with col4:
    st.metric(
        "Configured R:R",
        metric_number(
            take_profit_pct
            / stop_loss_pct
        ),
    )


# ----------------------------------------------------------------------
# Equity curve
# ----------------------------------------------------------------------

st.subheader(
    "Equity Curve"
)

equity = getattr(
    result,
    "equity_curve",
    None,
)

if equity is not None:
    try:
        equity_series = pd.Series(
            equity,
            index=(
                data.index[-len(equity):]
                if len(equity) <= len(data)
                else None
            ),
        )

        figure = plot_equity_curve(
            equity_series
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

    except Exception:
        st.line_chart(
            equity
        )
else:
    st.info(
        "Equity curve unavailable."
    )


# ----------------------------------------------------------------------
# Drawdown
# ----------------------------------------------------------------------

st.subheader(
    "Drawdown"
)

if equity is not None:
    try:
        equity_series = pd.Series(
            equity,
            index=(
                data.index[-len(equity):]
                if len(equity) <= len(data)
                else None
            ),
        )

        drawdown = calculate_drawdown(
            equity_series
        )

        figure = plot_drawdown(
            drawdown
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

    except Exception:
        st.info(
            "Unable to construct drawdown chart."
        )


# ----------------------------------------------------------------------
# Trade return distribution
# ----------------------------------------------------------------------

st.subheader(
    "Trade Return Distribution"
)

trade_returns = extract_trade_returns(
    result
)

if not trade_returns.empty:
    try:
        figure = plot_return_distribution(
            trade_returns
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )
    except Exception:
        st.bar_chart(
            trade_returns
        )
else:
    st.info(
        "No trade-return distribution is available."
    )


# ----------------------------------------------------------------------
# Trades
# ----------------------------------------------------------------------

st.subheader(
    "Trades"
)

trades = getattr(
    result,
    "trades",
    [],
)

if trades:
    trade_rows: list[
        dict[str, Any]
    ] = []

    for trade in trades:
        row: dict[
            str,
            Any,
        ] = {}

        for field in [
            "entry_time",
            "entry_timestamp",
            "exit_time",
            "exit_timestamp",
            "entry_price",
            "exit_price",
            "quantity",
            "return_pct",
            "return_fraction",
            "pnl",
            "exit_reason",
        ]:
            value = getattr(
                trade,
                field,
                None,
            )

            if value is not None:
                row[field] = value

        if row:
            trade_rows.append(
                row
            )

    if trade_rows:
        trade_table = pd.DataFrame(
            trade_rows
        )

        st.dataframe(
            trade_table,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "Trade objects were generated but no displayable fields "
            "were found."
        )
else:
    st.info(
        "No trades were generated under the current parameters."
    )


# ----------------------------------------------------------------------
# Research
