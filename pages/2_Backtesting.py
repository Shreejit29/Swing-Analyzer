"""
AI Swing Stock Analyzer
Walk-Forward Backtesting Dashboard

The dashboard evaluates the trading system chronologically and
displays portfolio, risk, trade and model diagnostics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.downloader import download_data
from src.models.backtest import (
    BacktestConfig,
    run_backtest,
)


# =====================================================================
# PAGE CONFIG
# =====================================================================

st.set_page_config(
    page_title="Swing Analyzer — Backtesting",
    page_icon="🧪",
    layout="wide",
)


# =====================================================================
# HEADER
# =====================================================================

st.title(
    "🧪 AI Swing Trading Backtest"
)

st.caption(
    "Walk-forward validation • Ensemble ML • "
    "Risk management • Transaction costs • Slippage"
)


# =====================================================================
# SIDEBAR
# =====================================================================

st.sidebar.header(
    "Backtest Settings"
)

ticker = st.sidebar.text_input(
    "Stock",
    value="RELIANCE.NS",
).strip().upper()

period = st.sidebar.selectbox(
    "Backtest Period",
    options=[
        "1y",
        "2y",
        "3y",
        "5y",
        "10y",
        "max",
    ],
    index=2,
)

horizon = st.sidebar.selectbox(
    "Prediction Horizon",
    options=[
        1,
        3,
        5,
        10,
        20,
    ],
    index=2,
)

st.sidebar.divider()

st.sidebar.caption(
    "Internal risk parameters"
)

probability_threshold = 0.60
stop_loss_pct = 0.03
target_pct = 0.06
risk_per_trade = 0.01
transaction_cost = 0.001
slippage = 0.0005
retrain_every = 20
min_train_rows = 180
min_model_agreement = 0.60

st.sidebar.info(
    f"""
**Probability threshold:** {probability_threshold:.0%}

**Stop loss:** {stop_loss_pct:.0%}

**Target:** {target_pct:.0%}

**Risk/trade:** {risk_per_trade:.0%}

**Transaction cost:** {transaction_cost:.2%}

**Slippage:** {slippage:.2%}

**Retraining:** every {retrain_every} observations

**Minimum model agreement:** {min_model_agreement:.0%}
"""
)


# =====================================================================
# VALIDATION
# =====================================================================

if not ticker:

    st.warning(
        "Enter a stock ticker."
    )

    st.stop()


# =====================================================================
# DATA DOWNLOAD
# =====================================================================

try:

    with st.spinner(
        f"Downloading {ticker} historical data..."
    ):

        data = download_data(
            ticker,
            period=period,
        )

except TypeError:

    # Compatibility with downloader versions
    # that only accept ticker.
    try:

        with st.spinner(
            f"Downloading {ticker} historical data..."
        ):

            data = download_data(
                ticker
            )

    except Exception as exc:

        st.error(
            f"Data download failed: {exc}"
        )

        st.exception(exc)

        st.stop()

except Exception as exc:

    st.error(
        f"Data download failed: {exc}"
    )

    st.exception(exc)

    st.stop()


if data is None or data.empty:

    st.error(
        "No historical data was returned."
    )

    st.stop()


# =====================================================================
# NORMALIZE DATA
# =====================================================================

data = data.copy()

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


# =====================================================================
# RUN BACKTEST
# =====================================================================

config = BacktestConfig(
    initial_capital=100000.0,
    horizon=int(
        horizon
    ),
    probability_threshold=(
        probability_threshold
    ),
    stop_loss_pct=stop_loss_pct,
    target_pct=target_pct,
    risk_per_trade=risk_per_trade,
    transaction_cost=transaction_cost,
    slippage=slippage,
    retrain_every=retrain_every,
    min_train_rows=min_train_rows,
    min_model_agreement=(
        min_model_agreement
    ),
)


try:

    with st.spinner(
        "Running walk-forward backtest..."
    ):

        result = run_backtest(
            data,
            config=config,
        )

except Exception as exc:

    st.error(
        f"Backtest failed: {exc}"
    )

    st.exception(exc)

    st.stop()


# =====================================================================
# EXTRACT RESULTS
# =====================================================================

metrics = result.get(
    "metrics",
    {},
)

trades = result.get(
    "trades",
    pd.DataFrame(),
)

equity_curve = result.get(
    "equity_curve",
    pd.Series(dtype=float),
)


# =====================================================================
# SUMMARY
# =====================================================================

st.subheader(
    f"{ticker} — Walk-Forward Results"
)

initial_capital = float(
    metrics.get(
        "initial_capital",
        100000.0,
    )
)

final_capital = float(
    metrics.get(
        "final_capital",
        initial_capital,
    )
)

total_return = float(
    metrics.get(
        "total_return",
        0.0,
    )
)

cagr = float(
    metrics.get(
        "cagr",
        0.0,
    )
)

max_drawdown_pct = float(
    metrics.get(
        "max_drawdown_pct",
        0.0,
    )
)

sharpe = float(
    metrics.get(
        "sharpe",
        0.0,
    )
)

total_trades = int(
    metrics.get(
        "total_trades",
        0,
    )
)

win_rate = float(
    metrics.get(
        "win_rate",
        0.0,
    )
)

profit_factor = float(
    metrics.get(
        "profit_factor",
        0.0,
    )
)


# =====================================================================
# TOP METRICS
# =====================================================================

c1, c2, c3, c4 = st.columns(
    4
)

with c1:

    st.metric(
        "Initial Capital",
        f"₹{initial_capital:,.0f}",
    )

with c2:

    st.metric(
        "Final Capital",
        f"₹{final_capital:,.0f}",
    )

with c3:

    st.metric(
        "Total Return",
        f"{total_return:.2%}",
    )

with c4:

    st.metric(
        "CAGR",
        f"{cagr:.2%}",
    )


c5, c6, c7, c8 = st.columns(
    4
)

with c5:

    st.metric(
        "Max Drawdown",
        f"{max_drawdown_pct:.2%}",
    )

with c6:

    st.metric(
        "Sharpe Ratio",
        f"{sharpe:.2f}",
    )

with c7:

    st.metric(
        "Win Rate",
        f"{win_rate:.2%}",
    )

with c8:

    if np.isfinite(
        profit_factor
    ):

        pf_text = f"{profit_factor:.2f}"

    else:

        pf_text = "∞"

    st.metric(
        "Profit Factor",
        pf_text,
    )


# =====================================================================
# EQUITY CURVE
# =====================================================================

st.subheader(
    "📈 Equity Curve"
)

if (
    isinstance(
        equity_curve,
        pd.Series,
    )
    and not equity_curve.empty
):

    equity_plot = equity_curve.copy()

    equity_plot = pd.to_numeric(
        equity_plot,
        errors="coerce",
    ).dropna()

    fig_equity = go.Figure()

    fig_equity.add_trace(
        go.Scatter(
            x=equity_plot.index,
            y=equity_plot.values,
            mode="lines",
            name="Portfolio Equity",
        )
    )

    fig_equity.add_hline(
        y=initial_capital,
        line_dash="dash",
        annotation_text="Initial Capital",
    )

    fig_equity.update_layout(
        height=450,
        xaxis_title="Date",
        yaxis_title="Portfolio Value (₹)",
        hovermode="x unified",
    )

    st.plotly_chart(
        fig_equity,
        use_container_width=True,
    )

else:

    st.info(
        "No equity-curve data available."
    )


# =====================================================================
# DRAWDOWN
# =====================================================================

st.subheader(
    "📉 Drawdown"

)

if (
    isinstance(
        equity_curve,
        pd.Series,
    )
    and not equity_curve.empty
):

    equity_plot = pd.to_numeric(
        equity_curve,
        errors="coerce",
    ).dropna()

    running_max = (
        equity_plot.cummax()
    )

    drawdown = (
        equity_plot
        / running_max
        - 1.0
    )

    fig_drawdown = go.Figure()

    fig_drawdown.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
        )
    )

    fig_drawdown.update_layout(
        height=350,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
    )

    st.plotly_chart(
        fig_drawdown,
        use_container_width=True,
    )


# =====================================================================
# TRADE STATISTICS
# =====================================================================

st.subheader(
    "Trade Statistics"
)

t1, t2, t3, t4, t5 = st.columns(
    5
)

average_trade = float(
    metrics.get(
        "average_trade",
        0.0,
    )
)

best_trade = float(
    metrics.get(
        "best_trade",
        0.0,
    )
)

worst_trade = float(
    metrics.get(
        "worst_trade",
        0.0,
    )
)

winning_trades = int(
    metrics.get(
        "winning_trades",
        0,
    )
)

losing_trades = int(
    metrics.get(
        "losing_trades",
        0,
    )
)

with t1:

    st.metric(
        "Total Trades",
        str(total_trades),
    )

with t2:

    st.metric(
        "Winning",
        str(winning_trades),
    )

with t3:

    st.metric(
        "Losing",
        str(losing_trades),
    )

with t4:

    st.metric(
        "Average Trade",
        f"{average_trade:.2%}",
    )

with t5:

    st.metric(
        "Worst Trade",
        f"{worst_trade:.2%}",
    )


# =====================================================================
# MODEL DIAGNOSTICS
# =====================================================================

st.subheader(
    "🤖 Ensemble Diagnostics"
)

average_probability = float(
    metrics.get(
        "average_probability",
        0.0,
    )
)

average_agreement = float(
    metrics.get(
        "average_model_agreement",
        0.0,
    )
)

d1, d2, d3, d4 = st.columns(
    4
)

with d1:

    st.metric(
        "Average P(UP)",
        f"{average_probability:.1%}",
    )

with d2:

    st.metric(
        "Average Agreement",
        f"{average_agreement:.1%}",
    )

with d3:

    st.metric(
        "Prediction Horizon",
        f"{horizon} day(s)",
    )

with d4:

    st.metric(
        "Retraining",
        f"Every {retrain_every}",
    )


# =====================================================================
# TRADE ANALYSIS
# =====================================================================

if (
    isinstance(
        trades,
        pd.DataFrame,
    )
    and not trades.empty
):

    # ---------------------------------------------------------------
    # TRADE RETURN DISTRIBUTION
    # ---------------------------------------------------------------

    st.subheader(
        "Trade Return Distribution"
    )

    trade_returns = pd.to_numeric(
        trades[
            "net_return"
        ],
        errors="coerce",
    ).dropna()

    if not trade_returns.empty:

        fig_returns = go.Figure()

        fig_returns.add_trace(
            go.Histogram(
                x=trade_returns * 100,
                nbinsx=25,
                name="Trade Returns",
            )
        )

        fig_returns.add_vline(
            x=0,
            line_dash="dash",
        )

        fig_returns.update_layout(
            height=350,
            xaxis_title="Net Trade Return (%)",
            yaxis_title="Number of Trades",
        )

        st.plotly_chart(
            fig_returns,
            use_container_width=True,
        )

    # ---------------------------------------------------------------
    # BUY / SELL ANALYSIS
    # ---------------------------------------------------------------

    st.subheader(
        "Performance by Direction"
    )

    direction_stats = (
        trades
        .groupby(
            "signal"
        )
        .agg(
            Trades=(
                "signal",
                "count",
            ),
            Average_Return=(
                "net_return",
                "mean",
            ),
            Total_PnL=(
                "pnl",
                "sum",
            ),
            Average_Probability=(
                "probability_up",
                "mean",
            ),
            Average_Agreement=(
                "model_agreement",
                "mean",
            ),
        )
        .reset_index()
    )

    if not direction_stats.empty:

        direction_display = (
            direction_stats.copy()
        )

        direction_display[
            "Average_Return"
        ] = (
            direction_display[
                "Average_Return"
            ]
            .map(
                lambda x: f"{x:.2%}"
            )
        )

        direction_display[
            "Average_Probability"
        ] = (
            direction_display[
                "Average_Probability"
            ]
            .map(
                lambda x: f"{x:.2%}"
            )
        )

        direction_display[
            "Average_Agreement"
        ] = (
            direction_display[
                "Average_Agreement"
            ]
            .map(
                lambda x: f"{x:.2%}"
            )
        )

        direction_display[
            "Total_PnL"
        ] = (
            direction_display[
                "Total_PnL"
            ]
            .map(
                lambda x: f"₹{x:,.2f}"
            )
        )

        st.dataframe(
            direction_display,
            use_container_width=True,
            hide_index=True,
        )

    # ---------------------------------------------------------------
    # EXIT REASONS
    # ---------------------------------------------------------------

    st.subheader(
        "Exit Analysis"
    )

    exit_stats = (
        trades[
            "exit_reason"
        ]
        .value_counts()
        .rename_axis(
            "Exit Reason"
        )
        .reset_index(
            name="Trades"
        )
    )

    e1, e2 = st.columns(
        2
    )

    with e1:

        st.dataframe(
            exit_stats,
            use_container_width=True,
            hide_index=True,
        )

    with e2:

        fig_exit = go.Figure()

        fig_exit.add_trace(
            go.Bar(
                x=exit_stats[
                    "Exit Reason"
                ],
                y=exit_stats[
                    "Trades"
                ],
                name="Exits",
            )
        )

        fig_exit.update_layout(
            height=300,
            xaxis_title="Exit Reason",
            yaxis_title="Number of Trades",
        )

        st.plotly_chart(
            fig_exit,
            use_container_width=True,
        )

    # ---------------------------------------------------------------
    # TRADE TABLE
    # ---------------------------------------------------------------

    with st.expander(
        "📋 Complete Trade Log"
    ):

        display_trades = trades.copy()

        percentage_columns = [
            "probability_up",
            "model_agreement",
            "gross_return",
            "net_return",
        ]

        for column in percentage_columns:

            if column in display_trades.columns:

                display_trades[
                    column
                ] = pd.to_numeric(
                    display_trades[
                        column
                    ],
                    errors="coerce",
                )

        st.dataframe(
            display_trades,
            use_container_width=True,
            hide_index=True,
        )

else:

    st.warning(
        "No trades were generated during the selected period."
    )


# =====================================================================
# BACKTEST CONFIGURATION
# =====================================================================

with st.expander(
    "⚙️ Backtest Configuration"
):

    config_rows = [
        {
            "Parameter": "Initial Capital",
            "Value": (
                f"₹{config.initial_capital:,.0f}"
            ),
        },
        {
            "Parameter": "Prediction Horizon",
            "Value": str(
                config.horizon
            ),
        },
        {
            "Parameter": "Probability Threshold",
            "Value": (
                f"{config.probability_threshold:.2%}"
            ),
        },
        {
            "Parameter": "Stop Loss",
            "Value": (
                f"{config.stop_loss_pct:.2%}"
            ),
        },
        {
            "Parameter": "Target",
            "Value": (
                f"{config.target_pct:.2%}"
            ),
        },
        {
            "Parameter": "Risk per Trade",
            "Value": (
                f"{config.risk_per_trade:.2%}"
            ),
        },
        {
            "Parameter": "Transaction Cost",
            "Value": (
                f"{config.transaction_cost:.2%}"
            ),
        },
        {
            "Parameter": "Slippage",
            "Value": (
                f"{config.slippage:.2%}"
            ),
        },
        {
            "Parameter": "Retrain Every",
            "Value": str(
                config.retrain_every
            ),
        },
        {
            "Parameter": "Minimum Training Rows",
            "Value": str(
                config.min_train_rows
            ),
        },
        {
            "Parameter": "Minimum Model Agreement",
            "Value": (
                f"{config.min_model_agreement:.2%}"
            ),
        },
    ]

    st.dataframe(
        pd.DataFrame(
            config_rows
        ),
        use_container_width=True,
        hide_index=True,
    )


# =====================================================================
# IMPORTANT VALIDATION NOTE
# =====================================================================

st.divider()

st.info(
    """
**How to interpret this backtest**

This is a historical simulation, not a guarantee of future performance.

The backtest uses chronological walk-forward training and enters
at the next trading day's open. Transaction costs and slippage are
included.

Historical news sentiment is deliberately not injected into this
backtest yet. Using today's news to explain historical trades would
introduce look-ahead bias. Historical sentiment will be added only
after we have a properly time-aligned historical news dataset.
"""
)


# =====================================================================
# DISCLAIMER
# =====================================================================

st.caption(
    "Educational/research software only. "
    "Past backtest performance does not guarantee future results. "
    "Actual execution may differ because of spreads, liquidity, "
    "slippage, market gaps and other real-world conditions."
)
