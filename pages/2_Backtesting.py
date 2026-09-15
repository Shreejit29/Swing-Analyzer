"""
Backtesting page for the AI Swing Stock Analyzer.

User inputs are intentionally kept simple:
    1. Stock
    2. Backtest period
    3. Prediction horizon

Risk-management parameters remain internal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Make project root importable when Streamlit runs /pages/*
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.data.downloader import download_data
from src.models.backtest import (
    BacktestConfig,
    run_backtest,
)


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Swing Analyzer - Backtesting",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("📊 Swing Stock Analyzer — Backtesting")

st.caption(
    "Walk-forward backtesting using the same feature and "
    "Gradient Boosting pipeline used by the Stock Analyzer."
)


# ---------------------------------------------------------
# User inputs
# ---------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    symbol = st.text_input(
        "Stock",
        value="RELIANCE.NS",
        help=(
            "Enter a Yahoo Finance symbol, "
            "for example RELIANCE.NS or TCS.NS."
        ),
    ).strip().upper()

with col2:
    period = st.selectbox(
        "Backtest Period",
        options=[
            "6mo",
            "1y",
            "2y",
            "3y",
            "5y",
        ],
        index=1,
    )

with col3:
    horizon = st.selectbox(
        "Prediction Horizon",
        options=[
            1,
            3,
            5,
            10,
            20,
        ],
        index=2,
        format_func=lambda x: f"{x} trading days",
    )


# ---------------------------------------------------------
# Internal strategy configuration
# ---------------------------------------------------------

config = BacktestConfig(
    initial_capital=100000.0,

    horizon=int(horizon),

    probability_threshold=0.60,

    stop_loss_pct=0.03,
    target_pct=0.06,

    risk_per_trade=0.01,

    transaction_cost=0.001,
    slippage=0.0005,

    retrain_every=20,
    min_train_rows=180,
)


# ---------------------------------------------------------
# Run button
# ---------------------------------------------------------

run_button = st.button(
    "🚀 Run Backtest",
    type="primary",
    use_container_width=True,
)


if run_button:

    if not symbol:
        st.error(
            "Please enter a stock symbol."
        )
        st.stop()

    # -----------------------------------------------------
    # Download data
    # -----------------------------------------------------

    with st.spinner(
        f"Downloading {symbol} data..."
    ):

        try:

            df = download_data(
                symbol,
                period=period,
                interval="1d",
            )

        except Exception as exc:

            st.error(
                f"Unable to download data: {exc}"
            )
            st.stop()

    if df is None or df.empty:

        st.error(
            "No market data was returned."
        )
        st.stop()

    st.success(
        f"Loaded {len(df):,} rows for {symbol}."
    )

    # -----------------------------------------------------
    # Run backtest
    # -----------------------------------------------------

    with st.spinner(
        "Running walk-forward backtest..."
    ):

        try:

            result = run_backtest(
                df,
                config=config,
            )

        except Exception as exc:

            st.error(
                f"Backtest failed: {exc}"
            )

            with st.expander(
                "Technical details"
            ):
                st.exception(exc)

            st.stop()

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    st.subheader("Backtest Results")

    initial_capital = float(
        result.get(
            "initial_capital",
            config.initial_capital,
        )
    )

    final_capital = float(
        result.get(
            "final_capital",
            initial_capital,
        )
    )

    total_return = float(
        result.get(
            "total_return",
            0.0,
        )
    )

    max_drawdown = float(
        result.get(
            "max_drawdown",
            0.0,
        )
    )

    trades = int(
        result.get(
            "trades",
            0,
        )
    )

    win_rate = float(
        result.get(
            "win_rate",
            0.0,
        )
    )

    profit_factor = float(
        result.get(
            "profit_factor",
            0.0,
        )
    )

    average_trade = float(
        result.get(
            "average_trade",
            0.0,
        )
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "Final Capital",
            f"₹{final_capital:,.0f}",
            delta=(
                f"{total_return * 100:.2f}%"
            ),
        )

    with m2:
        st.metric(
            "Total Return",
            f"{total_return * 100:.2f}%",
        )

    with m3:
        st.metric(
            "Max Drawdown",
            f"{max_drawdown * 100:.2f}%",
        )

    with m4:
        st.metric(
            "Trades",
            f"{trades}",
        )

    m5, m6, m7 = st.columns(3)

    with m5:
        st.metric(
            "Win Rate",
            f"{win_rate * 100:.1f}%",
        )

    with m6:

        if profit_factor == float("inf"):
            pf_text = "∞"
        else:
            pf_text = f"{profit_factor:.2f}"

        st.metric(
            "Profit Factor",
            pf_text,
        )

    with m7:
        st.metric(
            "Average Trade",
            f"₹{average_trade:,.2f}",
        )

    # -----------------------------------------------------
    # Equity curve
    # -----------------------------------------------------

    equity_curve = result.get(
        "equity_curve"
    )

    if (
        isinstance(
            equity_curve,
            pd.DataFrame,
        )
        and not equity_curve.empty
    ):

        st.subheader(
            "📈 Equity Curve"
        )

        equity_data = equity_curve.copy()

        if "equity" in equity_data.columns:

            st.line_chart(
                equity_data[
                    ["equity"]
                ],
                use_container_width=True,
            )

    # -----------------------------------------------------
    # Trade history
    # -----------------------------------------------------

    trades_df = result.get(
        "trades_df"
    )

    if (
        isinstance(
            trades_df,
            pd.DataFrame,
        )
        and not trades_df.empty
    ):

        st.subheader(
            "📋 Trade History"
        )

        display_df = trades_df.copy()

        # Format dates when possible.
        for column in [
            "entry_date",
            "exit_date",
        ]:

            if column in display_df.columns:

                try:
                    display_df[column] = (
                        pd.to_datetime(
                            display_df[column]
                        ).dt.strftime(
                            "%Y-%m-%d"
                        )
                    )

                except Exception:
                    pass

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        # -------------------------------------------------
        # Download trades
        # -------------------------------------------------

        csv_data = display_df.to_csv(
            index=False
        )

        st.download_button(
            "⬇️ Download Trade History",
            data=csv_data,
            file_name=(
                f"{symbol}_backtest_trades.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

    else:

        st.info(
            "No trades were generated during the selected period."
        )

    # -----------------------------------------------------
    # Strategy information
    # -----------------------------------------------------

    with st.expander(
        "ℹ️ Backtest Method"
    ):

        st.write(
            f"""
**Stock:** {symbol}

**Prediction horizon:** {horizon} trading days

**Initial capital:** ₹{initial_capital:,.0f}

The backtest uses walk-forward training. The model is trained
only on information available before each prediction point.

Internal strategy settings:

- Probability threshold: 60%
- Stop loss: 3%
- Target: 6%
- Risk per trade: 1%
- Transaction cost: 0.10%
- Slippage: 0.05%
- Model retraining interval: every 20 observations
"""
        )


# ---------------------------------------------------------
# Initial page message
# ---------------------------------------------------------

else:

    st.info(
        "Select a stock, backtest period and prediction horizon, "
        "then click **Run Backtest**."
    )
