# pages/1_Stock_Analyzer.py

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.downloader import download_data
from src.models.predictor import analyze_stock


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Stock Analyzer",
    page_icon="📈",
    layout="wide",
)


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .metric-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        background: rgba(128,128,128,0.06);
        min-height: 120px;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }

    .signal-buy {
        font-size: 2rem;
        font-weight: 800;
    }

    .small-label {
        font-size: 0.85rem;
        opacity: 0.7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

st.title("📈 AI Swing Stock Analyzer")
st.caption(
    "Technical analysis + market structure + multi-timeframe confirmation "
    "using Gradient Boosting."
)


# ---------------------------------------------------------------------
# User inputs
# ---------------------------------------------------------------------

col1, col2 = st.columns([2, 1])

with col1:
    symbol = st.text_input(
        "Stock",
        value="RELIANCE.NS",
        help="Use Yahoo Finance symbols, e.g. RELIANCE.NS, TCS.NS, INFY.NS",
    ).strip().upper()

with col2:
    horizon = st.selectbox(
        "Prediction Horizon",
        options=[1, 3, 5, 10, 20],
        index=2,
        format_func=lambda x: f"{x} trading days",
    )


analyze_clicked = st.button(
    "🔍 Analyze Stock",
    type="primary",
    use_container_width=True,
)


# ---------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------

if analyze_clicked or symbol:

    if not symbol:
        st.warning("Please enter a stock symbol.")
        st.stop()

    with st.spinner(f"Analyzing {symbol}..."):

        try:
            df = download_data(
                symbol,
                period="1y",
                interval="1d",
            )

            if df is None or df.empty:
                st.error(
                    "No market data was returned. "
                    "Check the stock symbol and try again."
                )
                st.stop()

            prediction = analyze_stock(
                df,
                horizon=horizon,
                probability_threshold=0.60,
                stop_loss_pct=0.03,
                target_pct=0.06,
            )

        except Exception as exc:
            st.error("Analysis failed.")
            st.exception(exc)
            st.stop()


    # -----------------------------------------------------------------
    # Extract result safely
    # -----------------------------------------------------------------

    price = float(
        getattr(
            prediction,
            "current_price",
            getattr(prediction, "price", 0.0),
        )
        or 0.0
    )

    probability_up = float(
        getattr(prediction, "probability_up", 0.5)
        or 0.5
    )

    probability_down = float(
        getattr(prediction, "probability_down", 0.5)
        or 0.5
    )

    confidence = float(
        getattr(prediction, "confidence", 0.0)
        or 0.0
    )

    signal = str(
        getattr(prediction, "signal", "WAIT")
    ).upper()

    regime = str(
        getattr(prediction, "regime", "UNKNOWN")
    ).upper()

    trend = str(
        getattr(prediction, "trend", "UNKNOWN")
    ).upper()

    momentum = str(
        getattr(prediction, "momentum", "UNKNOWN")
    ).upper()

    volume = str(
        getattr(prediction, "volume", "UNKNOWN")
    ).upper()

    weekly_trend = str(
        getattr(prediction, "weekly_trend", "UNKNOWN")
    ).upper()

    monthly_trend = str(
        getattr(prediction, "monthly_trend", "UNKNOWN")
    ).upper()

    # -----------------------------------------------------------------
    # Main summary
    # -----------------------------------------------------------------

    st.divider()

    st.subheader(f"{symbol}")

    # -----------------------------------------------------------------
    # Top metrics
    # -----------------------------------------------------------------

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            "Current Price",
            f"₹{price:,.2f}",
        )

    with c2:
        st.metric(
            "AI Signal",
            signal,
        )

    with c3:
        st.metric(
            "Probability Up",
            f"{probability_up:.1%}",
        )

    with c4:
        st.metric(
            "Probability Down",
            f"{probability_down:.1%}",
        )

    with c5:
        st.metric(
            "Confidence",
            f"{confidence:.1%}",
        )


    # -----------------------------------------------------------------
    # Market Structure
    # -----------------------------------------------------------------

    st.markdown(
        '<div class="section-title">📐 Market Structure</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown("**Trend**")
        st.markdown(f"### {trend}")

    with m2:
        st.markdown("**Momentum**")
        st.markdown(f"### {momentum}")

    with m3:
        st.markdown("**Volume**")
        st.markdown(f"### {volume}")


    # -----------------------------------------------------------------
    # Regime / MTF
    # -----------------------------------------------------------------

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric(
            "Market Regime",
            regime,
        )

    with r2:
        st.metric(
            "Weekly Trend",
            weekly_trend,
        )

    with r3:
        st.metric(
            "Monthly Trend",
            monthly_trend,
        )


    # -----------------------------------------------------------------
    # Price Chart
    # -----------------------------------------------------------------

    st.markdown(
        '<div class="section-title">📊 Price Chart</div>',
        unsafe_allow_html=True,
    )

    chart_df = df.copy()

    # Normalize possible MultiIndex columns
    if isinstance(chart_df.columns, pd.MultiIndex):
        flattened = []

        for col in chart_df.columns:
            parts = [
                str(x).strip()
                for x in col
                if str(x).strip().lower() != "nan"
            ]

            flattened.append("_".join(parts))

        chart_df.columns = flattened

    # Normalize column names
    rename_map = {}

    for col in chart_df.columns:
        name = str(col).strip().lower()

        if name in {"open", "open_price"}:
            rename_map[col] = "open"

        elif name in {"high", "high_price"}:
            rename_map[col] = "high"

        elif name in {"low", "low_price"}:
            rename_map[col] = "low"

        elif name in {"close", "close_price"}:
            rename_map[col] = "close"

        elif name in {"volume", "vol"}:
            rename_map[col] = "volume"

    chart_df = chart_df.rename(columns=rename_map)

    required = {"open", "high", "low", "close"}

    if required.issubset(chart_df.columns):

        chart_df = chart_df.copy()

        for col in ["open", "high", "low", "close"]:
            chart_df[col] = pd.to_numeric(
                chart_df[col],
                errors="coerce",
            )

        chart_df = chart_df.dropna(
            subset=["open", "high", "low", "close"]
        )

        # Last 6 months for cleaner visualization
        chart_view = chart_df.tail(130)

        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=chart_view.index,
                open=chart_view["open"],
                high=chart_view["high"],
                low=chart_view["low"],
                close=chart_view["close"],
                name="Price",
            )
        )

        # Add EMA lines if available from feature data
        try:
            features = getattr(prediction, "features", None)

            if isinstance(features, pd.DataFrame):

                feature_view = features.tail(len(chart_view))

                for ema_name, label in [
                    ("ema20", "EMA 20"),
                    ("ema50", "EMA 50"),
                    ("ema200", "EMA 200"),
                ]:

                    if ema_name in feature_view.columns:

                        ema = pd.to_numeric(
                            feature_view[ema_name],
                            errors="coerce",
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=feature_view.index,
                                y=ema,
                                mode="lines",
                                name=label,
                            )
                        )

        except Exception:
            pass

        fig.update_layout(
            height=600,
            xaxis_rangeslider_visible=False,
            margin=dict(
                l=10,
                r=10,
                t=30,
                b=10,
            ),
            hovermode="x unified",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    else:
        st.info(
            "OHLC data is not available in the expected format, "
            "so the price chart cannot be displayed."
        )


    # -----------------------------------------------------------------
    # Trade Plan
    # -----------------------------------------------------------------

    st.markdown(
        '<div class="section-title">🎯 Trade Plan</div>',
        unsafe_allow_html=True,
    )

    trade_plan = getattr(
        prediction,
        "trade_plan",
        {},
    )

    entry = getattr(
        prediction,
        "entry",
        None,
    )

    stop_loss = getattr(
        prediction,
        "stop_loss",
        None,
    )

    target = getattr(
        prediction,
        "target",
        None,
    )

    if isinstance(trade_plan, dict):

        entry = trade_plan.get(
            "entry",
            entry,
        )

        stop_loss = trade_plan.get(
            "stop_loss",
            stop_loss,
        )

        target = trade_plan.get(
            "target",
            target,
        )

    t1, t2, t3 = st.columns(3)

    with t1:
        st.metric(
            "Entry",
            f"₹{float(entry):,.2f}"
            if entry is not None
            else "—",
        )

    with t2:
        st.metric(
            "Stop Loss",
            f"₹{float(stop_loss):,.2f}"
            if stop_loss is not None
            else "—",
        )

    with t3:
        st.metric(
            "Target",
            f"₹{float(target):,.2f}"
            if target is not None
            else "—",
        )


    # -----------------------------------------------------------------
    # Technical Indicators
    # -----------------------------------------------------------------

    st.markdown(
        '<div class="section-title">📊 Technical Indicators</div>',
        unsafe_allow_html=True,
    )

    indicators = getattr(
        prediction,
        "indicators",
        {},
    )

    if isinstance(indicators, dict) and indicators:

        indicator_df = pd.DataFrame(
            {
                "Indicator": list(indicators.keys()),
                "Value": [
                    round(float(value), 4)
                    for value in indicators.values()
                    if value is not None
                ],
            }
        )

        st.dataframe(
            indicator_df,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info("No indicator data available.")


    # -----------------------------------------------------------------
    # Model Information
    # -----------------------------------------------------------------

    with st.expander("🤖 Model Information"):

        model_col1, model_col2, model_col3 = st.columns(3)

        with model_col1:
            st.metric(
                "Prediction Horizon",
                f"{horizon} days",
            )

        with model_col2:
            accuracy = float(
                getattr(
                    prediction,
                    "training_accuracy",
                    0.0,
                )
                or 0.0
            )

            st.metric(
                "Training Accuracy",
                f"{accuracy:.1%}",
            )

        with model_col3:
            n_rows = int(
                getattr(
                    prediction,
                    "n_rows",
                    len(df),
                )
                or len(df)
            )

            st.metric(
                "Training Rows",
                f"{n_rows:,}",
            )


    # -----------------------------------------------------------------
    # Top Features
    # -----------------------------------------------------------------

    top_features = getattr(
        prediction,
        "top_features",
        None,
    )

    if top_features is not None:

        with st.expander("🔎 Top Model Features"):

            if isinstance(top_features, pd.DataFrame):

                st.dataframe(
                    top_features,
                    use_container_width=True,
                    hide_index=True,
                )

            elif isinstance(top_features, dict):

                feature_df = pd.DataFrame(
                    {
                        "Feature": list(top_features.keys()),
                        "Importance": list(top_features.values()),
                    }
                )

                st.dataframe(
                    feature_df,
                    use_container_width=True,
                    hide_index=True,
                )

            else:
                st.write(top_features)


    # -----------------------------------------------------------------
    # Latest OHLCV
    # -----------------------------------------------------------------

    with st.expander("📋 Latest Market Data"):

        display_df = df.tail(10).copy()

        st.dataframe(
            display_df,
            use_container_width=True,
        )


    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    with st.expander("🛠 Diagnostics"):

        st.write(
            {
                "Symbol": symbol,
                "Rows downloaded": len(df),
                "Prediction horizon": horizon,
                "Signal": signal,
                "Market regime": regime,
                "Trend": trend,
                "Momentum": momentum,
                "Volume": volume,
                "Weekly trend": weekly_trend,
                "Monthly trend": monthly_trend,
            }
        )
