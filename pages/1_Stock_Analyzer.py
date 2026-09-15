import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from src.data.downloader import download_data
from src.models.predictor import analyze_stock


st.set_page_config(
    page_title="Stock Analyzer",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AI Swing Stock Analyzer")
st.caption("Technical analysis + Gradient Boosting probability model")


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Analysis Settings")

    symbol = st.text_input(
        "NSE Symbol",
        "RELIANCE.NS",
        help="Examples: RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS",
    ).strip().upper()

    period = st.selectbox(
        "Yahoo Finance data",
        ["1y", "2y", "5y", "10y", "max"],
        index=0,
        help="Use CSV backtesting for very long/custom historical datasets.",
    )

    horizon = st.selectbox(
        "Prediction horizon",
        [1, 3, 5, 10, 20],
        index=2,
        help="Number of trading days used to define the future return target.",
    )

    threshold = st.slider(
        "AI probability threshold",
        0.50,
        0.80,
        0.60,
        0.01,
    )

    stop_pct = st.slider(
        "Stop loss %",
        1.0,
        10.0,
        3.0,
        0.5,
    ) / 100

    target_pct = st.slider(
        "Target %",
        2.0,
        20.0,
        6.0,
        0.5,
    ) / 100

    st.divider()
    st.caption("Yahoo Finance • Daily candles • Gradient Boosting")


# -------------------------------------------------------------------
# Analyze
# -------------------------------------------------------------------
if st.button("🚀 Analyze Stock", type="primary", use_container_width=True):
    if not symbol:
        st.error("Please enter an NSE symbol.")
    else:
        try:
            with st.spinner("Downloading data and training AI model..."):
                df = download_data(
                    symbol,
                    period=period,
                    interval="1d",
                )

                pred = analyze_stock(
                    df,
                    horizon=horizon,
                    probability_threshold=threshold,
                    stop_loss_pct=stop_pct,
                    target_pct=target_pct,
                )

            st.session_state["analysis_df"] = df
            st.session_state["prediction"] = pred
            st.session_state["analysis_symbol"] = symbol

        except Exception as exc:
            st.error(str(exc))


# -------------------------------------------------------------------
# Results
# -------------------------------------------------------------------
if "prediction" in st.session_state:
    pred = st.session_state["prediction"]
    df = st.session_state["analysis_df"]
    active_symbol = st.session_state.get("analysis_symbol", symbol)

    last_close = float(df["close"].iloc[-1])
    previous_close = (
        float(df["close"].iloc[-2])
        if len(df) > 1
        else last_close
    )
    daily_change = last_close - previous_close
    daily_change_pct = (
        daily_change / previous_close
        if previous_close
        else 0
    )

    st.subheader(f"📈 {active_symbol}")

    # ---------------------------------------------------------------
    # Market snapshot
    # ---------------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Last Price",
        f"₹{last_close:,.2f}",
        f"{daily_change_pct:+.2%}",
    )

    c2.metric(
        "AI UP Probability",
        f"{pred.probability_up:.1%}",
    )

    c3.metric(
        "AI DOWN Probability",
        f"{pred.probability_down:.1%}",
    )

    c4.metric(
        "Market Regime",
        pred.regime,
    )

    st.divider()

    # ---------------------------------------------------------------
    # Signal
    # ---------------------------------------------------------------
    signal_col, confidence_col = st.columns([2, 1])

    with signal_col:
        if pred.signal == "BUY":
            st.success("🟢 BUY — Bullish setup detected")
        elif pred.signal == "SELL":
            st.error("🔴 SELL — Bearish setup detected")
        else:
            st.warning("🟡 WAIT — No sufficiently strong setup")

    with confidence_col:
        st.metric("Confidence", pred.confidence)

    # ---------------------------------------------------------------
    # Trade plan
    # ---------------------------------------------------------------
    if pred.signal != "WAIT":
        st.subheader("🎯 Trade Plan")

        a, b, c, d = st.columns(4)

        a.metric(
            "Entry",
            f"₹{pred.entry:,.2f}",
        )

        b.metric(
            "Stop Loss",
            f"₹{pred.stop_loss:,.2f}",
        )

        c.metric(
            "Target",
            f"₹{pred.target:,.2f}",
        )

        d.metric(
            "Risk / Reward",
            f"1 : {pred.risk_reward:.2f}",
        )

        if pred.signal == "BUY":
            st.info(
                f"Potential upside: "
                f"{target_pct:.1%} | "
                f"Risk: {stop_pct:.1%}"
            )
        else:
            st.info(
                f"Potential downside: "
                f"{target_pct:.1%} | "
                f"Risk: {stop_pct:.1%}"
            )

    # ---------------------------------------------------------------
    # Probability visualization
    # ---------------------------------------------------------------
    st.subheader("🤖 AI Probability")

    probability_df = pd.DataFrame(
        {
            "Direction": ["UP", "DOWN"],
            "Probability": [
                pred.probability_up,
                pred.probability_down,
            ],
        }
    )

    fig_probability = px.bar(
        probability_df,
        x="Direction",
        y="Probability",
        text="Probability",
    )

    fig_probability.update_traces(
        texttemplate="%{text:.1%}",
        textposition="outside",
    )

    fig_probability.update_yaxes(
        range=[0, 1],
        tickformat=".0%",
    )

    fig_probability.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
    )

    st.plotly_chart(
        fig_probability,
        use_container_width=True,
    )

    # ---------------------------------------------------------------
    # Price chart
    # ---------------------------------------------------------------
    st.subheader("📊 Price & Moving Averages")

    chart_df = df.copy()

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=chart_df.index,
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            name="Price",
        )
    )

    if "ema20" in chart_df.columns:
        fig.add_trace(
            go.Scatter(
                x=chart_df.index,
                y=chart_df["ema20"],
                name="EMA 20",
                mode="lines",
            )
        )

    if "ema50" in chart_df.columns:
        fig.add_trace(
            go.Scatter(
                x=chart_df.index,
                y=chart_df["ema50"],
                name="EMA 50",
                mode="lines",
            )
        )

    if "ema200" in chart_df.columns:
        fig.add_trace(
            go.Scatter(
                x=chart_df.index,
                y=chart_df["ema200"],
                name="EMA 200",
                mode="lines",
            )
        )

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # ---------------------------------------------------------------
    # Technical indicators
    # ---------------------------------------------------------------
    st.subheader("📐 Technical Indicators")

    indicator_names = {
        "rsi14": "RSI 14",
        "adx14": "ADX 14",
        "atr14": "ATR 14",
        "macd": "MACD",
        "macd_signal": "MACD Signal",
        "relative_volume": "Relative Volume",
        "ema20": "EMA 20",
        "ema50": "EMA 50",
        "ema200": "EMA 200",
        "bb_upper": "Bollinger Upper",
        "bb_lower": "Bollinger Lower",
    }

    indicator_items = []

    for key, label in indicator_names.items():
        value = pred.indicators.get(key)

        if value is not None:
            indicator_items.append(
                {
                    "Indicator": label,
                    "Value": round(float(value), 4),
                }
            )

    if indicator_items:
        indicator_df = pd.DataFrame(indicator_items)

        left, right = st.columns(2)

        with left:
            st.dataframe(
                indicator_df,
                use_container_width=True,
                hide_index=True,
            )

        with right:
            st.write("**Quick interpretation**")

            rsi = pred.indicators.get("rsi14")
            adx = pred.indicators.get("adx14")
            rel_volume = pred.indicators.get("relative_volume")

            if rsi is not None:
                if rsi >= 70:
                    st.write("• RSI: Overbought zone")
                elif rsi <= 30:
                    st.write("• RSI: Oversold zone")
                else:
                    st.write("• RSI: Neutral zone")

            if adx is not None:
                if adx >= 25:
                    st.write("• ADX: Trend is relatively strong")
                else:
                    st.write("• ADX: Trend is relatively weak")

            if rel_volume is not None:
                if rel_volume >= 1.5:
                    st.write("• Volume: Strong relative volume")
                elif rel_volume < 0.75:
                    st.write("• Volume: Below average")
                else:
                    st.write("• Volume: Normal")

    # ---------------------------------------------------------------
    # Data information
    # ---------------------------------------------------------------
    st.subheader("📋 Data Information")

    d1, d2, d3, d4 = st.columns(4)

    d1.metric("Rows", f"{len(df):,}")
    d2.metric("Start", str(df.index.min().date()))
    d3.metric("End", str(df.index.max().date()))
    d4.metric("Prediction Horizon", f"{horizon} day(s)")

    with st.expander("Model diagnostic"):
        accuracy = getattr(pred, "model_training_accuracy", None)

        if accuracy is not None:
            st.write(
                f"Training accuracy: **{accuracy:.2%}**"
            )

        st.caption(
            "Training accuracy is only a diagnostic. "
            "It does not represent expected live trading performance."
        )

        st.caption(
            "The AI probability is a model estimate, not a guarantee "
            "of future price movement."
        )

    with st.expander("View latest OHLCV data"):
        st.dataframe(
            df.tail(20),
            use_container_width=True,
        )
