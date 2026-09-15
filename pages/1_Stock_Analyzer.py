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
st.caption("Gradient Boosting prediction + technical market analysis")


# -------------------------------------------------------------------
# User inputs — intentionally kept minimal
# -------------------------------------------------------------------
with st.sidebar:
    st.header("🔎 Stock Analysis")

    symbol = st.text_input(
        "Stock",
        "RELIANCE.NS",
        help="Enter an NSE Yahoo Finance symbol, e.g. RELIANCE.NS, TCS.NS or INFY.NS.",
    ).strip().upper()

    horizon = st.selectbox(
        "Prediction Horizon",
        [1, 3, 5, 10, 20],
        index=2,
        format_func=lambda x: f"{x} Trading Day{'s' if x != 1 else ''}",
        help="How far ahead the AI should predict the direction.",
    )

    st.divider()
    st.caption("Yahoo Finance data • Daily timeframe")


# -------------------------------------------------------------------
# Analyze
# -------------------------------------------------------------------
if st.button(
    "🚀 Analyze Stock",
    type="primary",
    use_container_width=True,
):
    if not symbol:
        st.error("Please enter a stock symbol.")
    else:
        try:
            with st.spinner("Downloading data and running AI analysis..."):
                # All model/risk parameters are internal.
                # The user only selects stock and prediction horizon.
                df = download_data(
                    symbol,
                    period="1y",
                    interval="1d",
                )

                pred = analyze_stock(
                    df,
                    horizon=horizon,
                    probability_threshold=0.60,
                    stop_loss_pct=0.03,
                    target_pct=0.06,
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
    active_symbol = st.session_state.get(
        "analysis_symbol",
        symbol,
    )

    last_close = float(df["close"].iloc[-1])

    if len(df) > 1:
        previous_close = float(df["close"].iloc[-2])
        daily_change_pct = (
            (last_close - previous_close)
            / previous_close
            if previous_close
            else 0
        )
    else:
        daily_change_pct = 0

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
        "UP Probability",
        f"{pred.probability_up:.1%}",
    )

    c3.metric(
        "DOWN Probability",
        f"{pred.probability_down:.1%}",
    )

    c4.metric(
        "Market Regime",
        pred.regime,
    )

    st.divider()

    # ---------------------------------------------------------------
    # Main signal
    # ---------------------------------------------------------------
    signal_col, confidence_col = st.columns([2, 1])

    with signal_col:
        if pred.signal == "BUY":
            st.success("🟢 BUY — Bullish setup detected")
        elif pred.signal == "SELL":
            st.error("🔴 SELL — Bearish setup detected")
        else:
            st.warning("🟡 WAIT — No sufficiently strong setup")

        if getattr(pred, "signal_reason", ""):
            st.caption(pred.signal_reason)

    with confidence_col:
        st.metric(
            "AI Confidence",
            pred.confidence,
        )

    # ---------------------------------------------------------------
    # Automatically calculated trade plan
    # ---------------------------------------------------------------
    if pred.signal != "WAIT":
        st.subheader("🎯 Suggested Trade Levels")

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

        st.caption(
            "Entry, stop loss and target are calculated automatically by the model."
        )

    # ---------------------------------------------------------------
    # AI probability
    # ---------------------------------------------------------------
    st.subheader("🤖 AI Prediction")

    probability_df = pd.DataFrame(
        {
            "Direction": ["UP", "DOWN"],
            "Probability": [
                pred.probability_up,
                pred.probability_down,
            ],
        }
    )

    probability_fig = px.bar(
        probability_df,
        x="Direction",
        y="Probability",
        text="Probability",
    )

    probability_fig.update_traces(
        texttemplate="%{text:.1%}",
        textposition="outside",
    )

    probability_fig.update_yaxes(
        range=[0, 1],
        tickformat=".0%",
    )

    probability_fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
    )

    st.plotly_chart(
        probability_fig,
        use_container_width=True,
    )

    # ---------------------------------------------------------------
    # Market structure
    # ---------------------------------------------------------------
    st.subheader("📐 Market Structure")

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Trend",
        getattr(pred, "trend", "UNKNOWN"),
    )

    m2.metric(
        "Momentum",
        getattr(pred, "momentum", "UNKNOWN"),
    )

    m3.metric(
        "Volume",
        getattr(pred, "volume_status", "UNKNOWN"),
    )

    # ---------------------------------------------------------------
    # Price chart
    # ---------------------------------------------------------------
    st.subheader("📊 Price Chart")

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
        )
    )

    # The feature columns are generated inside analyze_stock, so
    # calculate them here only if available in the downloaded data.
    # This keeps the chart compatible with the existing data layer.
    try:
        from src.features.engine import build_features

        chart_df = build_features(df)

        for column, label in [
            ("ema20", "EMA 20"),
            ("ema50", "EMA 50"),
            ("ema200", "EMA 200"),
        ]:
            if column in chart_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=chart_df.index,
                        y=chart_df[column],
                        name=label,
                        mode="lines",
                    )
                )
    except Exception:
        pass

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
    # Indicators
    # ---------------------------------------------------------------
    st.subheader("📋 Technical Indicators")

    indicator_names = {
        "rsi14": "RSI 14",
        "adx14": "ADX 14",
        "atr14": "ATR 14",
        "macd": "MACD",
        "macd_signal": "MACD Signal",
        "macd_hist": "MACD Histogram",
        "relative_volume": "Relative Volume",
        "ema20": "EMA 20",
        "ema50": "EMA 50",
        "ema200": "EMA 200",
        "bb_upper": "Bollinger Upper",
        "bb_lower": "Bollinger Lower",
        "roc10": "ROC 10",
        "volatility20": "20D Volatility",
    }

    indicator_rows = []

    for key, label in indicator_names.items():
        value = pred.indicators.get(key)

        if value is not None:
            indicator_rows.append(
                {
                    "Indicator": label,
                    "Value": round(float(value), 4),
                }
            )

    if indicator_rows:
        st.dataframe(
            pd.DataFrame(indicator_rows),
            use_container_width=True,
            hide_index=True,
        )

    # ---------------------------------------------------------------
    # Data information
    # ---------------------------------------------------------------
    with st.expander("📊 Data Information"):
        d1, d2, d3, d4 = st.columns(4)

        d1.metric("Rows", f"{len(df):,}")
        d2.metric("Start", str(df.index.min().date()))
        d3.metric("End", str(df.index.max().date()))
        d4.metric(
            "Prediction Horizon",
            f"{horizon} day{'s' if horizon != 1 else ''}",
        )

    with st.expander("🤖 Model Diagnostic"):
        accuracy = getattr(
            pred,
            "model_training_accuracy",
            None,
        )

        if accuracy is not None:
            st.write(
                f"Training accuracy: **{accuracy:.2%}**"
            )

        st.caption(
            "Training accuracy is a diagnostic only and does not represent "
            "expected live trading performance."
        )

        st.caption(
            "AI probabilities are model estimates, not guarantees of future "
            "price movement."
        )

    with st.expander("📄 Latest OHLCV Data"):
        st.dataframe(
            df.tail(20),
            use_container_width=True,
        )
