import streamlit as st
import plotly.graph_objects as go

from src.data.downloader import download_data
from src.models.predictor import analyze_stock

st.set_page_config(page_title="Stock Analyzer", page_icon="📊", layout="wide")
st.title("📊 Stock Analyzer")

with st.sidebar:
    st.header("Analysis")
    symbol = st.text_input("NSE Symbol", "RELIANCE.NS").strip().upper()
    period = st.selectbox("Training data", ["1y", "2y", "5y", "10y", "max"], index=1)
    horizon = st.selectbox("Prediction horizon", [1, 3, 5, 10, 20], index=2)
    threshold = st.slider("AI probability threshold", 0.50, 0.80, 0.60, 0.01)
    stop_pct = st.slider("Stop loss %", 1.0, 10.0, 3.0, 0.5) / 100
    target_pct = st.slider("Target %", 2.0, 20.0, 6.0, 0.5) / 100

if st.button("Analyze", type="primary", use_container_width=True):
    try:
        with st.spinner("Downloading data and training AI model..."):
            df = download_data(symbol, period=period, interval="1d")
            pred = analyze_stock(df, horizon=horizon, probability_threshold=threshold, stop_loss_pct=stop_pct, target_pct=target_pct)
        st.session_state["analysis_df"] = df
        st.session_state["prediction"] = pred
    except Exception as exc:
        st.error(str(exc))

if "prediction" in st.session_state:
    pred = st.session_state["prediction"]
    df = st.session_state["analysis_df"]
    cols = st.columns(5)
    cols[0].metric("Signal", pred.signal)
    cols[1].metric("UP probability", f"{pred.probability_up:.1%}")
    cols[2].metric("DOWN probability", f"{pred.probability_down:.1%}")
    cols[3].metric("Confidence", pred.confidence)
    cols[4].metric("Regime", pred.regime)

    if pred.signal != "WAIT":
        a, b, c, d = st.columns(4)
        a.metric("Entry", f"₹{pred.entry:,.2f}")
        b.metric("Stop Loss", f"₹{pred.stop_loss:,.2f}")
        c.metric("Target", f"₹{pred.target:,.2f}")
        d.metric("Risk / Reward", f"1 : {pred.risk_reward:.2f}")
    else:
        st.warning("No high-confidence setup. Wait for stronger evidence.")

    st.subheader("Price")
    fig = go.Figure(go.Candlestick(x=df.index, open=df.open, high=df.high, low=df.low, close=df.close, name="Price"))
    fig.update_layout(height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Indicators")
    st.dataframe(pred.indicators, use_container_width=True)
    st.caption(f"Model training accuracy is shown only as a diagnostic; it is not a measure of live profitability. Training rows: {getattr(pred, 'model_training_accuracy', None)}")
