import streamlit as st
import plotly.graph_objects as go

from src.data.downloader import download_data, load_csv_data
from src.models.backtest import BacktestConfig, run_backtest

st.set_page_config(page_title="Backtesting", page_icon="📈", layout="wide")
st.title("📈 Backtesting")
st.caption("Use Yahoo Finance for convenience or upload a long-history OHLCV CSV for multi-year testing.")

with st.sidebar:
    st.header("Data")
    source = st.radio("Data source", ["Yahoo Finance", "CSV Upload"])
    symbol = st.text_input("NSE Symbol", "RELIANCE.NS").strip().upper()
    if source == "Yahoo Finance":
        period = st.selectbox("Period", ["1y", "2y", "5y", "10y", "max"], index=2)
    else:
        uploaded = st.file_uploader("OHLCV CSV", type=["csv"])

    st.header("Strategy")
    horizon = st.selectbox("Holding horizon", [1, 3, 5, 10, 20], index=2)
    threshold = st.slider("Probability threshold", 0.50, 0.80, 0.60, 0.01)
    stop_pct = st.slider("Stop loss %", 1.0, 10.0, 3.0, 0.5) / 100
    target_pct = st.slider("Target %", 2.0, 20.0, 6.0, 0.5) / 100
    capital = st.number_input("Initial capital", min_value=10000.0, value=100000.0, step=10000.0)
    risk = st.slider("Risk per trade %", 0.25, 5.0, 1.0, 0.25) / 100

if st.button("Run Backtest", type="primary", use_container_width=True):
    try:
        with st.spinner("Running walk-forward backtest..."):
            if source == "Yahoo Finance":
                df = download_data(symbol, period=period, interval="1d")
            else:
                if uploaded is None:
                    st.error("Upload a CSV file first.")
                    st.stop()
                df = load_csv_data(uploaded)
            cfg = BacktestConfig(initial_capital=capital, horizon=horizon, probability_threshold=threshold, stop_loss_pct=stop_pct, target_pct=target_pct, risk_per_trade=risk)
            result = run_backtest(df, cfg)
        st.session_state["backtest_result"] = result
        st.session_state["backtest_df"] = df
    except Exception as exc:
        st.error(str(exc))

if "backtest_result" in st.session_state:
    result = st.session_state["backtest_result"]
    curve = result["equity_curve"]
    trades = result["trades_df"]
    c = st.columns(6)
    c[0].metric("Final capital", f"₹{result['final_capital']:,.0f}")
    c[1].metric("Total return", f"{result['total_return']:.1%}")
    c[2].metric("Win rate", f"{result['win_rate']:.1%}")
    c[3].metric("Profit factor", f"{result['profit_factor']:.2f}" if result['profit_factor'] != float('inf') else "∞")
    c[4].metric("Max drawdown", f"{result['max_drawdown']:.1%}")
    c[5].metric("Trades", str(result['trades']))

    if not curve.empty:
        fig = go.Figure(go.Scatter(x=curve.index, y=curve.equity, mode="lines", name="Equity"))
        fig.update_layout(title="Equity Curve", height=450, yaxis_title="Capital")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Trades")
    st.dataframe(trades, use_container_width=True)
