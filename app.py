import streamlit as st

st.set_page_config(page_title="AI Swing Analyzer", page_icon="📈", layout="wide")

st.title("📈 AI Swing Stock Analyzer")
st.markdown("Simple stock analysis and backtesting — no research workflow required.")

st.info("Use **Stock Analyzer** for a current AI swing setup and **Backtesting** to test the same strategy on historical data.")

c1, c2 = st.columns(2)
with c1:
    st.subheader("📊 Stock Analyzer")
    st.write("Technical indicators + Gradient Boosting probability + entry, stop and target.")
    if st.button("Open Stock Analyzer", use_container_width=True):
        st.switch_page("pages/1_Stock_Analyzer.py")
with c2:
    st.subheader("📈 Backtesting")
    st.write("Test the strategy using Yahoo Finance or your own long-history CSV.")
    if st.button("Open Backtesting", use_container_width=True):
        st.switch_page("pages/2_Backtesting.py")

st.divider()
st.caption("Data source: Yahoo Finance via yfinance. For long historical backtests, upload OHLCV CSV data.")
