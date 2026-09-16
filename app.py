from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="AI Swing Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------

st.markdown(
    """
    <style>

    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,0.18);
    }

    /* Header */
    .app-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .app-subtitle {
        font-size: 1rem;
        opacity: 0.70;
        margin-bottom: 1.5rem;
    }

    /* Cards */
    .home-card {
        padding: 1.3rem;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.20);
        background: rgba(128,128,128,0.05);
        min-height: 150px;
    }

    .home-card h3 {
        margin-top: 0;
        margin-bottom: 0.5rem;
    }

    .home-card p {
        opacity: 0.75;
        margin-bottom: 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        opacity: 0.55;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(128,128,128,0.15);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

with st.sidebar:

    st.markdown("## 📈 AI Swing Analyzer")

    st.caption(
        "AI-assisted technical and swing-trading analysis"
    )

    st.divider()

    st.markdown("### Navigation")

    st.page_link(
        "pages/1_Stock_Analyzer.py",
        label="📊 Stock Analyzer",
        icon="📊",
    )

    st.page_link(
        "pages/2_Backtesting.py",
        label="🧪 Backtesting",
        icon="🧪",
    )

    st.divider()

    st.markdown("### Analysis Engine")

    st.caption("Technical Indicators")
    st.caption("Price Action")
    st.caption("Volume Analysis")
    st.caption("Market Regime")
    st.caption("Multi-Timeframe Analysis")
    st.caption("Gradient Boosting Model")

    st.divider()

    st.caption("v1.0")


# ---------------------------------------------------------------------
# Home page
# ---------------------------------------------------------------------

st.markdown(
    '<div class="app-title">📈 AI Swing Stock Analyzer</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-subtitle">'
    "A systematic framework for analyzing Indian equities using "
    "technical indicators, price action, volume, market regime, "
    "multi-timeframe confirmation and machine learning."
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Quick navigation
# ---------------------------------------------------------------------

st.subheader("Start Analysis")

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        """
        <div class="home-card">

        ### 📊 Stock Analyzer

        Analyze an individual stock using the AI swing-analysis
        pipeline.

        <br>

        <b>Includes:</b>

        • AI signal  
        • Probability  
        • Market structure  
        • Momentum  
        • Volume  
        • Multi-timeframe trend  
        • Trade plan  
        • Technical indicators  

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    st.page_link(
        "pages/1_Stock_Analyzer.py",
        label="Open Stock Analyzer →",
        icon="📊",
    )


with col2:

    st.markdown(
        """
        <div class="home-card">

        ### 🧪 Backtesting

        Evaluate the swing strategy against historical market data
        using a walk-forward testing approach.

        <br>

        <b>Includes:</b>

        • Historical simulation  
        • Model retraining  
        • Entry / exit logic  
        • Stop loss  
        • Target  
        • Equity curve  
        • Performance metrics  

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    st.page_link(
        "pages/2_Backtesting.py",
        label="Open Backtesting →",
        icon="🧪",
    )


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

st.divider()

st.subheader("Analysis Pipeline")

pipeline = [
    ("01", "Market Data", "Yahoo Finance / OHLCV"),
    ("02", "Feature Engineering", "Technical + Price Action + Volume"),
    ("03", "Market Context", "Regime + Multi-Timeframe"),
    ("04", "Machine Learning", "Gradient Boosting"),
    ("05", "Prediction", "Probability + Signal"),
    ("06", "Trade Plan", "Entry + Stop Loss + Target"),
]

cols = st.columns(3)

for index, (number, title, description) in enumerate(pipeline):

    with cols[index % 3]:

        st.markdown(
            f"""
            <div class="home-card">

            <div style="font-size:0.8rem;opacity:0.55;">
            {number}
            </div>

            <h3>{title}</h3>

            <p>{description}</p>

            </div>
            """,
            unsafe_allow_html=True,
        )

    if index % 3 == 2:
        st.write("")


# ---------------------------------------------------------------------
# Important note
# ---------------------------------------------------------------------

st.divider()

st.info(
    "⚠️ The analyzer is a decision-support tool, not financial advice. "
    "Model probabilities and technical signals can be wrong, especially "
    "during unusual market conditions."
)


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------

st.markdown(
    """
    <div class="footer">
        AI Swing Stock Analyzer · Technical Analysis · Machine Learning
    </div>
    """,
    unsafe_allow_html=True,
)
