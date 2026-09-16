"""
AI Swing Stock Analyzer
Streamlit UI

Displays:
- Price action
- Technical indicators
- ML ensemble
- Model agreement
- News sentiment
- Market regime
- Multi-timeframe context
- Risk / trade plan
- Validation diagnostics
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.downloader import download_data
from src.models.predictor import analyze_stock


# =====================================================================
# PAGE
# =====================================================================

st.set_page_config(
    page_title="AI Swing Stock Analyzer",
    page_icon="📈",
    layout="wide",
)


# =====================================================================
# HEADER
# =====================================================================

st.title("📈 AI Swing Stock Analyzer")

st.caption(
    "Ensemble ML + Technical Analysis + Market Regime "
    "+ Multi-Timeframe + News Sentiment"
)


# =====================================================================
# SIDEBAR
# =====================================================================

st.sidebar.header("Analysis Settings")

ticker = st.sidebar.text_input(
    "Stock",
    value="RELIANCE.NS",
).strip().upper()

horizon = st.sidebar.selectbox(
    "Prediction Horizon",
    options=[1, 3, 5, 10, 20],
    index=2,
)

st.sidebar.divider()

st.sidebar.caption(
    "Model configuration"
)

probability_threshold = 0.60
stop_loss_pct = 0.03
target_pct = 0.06

st.sidebar.info(
    f"""
**ML threshold:** {probability_threshold:.0%}

**Default stop:** {stop_loss_pct:.0%}

**Default target:** {target_pct:.0%}
"""
)


# =====================================================================
# LOAD DATA
# =====================================================================

if not ticker:

    st.warning(
        "Please enter a stock ticker."
    )

    st.stop()


try:

    with st.spinner(
        f"Downloading data for {ticker}..."
    ):

        df = download_data(
            ticker
        )

except Exception as exc:

    st.error(
        f"Unable to download market data: {exc}"
    )

    st.stop()


if df is None or df.empty:

    st.error(
        "No market data was returned."
    )

    st.stop()


# =====================================================================
# NORMALIZE DATA
# =====================================================================

data = df.copy()

if isinstance(
    data.columns,
    pd.MultiIndex,
):

    data.columns = [
        "_".join(
            str(x)
            for x in col
            if str(x).lower() != "nan"
        ).strip("_")
        for col in data.columns
    ]

data.columns = [
    str(c).strip().lower()
    for c in data.columns
]


# =====================================================================
# ANALYSIS
# =====================================================================

try:

    with st.spinner(
        "Running ensemble analysis..."
    ):

        result = analyze_stock(
            data,
            horizon=horizon,
            probability_threshold=(
                probability_threshold
            ),
            stop_loss_pct=stop_loss_pct,
            target_pct=target_pct,
            ticker=ticker,
        )

except Exception as exc:

    st.error(
        f"Analysis failed: {exc}"
    )

    st.exception(exc)

    st.stop()


# =====================================================================
# BASIC VALUES
# =====================================================================

current_price = float(
    result.get(
        "current_price",
        0.0,
    )
)

signal = str(
    result.get(
        "signal",
        "WAIT",
    )
)

base_signal = str(
    result.get(
        "base_signal",
        "WAIT",
    )
)

probability_up = float(
    result.get(
        "probability_up",
        0.5,
    )
)

probability_down = float(
    result.get(
        "probability_down",
        0.5,
    )
)

ml_probability = float(
    result.get(
        "prediction_probability_ml",
        probability_up,
    )
)

adjusted_probability = float(
    result.get(
        "prediction_probability",
        probability_up,
    )
)

confidence = float(
    result.get(
        "confidence",
        0.0,
    )
)

uncertainty = float(
    result.get(
        "uncertainty",
        1.0,
    )
)


# =====================================================================
# SIGNAL HEADER
# =====================================================================

st.subheader(
    f"{ticker} — AI Analysis"
)

col1, col2, col3, col4 = st.columns(
    4
)

with col1:

    st.metric(
        "Current Price",
        f"₹{current_price:,.2f}",
    )

with col2:

    st.metric(
        "Final Signal",
        signal,
    )

with col3:

    st.metric(
        "ML Probability",
        f"{ml_probability:.1%}",
    )

with col4:

    st.metric(
        "Confidence",
        f"{confidence:.1%}",
    )


# =====================================================================
# SIGNAL EXPLANATION
# =====================================================================

if signal == "BUY":

    st.success(
        f"AI Signal: **BUY** | "
        f"Adjusted probability: "
        f"**{adjusted_probability:.1%}**"
    )

elif signal == "SELL":

    st.error(
        f"AI Signal: **SELL** | "
        f"Adjusted probability: "
        f"**{adjusted_probability:.1%}**"
    )

else:

    st.warning(
        f"AI Signal: **WAIT** | "
        f"Adjusted probability: "
        f"**{adjusted_probability:.1%}**"
    )


if base_signal != signal:

    st.caption(
        f"Base ensemble signal: **{base_signal}** → "
        f"final signal after evidence/risk filters: "
        f"**{signal}**"
    )


# =====================================================================
# PROBABILITY
# =====================================================================

st.subheader(
    "Probability & Uncertainty"
)

p1, p2, p3 = st.columns(
    3
)

with p1:

    st.metric(
        "Probability Up",
        f"{probability_up:.1%}",
    )

with p2:

    st.metric(
        "Probability Down",
        f"{probability_down:.1%}",
    )

with p3:

    st.metric(
        "Uncertainty",
        f"{uncertainty:.1%}",
    )


# =====================================================================
# SENTIMENT
# =====================================================================

sentiment = result.get(
    "sentiment",
    {},
)

sentiment_score = float(
    sentiment.get(
        "score",
        0.0,
    )
)

sentiment_label = str(
    sentiment.get(
        "label",
        "NO DATA",
    )
)

sentiment_articles = int(
    sentiment.get(
        "articles",
        0,
    )
)

sentiment_adjustment = float(
    sentiment.get(
        "adjustment",
        0.0,
    )
)

sentiment_weight = float(
    sentiment.get(
        "weight",
        0.0,
    )
)

st.subheader(
    "📰 News Sentiment"
)

s1, s2, s3, s4 = st.columns(
    4
)

with s1:

    st.metric(
        "Sentiment",
        sentiment_label,
    )

with s2:

    st.metric(
        "Sentiment Score",
        f"{sentiment_score:+.2f}",
    )

with s3:

    st.metric(
        "News Articles",
        str(sentiment_articles),
    )

with s4:

    st.metric(
        "ML Adjustment",
        f"{sentiment_adjustment:+.2%}",
    )


positive = int(
    sentiment.get(
        "positive",
        0,
    )
)

neutral = int(
    sentiment.get(
        "neutral",
        0,
    )
)

negative = int(
    sentiment.get(
        "negative",
        0,
    )
)

st.caption(
    f"Positive: {positive}  |  "
    f"Neutral: {neutral}  |  "
    f"Negative: {negative}  |  "
    f"Sentiment influence: {sentiment_weight:.1%}"
)


# =====================================================================
# MODEL AGREEMENT
# =====================================================================

st.subheader(
    "🤖 Ensemble Model Agreement"
)

agreement_data = result.get(
    "model_agreement",
    {},
)

agreement = float(
    agreement_data.get(
        "agreement",
        0.0,
    )
)

disagreement = float(
    agreement_data.get(
        "disagreement",
        1.0,
    )
)

bullish_models = int(
    agreement_data.get(
        "bullish_models",
        0,
    )
)

bearish_models = int(
    agreement_data.get(
        "bearish_models",
        0,
    )
)

total_models = int(
    agreement_data.get(
        "total_models",
        0,
    )
)

agreement_status = str(
    agreement_data.get(
        "status",
        "UNKNOWN",
    )
)

a1, a2, a3, a4 = st.columns(
    4
)

with a1:

    st.metric(
        "Agreement",
        f"{agreement:.0%}",
    )

with a2:

    st.metric(
        "Disagreement",
        f"{disagreement:.0%}",
    )

with a3:

    st.metric(
        "Bullish Models",
        f"{bullish_models}/{total_models}",
    )

with a4:

    st.metric(
        "Bearish Models",
        f"{bearish_models}/{total_models}",
    )


st.caption(
    f"Ensemble status: **{agreement_status}**"
)


# =====================================================================
# COMPONENT PROBABILITIES
# =====================================================================

component_probabilities = result.get(
    "component_probabilities",
    {},
)

if component_probabilities:

    with st.expander(
        "View Individual Model Probabilities"
    ):

        component_rows = []

        for name, probability in (
            component_probabilities.items()
        ):

            try:

                p = float(
                    probability
                )

                if np.isfinite(p):

                    component_rows.append(
                        {
                            "Model": name,
                            "Probability Up": (
                                f"{p:.1%}"
                            ),
                            "Direction": (
                                "Bullish"
                                if p >= 0.50
                                else "Bearish"
                            ),
                        }
                    )

            except Exception:
                continue

        if component_rows:

            st.dataframe(
                pd.DataFrame(
                    component_rows
                ),
                use_container_width=True,
                hide_index=True,
            )


# =====================================================================
# MARKET STRUCTURE
# =====================================================================

st.subheader(
    "Market Structure"
)

structure = result.get(
    "market_structure",
    {},
)

m1, m2, m3, m4 = st.columns(
    4
)

with m1:

    st.metric(
        "Trend",
        structure.get(
            "trend",
            "UNKNOWN",
        ),
    )

with m2:

    st.metric(
        "Momentum",
        structure.get(
            "momentum",
            "UNKNOWN",
        ),
    )

with m3:

    st.metric(
        "Volume",
        structure.get(
            "volume",
            "UNKNOWN",
        ),
    )

with m4:

    st.metric(
        "ADX",
        f"{float(structure.get('adx', 0.0)):.1f}",
    )


# =====================================================================
# REGIME / MTF
# =====================================================================

st.subheader(
    "Market Regime & Multi-Timeframe"
)

r1, r2, r3, r4 = st.columns(
    4
)

with r1:

    st.metric(
        "Market Regime",
        result.get(
            "regime",
            "UNKNOWN",
        ),
    )

with r2:

    st.metric(
        "Weekly Trend",
        result.get(
            "weekly_trend",
            "UNKNOWN",
        ),
    )

with r3:

    st.metric(
        "Monthly Trend",
        result.get(
            "monthly_trend",
            "UNKNOWN",
        ),
    )

with r4:

    st.metric(
        "3-Timeframe Alignment",
        result.get(
            "three_timeframe_alignment",
            "UNKNOWN",
        ),
    )


# =====================================================================
# PRICE CHART
# =====================================================================

st.subheader(
    "Price & Trend"
)

chart_data = data.copy()

required_columns = {
    "open",
    "high",
    "low",
    "close",
}

if required_columns.issubset(
    chart_data.columns
):

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=chart_data.index,
            open=chart_data["open"],
            high=chart_data["high"],
            low=chart_data["low"],
            close=chart_data["close"],
            name="Price",
        )
    )

    for column, name in [
        ("ema20", "EMA 20"),
        ("ema50", "EMA 50"),
        ("ema200", "EMA 200"),
    ]:

        if column in chart_data.columns:

            fig.add_trace(
                go.Scatter(
                    x=chart_data.index,
                    y=chart_data[column],
                    mode="lines",
                    name=name,
                )
            )

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        margin=dict(
            l=10,
            r=10,
            t=30,
            b=10,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# =====================================================================
# TRADE PLAN
# =====================================================================

st.subheader(
    "Risk / Trade Plan"
)

trade_plan = result.get(
    "trade_plan",
    {},
)

t1, t2, t3, t4 = st.columns(
    4
)

entry = trade_plan.get(
    "entry"
)

stop_loss = trade_plan.get(
    "stop_loss"
)

target = trade_plan.get(
    "target"
)

risk_reward = trade_plan.get(
    "risk_reward",
    0.0,
)

with t1:

    st.metric(
        "Entry",
        (
            f"₹{float(entry):,.2f}"
            if entry is not None
            else "—"
        ),
    )

with t2:

    st.metric(
        "Stop Loss",
        (
            f"₹{float(stop_loss):,.2f}"
            if stop_loss is not None
            else "—"
        ),
    )

with t3:

    st.metric(
        "Target",
        (
            f"₹{float(target):,.2f}"
            if target is not None
            else "—"
        ),
    )

with t4:

    st.metric(
        "Risk / Reward",
        f"{float(risk_reward):.2f}",
    )


# =====================================================================
# TECHNICAL INDICATORS
# =====================================================================

st.subheader(
    "Technical Indicators"
)

indicators = result.get(
    "indicators",
    {},
)

indicator_rows = []

for name, value in indicators.items():

    try:

        numeric_value = float(
            value
        )

        if np.isfinite(
            numeric_value
        ):

            indicator_rows.append(
                {
                    "Indicator": name.upper(),
                    "Value": round(
                        numeric_value,
                        4,
                    ),
                }
            )

    except Exception:
        continue


if indicator_rows:

    st.dataframe(
        pd.DataFrame(
            indicator_rows
        ),
        use_container_width=True,
        hide_index=True,
    )


# =====================================================================
# MODEL VALIDATION
# =====================================================================

st.subheader(
    "Model Validation Diagnostics"
)

v1, v2, v3, v4, v5 = st.columns(
    5
)

validation_accuracy = result.get(
    "validation_accuracy"
)

validation_precision = result.get(
    "validation_precision"
)

validation_recall = result.get(
    "validation_recall"
)

validation_f1 = result.get(
    "validation_f1"
)

validation_roc_auc = result.get(
    "validation_roc_auc"
)

validation_brier = result.get(
    "validation_brier"
)

with v1:

    st.metric(
        "Validation Accuracy",
        (
            f"{float(validation_accuracy):.1%}"
            if validation_accuracy is not None
            else "N/A"
        ),
    )

with v2:

    st.metric(
        "Precision",
        (
            f"{float(validation_precision):.1%}"
            if validation_precision is not None
            else "N/A"
        ),
    )

with v3:

    st.metric(
        "Recall",
        (
            f"{float(validation_recall):.1%}"
            if validation_recall is not None
            else "N/A"
        ),
    )

with v4:

    st.metric(
        "F1",
        (
            f"{float(validation_f1):.1%}"
            if validation_f1 is not None
            else "N/A"
        ),
    )

with v5:

    st.metric(
        "ROC-AUC",
        (
            f"{float(validation_roc_auc):.1%}"
            if validation_roc_auc is not None
            else "N/A"
        ),
    )


if validation_brier is not None:

    st.caption(
        f"Brier Score: **{float(validation_brier):.4f}** "
        "(lower indicates better probability calibration)"
    )


# =====================================================================
# FEATURE IMPORTANCE
# =====================================================================

top_features = result.get(
    "top_features",
    [],
)

if top_features:

    st.subheader(
        "Top Model Features"
    )

    feature_df = pd.DataFrame(
        top_features
    )

    if (
        "feature" in feature_df.columns
        and "importance" in feature_df.columns
    ):

        feature_df["importance"] = (
            pd.to_numeric(
                feature_df[
                    "importance"
                ],
                errors="coerce",
            )
        )

        feature_df["importance"] = (
            feature_df[
                "importance"
            ].fillna(0.0)
        )

        feature_df["importance"] = (
            feature_df[
                "importance"
            ].map(
                lambda x: f"{x:.2%}"
            )
        )

        st.dataframe(
            feature_df[
                [
                    "feature",
                    "importance",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


# =====================================================================
# NEWS
# =====================================================================

news_df = sentiment.get(
    "news"
)

if (
    isinstance(
        news_df,
        pd.DataFrame,
    )
    and not news_df.empty
):

    with st.expander(
        "📰 Recent News Used for Sentiment"
    ):

        display_columns = [
            column
            for column in [
                "headline",
                "publisher",
                "sentiment",
                "sentiment_score",
                "published",
                "link",
            ]
            if column in news_df.columns
        ]

        if display_columns:

            st.dataframe(
                news_df[
                    display_columns
                ],
                use_container_width=True,
                hide_index=True,
            )


# =====================================================================
# LATEST DATA
# =====================================================================

with st.expander(
    "Latest Market Data"
):

    st.dataframe(
        data.tail(20),
        use_container_width=True,
    )


# =====================================================================
# DIAGNOSTICS
# =====================================================================

with st.expander(
    "System Diagnostics"
):

    st.json(
        {
            "ticker": ticker,
            "horizon": horizon,
            "rows": result.get(
                "n_rows",
                0,
            ),
            "features": result.get(
                "n_features",
                0,
            ),
            "signal": signal,
            "base_signal": base_signal,
            "ml_probability": ml_probability,
            "adjusted_probability": (
                adjusted_probability
            ),
            "confidence": confidence,
            "uncertainty": uncertainty,
            "model_agreement": agreement,
            "sentiment_score": sentiment_score,
            "sentiment_articles": (
                sentiment_articles
            ),
            "validation_accuracy": (
                validation_accuracy
            ),
            "validation_roc_auc": (
                validation_roc_auc
            ),
            "validation_brier": (
                validation_brier
            ),
        }
    )


# =====================================================================
# DISCLAIMER
# =====================================================================

st.divider()

st.caption(
    "Educational/research tool only. "
    "Model probabilities, sentiment, technical indicators and "
    "backtest results are estimates and can be wrong. "
    "They are not guarantees of future returns."
)
