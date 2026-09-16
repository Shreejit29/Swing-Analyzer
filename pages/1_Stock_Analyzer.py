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
- Probability reliability and calibration
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
# SIMPLE INVESTOR SUMMARY HELPERS
# =====================================================================

def _safe_num(value, default=0.0):
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except Exception:
        return default


def _plain_direction(signal_value):
    signal_value = str(signal_value).upper()
    if signal_value == "BUY":
        return "The model sees more evidence for an upward move."
    if signal_value == "SELL":
        return "The model sees more evidence for a downward move."
    return "The model does not see enough evidence in either direction."


def _plain_support(label, score=0.0):
    text = str(label).upper()
    if "STRONG BULL" in text or "BULLISH" in text or text == "BULL":
        return "Supportive"
    if "STRONG BEAR" in text or "BEARISH" in text or text == "BEAR":
        return "Not supportive"
    if "SIDEWAYS" in text or "NEUTRAL" in text:
        return "Mixed / neutral"

    score = _safe_num(score)
    if score > 0.20:
        return "Supportive"
    if score < -0.20:
        return "Not supportive"
    return "Mixed / neutral"


def _plain_trend(value):
    return str(value).replace("_", " ").title()


# =====================================================================
# SIMPLE INVESTOR SUMMARY
# =====================================================================

market_context = result.get("market_context", {})
sector_context = result.get("sector_context", {})
agreement_data = result.get("model_agreement", {})

agreement_simple = _safe_num(agreement_data.get("agreement", 0.0))
bullish_simple = int(agreement_data.get("bullish_models", 0))
bearish_simple = int(agreement_data.get("bearish_models", 0))
total_simple = int(agreement_data.get("total_models", 0))

weekly_simple = result.get("weekly_trend", "UNKNOWN")
monthly_simple = result.get("monthly_trend", "UNKNOWN")
alignment_simple = result.get("three_timeframe_alignment", "UNKNOWN")
regime_simple = result.get("regime", "UNKNOWN")

market_trend_simple = (
    market_context.get("trend", "UNKNOWN")
    if isinstance(market_context, dict)
    else "UNKNOWN"
)
market_score_simple = (
    market_context.get("score", 0.0)
    if isinstance(market_context, dict)
    else 0.0
)

sector_label_simple = (
    sector_context.get("summary", "UNKNOWN")
    if isinstance(sector_context, dict)
    else "UNKNOWN"
)
sector_score_simple = (
    sector_context.get("score", 0.0)
    if isinstance(sector_context, dict)
    else 0.0
)

if total_simple > 0:
    if probability_up >= probability_down:
        model_agreement_text = (
            f"{bullish_simple}/{total_simple} models support an upward move"
        )
    else:
        model_agreement_text = (
            f"{bearish_simple}/{total_simple} models support a downward move"
        )
else:
    model_agreement_text = "Model agreement data is not available"

sentiment_data_simple = result.get("sentiment", {})
if isinstance(sentiment_data_simple, dict):
    sentiment_label_simple = sentiment_data_simple.get(
        "label", sentiment_data_simple.get("sentiment", "UNKNOWN")
    )
    sentiment_articles_simple = int(
        _safe_num(
            sentiment_data_simple.get(
                "articles", sentiment_data_simple.get("article_count", 0)
            )
        )
    )
else:
    sentiment_label_simple = "UNKNOWN"
    sentiment_articles_simple = 0

sentiment_words_simple = str(sentiment_label_simple).upper()
if "POSITIVE" in sentiment_words_simple or "BULL" in sentiment_words_simple:
    news_text_simple = "News is generally positive"
elif "NEGATIVE" in sentiment_words_simple or "BEAR" in sentiment_words_simple:
    news_text_simple = "News is generally negative"
elif sentiment_articles_simple > 0:
    news_text_simple = "News is mixed / neutral"
else:
    news_text_simple = "There is not enough news data"

current_gate_confidence = max(
    probability_up,
    probability_down,
)

st.subheader("🧠 Simple Investor Summary")

if signal == "BUY":
    st.success(
        f"### {ticker} — BUY\n\n"
        f"{_plain_direction(signal)}"
    )
elif signal == "SELL":
    st.error(
        f"### {ticker} — SELL\n\n"
        f"{_plain_direction(signal)}"
    )
else:
    st.warning(
        f"### {ticker} — WAIT\n\n"
        f"{_plain_direction(signal)}"
    )

st.caption(
    "This is a plain-language explanation of the model output. "
    "It is not a guarantee of future price movement."
)

q1, q2, q3, q4 = st.columns(4)

with q1:
    st.metric("AI Confidence", f"{confidence:.1%}")

with q2:
    st.metric(
        "Model Agreement",
        f"{agreement_simple:.0%}" if total_simple else "N/A",
    )

with q3:
    st.metric("Price Trend", _plain_trend(weekly_simple))

with q4:
    st.metric(
        "90% Check",
        "REACHED" if current_gate_confidence >= 0.90 else "NOT REACHED",
    )

st.markdown("#### Why is the model saying this?")

reason_col1, reason_col2 = st.columns(2)

with reason_col1:
    st.write(f"• **AI models:** {model_agreement_text}")
    st.write(f"• **Weekly trend:** {_plain_trend(weekly_simple)}")
    st.write(f"• **Monthly trend:** {_plain_trend(monthly_simple)}")
    st.write(
        f"• **Market environment:** "
        f"{_plain_support(market_trend_simple, market_score_simple)}"
    )

with reason_col2:
    st.write(
        f"• **Sector environment:** "
        f"{_plain_support(sector_label_simple, sector_score_simple)}"
    )
    st.write(
        f"• **News:** {news_text_simple} "
        f"({sentiment_articles_simple} article(s))"
    )
    st.write(
        f"• **Timeframe agreement:** {_plain_trend(alignment_simple)}"
    )
    st.write(
        f"• **Market regime:** {_plain_trend(regime_simple)}"
    )

if current_gate_confidence >= 0.90:
    st.success(
        f"**90% confidence check:** reached at "
        f"**{current_gate_confidence:.1%}**. "
        "The detailed reliability section below shows how this "
        "confidence range performed in historical validation."
    )
elif current_gate_confidence >= 0.80:
    st.warning(
        f"**90% confidence check:** not reached. "
        f"Current model confidence is **{current_gate_confidence:.1%}**."
    )
else:
    st.info(
        f"**90% confidence check:** not reached. "
        f"Current model confidence is **{current_gate_confidence:.1%}**."
    )

if base_signal != signal:
    st.caption(
        f"The initial ensemble signal was **{base_signal}**, "
        f"but additional evidence/risk filters produced the final "
        f"signal **{signal}**."
    )

simple_plan = result.get("trade_plan", {})

if signal in {"BUY", "SELL"} and isinstance(simple_plan, dict):
    entry_simple = simple_plan.get("entry")
    stop_simple = simple_plan.get("stop_loss")
    target_simple = simple_plan.get("target")

    if any(
        value is not None
        for value in [entry_simple, stop_simple, target_simple]
    ):
        st.markdown("#### Risk levels shown by the model")

        rp1, rp2, rp3 = st.columns(3)

        with rp1:
            st.metric(
                "Entry",
                f"₹{float(entry_simple):,.2f}"
                if entry_simple is not None
                else "—",
            )

        with rp2:
            st.metric(
                "Stop Loss",
                f"₹{float(stop_simple):,.2f}"
                if stop_simple is not None
                else "—",
            )

        with rp3:
            st.metric(
                "Target",
                f"₹{float(target_simple):,.2f}"
                if target_simple is not None
                else "—",
            )

st.divider()


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
# COMPONENT VALIDATION & WEIGHTS
# =====================================================================

st.subheader(
    "📊 Ensemble Component Validation"
)

reliability_model = result.get("model")

component_metrics = getattr(
    reliability_model,
    "component_validation_metrics_",
    {},
)

component_weights = getattr(
    reliability_model,
    "component_weights",
    {},
)

if isinstance(component_metrics, dict) and component_metrics:

    validation_rows = []

    for name, metrics in component_metrics.items():

        if not isinstance(metrics, dict):
            continue

        try:
            oof_samples = int(metrics.get("oof_samples", 0))
        except Exception:
            oof_samples = 0

        try:
            logloss = float(metrics.get("log_loss", np.nan))
        except Exception:
            logloss = np.nan

        try:
            roc_auc = float(metrics.get("roc_auc", np.nan))
        except Exception:
            roc_auc = np.nan

        try:
            weight = float(component_weights.get(name, np.nan))
        except Exception:
            weight = np.nan

        validation_rows.append(
            {
                "Model": name,
                "Ensemble Weight": (
                    f"{weight:.1%}"
                    if np.isfinite(weight)
                    else "N/A"
                ),
                "OOF Samples": oof_samples,
                "ROC-AUC": (
                    f"{roc_auc:.3f}"
                    if np.isfinite(roc_auc)
                    else "N/A"
                ),
                "Log Loss": (
                    f"{logloss:.4f}"
                    if np.isfinite(logloss)
                    else "N/A"
                ),
            }
        )

    if validation_rows:

        st.dataframe(
            pd.DataFrame(validation_rows),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "OOF = out-of-fold validation. Ensemble weights are "
            "derived conservatively from chronological validation "
            "performance; lower log loss indicates better probability "
            "quality."
        )

else:

    st.info(
        "Component-level validation diagnostics are not available "
        "for this model run."
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
# PROBABILITY RELIABILITY HELPERS
# =====================================================================

def _extract_validation_probability_data(model):
    """
    Extract out-of-sample validation probabilities and actual labels
    from the fitted classifier.

    Multiple attribute names are supported so the UI remains compatible
    with classifier revisions.
    """
    if model is None:
        return None, None

    probability_names = [
        "validation_probabilities_",
        "validation_probability_",
        "validation_probs_",
    ]

    actual_names = [
        "validation_actuals_",
        "validation_actual_",
        "validation_y_",
        "validation_targets_",
    ]

    probabilities = None
    actuals = None

    for name in probability_names:
        value = getattr(model, name, None)
        if value is not None:
            probabilities = value
            break

    for name in actual_names:
        value = getattr(model, name, None)
        if value is not None:
            actuals = value
            break

    if probabilities is None or actuals is None:
        return None, None

    try:
        probabilities = np.asarray(probabilities, dtype=float)
        actuals = np.asarray(actuals, dtype=int).reshape(-1)

        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            up_probability = probabilities[:, 1]
            down_probability = probabilities[:, 0]
        elif probabilities.ndim == 1:
            up_probability = probabilities
            down_probability = 1.0 - probabilities
        else:
            return None, None

        n = min(
            len(up_probability),
            len(down_probability),
            len(actuals),
        )

        if n == 0:
            return None, None

        frame = pd.DataFrame(
            {
                "probability_up": up_probability[:n],
                "probability_down": down_probability[:n],
                "actual": actuals[:n],
            }
        )

        frame = frame.replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        frame["actual"] = (
            pd.to_numeric(
                frame["actual"],
                errors="coerce",
            )
            .round()
            .astype("Int64")
        )

        frame = frame[
            frame["actual"].isin([0, 1])
        ].copy()

        if frame.empty:
            return None, None

        frame["confidence"] = frame[
            [
                "probability_up",
                "probability_down",
            ]
        ].max(axis=1)

        frame["predicted_class"] = (
            frame["probability_up"] >= 0.50
        ).astype(int)

        frame["correct"] = (
            frame["predicted_class"]
            == frame["actual"].astype(int)
        )

        frame["confidence"] = frame[
            "confidence"
        ].clip(0.0, 1.0)

        return frame, None

    except Exception:
        return None, None


def _build_confidence_calibration(validation_frame):
    """
    Build confidence buckets for both bullish and bearish predictions.
    """
    if (
        validation_frame is None
        or validation_frame.empty
    ):
        return pd.DataFrame()

    bins = [
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.00,
    ]

    labels = [
        "50–60%",
        "60–70%",
        "70–80%",
        "80–90%",
        "90–100%",
    ]

    work = validation_frame.copy()

    work["bucket"] = pd.cut(
        work["confidence"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    rows = []

    for label in labels:
        subset = work[
            work["bucket"] == label
        ]

        if subset.empty:
            continue

        predicted_confidence = float(
            subset["confidence"].mean()
        )

        observed_accuracy = float(
            subset["correct"].mean()
        )

        rows.append(
            {
                "Confidence Bucket": label,
                "Samples": int(len(subset)),
                "Mean Model Confidence": predicted_confidence,
                "Observed Accuracy": observed_accuracy,
                "Calibration Gap": (
                    observed_accuracy
                    - predicted_confidence
                ),
            }
        )

    return pd.DataFrame(rows)


# =====================================================================
# PROBABILITY RELIABILITY DASHBOARD
# =====================================================================

st.subheader(
    "🎯 Probability Reliability"
)

reliability_model = result.get(
    "model"
)

validation_frame, _ = (
    _extract_validation_probability_data(
        reliability_model
    )
)

if (
    validation_frame is None
    or validation_frame.empty
):
    st.info(
        "Historical validation probabilities are not exposed "
        "by the current classifier, so the 90% reliability test "
        "cannot be calculated yet."
    )
    st.caption(
        "This dashboard deliberately does not manufacture a "
        "90% probability from accuracy or other summary metrics."
    )

else:
    calibration_df = (
        _build_confidence_calibration(
            validation_frame
        )
    )

    high_confidence = validation_frame[
        validation_frame["confidence"] >= 0.90
    ]

    validation_samples = len(
        validation_frame
    )

    high_confidence_samples = len(
        high_confidence
    )

    if high_confidence_samples > 0:
        high_confidence_accuracy = float(
            high_confidence["correct"].mean()
        )
        high_confidence_mean = float(
            high_confidence["confidence"].mean()
        )
    else:
        high_confidence_accuracy = np.nan
        high_confidence_mean = np.nan

    # Expected Calibration Error using confidence buckets.
    if not calibration_df.empty:
        calibration_df["_weight"] = (
            calibration_df["Samples"]
            / validation_samples
        )
        ece = float(
            (
                calibration_df["Calibration Gap"].abs()
                * calibration_df["_weight"]
            ).sum()
        )
        calibration_df = calibration_df.drop(
            columns=["_weight"]
        )
    else:
        ece = np.nan

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.metric(
            "Validation Samples",
            f"{validation_samples:,}",
        )

    with r2:
        if np.isfinite(high_confidence_mean):
            st.metric(
                "Current 90% Gate",
                (
                    "PASS"
                    if max(
                        probability_up,
                        probability_down,
                    ) >= 0.90
                    else "NOT REACHED"
                ),
            )
        else:
            st.metric(
                "Current 90% Gate",
                "NO DATA",
            )

    with r3:
        if np.isfinite(
            high_confidence_accuracy
        ):
            st.metric(
                "Observed Accuracy ≥90%",
                f"{high_confidence_accuracy:.1%}",
            )
        else:
            st.metric(
                "Observed Accuracy ≥90%",
                "No samples",
            )

    with r4:
        if np.isfinite(ece):
            st.metric(
                "Confidence ECE",
                f"{ece:.3f}",
            )
        else:
            st.metric(
                "Confidence ECE",
                "N/A",
            )

    st.caption(
        "The ≥90% figure is a validation bucket, not a forced "
        "model output. Its observed accuracy is calculated only "
        "from historical validation predictions."
    )

    if high_confidence_samples > 0:
        st.write(
            f"**≥90% confidence validation sample:** "
            f"{high_confidence_samples:,} observations"
        )
    else:
        st.write(
            "**≥90% confidence validation sample:** "
            "No observations in the current validation split."
        )

    if not calibration_df.empty:
        display_calibration = calibration_df.copy()

        display_calibration[
            "Mean Model Confidence"
        ] = display_calibration[
            "Mean Model Confidence"
        ].map(
            lambda x: f"{x:.1%}"
        )

        display_calibration[
            "Observed Accuracy"
        ] = display_calibration[
            "Observed Accuracy"
        ].map(
            lambda x: f"{x:.1%}"
        )

        display_calibration[
            "Calibration Gap"
        ] = display_calibration[
            "Calibration Gap"
        ].map(
            lambda x: f"{x:+.1%}"
        )

        with st.expander(
            "View Probability Calibration Table",
            expanded=True,
        ):
            st.dataframe(
                display_calibration,
                use_container_width=True,
                hide_index=True,
            )

        chart = go.Figure()

        chart.add_trace(
            go.Scatter(
                x=calibration_df[
                    "Mean Model Confidence"
                ],
                y=calibration_df[
                    "Observed Accuracy"
                ],
                mode="lines+markers",
                name="Observed",
            )
        )

        chart.add_trace(
            go.Scatter(
                x=[0.50, 1.00],
                y=[0.50, 1.00],
                mode="lines",
                name="Perfect Calibration",
                line=dict(
                    dash="dash"
                ),
            )
        )

        chart.update_layout(
            title="Confidence Calibration",
            xaxis_title="Mean Model Confidence",
            yaxis_title="Observed Accuracy",
            xaxis=dict(
                range=[0.50, 1.00],
                tickformat=".0%",
            ),
            yaxis=dict(
                range=[0.50, 1.00],
                tickformat=".0%",
            ),
            height=420,
            margin=dict(
                l=20,
                r=20,
                t=50,
                b=20,
            ),
        )

        st.plotly_chart(
            chart,
            use_container_width=True,
        )

    # Explicit current probability interpretation.
    current_confidence = max(
        probability_up,
        probability_down,
    )

    if current_confidence >= 0.90:
        st.success(
            f"Current ensemble confidence is "
            f"**{current_confidence:.1%}**. "
            "This qualifies for the ≥90% confidence bucket."
        )
    elif current_confidence >= 0.80:
        st.warning(
            f"Current ensemble confidence is "
            f"**{current_confidence:.1%}**. "
            "It has not reached the ≥90% high-confidence gate."
        )
    else:
        st.info(
            f"Current ensemble confidence is "
            f"**{current_confidence:.1%}**. "
            "The model is not presenting this setup as a ≥90% "
            "confidence prediction."
        )

    st.caption(
        "Calibration gap = observed accuracy − mean model confidence. "
        "A negative gap means the model was overconfident in that bucket; "
        "a positive gap means it was underconfident."
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
