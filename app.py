"""
AI Swing Analyser
=================

Final Streamlit application shell.

The dashboard is intentionally fail-closed:

    Data
      ↓
    Research
      ↓
    Final Holdout
      ↓
    Production Approval
      ↓
    Production Prediction
      ↓
    Swing Setup

If a production-approved model is not available, the application will
display WAIT / RESEARCH STATUS rather than fabricate a trading signal.

This UI does not train models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.research.api_contracts import (
    extract_approved,
    extract_final_holdout_used,
    normalize_horizon_contract,
)
from src.research.production_prediction_integration import (
    ProductionPrediction,
    ProductionPredictionConfig,
    ProductionPredictionIntegration,
    production_prediction_summary,
)


# ---------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="AI Swing Analyser",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------

HORIZONS = (
    1,
    3,
    5,
    10,
    20,
)

MODEL_DIR = Path("models")
ARTIFACT_DIR = Path("artifacts")

STATUS_RESEARCH = "RESEARCH"
STATUS_WAIT = "WAIT"


# ---------------------------------------------------------------------
# STYLING
# ---------------------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .metric-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 10px;
    }

    .status-box {
        border: 1px solid rgba(128,128,128,0.3);
        border-radius: 12px;
        padding: 18px;
        margin: 10px 0;
    }

    .small-text {
        font-size: 0.85rem;
        opacity: 0.75;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "approval_result" not in st.session_state:
    st.session_state.approval_result = None

if "price_data" not in st.session_state:
    st.session_state.price_data = None


# ---------------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_price_data(
    symbol: str,
    period: str = "1y",
) -> pd.DataFrame:
    """
    Load market data.

    This is intentionally isolated from the model pipeline so that the
    UI remains replaceable when the final historical/live data provider
    is connected.
    """

    try:
        import yfinance as yf

        ticker = symbol.strip().upper()

        if not ticker.endswith(".NS"):
            ticker = f"{ticker}.NS"

        data = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )

        if data is None or data.empty:
            return pd.DataFrame()

        if isinstance(
            data.columns,
            pd.MultiIndex,
        ):
            data.columns = [
                column[0]
                for column in data.columns
            ]

        data = data.copy()

        data.columns = [
            str(column).strip().title()
            for column in data.columns
        ]

        required = {
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        }

        if not required.issubset(
            set(data.columns)
        ):
            return pd.DataFrame()

        data = data[
            [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]
        ].dropna()

        return data

    except Exception:
        return pd.DataFrame()


def calculate_basic_indicators(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate dashboard-level indicators."""

    if data.empty:
        return data

    frame = data.copy()

    close = frame["Close"]

    frame["EMA20"] = (
        close.ewm(
            span=20,
            adjust=False,
        ).mean()
    )

    frame["EMA50"] = (
        close.ewm(
            span=50,
            adjust=False,
        ).mean()
    )

    frame["EMA200"] = (
        close.ewm(
            span=200,
            adjust=False,
        ).mean()
    )

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.rolling(
        14
    ).mean()

    avg_loss = loss.rolling(
        14
    ).mean()

    rs = (
        avg_gain
        / avg_loss.replace(
            0,
            np.nan,
        )
    )

    frame["RSI14"] = (
        100
        - (
            100
            / (1 + rs)
        )
    )

    ema12 = close.ewm(
        span=12,
        adjust=False,
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False,
    ).mean()

    frame["MACD"] = (
        ema12 - ema26
    )

    frame["MACD_Signal"] = (
        frame["MACD"]
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    middle = close.rolling(
        20
    ).mean()

    std = close.rolling(
        20
    ).std()

    frame["BB_Middle"] = middle
    frame["BB_Upper"] = (
        middle + 2 * std
    )
    frame["BB_Lower"] = (
        middle - 2 * std
    )

    frame["ATR14"] = (
        pd.concat(
            [
                frame["High"]
                - frame["Low"],
                (
                    frame["High"]
                    - frame["Close"].shift()
                ).abs(),
                (
                    frame["Low"]
                    - frame["Close"].shift()
                ).abs(),
            ],
            axis=1,
        )
        .max(axis=1)
        .rolling(14)
        .mean()
    )

    frame["Volume_SMA20"] = (
        frame["Volume"]
        .rolling(20)
        .mean()
    )

    return frame


def latest_value(
    data: pd.DataFrame,
    column: str,
) -> float | None:
    if (
        data.empty
        or column not in data.columns
    ):
        return None

    value = data[column].iloc[-1]

    if pd.isna(value):
        return None

    return float(value)


def format_probability(
    value: float | None,
) -> str:
    if value is None:
        return "—"

    return f"{value * 100:.1f}%"


def format_price(
    value: float | None,
) -> str:
    if value is None:
        return "—"

    return f"₹{value:,.2f}"


def model_artifacts_available() -> bool:
    """
    Detect whether a production artifact directory exists.

    Existence alone does NOT mean the model is approved.
    """

    if not MODEL_DIR.exists():
        return False

    return any(
        MODEL_DIR.iterdir()
    )


def load_approval_artifact() -> Any:
    """
    Load an explicitly serialized approval result when available.

    JSON is supported as a lightweight interchange format.

    This function never assumes that artifact existence means approval.
    """

    candidates = [
        ARTIFACT_DIR
        / "production_approval.json",
        ARTIFACT_DIR
        / "approval.json",
    ]

    for path in candidates:
        if not path.exists():
            continue

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as handle:
                return json.load(handle)
        except Exception:
            continue

    return None


def safe_target_range(
    current_price: float | None,
    horizon: int,
) -> tuple[
    float | None,
    float | None,
    float | None,
]:
    """
    Return no target unless a real range model is connected.

    The application must never invent price targets.
    """

    return (
        None,
        None,
        None,
    )


def render_price_chart(
    data: pd.DataFrame,
) -> None:
    """Render price and moving-average chart."""

    if data.empty:
        st.info(
            "Price data is not available."
        )
        return

    recent = data.tail(180)

    figure = go.Figure()

    figure.add_trace(
        go.Candlestick(
            x=recent.index,
            open=recent["Open"],
            high=recent["High"],
            low=recent["Low"],
            close=recent["Close"],
            name="Price",
        )
    )

    for column in (
        "EMA20",
        "EMA50",
        "EMA200",
    ):
        if column in recent.columns:
            figure.add_trace(
                go.Scatter(
                    x=recent.index,
                    y=recent[column],
                    mode="lines",
                    name=column,
                )
            )

    figure.update_layout(
        height=500,
        xaxis_rangeslider_visible=False,
        margin=dict(
            l=10,
            r=10,
            t=30,
            b=10,
        ),
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )


def render_indicator_table(
    data: pd.DataFrame,
) -> None:
    """Display current technical indicator state."""

    columns = [
        "Close",
        "EMA20",
        "EMA50",
        "EMA200",
        "RSI14",
        "MACD",
        "MACD_Signal",
        "ATR14",
        "Volume",
        "Volume_SMA20",
    ]

    available = [
        column
        for column in columns
        if column in data.columns
    ]

    if not available:
        return

    latest = data.iloc[-1]

    values = {}

    for column in available:
        value = latest[column]

        if pd.isna(value):
            values[column] = "—"
        elif column == "Close":
            values[column] = (
                f"₹{float(value):,.2f}"
            )
        elif column.startswith(
            "Volume"
        ):
            values[column] = (
                f"{float(value):,.0f}"
            )
        else:
            values[column] = (
                f"{float(value):.2f}"
            )

    indicator_frame = pd.DataFrame(
        {
            "Indicator": list(
                values.keys()
            ),
            "Value": list(
                values.values()
            ),
        }
    )

    st.dataframe(
        indicator_frame,
        use_container_width=True,
        hide_index=True,
    )


def render_horizon_table(
    approval_result: Any,
) -> None:
    """Display approval status across forecast horizons."""

    rows = []

    for horizon in HORIZONS:
        contract = normalize_horizon_contract(
            approval_result,
            horizon,
        )

        rows.append(
            {
                "Horizon": f"{horizon}D",
                "Evaluated": (
                    "YES"
                    if contract.evaluated
                    else "NO"
                ),
                "Passed": (
                    "YES"
                    if contract.passed
                    else "NO"
                ),
                "Approved": (
                    "YES"
                    if contract.approved
                    else "NO"
                ),
                "Accuracy": (
                    format_probability(
                        contract.accuracy
                    )
                ),
                "Confidence": (
                    format_probability(
                        contract.confidence
                    )
                ),
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


def render_research_status(
    approval_result: Any,
) -> None:
    """Display the governance status."""

    st.subheader(
        "Research & Model Governance"
    )

    approved = extract_approved(
        approval_result
    )

    holdout = extract_final_holdout_used(
        approval_result
    )

    if approved and holdout:
        st.success(
            "PRODUCTION APPROVED — "
            "approval and final-holdout evidence detected."
        )
    elif holdout:
        st.warning(
            "FINAL HOLDOUT COMPLETED — "
            "production approval has not been granted."
        )
    else:
        st.info(
            "RESEARCH MODE — "
            "final holdout / production approval evidence "
            "is not available."
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Final Holdout",
            "PASS"
            if holdout
            else "NOT VERIFIED",
        )

    with col2:
        st.metric(
            "Production Approval",
            "PASS"
            if approved
            else "NOT APPROVED",
        )

    with col3:
        st.metric(
            "Executable Signals",
            "ENABLED"
            if approved
            else "DISABLED",
        )


def create_wait_prediction(
    symbol: str,
    horizon: int,
    reason: str,
    *,
    probability_up: float | None = None,
) -> ProductionPrediction:
    """Create an explicit fail-closed WAIT prediction."""

    if probability_up is None:
        probability_up = 0.5

    probability_down = (
        1.0 - probability_up
    )

    return ProductionPrediction(
        symbol=symbol.upper(),
        horizon=horizon,
        probability_up=probability_up,
        probability_down=probability_down,
        predicted_direction=STATUS_WAIT,
        confidence=max(
            probability_up,
            probability_down,
        ),
        production_approved=False,
        final_holdout_used=False,
        executable=False,
        reasons=[reason],
        warnings=[
            "No production-approved model "
            "is available to generate an executable signal."
        ],
        metadata={
            "research_only": True,
            "production_approved": False,
            "final_holdout_used": False,
        },
    )


def generate_dashboard_prediction(
    symbol: str,
    horizon: int,
    approval_result: Any,
) -> ProductionPrediction:
    """
    Generate a production prediction only when a real model output is
    available.

    The current application shell intentionally does not manufacture a
    model probability.
    """

    if approval_result is None:
        return create_wait_prediction(
            symbol,
            horizon,
            "Production approval artifact is unavailable.",
        )

    if not extract_approved(
        approval_result
    ):
        return create_wait_prediction(
            symbol,
            horizon,
            "Model has not passed the production approval gate.",
        )

    if not extract_final_holdout_used(
        approval_result
    ):
        return create_wait_prediction(
            symbol,
            horizon,
            "Final holdout evidence is unavailable.",
        )

    # A real inference adapter will replace this section.
    return create_wait_prediction(
        symbol,
        horizon,
        "Production model inference adapter is not connected.",
    )


# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------

st.sidebar.title(
    "AI Swing Analyser"
)

st.sidebar.caption(
    "Research-first Indian swing trading system"
)

symbol = st.sidebar.text_input(
    "Stock Symbol",
    value="RELIANCE",
    help=(
        "Enter an NSE symbol such as RELIANCE, "
        "TCS, INFY, HDFCBANK, etc."
    ),
).strip().upper()

horizon = st.sidebar.selectbox(
    "Forecast Horizon",
    options=HORIZONS,
    format_func=lambda x: f"{x} Day",
)

period = st.sidebar.selectbox(
    "Historical Chart",
    options=[
        "6mo",
        "1y",
        "2y",
        "5y",
    ],
    index=1,
)

run_analysis = st.sidebar.button(
    "Analyse Stock",
    type="primary",
    use_container_width=True,
)

st.sidebar.divider()

st.sidebar.caption(
    "System policy"
)

st.sidebar.caption(
    "The analyser can return WAIT when "
    "research evidence is insufficient."
)


# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------

st.title(
    "📈 AI Swing Analyser"
)

st.caption(
    "Multi-horizon • Multi-timeframe • "
    "Market-aware • Research-first"
)

st.warning(
    "Research-grade system: no trading signal is considered "
    "executable until the production approval gate passes."
)


# ---------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------

if run_analysis or symbol:

    with st.spinner(
        f"Loading {symbol} market data..."
    ):
        price_data = load_price_data(
            symbol,
            period,
        )

    if price_data.empty:
        st.error(
            f"Unable to retrieve price data for {symbol}."
        )

        st.stop()

    price_data = calculate_basic_indicators(
        price_data
    )

    st.session_state.price_data = (
        price_data
    )


data = st.session_state.price_data

if data is None or data.empty:
    st.info(
        "Enter a stock symbol to begin analysis."
    )
    st.stop()


# ---------------------------------------------------------------------
# MARKET SNAPSHOT
# ---------------------------------------------------------------------

current_price = latest_value(
    data,
    "Close",
)

previous_close = (
    float(data["Close"].iloc[-2])
    if len(data) >= 2
    else None
)

daily_change = None
daily_change_pct = None

if (
    current_price is not None
    and previous_close
    and previous_close != 0
):
    daily_change = (
        current_price
        - previous_close
    )

    daily_change_pct = (
        daily_change
        / previous_close
    )


st.subheader(
    f"{symbol} — Market Snapshot"
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Current Price",
        format_price(
            current_price
        ),
        (
            f"{daily_change_pct * 100:.2f}%"
            if daily_change_pct is not None
            else None
        ),
    )

with col2:
    rsi = latest_value(
        data,
        "RSI14",
    )

    st.metric(
        "RSI(14)",
        (
            f"{rsi:.2f}"
            if rsi is not None
            else "—"
        ),
    )

with col3:
    atr = latest_value(
        data,
        "ATR14",
    )

    st.metric(
        "ATR(14)",
        (
            f"{atr:.2f}"
            if atr is not None
            else "—"
        ),
    )

with col4:
    volume = latest_value(
        data,
        "Volume",
    )

    st.metric(
        "Volume",
        (
            f"{volume:,.0f}"
            if volume is not None
            else "—"
        ),
    )


# ---------------------------------------------------------------------
# PRICE CHART
# ---------------------------------------------------------------------

st.subheader(
    "Price Structure"
)

render_price_chart(
    data
)


# ---------------------------------------------------------------------
# TECHNICAL INDICATORS
# ---------------------------------------------------------------------

with st.expander(
    "Technical Indicators",
    expanded=True,
):
    render_indicator_table(
        data
    )


# ---------------------------------------------------------------------
# MULTI-TIMEFRAME SECTION
# ---------------------------------------------------------------------

st.subheader(
    "Multi-Timeframe Analysis"
)

timeframe_columns = st.columns(4)

timeframes = [
    ("4H", "Research adapter"),
    ("1D", "Daily structure"),
    ("1W", "Weekly structure"),
    ("1M", "Monthly structure"),
]

for column, (
    timeframe,
    description,
) in zip(
    timeframe_columns,
    timeframes,
):
    with column:
        st.markdown(
            f"### {timeframe}"
        )

        st.info(
            description
        )

        st.metric(
            "Status",
            "NOT CONNECTED",
        )

        st.caption(
            "Production timeframe model "
            "will populate this section."
        )


# ---------------------------------------------------------------------
# MARKET REGIME
# ---------------------------------------------------------------------

st.subheader(
    "Market & Sector Regime"
)

regime_col1, regime_col2, regime_col3 = (
    st.columns(3)
)

with regime_col1:
    st.metric(
        "NIFTY Regime",
        "NOT CONNECTED",
    )

with regime_col2:
    st.metric(
        "Sector Strength",
        "NOT CONNECTED",
    )

with regime_col3:
    st.metric(
        "India VIX",
        "NOT CONNECTED",
    )

st.caption(
    "Market, sector and volatility context will be supplied "
    "by the validated market-context pipeline."
)


# ---------------------------------------------------------------------
# PRODUCTION APPROVAL
# ---------------------------------------------------------------------

approval_result = (
    st.session_state.approval_result
)

if approval_result is None:
    approval_result = (
        load_approval_artifact()
    )

render_research_status(
    approval_result
)

if approval_result is not None:
    with st.expander(
        "Approval Evidence by Horizon",
        expanded=False,
    ):
        render_horizon_table(
            approval_result
        )


# ---------------------------------------------------------------------
# AI PREDICTION
# ---------------------------------------------------------------------

st.subheader(
    "AI Direction Forecast"
)

prediction = generate_dashboard_prediction(
    symbol=symbol,
    horizon=horizon,
    approval_result=approval_result,
)

st.session_state.prediction = prediction

prediction_col1, prediction_col2, prediction_col3 = (
    st.columns(3)
)

with prediction_col1:
    st.metric(
        "Direction",
        prediction.predicted_direction,
    )

with prediction_col2:
    st.metric(
        "Probability Up",
        format_probability(
            prediction.probability_up
        ),
    )

with prediction_col3:
    st.metric(
        "Probability Down",
        format_probability(
            prediction.probability_down
        ),
    )


if prediction.executable:
    st.success(
        f"Production signal: "
        f"{prediction.predicted_direction}"
    )
else:
    st.info(
        "WAIT — no executable production signal."
    )

for reason in prediction.reasons:
    st.warning(
        reason
    )


# ---------------------------------------------------------------------
# TARGET RANGE
# ---------------------------------------------------------------------

st.subheader(
    f"{horizon}D Target Range"
)

target_low, target_mid, target_high = (
    safe_target_range(
        current_price,
        horizon,
    )
)

target_col1, target_col2, target_col3 = (
    st.columns(3)
)

with target_col1:
    st.metric(
        "Lower Range",
        format_price(
            target_low
        ),
    )

with target_col2:
    st.metric(
        "Central Target",
        format_price(
            target_mid
        ),
    )

with target_col3:
    st.metric(
        "Upper Range",
        format_price(
            target_high
        ),
    )

st.caption(
    "Targets remain blank until the validated range-prediction "
    "model is connected. No synthetic target prices are generated."
)


# ---------------------------------------------------------------------
# SUPPORT / RESISTANCE
# ---------------------------------------------------------------------

st.subheader(
    "Support & Resistance"
)

recent_window = data.tail(60)

support = (
    float(
        recent_window["Low"].min()
    )
    if not recent_window.empty
    else None
)

resistance = (
    float(
        recent_window["High"].max()
    )
    if not recent_window.empty
    else None
)

sr_col1, sr_col2 = st.columns(2)

with sr_col1:
    st.metric(
        "Recent Support",
        format_price(
            support
        ),
    )

with sr_col2:
    st.metric(
        "Recent Resistance",
        format_price(
            resistance
        ),
    )


# ---------------------------------------------------------------------
# SWING SETUP
# ---------------------------------------------------------------------

st.subheader(
    "Swing Setup"
)

setup_col1, setup_col2 = st.columns(2)

with setup_col1:
    st.markdown(
        "### Trade Decision"
    )

    if prediction.executable:
        st.success(
            prediction.predicted_direction
        )
    else:
        st.info(
            "WAIT"
        )

with setup_col2:
    st.markdown(
        "### Execution Status"
    )

    if prediction.executable:
        st.success(
            "EXECUTABLE"
        )
    else:
        st.error(
            "BLOCKED"
        )

st.caption(
    "A high model probability does not override governance gates."
)


# ---------------------------------------------------------------------
# RISK / POSITION SIZING
# ---------------------------------------------------------------------

st.subheader(
    "Risk Management"
)

risk_col1, risk_col2, risk_col3 = st.columns(3)

with risk_col1:
    capital = st.number_input(
        "Trading Capital (₹)",
        min_value=0.0,
        value=100000.0,
        step=5000.0,
    )

with risk_col2:
    risk_percent = st.number_input(
        "Risk per Trade (%)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
    )

with risk_col3:
    stop_loss_percent = st.number_input(
        "Stop Loss (%)",
        min_value=0.5,
        max_value=20.0,
        value=5.0,
        step=0.5,
    )

risk_amount = (
    capital
    * risk_percent
    / 100.0
)

if current_price is not None:
    stop_distance = (
        current_price
        * stop_loss_percent
        / 100.0
    )

    if stop_distance > 0:
        position_quantity = int(
            risk_amount
            / stop_distance
        )
    else:
        position_quantity = 0
else:
    position_quantity = 0

st.metric(
    "Indicative Position Quantity",
    f"{position_quantity:,}",
)

st.caption(
    "Position sizing is indicative and does not override "
    "the production signal gate."
)


# ---------------------------------------------------------------------
# SAFETY GATES
# ---------------------------------------------------------------------

st.subheader(
    "Safety Gates"
)

gate_data = [
    {
        "Gate": "Historical Data",
        "Status": "PASS"
        if not data.empty
        else "FAIL",
    },
    {
        "Gate": "Feature Pipeline",
        "Status": "RESEARCH",
    },
    {
        "Gate": "Model Selection",
        "Status": (
            "PASS"
            if extract_approved(
                approval_result
            )
            else "NOT APPROVED"
        ),
    },
    {
        "Gate": "Backtest",
        "Status": "RESEARCH",
    },
    {
        "Gate": "Robustness",
        "Status": "RESEARCH",
    },
    {
        "Gate": "Final Holdout",
        "Status": (
            "PASS"
            if extract_final_holdout_used(
                approval_result
            )
            else "NOT VERIFIED"
        ),
    },
    {
        "Gate": "Production Approval",
        "Status": (
            "PASS"
            if extract_approved(
                approval_result
            )
            else "BLOCKED"
        ),
    },
    {
        "Gate": "Executable Signal",
        "Status": (
            "PASS"
            if prediction.executable
            else "BLOCKED"
        ),
    },
]

st.dataframe(
    pd.DataFrame(
        gate_data
    ),
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------------------
# MODEL / RESEARCH STATUS
# ---------------------------------------------------------------------

st.subheader(
    "Model Status"
)

model_col1, model_col2 = st.columns(2)

with model_col1:
    st.metric(
        "Production Artifact",
        (
            "AVAILABLE"
            if model_artifacts_available()
            else "NOT AVAILABLE"
        ),
    )

with model_col2:
    st.metric(
        "Inference",
        (
            "ENABLED"
            if prediction.production_approved
            else "RESEARCH / WAIT"
        ),
    )

st.caption(
    "Artifact availability is not equivalent to production approval."
)


# ---------------------------------------------------------------------
# DISCLAIMER
# ---------------------------------------------------------------------

st.divider()

st.caption(
    """
    Research tool only. This application does not guarantee trading
    performance, profitability, or future accuracy. A model that achieves
    a high historical or holdout accuracy can still fail in live markets.
    Always consider transaction costs, slippage, liquidity, regime changes,
    and position-level risk.
    """
)
