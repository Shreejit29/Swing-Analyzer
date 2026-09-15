"""
AI Swing Analyser - Stock Analyzer.

This page provides the user interface for analysing an individual
Indian stock.

Important:
    - It does not bypass model approval.
    - It does not manufacture a BUY/SELL signal.
    - It does not treat raw model probability as calibrated confidence.
    - It will only enable production prediction once an approved
      artifact is available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.data.cache import (
    CacheConfig,
    DataCache,
)
from src.data.downloader import (
    download_daily,
)
from src.data.quality import (
    QualityReport,
    clean_ohlcv,
    validate_ohlcv,
)
from src.features.engine import (
    engineer_features,
)


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

ROOT_DIR = Path(
    __file__
).resolve().parents[1]

CACHE_DIR = (
    ROOT_DIR
    / "data"
    / "cache"
)


# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="AI Swing Analyser — Stock Analyzer",
    page_icon="📈",
    layout="wide",
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def normalize_symbol(
    symbol: str,
) -> str:
    """
    Normalize an Indian NSE symbol.

    Examples:
        RELIANCE
        RELIANCE.NS
    """

    symbol = (
        symbol.strip()
        .upper()
    )

    if not symbol:
        return ""

    if (
        symbol.endswith(
            ".NS"
        )
        or symbol.endswith(
            ".BO"
        )
    ):
        return symbol

    return (
        f"{symbol}.NS"
    )


def format_price(
    value: Any,
) -> str:
    try:
        return (
            f"₹{float(value):,.2f}"
        )
    except (
        TypeError,
        ValueError,
    ):
        return "N/A"


def format_percent(
    value: Any,
) -> str:
    try:
        return (
            f"{float(value) * 100:.2f}%"
        )
    except (
        TypeError,
        ValueError,
    ):
        return "N/A"


def calculate_summary(
    data: pd.DataFrame,
) -> dict[str, Any]:
    """
    Calculate simple descriptive statistics.

    These are descriptive only and must not be interpreted as forecasts.
    """

    close = data[
        "Close"
    ].dropna()

    volume = data[
        "Volume"
    ].dropna()

    if close.empty:
        return {}

    latest = float(
        close.iloc[-1]
    )

    previous = (
        float(
            close.iloc[-2]
        )
        if len(close) >= 2
        else latest
    )

    daily_return = (
        latest / previous
        - 1.0
        if previous != 0
        else np.nan
    )

    return {
        "latest": latest,
        "daily_return": daily_return,
        "high_52w": float(
            close.tail(
                252
            ).max()
        ),
        "low_52w": float(
            close.tail(
                252
            ).min()
        ),
        "volume": float(
            volume.iloc[-1]
        )
        if not volume.empty
        else np.nan,
    }


def quality_message(
    report: QualityReport,
) -> None:
    if report.passed:
        st.success(
            "Data quality checks passed."
        )
    else:
        st.error(
            "Data quality checks failed. "
            "This dataset must not be used for model training."
        )

    if report.errors:
        with st.expander(
            "Quality errors"
        ):
            for error in report.errors:
                st.write(
                    f"• {error}"
                )

    if report.warnings:
        with st.expander(
            "Quality warnings"
        ):
            for warning in report.warnings:
                st.write(
                    f"• {warning}"
                )


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------

st.title(
    "📈 Stock Analyzer"
)

st.caption(
    "Research-first Indian equity analysis. "
    "Production signals require an approved model."
)


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------

st.sidebar.header(
    "Stock Selection"
)

symbol_input = st.sidebar.text_input(
    "NSE Symbol",
    value="RELIANCE",
    help=(
        "Enter an NSE equity symbol, "
        "for example RELIANCE, TCS, INFY or HDFCBANK."
    ),
)

symbol = normalize_symbol(
    symbol_input
)

period = st.sidebar.selectbox(
    "Historical Period",
    options=[
        "1y",
        "2y",
        "5y",
        "10y",
    ],
    index=1,
)

run_analysis = st.sidebar.button(
    "🔍 Analyze Stock",
    type="primary",
    use_container_width=True,
)


# ----------------------------------------------------------------------
# Empty state
# ----------------------------------------------------------------------

if not symbol:
    st.info(
        "Enter an NSE stock symbol to begin."
    )
    st.stop()


# ----------------------------------------------------------------------
# Analyze
# ----------------------------------------------------------------------

if run_analysis:
    st.session_state[
        "analysis_symbol"
    ] = symbol

    st.session_state[
        "analysis_period"
    ] = period

    st.session_state[
        "analysis_requested"
    ] = True


if not st.session_state.get(
    "analysis_requested",
    False,
):
    st.info(
        f"Ready to analyze **{symbol}**. "
        "Click **Analyze Stock**."
    )
    st.stop()


symbol = st.session_state.get(
    "analysis_symbol",
    symbol,
)

period = st.session_state.get(
    "analysis_period",
    period,
)


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------

with st.spinner(
    f"Loading historical data for {symbol}..."
):
    try:
        data = download_daily(
            symbol,
            period=period,
        )

    except Exception as exc:
        st.error(
            "Unable to download market data."
        )

        st.exception(
            exc
        )

        st.stop()


if data.empty:
    st.error(
        "No market data was returned."
    )
    st.stop()


# ----------------------------------------------------------------------
# Data cleaning and quality
# ----------------------------------------------------------------------

try:
    report = validate_ohlcv(
        data
    )

except Exception as exc:
    st.error(
        "Data validation failed."
    )
    st.exception(
        exc
    )
    st.stop()


quality_message(
    report
)

if not report.passed:
    st.stop()


cleaned = clean_ohlcv(
    data
)

if cleaned.empty:
    st.error(
        "No usable observations remain after cleaning."
    )
    st.stop()


# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------

summary = calculate_summary(
    cleaned
)

st.subheader(
    f"{symbol} — Market Overview"
)

col1, col2, col3, col4, col5 = st.columns(
    5
)

with col1:
    st.metric(
        "Latest Price",
        format_price(
            summary.get(
                "latest"
            )
        ),
    )

with col2:
    st.metric(
        "Daily Change",
        format_percent(
            summary.get(
                "daily_return"
            )
        ),
    )

with col3:
    st.metric(
        "52W High",
        format_price(
            summary.get(
                "high_52w"
            )
        ),
    )

with col4:
    st.metric(
        "52W Low",
        format_price(
            summary.get(
                "low_52w"
            )
        ),
    )

with col5:
    volume_value = summary.get(
        "volume"
    )

    if volume_value is not None:
        volume_text = (
            f"{volume_value:,.0f}"
        )
    else:
        volume_text = "N/A"

    st.metric(
        "Latest Volume",
        volume_text,
    )


# ----------------------------------------------------------------------
# Price chart
# ----------------------------------------------------------------------

st.subheader(
    "Price History"
)

chart_data = cleaned[
    [
        "Open",
        "High",
        "Low",
        "Close",
    ]
].copy()

st.line_chart(
    chart_data[
        "Close"
    ],
    use_container_width=True,
)


# ----------------------------------------------------------------------
# Technical feature engineering
# ----------------------------------------------------------------------

st.subheader(
    "Technical Analysis"
)

with st.spinner(
    "Engineering technical, price-action, volume and regime features..."
):
    try:
        feature_result = engineer_features(
            cleaned
        )

    except Exception as exc:
        st.error(
            "Feature engineering failed."
        )
        st.exception(
            exc
        )
        st.stop()


features = feature_result.data

feature_names = (
    feature_result.feature_names
)


# ----------------------------------------------------------------------
# Technical indicators
# ----------------------------------------------------------------------

indicator_col1, indicator_col2, indicator_col3, indicator_col4 = (
    st.columns(
        4
    )
)

latest_features = features.iloc[
    -1
]

with indicator_col1:
    rsi = latest_features.get(
        "RSI_14",
        np.nan,
    )

    st.metric(
        "RSI (14)",
        (
            f"{float(rsi):.2f}"
            if pd.notna(rsi)
            else "N/A"
        ),
    )

with indicator_col2:
    macd = latest_features.get(
        "MACD",
        np.nan,
    )

    st.metric(
        "MACD",
        (
            f"{float(macd):.4f}"
            if pd.notna(macd)
            else "N/A"
        ),
    )

with indicator_col3:
    atr_pct = latest_features.get(
        "ATR_Pct",
        np.nan,
    )

    st.metric(
        "ATR %",
        format_percent(
            atr_pct
        ),
    )

with indicator_col4:
    adx = latest_features.get(
        "ADX_14",
        np.nan,
    )

    st.metric(
        "ADX",
        (
            f"{float(adx):.2f}"
            if pd.notna(adx)
            else "N/A"
        ),
    )


# ----------------------------------------------------------------------
# Moving averages
# ----------------------------------------------------------------------

st.subheader(
    "Moving Average Structure"
)

ma_columns = [
    column
    for column in [
        "EMA_9",
        "EMA_20",
        "EMA_50",
        "EMA_100",
        "EMA_200",
        "SMA_20",
        "SMA_50",
        "SMA_200",
    ]
    if column in features.columns
]

if ma_columns:
    ma_table = pd.DataFrame(
        {
            "Indicator": ma_columns,
            "Latest Value": [
                features.iloc[
                    -1
                ][column]
                for column in ma_columns
            ],
        }
    )

    ma_table[
        "Latest Value"
    ] = ma_table[
        "Latest Value"
    ].map(
        lambda value: (
            f"₹{value:,.2f}"
            if pd.notna(value)
            else "N/A"
        )
    )

    st.dataframe(
        ma_table,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info(
        "Moving-average features are not available."
    )


# ----------------------------------------------------------------------
# Regime
# ----------------------------------------------------------------------

st.subheader(
    "Market Regime"
)

regime_columns = [
    column
    for column in [
        "Trend_Regime",
        "ADX_Regime",
        "Volatility_Regime",
        "Momentum_Regime",
        "Market_State_Score",
    ]
    if column in features.columns
]

if regime_columns:
    regime_table = pd.DataFrame(
        {
            "Regime Feature": regime_columns,
            "Current Value": [
                features.iloc[
                    -1
                ][column]
                for column in regime_columns
            ],
        }
    )

    st.dataframe(
        regime_table,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info(
        "Regime features are not available."
    )


# ----------------------------------------------------------------------
# Feature snapshot
# ----------------------------------------------------------------------

with st.expander(
    "View latest feature snapshot"
):
    snapshot = (
        features[
            feature_names
        ]
        .tail(1)
        .T
    )

    snapshot.columns = [
        "Latest Value"
    ]

    st.dataframe(
        snapshot,
        use_container_width=True,
    )


# ----------------------------------------------------------------------
# Model status
# ----------------------------------------------------------------------

st.subheader(
    "AI Prediction Status"
)

st.markdown(
    """
    <div class="warning-box">
        <strong>Prediction protection is active.</strong><br>
        This page will not generate a live BUY/SELL recommendation until
        a model has passed the required validation, holdout, leakage,
        calibration, range, regime, backtest and robustness gates.
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Production artifact discovery
# ----------------------------------------------------------------------

artifact_files: list[
    Path
] = []

artifact_dir = (
    ROOT_DIR
    / "artifacts"
)

if artifact_dir.exists():
    artifact_files = sorted(
        artifact_dir.glob(
            "*.joblib"
        )
    )

if not artifact_files:
    st.info(
        "No model artifacts are currently available for production "
        "inference."
    )
else:
    st.write(
        f"Found {len(artifact_files)} persisted model artifact(s)."
    )

    artifact_table = pd.DataFrame(
        {
            "Artifact": [
                path.name
                for path in artifact_files
            ],
            "Status": [
                "Approval required"
                for _ in artifact_files
            ],
        }
    )

    st.dataframe(
        artifact_table,
        use_container_width=True,
        hide_index=True,
    )


# ----------------------------------------------------------------------
# Research interpretation
# ----------------------------------------------------------------------

st.subheader(
    "Research Interpretation"
)

st.info(
    "The indicators shown above are descriptive market features. "
    "They are not independently sufficient to predict future price "
    "direction."
)

interpretation_items = [
    "RSI measures recent momentum and should be interpreted with trend and regime.",
    "MACD describes momentum/trend structure rather than guaranteeing reversal.",
    "ATR describes volatility and can help determine risk distance.",
    "ADX measures trend strength, not direction by itself.",
    "Moving-average structure helps describe trend alignment.",
    "Volume features help evaluate price-volume confirmation.",
    "Support/resistance features are generated from historical observations.",
    "The eventual AI prediction must combine these features with multi-timeframe and market context.",
]

for item in interpretation_items:
    st.write(
        f"• {item}"
    )


# ----------------------------------------------------------------------
# Data table
# ----------------------------------------------------------------------

with st.expander(
    "View recent OHLCV data"
):
    st.dataframe(
        cleaned.tail(
            50
        ),
        use_container_width=True,
    )


# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------

st.divider()

st.caption(
    "AI Swing Analyser • Stock Analyzer • "
    "Research mode until production approval is achieved."
)
