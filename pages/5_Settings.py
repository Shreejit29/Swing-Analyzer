"""
AI Swing Analyser - Settings Dashboard.

This page exposes configuration used by the research application.

Important:
    - UI changes are session-level only.
    - settings.yaml remains the controlled configuration source.
    - Production approval rules must not be silently weakened from the UI.
    - No setting on this page can approve a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st
import yaml


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)

CONFIG_PATH = (
    ROOT_DIR
    / "config"
    / "settings.yaml"
)


# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="AI Swing Analyser — Settings",
    page_icon="⚙️",
    layout="wide",
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(
            file
        )

    return data or {}


def get_nested(
    data: dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    current: Any = data

    for key in keys:
        if not isinstance(
            current,
            dict,
        ):
            return default

        current = current.get(
            key
        )

    return current


def initialize_session_state(
    config: dict[str, Any],
) -> None:
    defaults = {
        "settings_direction_threshold": get_nested(
            config,
            "prediction",
            "direction_threshold",
            default=0.0,
        ),
        "settings_min_accuracy": get_nested(
            config,
            "validation",
            "minimum_accuracy",
            default=0.95,
        ),
        "settings_risk": get_nested(
            config,
            "risk",
            "max_risk_per_trade",
            default=0.01,
        ),
        "settings_min_rr": get_nested(
            config,
            "risk",
            "minimum_risk_reward",
            default=2.0,
        ),
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[
                key
            ] = value


def reset_session_settings(
    config: dict[str, Any],
) -> None:
    st.session_state[
        "settings_direction_threshold"
    ] = get_nested(
        config,
        "prediction",
        "direction_threshold",
        default=0.0,
    )

    st.session_state[
        "settings_min_accuracy"
    ] = get_nested(
        config,
        "validation",
        "minimum_accuracy",
        default=0.95,
    )

    st.session_state[
        "settings_risk"
    ] = get_nested(
        config,
        "risk",
        "max_risk_per_trade",
        default=0.01,
    )

    st.session_state[
        "settings_min_rr"
    ] = get_nested(
        config,
        "risk",
        "minimum_risk_reward",
        default=2.0,
    )


# ----------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------

try:
    config = load_config()
except Exception as exc:
    st.error(
        "Unable to load settings.yaml."
    )
    st.exception(
        exc
    )
    st.stop()


initialize_session_state(
    config
)


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------

st.title(
    "⚙️ Settings"
)

st.caption(
    "Research configuration and system controls."
)


# ----------------------------------------------------------------------
# Safety notice
# ----------------------------------------------------------------------

st.warning(
    "Settings shown here do not directly approve models or bypass "
    "production safety gates. Configuration changes are session-level "
    "until a controlled configuration workflow is implemented."
)


# ----------------------------------------------------------------------
# Market configuration
# ----------------------------------------------------------------------

st.subheader(
    "Market Configuration"
)

market = config.get(
    "market",
    {},
)

market_col1, market_col2 = st.columns(
    2
)

with market_col1:
    st.text_input(
        "Country",
        value=str(
            market.get(
                "country",
                "India",
            )
        ),
        disabled=True,
    )

with market_col2:
    st.text_input(
        "Timezone",
        value=str(
            market.get(
                "timezone",
                "Asia/Kolkata",
            )
        ),
        disabled=True,
    )


indices = market.get(
    "indices",
    {},
)

st.write(
    "**Configured Market Indices**"
)

index_rows = [
    {
        "Market": "NIFTY 50",
        "Symbol": indices.get(
            "nifty50",
            "^NSEI",
        ),
    },
    {
        "Market": "SENSEX",
        "Symbol": indices.get(
            "sensex",
            "^BSESN",
        ),
    },
    {
        "Market": "NIFTY Bank",
        "Symbol": indices.get(
            "nifty_bank",
            "^NSEBANK",
        ),
    },
]

st.dataframe(
    index_rows,
    use_container_width=True,
    hide_index=True,
)


# ----------------------------------------------------------------------
# Timeframes
# ----------------------------------------------------------------------

st.subheader(
    "Timeframes"
)

timeframes = config.get(
    "timeframes",
    {},
)

configured_timeframes = timeframes.get(
    "primary",
    [
        "4H",
        "1D",
        "1W",
        "1M",
    ],
)

st.multiselect(
    "Primary Analysis Timeframes",
    options=[
        "4H",
        "1D",
        "1W",
        "1M",
    ],
    default=[
        timeframe
        for timeframe in configured_timeframes
        if timeframe
        in [
            "4H",
            "1D",
            "1W",
            "1M",
        ]
    ],
    disabled=True,
    help=(
        "The production multi-timeframe architecture is controlled "
        "through the project configuration."
    ),
)


# ----------------------------------------------------------------------
# Prediction horizons
# ----------------------------------------------------------------------

st.subheader(
    "Prediction Horizons"
)

prediction = config.get(
    "prediction",
    {},
)

configured_horizons = prediction.get(
    "horizons",
    [
        1,
        3,
        5,
        10,
        20,
    ],
)

st.multiselect(
    "Forecast Horizons",
    options=[
        1,
        3,
        5,
        10,
        20,
    ],
    default=[
        int(
            horizon
        )
        for horizon in configured_horizons
        if int(
            horizon
        )
        in [
            1,
            3,
            5,
            10,
            20,
        ]
    ],
    disabled=True,
)


# ----------------------------------------------------------------------
# Validation configuration
# ----------------------------------------------------------------------

st.subheader(
    "Validation"
)

validation = config.get(
    "validation",
    {},
)

validation_col1, validation_col2 = st.columns(
    2
)

with validation_col1:
    st.number_input(
        "Test Fraction",
        min_value=0.05,
        max_value=0.40,
        value=float(
            validation.get(
                "test_fraction",
                0.20,
            )
        ),
        step=0.05,
        disabled=True,
    )

with validation_col2:
    st.number_input(
        "Validation Fraction",
        min_value=0.05,
        max_value=0.40,
        value=float(
            validation.get(
                "validation_fraction",
                0.20,
            )
        ),
        step=0.05,
        disabled=True,
    )


# ----------------------------------------------------------------------
# 95% gate
# ----------------------------------------------------------------------

st.markdown(
    "### Minimum Accuracy Gate"
)

minimum_accuracy = st.number_input(
    "Minimum Accuracy",
    min_value=0.50,
    max_value=0.999,
    value=float(
        st.session_state[
            "settings_min_accuracy"
        ]
    ),
    step=0.01,
    key="settings_min_accuracy",
)

if minimum_accuracy < 0.95:
    st.error(
        "The project's strict production research gate is 95%. "
        "Lowering this session value cannot make a model production "
        "approved."
    )
else:
    st.success(
        f"Configured research gate: "
        f"{minimum_accuracy * 100:.2f}%"
    )


# ----------------------------------------------------------------------
# Risk settings
# ----------------------------------------------------------------------

st.subheader(
    "Risk Management"
)

risk = config.get(
    "risk",
    {},
)

risk_col1, risk_col2 = st.columns(
    2
)

with risk_col1:
    max_risk = st.number_input(
        "Maximum Risk Per Trade",
        min_value=0.001,
        max_value=0.05,
        value=float(
            st.session_state[
                "settings_risk"
            ]
        ),
        step=0.001,
        format="%.3f",
        key="settings_risk",
    )

with risk_col2:
    minimum_rr = st.number_input(
        "Minimum Risk / Reward",
        min_value=1.0,
        max_value=10.0,
        value=float(
            st.session_state[
                "settings_min_rr"
            ]
        ),
        step=0.25,
        key="settings_min_rr",
    )


if max_risk > 0.02:
    st.warning(
        "Risk above 2% per trade is aggressive for a research swing "
        "trading system."
    )

if minimum_rr < 2.0:
    st.warning(
        "The current production target is a minimum 2.0 risk/reward "
        "ratio."
    )


# ----------------------------------------------------------------------
# Model configuration
# ----------------------------------------------------------------------

st.subheader(
    "Model Configuration"
)

model_config = config.get(
    "model",
    {},
)

model_rows = [
    {
        "Model",
        "Enabled",
    }
]

model_rows = [
    {
        "Model": "Logistic Regression",
        "Enabled": bool(
            get_nested(
                model_config,
                "logistic_regression",
                "enabled",
                default=True,
            )
        ),
    },
    {
        "Model": "Random Forest",
        "Enabled": bool(
            get_nested(
                model_config,
                "random_forest",
                "enabled",
                default=True,
            )
        ),
    },
    {
        "Model": "Gradient Boosting",
        "Enabled": bool(
            get_nested(
                model_config,
                "gradient_boosting",
                "enabled",
                default=True,
            )
        ),
    },
]

st.dataframe(
    model_rows,
    use_container_width=True,
    hide_index=True,
)


# ----------------------------------------------------------------------
# Feature configuration
# ----------------------------------------------------------------------

st.subheader(
    "Feature Configuration"
)

features = config.get(
    "features",
    {},
)

feature_rows: list[
    dict[str, Any]
] = []

for feature_name, feature_config in features.items():
    if not isinstance(
        feature_config,
        dict,
    ):
        feature_rows.append(
            {
                "Feature": feature_name,
                "Enabled": feature_config,
            }
        )
        continue

    feature_rows.append(
        {
            "Feature": feature_name,
            "Enabled": feature_config.get(
                "enabled",
                True,
            ),
        }
    )

if feature_rows:
    st.dataframe(
        feature_rows,
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info(
        "No feature configuration found."
    )


# ----------------------------------------------------------------------
# Session controls
# ----------------------------------------------------------------------

st.subheader(
    "Session Controls"
)

col1, col2 = st.columns(
    2
)

with col1:
    if st.button(
        "↩ Reset Session Settings",
        use_container_width=True,
    ):
        reset_session_settings(
            config
        )
        st.rerun()

with col2:
    if st.button(
        "🔄 Reload settings.yaml",
        use_container_width=True,
    ):
        st.rerun()


# ----------------------------------------------------------------------
# Configuration preview
# ----------------------------------------------------------------------

st.subheader(
    "Configuration Preview"
)

with st.expander(
    "View current settings.yaml"
):
    st.code(
        yaml.safe_dump(
            config,
            sort_keys=False,
        ),
        language="yaml",
    )


# ----------------------------------------------------------------------
# Current limitations
# ----------------------------------------------------------------------

with st.expander(
    "Configuration safety notes"
):
    notes = [
        "The Streamlit UI cannot approve models.",
        "The final holdout cannot be weakened or bypassed through this page.",
        "The 95% production research threshold is intentionally strict.",
        "Changing a research threshold does not retroactively change stored experiment results.",
        "Model artifacts must remain tied to their feature schema.",
        "Risk settings are not a substitute for portfolio-level risk controls.",
        "Production configuration should eventually be version-controlled with experiment IDs.",
        "Any change to a production model's feature or preprocessing schema should create a new model version.",
    ]

    for note in notes:
        st.write(
            f"• {note}"
        )


# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------

st.divider()

st.caption(
    "AI Swing Analyser • Configuration • "
    "Session changes are not production approval."
)
