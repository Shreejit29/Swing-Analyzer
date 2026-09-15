"""
AI Swing Analyser - Model Status Dashboard.

This page exposes the model lifecycle:

    RESEARCH
       ↓
    VALIDATION
       ↓
    CALIBRATION
       ↓
    RANGE VALIDATION
       ↓
    REGIME VALIDATION
       ↓
    BACKTEST
       ↓
    ROBUSTNESS
       ↓
    APPROVED

The Streamlit interface is read-only with respect to approval.
Production approval must be performed by the model governance layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.models.model_registry import (
    ModelRegistry,
)


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

ROOT_DIR = Path(
    __file__
).resolve().parents[1]

ARTIFACT_DIR = (
    ROOT_DIR
    / "artifacts"
)

REGISTRY_PATHS = [
    ARTIFACT_DIR
    / "registry.json",
    ROOT_DIR
    / "models"
    / "registry.json",
]


# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="AI Swing Analyser — Model Status",
    page_icon="🛡️",
    layout="wide",
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def locate_registry() -> Path | None:
    for path in REGISTRY_PATHS:
        if path.exists():
            return path

    return None


def safe_float(
    value: Any,
) -> float | None:
    try:
        return float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return None


def percentage(
    value: Any,
) -> str:
    number = safe_float(
        value
    )

    if number is None:
        return "N/A"

    return (
        f"{number * 100:.2f}%"
    )


def number(
    value: Any,
    decimals: int = 2,
) -> str:
    number_value = safe_float(
        value
    )

    if number_value is None:
        return "N/A"

    return (
        f"{number_value:.{decimals}f}"
    )


def get_field(
    entry: Any,
    *names: str,
    default: Any = None,
) -> Any:
    """
    Retrieve a value from either a registry dataclass or dict.
    """

    for name in names:
        if isinstance(
            entry,
            dict,
        ):
            if name in entry:
                return entry[
                    name
                ]

        else:
            value = getattr(
                entry,
                name,
                None,
            )

            if value is not None:
                return value

    return default


def bool_status(
    value: Any,
) -> str:
    if value is True:
        return "PASS"

    if value is False:
        return "FAIL"

    return "NOT EVALUATED"


def gate_class(
    status: str,
) -> str:
    status = status.upper()

    if status == "PASS":
        return "success"

    if status == "FAIL":
        return "error"

    return "warning"


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------

st.title(
    "🛡️ Model Status & Governance"
)

st.caption(
    "Controlled model lifecycle and production-readiness monitoring."
)


# ----------------------------------------------------------------------
# Safety statement
# ----------------------------------------------------------------------

st.warning(
    "This page is informational. It does not override the model "
    "registry or production approval engine."
)


# ----------------------------------------------------------------------
# Registry loading
# ----------------------------------------------------------------------

registry_path = locate_registry()

if registry_path is None:
    st.info(
        "No model registry has been created yet."
    )

    st.markdown(
        """
        ### Model lifecycle

        ```text
        RESEARCH
            ↓
        VALIDATION
            ↓
        HOLDOUT
            ↓
        CALIBRATION
            ↓
        RANGE VALIDATION
            ↓
        REGIME STABILITY
            ↓
        BACKTEST
            ↓
        ROBUSTNESS
            ↓
        APPROVED
        ```
        """
    )

    st.stop()


try:
    registry = ModelRegistry(
        registry_path
    )

except Exception as exc:
    st.error(
        "Unable to load the model registry."
    )
    st.exception(
        exc
    )
    st.stop()


# ----------------------------------------------------------------------
# Retrieve registry entries
# ----------------------------------------------------------------------

try:
    research_models = (
        registry.research_models()
    )
except Exception:
    research_models = []

try:
    approved_models = (
        registry.approved_models()
    )
except Exception:
    approved_models = []

try:
    rejected_models = (
        registry.rejected_models()
    )
except Exception:
    rejected_models = []


all_models: list[
    Any
] = []

seen_ids: set[
    str
] = set()

for collection in [
    research_models,
    approved_models,
    rejected_models,
]:
    for entry in collection:
        model_id = str(
            get_field(
                entry,
                "model_id",
                "id",
                default="",
            )
        )

        if model_id in seen_ids:
            continue

        seen_ids.add(
            model_id
        )

        all_models.append(
            entry
        )


# ----------------------------------------------------------------------
# Summary metrics
# ----------------------------------------------------------------------

st.subheader(
    "Model Inventory"
)

col1, col2, col3, col4 = st.columns(
    4
)

with col1:
    st.metric(
        "Total Models",
        len(
            all_models
        ),
    )

with col2:
    st.metric(
        "Research",
        len(
            research_models
        ),
    )

with col3:
    st.metric(
        "Approved",
        len(
            approved_models
        ),
    )

with col4:
    st.metric(
        "Rejected",
        len(
            rejected_models
        ),
    )


# ----------------------------------------------------------------------
# Model selector
# ----------------------------------------------------------------------

if not all_models:
    st.info(
        "The registry contains no model entries yet."
    )
    st.stop()


model_options: dict[
    str,
    Any,
] = {}

for entry in all_models:
    model_id = str(
        get_field(
            entry,
            "model_id",
            "id",
            default="UNKNOWN",
        )
    )

    model_options[
        model_id
    ] = entry


selected_model_id = st.selectbox(
    "Select Model",
    options=list(
        model_options.keys()
    ),
)

selected_model = model_options[
    selected_model_id
]


# ----------------------------------------------------------------------
# Model metadata
# ----------------------------------------------------------------------

st.subheader(
    "Model Information"
)

metadata_col1, metadata_col2, metadata_col3, metadata_col4 = (
    st.columns(
        4
    )
)

with metadata_col1:
    st.write(
        "**Symbol**"
    )
    st.write(
        get_field(
            selected_model,
            "symbol",
            default="N/A",
        )
    )

with metadata_col2:
    st.write(
        "**Timeframe**"
    )
    st.write(
        get_field(
            selected_model,
            "timeframe",
            default="N/A",
        )
    )

with metadata_col3:
    st.write(
        "**Horizon**"
    )
    st.write(
        get_field(
            selected_model,
            "horizon",
            default="N/A",
        )
    )

with metadata_col4:
    st.write(
        "**Status**"
    )

    status = str(
        get_field(
            selected_model,
            "status",
            default="UNKNOWN",
        )
    ).upper()

    if status == "APPROVED":
        st.success(
            status
        )
    elif status == "REJECTED":
        st.error(
            status
        )
    else:
        st.warning(
            status
        )


# ----------------------------------------------------------------------
# Performance
# ----------------------------------------------------------------------

st.subheader(
    "Performance Evidence"
)

performance = pd.DataFrame(
    {
        "Metric": [
            "Validation Accuracy",
            "Final Holdout Accuracy",
            "Generalization Gap",
            "Feature Hash",
        ],
        "Value": [
            percentage(
                get_field(
                    selected_model,
                    "validation_accuracy",
                    "validation_acc",
                )
            ),
            percentage(
                get_field(
                    selected_model,
                    "holdout_accuracy",
                    "final_holdout_accuracy",
                    "test_accuracy",
                )
            ),
            percentage(
                get_field(
                    selected_model,
                    "generalization_gap",
                )
            ),
            str(
                get_field(
                    selected_model,
                    "feature_hash",
                    default="N/A",
                )
            ),
        ],
    }
)

st.dataframe(
    performance,
    use_container_width=True,
    hide_index=True,
)


# ----------------------------------------------------------------------
# Mandatory gates
# ----------------------------------------------------------------------

st.subheader(
    "Mandatory Approval Gates"
)

gate_definitions = [
    (
        "Validation",
        [
            "validation_passed",
            "validation",
        ],
    ),
    (
        "Final Holdout",
        [
            "holdout_passed",
            "final_holdout_passed",
        ],
    ),
    (
        "Leakage Check",
        [
            "leakage_free",
            "no_leakage",
        ],
    ),
    (
        "Calibration",
        [
            "calibration_passed",
        ],
    ),
    (
        "Range Validation",
        [
            "range_validation_passed",
            "range_passed",
        ],
    ),
    (
        "Regime Stability",
        [
            "regime_stability_passed",
            "regime_passed",
        ],
    ),
    (
        "Backtest",
        [
            "backtest_passed",
        ],
    ),
    (
        "Robustness",
        [
            "robustness_passed",
        ],
    ),
]


gate_rows: list[
    dict[str, Any]
] = []

for label, names in gate_definitions:
    value = get_field(
        selected_model,
        *names,
        default=None,
    )

    gate_rows.append(
        {
            "Gate": label,
            "Status": bool_status(
                value
            ),
        }
    )

gate_table = pd.DataFrame(
    gate_rows
)

st.dataframe(
    gate_table,
    use_container_width=True,
    hide_index=True,
)


# ----------------------------------------------------------------------
# Visual gate summary
# ----------------------------------------------------------------------

for row in gate_rows:
    gate = row[
        "Gate"
    ]

    gate_status = row[
        "Status"
    ]

    if gate_status == "PASS":
        st.success(
            f"✓ {gate}: PASS"
        )
    elif gate_status == "FAIL":
        st.error(
            f"✗ {gate}: FAIL"
        )
    else:
        st.warning(
            f"• {gate}: NOT EVALUATED"
        )


# ----------------------------------------------------------------------
# Approval interpretation
# ----------------------------------------------------------------------

st.subheader(
    "Production Interpretation"
)

required_gates = [
    get_field(
        selected_model,
        *names,
        default=None,
    )
    for _, names in gate_definitions
]

all_passed = (
    all(
        value is True
        for value in required_gates
    )
    and status == "APPROVED"
)

if all_passed:
    st.success(
        "This registry entry reports APPROVED status and all displayed "
        "mandatory gates are passing."
    )

    st.info(
        "Inference should still load the artifact through the controlled "
        "registry/approval path rather than directly from a file."
    )
else:
    st.error(
        "This model is NOT production-approved."
    )

    st.write(
        "The model must remain in research, be corrected, or be rejected "
        "until every mandatory gate is satisfied."
    )


# ----------------------------------------------------------------------
# Artifact verification
# ----------------------------------------------------------------------

st.subheader(
    "Artifact"
)

artifact_path = get_field(
    selected_model,
    "artifact_path",
)

if artifact_path:
    artifact = Path(
        str(
            artifact_path
        )
    )

    if not artifact.is_absolute():
        artifact = (
            ROOT_DIR
            / artifact
        )

    if artifact.exists():
        st.success(
            f"Artifact exists: {artifact}"
        )

        st.write(
            f"Size: {artifact.stat().st_size:,} bytes"
        )
    else:
        st.error(
            "Registry references an artifact that does not exist."
        )

        st.code(
            str(
                artifact
            )
        )
else:
    st.warning(
        "No artifact path is recorded for this model."
    )


# ----------------------------------------------------------------------
# Notes
# ----------------------------------------------------------------------

notes = get_field(
    selected_model,
    "notes",
    default=None,
)

if notes:
    st.subheader(
        "Research Notes"
    )

    st.write(
        notes
    )


# ----------------------------------------------------------------------
# Lifecycle explanation
# ----------------------------------------------------------------------

with st.expander(
    "Understand the approval lifecycle"
):
    st.markdown(
        """
        ### 1. Research

        Models can be experimented with freely, but they cannot generate
        production trading decisions.

        ### 2. Validation

        Performance is measured on chronologically unseen observations.

        ### 3. Final Holdout

        The final holdout remains untouched until model selection is frozen.

        ### 4. Calibration

        Probabilities are calibrated on a separate temporal dataset.

        ### 5. Range Validation

        Target-price intervals are evaluated independently from direction.

        ### 6. Regime Stability

        Performance is checked across different market conditions.

        ### 7. Backtest

        Trading performance is evaluated with execution assumptions,
        transaction costs and slippage.

        ### 8. Robustness

        The strategy is stress-tested against adverse assumptions.

        ### 9. Approval

        Only when every mandatory gate passes can the model enter the
        approved registry state.
        """
    )


# ----------------------------------------------------------------------
# Refresh
# ----------------------------------------------------------------------

st.divider()

if st.button(
    "🔄 Refresh Registry",
    use_container_width=True,
):
    st.rerun()


st.caption(
    "AI Swing Analyser • Model Governance • "
    "Approval is controlled outside the Streamlit UI."
)
