"""
AI Swing Analyser - Research Dashboard.

This page is intentionally research-first.

It does not generate live trading signals. Its purpose is to expose:
    - research status
    - data quality
    - validation gates
    - model approval status
    - experiment information
    - important safety warnings

Production trading decisions must come through the approved inference
and gating layers.
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import pandas as pd
import streamlit as st


# ----------------------------------------------------------------------
# Application paths
# ----------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    ROOT_DIR
    / "config"
    / "settings.yaml"
)

EXPERIMENT_DIR = (
    ROOT_DIR
    / "experiments"
)

ARTIFACT_DIR = (
    ROOT_DIR
    / "artifacts"
)


# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="AI Swing Analyser — Research",
    page_icon="📊",
    layout="wide",
)


# ----------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #666;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }

        .warning-box {
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #e0a800;
            background-color: #fff8e1;
            margin: 1rem 0;
        }

        .danger-box {
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #dc3545;
            background-color: #fff0f0;
            margin: 1rem 0;
        }

        .success-box {
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #198754;
            background-color: #f0fff5;
            margin: 1rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def load_yaml_config() -> dict[str, Any]:
    """
    Load the project configuration.

    PyYAML is already part of requirements.txt.
    """

    if not CONFIG_PATH.exists():
        return {}

    try:
        import yaml

        with CONFIG_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = yaml.safe_load(
                file
            )

        return data or {}

    except Exception as exc:
        st.warning(
            f"Could not load configuration: {exc}"
        )
        return {}


def load_experiment_files() -> list[dict[str, Any]]:
    """
    Load saved experiment JSON files.

    Invalid files are skipped rather than crashing the dashboard.
    """

    if not EXPERIMENT_DIR.exists():
        return []

    records: list[
        dict[str, Any]
    ] = []

    for path in sorted(
        EXPERIMENT_DIR.glob(
            "*.json"
        )
    ):
        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(
                    file
                )

            if isinstance(
                data,
                dict,
            ):
                data["_file"] = path.name
                records.append(
                    data
                )

        except Exception:
            continue

    return records


def load_registry() -> dict[str, Any]:
    """
    Read the model registry if it exists.

    The registry remains the source of truth for model lifecycle.
    """

    registry_path = (
        ROOT_DIR
        / "models"
        / "registry.json"
    )

    if not registry_path.exists():
        registry_path = (
            ROOT_DIR
            / "artifacts"
            / "registry.json"
        )

    if not registry_path.exists():
        return {}

    try:
        with registry_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(
                file
            )

        return data or {}

    except Exception:
        return {}


def count_artifacts() -> int:
    """
    Count persisted model artifacts.
    """

    if not ARTIFACT_DIR.exists():
        return 0

    return len(
        list(
            ARTIFACT_DIR.glob(
                "*.joblib"
            )
        )
    )


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


def extract_accuracy(
    record: dict[str, Any],
) -> float | None:
    """
    Try common locations used by experiment results.

    This function is deliberately defensive because experiment schema
    will evolve as the research pipeline becomes more sophisticated.
    """

    candidates = [
        record.get(
            "accuracy"
        ),
        record.get(
            "validation_accuracy"
        ),
        record.get(
            "test_accuracy"
        ),
        record.get(
            "final_holdout_accuracy"
        ),
    ]

    metrics = record.get(
        "metrics"
    )

    if isinstance(
        metrics,
        dict,
    ):
        candidates.extend(
            [
                metrics.get(
                    "accuracy"
                ),
                metrics.get(
                    "validation_accuracy"
                ),
                metrics.get(
                    "test_accuracy"
                ),
            ]
        )

    for candidate in candidates:
        value = safe_float(
            candidate
        )

        if value is not None:
            return value

    return None


def format_percent(
    value: float | None,
) -> str:
    if value is None:
        return "N/A"

    return (
        f"{value * 100:.2f}%"
    )


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------

st.markdown(
    '<div class="main-title">📊 AI Swing Analyser</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Research & Validation Dashboard"
    "</div>",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

config = load_yaml_config()

validation_config = config.get(
    "validation",
    {},
)

prediction_config = config.get(
    "prediction",
    {},
)

timeframe_config = config.get(
    "timeframes",
    {},
)


minimum_accuracy = safe_float(
    validation_config.get(
        "minimum_accuracy",
        0.95,
    )
)

if minimum_accuracy is None:
    minimum_accuracy = 0.95


# ----------------------------------------------------------------------
# Safety notice
# ----------------------------------------------------------------------

st.markdown(
    """
    <div class="warning-box">
        <strong>Research Mode</strong><br>
        This dashboard does not imply that any model is profitable,
        production-ready, or suitable for live trading.
        A model must pass the complete approval framework before it can
        be used by the production inference layer.
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Research status
# ----------------------------------------------------------------------

experiments = load_experiment_files()
registry = load_registry()
artifact_count = count_artifacts()


approved_count = 0
research_count = 0
rejected_count = 0

if isinstance(
    registry,
    dict,
):
    entries = registry.get(
        "models",
        registry.get(
            "entries",
            [],
        ),
    )

    if isinstance(
        entries,
        list,
    ):
        for entry in entries:
            if not isinstance(
                entry,
                dict,
            ):
                continue

            status = str(
                entry.get(
                    "status",
                    ""
                )
            ).upper()

            if status == "APPROVED":
                approved_count += 1
            elif status == "RESEARCH":
                research_count += 1
            elif status == "REJECTED":
                rejected_count += 1


# ----------------------------------------------------------------------
# KPI row
# ----------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(
    4
)

with col1:
    st.metric(
        "Experiments",
        len(
            experiments
        ),
    )

with col2:
    st.metric(
        "Model Artifacts",
        artifact_count,
    )

with col3:
    st.metric(
        "Approved Models",
        approved_count,
    )

with col4:
    st.metric(
        "95% Accuracy Gate",
        format_percent(
            minimum_accuracy
        ),
    )


# ----------------------------------------------------------------------
# Research framework
# ----------------------------------------------------------------------

st.subheader(
    "Research Framework"
)

framework = pd.DataFrame(
    {
        "Stage": [
            "Historical data",
            "Data quality",
            "Feature engineering",
            "Temporal validation",
            "Calibration",
            "Target-price range validation",
            "Regime stability",
            "Walk-forward backtest",
            "Robustness testing",
            "Production approval",
        ],
        "Purpose": [
            "Build reproducible historical datasets",
            "Detect missing, invalid and inconsistent market data",
            "Generate technical, price-action, volume and regime features",
            "Measure genuine out-of-sample predictive performance",
            "Make probabilities meaningful and usable for confidence gating",
            "Validate predicted future price intervals",
            "Check performance across market conditions",
            "Evaluate trading performance without look-ahead",
            "Stress-test costs, slippage and trade sequencing",
            "Allow only models that pass all mandatory gates",
        ],
        "Status": [
            "Implemented",
            "Implemented",
            "Implemented",
            "Implemented",
            "Implemented",
            "Implemented",
            "Implemented",
            "Implemented",
            "Implemented",
            "Implemented",
        ],
    }
)

st.dataframe(
    framework,
    use_container_width=True,
    hide_index=True,
)


# ----------------------------------------------------------------------
# Validation gates
# ----------------------------------------------------------------------

st.subheader(
    "Validation Gates"
)

gate_data = pd.DataFrame(
    {
        "Gate": [
            "Minimum validation accuracy",
            "Final holdout accuracy",
            "Out-of-sample validation",
            "Walk-forward validation",
            "No data leakage",
            "Probability calibration",
            "Range calibration",
            "Regime stability",
            "Realistic backtest",
            "Robustness",
        ],
        "Required": [
            format_percent(
                minimum_accuracy
            ),
            format_percent(
                minimum_accuracy
            ),
            "YES",
            "YES",
            "YES",
            "YES",
            "YES",
            "YES",
            "YES",
            "YES",
        ],
        "Current dashboard status": [
            "NOT EVALUATED",
            "NOT EVALUATED",
            "NOT EVALUATED",
            "NOT EVALUATED",
            "NOT EVALUATED",
            "NOT EVALUATED",
            "NOT EVALUATED",
            "NOT EVALUATED",
            "NOT EVALUATED",
            "NOT EVALUATED",
        ],
    }
)

st.dataframe(
    gate_data,
    use_container_width=True,
    hide_index=True,
)


# ----------------------------------------------------------------------
# Important interpretation
# ----------------------------------------------------------------------

st.markdown(
    """
    <div class="danger-box">
        <strong>Important:</strong>
        The 95% threshold is a research approval gate, not a claim that
        stock-market direction can reliably be predicted with 95%
        accuracy. If the model fails the threshold on genuinely unseen
        data, it must remain in research or be rejected.
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Timeframes and horizons
# ----------------------------------------------------------------------

left, right = st.columns(
    2
)

with left:
    st.subheader(
        "Configured Timeframes"
    )

    configured_timeframes = (
        timeframe_config.get(
            "primary",
            [
                "4H",
                "1D",
                "1W",
                "1M",
            ],
        )
    )

    for timeframe in configured_timeframes:
        st.write(
            f"• {timeframe}"
        )

with right:
    st.subheader(
        "Prediction Horizons"
    )

    horizons = prediction_config.get(
        "horizons",
        [
            1,
            3,
            5,
            10,
            20,
        ],
    )

    for horizon in horizons:
        st.write(
            f"• {horizon} trading day(s)"
        )


# ----------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------

st.subheader(
    "Recent Experiments"
)

if not experiments:
    st.info(
        "No saved experiments found yet. "
        "Run the research pipeline to populate this section."
    )
else:
    rows: list[
        dict[str, Any]
    ] = []

    for experiment in experiments:
        accuracy = extract_accuracy(
            experiment
        )

        rows.append(
            {
                "Experiment": experiment.get(
                    "experiment_id",
                    experiment.get(
                        "id",
                        experiment.get(
                            "_file",
                            "Unknown",
                        ),
                    ),
                ),
                "Symbol": experiment.get(
                    "symbol",
                    "N/A",
                ),
                "Timeframe": experiment.get(
                    "timeframe",
                    "N/A",
                ),
                "Horizon": experiment.get(
                    "horizon",
                    "N/A",
                ),
                "Accuracy": format_percent(
                    accuracy
                ),
                "Status": str(
                    experiment.get(
                        "status",
                        "RESEARCH",
                    )
                ).upper(),
            }
        )

    experiment_table = pd.DataFrame(
        rows
    )

    st.dataframe(
        experiment_table,
        use_container_width=True,
        hide_index=True,
    )


# ----------------------------------------------------------------------
# Research controls
# ----------------------------------------------------------------------

st.subheader(
    "Research Controls"
)

control_col1, control_col2 = st.columns(
    2
)

with control_col1:
    if st.button(
        "🔄 Refresh Research State",
        use_container_width=True,
    ):
        st.cache_data.clear()
        st.rerun()

with control_col2:
    if st.button(
        "🧹 Clear Dashboard Messages",
        use_container_width=True,
    ):
        st.rerun()


# ----------------------------------------------------------------------
# Current limitations
# ----------------------------------------------------------------------

st.subheader(
    "Current Research Limitations"
)

limitations = [
    "Historical intraday data must be sufficiently long before 4H models can be trusted.",
    "Yahoo Finance intraday history is not sufficient for multi-year 4H research.",
    "NSE session-aware 4H aggregation must be used rather than naive calendar resampling.",
    "The final holdout must remain untouched during model selection.",
    "Feature selection and hyperparameter optimization must not use the final holdout.",
    "Model probabilities must be calibrated before being treated as confidence.",
    "Predicted ranges must be validated independently from directional accuracy.",
    "Backtest performance must include realistic transaction costs and slippage.",
    "No model should be forced to generate a trade when confidence is insufficient.",
]

for limitation in limitations:
    st.write(
        f"• {limitation}"
    )


# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------

st.divider()

st.caption(
    "AI Swing Analyser • Research-first architecture • "
    "No live-trading recommendation is generated by this page."
)
