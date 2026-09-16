"""
Probability Reliability & Calibration
=====================================

Provides probability reliability analysis for the swing-trading
ensemble.

The purpose is NOT to artificially increase probabilities.

Instead, this module answers:

    "When the model says 90%, how often was the outcome actually
     positive?"

This allows the application to distinguish:

    Model probability
    from
    Empirically validated probability.

The module is intentionally independent of the classifier so that
the existing ML pipeline remains backward compatible.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd


# =====================================================================
# CONSTANTS
# =====================================================================


DEFAULT_BINS = np.array(
    [
        0.00,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.00,
    ],
    dtype=float,
)


# =====================================================================
# SAFE HELPERS
# =====================================================================


def _safe_float(
    value: Any,
    default: float = np.nan,
) -> float:

    try:

        result = float(value)

        if np.isfinite(result):
            return result

    except Exception:
        pass

    return float(default)


def _to_numeric_array(
    values: Any,
) -> np.ndarray:

    if values is None:
        return np.array([], dtype=float)

    try:

        if isinstance(values, pd.Series):
            array = pd.to_numeric(
                values,
                errors="coerce",
            ).to_numpy(dtype=float)

        elif isinstance(values, pd.DataFrame):

            if values.shape[1] == 0:
                return np.array([], dtype=float)

            array = pd.to_numeric(
                values.iloc[:, 0],
                errors="coerce",
            ).to_numpy(dtype=float)

        else:

            array = np.asarray(
                values,
                dtype=float,
            ).reshape(-1)

    except Exception:

        return np.array([], dtype=float)

    return array


def _extract_validation_data(
    model: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract validation probabilities and actual outcomes from the
    existing classifier.

    Multiple attribute names are supported for backward compatibility.
    """

    probability_candidates = [
        "validation_probabilities_",
        "validation_probabilities",
        "validation_probs_",
        "validation_probs",
        "validation_probability_",
        "validation_probability",
    ]

    actual_candidates = [
        "validation_actual_",
        "validation_actual",
        "validation_actuals_",
        "validation_actuals",
        "validation_y_",
        "validation_y",
        "y_validation_",
        "y_validation",
    ]

    probabilities = None
    actuals = None

    for name in probability_candidates:

        if hasattr(model, name):

            probabilities = getattr(
                model,
                name,
                None,
            )

            if probabilities is not None:
                break

    for name in actual_candidates:

        if hasattr(model, name):

            actuals = getattr(
                model,
                name,
                None,
            )

            if actuals is not None:
                break

    # ---------------------------------------------------------------
    # Some implementations store these values inside model_summary.
    # ---------------------------------------------------------------

    if probabilities is None:

        try:

            summary = model.summary()

            if isinstance(summary, dict):

                for name in probability_candidates:

                    if name in summary:

                        probabilities = summary[name]
                        break

                if probabilities is None:

                    for name in [
                        "validation_probabilities",
                        "validation_probs",
                        "validation_probability",
                    ]:

                        if name in summary:

                            probabilities = summary[name]
                            break

        except Exception:
            pass

    if actuals is None:

        try:

            summary = model.summary()

            if isinstance(summary, dict):

                for name in actual_candidates:

                    if name in summary:

                        actuals = summary[name]
                        break

                if actuals is None:

                    for name in [
                        "validation_actual",
                        "validation_actuals",
                        "validation_y",
                    ]:

                        if name in summary:

                            actuals = summary[name]
                            break

        except Exception:
            pass

    probabilities = _to_numeric_array(
        probabilities
    )

    actuals = _to_numeric_array(
        actuals
    )

    # ---------------------------------------------------------------
    # If probabilities are stored as two-class probabilities,
    # extract P(UP).
    # ---------------------------------------------------------------

    try:

        raw = probabilities

        if raw.ndim == 2 and raw.shape[1] >= 2:
            probabilities = raw[:, 1]

    except Exception:
        pass

    # ---------------------------------------------------------------
    # Ensure equal lengths.
    # ---------------------------------------------------------------

    n = min(
        len(probabilities),
        len(actuals),
    )

    if n <= 0:

        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    probabilities = probabilities[:n]
    actuals = actuals[:n]

    # ---------------------------------------------------------------
    # Keep only valid values.
    # ---------------------------------------------------------------

    mask = (
        np.isfinite(probabilities)
        & np.isfinite(actuals)
    )

    probabilities = probabilities[mask]
    actuals = actuals[mask]

    probabilities = np.clip(
        probabilities,
        0.0,
        1.0,
    )

    actuals = (
        actuals >= 0.5
    ).astype(int)

    return probabilities, actuals


# =====================================================================
# BRIER SCORE
# =====================================================================


def brier_score(
    probabilities: Iterable[float],
    actuals: Iterable[int],
) -> float:

    probabilities = _to_numeric_array(
        probabilities
    )

    actuals = _to_numeric_array(
        actuals
    )

    n = min(
        len(probabilities),
        len(actuals),
    )

    if n == 0:
        return np.nan

    probabilities = probabilities[:n]
    actuals = actuals[:n]

    mask = (
        np.isfinite(probabilities)
        & np.isfinite(actuals)
    )

    if not mask.any():
        return np.nan

    return float(
        np.mean(
            (
                probabilities[mask]
                - actuals[mask]
            )
            ** 2
        )
    )


# =====================================================================
# CALIBRATION TABLE
# =====================================================================


def calibration_table(
    probabilities: Iterable[float],
    actuals: Iterable[int],
    bins: Optional[Iterable[float]] = None,
) -> pd.DataFrame:
    """
    Build a probability calibration table.

    Columns:

        bucket
        lower
        upper
        predicted_probability
        actual_rate
        calibration_error
        count

    actual_rate is the historical positive-outcome frequency within
    each probability bucket.
    """

    probabilities = _to_numeric_array(
        probabilities
    )

    actuals = _to_numeric_array(
        actuals
    )

    n = min(
        len(probabilities),
        len(actuals),
    )

    if n == 0:

        return pd.DataFrame(
            columns=[
                "bucket",
                "lower",
                "upper",
                "predicted_probability",
                "actual_rate",
                "calibration_error",
                "count",
            ]
        )

    probabilities = probabilities[:n]
    actuals = actuals[:n]

    mask = (
        np.isfinite(probabilities)
        & np.isfinite(actuals)
    )

    probabilities = np.clip(
        probabilities[mask],
        0.0,
        1.0,
    )

    actuals = (
        actuals[mask] >= 0.5
    ).astype(int)

    if bins is None:
        bins = DEFAULT_BINS

    bins = np.asarray(
        list(bins),
        dtype=float,
    )

    bins = np.unique(
        np.clip(
            bins,
            0.0,
            1.0,
        )
    )

    if len(bins) < 2:

        bins = DEFAULT_BINS.copy()

    rows = []

    for i in range(len(bins) - 1):

        lower = float(
            bins[i]
        )

        upper = float(
            bins[i + 1]
        )

        # Last bucket includes 1.0.
        if i == len(bins) - 2:

            mask_bucket = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )

        else:

            mask_bucket = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        count = int(
            mask_bucket.sum()
        )

        if count > 0:

            predicted = float(
                np.mean(
                    probabilities[
                        mask_bucket
                    ]
                )
            )

            actual = float(
                np.mean(
                    actuals[
                        mask_bucket
                    ]
                )
            )

            error = float(
                actual - predicted
            )

        else:

            predicted = np.nan
            actual = np.nan
            error = np.nan

        rows.append(
            {
                "bucket": (
                    f"{lower:.0%}"
                    f"–"
                    f"{upper:.0%}"
                ),
                "lower": lower,
                "upper": upper,
                "predicted_probability": predicted,
                "actual_rate": actual,
                "calibration_error": error,
                "count": count,
            }
        )

    return pd.DataFrame(rows)


# =====================================================================
# 90% CONFIDENCE ANALYSIS
# =====================================================================


def high_confidence_analysis(
    probabilities: Iterable[float],
    actuals: Iterable[int],
    threshold: float = 0.90,
    minimum_samples: int = 10,
) -> Dict[str, Any]:
    """
    Validate the high-confidence probability region.

    A 90% model probability is considered empirically reliable only
    when enough historical validation samples exist and the observed
    outcome rate is close to the predicted probability.

    This function does not artificially modify probabilities.
    """

    probabilities = _to_numeric_array(
        probabilities
    )

    actuals = _to_numeric_array(
        actuals
    )

    n = min(
        len(probabilities),
        len(actuals),
    )

    if n == 0:

        return {
            "threshold": float(threshold),
            "samples": 0,
            "minimum_samples": int(
                minimum_samples
            ),
            "predicted_probability": np.nan,
            "actual_rate": np.nan,
            "calibration_error": np.nan,
            "reliable_sample_size": False,
            "status": "NO VALIDATION DATA",
        }

    probabilities = probabilities[:n]
    actuals = actuals[:n]

    mask = (
        np.isfinite(probabilities)
        & np.isfinite(actuals)
    )

    probabilities = np.clip(
        probabilities[mask],
        0.0,
        1.0,
    )

    actuals = (
        actuals[mask] >= 0.5
    ).astype(int)

    threshold = float(
        np.clip(
            threshold,
            0.50,
            0.99,
        )
    )

    selected = (
        probabilities >= threshold
    )

    count = int(
        selected.sum()
    )

    if count == 0:

        return {
            "threshold": threshold,
            "samples": 0,
            "minimum_samples": int(
                minimum_samples
            ),
            "predicted_probability": np.nan,
            "actual_rate": np.nan,
            "calibration_error": np.nan,
            "reliable_sample_size": False,
            "status": "NO SAMPLES ABOVE THRESHOLD",
        }

    predicted = float(
        np.mean(
            probabilities[selected]
        )
    )

    actual = float(
        np.mean(
            actuals[selected]
        )
    )

    error = float(
        actual - predicted
    )

    enough_samples = (
        count >= int(minimum_samples)
    )

    absolute_error = abs(error)

    if not enough_samples:

        status = "INSUFFICIENT SAMPLE"

    elif absolute_error <= 0.05:

        status = "WELL CALIBRATED"

    elif absolute_error <= 0.10:

        status = "MODERATELY CALIBRATED"

    else:

        status = "POORLY CALIBRATED"

    return {
        "threshold": threshold,
        "samples": count,
        "minimum_samples": int(
            minimum_samples
        ),
        "predicted_probability": predicted,
        "actual_rate": actual,
        "calibration_error": error,
        "absolute_calibration_error": (
            absolute_error
        ),
        "reliable_sample_size": (
            enough_samples
        ),
        "status": status,
    }


# =====================================================================
# OVERALL RELIABILITY REPORT
# =====================================================================


def reliability_report(
    model: Any = None,
    probabilities: Any = None,
    actuals: Any = None,
    high_confidence_threshold: float = 0.90,
    minimum_high_confidence_samples: int = 10,
) -> Dict[str, Any]:
    """
    Generate the complete reliability report.

    Either:

        reliability_report(model=model)

    or:

        reliability_report(
            probabilities=...,
            actuals=...,
        )
    """

    # ---------------------------------------------------------------
    # Obtain data.
    # ---------------------------------------------------------------

    if (
        probabilities is None
        or actuals is None
    ):

        if model is None:

            probabilities_array = (
                np.array([], dtype=float)
            )

            actuals_array = (
                np.array([], dtype=float)
            )

        else:

            (
                probabilities_array,
                actuals_array,
            ) = _extract_validation_data(
                model
            )

    else:

        probabilities_array = (
            _to_numeric_array(
                probabilities
            )
        )

        actuals_array = (
            _to_numeric_array(
                actuals
            )
        )

    # ---------------------------------------------------------------
    # Align arrays.
    # ---------------------------------------------------------------

    n = min(
        len(probabilities_array),
        len(actuals_array),
    )

    probabilities_array = (
        probabilities_array[:n]
    )

    actuals_array = (
        actuals_array[:n]
    )

    # ---------------------------------------------------------------
    # Core metrics.
    # ---------------------------------------------------------------

    score = brier_score(
        probabilities_array,
        actuals_array,
    )

    table = calibration_table(
        probabilities_array,
        actuals_array,
    )

    high_confidence = (
        high_confidence_analysis(
            probabilities_array,
            actuals_array,
            threshold=(
                high_confidence_threshold
            ),
            minimum_samples=(
                minimum_high_confidence_samples
            ),
        )
    )

    # ---------------------------------------------------------------
    # Mean calibration error.
    #
    # Weighted by number of observations in each bucket.
    # ---------------------------------------------------------------

    mean_absolute_error = np.nan

    if not table.empty:

        valid = table[
            table["count"] > 0
        ]

        if not valid.empty:

            weights = valid[
                "count"
            ].to_numpy(
                dtype=float
            )

            errors = np.abs(
                valid[
                    "calibration_error"
                ].to_numpy(
                    dtype=float
                )
            )

            if weights.sum() > 0:

                mean_absolute_error = float(
                    np.average(
                        errors,
                        weights=weights,
                    )
                )

    # ---------------------------------------------------------------
    # Reliability interpretation.
    # ---------------------------------------------------------------

    if n == 0:

        overall_status = (
            "NO VALIDATION DATA"
        )

    elif np.isfinite(score):

        if score <= 0.05:
            overall_status = (
                "GOOD PROBABILITY QUALITY"
            )

        elif score <= 0.10:
            overall_status = (
                "MODERATE PROBABILITY QUALITY"
            )

        else:
            overall_status = (
                "PROBABILITY NEEDS CALIBRATION"
            )

    else:

        overall_status = "UNKNOWN"

    return {

        "n_validation_samples": int(n),

        "brier_score": (
            float(score)
            if np.isfinite(score)
            else np.nan
        ),

        "mean_absolute_calibration_error": (
            mean_absolute_error
        ),

        "calibration_table": table,

        "high_confidence": (
            high_confidence
        ),

        "overall_status": (
            overall_status
        ),

        "has_validation_data": (
            n > 0
        ),

        "probabilities": (
            probabilities_array
        ),

        "actuals": (
            actuals_array
        ),
    }


# =====================================================================
# MODEL COMPATIBILITY HELPER
# =====================================================================


def get_model_reliability(
    model: Any,
    high_confidence_threshold: float = 0.90,
    minimum_high_confidence_samples: int = 10,
) -> Dict[str, Any]:
    """
    Convenience wrapper for the existing SwingClassifier.
    """

    return reliability_report(
        model=model,
        high_confidence_threshold=(
            high_confidence_threshold
        ),
        minimum_high_confidence_samples=(
            minimum_high_confidence_samples
        ),
    )


# =====================================================================
# CURRENT PROBABILITY CLASSIFICATION
# =====================================================================


def classify_probability(
    probability: float,
) -> Dict[str, Any]:
    """
    Describe the current probability without implying that it is
    historically calibrated.

    Example:

        0.63 -> MODERATE
        0.78 -> STRONG
        0.93 -> VERY HIGH
    """

    p = float(
        np.clip(
            probability,
            0.0,
            1.0,
        )
    )

    if p >= 0.90:

        label = "VERY HIGH"

    elif p >= 0.75:

        label = "STRONG"

    elif p >= 0.60:

        label = "MODERATE"

    elif p >= 0.40:

        label = "UNCERTAIN"

    elif p >= 0.25:

        label = "MODERATE BEARISH"

    elif p >= 0.10:

        label = "STRONG BEARISH"

    else:

        label = "VERY HIGH BEARISH"

    return {
        "probability": p,
        "percentage": p * 100.0,
        "label": label,
        "is_high_confidence": (
            p >= 0.90
        ),
    }


# =====================================================================
# EXPORTS
# =====================================================================


__all__ = [
    "brier_score",
    "calibration_table",
    "high_confidence_analysis",
    "reliability_report",
    "get_model_reliability",
    "classify_probability",
]
