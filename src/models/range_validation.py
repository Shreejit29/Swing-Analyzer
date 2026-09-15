"""
Prediction-range validation for AI Swing Analyser.

Validates predicted future-price/return intervals.

The objective is to determine whether predicted ranges are:

    - Calibrated
    - Wide enough to contain actual outcomes
    - Not unnecessarily wide
    - Stable across time
    - Symmetric/asymmetric in a sensible way

IMPORTANT:

Range validation must be performed on genuinely unseen data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class RangeValidationConfig:
    """
    Configuration for prediction-range validation.
    """

    expected_coverage: float = 0.80

    minimum_coverage: float = 0.70

    maximum_coverage: float = 0.95

    maximum_average_width: Optional[
        float
    ] = None

    minimum_samples: int = 50


@dataclass
class RangeValidationResult:
    """Complete range-validation result."""

    samples: int

    coverage: float

    average_width: float

    median_width: float

    lower_violation_rate: float

    upper_violation_rate: float

    average_lower_distance: float

    average_upper_distance: float

    calibration_error: float

    passed_coverage: bool

    passed_width: bool

    passed_samples: bool

    passed: bool


@dataclass
class RangeCalibrationBin:
    """Range calibration result for one confidence level."""

    nominal_coverage: float

    actual_coverage: float

    calibration_error: float

    samples: int


def _validate_arrays(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Validate range arrays."""

    actual = np.asarray(
        actual,
        dtype=float,
    ).reshape(-1)

    lower = np.asarray(
        lower,
        dtype=float,
    ).reshape(-1)

    upper = np.asarray(
        upper,
        dtype=float,
    ).reshape(-1)

    if not (
        len(actual)
        == len(lower)
        == len(upper)
    ):
        raise ValueError(
            "Actual, lower and upper arrays must have equal length."
        )

    if len(actual) == 0:
        raise ValueError(
            "Range arrays cannot be empty."
        )

    if not (
        np.isfinite(actual).all()
        and np.isfinite(lower).all()
        and np.isfinite(upper).all()
    ):
        raise ValueError(
            "Range arrays contain non-finite values."
        )

    if np.any(
        lower > upper
    ):
        raise ValueError(
            "Lower range bound cannot exceed upper range bound."
        )

    return (
        actual,
        lower,
        upper,
    )


def interval_coverage(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """
    Calculate fraction of actual observations inside predicted range.
    """

    actual, lower, upper = (
        _validate_arrays(
            actual,
            lower,
            upper,
        )
    )

    inside = (
        (actual >= lower)
        & (actual <= upper)
    )

    return float(
        np.mean(inside)
    )


def interval_width(
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Calculate prediction interval width."""

    lower = np.asarray(
        lower,
        dtype=float,
    )

    upper = np.asarray(
        upper,
        dtype=float,
    )

    if lower.shape != upper.shape:
        raise ValueError(
            "lower and upper must have identical shapes."
        )

    width = (
        upper
        - lower
    )

    if np.any(
        width < 0
    ):
        raise ValueError(
            "Interval width cannot be negative."
        )

    return width


def range_calibration(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal_coverage: float,
) -> RangeCalibrationBin:
    """
    Evaluate calibration of one prediction interval.

    Example:

        nominal coverage = 0.80

    means approximately 80% of future observations should
    fall inside the interval if the model is well calibrated.
    """

    if not 0.0 < nominal_coverage < 1.0:
        raise ValueError(
            "nominal_coverage must be between 0 and 1."
        )

    actual, lower, upper = (
        _validate_arrays(
            actual,
            lower,
            upper,
        )
    )

    coverage = interval_coverage(
        actual,
        lower,
        upper,
    )

    return RangeCalibrationBin(
        nominal_coverage=(
            nominal_coverage
        ),
        actual_coverage=coverage,
        calibration_error=abs(
            coverage
            - nominal_coverage
        ),
        samples=len(actual),
    )


def evaluate_range(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    config: Optional[
        RangeValidationConfig
    ] = None,
) -> RangeValidationResult:
    """
    Evaluate a predicted return/price range.

    The default objective is approximately 80% coverage.

    Coverage that is too low means the range is unreliable.

    Coverage that is extremely high can also be suspicious if the
    model simply produces excessively wide ranges.
    """

    config = (
        config
        if config is not None
        else RangeValidationConfig()
    )

    actual, lower, upper = (
        _validate_arrays(
            actual,
            lower,
            upper,
        )
    )

    samples = len(actual)

    coverage = interval_coverage(
        actual,
        lower,
        upper,
    )

    widths = interval_width(
        lower,
        upper,
    )

    inside = (
        (actual >= lower)
        & (actual <= upper)
    )

    lower_violation = (
        actual < lower
    )

    upper_violation = (
        actual > upper
    )

    lower_distance = np.maximum(
        lower - actual,
        0.0,
    )

    upper_distance = np.maximum(
        actual - upper,
        0.0,
    )

    calibration_error = abs(
        coverage
        - config.expected_coverage
    )

    if config.maximum_average_width is None:

        passed_width = True

    else:

        passed_width = (
            float(
                np.mean(widths)
            )
            <= config.maximum_average_width
        )

    passed_coverage = (
        config.minimum_coverage
        <= coverage
        <= config.maximum_coverage
    )

    passed_samples = (
        samples
        >= config.minimum_samples
    )

    passed = (
        passed_coverage
        and passed_width
        and passed_samples
    )

    return RangeValidationResult(
        samples=samples,
        coverage=coverage,
        average_width=float(
            np.mean(widths)
        ),
        median_width=float(
            np.median(widths)
        ),
        lower_violation_rate=float(
            np.mean(
                lower_violation
            )
        ),
        upper_violation_rate=float(
            np.mean(
                upper_violation
            )
        ),
        average_lower_distance=float(
            np.mean(
                lower_distance
            )
        ),
        average_upper_distance=float(
            np.mean(
                upper_distance
            )
        ),
        calibration_error=float(
            calibration_error
        ),
        passed_coverage=bool(
            passed_coverage
        ),
        passed_width=bool(
            passed_width
        ),
        passed_samples=bool(
            passed_samples
        ),
        passed=bool(
            passed
        ),
    )


def rolling_range_validation(
    actual: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    window: int = 100,
    minimum_samples: int = 50,
) -> pd.DataFrame:
    """
    Calculate rolling range coverage.

    Useful for detecting periods where range predictions
    suddenly become unreliable.
    """

    frame = pd.concat(
        [
            actual.rename(
                "actual"
            ),
            lower.rename(
                "lower"
            ),
            upper.rename(
                "upper"
            ),
        ],
        axis=1,
    ).dropna()

    if frame.empty:
        raise ValueError(
            "No overlapping range observations."
        )

    rows = []

    for end in range(
        window,
        len(frame) + 1,
    ):

        subset = frame.iloc[
            end - window:end
        ]

        if len(subset) < (
            minimum_samples
        ):
            continue

        coverage = interval_coverage(
            subset["actual"].to_numpy(),
            subset["lower"].to_numpy(),
            subset["upper"].to_numpy(),
        )

        width = (
            subset["upper"]
            - subset["lower"]
        )

        rows.append(
            {
                "Timestamp": subset.index[-1],
                "Coverage": coverage,
                "Average Width": float(
                    width.mean()
                ),
                "Samples": len(subset),
            }
        )

    return pd.DataFrame(
        rows
    ).set_index(
        "Timestamp"
    )


def asymmetric_range_metrics(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict:
    """
    Analyse whether lower and upper errors are balanced.

    Useful for identifying systematic underestimation of downside
    or upside.
    """

    actual, lower, upper = (
        _validate_arrays(
            actual,
            lower,
            upper,
        )
    )

    lower_violation = (
        actual < lower
    )

    upper_violation = (
        actual > upper
    )

    lower_error = np.maximum(
        lower - actual,
        0.0,
    )

    upper_error = np.maximum(
        actual - upper,
        0.0,
    )

    return {
        "lower_violation_rate": float(
            np.mean(
                lower_violation
            )
        ),
        "upper_violation_rate": float(
            np.mean(
                upper_violation
            )
        ),
        "lower_average_error": float(
            np.mean(
                lower_error
            )
        ),
        "upper_average_error": float(
            np.mean(
                upper_error
            )
        ),
        "violation_rate_difference": float(
            abs(
                np.mean(
                    lower_violation
                )
                - np.mean(
                    upper_violation
                )
            )
        ),
    }


def compare_range_models(
    actual: np.ndarray,
    model_ranges: dict[
        str,
        tuple[
            np.ndarray,
            np.ndarray,
        ],
    ],
    expected_coverage: float = 0.80,
) -> pd.DataFrame:
    """
    Compare several range models.

    Each model must provide:

        (lower_predictions, upper_predictions)
    """

    rows = []

    for name, (
        lower,
        upper,
    ) in model_ranges.items():

        result = range_calibration(
            actual=actual,
            lower=lower,
            upper=upper,
            nominal_coverage=(
                expected_coverage
            ),
        )

        widths = interval_width(
            lower,
            upper,
        )

        rows.append(
            {
                "Model": name,
                "Nominal Coverage": (
                    expected_coverage
                ),
                "Actual Coverage": (
                    result.actual_coverage
                ),
                "Calibration Error": (
                    result.calibration_error
                ),
                "Average Width": float(
                    np.mean(widths)
                ),
                "Median Width": float(
                    np.median(widths)
                ),
                "Samples": result.samples,
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "Calibration Error",
            "Average Width",
        ]
    ).reset_index(
        drop=True
    )


def range_validation_summary(
    result: RangeValidationResult,
) -> pd.DataFrame:
    """Convert range result into dashboard table."""

    return pd.DataFrame(
        [
            {
                "Metric": "Samples",
                "Value": result.samples,
            },
            {
                "Metric": "Coverage",
                "Value": result.coverage,
            },
            {
                "Metric": "Average Width",
                "Value": result.average_width,
            },
            {
                "Metric": "Median Width",
                "Value": result.median_width,
            },
            {
                "Metric": "Lower Violation Rate",
                "Value": result.lower_violation_rate,
            },
            {
                "Metric": "Upper Violation Rate",
                "Value": result.upper_violation_rate,
            },
            {
                "Metric": "Calibration Error",
                "Value": result.calibration_error,
            },
            {
                "Metric": "Coverage Gate",
                "Value": result.passed_coverage,
            },
            {
                "Metric": "Width Gate",
                "Value": result.passed_width,
            },
            {
                "Metric": "Sample Gate",
                "Value": result.passed_samples,
            },
            {
                "Metric": "Overall Range Gate",
                "Value": result.passed,
            },
        ]
    )
