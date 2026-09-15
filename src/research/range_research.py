"""
AI Swing Analyser — Target Price Range Research.

Researches probabilistic future-price ranges rather than relying on
a single point prediction.

For each prediction horizon the system can estimate:

    Lower bound
    Median / expected path
    Upper bound

The range is evaluated for:

    - interval coverage
    - interval width
    - lower/upper miss rates
    - calibration
    - temporal integrity

The final holdout must remain untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.models.range_validation import (
    RangeValidationConfig,
    RangeValidationResult,
    evaluate_range,
)


# ---------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------


@dataclass
class RangeResearchResult:
    """
    Result of probabilistic target-range research.
    """

    validation: RangeValidationResult

    horizon: int

    samples: int

    expected_coverage: float
    minimum_coverage: float
    maximum_coverage: float

    actual_coverage: float
    average_width: float
    median_width: float

    lower_miss_rate: float
    upper_miss_rate: float

    passed_coverage_gate: bool
    passed_width_gate: bool

    final_holdout_used: bool

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    warnings: list[str] = field(
        default_factory=list
    )

    @property
    def passed(self) -> bool:
        return (
            self.passed_coverage_gate
            and self.passed_width_gate
        )

    def summary(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "samples": self.samples,
            "expected_coverage": (
                self.expected_coverage
            ),
            "minimum_coverage": (
                self.minimum_coverage
            ),
            "maximum_coverage": (
                self.maximum_coverage
            ),
            "actual_coverage": (
                self.actual_coverage
            ),
            "average_width": (
                self.average_width
            ),
            "median_width": (
                self.median_width
            ),
            "lower_miss_rate": (
                self.lower_miss_rate
            ),
            "upper_miss_rate": (
                self.upper_miss_rate
            ),
            "passed_coverage_gate": (
                self.passed_coverage_gate
            ),
            "passed_width_gate": (
                self.passed_width_gate
            ),
            "passed": self.passed,
            "final_holdout_used": (
                self.final_holdout_used
            ),
            "metadata": dict(
                self.metadata
            ),
            "warnings": list(
                self.warnings
            ),
        }


# ---------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------


class RangeResearchEngine:
    """
    Research evaluator for target-price prediction ranges.

    This engine evaluates already-generated out-of-sample ranges.

    It never trains a range model.
    """

    def __init__(
        self,
        config: RangeValidationConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else RangeValidationConfig()
        )

        self._validate_config()

    # -----------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------

    def _validate_config(self) -> None:

        expected = getattr(
            self.config,
            "expected_coverage",
            0.80,
        )

        minimum = getattr(
            self.config,
            "minimum_coverage",
            0.70,
        )

        maximum = getattr(
            self.config,
            "maximum_coverage",
            0.95,
        )

        minimum_samples = getattr(
            self.config,
            "minimum_samples",
            50,
        )

        if not (
            0.0
            < minimum
            <= expected
            <= maximum
            < 1.0
        ):
            raise ValueError(
                "Range coverage thresholds must satisfy "
                "0 < minimum <= expected <= maximum < 1."
            )

        if minimum_samples < 1:
            raise ValueError(
                "minimum_samples must be positive."
            )

    # -----------------------------------------------------------------
    # Input validation
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        actual: np.ndarray,
        lower: np.ndarray,
        median: np.ndarray,
        upper: np.ndarray,
        timestamps: pd.DatetimeIndex,
    ) -> None:

        actual = np.asarray(
            actual,
            dtype=float,
        )

        lower = np.asarray(
            lower,
            dtype=float,
        )

        median = np.asarray(
            median,
            dtype=float,
        )

        upper = np.asarray(
            upper,
            dtype=float,
        )

        lengths = {
            len(actual),
            len(lower),
            len(median),
            len(upper),
            len(timestamps),
        }

        if len(lengths) != 1:
            raise ValueError(
                "Actual values, range predictions, and "
                "timestamps must have equal lengths."
            )

        if len(actual) == 0:
            raise ValueError(
                "Range evaluation dataset is empty."
            )

        for name, values in (
            ("actual", actual),
            ("lower", lower),
            ("median", median),
            ("upper", upper),
        ):
            if not np.isfinite(
                values
            ).all():
                raise ValueError(
                    f"{name} contains non-finite values."
                )

        if not (
            np.all(
                lower <= median
            )
            and np.all(
                median <= upper
            )
        ):
            raise ValueError(
                "Range predictions must satisfy "
                "lower <= median <= upper."
            )

        if not isinstance(
            timestamps,
            pd.DatetimeIndex,
        ):
            raise TypeError(
                "timestamps must be a DatetimeIndex."
            )

        if timestamps.has_duplicates:
            raise ValueError(
                "timestamps contain duplicates."
            )

        if not timestamps.is_monotonic_increasing:
            raise ValueError(
                "timestamps must be chronological."
            )

    # -----------------------------------------------------------------
    # Main evaluation
    # -----------------------------------------------------------------

    def evaluate(
        self,
        actual: np.ndarray,
        lower: np.ndarray,
        median: np.ndarray,
        upper: np.ndarray,
        timestamps: pd.DatetimeIndex,
        horizon: int,
        final_holdout_used: bool = False,
    ) -> RangeResearchResult:

        if horizon <= 0:
            raise ValueError(
                "horizon must be positive."
            )

        self._validate_inputs(
            actual=actual,
            lower=lower,
            median=median,
            upper=upper,
            timestamps=timestamps,
        )

        if final_holdout_used:
            raise ValueError(
                "Final holdout data must not be used "
                "for range-model research."
            )

        minimum_samples = getattr(
            self.config,
            "minimum_samples",
            50,
        )

        if len(actual) < minimum_samples:
            raise ValueError(
                "Range evaluation dataset contains fewer "
                f"than {minimum_samples} observations."
            )

        validation = evaluate_range(
            actual=np.asarray(
                actual,
                dtype=float,
            ),
            lower=np.asarray(
                lower,
                dtype=float,
            ),
            median=np.asarray(
                median,
                dtype=float,
            ),
            upper=np.asarray(
                upper,
                dtype=float,
            ),
            expected_coverage=getattr(
                self.config,
                "expected_coverage",
                0.80,
            ),
        )

        actual_array = np.asarray(
            actual,
            dtype=float,
        )

        lower_array = np.asarray(
            lower,
            dtype=float,
        )

        median_array = np.asarray(
            median,
            dtype=float,
        )

        upper_array = np.asarray(
            upper,
            dtype=float,
        )

        inside = (
            (
                actual_array
                >= lower_array
            )
            & (
                actual_array
                <= upper_array
            )
        )

        actual_coverage = float(
            inside.mean()
        )

        widths = (
            upper_array
            - lower_array
        )

        average_width = float(
            np.mean(widths)
        )

        median_width = float(
            np.median(widths)
        )

        lower_miss_rate = float(
            (
                actual_array
                < lower_array
            ).mean()
        )

        upper_miss_rate = float(
            (
                actual_array
                > upper_array
            ).mean()
        )

        minimum_coverage = getattr(
            self.config,
            "minimum_coverage",
            0.70,
        )

        maximum_coverage = getattr(
            self.config,
            "maximum_coverage",
            0.95,
        )

        expected_coverage = getattr(
            self.config,
            "expected_coverage",
            0.80,
        )

        passed_coverage = (
            minimum_coverage
            <= actual_coverage
            <= maximum_coverage
        )

        # If an explicit maximum width is configured, enforce it.
        maximum_width = getattr(
            self.config,
            "maximum_average_width",
            None,
        )

        if maximum_width is None:
            passed_width = True
        else:
            passed_width = (
                average_width
                <= maximum_width
            )

        warnings: list[str] = []

        if not passed_coverage:
            warnings.append(
                "Prediction interval coverage is outside "
                "the configured acceptable range."
            )

        if (
            actual_coverage
            < expected_coverage
        ):
            warnings.append(
                "Prediction intervals are under-covering."
            )

        if (
            maximum_width is not None
            and not passed_width
        ):
            warnings.append(
                "Prediction interval width exceeds "
                "the configured maximum."
            )

        return RangeResearchResult(
            validation=validation,
            horizon=horizon,
            samples=len(
                actual_array
            ),
            expected_coverage=(
                expected_coverage
            ),
            minimum_coverage=(
                minimum_coverage
            ),
            maximum_coverage=(
                maximum_coverage
            ),
            actual_coverage=(
                actual_coverage
            ),
            average_width=(
                average_width
            ),
            median_width=(
                median_width
            ),
            lower_miss_rate=(
                lower_miss_rate
            ),
            upper_miss_rate=(
                upper_miss_rate
            ),
            passed_coverage_gate=(
                passed_coverage
            ),
            passed_width_gate=(
                passed_width
            ),
            final_holdout_used=False,
            metadata={
                "evaluation_only": True,
                "model_fitted": False,
                "final_holdout_used": False,
                "range_order_validated": True,
                "research_only": True,
                "median_prediction_evaluated": True,
            },
            warnings=warnings,
        )

    # -----------------------------------------------------------------
    # Range from returns to prices
    # -----------------------------------------------------------------

    @staticmethod
    def returns_to_price_range(
        current_price: np.ndarray | float,
        lower_return: np.ndarray | float,
        median_return: np.ndarray | float,
        upper_return: np.ndarray | float,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:

        current = np.asarray(
            current_price,
            dtype=float,
        )

        lower = np.asarray(
            lower_return,
            dtype=float,
        )

        median = np.asarray(
            median_return,
            dtype=float,
        )

        upper = np.asarray(
            upper_return,
            dtype=float,
        )

        try:
            current, lower, median, upper = (
                np.broadcast_arrays(
                    current,
                    lower,
                    median,
                    upper,
                )
            )
        except ValueError as exc:
            raise ValueError(
                "Price and return arrays cannot be broadcast."
            ) from exc

        if not np.isfinite(
            current
        ).all():
            raise ValueError(
                "Current price contains non-finite values."
            )

        if (
            current <= 0
        ).any():
            raise ValueError(
                "Current price must be positive."
            )

        if not np.isfinite(
            lower
        ).all() or not np.isfinite(
            median
        ).all() or not np.isfinite(
            upper
        ).all():
            raise ValueError(
                "Return predictions contain non-finite values."
            )

        lower_price = (
            current
            * (
                1.0
                + lower
            )
        )

        median_price = (
            current
            * (
                1.0
                + median
            )
        )

        upper_price = (
            current
            * (
                1.0
                + upper
            )
        )

        return (
            lower_price,
            median_price,
            upper_price,
        )

    # -----------------------------------------------------------------
    # Range quality diagnostics
    # -----------------------------------------------------------------

    @staticmethod
    def range_width_percentage(
        lower_price: np.ndarray | float,
        upper_price: np.ndarray | float,
        current_price: np.ndarray | float,
    ) -> np.ndarray:

        lower = np.asarray(
            lower_price,
            dtype=float,
        )

        upper = np.asarray(
            upper_price,
            dtype=float,
        )

        current = np.asarray(
            current_price,
            dtype=float,
        )

        if not np.isfinite(
            lower
        ).all():
            raise ValueError(
                "lower_price contains non-finite values."
            )

        if not np.isfinite(
            upper
        ).all():
            raise ValueError(
                "upper_price contains non-finite values."
            )

        if not np.isfinite(
            current
        ).all():
            raise ValueError(
                "current_price contains non-finite values."
            )

        if (
            current <= 0
        ).any():
            raise ValueError(
                "current_price must be positive."
            )

        if (
            lower > upper
        ).any():
            raise ValueError(
                "lower_price cannot exceed upper_price."
            )

        return (
            (
                upper
                - lower
            )
            / current
        )

    # -----------------------------------------------------------------
    # Summary helpers
    # -----------------------------------------------------------------

    @staticmethod
    def compare_ranges(
        first: RangeResearchResult,
        second: RangeResearchResult,
    ) -> dict[str, float]:

        if first is None or second is None:
            raise ValueError(
                "Both range results are required."
            )

        return {
            "coverage_difference": (
                second.actual_coverage
                - first.actual_coverage
            ),
            "average_width_difference": (
                second.average_width
                - first.average_width
            ),
            "stability_difference": (
                (
                    float(
                        second.passed
                    )
                )
                - float(
                    first.passed
                )
            ),
        }


# ---------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------


def run_range_research(
    actual: np.ndarray,
    lower: np.ndarray,
    median: np.ndarray,
    upper: np.ndarray,
    timestamps: pd.DatetimeIndex,
    horizon: int,
    config: RangeValidationConfig | None = None,
) -> RangeResearchResult:

    engine = RangeResearchEngine(
        config=config
    )

    return engine.evaluate(
        actual=actual,
        lower=lower,
        median=median,
        upper=upper,
        timestamps=timestamps,
        horizon=horizon,
        final_holdout_used=False,
    )


__all__ = [
    "RangeResearchResult",
    "RangeResearchEngine",
    "run_range_research",
]
