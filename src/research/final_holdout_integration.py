"""
Final holdout integration for the AI Swing Analyser research pipeline.

The final holdout is an untouched dataset reserved for the final,
one-time out-of-sample evaluation of a selected candidate.

This module deliberately does NOT:
- fit models,
- tune hyperparameters,
- optimize thresholds,
- select features,
- modify the candidate,
- or make production approval decisions.

It only evaluates an already-selected candidate against the final holdout
and records the evidence needed by the later approval stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_HOLDOUT_HORIZONS = (
    1,
    3,
    5,
    10,
    20,
)


@dataclass(frozen=True)
class FinalHoldoutIntegrationConfig:
    """Configuration for final holdout evaluation."""

    horizons: tuple[int, ...] = DEFAULT_HOLDOUT_HORIZONS

    minimum_accuracy: float = 0.95
    minimum_coverage: float = 0.80
    minimum_direction_samples: int = 30

    require_untouched_holdout: bool = True

    def __post_init__(self) -> None:
        horizons = tuple(int(x) for x in self.horizons)

        if not horizons:
            raise ValueError(
                "At least one holdout horizon is required."
            )

        if any(x <= 0 for x in horizons):
            raise ValueError(
                "Holdout horizons must be positive."
            )

        if len(set(horizons)) != len(horizons):
            raise ValueError(
                "Holdout horizons must be unique."
            )

        if not 0.0 <= self.minimum_accuracy <= 1.0:
            raise ValueError(
                "minimum_accuracy must be between 0 and 1."
            )

        if not 0.0 <= self.minimum_coverage <= 1.0:
            raise ValueError(
                "minimum_coverage must be between 0 and 1."
            )

        if self.minimum_direction_samples < 1:
            raise ValueError(
                "minimum_direction_samples must be positive."
            )

        object.__setattr__(
            self,
            "horizons",
            horizons,
        )


@dataclass
class FinalHoldoutHorizonResult:
    """Final holdout evidence for one forecast horizon."""

    horizon: int

    evaluated: bool = False
    passed: bool = False

    sample_count: int = 0
    correct_direction_count: int = 0

    accuracy: float | None = None
    coverage: float | None = None

    warnings: list[str] = field(
        default_factory=list
    )
    errors: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class FinalHoldoutIntegrationResult:
    """Complete final holdout evaluation result."""

    horizons: tuple[int, ...]

    results: dict[int, FinalHoldoutHorizonResult]

    evaluated_horizons: tuple[int, ...] = ()
    passed_horizons: tuple[int, ...] = ()
    failed_horizons: tuple[int, ...] = ()

    candidate_passed: bool = False
    production_ready: bool = False

    final_holdout_used: bool = True

    warnings: list[str] = field(
        default_factory=list
    )
    errors: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def get(
        self,
        horizon: int,
    ) -> FinalHoldoutHorizonResult:
        """Return the result for a horizon."""
        if horizon not in self.results:
            raise KeyError(
                f"No holdout result for horizon={horizon}."
            )

        return self.results[horizon]


class FinalHoldoutIntegration:
    """
    Integrate final holdout evaluation into the research pipeline.

    The holdout must already contain predictions generated without using
    the holdout observations for training, feature selection, threshold
    optimization, calibration, or model selection.
    """

    def __init__(
        self,
        config: FinalHoldoutIntegrationConfig | None = None,
    ) -> None:
        self.config = (
            config
            or FinalHoldoutIntegrationConfig()
        )

    @staticmethod
    def _as_series(
        values: Any,
        name: str,
    ) -> pd.Series:
        if isinstance(values, pd.Series):
            series = values.copy(deep=True)
        else:
            series = pd.Series(values)

        series.name = name

        if series.empty:
            raise ValueError(
                f"{name} cannot be empty."
            )

        if not pd.api.types.is_numeric_dtype(
            series
        ):
            raise TypeError(
                f"{name} must be numeric."
            )

        if not np.isfinite(
            series.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"{name} contains NaN or infinite values."
            )

        return series

    @staticmethod
    def _extract(
        source: Any,
        names: Sequence[str],
    ) -> Any:
        if source is None:
            return None

        if isinstance(source, Mapping):
            for name in names:
                if name in source:
                    return source[name]

        if isinstance(source, pd.DataFrame):
            for name in names:
                if name in source.columns:
                    return source[name]

        for name in names:
            if hasattr(source, name):
                return getattr(
                    source,
                    name,
                )

        return None

    @classmethod
    def _validate_holdout_provenance(
        cls,
        holdout_data: Any,
    ) -> None:
        """
        Reject holdout data when provenance explicitly indicates that
        it was used for model development.
        """

        forbidden_flags = (
            "used_for_training",
            "used_for_tuning",
            "used_for_feature_selection",
            "used_for_threshold_optimization",
            "used_for_calibration",
            "used_for_model_selection",
            "production_approved",
        )

        for flag in forbidden_flags:
            value = cls._extract(
                holdout_data,
                (flag,),
            )

            if value is True:
                raise ValueError(
                    "Final holdout provenance violation: "
                    f"{flag}=True."
                )

    @staticmethod
    def _calculate_direction_accuracy(
        actual: pd.Series,
        predicted: pd.Series,
    ) -> tuple[float, int]:
        if len(actual) != len(predicted):
            raise ValueError(
                "Actual and predicted direction "
                "series must have equal length."
            )

        actual_values = (
            actual.to_numpy()
            .astype(int)
        )

        predicted_values = (
            predicted.to_numpy()
            .astype(int)
        )

        correct = int(
            np.sum(
                actual_values
                == predicted_values
            )
        )

        accuracy = (
            correct / len(actual_values)
        )

        return (
            float(accuracy),
            correct,
        )

    @staticmethod
    def _calculate_coverage(
        predicted_probability: pd.Series,
    ) -> float:
        """
        Coverage means the fraction of predictions that contain a finite
        probability and are therefore eligible for evaluation.
        """

        values = (
            predicted_probability
            .to_numpy(dtype=float)
        )

        if len(values) == 0:
            return 0.0

        valid = np.isfinite(values)

        return float(
            np.mean(valid)
        )

    def _evaluate_horizon(
        self,
        horizon: int,
        actual_direction: Any,
        predicted_direction: Any,
        predicted_probability: Any = None,
    ) -> FinalHoldoutHorizonResult:
        result = FinalHoldoutHorizonResult(
            horizon=horizon,
            evaluated=False,
            final_holdout_used=True
            if False
            else True,
        )

        try:
            actual = self._as_series(
                actual_direction,
                "actual_direction",
            )

            predicted = self._as_series(
                predicted_direction,
                "predicted_direction",
            )

            if len(actual) != len(predicted):
                raise ValueError(
                    "Actual and predicted direction "
                    "lengths do not match."
                )

            if not set(
                actual.unique()
            ).issubset({0, 1}):
                raise ValueError(
                    "actual_direction must contain only 0/1."
                )

            if not set(
                predicted.unique()
            ).issubset({0, 1}):
                raise ValueError(
                    "predicted_direction must contain only 0/1."
                )

            probability = None

            if predicted_probability is not None:
                probability = self._as_series(
                    predicted_probability,
                    "predicted_probability",
                )

                if len(probability) != len(
                    actual
                ):
                    raise ValueError(
                        "Probability and direction "
                        "lengths do not match."
                    )

                coverage = (
                    self._calculate_coverage(
                        probability
                    )
                )
            else:
                coverage = 1.0

            accuracy, correct = (
                self._calculate_direction_accuracy(
                    actual,
                    predicted,
                )
            )

            sample_count = len(actual)

            result.sample_count = (
                sample_count
            )

            result.correct_direction_count = (
                correct
            )

            result.accuracy = accuracy
            result.coverage = coverage
            result.evaluated = True

            if (
                sample_count
                < self.config.minimum_direction_samples
            ):
                result.warnings.append(
                    "Insufficient final holdout "
                    "samples for the configured "
                    "minimum sample requirement."
                )

            result.passed = bool(
                sample_count
                >= self.config.minimum_direction_samples
                and accuracy
                >= self.config.minimum_accuracy
                and coverage
                >= self.config.minimum_coverage
            )

            result.metadata.update(
                {
                    "final_holdout_used": True,
                    "model_fitted_here": False,
                    "threshold_optimized_here": False,
                    "feature_selection_here": False,
                    "calibration_here": False,
                    "research_only": True,
                    "production_approved": False,
                }
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            result.errors.append(
                str(exc)
            )
            result.passed = False
            result.evaluated = False

        return result

    def run(
        self,
        holdout_data: Any,
        actual_directions: Mapping[int, Any],
        predicted_directions: Mapping[int, Any],
        predicted_probabilities: Mapping[int, Any]
        | None = None,
    ) -> FinalHoldoutIntegrationResult:
        """
        Evaluate final holdout predictions.

        No fitting or optimization is performed.
        """

        self._validate_holdout_provenance(
            holdout_data
        )

        results: dict[
            int,
            FinalHoldoutHorizonResult,
        ] = {}

        warnings: list[str] = []
        errors: list[str] = []

        evaluated = []
        passed = []
        failed = []

        for horizon in self.config.horizons:
            result = self._evaluate_horizon(
                horizon=horizon,
                actual_direction=actual_directions.get(
                    horizon
                ),
                predicted_direction=predicted_directions.get(
                    horizon
                ),
                predicted_probability=(
                    predicted_probabilities.get(
                        horizon
                    )
                    if predicted_probabilities
                    is not None
                    else None
                ),
            )

            results[horizon] = result

            warnings.extend(
                [
                    f"{horizon}D: {warning}"
                    for warning in result.warnings
                ]
            )

            errors.extend(
                [
                    f"{horizon}D: {error}"
                    for error in result.errors
                ]
            )

            if result.evaluated:
                evaluated.append(
                    horizon
                )

            if result.passed:
                passed.append(
                    horizon
                )
            else:
                failed.append(
                    horizon
                )

        candidate_passed = bool(
            evaluated
            and not errors
            and len(passed)
            == len(evaluated)
            and len(failed) == 0
        )

        return FinalHoldoutIntegrationResult(
            horizons=self.config.horizons,
            results=results,
            evaluated_horizons=tuple(
                evaluated
            ),
            passed_horizons=tuple(
                passed
            ),
            failed_horizons=tuple(
                failed
            ),
            candidate_passed=candidate_passed,
            production_ready=False,
            final_holdout_used=True,
            warnings=warnings,
            errors=errors,
            metadata={
                "final_holdout_used": True,
                "model_fitted_here": False,
                "threshold_optimized_here": False,
                "feature_selection_here": False,
                "calibration_here": False,
                "research_only": True,
                "production_ready": False,
                "production_approved": False,
            },
        )


def integrate_final_holdout(
    holdout_data: Any,
    actual_directions: Mapping[int, Any],
    predicted_directions: Mapping[int, Any],
    predicted_probabilities: Mapping[int, Any]
    | None = None,
    config: FinalHoldoutIntegrationConfig | None = None,
) -> FinalHoldoutIntegrationResult:
    """Convenience wrapper for final holdout evaluation."""

    pipeline = FinalHoldoutIntegration(
        config=config
    )

    return pipeline.run(
        holdout_data=holdout_data,
        actual_directions=actual_directions,
        predicted_directions=predicted_directions,
        predicted_probabilities=predicted_probabilities,
    )


def final_holdout_summary(
    result: FinalHoldoutIntegrationResult,
) -> dict[str, Any]:
    """Convert final holdout evidence into a dashboard-safe summary."""

    if not isinstance(
        result,
        FinalHoldoutIntegrationResult,
    ):
        raise TypeError(
            "result must be a "
            "FinalHoldoutIntegrationResult."
        )

    horizon_results = {}

    for horizon, item in result.results.items():
        horizon_results[str(horizon)] = {
            "evaluated": item.evaluated,
            "passed": item.passed,
            "sample_count": item.sample_count,
            "correct_direction_count": (
                item.correct_direction_count
            ),
            "accuracy": item.accuracy,
            "coverage": item.coverage,
            "warnings": list(
                item.warnings
            ),
            "errors": list(
                item.errors
            ),
        }

    return {
        "horizons": list(
            result.horizons
        ),
        "evaluated_horizons": list(
            result.evaluated_horizons
        ),
        "passed_horizons": list(
            result.passed_horizons
        ),
        "failed_horizons": list(
            result.failed_horizons
        ),
        "candidate_passed": (
            result.candidate_passed
        ),
        "production_ready": (
            result.production_ready
        ),
        "final_holdout_used": (
            result.final_holdout_used
        ),
        "research_only": True,
        "production_approved": False,
        "horizon_results": horizon_results,
    }
