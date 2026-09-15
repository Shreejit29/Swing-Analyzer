"""
AI Swing Analyser — Range Prediction Integration.

Connects selected multi-horizon research models with probabilistic
target-price range evaluation.

Research boundary
-----------------
This module does NOT:

- use the final holdout for fitting
- perform production approval
- optimize trading thresholds
- create live trading signals
- perform probability calibration
- guarantee a particular prediction accuracy

It prepares and evaluates target-price ranges that can later be
combined with direction, market regime, calibration, backtesting,
robustness, and final-holdout evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.research.range_research import (
    RangeResearchEngine,
    RangeResearchResult,
)
from src.multi_horizon_selection_pipeline import (
    MultiHorizonSelectionResult,
)


DEFAULT_RANGE_HORIZONS = (
    1,
    3,
    5,
    10,
    20,
)


@dataclass(frozen=True)
class RangePredictionIntegrationConfig:
    """Configuration for multi-horizon range integration."""

    horizons: tuple[int, ...] = (
        DEFAULT_RANGE_HORIZONS
    )

    lower_quantile: float = 0.10

    median_quantile: float = 0.50

    upper_quantile: float = 0.90

    expected_coverage: float = 0.80

    minimum_coverage: float = 0.70

    maximum_coverage: float = 0.95

    minimum_observations: int = 50

    require_selection_completed: bool = True

    def __post_init__(self) -> None:
        normalized = tuple(
            int(horizon)
            for horizon in self.horizons
        )

        if not normalized:
            raise ValueError(
                "At least one range horizon is required."
            )

        if any(
            horizon <= 0
            for horizon in normalized
        ):
            raise ValueError(
                "Range horizons must be positive."
            )

        if len(normalized) != len(
            set(normalized)
        ):
            raise ValueError(
                "Range horizons must be unique."
            )

        if not (
            0.0
            < self.lower_quantile
            < self.median_quantile
            < self.upper_quantile
            < 1.0
        ):
            raise ValueError(
                "Range quantiles must satisfy "
                "lower < median < upper."
            )

        if not (
            0.0
            < self.minimum_coverage
            <= self.expected_coverage
            <= self.maximum_coverage
            <= 1.0
        ):
            raise ValueError(
                "Invalid range coverage limits."
            )

        if (
            self.minimum_observations
            < 1
        ):
            raise ValueError(
                "minimum_observations must be positive."
            )


@dataclass
class HorizonRangePrediction:
    """Range prediction/evaluation for one horizon."""

    horizon: int

    lower_return: pd.Series | None

    median_return: pd.Series | None

    upper_return: pd.Series | None

    lower_price: pd.Series | None

    median_price: pd.Series | None

    upper_price: pd.Series | None

    coverage: float | None

    average_width: float | None

    selected_model_id: str | None

    evaluated: bool

    passed: bool

    warnings: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, object] = field(
        default_factory=dict
    )


@dataclass
class MultiHorizonRangePredictionResult:
    """Aggregated range results across horizons."""

    horizons: tuple[int, ...]

    results: dict[
        int,
        HorizonRangePrediction,
    ]

    evaluated_horizons: tuple[int, ...]

    passed_horizons: tuple[int, ...]

    failed_horizons: tuple[int, ...]

    candidate_passed: bool

    production_ready: bool

    warnings: list[str] = field(
        default_factory=list
    )

    errors: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, object] = field(
        default_factory=dict
    )

    @property
    def evaluated_count(self) -> int:
        return len(
            self.evaluated_horizons
        )

    @property
    def passed_count(self) -> int:
        return len(
            self.passed_horizons
        )

    @property
    def final_holdout_used(self) -> bool:
        return bool(
            self.metadata.get(
                "final_holdout_used",
                False,
            )
        )

    def get(
        self,
        horizon: int,
    ) -> HorizonRangePrediction:
        if horizon not in self.results:
            raise KeyError(
                f"Horizon {horizon}D was not evaluated."
            )

        return self.results[horizon]

    def summary(self) -> dict[str, object]:
        horizon_summary = {}

        for horizon, result in (
            self.results.items()
        ):
            horizon_summary[
                str(horizon)
            ] = {
                "evaluated": result.evaluated,
                "passed": result.passed,
                "coverage": result.coverage,
                "average_width": (
                    result.average_width
                ),
                "selected_model_id": (
                    result.selected_model_id
                ),
            }

        return {
            "horizons": list(
                self.horizons
            ),
            "evaluated_horizons": list(
                self.evaluated_horizons
            ),
            "passed_horizons": list(
                self.passed_horizons
            ),
            "failed_horizons": list(
                self.failed_horizons
            ),
            "evaluated_count": (
                self.evaluated_count
            ),
            "passed_count": (
                self.passed_count
            ),
            "candidate_passed": (
                self.candidate_passed
            ),
            "production_ready": False,
            "research_only": True,
            "final_holdout_used": (
                self.final_holdout_used
            ),
            "horizon_results": horizon_summary,
        }


class RangePredictionIntegration:
    """
    Integrate target-price range research with selected models.

    The selected-model result is used only as model identity/evidence.
    Actual range predictions are supplied to `evaluate_ranges`, keeping
    model fitting outside this integration boundary.
    """

    def __init__(
        self,
        *,
        config: RangePredictionIntegrationConfig
        | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else RangePredictionIntegrationConfig()
        )

        self.range_engine = (
            RangeResearchEngine()
        )

    @staticmethod
    def _validate_selection(
        selection: MultiHorizonSelectionResult,
    ) -> None:
        if not isinstance(
            selection,
            MultiHorizonSelectionResult,
        ):
            raise TypeError(
                "selection must be MultiHorizonSelectionResult."
            )

        if selection.final_holdout_used:
            raise ValueError(
                "Range integration cannot use a selection "
                "result that claims final holdout usage."
            )

        if selection.metadata.get(
            "production_approved",
            False,
        ):
            raise ValueError(
                "Research range integration cannot consume "
                "a production-approved selection."
            )

    @staticmethod
    def _validate_series(
        value: Any,
        name: str,
    ) -> pd.Series:
        if not isinstance(
            value,
            pd.Series,
        ):
            raise TypeError(
                f"{name} must be a pandas Series."
            )

        if not isinstance(
            value.index,
            pd.DatetimeIndex,
        ):
            raise TypeError(
                f"{name} must have a DatetimeIndex."
            )

        if value.index.has_duplicates:
            raise ValueError(
                f"{name} contains duplicate timestamps."
            )

        if not value.index.is_monotonic_increasing:
            raise ValueError(
                f"{name} must be chronological."
            )

        numeric = pd.to_numeric(
            value,
            errors="coerce",
        )

        if numeric.isna().any():
            raise ValueError(
                f"{name} contains invalid values."
            )

        if not np.isfinite(
            numeric.to_numpy()
        ).all():
            raise ValueError(
                f"{name} contains non-finite values."
            )

        return numeric.astype(float)

    @staticmethod
    def _align_series(
        lower: pd.Series,
        median: pd.Series,
        upper: pd.Series,
    ) -> tuple[
        pd.Series,
        pd.Series,
        pd.Series,
    ]:
        frame = pd.concat(
            [
                lower.rename("lower"),
                median.rename("median"),
                upper.rename("upper"),
            ],
            axis=1,
            join="inner",
        ).dropna()

        if frame.empty:
            raise ValueError(
                "Range predictions have no overlapping timestamps."
            )

        return (
            frame["lower"],
            frame["median"],
            frame["upper"],
        )

    @staticmethod
    def _validate_range_order(
        lower: pd.Series,
        median: pd.Series,
        upper: pd.Series,
    ) -> None:
        invalid = (
            (lower > median)
            | (median > upper)
        )

        if invalid.any():
            raise ValueError(
                "Range predictions violate "
                "lower <= median <= upper."
            )

    @staticmethod
    def _return_to_price(
        current_price: pd.Series,
        returns: pd.Series,
    ) -> pd.Series:
        current_price, returns = (
            current_price.align(
                returns,
                join="inner",
            )
        )

        if current_price.empty:
            raise ValueError(
                "Current price and return series "
                "have no overlapping timestamps."
            )

        price = (
            current_price.astype(float)
            * (1.0 + returns.astype(float))
        )

        if (
            ~np.isfinite(
                price.to_numpy()
            )
        ).any():
            raise ValueError(
                "Converted price range contains "
                "non-finite values."
            )

        if (price <= 0).any():
            raise ValueError(
                "Converted price range contains "
                "non-positive prices."
            )

        return price

    @staticmethod
    def _selected_model_id(
        selection: MultiHorizonSelectionResult,
        horizon: int,
    ) -> str | None:
        if horizon not in selection.results:
            return None

        result = selection.results[
            horizon
        ]

        return result.selected_model_id

    def evaluate_ranges(
        self,
        *,
        selection: MultiHorizonSelectionResult,
        predictions: dict[
            int,
            dict[str, pd.Series],
        ],
        current_prices: dict[
            int,
            pd.Series,
        ],
    ) -> MultiHorizonRangePredictionResult:
        """
        Evaluate already-generated range predictions.

        Parameters
        ----------
        selection:
            Development-only selected model evidence.

        predictions:
            Mapping:
                horizon -> {
                    "lower": Series,
                    "median": Series,
                    "upper": Series,
                    "actual_lower": Series,
                    "actual_upper": Series,
                }

            The actual lower/upper series represent the realized
            future path used to assess whether the predicted interval
            contained the observed outcome.

        current_prices:
            Current/reference price series for each horizon.

        Notes
        -----
        This method deliberately does not fit a range model.
        """

        self._validate_selection(
            selection
        )

        if not isinstance(
            predictions,
            dict,
        ):
            raise TypeError(
                "predictions must be a dictionary."
            )

        if not isinstance(
            current_prices,
            dict,
        ):
            raise TypeError(
                "current_prices must be a dictionary."
            )

        results = {}

        evaluated = []
        passed = []
        failed = []

        warnings = []
        errors = []

        for horizon in self.config.horizons:
            horizon_predictions = (
                predictions.get(horizon)
            )

            if horizon_predictions is None:
                failed.append(
                    horizon
                )

                warnings.append(
                    f"{horizon}D range predictions "
                    "were not supplied."
                )

                continue

            required = {
                "lower",
                "median",
                "upper",
                "actual_lower",
                "actual_upper",
            }

            missing = (
                required
                - set(horizon_predictions)
            )

            if missing:
                failed.append(
                    horizon
                )

                errors.append(
                    f"{horizon}D range predictions "
                    f"missing: {sorted(missing)}"
                )

                continue

            try:
                lower = self._validate_series(
                    horizon_predictions["lower"],
                    f"{horizon}D lower",
                )

                median = self._validate_series(
                    horizon_predictions["median"],
                    f"{horizon}D median",
                )

                upper = self._validate_series(
                    horizon_predictions["upper"],
                    f"{horizon}D upper",
                )

                actual_lower = self._validate_series(
                    horizon_predictions["actual_lower"],
                    f"{horizon}D actual lower",
                )

                actual_upper = self._validate_series(
                    horizon_predictions["actual_upper"],
                    f"{horizon}D actual upper",
                )

                lower, median, upper = (
                    self._align_series(
                        lower,
                        median,
                        upper,
                    )
                )

                actual_lower, actual_upper = (
                    actual_lower.align(
                        actual_upper,
                        join="inner",
                    )
                )

                if actual_lower.empty:
                    raise ValueError(
                        "Actual range contains no "
                        "overlapping timestamps."
                    )

                self._validate_range_order(
                    lower,
                    median,
                    upper,
                )

                actual_lower, lower = (
                    actual_lower.align(
                        lower,
                        join="inner",
                    )
                )

                actual_upper, upper = (
                    actual_upper.align(
                        upper,
                        join="inner",
                    )
                )

                median = median.reindex(
                    lower.index
                )

                if len(lower) < (
                    self.config.minimum_observations
                ):
                    raise ValueError(
                        f"{horizon}D has only "
                        f"{len(lower)} overlapping observations; "
                        f"minimum is "
                        f"{self.config.minimum_observations}."
                    )

                current_price = (
                    current_prices.get(horizon)
                )

                if current_price is None:
                    raise ValueError(
                        f"No current-price series supplied "
                        f"for {horizon}D."
                    )

                current_price = (
                    self._validate_series(
                        current_price,
                        f"{horizon}D current price",
                    )
                )

                lower_price = (
                    self._return_to_price(
                        current_price,
                        lower,
                    )
                )

                median_price = (
                    self._return_to_price(
                        current_price,
                        median,
                    )
                )

                upper_price = (
                    self._return_to_price(
                        current_price,
                        upper,
                    )
                )

                actual_frame = pd.concat(
                    [
                        actual_lower.rename(
                            "actual_lower"
                        ),
                        actual_upper.rename(
                            "actual_upper"
                        ),
                        lower.rename(
                            "lower"
                        ),
                        upper.rename(
                            "upper"
                        ),
                    ],
                    axis=1,
                    join="inner",
                ).dropna()

                if actual_frame.empty:
                    raise ValueError(
                        "No observations remain after "
                        "range alignment."
                    )

                coverage = float(
                    (
                        (
                            actual_frame[
                                "actual_lower"
                            ]
                            >= actual_frame[
                                "lower"
                            ]
                        )
                        & (
                            actual_frame[
                                "actual_upper"
                            ]
                            <= actual_frame[
                                "upper"
                            ]
                        )
                    ).mean()
                )

                width = (
                    actual_frame["upper"]
                    - actual_frame["lower"]
                )

                average_width = float(
                    width.mean()
                )

                passed = (
                    self.config.minimum_coverage
                    <= coverage
                    <= self.config.maximum_coverage
                )

                selected_model_id = (
                    self._selected_model_id(
                        selection,
                        horizon,
                    )
                )

                if selected_model_id is None:
                    warnings.append(
                        f"{horizon}D has no selected "
                        "model identity."
                    )

                result = (
                    HorizonRangePrediction(
                        horizon=horizon,
                        lower_return=lower,
                        median_return=median,
                        upper_return=upper,
                        lower_price=lower_price,
                        median_price=median_price,
                        upper_price=upper_price,
                        coverage=coverage,
                        average_width=average_width,
                        selected_model_id=(
                            selected_model_id
                        ),
                        evaluated=True,
                        passed=passed,
                        warnings=[],
                        metadata={
                            "expected_coverage": (
                                self.config.expected_coverage
                            ),
                            "minimum_coverage": (
                                self.config.minimum_coverage
                            ),
                            "maximum_coverage": (
                                self.config.maximum_coverage
                            ),
                            "final_holdout_used": False,
                            "research_only": True,
                            "production_approved": False,
                        },
                    )
                )

                results[horizon] = result
                evaluated.append(
                    horizon
                )

                if passed:
                    passed_horizons = True
                    passed.append(
                        horizon
                    )
                else:
                    passed_horizons = False
                    failed.append(
                        horizon
                    )

                results[horizon].passed = (
                    passed_horizons
                )

            except Exception as exc:
                failed.append(
                    horizon
                )

                errors.append(
                    f"{horizon}D range evaluation failed: "
                    f"{type(exc).__name__}: {exc}"
                )

        candidate_passed = (
            len(passed) >= 1
            and len(passed) == len(evaluated)
        )

        metadata = {
            "research_only": True,
            "production_ready": False,
            "production_approved": False,
            "final_holdout_used": False,
            "range_model_fitted_here": False,
            "calibration_fitted_here": False,
            "threshold_optimization_completed": False,
            "selection_used_for_identity_only": True,
        }

        return MultiHorizonRangePredictionResult(
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
            candidate_passed=(
                candidate_passed
            ),
            production_ready=False,
            warnings=warnings,
            errors=errors,
            metadata=metadata,
        )


def integrate_range_predictions(
    *,
    selection: MultiHorizonSelectionResult,
    predictions: dict[
        int,
        dict[str, pd.Series],
    ],
    current_prices: dict[
        int,
        pd.Series,
    ],
    config: RangePredictionIntegrationConfig
    | None = None,
) -> MultiHorizonRangePredictionResult:
    """Convenience API for range prediction integration."""

    pipeline = RangePredictionIntegration(
        config=config
    )

    return pipeline.evaluate_ranges(
        selection=selection,
        predictions=predictions,
        current_prices=current_prices,
    )


def range_prediction_integration_summary(
    result: MultiHorizonRangePredictionResult,
) -> dict[str, object]:
    """Return a compact range integration summary."""

    if not isinstance(
        result,
        MultiHorizonRangePredictionResult,
    ):
        raise TypeError(
            "result must be MultiHorizonRangePredictionResult."
        )

    return result.summary()


__all__ = [
    "DEFAULT_RANGE_HORIZONS",
    "RangePredictionIntegrationConfig",
    "HorizonRangePrediction",
    "MultiHorizonRangePredictionResult",
    "RangePredictionIntegration",
    "integrate_range_predictions",
    "range_prediction_integration_summary",
]
