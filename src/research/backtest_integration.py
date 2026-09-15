"""
AI Swing Analyser — Backtest Integration.

Integrates model predictions, target ranges, and market data with
the research backtesting boundary.

Research rules
--------------
- Predictions must be point-in-time.
- Prediction timestamps must exactly align with market data.
- No forward filling of predictions.
- Final holdout data is forbidden.
- No model fitting is performed here.
- No threshold optimization is performed here.
- No production approval is granted here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.models.backtest import BacktestConfig
from .backtest_research import (
    BacktestResearchEngine,
    BacktestResearchResult,
)
from src.multi_horizon_selection_pipeline import (
    MultiHorizonSelectionResult,
)
from src.range_prediction_integration import (
    MultiHorizonRangePredictionResult,
)


DEFAULT_BACKTEST_HORIZONS = (
    1,
    3,
    5,
    10,
    20,
)


@dataclass(frozen=True)
class BacktestIntegrationConfig:
    """Configuration for research backtest integration."""

    horizons: tuple[int, ...] = (
        DEFAULT_BACKTEST_HORIZONS
    )

    probability_threshold: float = 0.60

    minimum_observations: int = 50

    require_selected_model: bool = True

    require_range_evidence: bool = False

    allow_unselected_horizons: bool = True

    def __post_init__(self) -> None:
        normalized = tuple(
            int(horizon)
            for horizon in self.horizons
        )

        if not normalized:
            raise ValueError(
                "At least one backtest horizon is required."
            )

        if any(
            horizon <= 0
            for horizon in normalized
        ):
            raise ValueError(
                "Backtest horizons must be positive."
            )

        if len(normalized) != len(
            set(normalized)
        ):
            raise ValueError(
                "Backtest horizons must be unique."
            )

        if not (
            0.50
            <= self.probability_threshold
            < 1.0
        ):
            raise ValueError(
                "probability_threshold must be >= 0.50 and < 1.0."
            )

        if (
            self.minimum_observations
            < 1
        ):
            raise ValueError(
                "minimum_observations must be positive."
            )


@dataclass
class HorizonBacktestResult:
    """Backtest result for one horizon."""

    horizon: int

    model_id: str | None

    result: BacktestResearchResult | None

    evaluated: bool

    passed: bool

    total_return: float | None

    annualized_return: float | None

    sharpe_ratio: float | None

    max_drawdown: float | None

    win_rate: float | None

    profit_factor: float | None

    trade_count: int

    warnings: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, object] = field(
        default_factory=dict
    )


@dataclass
class MultiHorizonBacktestIntegrationResult:
    """Aggregated research backtest results."""

    horizons: tuple[int, ...]

    results: dict[
        int,
        HorizonBacktestResult,
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
    ) -> HorizonBacktestResult:
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
                "model_id": result.model_id,
                "evaluated": result.evaluated,
                "passed": result.passed,
                "total_return": result.total_return,
                "annualized_return": (
                    result.annualized_return
                ),
                "sharpe_ratio": (
                    result.sharpe_ratio
                ),
                "max_drawdown": (
                    result.max_drawdown
                ),
                "win_rate": result.win_rate,
                "profit_factor": (
                    result.profit_factor
                ),
                "trade_count": result.trade_count,
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


class BacktestIntegration:
    """
    Integrate selected research candidates with the backtest engine.

    The backtest itself is delegated to BacktestResearchEngine.
    """

    def __init__(
        self,
        *,
        config: BacktestIntegrationConfig
        | None = None,
        backtest_config: BacktestConfig
        | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else BacktestIntegrationConfig()
        )

        self.backtest_engine = (
            BacktestResearchEngine(
                config=backtest_config
            )
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
                "Backtesting cannot consume a selection "
                "result that claims final holdout usage."
            )

        if selection.metadata.get(
            "production_approved",
            False,
        ):
            raise ValueError(
                "Research backtesting cannot consume a "
                "production-approved selection."
            )

    @staticmethod
    def _validate_range_result(
        range_result: MultiHorizonRangePredictionResult
        | None,
    ) -> None:
        if range_result is None:
            return

        if not isinstance(
            range_result,
            MultiHorizonRangePredictionResult,
        ):
            raise TypeError(
                "range_result must be "
                "MultiHorizonRangePredictionResult."
            )

        if range_result.final_holdout_used:
            raise ValueError(
                "Range evidence claims final holdout usage."
            )

    @staticmethod
    def _validate_market_data(
        market_data: pd.DataFrame,
    ) -> None:
        if not isinstance(
            market_data,
            pd.DataFrame,
        ):
            raise TypeError(
                "market_data must be a DataFrame."
            )

        if market_data.empty:
            raise ValueError(
                "market_data cannot be empty."
            )

        if not isinstance(
            market_data.index,
            pd.DatetimeIndex,
        ):
            raise TypeError(
                "market_data must use a DatetimeIndex."
            )

        if market_data.index.has_duplicates:
            raise ValueError(
                "market_data contains duplicate timestamps."
            )

        if not market_data.index.is_monotonic_increasing:
            raise ValueError(
                "market_data must be chronological."
            )

        required = {
            "Open",
            "High",
            "Low",
            "Close",
        }

        missing = (
            required
            - set(market_data.columns)
        )

        if missing:
            raise ValueError(
                f"market_data missing columns: "
                f"{sorted(missing)}"
            )

        for column in required:
            numeric = pd.to_numeric(
                market_data[column],
                errors="coerce",
            )

            if numeric.isna().any():
                raise ValueError(
                    f"{column} contains invalid values."
                )

            if not np.isfinite(
                numeric.to_numpy()
            ).all():
                raise ValueError(
                    f"{column} contains non-finite values."
                )

        if (
            market_data["Close"]
            <= 0
        ).any():
            raise ValueError(
                "Close contains non-positive values."
            )

        if (
            market_data["High"]
            < market_data["Low"]
        ).any():
            raise ValueError(
                "High cannot be below Low."
            )

    @staticmethod
    def _validate_prediction_frame(
        predictions: pd.DataFrame,
    ) -> None:
        if not isinstance(
            predictions,
            pd.DataFrame,
        ):
            raise TypeError(
                "predictions must be a DataFrame."
            )

        if predictions.empty:
            raise ValueError(
                "predictions cannot be empty."
            )

        if not isinstance(
            predictions.index,
            pd.DatetimeIndex,
        ):
            raise TypeError(
                "predictions must use a DatetimeIndex."
            )

        if predictions.index.has_duplicates:
            raise ValueError(
                "predictions contain duplicate timestamps."
            )

        if not predictions.index.is_monotonic_increasing:
            raise ValueError(
                "predictions must be chronological."
            )

        required = {
            "Probability",
        }

        missing = (
            required
            - set(predictions.columns)
        )

        if missing:
            raise ValueError(
                f"predictions missing columns: "
                f"{sorted(missing)}"
            )

        probability = pd.to_numeric(
            predictions["Probability"],
            errors="coerce",
        )

        if probability.isna().any():
            raise ValueError(
                "Probability contains invalid values."
            )

        values = probability.to_numpy()

        if not np.isfinite(
            values
        ).all():
            raise ValueError(
                "Probability contains non-finite values."
            )

        if (
            (values < 0.0)
            | (values > 1.0)
        ).any():
            raise ValueError(
                "Probability must be between 0 and 1."
            )

    @staticmethod
    def _extract_metric(
        result: BacktestResearchResult,
        names: tuple[str, ...],
    ) -> float | None:
        for name in names:
            value = getattr(
                result,
                name,
                None,
            )

            if value is None:
                continue

            try:
                numeric = float(value)

                if np.isfinite(
                    numeric
                ):
                    return numeric

            except (
                TypeError,
                ValueError,
            ):
                continue

        metrics = getattr(
            result,
            "metrics",
            None,
        )

        if isinstance(
            metrics,
            dict,
        ):
            for name in names:
                value = metrics.get(
                    name
                )

                if value is None:
                    continue

                try:
                    numeric = float(value)

                    if np.isfinite(
                        numeric
                    ):
                        return numeric

                except (
                    TypeError,
                    ValueError,
                ):
                    continue

        return None

    @staticmethod
    def _extract_trade_count(
        result: BacktestResearchResult,
    ) -> int:
        for name in (
            "trade_count",
            "trades",
            "number_of_trades",
        ):
            value = getattr(
                result,
                name,
                None,
            )

            if value is not None:
                try:
                    return int(value)
                except (
                    TypeError,
                    ValueError,
                ):
                    pass

        metrics = getattr(
            result,
            "metrics",
            None,
        )

        if isinstance(
            metrics,
            dict,
        ):
            for name in (
                "trade_count",
                "trades",
                "number_of_trades",
            ):
                value = metrics.get(
                    name
                )

                if value is not None:
                    try:
                        return int(value)
                    except (
                        TypeError,
                        ValueError,
                    ):
                        pass

        return 0

    @staticmethod
    def _range_available(
        range_result: MultiHorizonRangePredictionResult
        | None,
        horizon: int,
    ) -> bool:
        if range_result is None:
            return False

        item = range_result.results.get(
            horizon
        )

        return bool(
            item is not None
            and item.evaluated
            and item.passed
        )

    def _run_one(
        self,
        *,
        horizon: int,
        model_id: str | None,
        predictions: pd.DataFrame,
        market_data: pd.DataFrame,
    ) -> HorizonBacktestResult:
        """
        Execute one horizon's research backtest.
        """

        warnings: list[str] = []

        self._validate_prediction_frame(
            predictions
        )

        self._validate_market_data(
            market_data
        )

        if len(predictions) < (
            self.config.minimum_observations
        ):
            warnings.append(
                f"{horizon}D has fewer observations "
                "than the configured minimum."
            )

        aligned_index = (
            predictions.index
            .intersection(
                market_data.index
            )
        )

        if aligned_index.empty:
            raise ValueError(
                f"{horizon}D predictions have no "
                "exactly matching market timestamps."
            )

        aligned_predictions = (
            predictions
            .loc[aligned_index]
            .copy()
        )

        aligned_market = (
            market_data
            .loc[aligned_index]
            .copy()
        )

        # The intersection is intentional. We do not forward-fill
        # predictions or market data.
        if len(aligned_predictions) != len(
            predictions
        ):
            warnings.append(
                f"{horizon}D prediction rows without exact "
                "market timestamps were excluded."
            )

        if (
            aligned_predictions.index
            != aligned_market.index
        ).any():
            raise RuntimeError(
                "Prediction and market timelines "
                "are not exactly aligned."
            )

        combined = aligned_market.copy()

        combined[
            "Probability"
        ] = aligned_predictions[
            "Probability"
        ]

        if "Direction" in aligned_predictions:
            combined[
                "Direction"
            ] = aligned_predictions[
                "Direction"
            ]

        result = self.backtest_engine.run(
            market_data=combined,
            predictions=combined[
                [
                    "Probability"
                ]
            ],
        )

        total_return = (
            self._extract_metric(
                result,
                (
                    "total_return",
                    "return",
                ),
            )
        )

        annualized_return = (
            self._extract_metric(
                result,
                (
                    "annualized_return",
                    "annual_return",
                ),
            )
        )

        sharpe_ratio = (
            self._extract_metric(
                result,
                (
                    "sharpe_ratio",
                    "sharpe",
                ),
            )
        )

        max_drawdown = (
            self._extract_metric(
                result,
                (
                    "max_drawdown",
                    "maximum_drawdown",
                    "drawdown",
                ),
            )
        )

        win_rate = (
            self._extract_metric(
                result,
                (
                    "win_rate",
                    "winning_rate",
                ),
            )
        )

        profit_factor = (
            self._extract_metric(
                result,
                (
                    "profit_factor",
                    "profit_factor_ratio",
                ),
            )
        )

        trade_count = (
            self._extract_trade_count(
                result
            )
        )

        # This integration layer does not invent a new approval
        # threshold. The underlying research result remains the
        # authoritative backtest evidence.
        passed = bool(
            getattr(
                result,
                "passed",
                False,
            )
        )

        return HorizonBacktestResult(
            horizon=horizon,
            model_id=model_id,
            result=result,
            evaluated=True,
            passed=passed,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            trade_count=trade_count,
            warnings=warnings,
            metadata={
                "research_only": True,
                "final_holdout_used": False,
                "model_fitted_here": False,
                "threshold_optimized_here": False,
                "production_approved": False,
            },
        )

    def run(
        self,
        *,
        selection: MultiHorizonSelectionResult,
        predictions: dict[
            int,
            pd.DataFrame,
        ],
        market_data: pd.DataFrame,
        range_result: MultiHorizonRangePredictionResult
        | None = None,
    ) -> MultiHorizonBacktestIntegrationResult:
        """
        Run research backtests independently for each horizon.

        Parameters
        ----------
        selection:
            Development-only model-selection evidence.

        predictions:
            Mapping of horizon to point-in-time prediction DataFrame.
            Each frame must contain a `Probability` column.

        market_data:
            OHLC market data indexed by timestamp.

        range_result:
            Optional range-validation evidence. It is not required by
            default because range validation is an independent gate.
        """

        self._validate_selection(
            selection
        )

        self._validate_range_result(
            range_result
        )

        self._validate_market_data(
            market_data
        )

        if not isinstance(
            predictions,
            dict,
        ):
            raise TypeError(
                "predictions must be a dictionary."
            )

        results = {}

        evaluated = []
        passed = []
        failed = []

        warnings: list[str] = []
        errors: list[str] = []

        for horizon in self.config.horizons:

            selection_item = (
                selection.results.get(
                    horizon
                )
            )

            if (
                self.config.require_selected_model
                and (
                    selection_item is None
                    or not selection_item.selected
                )
            ):
                failed.append(
                    horizon
                )

                warnings.append(
                    f"{horizon}D has no selected model; "
                    "backtest skipped."
                )

                continue

            prediction_frame = (
                predictions.get(horizon)
            )

            if prediction_frame is None:
                failed.append(
                    horizon
                )

                errors.append(
                    f"{horizon}D predictions were not supplied."
                )

                continue

            if (
                self.config.require_range_evidence
                and not self._range_available(
                    range_result,
                    horizon,
                )
            ):
                failed.append(
                    horizon
                )

                warnings.append(
                    f"{horizon}D range evidence did not pass; "
                    "backtest skipped."
                )

                continue

            model_id = (
                selection_item.selected_model_id
                if selection_item is not None
                else None
            )

            try:
                horizon_result = self._run_one(
                    horizon=horizon,
                    model_id=model_id,
                    predictions=prediction_frame,
                    market_data=market_data,
                )

                results[
                    horizon
                ] = horizon_result

                evaluated.append(
                    horizon
                )

                if horizon_result.passed:
                    passed.append(
                        horizon
                    )
                else:
                    failed.append(
                        horizon
                    )

                warnings.extend(
                    horizon_result.warnings
                )

            except Exception as exc:
                failed.append(
                    horizon
                )

                errors.append(
                    f"{horizon}D backtest failed: "
                    f"{type(exc).__name__}: {exc}"
                )

        candidate_passed = (
            len(passed) > 0
            and len(passed) == len(evaluated)
        )

        metadata = {
            "research_only": True,
            "production_ready": False,
            "production_approved": False,
            "final_holdout_used": False,
            "model_fitted_here": False,
            "threshold_optimized_here": False,
            "forward_fill_used": False,
            "prediction_timestamp_alignment": "exact",
            "selection_used": True,
            "range_evidence_used": (
                range_result is not None
            ),
        }

        return MultiHorizonBacktestIntegrationResult(
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


def integrate_backtest(
    *,
    selection: MultiHorizonSelectionResult,
    predictions: dict[
        int,
        pd.DataFrame,
    ],
    market_data: pd.DataFrame,
    range_result: MultiHorizonRangePredictionResult
    | None = None,
    config: BacktestIntegrationConfig
    | None = None,
    backtest_config: BacktestConfig
    | None = None,
) -> MultiHorizonBacktestIntegrationResult:
    """Convenience API for research backtest integration."""

    pipeline = BacktestIntegration(
        config=config,
        backtest_config=backtest_config,
    )

    return pipeline.run(
        selection=selection,
        predictions=predictions,
        market_data=market_data,
        range_result=range_result,
    )


def backtest_integration_summary(
    result: MultiHorizonBacktestIntegrationResult,
) -> dict[str, object]:
    """Return a compact backtest integration summary."""

    if not isinstance(
        result,
        MultiHorizonBacktestIntegrationResult,
    ):
        raise TypeError(
            "result must be "
            "MultiHorizonBacktestIntegrationResult."
        )

    return result.summary()


__all__ = [
    "DEFAULT_BACKTEST_HORIZONS",
    "BacktestIntegrationConfig",
    "HorizonBacktestResult",
    "MultiHorizonBacktestIntegrationResult",
    "BacktestIntegration",
    "integrate_backtest",
    "backtest_integration_summary",
]
