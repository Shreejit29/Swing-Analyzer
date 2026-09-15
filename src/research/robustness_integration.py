"""
AI Swing Analyser — Robustness Integration.

Connects multi-horizon backtest evidence with robustness analysis.

Research rules
--------------
- Final holdout is never used.
- Trade order is preserved.
- No model fitting occurs here.
- No threshold optimization occurs here.
- Robustness does not grant production approval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .backtest_integration import (
    MultiHorizonBacktestIntegrationResult,
)
from src.models.robustness import RobustnessConfig
from .robustness_research import (
    RobustnessResearchEngine,
    RobustnessResearchResult,
)


DEFAULT_ROBUSTNESS_HORIZONS = (
    1,
    3,
    5,
    10,
    20,
)


@dataclass(frozen=True)
class RobustnessIntegrationConfig:
    """Configuration for multi-horizon robustness analysis."""

    horizons: tuple[int, ...] = (
        DEFAULT_ROBUSTNESS_HORIZONS
    )

    simulations: int = 5000

    random_state: int = 42

    transaction_cost_multiplier: float = 2.0

    slippage_multiplier: float = 2.0

    removal_fraction: float = 0.10

    minimum_trades: int = 30

    minimum_robustness_score: float = 0.60

    require_backtest_pass: bool = True

    def __post_init__(self) -> None:
        horizons = tuple(
            int(h)
            for h in self.horizons
        )

        if not horizons:
            raise ValueError(
                "At least one robustness horizon is required."
            )

        if any(
            h <= 0
            for h in horizons
        ):
            raise ValueError(
                "Robustness horizons must be positive."
            )

        if len(horizons) != len(
            set(horizons)
        ):
            raise ValueError(
                "Robustness horizons must be unique."
            )

        if self.simulations < 100:
            raise ValueError(
                "At least 100 simulations are required."
            )

        if self.random_state < 0:
            raise ValueError(
                "random_state must be non-negative."
            )

        if (
            self.transaction_cost_multiplier
            < 0
        ):
            raise ValueError(
                "transaction_cost_multiplier cannot be negative."
            )

        if (
            self.slippage_multiplier
            < 0
        ):
            raise ValueError(
                "slippage_multiplier cannot be negative."
            )

        if not (
            0.0
            <= self.removal_fraction
            < 1.0
        ):
            raise ValueError(
                "removal_fraction must be in [0, 1)."
            )

        if self.minimum_trades < 1:
            raise ValueError(
                "minimum_trades must be positive."
            )

        if not (
            0.0
            <= self.minimum_robustness_score
            <= 1.0
        ):
            raise ValueError(
                "minimum_robustness_score must be "
                "between 0 and 1."
            )


@dataclass
class HorizonRobustnessResult:
    """Robustness result for one horizon."""

    horizon: int

    robustness_result: RobustnessResearchResult | None

    evaluated: bool

    passed: bool

    trade_count: int

    robustness_score: float | None

    monte_carlo_profit_probability: float | None

    monte_carlo_drawdown_probability: float | None

    cost_stress_passed: bool | None

    slippage_stress_passed: bool | None

    best_trade_removal_passed: bool | None

    worst_trade_removal_passed: bool | None

    warnings: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, object] = field(
        default_factory=dict
    )


@dataclass
class MultiHorizonRobustnessIntegrationResult:
    """Aggregated robustness evidence."""

    horizons: tuple[int, ...]

    results: dict[
        int,
        HorizonRobustnessResult,
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
    ) -> HorizonRobustnessResult:
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
                "trade_count": result.trade_count,
                "robustness_score": (
                    result.robustness_score
                ),
                "monte_carlo_profit_probability": (
                    result.monte_carlo_profit_probability
                ),
                "monte_carlo_drawdown_probability": (
                    result.monte_carlo_drawdown_probability
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


class RobustnessIntegration:
    """
    Connect backtest trade returns to robustness analysis.

    This layer consumes completed research backtests and performs
    robustness testing independently for each horizon.
    """

    def __init__(
        self,
        *,
        config: RobustnessIntegrationConfig
        | None = None,
        robustness_config: RobustnessConfig
        | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else RobustnessIntegrationConfig()
        )

        if robustness_config is None:
            robustness_config = (
                RobustnessResearchConfig(
                    simulations=self.config.simulations,
                    random_state=self.config.random_state,
                    transaction_cost_multiplier=(
                        self.config.transaction_cost_multiplier
                    ),
                    slippage_multiplier=(
                        self.config.slippage_multiplier
                    ),
                    removal_fraction=(
                        self.config.removal_fraction
                    ),
                    minimum_trades=(
                        self.config.minimum_trades
                    ),
                    minimum_robustness_score=(
                        self.config.minimum_robustness_score
                    ),
                )
            )

        self.engine = (
            RobustnessResearchEngine(
                config=robustness_config
            )
        )

    @staticmethod
    def _validate_backtest(
        result: MultiHorizonBacktestIntegrationResult,
    ) -> None:
        if not isinstance(
            result,
            MultiHorizonBacktestIntegrationResult,
        ):
            raise TypeError(
                "backtest_result must be "
                "MultiHorizonBacktestIntegrationResult."
            )

        if result.final_holdout_used:
            raise ValueError(
                "Robustness analysis cannot consume "
                "backtest evidence claiming final holdout usage."
            )

        if result.metadata.get(
            "production_approved",
            False,
        ):
            raise ValueError(
                "Robustness research cannot consume "
                "production-approved evidence."
            )

    @staticmethod
    def _validate_returns(
        returns: Any,
        horizon: int,
    ) -> pd.Series:
        if isinstance(
            returns,
            pd.Series,
        ):
            series = returns.copy()
        elif isinstance(
            returns,
            (list, tuple, np.ndarray),
        ):
            series = pd.Series(
                returns,
                dtype=float,
            )
        else:
            raise TypeError(
                f"{horizon}D trade returns must "
                "be a Series, list, tuple, or ndarray."
            )

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        if numeric.isna().any():
            raise ValueError(
                f"{horizon}D trade returns contain invalid values."
            )

        values = numeric.to_numpy(
            dtype=float
        )

        if not np.isfinite(
            values
        ).all():
            raise ValueError(
                f"{horizon}D trade returns contain non-finite values."
            )

        return pd.Series(
            values,
            index=series.index,
            dtype=float,
            name="Trade_Return",
        )

    @staticmethod
    def _extract_metric(
        result: Any,
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
                value = float(value)

                if np.isfinite(value):
                    return value
            except (
                TypeError,
                ValueError,
            ):
                pass

        return None

    @staticmethod
    def _extract_trade_count(
        result: Any,
        returns: pd.Series,
    ) -> int:
        for name in (
            "trade_count",
            "number_of_trades",
            "trades",
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

        return len(returns)

    @staticmethod
    def _extract_passed(
        result: Any,
    ) -> bool:
        value = getattr(
            result,
            "passed",
            None,
        )

        if value is not None:
            return bool(value)

        value = getattr(
            result,
            "robustness_passed",
            None,
        )

        if value is not None:
            return bool(value)

        score = (
            RobustnessIntegration
            ._extract_metric(
                result,
                (
                    "robustness_score",
                    "score",
                ),
            )
        )

        if score is None:
            return False

        return (
            score
            >= 0.60
        )

    @staticmethod
    def _extract_subresult(
        result: Any,
        names: tuple[str, ...],
    ) -> Any:
        for name in names:
            value = getattr(
                result,
                name,
                None,
            )

            if value is not None:
                return value

        return None

    @staticmethod
    def _extract_boolean(
        result: Any,
        names: tuple[str, ...],
    ) -> bool | None:
        value = (
            RobustnessIntegration
            ._extract_subresult(
                result,
                names,
            )
        )

        if value is None:
            return None

        if isinstance(
            value,
            bool,
        ):
            return value

        passed = getattr(
            value,
            "passed",
            None,
        )

        if passed is not None:
            return bool(passed)

        return None

    def _run_one(
        self,
        *,
        horizon: int,
        trade_returns: pd.Series,
    ) -> HorizonRobustnessResult:
        warnings: list[str] = []

        if len(trade_returns) < (
            self.config.minimum_trades
        ):
            warnings.append(
                f"{horizon}D contains only "
                f"{len(trade_returns)} trades; "
                f"minimum recommended is "
                f"{self.config.minimum_trades}."
            )

        result = self.engine.analyze(
            trade_returns
        )

        robustness_score = (
            self._extract_metric(
                result,
                (
                    "robustness_score",
                    "score",
                ),
            )
        )

        mc_result = (
            self._extract_subresult(
                result,
                (
                    "monte_carlo",
                    "monte_carlo_result",
                ),
            )
        )

        cost_result = (
            self._extract_subresult(
                result,
                (
                    "cost_stress",
                    "transaction_cost_stress",
                ),
            )
        )

        slippage_result = (
            self._extract_subresult(
                result,
                (
                    "slippage_stress",
                ),
            )
        )

        best_result = (
            self._extract_subresult(
                result,
                (
                    "best_trade_removal",
                    "best_trade_removal_result",
                ),
            )
        )

        worst_result = (
            self._extract_subresult(
                result,
                (
                    "worst_trade_removal",
                    "worst_trade_removal_result",
                ),
            )
        )

        mc_profit_probability = (
            self._extract_metric(
                mc_result,
                (
                    "profit_probability",
                    "probability_of_profit",
                ),
            )
        )

        mc_drawdown_probability = (
            self._extract_metric(
                mc_result,
                (
                    "drawdown_probability",
                    "probability_of_drawdown",
                ),
            )
        )

        cost_passed = (
            self._extract_boolean(
                result,
                (
                    "cost_stress_passed",
                ),
            )
        )

        if cost_passed is None:
            cost_passed = (
                self._extract_boolean(
                    cost_result,
                    (
                        "passed",
                        "robust",
                    ),
                )
            )

        slippage_passed = (
            self._extract_boolean(
                result,
                (
                    "slippage_stress_passed",
                ),
            )
        )

        if slippage_passed is None:
            slippage_passed = (
                self._extract_boolean(
                    slippage_result,
                    (
                        "passed",
                        "robust",
                    ),
                )
            )

        best_passed = (
            self._extract_boolean(
                result,
                (
                    "best_trade_removal_passed",
                ),
            )
        )

        if best_passed is None:
            best_passed = (
                self._extract_boolean(
                    best_result,
                    (
                        "passed",
                        "robust",
                    ),
                )
            )

        worst_passed = (
            self._extract_boolean(
                result,
                (
                    "worst_trade_removal_passed",
                ),
            )
        )

        if worst_passed is None:
            worst_passed = (
                self._extract_boolean(
                    worst_result,
                    (
                        "passed",
                        "robust",
                    ),
                )
            )

        passed = self._extract_passed(
            result
        )

        return HorizonRobustnessResult(
            horizon=horizon,
            robustness_result=result,
            evaluated=True,
            passed=passed,
            trade_count=len(
                trade_returns
            ),
            robustness_score=robustness_score,
            monte_carlo_profit_probability=(
                mc_profit_probability
            ),
            monte_carlo_drawdown_probability=(
                mc_drawdown_probability
            ),
            cost_stress_passed=(
                cost_passed
            ),
            slippage_stress_passed=(
                slippage_passed
            ),
            best_trade_removal_passed=(
                best_passed
            ),
            worst_trade_removal_passed=(
                worst_passed
            ),
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
        backtest_result: MultiHorizonBacktestIntegrationResult,
        trade_returns: dict[
            int,
            pd.Series | list[float] | np.ndarray,
        ],
    ) -> MultiHorizonRobustnessIntegrationResult:
        """
        Run robustness analysis independently for each horizon.

        `trade_returns` must contain the realized trade-return
        sequence generated by the corresponding research backtest.
        The original chronological order is preserved.
        """

        self._validate_backtest(
            backtest_result
        )

        if not isinstance(
            trade_returns,
            dict,
        ):
            raise TypeError(
                "trade_returns must be a dictionary."
            )

        results = {}

        evaluated = []
        passed = []
        failed = []

        warnings: list[str] = []
        errors: list[str] = []

        for horizon in self.config.horizons:

            if horizon not in (
                backtest_result.results
            ):
                failed.append(
                    horizon
                )

                warnings.append(
                    f"{horizon}D has no backtest result."
                )

                continue

            raw_returns = trade_returns.get(
                horizon
            )

            if raw_returns is None:
                failed.append(
                    horizon
                )

                errors.append(
                    f"{horizon}D trade returns were not supplied."
                )

                continue

            backtest_item = (
                backtest_result.results[
                    horizon
                ]
            )

            if (
                self.config.require_backtest_pass
                and not backtest_item.passed
            ):
                failed.append(
                    horizon
                )

                warnings.append(
                    f"{horizon}D backtest did not pass; "
                    "robustness analysis skipped."
                )

                continue

            try:
                returns = self._validate_returns(
                    raw_returns,
                    horizon,
                )

                horizon_result = self._run_one(
                    horizon=horizon,
                    trade_returns=returns,
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
                    f"{horizon}D robustness analysis failed: "
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
            "trade_order_preserved": True,
            "best_worst_trade_removal_preserves_order": True,
        }

        return MultiHorizonRobustnessIntegrationResult(
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


def integrate_robustness(
    *,
    backtest_result: MultiHorizonBacktestIntegrationResult,
    trade_returns: dict[
        int,
        pd.Series | list[float] | np.ndarray,
    ],
    config: RobustnessIntegrationConfig
    | None = None,
    robustness_config: RobustnessConfig
    | None = None,
) -> MultiHorizonRobustnessIntegrationResult:
    """Convenience API for robustness integration."""

    pipeline = RobustnessIntegration(
        config=config,
        robustness_config=robustness_config,
    )

    return pipeline.run(
        backtest_result=backtest_result,
        trade_returns=trade_returns,
    )


def robustness_integration_summary(
    result: MultiHorizonRobustnessIntegrationResult,
) -> dict[str, object]:
    """Return a compact robustness integration summary."""

    if not isinstance(
        result,
        MultiHorizonRobustnessIntegrationResult,
    ):
        raise TypeError(
            "result must be "
            "MultiHorizonRobustnessIntegrationResult."
        )

    return result.summary()


__all__ = [
    "DEFAULT_ROBUSTNESS_HORIZONS",
    "RobustnessIntegrationConfig",
    "HorizonRobustnessResult",
    "MultiHorizonRobustnessIntegrationResult",
    "RobustnessIntegration",
    "integrate_robustness",
    "robustness_integration_summary",
]
