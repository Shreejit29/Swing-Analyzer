"""
AI Swing Analyser — Regime Research Engine.

Evaluates model performance across different market regimes.

The goal is to determine whether a model is stable across:

    - Bull / Bear / Neutral trends
    - Low / Normal / High volatility
    - Combined trend-volatility regimes

This module is evaluation-only.

It must never modify a trained model, fit a model, or use the
final holdout for model development.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.models.regime_validation import (
    RegimeValidationConfig,
    RegimeValidationReport,
    evaluate_by_regime,
    build_regime_report,
)


# ---------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------


@dataclass
class RegimeResearchResult:
    """
    Result of regime-specific model evaluation.
    """

    report: RegimeValidationReport

    total_samples: int

    trend_regimes: list[str]

    volatility_regimes: list[str]

    combined_regimes: list[str]

    minimum_regime_accuracy: float | None

    maximum_regime_accuracy: float | None

    regime_accuracy_std: float | None

    stability_score: float

    passed_stability_gate: bool

    final_holdout_used: bool

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    warnings: list[str] = field(
        default_factory=list
    )

    @property
    def stable(self) -> bool:
        return self.passed_stability_gate

    def summary(self) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "trend_regimes": list(
                self.trend_regimes
            ),
            "volatility_regimes": list(
                self.volatility_regimes
            ),
            "combined_regimes": list(
                self.combined_regimes
            ),
            "minimum_regime_accuracy": (
                self.minimum_regime_accuracy
            ),
            "maximum_regime_accuracy": (
                self.maximum_regime_accuracy
            ),
            "regime_accuracy_std": (
                self.regime_accuracy_std
            ),
            "stability_score": (
                self.stability_score
            ),
            "passed_stability_gate": (
                self.passed_stability_gate
            ),
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


class RegimeResearchEngine:
    """
    Regime-aware research evaluator.

    This engine does not train or modify models.
    """

    def __init__(
        self,
        config: RegimeValidationConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else RegimeValidationConfig()
        )

        self._validate_config()

    # -----------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------

    def _validate_config(self) -> None:

        minimum_samples = getattr(
            self.config,
            "minimum_samples",
            30,
        )

        minimum_accuracy = getattr(
            self.config,
            "minimum_accuracy",
            0.55,
        )

        maximum_accuracy_std = getattr(
            self.config,
            "maximum_accuracy_std",
            0.15,
        )

        if minimum_samples < 1:
            raise ValueError(
                "minimum_samples must be positive."
            )

        if not (
            0.0
            <= minimum_accuracy
            <= 1.0
        ):
            raise ValueError(
                "minimum_accuracy must be between 0 and 1."
            )

        if maximum_accuracy_std < 0:
            raise ValueError(
                "maximum_accuracy_std cannot be negative."
            )

    # -----------------------------------------------------------------
    # Input validation
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_predictions(
        actuals: np.ndarray,
        predictions: np.ndarray,
        timestamps: pd.Index,
    ) -> None:

        actuals = np.asarray(
            actuals
        )

        predictions = np.asarray(
            predictions
        )

        if len(actuals) == 0:
            raise ValueError(
                "Regime evaluation dataset is empty."
            )

        if len(actuals) != len(
            predictions
        ):
            raise ValueError(
                "Actuals and predictions must have "
                "the same length."
            )

        if len(actuals) != len(
            timestamps
        ):
            raise ValueError(
                "Timestamps must have the same "
                "length as predictions."
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

        if len(
            np.unique(actuals)
        ) < 2:
            raise ValueError(
                "Actuals must contain both classes."
            )

    @staticmethod
    def _validate_regime_columns(
        dataframe: pd.DataFrame,
        columns: list[str],
    ) -> None:

        missing = [
            column
            for column in columns
            if column not in dataframe.columns
        ]

        if missing:
            raise ValueError(
                "Missing regime columns: "
                + ", ".join(missing)
            )

    # -----------------------------------------------------------------
    # Regime construction
    # -----------------------------------------------------------------

    @staticmethod
    def add_trend_regime(
        dataframe: pd.DataFrame,
        fast_period: int = 50,
        slow_period: int = 200,
    ) -> pd.DataFrame:

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                "dataframe must be a DataFrame."
            )

        if "Close" not in dataframe.columns:
            raise ValueError(
                "Close column is required."
            )

        if fast_period <= 0:
            raise ValueError(
                "fast_period must be positive."
            )

        if slow_period <= fast_period:
            raise ValueError(
                "slow_period must be greater than fast_period."
            )

        result = dataframe.copy()

        fast = result[
            "Close"
        ].rolling(
            fast_period,
            min_periods=fast_period,
        ).mean()

        slow = result[
            "Close"
        ].rolling(
            slow_period,
            min_periods=slow_period,
        ).mean()

        regime = np.where(
            fast > slow,
            "BULL",
            np.where(
                fast < slow,
                "BEAR",
                "NEUTRAL",
            ),
        )

        regime = pd.Series(
            regime,
            index=result.index,
        )

        regime.loc[
            slow.isna()
        ] = "UNKNOWN"

        result[
            "Trend_Regime"
        ] = regime

        return result

    @staticmethod
    def add_volatility_regime(
        dataframe: pd.DataFrame,
        return_period: int = 20,
        low_quantile: float = 0.33,
        high_quantile: float = 0.67,
    ) -> pd.DataFrame:

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                "dataframe must be a DataFrame."
            )

        if "Close" not in dataframe.columns:
            raise ValueError(
                "Close column is required."
            )

        if return_period <= 1:
            raise ValueError(
                "return_period must be greater than 1."
            )

        if not (
            0.0
            < low_quantile
            < high_quantile
            < 1.0
        ):
            raise ValueError(
                "Quantiles must satisfy "
                "0 < low < high < 1."
            )

        result = dataframe.copy()

        returns = (
            result[
                "Close"
            ].pct_change()
        )

        volatility = (
            returns
            .rolling(
                return_period,
                min_periods=return_period,
            )
            .std()
        )

        # Expanding quantiles are used instead of full-sample
        # quantiles to avoid using future observations.
        low_threshold = (
            volatility
            .expanding(
                min_periods=return_period
            )
            .quantile(
                low_quantile
            )
        )

        high_threshold = (
            volatility
            .expanding(
                min_periods=return_period
            )
            .quantile(
                high_quantile
            )
        )

        regime = np.where(
            volatility <= low_threshold,
            "LOW",
            np.where(
                volatility >= high_threshold,
                "HIGH",
                "NORMAL",
            ),
        )

        regime = pd.Series(
            regime,
            index=result.index,
        )

        regime.loc[
            volatility.isna()
        ] = "UNKNOWN"

        result[
            "Volatility_Regime"
        ] = regime

        result[
            "Regime_Volatility"
        ] = volatility

        return result

    @classmethod
    def add_all_regimes(
        cls,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        result = cls.add_trend_regime(
            dataframe
        )

        result = cls.add_volatility_regime(
            result
        )

        result[
            "Combined_Regime"
        ] = (
            result[
                "Trend_Regime"
            ].astype(str)
            + "_"
            + result[
                "Volatility_Regime"
            ].astype(str)
        )

        return result

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------

    def evaluate(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray,
        timestamps: pd.DatetimeIndex,
        regime_data: pd.DataFrame,
        probability: np.ndarray | None = None,
        final_holdout_used: bool = False,
    ) -> RegimeResearchResult:

        self._validate_predictions(
            actuals,
            predictions,
            timestamps,
        )

        if not isinstance(
            regime_data,
            pd.DataFrame,
        ):
            raise TypeError(
                "regime_data must be a DataFrame."
            )

        if not isinstance(
            regime_data.index,
            pd.DatetimeIndex,
        ):
            raise TypeError(
                "regime_data must use a DatetimeIndex."
            )

        self._validate_regime_columns(
            regime_data,
            [
                "Trend_Regime",
                "Volatility_Regime",
                "Combined_Regime",
            ],
        )

        if not (
            regime_data.index.equals(
                timestamps
            )
        ):
            raise ValueError(
                "regime_data index must exactly match "
                "prediction timestamps."
            )

        if final_holdout_used:
            raise ValueError(
                "Final holdout data must not be used "
                "for regime-based model development."
            )

        if probability is not None:
            probability = np.asarray(
                probability,
                dtype=float,
            )

            if len(probability) != len(
                predictions
            ):
                raise ValueError(
                    "Probability length must match predictions."
                )

            if not np.isfinite(
                probability
            ).all():
                raise ValueError(
                    "Probability contains non-finite values."
                )

            if (
                (probability < 0).any()
                or (probability > 1).any()
            ):
                raise ValueError(
                    "Probability must be between 0 and 1."
                )

        evaluation = regime_data.copy()

        evaluation[
            "Actual"
        ] = np.asarray(
            actuals
        )

        evaluation[
            "Prediction"
        ] = np.asarray(
            predictions
        )

        if probability is not None:
            evaluation[
                "Probability"
            ] = probability

        report = build_regime_report(
            evaluation
        )

        accuracies: list[float] = []

        for section in (
            getattr(
                report,
                "results",
                [],
            )
        ):
            accuracy = getattr(
                section,
                "accuracy",
                None,
            )

            if accuracy is not None:
                accuracies.append(
                    float(accuracy)
                )

        if accuracies:
            minimum_accuracy = float(
                min(accuracies)
            )

            maximum_accuracy = float(
                max(accuracies)
            )

            accuracy_std = float(
                np.std(
                    accuracies
                )
            )
        else:
            minimum_accuracy = None
            maximum_accuracy = None
            accuracy_std = None

        stability_score = self._calculate_stability_score(
            accuracies
        )

        maximum_accuracy_std = getattr(
            self.config,
            "maximum_accuracy_std",
            0.15,
        )

        minimum_accuracy_gate = getattr(
            self.config,
            "minimum_accuracy",
            0.55,
        )

        passed = (
            minimum_accuracy is not None
            and accuracy_std is not None
            and minimum_accuracy
            >= minimum_accuracy_gate
            and accuracy_std
            <= maximum_accuracy_std
        )

        warnings: list[str] = []

        if minimum_accuracy is None:
            warnings.append(
                "No valid regime groups contained enough observations."
            )

        if (
            minimum_accuracy is not None
            and minimum_accuracy
            < minimum_accuracy_gate
        ):
            warnings.append(
                "At least one regime failed the minimum "
                "accuracy requirement."
            )

        if (
            accuracy_std is not None
            and accuracy_std
            > maximum_accuracy_std
        ):
            warnings.append(
                "Regime accuracy is unstable across regimes."
            )

        return RegimeResearchResult(
            report=report,
            total_samples=len(
                actuals
            ),
            trend_regimes=sorted(
                evaluation[
                    "Trend_Regime"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            ),
            volatility_regimes=sorted(
                evaluation[
                    "Volatility_Regime"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            ),
            combined_regimes=sorted(
                evaluation[
                    "Combined_Regime"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            ),
            minimum_regime_accuracy=(
                minimum_accuracy
            ),
            maximum_regime_accuracy=(
                maximum_accuracy
            ),
            regime_accuracy_std=(
                accuracy_std
            ),
            stability_score=stability_score,
            passed_stability_gate=passed,
            final_holdout_used=False,
            metadata={
                "evaluation_only": True,
                "model_fitted": False,
                "final_holdout_used": False,
                "future_regime_information_used": False,
                "research_only": True,
            },
            warnings=warnings,
        )

    # -----------------------------------------------------------------
    # Stability
    # -----------------------------------------------------------------

    def _calculate_stability_score(
        self,
        accuracies: list[float],
    ) -> float:

        if not accuracies:
            return 0.0

        minimum_accuracy = min(
            accuracies
        )

        accuracy_std = float(
            np.std(
                accuracies
            )
        )

        minimum_required = getattr(
            self.config,
            "minimum_accuracy",
            0.55,
        )

        maximum_std = getattr(
            self.config,
            "maximum_accuracy_std",
            0.15,
        )

        if minimum_required >= 1.0:
            accuracy_component = (
                1.0
                if minimum_accuracy
                >= minimum_required
                else 0.0
            )
        else:
            accuracy_component = (
                minimum_accuracy
                - 0.5
            ) / (
                minimum_required
                - 0.5
            )

        accuracy_component = float(
            np.clip(
                accuracy_component,
                0.0,
                1.0,
            )
        )

        if maximum_std == 0:
            stability_component = (
                1.0
                if accuracy_std == 0
                else 0.0
            )
        else:
            stability_component = 1.0 - (
                accuracy_std
                / maximum_std
            )

        stability_component = float(
            np.clip(
                stability_component,
                0.0,
                1.0,
            )
        )

        return float(
            0.5 * accuracy_component
            + 0.5 * stability_component
        )


# ---------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------


def run_regime_research(
    actuals: np.ndarray,
    predictions: np.ndarray,
    timestamps: pd.DatetimeIndex,
    regime_data: pd.DataFrame,
    probability: np.ndarray | None = None,
    config: RegimeValidationConfig | None = None,
) -> RegimeResearchResult:

    engine = RegimeResearchEngine(
        config=config
    )

    return engine.evaluate(
        actuals=actuals,
        predictions=predictions,
        timestamps=timestamps,
        regime_data=regime_data,
        probability=probability,
        final_holdout_used=False,
    )


__all__ = [
    "RegimeResearchResult",
    "RegimeResearchEngine",
    "run_regime_research",
]
