"""
AI Swing Analyser — Research Backtest Engine.

Runs research-only trading evaluation using already-generated
out-of-sample predictions.

Important principles:

1. Predictions must be generated without using future information.
2. Trades are entered only after the prediction becomes available.
3. Transaction costs and slippage must be included.
4. The final holdout must not be used for strategy development.
5. Backtest results are evidence, not automatic production approval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.models.backtest import (
    BacktestConfig,
    BacktestResult,
    run_backtest,
)


# ---------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------


@dataclass
class BacktestResearchResult:
    """
    Research backtest result.

    The underlying BacktestResult contains the detailed trading
    statistics and trade/equity information.
    """

    backtest: BacktestResult

    samples: int

    prediction_start: pd.Timestamp
    prediction_end: pd.Timestamp

    probability_threshold: float

    transaction_cost: float
    slippage: float

    final_holdout_used: bool

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    warnings: list[str] = field(
        default_factory=list
    )

    @property
    def metrics(self):
        """
        Return underlying trading metrics.
        """
        return getattr(
            self.backtest,
            "metrics",
            self.backtest,
        )

    @property
    def total_return(self) -> float | None:
        return self._metric(
            "total_return"
        )

    @property
    def sharpe_ratio(self) -> float | None:
        return self._metric(
            "sharpe_ratio"
        )

    @property
    def max_drawdown(self) -> float | None:
        return self._metric(
            "max_drawdown"
        )

    @property
    def profit_factor(self) -> float | None:
        return self._metric(
            "profit_factor"
        )

    @property
    def win_rate(self) -> float | None:
        return self._metric(
            "win_rate"
        )

    @property
    def trade_count(self) -> int | None:
        value = self._metric(
            "trades"
        )

        if value is None:
            value = self._metric(
                "trade_count"
            )

        if value is None:
            return None

        return int(value)

    def _metric(
        self,
        name: str,
    ):
        metrics = self.metrics

        if isinstance(
            metrics,
            dict,
        ):
            return metrics.get(
                name
            )

        return getattr(
            metrics,
            name,
            None,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "prediction_start": (
                self.prediction_start
            ),
            "prediction_end": (
                self.prediction_end
            ),
            "probability_threshold": (
                self.probability_threshold
            ),
            "transaction_cost": (
                self.transaction_cost
            ),
            "slippage": (
                self.slippage
            ),
            "total_return": (
                self.total_return
            ),
            "sharpe_ratio": (
                self.sharpe_ratio
            ),
            "max_drawdown": (
                self.max_drawdown
            ),
            "profit_factor": (
                self.profit_factor
            ),
            "win_rate": (
                self.win_rate
            ),
            "trade_count": (
                self.trade_count
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


class BacktestResearchEngine:
    """
    Research-only backtesting engine.

    This class does not train models.
    """

    REQUIRED_OHLCV = (
        "Open",
        "High",
        "Low",
        "Close",
    )

    def __init__(
        self,
        config: BacktestConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else BacktestConfig()
        )

        self._validate_config()

    # -----------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------

    def _validate_config(self) -> None:

        threshold = getattr(
            self.config,
            "probability_threshold",
            0.60,
        )

        if not (
            0.0
            < threshold
            < 1.0
        ):
            raise ValueError(
                "probability_threshold must be between 0 and 1."
            )

        transaction_cost = getattr(
            self.config,
            "transaction_cost",
            None,
        )
        if transaction_cost is None:
            transaction_cost = getattr(
                self.config,
                "transaction_cost_pct",
                0.0,
            )

        slippage = getattr(
            self.config,
            "slippage",
            None,
        )
        if slippage is None:
            slippage = getattr(
                self.config,
                "slippage_pct",
                0.0,
            )

        if transaction_cost < 0:
            raise ValueError(
                "transaction_cost cannot be negative."
            )

        if slippage < 0:
            raise ValueError(
                "slippage cannot be negative."
            )

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    @classmethod
    def _validate_market_data(
        cls,
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
                "market_data must not be empty."
            )

        missing = [
            column
            for column in cls.REQUIRED_OHLCV
            if column not in market_data.columns
        ]

        if missing:
            raise ValueError(
                "Missing OHLCV columns: "
                + ", ".join(missing)
            )

        if not isinstance(
            market_data.index,
            pd.DatetimeIndex,
        ):
            raise TypeError(
                "market_data must use DatetimeIndex."
            )

        if market_data.index.has_duplicates:
            raise ValueError(
                "market_data contains duplicate timestamps."
            )

        if not market_data.index.is_monotonic_increasing:
            raise ValueError(
                "market_data must be chronological."
            )

        values = market_data[
            list(cls.REQUIRED_OHLCV)
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        if not np.isfinite(
            values.to_numpy(
                dtype=float
            )
        ).all():
            raise ValueError(
                "market_data contains non-finite OHLC values."
            )

        if (
            values["Open"] <= 0
        ).any() or (
            values["High"] <= 0
        ).any() or (
            values["Low"] <= 0
        ).any() or (
            values["Close"] <= 0
        ).any():
            raise ValueError(
                "OHLC prices must be positive."
            )

        if (
            values["High"]
            < values[
                ["Open", "Close"]
            ].max(axis=1)
        ).any():
            raise ValueError(
                "High is below Open or Close."
            )

        if (
            values["Low"]
            > values[
                ["Open", "Close"]
            ].min(axis=1)
        ).any():
            raise ValueError(
                "Low is above Open or Close."
            )

        if (
            values["High"]
            < values["Low"]
        ).any():
            raise ValueError(
                "High cannot be below Low."
            )

    @staticmethod
    def _validate_predictions(
        probabilities: np.ndarray,
        timestamps: pd.DatetimeIndex,
    ) -> None:

        probabilities = np.asarray(
            probabilities,
            dtype=float,
        )

        if len(probabilities) == 0:
            raise ValueError(
                "Predictions must not be empty."
            )

        if len(probabilities) != len(
            timestamps
        ):
            raise ValueError(
                "Predictions and timestamps must have "
                "the same length."
            )

        if not np.isfinite(
            probabilities
        ).all():
            raise ValueError(
                "Predictions contain non-finite probabilities."
            )

        if (
            (probabilities < 0).any()
            or (probabilities > 1).any()
        ):
            raise ValueError(
                "Probabilities must be between 0 and 1."
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
                "Prediction timestamps contain duplicates."
            )

        if not timestamps.is_monotonic_increasing:
            raise ValueError(
                "Prediction timestamps must be chronological."
            )

    # -----------------------------------------------------------------
    # Prediction alignment
    # -----------------------------------------------------------------

    @staticmethod
    def align_predictions(
        market_data: pd.DataFrame,
        probabilities: np.ndarray,
        prediction_timestamps: pd.DatetimeIndex,
    ) -> pd.DataFrame:

        probabilities = np.asarray(
            probabilities,
            dtype=float,
        )

        prediction_timestamps = (
            pd.DatetimeIndex(
                prediction_timestamps
            )
        )

        BacktestResearchEngine._validate_market_data(
            market_data
        )

        BacktestResearchEngine._validate_predictions(
            probabilities,
            prediction_timestamps,
        )

        prediction_frame = pd.DataFrame(
            {
                "Probability": probabilities,
            },
            index=prediction_timestamps,
        )

        # Exact timestamp alignment is intentional.
        #
        # We do NOT forward-fill predictions onto future bars because
        # doing so can accidentally turn a prediction into information
        # available before the original prediction timestamp.
        aligned = market_data.join(
            prediction_frame,
            how="left",
        )

        return aligned

    # -----------------------------------------------------------------
    # Main backtest
    # -----------------------------------------------------------------

    def run(
        self,
        market_data: pd.DataFrame,
        probabilities: np.ndarray,
        prediction_timestamps: pd.DatetimeIndex,
        final_holdout_used: bool = False,
    ) -> BacktestResearchResult:

        self._validate_market_data(
            market_data
        )

        self._validate_predictions(
            probabilities,
            prediction_timestamps,
        )

        if final_holdout_used:
            raise ValueError(
                "Final holdout data must not be used "
                "for research backtesting."
            )

        if (
            not prediction_timestamps.isin(
                market_data.index
            ).all()
        ):
            raise ValueError(
                "Every prediction timestamp must exist in market_data."
            )

        aligned = self.align_predictions(
            market_data=market_data,
            probabilities=probabilities,
            prediction_timestamps=prediction_timestamps,
        )

        # Keep only bars where an out-of-sample prediction exists.
        backtest_data = aligned.loc[
            aligned["Probability"].notna()
        ].copy()

        if backtest_data.empty:
            raise ValueError(
                "No prediction observations remain after alignment."
            )

        if len(backtest_data) < 2:
            raise ValueError(
                "At least two observations are required for a backtest."
            )

        # -------------------------------------------------------------
        # The underlying backtest engine is responsible for execution
        # mechanics such as next-bar entry, stop loss, take profit,
        # position sizing, costs and slippage.
        # -------------------------------------------------------------

        result = run_backtest(
            backtest_data,
            config=self.config,
        )

        warnings: list[str] = []

        if len(backtest_data) < 100:
            warnings.append(
                "Backtest contains fewer than 100 prediction observations."
            )

        trade_count = self._extract_trade_count(
            result
        )

        if trade_count is not None and trade_count < 30:
            warnings.append(
                "Backtest contains fewer than 30 trades; "
                "performance estimates may be statistically unstable."
            )

        return BacktestResearchResult(
            backtest=result,
            samples=len(
                backtest_data
            ),
            prediction_start=(
                backtest_data.index.min()
            ),
            prediction_end=(
                backtest_data.index.max()
            ),
            probability_threshold=getattr(
                self.config,
                "probability_threshold",
                0.60,
            ),
            transaction_cost=getattr(
                self.config,
                "transaction_cost",
                0.0,
            ),
            slippage=getattr(
                self.config,
                "slippage",
                0.0,
            ),
            final_holdout_used=False,
            metadata={
                "evaluation_only": True,
                "model_fitted": False,
                "final_holdout_used": False,
                "prediction_alignment": "exact_timestamp",
                "forward_fill_predictions": False,
                "research_only": True,
            },
            warnings=warnings,
        )

    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------

    @staticmethod
    def _extract_trade_count(
        result: BacktestResult,
    ) -> int | None:

        metrics = getattr(
            result,
            "metrics",
            result,
        )

        if isinstance(
            metrics,
            dict,
        ):
            value = metrics.get(
                "trades"
            )

            if value is None:
                value = metrics.get(
                    "trade_count"
                )

            return (
                None
                if value is None
                else int(value)
            )

        value = getattr(
            metrics,
            "trades",
            None,
        )

        if value is None:
            value = getattr(
                metrics,
                "trade_count",
                None,
            )

        return (
            None
            if value is None
            else int(value)
        )


# ---------------------------------------------------------------------
# Robustness helpers
# ---------------------------------------------------------------------


def calculate_return_distribution(
    trade_returns: np.ndarray,
) -> dict[str, float]:

    returns = np.asarray(
        trade_returns,
        dtype=float,
    )

    if len(returns) == 0:
        raise ValueError(
            "trade_returns must not be empty."
        )

    if not np.isfinite(
        returns
    ).all():
        raise ValueError(
            "trade_returns contains non-finite values."
        )

    return {
        "mean": float(
            np.mean(returns)
        ),
        "median": float(
            np.median(returns)
        ),
        "std": float(
            np.std(returns)
        ),
        "minimum": float(
            np.min(returns)
        ),
        "maximum": float(
            np.max(returns)
        ),
        "p05": float(
            np.percentile(
                returns,
                5,
            )
        ),
        "p25": float(
            np.percentile(
                returns,
                25,
            )
        ),
        "p75": float(
            np.percentile(
                returns,
                75,
            )
        ),
        "p95": float(
            np.percentile(
                returns,
                95,
            )
        ),
    }


def stress_trade_returns(
    trade_returns: np.ndarray,
    additional_cost: float = 0.0,
) -> np.ndarray:

    returns = np.asarray(
        trade_returns,
        dtype=float,
    )

    if not np.isfinite(
        returns
    ).all():
        raise ValueError(
            "trade_returns contains non-finite values."
        )

    if additional_cost < 0:
        raise ValueError(
            "additional_cost cannot be negative."
        )

    return (
        returns
        - additional_cost
    )


__all__ = [
    "BacktestResearchResult",
    "BacktestResearchEngine",
    "calculate_return_distribution",
    "stress_trade_returns",
]
