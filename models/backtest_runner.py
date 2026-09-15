"""
Walk-forward trading backtest runner.

Purpose
-------
Connect out-of-sample model predictions to a realistic trading simulation.

The runner is intentionally conservative:

    historical data
          ↓
    chronological folds
          ↓
    train model on past
          ↓
    predict future
          ↓
    generate trade permission
          ↓
    execute on next available bar
          ↓
    include costs + slippage
          ↓
    calculate equity / drawdown / returns

IMPORTANT
---------
This module is a research backtester.

It must never be used to manufacture an attractive historical result.

Predictions must be generated strictly out-of-sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .backtest import (
    BacktestConfig,
    BacktestResult,
    Trade,
    run_backtest,
)
from .preprocessing import (
    PreprocessorConfig,
    SafePreprocessor,
)
from .splitter import purged_walk_forward_splits


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass
class BacktestRunnerConfig:
    """
    Configuration for walk-forward backtesting.
    """

    n_splits: int = 5

    train_size: Optional[int] = None

    validation_size: Optional[int] = None

    gap: int = 0

    embargo: int = 0

    expanding: bool = True

    minimum_train_samples: int = 100

    minimum_validation_samples: int = 30

    probability_threshold: float = 0.60

    long_enabled: bool = True

    short_enabled: bool = False

    use_next_open: bool = True

    random_state: int = 42

    def __post_init__(self) -> None:
        if self.n_splits < 1:
            raise ValueError(
                "n_splits must be positive."
            )

        if self.train_size is not None:
            if self.train_size < 1:
                raise ValueError(
                    "train_size must be positive."
                )

        if self.validation_size is not None:
            if self.validation_size < 1:
                raise ValueError(
                    "validation_size must be positive."
                )

        if self.gap < 0:
            raise ValueError(
                "gap cannot be negative."
            )

        if self.embargo < 0:
            raise ValueError(
                "embargo cannot be negative."
            )

        if self.minimum_train_samples < 1:
            raise ValueError(
                "minimum_train_samples must be positive."
            )

        if self.minimum_validation_samples < 1:
            raise ValueError(
                "minimum_validation_samples must be positive."
            )

        if not 0.0 < self.probability_threshold < 1.0:
            raise ValueError(
                "probability_threshold must be between 0 and 1."
            )


# ----------------------------------------------------------------------
# Fold result
# ----------------------------------------------------------------------


@dataclass
class BacktestFoldResult:
    """
    Result for one walk-forward backtest fold.
    """

    fold_number: int

    train_start: pd.Timestamp

    train_end: pd.Timestamp

    validation_start: pd.Timestamp

    validation_end: pd.Timestamp

    train_samples: int

    validation_samples: int

    predictions: pd.DataFrame

    backtest_result: Optional[
        BacktestResult
    ]

    passed_temporal_check: bool

    notes: list[str] = field(
        default_factory=list
    )


# ----------------------------------------------------------------------
# Complete result
# ----------------------------------------------------------------------


@dataclass
class WalkForwardBacktestResult:
    """
    Complete walk-forward backtest result.
    """

    folds: list[
        BacktestFoldResult
    ]

    combined_predictions: pd.DataFrame

    combined_trades: pd.DataFrame

    equity_curve: pd.DataFrame

    metrics: Dict[str, Any]

    passed_temporal_checks: bool

    notes: list[str] = field(
        default_factory=list
    )

    def fold_table(self) -> pd.DataFrame:
        """Return fold-level performance."""

        rows = []

        for fold in self.folds:
            result = fold.backtest_result

            row = {
                "fold": fold.fold_number,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "validation_start": (
                    fold.validation_start
                ),
                "validation_end": (
                    fold.validation_end
                ),
                "train_samples": (
                    fold.train_samples
                ),
                "validation_samples": (
                    fold.validation_samples
                ),
                "temporal_check": (
                    fold.passed_temporal_check
                ),
            }

            if result is not None:
                row.update(
                    self._extract_backtest_metrics(
                        result
                    )
                )

            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def _extract_backtest_metrics(
        result: BacktestResult,
    ) -> Dict[str, Any]:
        metrics = getattr(
            result,
            "metrics",
            None,
        )

        if metrics is None:
            return {}

        if isinstance(
            metrics,
            dict,
        ):
            return dict(metrics)

        if hasattr(
            metrics,
            "__dataclass_fields__",
        ):
            return {
                name: getattr(
                    metrics,
                    name,
                )
                for name in metrics.__dataclass_fields__
            }

        return {
            key: value
            for key, value in vars(
                metrics
            ).items()
            if not key.startswith("_")
        }


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------


class WalkForwardBacktestRunner:
    """
    Leakage-aware walk-forward backtest engine.

    ``model_factory`` must create a completely fresh model.

    The prediction callback must use only information available at the
    prediction timestamp.
    """

    def __init__(
        self,
        config: Optional[
            BacktestRunnerConfig
        ] = None,
        backtest_config: Optional[
            BacktestConfig
        ] = None,
        preprocessor_config: Optional[
            PreprocessorConfig
        ] = None,
    ) -> None:
        self.config = (
            config
            or BacktestRunnerConfig()
        )

        self.backtest_config = (
            backtest_config
            or BacktestConfig(
                probability_threshold=(
                    self.config
                    .probability_threshold
                ),
                long_enabled=(
                    self.config.long_enabled
                ),
                short_enabled=(
                    self.config.short_enabled
                ),
            )
        )

        self.preprocessor_config = (
            preprocessor_config
            or PreprocessorConfig()
        )

    # ------------------------------------------------------------------
    # Main method
    # ------------------------------------------------------------------

    def run(
        self,
        data: pd.DataFrame,
        *,
        target_column: str,
        feature_columns: Sequence[str],
        model_factory: Callable[[], Any],
        probability_function: Optional[
            Callable[
                [Any, Any],
                np.ndarray,
            ]
        ] = None,
        price_columns: Sequence[str] = (
            "Open",
            "High",
            "Low",
            "Close",
        ),
    ) -> WalkForwardBacktestResult:
        """
        Run a walk-forward model/trading backtest.

        Parameters
        ----------
        data:
            Chronological OHLCV + feature + target data.

        target_column:
            Historical training target.

        feature_columns:
            Features used by the model.

        model_factory:
            Function returning a new unfitted model for every fold.

        probability_function:
            Optional custom probability prediction function.

        price_columns:
            Required OHLC columns used by the trading simulator.
        """

        frame = self._prepare_data(
            data=data,
            target_column=target_column,
            feature_columns=feature_columns,
            price_columns=price_columns,
        )

        splits = self._create_splits(
            frame
        )

        folds: list[
            BacktestFoldResult
        ] = []

        all_predictions = []
        all_trades = []
        all_equity = []

        temporal_checks = []

        for fold_number, (
            train_indices,
            validation_indices,
        ) in enumerate(
            splits,
            start=1,
        ):
            train = frame.iloc[
                train_indices
            ]

            validation = frame.iloc[
                validation_indices
            ]

            # ----------------------------------------------------------
            # Temporal safety
            # ----------------------------------------------------------

            temporal_ok = (
                train.index.max()
                < validation.index.min()
            )

            temporal_checks.append(
                temporal_ok
            )

            if not temporal_ok:
                raise RuntimeError(
                    "Temporal leakage detected in backtest fold."
                )

            # ----------------------------------------------------------
            # Features / target
            # ----------------------------------------------------------

            X_train = train[
                list(feature_columns)
            ]

            X_validation = validation[
                list(feature_columns)
            ]

            y_train = train[
                target_column
            ]

            # ----------------------------------------------------------
            # Fresh preprocessing
            # ----------------------------------------------------------

            preprocessor = SafePreprocessor(
                self.preprocessor_config
            )

            X_train_transformed = (
                preprocessor.fit_transform(
                    X_train
                )
            )

            X_validation_transformed = (
                preprocessor.transform(
                    X_validation
                )
            )

            # ----------------------------------------------------------
            # Fresh model
            # ----------------------------------------------------------

            model = model_factory()

            if not hasattr(
                model,
                "fit",
            ):
                raise TypeError(
                    "model_factory must return an object with fit()."
                )

            model.fit(
                X_train_transformed,
                y_train,
            )

            # ----------------------------------------------------------
            # Out-of-sample prediction
            # ----------------------------------------------------------

            if probability_function is not None:
                probability = np.asarray(
                    probability_function(
                        model,
                        X_validation_transformed,
                    ),
                    dtype=float,
                ).reshape(-1)
            else:
                probability = (
                    self._predict_probability(
                        model,
                        X_validation_transformed,
                    )
                )

            if len(probability) != len(
                validation
            ):
                raise ValueError(
                    "Probability prediction length does not match "
                    "validation data."
                )

            probability = np.clip(
                probability,
                0.0,
                1.0,
            )

            direction = (
                probability
                >= self.config
                .probability_threshold
            ).astype(int)

            # ----------------------------------------------------------
            # Build prediction frame
            # ----------------------------------------------------------

            predictions = validation[
                list(price_columns)
            ].copy()

            predictions[
                "Probability_Up"
            ] = probability

            predictions[
                "Predicted_Direction"
            ] = direction

            predictions[
                "Trade_Allowed"
            ] = (
                (
                    probability
                    >= self.config
                    .probability_threshold
                )
                & (
                    self.config.long_enabled
                )
            )

            predictions[
                "Fold"
            ] = fold_number

            # ----------------------------------------------------------
            # Run historical trade simulation
            # ----------------------------------------------------------

            backtest_result = self._run_fold_backtest(
                validation,
                predictions,
            )

            trades = self._extract_trades(
                backtest_result
            )

            equity = self._extract_equity(
                backtest_result
            )

            all_predictions.append(
                predictions
            )

            if not trades.empty:
                all_trades.append(
                    trades
                )

            if not equity.empty:
                all_equity.append(
                    equity
                )

            notes = [
                (
                    "Model and preprocessing were fitted "
                    "only on the training fold."
                ),
                (
                    "Predictions are out-of-sample "
                    "for this validation fold."
                ),
            ]

            folds.append(
                BacktestFoldResult(
                    fold_number=fold_number,
                    train_start=train.index.min(),
                    train_end=train.index.max(),
                    validation_start=(
                        validation.index.min()
                    ),
                    validation_end=(
                        validation.index.max()
                    ),
                    train_samples=len(train),
                    validation_samples=len(
                        validation
                    ),
                    predictions=predictions,
                    backtest_result=(
                        backtest_result
                    ),
                    passed_temporal_check=(
                        temporal_ok
                    ),
                    notes=notes,
                )
            )

        combined_predictions = (
            pd.concat(
                all_predictions,
                axis=0,
            )
            if all_predictions
            else pd.DataFrame()
        )

        combined_predictions = (
            combined_predictions[
                ~combined_predictions.index.duplicated(
                    keep="first"
                )
            ]
            .sort_index()
        )

        combined_trades = (
            pd.concat(
                all_trades,
                axis=0,
                ignore_index=True,
            )
            if all_trades
            else pd.DataFrame()
        )

        equity_curve = self._combine_equity(
            all_equity
        )

        metrics = self._aggregate_metrics(
            folds
        )

        notes = [
            (
                "Walk-forward predictions were generated "
                "strictly out-of-sample."
            ),
            (
                "Transaction costs and slippage are included "
                "through the backtest configuration."
            ),
            (
                "The final holdout must remain separate from "
                "this research backtest."
            ),
        ]

        return WalkForwardBacktestResult(
            folds=folds,
            combined_predictions=(
                combined_predictions
            ),
            combined_trades=(
                combined_trades
            ),
            equity_curve=equity_curve,
            metrics=metrics,
            passed_temporal_checks=all(
                temporal_checks
            ),
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Fold backtest
    # ------------------------------------------------------------------

    def _run_fold_backtest(
        self,
        validation: pd.DataFrame,
        predictions: pd.DataFrame,
    ) -> BacktestResult:
        """
        Convert model predictions into the format expected by the
        historical backtest engine.
        """

        frame = validation.copy()

        frame[
            "Probability_Up"
        ] = predictions[
            "Probability_Up"
        ].to_numpy()

        frame[
            "Predicted_Direction"
        ] = predictions[
            "Predicted_Direction"
        ].to_numpy()

        frame[
            "Trade_Allowed"
        ] = predictions[
            "Trade_Allowed"
        ].to_numpy()

        return run_backtest(
            frame,
            config=self.backtest_config,
        )

    # ------------------------------------------------------------------
    # Splits
    # ------------------------------------------------------------------

    def _create_splits(
        self,
        frame: pd.DataFrame,
    ):
        splits = purged_walk_forward_splits(
            frame,
            n_splits=self.config.n_splits,
            train_size=self.config.train_size,
            test_size=self.config.validation_size,
            gap=self.config.gap,
            embargo=self.config.embargo,
            expanding=self.config.expanding,
        )

        valid = []

        for train_indices, validation_indices in splits:
            train_indices = np.asarray(
                train_indices,
                dtype=int,
            )

            validation_indices = np.asarray(
                validation_indices,
                dtype=int,
            )

            if len(train_indices) < (
                self.config.minimum_train_samples
            ):
                continue

            if len(validation_indices) < (
                self.config.minimum_validation_samples
            ):
                continue

            valid.append(
                (
                    train_indices,
                    validation_indices,
                )
            )

        if not valid:
            raise ValueError(
                "No valid walk-forward backtest folds were produced."
            )

        return valid

    # ------------------------------------------------------------------
    # Data validation
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_data(
        *,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: Sequence[str],
        price_columns: Sequence[str],
    ) -> pd.DataFrame:
        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        if not isinstance(
            data.index,
            pd.DatetimeIndex,
        ):
            raise TypeError(
                "Backtest data requires a DatetimeIndex."
            )

        required = set(
            price_columns
        )
        required.add(
            target_column
        )
        required.update(
            feature_columns
        )

        missing = [
            column
            for column in required
            if column not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        frame = data.copy().sort_index()

        if frame.index.has_duplicates:
            raise ValueError(
                "Duplicate timestamps detected."
            )

        suspicious = [
            feature
            for feature in feature_columns
            if (
                "future" in feature.lower()
                or "target" in feature.lower()
                or "direction" in feature.lower()
            )
        ]

        if suspicious:
            raise ValueError(
                "Potential target leakage in backtest features: "
                f"{suspicious}"
            )

        # Do not silently create fake historical prices.
        frame = frame.dropna(
            subset=list(
                required
            )
        )

        if frame.empty:
            raise ValueError(
                "No usable rows remain after removing missing values."
            )

        # Validate OHLC relationships.
        high = frame[
            "High"
        ].to_numpy(dtype=float)

        low = frame[
            "Low"
        ].to_numpy(dtype=float)

        open_price = frame[
            "Open"
        ].to_numpy(dtype=float)

        close = frame[
            "Close"
        ].to_numpy(dtype=float)

        if not (
            np.isfinite(high).all()
            and np.isfinite(low).all()
            and np.isfinite(open_price).all()
            and np.isfinite(close).all()
        ):
            raise ValueError(
                "OHLC data contains invalid values."
            )

        if (
            high < low
        ).any():
            raise ValueError(
                "High price cannot be below low price."
            )

        if (
            open_price > high
        ).any() or (
            open_price < low
        ).any():
            raise ValueError(
                "Open price falls outside High/Low."
            )

        if (
            close > high
        ).any() or (
            close < low
        ).any():
            raise ValueError(
                "Close price falls outside High/Low."
            )

        return frame

    # ------------------------------------------------------------------
    # Probability helper
    # ------------------------------------------------------------------

    @staticmethod
    def _predict_probability(
        model: Any,
        X: Any,
    ) -> np.ndarray:
        if not hasattr(
            model,
            "predict_proba",
        ):
            raise TypeError(
                "Backtest model must expose predict_proba()."
            )

        matrix = np.asarray(
            model.predict_proba(X),
            dtype=float,
        )

        if matrix.ndim != 2:
            raise ValueError(
                "predict_proba() must return a 2D array."
            )

        if matrix.shape[1] < 2:
            raise ValueError(
                "Binary classifier must provide two class probabilities."
            )

        return matrix[:, 1]

    # ------------------------------------------------------------------
    # Trade extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_trades(
        result: BacktestResult,
    ) -> pd.DataFrame:
        trades = getattr(
            result,
            "trades",
            None,
        )

        if trades is None:
            return pd.DataFrame()

        if isinstance(
            trades,
            pd.DataFrame,
        ):
            return trades.copy()

        if isinstance(
            trades,
            list,
        ):
            records = []

            for trade in trades:
                if isinstance(
                    trade,
                    dict,
                ):
                    records.append(
                        trade
                    )

                elif hasattr(
                    trade,
                    "__dataclass_fields__",
                ):
                    records.append(
                        {
                            field_name: getattr(
                                trade,
                                field_name,
                            )
                            for field_name in (
                                trade.__dataclass_fields__
                            )
                        }

                    )

            return pd.DataFrame(
                records
            )

        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Equity extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_equity(
        result: BacktestResult,
    ) -> pd.DataFrame:
        equity = getattr(
            result,
            "equity_curve",
            None,
        )

        if equity is None:
            return pd.DataFrame()

        if isinstance(
            equity,
            pd.DataFrame,
        ):
            return equity.copy()

        if isinstance(
            equity,
            pd.Series,
        ):
            return equity.to_frame(
                "Equity"
            )

        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Equity combination
    # ------------------------------------------------------------------

    @staticmethod
    def _combine_equity(
        curves: Sequence[pd.DataFrame],
    ) -> pd.DataFrame:
        valid = [
            curve
            for curve in curves
            if curve is not None
            and not curve.empty
        ]

        if not valid:
            return pd.DataFrame()

        combined = pd.concat(
            valid,
            axis=0,
        )

        combined = (
            combined[
                ~combined.index.duplicated(
                    keep="last"
                )
            ]
            .sort_index()
        )

        return combined

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    def _aggregate_metrics(
        self,
        folds: Sequence[
            BacktestFoldResult
        ],
    ) -> Dict[str, Any]:
        values = []

        for fold in folds:
            result = fold.backtest_result

            if result is None:
                continue

            metrics = getattr(
                result,
                "metrics",
                None,
            )

            if metrics is None:
                continue

            if isinstance(
                metrics,
                dict,
            ):
                values.append(
                    metrics
                )

            elif hasattr(
                metrics,
                "__dataclass_fields__",
            ):
                values.append(
                    {
                        name: getattr(
                            metrics,
                            name,
                        )
                        for name in (
                            metrics.__dataclass_fields__
                        )
                    }
                )

            else:
                values.append(
                    {
                        key: value
                        for key, value in vars(
                            metrics
                        ).items()
                        if not key.startswith("_")
                    }
                )

        if not values:
            return {}

        numeric_keys = set()

        for record in values:
            for key, value in record.items():
                if isinstance(
                    value,
                    (
                        int,
                        float,
                        np.integer,
                        np.floating,
                    ),
                ):
                    numeric_keys.add(
                        key
                    )

        aggregate = {}

        for key in sorted(
            numeric_keys
        ):
            numeric_values = []

            for record in values:
                value = record.get(
                    key
                )

                if value is None:
                    continue

                try:
                    value = float(
                        value
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                if np.isfinite(value):
                    numeric_values.append(
                        value
                    )

            if numeric_values:
                aggregate[
                    f"{key}_mean"
                ] = float(
                    np.mean(
                        numeric_values
                    )
                )

                aggregate[
                    f"{key}_median"
                ] = float(
                    np.median(
                        numeric_values
                    )
                )

                aggregate[
                    f"{key}_min"
                ] = float(
                    np.min(
                        numeric_values
                    )
                )

        aggregate[
            "fold_count"
        ] = len(values)

        return aggregate


# ----------------------------------------------------------------------
# Convenience function
# ----------------------------------------------------------------------


def run_walk_forward_backtest(
    data: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: Sequence[str],
    model_factory: Callable[[], Any],
    probability_function: Optional[
        Callable[
            [Any, Any],
            np.ndarray,
        ]
    ] = None,
    config: Optional[
        BacktestRunnerConfig
    ] = None,
    backtest_config: Optional[
        BacktestConfig
    ] = None,
    preprocessor_config: Optional[
        PreprocessorConfig
    ] = None,
) -> WalkForwardBacktestResult:
    """
    Convenience wrapper around WalkForwardBacktestRunner.
    """

    runner = WalkForwardBacktestRunner(
        config=config,
        backtest_config=backtest_config,
        preprocessor_config=preprocessor_config,
    )

    return runner.run(
        data,
        target_column=target_column,
        feature_columns=feature_columns,
        model_factory=model_factory,
        probability_function=probability_function,
    )


def backtest_summary(
    result: WalkForwardBacktestResult,
) -> Dict[str, Any]:
    """Return compact walk-forward backtest information."""

    return {
        "fold_count": len(
            result.folds
        ),
        "temporal_checks_passed": (
            result.passed_temporal_checks
        ),
        "prediction_rows": len(
            result.combined_predictions
        ),
        "trade_count": len(
            result.combined_trades
        ),
        "metrics": result.metrics,
    }


__all__ = [
    "BacktestRunnerConfig",
    "BacktestFoldResult",
    "WalkForwardBacktestResult",
    "WalkForwardBacktestRunner",
    "run_walk_forward_backtest",
    "backtest_summary",
]
