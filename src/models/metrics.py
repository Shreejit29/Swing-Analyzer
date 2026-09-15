"""
Model evaluation metrics for AI Swing Analyser.

Provides leakage-safe evaluation metrics for:

    - Direction classification
    - Probability calibration
    - Return prediction
    - Price-range prediction
    - Trading-oriented performance
    - Confidence thresholds

IMPORTANT:
Metrics must be calculated on data that was not used to fit
the model being evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    """Container for classification evaluation."""

    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    log_loss: float
    brier_score: float
    samples: int


@dataclass
class RegressionMetrics:
    """Container for regression evaluation."""

    mae: float
    rmse: float
    mean_error: float
    median_absolute_error: float
    directional_accuracy: float
    samples: int


@dataclass
class RangeMetrics:
    """Container for prediction-range evaluation."""

    coverage: float
    average_width: float
    median_width: float
    lower_violation_rate: float
    upper_violation_rate: float
    samples: int


@dataclass
class TradingMetrics:
    """Container for simple strategy-oriented metrics."""

    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    maximum_drawdown: float
    win_rate: float
    profit_factor: float
    trades: int


def _validate_equal_length(
    *arrays: np.ndarray,
) -> None:
    """Ensure arrays have equal length."""

    lengths = [
        len(np.asarray(array))
        for array in arrays
    ]

    if len(set(lengths)) != 1:
        raise ValueError(
            f"Array lengths do not match: {lengths}"
        )


def _validate_binary_target(
    y_true: np.ndarray,
) -> np.ndarray:
    """Validate binary target."""

    y_true = np.asarray(
        y_true,
        dtype=int,
    ).reshape(-1)

    unique = np.unique(y_true)

    if not np.isin(
        unique,
        [0, 1],
    ).all():
        raise ValueError(
            "Classification target must contain only 0 and 1."
        )

    return y_true


def classification_metrics(
    y_true: np.ndarray,
    predicted_probability: np.ndarray,
    threshold: float = 0.50,
) -> ClassificationMetrics:
    """
    Calculate comprehensive binary classification metrics.

    Parameters
    ----------
    y_true:
        Actual 0/1 direction.

    predicted_probability:
        Probability of the positive/up class.

    threshold:
        Probability threshold used to convert probabilities
        into binary predictions.
    """

    y_true = _validate_binary_target(
        y_true
    )

    probability = np.asarray(
        predicted_probability,
        dtype=float,
    ).reshape(-1)

    _validate_equal_length(
        y_true,
        probability,
    )

    if not np.isfinite(
        probability
    ).all():
        raise ValueError(
            "Predicted probabilities contain non-finite values."
        )

    probability = np.clip(
        probability,
        0.0,
        1.0,
    )

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    prediction = (
        probability >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        prediction,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            prediction,
        )
    )

    precision = precision_score(
        y_true,
        prediction,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        prediction,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        prediction,
        zero_division=0,
    )

    if len(np.unique(y_true)) == 2:

        roc_auc = roc_auc_score(
            y_true,
            probability,
        )

    else:

        roc_auc = float("nan")

    ll = log_loss(
        y_true,
        probability,
        labels=[0, 1],
    )

    brier = float(
        np.mean(
            (
                probability
                - y_true
            ) ** 2
        )
    )

    return ClassificationMetrics(
        accuracy=float(accuracy),
        balanced_accuracy=float(
            balanced_accuracy
        ),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=float(roc_auc),
        log_loss=float(ll),
        brier_score=brier,
        samples=len(y_true),
    )


def confusion_matrix_dataframe(
    y_true: np.ndarray,
    predicted_probability: np.ndarray,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """Return confusion matrix as a labelled dataframe."""

    y_true = _validate_binary_target(
        y_true
    )

    probability = np.asarray(
        predicted_probability,
        dtype=float,
    ).reshape(-1)

    _validate_equal_length(
        y_true,
        probability,
    )

    prediction = (
        probability >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1],
    )

    return pd.DataFrame(
        matrix,
        index=[
            "Actual_Down",
            "Actual_Up",
        ],
        columns=[
            "Predicted_Down",
            "Predicted_Up",
        ],
    )


def regression_metrics(
    y_true: np.ndarray,
    prediction: np.ndarray,
) -> RegressionMetrics:
    """Evaluate future-return predictions."""

    y_true = np.asarray(
        y_true,
        dtype=float,
    ).reshape(-1)

    prediction = np.asarray(
        prediction,
        dtype=float,
    ).reshape(-1)

    _validate_equal_length(
        y_true,
        prediction,
    )

    if not np.isfinite(
        y_true
    ).all():
        raise ValueError(
            "y_true contains non-finite values."
        )

    if not np.isfinite(
        prediction
    ).all():
        raise ValueError(
            "prediction contains non-finite values."
        )

    errors = (
        prediction
        - y_true
    )

    directional_accuracy = float(
        np.mean(
            np.sign(prediction)
            == np.sign(y_true)
        )
    )

    return RegressionMetrics(
        mae=float(
            mean_absolute_error(
                y_true,
                prediction,
            )
        ),
        rmse=float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    prediction,
                )
            )
        ),
        mean_error=float(
            np.mean(errors)
        ),
        median_absolute_error=float(
            np.median(
                np.abs(errors)
            )
        ),
        directional_accuracy=(
            directional_accuracy
        ),
        samples=len(y_true),
    )


def range_metrics(
    actual_future_return: np.ndarray,
    predicted_lower_return: np.ndarray,
    predicted_upper_return: np.ndarray,
) -> RangeMetrics:
    """
    Evaluate prediction interval quality.

    Coverage:
        Fraction of observations where the actual future return
        falls inside the predicted interval.

    Width:
        Average size of the predicted interval.
    """

    actual = np.asarray(
        actual_future_return,
        dtype=float,
    ).reshape(-1)

    lower = np.asarray(
        predicted_lower_return,
        dtype=float,
    ).reshape(-1)

    upper = np.asarray(
        predicted_upper_return,
        dtype=float,
    ).reshape(-1)

    _validate_equal_length(
        actual,
        lower,
        upper,
    )

    if not (
        np.isfinite(actual).all()
        and np.isfinite(lower).all()
        and np.isfinite(upper).all()
    ):
        raise ValueError(
            "Range inputs contain non-finite values."
        )

    invalid_order = (
        lower > upper
    )

    if invalid_order.any():
        raise ValueError(
            "Predicted lower bound exceeds upper bound."
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

    width = (
        upper
        - lower
    )

    return RangeMetrics(
        coverage=float(
            np.mean(inside)
        ),
        average_width=float(
            np.mean(width)
        ),
        median_width=float(
            np.median(width)
        ),
        lower_violation_rate=float(
            np.mean(lower_violation)
        ),
        upper_violation_rate=float(
            np.mean(upper_violation)
        ),
        samples=len(actual),
    )


def confidence_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    thresholds: Optional[
        list[float]
    ] = None,
) -> pd.DataFrame:
    """
    Evaluate performance at different confidence thresholds.

    Only observations satisfying:

        P(up) >= threshold

    or:

        P(up) <= 1 - threshold

    are considered actionable.

    This is useful for determining whether the model can
    responsibly say "no high-confidence setup".
    """

    y_true = _validate_binary_target(
        y_true
    )

    probability = np.asarray(
        probability,
        dtype=float,
    ).reshape(-1)

    _validate_equal_length(
        y_true,
        probability,
    )

    if thresholds is None:
        thresholds = [
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
        ]

    rows = []

    for threshold in thresholds:

        if not 0.50 < threshold < 1.0:
            continue

        actionable = (
            (probability >= threshold)
            | (
                probability
                <= 1.0 - threshold
            )
        )

        count = int(
            actionable.sum()
        )

        if count == 0:

            rows.append(
                {
                    "threshold": threshold,
                    "coverage": 0.0,
                    "accuracy": np.nan,
                    "samples": 0,
                }
            )

            continue

        actual = y_true[
            actionable
        ]

        predicted_direction = (
            probability[actionable]
            >= 0.50
        ).astype(int)

        accuracy = float(
            np.mean(
                actual
                == predicted_direction
            )
        )

        rows.append(
            {
                "threshold": threshold,
                "coverage": (
                    count
                    / len(y_true)
                ),
                "accuracy": accuracy,
                "samples": count,
            }
        )

    return pd.DataFrame(
        rows
    )


def trading_metrics(
    returns: np.ndarray,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> TradingMetrics:
    """
    Calculate basic trading performance statistics.

    `returns` should represent strategy returns per observation.

    This is intentionally a simple metric engine. A later
    backtesting engine will handle entries, exits, stops,
    targets, slippage and transaction costs explicitly.
    """

    returns = np.asarray(
        returns,
        dtype=float,
    ).reshape(-1)

    if len(returns) == 0:
        raise ValueError(
            "returns cannot be empty."
        )

    if not np.isfinite(
        returns
    ).all():
        raise ValueError(
            "returns contain non-finite values."
        )

    equity = np.cumprod(
        1.0 + returns
    )

    total_return = (
        equity[-1]
        - 1.0
    )

    years = (
        len(returns)
        / periods_per_year
    )

    if years > 0:
        annualized_return = (
            equity[-1]
            ** (1.0 / years)
            - 1.0
        )
    else:
        annualized_return = 0.0

    volatility = (
        np.std(
            returns,
            ddof=1,
        )
        * np.sqrt(
            periods_per_year
        )
        if len(returns) > 1
        else 0.0
    )

    excess_returns = (
        returns
        - (
            risk_free_rate
            / periods_per_year
        )
    )

    excess_std = np.std(
        excess_returns,
        ddof=1,
    )

    if excess_std > 0:

        sharpe = (
            np.mean(
                excess_returns
            )
            / excess_std
            * np.sqrt(
                periods_per_year
            )
        )

    else:

        sharpe = 0.0

    running_max = np.maximum.accumulate(
        equity
    )

    drawdown = (
        equity
        / running_max
        - 1.0
    )

    maximum_drawdown = float(
        drawdown.min()
    )

    winning_returns = (
        returns[returns > 0]
    )

    losing_returns = (
        returns[returns < 0]
    )

    trades = int(
        np.count_nonzero(
            returns
        )
    )

    if trades > 0:

        win_rate = float(
            np.sum(
                returns > 0
            )
            / trades
        )

    else:

        win_rate = 0.0

    gross_profit = (
        winning_returns.sum()
    )

    gross_loss = (
        np.abs(
            losing_returns.sum()
        )
    )

    if gross_loss > 0:

        profit_factor = float(
            gross_profit
            / gross_loss
        )

    elif gross_profit > 0:

        profit_factor = float(
            "inf"
        )

    else:

        profit_factor = 0.0

    return TradingMetrics(
        total_return=float(
            total_return
        ),
        annualized_return=float(
            annualized_return
        ),
        volatility=float(
            volatility
        ),
        sharpe_ratio=float(
            sharpe
        ),
        maximum_drawdown=(
            maximum_drawdown
        ),
        win_rate=win_rate,
        profit_factor=profit_factor,
        trades=trades,
    )


def evaluate_95_percent_gate(
    classification: ClassificationMetrics,
    minimum_accuracy: float = 0.95,
) -> dict:
    """
    Evaluate the project's strict directional-accuracy gate.

    IMPORTANT:
    Passing this function does NOT mean the model is production-ready.

    The model must also pass:

        - leakage checks
        - walk-forward testing
        - final holdout testing
        - calibration checks
        - range coverage checks
        - regime stability
        - trading/backtest checks
    """

    passed_accuracy = (
        classification.accuracy
        >= minimum_accuracy
    )

    return {
        "passed_accuracy_gate": bool(
            passed_accuracy
        ),
        "accuracy": classification.accuracy,
        "required_accuracy": (
            minimum_accuracy
        ),
        "samples": classification.samples,
        "production_ready": False,
        "reason": (
            "Accuracy gate is only one research requirement. "
            "Additional out-of-sample validation is mandatory."
        ),
    }


def metric_summary(
    classification: Optional[
        ClassificationMetrics
    ] = None,
    regression: Optional[
        RegressionMetrics
    ] = None,
    range_result: Optional[
        RangeMetrics
    ] = None,
) -> pd.DataFrame:
    """
    Convert available metrics into a compact dashboard dataframe.
    """

    rows = []

    if classification is not None:

        rows.extend(
            [
                {
                    "Category": "Classification",
                    "Metric": "Accuracy",
                    "Value": classification.accuracy,
                },
                {
                    "Category": "Classification",
                    "Metric": "Balanced Accuracy",
                    "Value": classification.balanced_accuracy,
                },
                {
                    "Category": "Classification",
                    "Metric": "Precision",
                    "Value": classification.precision,
                },
                {
                    "Category": "Classification",
                    "Metric": "Recall",
                    "Value": classification.recall,
                },
                {
                    "Category": "Classification",
                    "Metric": "F1",
                    "Value": classification.f1,
                },
                {
                    "Category": "Classification",
                    "Metric": "ROC AUC",
                    "Value": classification.roc_auc,
                },
                {
                    "Category": "Classification",
                    "Metric": "Brier Score",
                    "Value": classification.brier_score,
                },
            ]
        )

    if regression is not None:

        rows.extend(
            [
                {
                    "Category": "Regression",
                    "Metric": "MAE",
                    "Value": regression.mae,
                },
                {
                    "Category": "Regression",
                    "Metric": "RMSE",
                    "Value": regression.rmse,
                },
                {
                    "Category": "Regression",
                    "Metric": "Directional Accuracy",
                    "Value": regression.directional_accuracy,
                },
            ]
        )

    if range_result is not None:

        rows.extend(
            [
                {
                    "Category": "Range",
                    "Metric": "Coverage",
                    "Value": range_result.coverage,
                },
                {
                    "Category": "Range",
                    "Metric": "Average Width",
                    "Value": range_result.average_width,
                },
                {
                    "Category": "Range",
                    "Metric": "Median Width",
                    "Value": range_result.median_width,
                },
                {
                    "Category": "Range",
                    "Metric": "Lower Violation Rate",
                    "Value": range_result.lower_violation_rate,
                },
                {
                    "Category": "Range",
                    "Metric": "Upper Violation Rate",
                    "Value": range_result.upper_violation_rate,
                },
            ]
        )

    return pd.DataFrame(
        rows
    )
