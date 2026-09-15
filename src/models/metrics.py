```python
"""
Model evaluation metrics for AI Swing Analyser.

Provides leakage-safe evaluation metrics for:

    - Direction classification
    - Probability quality
    - Probability calibration
    - Return prediction
    - Price-range prediction
    - Trading-oriented performance
    - Confidence-threshold analysis

IMPORTANT
---------
Metrics must be calculated on observations that were not used to
fit the model being evaluated.

For financial prediction, accuracy alone is NOT sufficient.

The research layer should consider:

    - Balanced Accuracy
    - ROC-AUC
    - PR-AUC
    - F1
    - Log Loss
    - Brier Score
    - Matthews Correlation Coefficient
    - Calibration Error
    - Out-of-sample trading performance
    - Drawdown
    - Profit Factor
    - Stability across time
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------


@dataclass
class ClassificationMetrics:
    """Container for binary classification evaluation."""

    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    log_loss: float
    brier_score: float
    matthews_corrcoef: float
    expected_calibration_error: float
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


# ---------------------------------------------------------------------
# VALIDATION HELPERS
# ---------------------------------------------------------------------


def _validate_equal_length(
    *arrays: np.ndarray,
) -> None:
    """Ensure all supplied arrays have equal length."""

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
    """
    Validate and return a binary target.

    Accepted classes:

        0
        1
    """

    y_true = np.asarray(
        y_true,
        dtype=int,
    ).reshape(-1)

    if len(y_true) == 0:
        raise ValueError(
            "Classification target cannot be empty."
        )

    unique = np.unique(
        y_true
    )

    if not np.isin(
        unique,
        [0, 1],
    ).all():

        raise ValueError(
            "Classification target must contain only 0 and 1."
        )

    return y_true


def _validate_probability(
    probability: np.ndarray,
) -> np.ndarray:
    """
    Validate and safely clip predicted probabilities.
    """

    probability = np.asarray(
        probability,
        dtype=float,
    ).reshape(-1)

    if len(probability) == 0:
        raise ValueError(
            "Predicted probabilities cannot be empty."
        )

    if not np.isfinite(
        probability
    ).all():

        raise ValueError(
            "Predicted probabilities contain non-finite values."
        )

    return np.clip(
        probability,
        0.0,
        1.0,
    )


# ---------------------------------------------------------------------
# CALIBRATION
# ---------------------------------------------------------------------


def expected_calibration_error(
    y_true: np.ndarray,
    probability: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Calculate Expected Calibration Error (ECE).

    ECE measures the difference between:

        predicted probability

    and:

        observed frequency

    across probability bins.

    Lower is better.

    Example
    -------
    If predictions with probability around 0.80 are correct
    approximately 80% of the time, calibration is good.

    Parameters
    ----------
    y_true:
        Binary 0/1 outcomes.

    probability:
        Predicted probability of class 1.

    n_bins:
        Number of probability bins.
    """

    y_true = _validate_binary_target(
        y_true
    )

    probability = _validate_probability(
        probability
    )

    _validate_equal_length(
        y_true,
        probability,
    )

    if n_bins < 2:
        raise ValueError(
            "n_bins must be at least 2."
        )

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    total = len(
        y_true
    )

    ece = 0.0

    for index in range(
        n_bins
    ):

        lower = edges[index]
        upper = edges[index + 1]

        if index == n_bins - 1:

            mask = (
                (probability >= lower)
                & (probability <= upper)
            )

        else:

            mask = (
                (probability >= lower)
                & (probability < upper)
            )

        count = int(
            mask.sum()
        )

        if count == 0:
            continue

        mean_probability = float(
            probability[mask].mean()
        )

        observed_frequency = float(
            y_true[mask].mean()
        )

        ece += (
            count
            / total
        ) * abs(
            mean_probability
            - observed_frequency
        )

    return float(
        ece
    )


def calibration_table(
    y_true: np.ndarray,
    probability: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Return a probability calibration table.

    Columns
    -------
    bin_lower
    bin_upper
    samples
    mean_probability
    observed_frequency
    calibration_gap
    """

    y_true = _validate_binary_target(
        y_true
    )

    probability = _validate_probability(
        probability
    )

    _validate_equal_length(
        y_true,
        probability,
    )

    if n_bins < 2:
        raise ValueError(
            "n_bins must be at least 2."
        )

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    rows = []

    for index in range(
        n_bins
    ):

        lower = float(
            edges[index]
        )

        upper = float(
            edges[index + 1]
        )

        if index == n_bins - 1:

            mask = (
                (probability >= lower)
                & (probability <= upper)
            )

        else:

            mask = (
                (probability >= lower)
                & (probability < upper)
            )

        count = int(
            mask.sum()
        )

        if count == 0:

            rows.append(
                {
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "samples": 0,
                    "mean_probability": np.nan,
                    "observed_frequency": np.nan,
                    "calibration_gap": np.nan,
                }
            )

            continue

        mean_probability = float(
            probability[mask].mean()
        )

        observed_frequency = float(
            y_true[mask].mean()
        )

        rows.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "samples": count,
                "mean_probability": mean_probability,
                "observed_frequency": observed_frequency,
                "calibration_gap": (
                    observed_frequency
                    - mean_probability
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------
# CLASSIFICATION METRICS
# ---------------------------------------------------------------------


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

    Notes
    -----
    Accuracy is intentionally only one metric.

    For financial classification, probability quality and
    out-of-sample trading performance are also important.
    """

    y_true = _validate_binary_target(
        y_true
    )

    probability = _validate_probability(
        predicted_probability
    )

    _validate_equal_length(
        y_true,
        probability,
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

    unique_classes = np.unique(
        y_true
    )

    if len(unique_classes) == 2:

        roc_auc = roc_auc_score(
            y_true,
            probability,
        )

        pr_auc = average_precision_score(
            y_true,
            probability,
        )

    else:

        roc_auc = float("nan")
        pr_auc = float("nan")

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

    mcc = matthews_corrcoef(
        y_true,
        prediction,
    )

    ece = expected_calibration_error(
        y_true,
        probability,
    )

    return ClassificationMetrics(
        accuracy=float(
            accuracy
        ),
        balanced_accuracy=float(
            balanced_accuracy
        ),
        precision=float(
            precision
        ),
        recall=float(
            recall
        ),
        f1=float(
            f1
        ),
        roc_auc=float(
            roc_auc
        ),
        pr_auc=float(
            pr_auc
        ),
        log_loss=float(
            ll
        ),
        brier_score=float(
            brier
        ),
        matthews_corrcoef=float(
            mcc
        ),
        expected_calibration_error=float(
            ece
        ),
        samples=len(
            y_true
        ),
    )


def classification_research_metrics(
    y_true: np.ndarray,
    predicted_probability: np.ndarray,
    threshold: float = 0.50,
) -> dict[str, float]:
    """
    Return classification metrics as a flat dictionary.

    This format is convenient for:

        - pandas DataFrames
        - Streamlit
        - CSV logging
        - experiment tracking
        - model comparison

    The dictionary deliberately contains both traditional
    classification metrics and probability-quality metrics.
    """

    result = classification_metrics(
        y_true=y_true,
        predicted_probability=predicted_probability,
        threshold=threshold,
    )

    return {
        "accuracy": result.accuracy,
        "balanced_accuracy": result.balanced_accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "roc_auc": result.roc_auc,
        "pr_auc": result.pr_auc,
        "log_loss": result.log_loss,
        "brier_score": result.brier_score,
        "matthews_corrcoef": result.matthews_corrcoef,
        "expected_calibration_error": (
            result.expected_calibration_error
        ),
        "samples": float(
            result.samples
        ),
    }


def confusion_matrix_dataframe(
    y_true: np.ndarray,
    predicted_probability: np.ndarray,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """
    Return confusion matrix as a labelled dataframe.
    """

    y_true = _validate_binary_target(
        y_true
    )

    probability = _validate_probability(
        predicted_probability
    )

    _validate_equal_length(
        y_true,
        probability,
    )

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
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


# ---------------------------------------------------------------------
# REGRESSION METRICS
# ---------------------------------------------------------------------


def regression_metrics(
    y_true: np.ndarray,
    prediction: np.ndarray,
) -> RegressionMetrics:
    """
    Evaluate future-return predictions.
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    ).reshape(-1)

    prediction = np.asarray(
        prediction,
        dtype=float,
    ).reshape(-1)

    if len(y_true) == 0:
        raise ValueError(
            "y_true cannot be empty."
        )

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
        samples=len(
            y_true
        ),
    )


# ---------------------------------------------------------------------
# RANGE METRICS
# ---------------------------------------------------------------------


def range_metrics(
    actual_future_return: np.ndarray,
    predicted_lower_return: np.ndarray,
    predicted_upper_return: np.ndarray,
) -> RangeMetrics:
    """
    Evaluate prediction interval quality.

    Coverage:
        Fraction of observations where actual return is inside
        the predicted interval.

    Width:
        Average size of the predicted interval.

    Lower/upper violation:
        Fraction of observations outside the interval on
        each respective side.
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

    if len(actual) == 0:
        raise ValueError(
            "Range inputs cannot be empty."
        )

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
            np.mean(
                lower_violation
            )
        ),
        upper_violation_rate=float(
            np.mean(
                upper_violation
            )
        ),
        samples=len(
            actual
        ),
    )


# ---------------------------------------------------------------------
# CONFIDENCE / THRESHOLD ANALYSIS
# ---------------------------------------------------------------------


def confidence_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    thresholds: Optional[
        list[float]
    ] = None,
) -> pd.DataFrame:
    """
    Evaluate performance at different probability thresholds.

    For a threshold such as 0.70:

        probability >= 0.70
            -> predicted UP

        probability <= 0.30
            -> predicted DOWN

        otherwise
            -> WAIT

    This is much closer to the eventual swing-trading decision
    process than simply measuring accuracy on every observation.
    """

    y_true = _validate_binary_target(
        y_true
    )

    probability = _validate_probability(
        probability
    )

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
                    "balanced_accuracy": np.nan,
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

        if len(
            np.unique(actual)
        ) == 2:

            balanced = float(
                balanced_accuracy_score(
                    actual,
                    predicted_direction,
                )
            )

        else:

            balanced = float(
                accuracy
            )

        rows.append(
            {
                "threshold": threshold,
                "coverage": (
                    count
                    / len(y_true)
                ),
                "accuracy": accuracy,
                "balanced_accuracy": balanced,
                "samples": count,
            }
        )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------
# TRADING METRICS
# ---------------------------------------------------------------------


def trading_metrics(
    returns: np.ndarray,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> TradingMetrics:
    """
    Calculate basic strategy performance statistics.

    Parameters
    ----------
    returns:
        Strategy return for each observation.

    periods_per_year:
        Number of observations per year.

    risk_free_rate:
        Annual risk-free rate.

    Notes
    -----
    This function is intentionally a simple metric engine.

    It does NOT model:

        - entries
        - exits
        - intraday execution
        - stop loss
        - target execution
        - slippage
        - brokerage

    Those belong in the backtesting layer.
    """

    returns = np.asarray(
        returns,
        dtype=float,
    ).reshape(-1)

    if len(returns) == 0:
        raise ValueError(
            "returns cannot be empty."
        )

    if periods_per_year <= 0:
        raise ValueError(
            "periods_per_year must be positive."
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

    if years > 0 and equity[-1] > 0:

        annualized_return = (
            equity[-1]
            ** (1.0 / years)
            - 1.0
        )

    else:

        annualized_return = -1.0

    if len(returns) > 1:

        volatility = (
            np.std(
                returns,
                ddof=1,
            )
            * np.sqrt(
                periods_per_year
            )
        )

    else:

        volatility = 0.0

    excess_returns = (
        returns
        - (
            risk_free_rate
            / periods_per_year
        )
    )

    if len(excess_returns) > 1:

        excess_std = np.std(
            excess_returns,
            ddof=1,
        )

    else:

        excess_std = 0.0

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
        returns[
            returns > 0
        ]
    )

    losing_returns = (
        returns[
            returns < 0
        ]
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

    gross_profit = float(
        winning_returns.sum()
    )

    gross_loss = float(
        np.abs(
            losing_returns.sum()
        )
    )

    if gross_loss > 0:

        profit_factor = (
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
        win_rate=float(
            win_rate
        ),
        profit_factor=float(
            profit_factor
        ),
        trades=trades,
    )


# ---------------------------------------------------------------------
# LEGACY ACCURACY GATE
# ---------------------------------------------------------------------


def evaluate_95_percent_gate(
    classification: ClassificationMetrics,
    minimum_accuracy: float = 0.95,
) -> dict:
    """
    Evaluate the project's historical strict accuracy gate.

    This function is retained for backward compatibility.

    IMPORTANT
    ---------
    A 95% accuracy result is NOT considered sufficient evidence
    of a profitable trading model.

    New research should use:

        - ROC-AUC
        - PR-AUC
        - Log Loss
        - Brier Score
        - MCC
        - Calibration Error
        - Walk-forward performance
        - Trading expectancy
        - Drawdown
        - Robustness

    This function should eventually be replaced by a complete
    production approval framework.
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
            "Accuracy alone cannot establish production readiness. "
            "Out-of-sample validation, calibration, robustness and "
            "trading performance are required."
        ),
    }


# ---------------------------------------------------------------------
# RESEARCH MODEL SCORECARD
# ---------------------------------------------------------------------


def research_scorecard(
    classification: ClassificationMetrics,
) -> dict[str, float]:
    """
    Return a compact research scorecard.

    This is NOT a profitability score.

    It is a convenient summary of statistical model quality.

    Higher is generally better for:

        ROC-AUC
        PR-AUC
        Balanced Accuracy
        F1
        MCC

    Lower is better for:

        Log Loss
        Brier Score
        Expected Calibration Error
    """

    return {
        "ROC_AUC": classification.roc_auc,
        "PR_AUC": classification.pr_auc,
        "Balanced_Accuracy": (
            classification.balanced_accuracy
        ),
        "F1": classification.f1,
        "MCC": classification.matthews_corrcoef,
        "Log_Loss": classification.log_loss,
        "Brier_Score": classification.brier_score,
        "ECE": (
            classification.expected_calibration_error
        ),
        "Accuracy": classification.accuracy,
        "Samples": float(
            classification.samples
        ),
    }


# ---------------------------------------------------------------------
# METRIC SUMMARY
# ---------------------------------------------------------------------


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
    Convert available metrics into a compact dataframe.

    Useful for Streamlit dashboards and experiment reports.
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
                    "Metric": "PR AUC",
                    "Value": classification.pr_auc,
                },
                {
                    "Category": "Classification",
                    "Metric": "MCC",
                    "Value": classification.matthews_corrcoef,
                },
                {
                    "Category": "Classification",
                    "Metric": "Log Loss",
                    "Value": classification.log_loss,
                },
                {
                    "Category": "Classification",
                    "Metric": "Brier Score",
                    "Value": classification.brier_score,
                },
                {
                    "Category": "Classification",
                    "Metric": "Calibration Error",
                    "Value": (
                        classification
                        .expected_calibration_error
                    ),
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
                    "Metric": "Mean Error",
                    "Value": regression.mean_error,
                },
                {
                    "Category": "Regression",
                    "Metric": "Median Absolute Error",
                    "Value": (
                        regression
                        .median_absolute_error
                    ),
                },
                {
                    "Category": "Regression",
                    "Metric": "Directional Accuracy",
                    "Value": (
                        regression
                        .directional_accuracy
                    ),
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
                    "Value": (
                        range_result
                        .lower_violation_rate
                    ),
                },
                {
                    "Category": "Range",
                    "Metric": "Upper Violation Rate",
                    "Value": (
                        range_result
                        .upper_violation_rate
                    ),
                },
            ]
        )

    return pd.DataFrame(
        rows
    )


__all__ = [
    "ClassificationMetrics",
    "RegressionMetrics",
    "RangeMetrics",
    "TradingMetrics",
    "classification_metrics",
    "classification_research_metrics",
    "confusion_matrix_dataframe",
    "expected_calibration_error",
    "calibration_table",
    "regression_metrics",
    "range_metrics",
    "confidence_metrics",
    "trading_metrics",
    "evaluate_95_percent_gate",
    "research_scorecard",
    "metric_summary",
]
```
