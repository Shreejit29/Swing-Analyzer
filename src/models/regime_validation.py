"""
Market-regime validation for AI Swing Analyser.

Evaluates model performance separately across different market
conditions.

Supported regimes include:

    - Bull / Bear / Neutral trend
    - Low / Normal / High volatility
    - Strong / Weak trend
    - Positive / Negative momentum

The purpose is to identify models that only work under one
specific historical market condition.

IMPORTANT:

Regime labels must be generated using information available
at the prediction timestamp only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class RegimeValidationConfig:
    """Configuration for regime validation."""

    minimum_samples_per_regime: int = 30

    minimum_accuracy: float = 0.55

    maximum_accuracy_std: float = 0.15

    require_all_regimes: bool = False


@dataclass
class RegimeResult:
    """Performance of a model inside one regime."""

    regime: str

    samples: int

    accuracy: float

    precision: float

    recall: float

    f1: float

    average_probability: float

    actual_up_rate: float

    prediction_up_rate: float

    passed_sample_gate: bool

    passed_accuracy_gate: bool


@dataclass
class RegimeValidationReport:
    """Complete regime validation report."""

    results: pd.DataFrame

    regimes_tested: int

    regimes_with_enough_samples: int

    mean_accuracy: float

    minimum_accuracy: float

    maximum_accuracy: float

    accuracy_std: float

    passed_sample_gate: bool

    passed_accuracy_gate: bool

    passed_stability_gate: bool

    passed: bool


def _validate_inputs(
    actual: np.ndarray,
    probability: np.ndarray,
    regime: pd.Series,
) -> tuple[
    np.ndarray,
    np.ndarray,
    pd.Series,
]:
    """Validate regime-validation inputs."""

    actual = np.asarray(
        actual,
        dtype=int,
    ).reshape(-1)

    probability = np.asarray(
        probability,
        dtype=float,
    ).reshape(-1)

    regime = pd.Series(
        regime
    ).reset_index(
        drop=True
    )

    if not (
        len(actual)
        == len(probability)
        == len(regime)
    ):
        raise ValueError(
            "actual, probability and regime must have equal length."
        )

    if len(actual) == 0:
        raise ValueError(
            "Regime validation data cannot be empty."
        )

    if not np.isin(
        actual,
        [0, 1],
    ).all():
        raise ValueError(
            "actual must contain only 0 and 1."
        )

    if not np.isfinite(
        probability
    ).all():
        raise ValueError(
            "probability contains non-finite values."
        )

    probability = np.clip(
        probability,
        0.0,
        1.0,
    )

    if regime.isna().any():
        raise ValueError(
            "regime contains missing values."
        )

    return (
        actual,
        probability,
        regime,
    )


def _classification_statistics(
    actual: np.ndarray,
    prediction: np.ndarray,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    """Calculate simple classification statistics."""

    accuracy = float(
        np.mean(
            actual == prediction
        )
    )

    true_positive = np.sum(
        (actual == 1)
        & (prediction == 1)
    )

    false_positive = np.sum(
        (actual == 0)
        & (prediction == 1)
    )

    false_negative = np.sum(
        (actual == 1)
        & (prediction == 0)
    )

    precision = (
        float(
            true_positive
            / (
                true_positive
                + false_positive
            )
        )
        if (
            true_positive
            + false_positive
        ) > 0
        else 0.0
    )

    recall = (
        float(
            true_positive
            / (
                true_positive
                + false_negative
            )
        )
        if (
            true_positive
            + false_negative
        ) > 0
        else 0.0
    )

    if (
        precision
        + recall
    ) > 0:

        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )

    else:

        f1 = 0.0

    return (
        accuracy,
        precision,
        recall,
        float(f1),
    )


def evaluate_by_regime(
    actual: np.ndarray,
    probability: np.ndarray,
    regime: pd.Series,
    threshold: float = 0.50,
    minimum_samples: int = 30,
) -> pd.DataFrame:
    """
    Evaluate classification performance separately by regime.
    """

    (
        actual,
        probability,
        regime,
    ) = _validate_inputs(
        actual,
        probability,
        regime,
    )

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    frame = pd.DataFrame(
        {
            "actual": actual,
            "probability": probability,
            "regime": regime,
        }
    )

    frame["prediction"] = (
        frame["probability"]
        >= threshold
    ).astype(int)

    rows = []

    for regime_name, group in (
        frame.groupby(
            "regime",
            sort=True,
        )
    ):

        group_actual = (
            group["actual"]
            .to_numpy()
        )

        group_prediction = (
            group["prediction"]
            .to_numpy()
        )

        (
            accuracy,
            precision,
            recall,
            f1,
        ) = _classification_statistics(
            group_actual,
            group_prediction,
        )

        rows.append(
            {
                "Regime": str(
                    regime_name
                ),
                "Samples": len(group),
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "Average Probability": float(
                    group["probability"].mean()
                ),
                "Actual Up Rate": float(
                    group["actual"].mean()
                ),
                "Prediction Up Rate": float(
                    group["prediction"].mean()
                ),
                "Enough Samples": (
                    len(group)
                    >= minimum_samples
                ),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        rows
    )


def build_regime_report(
    regime_results: pd.DataFrame,
    config: Optional[
        RegimeValidationConfig
    ] = None,
) -> RegimeValidationReport:
    """
    Convert regime-level results into an overall regime report.
    """

    config = (
        config
        if config is not None
        else RegimeValidationConfig()
    )

    if regime_results.empty:
        return RegimeValidationReport(
            results=regime_results,
            regimes_tested=0,
            regimes_with_enough_samples=0,
            mean_accuracy=np.nan,
            minimum_accuracy=np.nan,
            maximum_accuracy=np.nan,
            accuracy_std=np.nan,
            passed_sample_gate=False,
            passed_accuracy_gate=False,
            passed_stability_gate=False,
            passed=False,
        )

    accuracies = (
        regime_results["Accuracy"]
        .to_numpy(
            dtype=float
        )
    )

    enough_samples = (
        regime_results[
            "Enough Samples"
        ].astype(bool)
    )

    valid_accuracies = accuracies[
        enough_samples.to_numpy()
    ]

    regimes_tested = len(
        regime_results
    )

    regimes_with_enough_samples = (
        len(valid_accuracies)
    )

    if len(valid_accuracies) > 0:

        mean_accuracy = float(
            np.mean(
                valid_accuracies
            )
        )

        minimum_accuracy = float(
            np.min(
                valid_accuracies
            )
        )

        maximum_accuracy = float(
            np.max(
                valid_accuracies
            )
        )

        accuracy_std = float(
            np.std(
                valid_accuracies,
                ddof=1,
            )
        ) if len(
            valid_accuracies
        ) > 1 else 0.0

    else:

        mean_accuracy = np.nan
        minimum_accuracy = np.nan
        maximum_accuracy = np.nan
        accuracy_std = np.nan

    if config.require_all_regimes:

        passed_sample_gate = (
            regimes_with_enough_samples
            == regimes_tested
        )

    else:

        passed_sample_gate = (
            regimes_with_enough_samples
            > 0
        )

    passed_accuracy_gate = (
        len(valid_accuracies) > 0
        and np.all(
            valid_accuracies
            >= config.minimum_accuracy
        )
    )

    passed_stability_gate = (
        len(valid_accuracies) > 0
        and accuracy_std
        <= config.maximum_accuracy_std
    )

    passed = (
        passed_sample_gate
        and passed_accuracy_gate
        and passed_stability_gate
    )

    return RegimeValidationReport(
        results=regime_results,
        regimes_tested=regimes_tested,
        regimes_with_enough_samples=(
            regimes_with_enough_samples
        ),
        mean_accuracy=mean_accuracy,
        minimum_accuracy=minimum_accuracy,
        maximum_accuracy=maximum_accuracy,
        accuracy_std=accuracy_std,
        passed_sample_gate=bool(
            passed_sample_gate
        ),
        passed_accuracy_gate=bool(
            passed_accuracy_gate
        ),
        passed_stability_gate=bool(
            passed_stability_gate
        ),
        passed=bool(
            passed
        ),
    )


def classify_trend_regime(
    close: pd.Series,
    fast_period: int = 50,
    slow_period: int = 200,
) -> pd.Series:
    """
    Classify trend regime using only historical prices.

    Returns:

        BULL
        BEAR
        NEUTRAL
    """

    close = pd.to_numeric(
        close,
        errors="coerce",
    )

    fast_ma = (
        close
        .rolling(
            fast_period,
            min_periods=fast_period,
        )
        .mean()
    )

    slow_ma = (
        close
        .rolling(
            slow_period,
            min_periods=slow_period,
        )
        .mean()
    )

    regime = pd.Series(
        "NEUTRAL",
        index=close.index,
        dtype="object",
    )

    bull = (
        (close > slow_ma)
        & (fast_ma > slow_ma)
    )

    bear = (
        (close < slow_ma)
        & (fast_ma < slow_ma)
    )

    regime.loc[
        bull
    ] = "BULL"

    regime.loc[
        bear
    ] = "BEAR"

    regime.loc[
        fast_ma.isna()
        | slow_ma.isna()
    ] = np.nan

    return regime


def classify_volatility_regime(
    close: pd.Series,
    lookback: int = 20,
    low_quantile: float = 0.33,
    high_quantile: float = 0.67,
) -> pd.Series:
    """
    Classify volatility as:

        LOW_VOL
        NORMAL_VOL
        HIGH_VOL

    Quantile thresholds are calculated using expanding historical
    observations, avoiding future-data leakage.
    """

    close = pd.to_numeric(
        close,
        errors="coerce",
    )

    returns = (
        close.pct_change()
    )

    volatility = (
        returns
        .rolling(
            lookback,
            min_periods=lookback,
        )
        .std()
    )

    historical_low = (
        volatility
        .expanding(
            min_periods=lookback,
        )
        .quantile(
            low_quantile
        )
    )

    historical_high = (
        volatility
        .expanding(
            min_periods=lookback,
        )
        .quantile(
            high_quantile
        )
    )

    regime = pd.Series(
        "NORMAL_VOL",
        index=close.index,
        dtype="object",
    )

    low = (
        volatility
        <= historical_low
    )

    high = (
        volatility
        >= historical_high
    )

    regime.loc[
        low
    ] = "LOW_VOL"

    regime.loc[
        high
    ] = "HIGH_VOL"

    regime.loc[
        volatility.isna()
    ] = np.nan

    return regime


def combine_regimes(
    trend_regime: pd.Series,
    volatility_regime: pd.Series,
) -> pd.Series:
    """
    Combine trend and volatility regimes.

    Examples:

        BULL_LOW_VOL
        BULL_HIGH_VOL
        BEAR_LOW_VOL
        BEAR_HIGH_VOL
        NEUTRAL_NORMAL_VOL
    """

    trend = pd.Series(
        trend_regime
    )

    volatility = pd.Series(
        volatility_regime
    )

    if not trend.index.equals(
        volatility.index
    ):
        raise ValueError(
            "Regime series must have identical indexes."
        )

    combined = (
        trend.astype("string")
        + "_"
        + volatility.astype("string")
    )

    combined = combined.where(
        trend.notna()
        & volatility.notna()
    )

    return combined


def regime_stability_score(
    report: RegimeValidationReport,
) -> float:
    """
    Produce a 0-1 regime stability score.

    This is a diagnostic score, not a calibrated probability.
    """

    if (
        report.results.empty
        or not np.isfinite(
            report.mean_accuracy
        )
    ):
        return 0.0

    accuracy_component = np.clip(
        report.minimum_accuracy,
        0.0,
        1.0,
    )

    if np.isfinite(
        report.accuracy_std
    ):

        stability_component = np.clip(
            1.0
            - (
                report.accuracy_std
                / 0.20
            ),
            0.0,
            1.0,
        )

    else:

        stability_component = 0.0

    sample_component = (
        report.regimes_with_enough_samples
        / max(
            report.regimes_tested,
            1,
        )
    )

    return float(
        np.clip(
            (
                0.50
                * accuracy_component
                + 0.30
                * stability_component
                + 0.20
                * sample_component
            ),
            0.0,
            1.0,
        )
    )
