"""
AI Swing Analyser — Research Stage Adapters.

Converts outputs from individual research stages into the common
ResearchEvidence format.

The adapters intentionally avoid:

- model training
- feature fitting
- holdout manipulation
- signal generation
- automatic production approval

They only interpret already-produced research results.

If a result is incomplete or ambiguous, the adapter returns
NOT_EVALUATED rather than assuming success.
"""

from __future__ import annotations

from typing import Any

from src.research.integration import (
    EvidenceItem,
    EvidenceStatus,
)


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------


def _safe_float(
    value: Any,
) -> float | None:

    if value is None:
        return None

    try:
        result = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not result == result:
        return None

    return result


def _safe_bool(
    value: Any,
) -> bool | None:

    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        return value

    return None


def _get_attr(
    obj: Any,
    name: str,
    default: Any = None,
) -> Any:

    if obj is None:
        return default

    return getattr(
        obj,
        name,
        default,
    )


def _status_from_passed(
    passed: bool | None,
) -> EvidenceStatus:

    if passed is True:
        return EvidenceStatus.PASS

    if passed is False:
        return EvidenceStatus.FAIL

    return EvidenceStatus.NOT_EVALUATED


def _base_metrics(
    result: Any,
) -> dict[str, Any]:

    metrics = {}

    if result is None:
        return metrics

    summary_method = getattr(
        result,
        "summary",
        None,
    )

    if callable(summary_method):
        try:
            summary = summary_method()

            if isinstance(
                summary,
                dict,
            ):
                metrics.update(
                    summary
                )
        except Exception:
            # Adapters must never make a production
            # decision because a diagnostic summary
            # failed.
            pass

    return metrics


# ---------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------


def adapt_walk_forward_result(
    result: Any,
    minimum_accuracy: float = 0.95,
) -> EvidenceItem:

    if result is None:
        return EvidenceItem(
            name="walk_forward_validation"
        )

    mean_accuracy = _safe_float(
        _get_attr(
            result,
            "mean_accuracy",
        )
    )

    if mean_accuracy is None:
        summary = _base_metrics(
            result
        )

        mean_accuracy = _safe_float(
            summary.get(
                "mean_accuracy"
            )
        )

    minimum_accuracy_observed = _safe_float(
        _get_attr(
            result,
            "minimum_accuracy",
        )
    )

    if minimum_accuracy_observed is None:
        summary = _base_metrics(
            result
        )

        minimum_accuracy_observed = (
            _safe_float(
                summary.get(
                    "minimum_accuracy"
                )
            )
        )

    passed_accuracy_gate = (
        _safe_bool(
            _get_attr(
                result,
                "passed_accuracy_gate",
            )
        )
    )

    if passed_accuracy_gate is None:
        if (
            mean_accuracy is not None
            and minimum_accuracy_observed
            is not None
        ):
            passed_accuracy_gate = (
                mean_accuracy
                >= minimum_accuracy
                and minimum_accuracy_observed
                >= minimum_accuracy
            )

    if passed_accuracy_gate is None:
        return EvidenceItem(
            name="walk_forward_validation",
            status=EvidenceStatus.NOT_EVALUATED,
            threshold=minimum_accuracy,
            details=(
                "Walk-forward accuracy evidence "
                "could not be established."
            ),
            critical=True,
            source="walk_forward_research",
        )

    metrics = _base_metrics(
        result
    )

    if mean_accuracy is not None:
        metrics[
            "mean_accuracy"
        ] = mean_accuracy

    if minimum_accuracy_observed is not None:
        metrics[
            "minimum_accuracy"
        ] = minimum_accuracy_observed

    return EvidenceItem(
        name="walk_forward_validation",
        status=_status_from_passed(
            passed_accuracy_gate
        ),
        score=mean_accuracy,
        threshold=minimum_accuracy,
        metrics=metrics,
        details=(
            "Walk-forward validation evaluated "
            "without using the final holdout."
        ),
        critical=True,
        source="walk_forward_research",
    )


# ---------------------------------------------------------------------
# Final holdout
# ---------------------------------------------------------------------


def adapt_holdout_result(
    result: Any,
    minimum_accuracy: float = 0.95,
) -> EvidenceItem:

    if result is None:
        return EvidenceItem(
            name="final_holdout"
        )

    accuracy = _safe_float(
        _get_attr(
            result,
            "accuracy",
        )
    )

    if accuracy is None:
        metrics = _get_attr(
            result,
            "metrics",
            {},
        )

        if isinstance(
            metrics,
            dict,
        ):
            accuracy = _safe_float(
                metrics.get(
                    "accuracy"
                )
            )

    passed = _safe_bool(
        _get_attr(
            result,
            "passed_accuracy_gate",
        )
    )

    holdout_used_for_training = (
        _safe_bool(
            _get_attr(
                result,
                "holdout_used_for_training",
            )
        )
    )

    holdout_used_for_selection = (
        _safe_bool(
            _get_attr(
                result,
                "holdout_used_for_selection",
            )
        )
    )

    if (
        holdout_used_for_training is True
        or holdout_used_for_selection is True
    ):
        return EvidenceItem(
            name="final_holdout",
            status=EvidenceStatus.FAIL,
            score=accuracy,
            threshold=minimum_accuracy,
            metrics={
                "holdout_used_for_training": (
                    holdout_used_for_training
                ),
                "holdout_used_for_selection": (
                    holdout_used_for_selection
                ),
            },
            details=(
                "Final holdout was used during "
                "model development."
            ),
            critical=True,
            source="final_holdout",
        )

    if passed is None and accuracy is not None:
        passed = (
            accuracy
            >= minimum_accuracy
        )

    if passed is None:
        return EvidenceItem(
            name="final_holdout",
            status=EvidenceStatus.NOT_EVALUATED,
            threshold=minimum_accuracy,
            details=(
                "Final holdout accuracy could not "
                "be established."
            ),
            critical=True,
            source="final_holdout",
        )

    metrics = _base_metrics(
        result
    )

    if accuracy is not None:
        metrics[
            "accuracy"
        ] = accuracy

    metrics[
        "holdout_used_for_training"
    ] = False

    metrics[
        "holdout_used_for_selection"
    ] = False

    return EvidenceItem(
        name="final_holdout",
        status=_status_from_passed(
            passed
        ),
        score=accuracy,
        threshold=minimum_accuracy,
        metrics=metrics,
        details=(
            "Final holdout evaluated separately "
            "from model development."
        ),
        critical=True,
        source="final_holdout",
    )


# ---------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------


def adapt_calibration_result(
    result: Any,
    maximum_brier: float = 0.25,
    maximum_ece: float = 0.15,
) -> EvidenceItem:

    if result is None:
        return EvidenceItem(
            name="probability_calibration"
        )

    brier = _safe_float(
        _get_attr(
            result,
            "calibrated_brier",
        )
    )

    if brier is None:
        brier = _safe_float(
            _get_attr(
                result,
                "brier_score",
            )
        )

    ece = _safe_float(
        _get_attr(
            result,
            "calibrated_ece",
        )
    )

    if ece is None:
        ece = _safe_float(
            _get_attr(
                result,
                "ece",
            )
        )

    passed = _safe_bool(
        _get_attr(
            result,
            "passed",
        )
    )

    calibration_used_holdout = _safe_bool(
        _get_attr(
            result,
            "final_holdout_used",
        )
    )

    if calibration_used_holdout is True:
        return EvidenceItem(
            name="probability_calibration",
            status=EvidenceStatus.FAIL,
            score=brier,
            threshold=maximum_brier,
            metrics={
                "brier_score": brier,
                "ece": ece,
                "final_holdout_used": True,
            },
            details=(
                "Calibration used final holdout data."
            ),
            critical=True,
            source="calibration_research",
        )

    if passed is None:
        if (
            brier is not None
            and ece is not None
        ):
            passed = (
                brier
                <= maximum_brier
                and ece
                <= maximum_ece
            )

    if passed is None:
        return EvidenceItem(
            name="probability_calibration",
            status=EvidenceStatus.NOT_EVALUATED,
            threshold=maximum_brier,
            metrics={
                "brier_score": brier,
                "ece": ece,
            },
            details=(
                "Calibration quality could not "
                "be established."
            ),
            critical=True,
            source="calibration_research",
        )

    return EvidenceItem(
        name="probability_calibration",
        status=_status_from_passed(
            passed
        ),
        score=brier,
        threshold=maximum_brier,
        metrics={
            "brier_score": brier,
            "ece": ece,
        },
        details=(
            "Probability calibration evaluated "
            "on a dedicated calibration period."
        ),
        critical=True,
        source="calibration_research",
    )


# ---------------------------------------------------------------------
# Range validation
# ---------------------------------------------------------------------


def adapt_range_result(
    result: Any,
    minimum_coverage: float = 0.70,
    maximum_coverage: float = 0.95,
) -> EvidenceItem:

    if result is None:
        return EvidenceItem(
            name="target_range_validation"
        )

    coverage = _safe_float(
        _get_attr(
            result,
            "coverage",
        )
    )

    if coverage is None:
        coverage = _safe_float(
            _get_attr(
                result,
                "interval_coverage",
            )
        )

    passed = _safe_bool(
        _get_attr(
            result,
            "passed",
        )
    )

    final_holdout_used = _safe_bool(
        _get_attr(
            result,
            "final_holdout_used",
        )
    )

    if final_holdout_used is True:
        return EvidenceItem(
            name="target_range_validation",
            status=EvidenceStatus.FAIL,
            score=coverage,
            threshold=minimum_coverage,
            metrics={
                "coverage": coverage,
                "final_holdout_used": True,
            },
            details=(
                "Range validation used final holdout data."
            ),
            critical=True,
            source="range_research",
        )

    if passed is None and coverage is not None:
        passed = (
            minimum_coverage
            <= coverage
            <= maximum_coverage
        )

    if passed is None:
        return EvidenceItem(
            name="target_range_validation",
            status=EvidenceStatus.NOT_EVALUATED,
            threshold=minimum_coverage,
            metrics={
                "coverage": coverage,
            },
            details=(
                "Target-range validation could "
                "not be established."
            ),
            critical=True,
            source="range_research",
        )

    metrics = _base_metrics(
        result
    )

    if coverage is not None:
        metrics[
            "coverage"
        ] = coverage

    return EvidenceItem(
        name="target_range_validation",
        status=_status_from_passed(
            passed
        ),
        score=coverage,
        threshold=minimum_coverage,
        metrics=metrics,
        details=(
            "Out-of-sample target-range coverage "
            "was evaluated."
        ),
        critical=True,
        source="range_research",
    )


# ---------------------------------------------------------------------
# Regime validation
# ---------------------------------------------------------------------


def adapt_regime_result(
    result: Any,
    minimum_stability: float = 0.60,
) -> EvidenceItem:

    if result is None:
        return EvidenceItem(
            name="regime_validation"
        )

    stability = _safe_float(
        _get_attr(
            result,
            "stability_score",
        )
    )

    if stability is None:
        summary = _base_metrics(
            result
        )

        stability = _safe_float(
            summary.get(
                "stability_score"
            )
        )

    passed = _safe_bool(
        _get_attr(
            result,
            "passed",
        )
    )

    final_holdout_used = _safe_bool(
        _get_attr(
            result,
            "final_holdout_used",
        )
    )

    if final_holdout_used is True:
        return EvidenceItem(
            name="regime_validation",
            status=EvidenceStatus.FAIL,
            score=stability,
            threshold=minimum_stability,
            metrics={
                "stability_score": stability,
                "final_holdout_used": True,
            },
            details=(
                "Regime analysis used final holdout data."
            ),
            critical=True,
            source="regime_research",
        )

    if passed is None and stability is not None:
        passed = (
            stability
            >= minimum_stability
        )

    if passed is None:
        return EvidenceItem(
            name="regime_validation",
            status=EvidenceStatus.NOT_EVALUATED,
            threshold=minimum_stability,
            metrics={
                "stability_score": stability,
            },
            details=(
                "Regime stability could not "
                "be established."
            ),
            critical=True,
            source="regime_research",
        )

    metrics = _base_metrics(
        result
    )

    if stability is not None:
        metrics[
            "stability_score"
        ] = stability

    return EvidenceItem(
        name="regime_validation",
        status=_status_from_passed(
            passed
        ),
        score=stability,
        threshold=minimum_stability,
        metrics=metrics,
        details=(
            "Performance stability was evaluated "
            "across market and volatility regimes."
        ),
        critical=True,
        source="regime_research",
    )


# ---------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------


def adapt_backtest_result(
    result: Any,
    minimum_profit_factor: float = 1.20,
    minimum_sharpe: float = 0.80,
    maximum_drawdown: float = 0.30,
    minimum_trades: int = 30,
) -> EvidenceItem:

    if result is None:
        return EvidenceItem(
            name="backtest"
        )

    profit_factor = _safe_float(
        _get_attr(
            result,
            "profit_factor",
        )
    )

    sharpe = _safe_float(
        _get_attr(
            result,
            "sharpe",
        )
    )

    max_drawdown = _safe_float(
        _get_attr(
            result,
            "max_drawdown",
        )
    )

    trades = _safe_float(
        _get_attr(
            result,
            "trades",
        )
    )

    metrics = _base_metrics(
        result
    )

    if profit_factor is None:
        profit_factor = _safe_float(
            metrics.get(
                "profit_factor"
            )
        )

    if sharpe is None:
        sharpe = _safe_float(
            metrics.get(
                "sharpe"
            )
        )

    if max_drawdown is None:
        max_drawdown = _safe_float(
            metrics.get(
                "max_drawdown"
            )
        )

    if trades is None:
        trades = _safe_float(
            metrics.get(
                "trades"
            )
        )

    passed = _safe_bool(
        _get_attr(
            result,
            "passed",
        )
    )

    if passed is None:
        if (
            profit_factor is not None
            and sharpe is not None
            and max_drawdown is not None
            and trades is not None
        ):
            passed = (
                profit_factor
                >= minimum_profit_factor
                and sharpe
                >= minimum_sharpe
                and abs(max_drawdown)
                <= maximum_drawdown
                and trades
                >= minimum_trades
            )

    if passed is None:
        return EvidenceItem(
            name="backtest",
            status=EvidenceStatus.NOT_EVALUATED,
            metrics={
                "profit_factor": profit_factor,
                "sharpe": sharpe,
                "max_drawdown": max_drawdown,
                "trades": trades,
            },
            details=(
                "Backtest quality could not "
                "be established."
            ),
            critical=True,
            source="backtest_research",
        )

    return EvidenceItem(
        name="backtest",
        status=_status_from_passed(
            passed
        ),
        score=profit_factor,
        metrics={
            "profit_factor": profit_factor,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "trades": trades,
        },
        details=(
            "Trading performance evaluated "
            "with configured execution assumptions."
        ),
        critical=True,
        source="backtest_research",
    )


# ---------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------


def adapt_robustness_result(
    result: Any,
    minimum_score: float = 0.60,
) -> EvidenceItem:

    if result is None:
        return EvidenceItem(
            name="robustness"
        )

    score = _safe_float(
        _get_attr(
            result,
            "robustness_score",
        )
    )

    passed = _safe_bool(
        _get_attr(
            result,
            "passed",
        )
    )

    final_holdout_used = _safe_bool(
        _get_attr(
            result,
            "final_holdout_used",
        )
    )

    if final_holdout_used is True:
        return EvidenceItem(
            name="robustness",
            status=EvidenceStatus.FAIL,
            score=score,
            threshold=minimum_score,
            metrics={
                "robustness_score": score,
                "final_holdout_used": True,
            },
            details=(
                "Robustness analysis used "
                "final holdout data."
            ),
            critical=True,
            source="robustness_research",
        )

    if passed is None and score is not None:
        passed = (
            score
            >= minimum_score
        )

    if passed is None:
        return EvidenceItem(
            name="robustness",
            status=EvidenceStatus.NOT_EVALUATED,
            score=score,
            threshold=minimum_score,
            metrics={
                "robustness_score": score,
            },
            details=(
                "Robustness quality could not "
                "be established."
            ),
            critical=True,
            source="robustness_research",
        )

    metrics = _base_metrics(
        result
    )

    if score is not None:
        metrics[
            "robustness_score"
        ] = score

    return EvidenceItem(
        name="robustness",
        status=_status_from_passed(
            passed
        ),
        score=score,
        threshold=minimum_score,
        metrics=metrics,
        details=(
            "Performance was stress-tested "
            "under unfavorable assumptions."
        ),
        critical=True,
        source="robustness_research",
    )


# ---------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------


def adapt_leakage_result(
    result: Any,
) -> EvidenceItem:

    if result is None:
        return EvidenceItem(
            name="leakage_audit"
        )

    passed = _safe_bool(
        _get_attr(
            result,
            "passed",
        )
    )

    if passed is None:
        passed = _safe_bool(
            _get_attr(
                result,
                "is_clean",
            )
        )

    if passed is None:
        passed = _safe_bool(
            _get_attr(
                result,
                "leakage_free",
            )
        )

    if passed is None:
        return EvidenceItem(
            name="leakage_audit",
            status=EvidenceStatus.NOT_EVALUATED,
            details=(
                "Leakage-free status could not "
                "be established."
            ),
            critical=True,
            source="leakage_audit",
        )

    metrics = _base_metrics(
        result
    )

    return EvidenceItem(
        name="leakage_audit",
        status=_status_from_passed(
            passed
        ),
        metrics=metrics,
        details=(
            "Formal temporal and feature leakage "
            "audit."
        ),
        critical=True,
        source="leakage_audit",
    )


# ---------------------------------------------------------------------
# Generic adapter
# ---------------------------------------------------------------------


def adapt_stage_result(
    stage_name: str,
    result: Any,
) -> EvidenceItem:

    adapters = {
        "walk_forward_validation":
            adapt_walk_forward_result,
        "final_holdout":
            adapt_holdout_result,
        "calibration":
            adapt_calibration_result,
        "range_validation":
            adapt_range_result,
        "regime_validation":
            adapt_regime_result,
        "backtest":
            adapt_backtest_result,
        "robustness":
            adapt_robustness_result,
        "leakage_audit":
            adapt_leakage_result,
    }

    adapter = adapters.get(
        stage_name
    )

    if adapter is None:
        raise ValueError(
            f"No adapter exists for stage '{stage_name}'."
        )

    return adapter(
        result
    )


__all__ = [
    "adapt_walk_forward_result",
    "adapt_holdout_result",
    "adapt_calibration_result",
    "adapt_range_result",
    "adapt_regime_result",
    "adapt_backtest_result",
    "adapt_robustness_result",
    "adapt_leakage_result",
    "adapt_stage_result",
]
