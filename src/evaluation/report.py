"""
Unified research evaluation report.

This module consolidates model-development and strategy-evaluation
results into one structured report.

The report can contain:

    - validation performance
    - final holdout performance
    - calibration
    - target-range validation
    - regime stability
    - walk-forward validation
    - backtesting
    - robustness
    - production approval

IMPORTANT
---------
This module does not train models and does not modify evaluation
results.

It is intentionally a reporting layer so that research conclusions
remain reproducible and auditable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional
import json

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Section result
# ----------------------------------------------------------------------


@dataclass
class EvaluationSection:
    """
    One section of an evaluation report.
    """

    name: str

    status: str = "NOT_EVALUATED"

    metrics: dict[str, Any] = field(
        default_factory=dict
    )

    warnings: list[str] = field(
        default_factory=list
    )

    errors: list[str] = field(
        default_factory=list
    )

    notes: list[str] = field(
        default_factory=list
    )

    def passed(self) -> bool:
        return self.status.upper() == "PASS"

    def failed(self) -> bool:
        return self.status.upper() == "FAIL"

    def evaluated(self) -> bool:
        return self.status.upper() not in {
            "NOT_EVALUATED",
            "NOT_AVAILABLE",
        }


# ----------------------------------------------------------------------
# Overall report
# ----------------------------------------------------------------------


@dataclass
class EvaluationReport:
    """
    Complete evaluation report for one research experiment/model.
    """

    experiment_id: str

    symbol: Optional[str] = None

    timeframe: Optional[str] = None

    horizon: Optional[int] = None

    target: Optional[str] = None

    model_id: Optional[str] = None

    generated_at: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    sections: dict[
        str,
        EvaluationSection
    ] = field(
        default_factory=dict
    )

    overall_status: str = "RESEARCH"

    production_ready: bool = False

    overall_score: Optional[float] = None

    critical_failures: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    notes: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    # ------------------------------------------------------------------
    # Section management
    # ------------------------------------------------------------------

    def add_section(
        self,
        section: EvaluationSection,
    ) -> None:
        self.sections[
            section.name
        ] = section

    def get_section(
        self,
        name: str,
    ) -> Optional[
        EvaluationSection
    ]:
        return self.sections.get(
            name
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def evaluated_sections(
        self,
    ) -> list[
        EvaluationSection
    ]:
        return [
            section
            for section in self.sections.values()
            if section.evaluated()
        ]

    def failed_sections(
        self,
    ) -> list[
        EvaluationSection
    ]:
        return [
            section
            for section in self.sections.values()
            if section.failed()
        ]

    def passed_sections(
        self,
    ) -> list[
        EvaluationSection
    ]:
        return [
            section
            for section in self.sections.values()
            if section.passed()
        ]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "horizon": self.horizon,
            "target": self.target,
            "model_id": self.model_id,
            "generated_at": self.generated_at,
            "sections": {
                name: asdict(
                    section
                )
                for name, section
                in self.sections.items()
            },
            "overall_status": self.overall_status,
            "production_ready": self.production_ready,
            "overall_score": self.overall_score,
            "critical_failures": self.critical_failures,
            "warnings": self.warnings,
            "notes": self.notes,
            "metadata": self.metadata,
        }

    def to_json(
        self,
        *,
        indent: int = 2,
    ) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            default=_json_default,
        )

    def save(
        self,
        path: str | Path,
    ) -> Path:
        destination = Path(
            path
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination.write_text(
            self.to_json(),
            encoding="utf-8",
        )

        return destination

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def section_table(
        self,
    ) -> pd.DataFrame:
        rows = []

        for section in self.sections.values():
            row = {
                "section": section.name,
                "status": section.status,
                "evaluated": section.evaluated(),
                "metrics": len(
                    section.metrics
                ),
                "warnings": len(
                    section.warnings
                ),
                "errors": len(
                    section.errors
                ),
            }

            rows.append(
                row
            )

        return pd.DataFrame(
            rows
        )

    def metric_table(
        self,
    ) -> pd.DataFrame:
        rows = []

        for section in self.sections.values():
            for metric, value in section.metrics.items():
                rows.append(
                    {
                        "section": section.name,
                        "metric": metric,
                        "value": value,
                    }
                )

        return pd.DataFrame(
            rows
        )

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------

    def summary(
        self,
    ) -> str:
        lines = [
            "AI Swing Analyser — Evaluation Report",
            "",
            f"Experiment: {self.experiment_id}",
        ]

        if self.symbol:
            lines.append(
                f"Symbol: {self.symbol}"
            )

        if self.timeframe:
            lines.append(
                f"Timeframe: {self.timeframe}"
            )

        if self.horizon is not None:
            lines.append(
                f"Horizon: {self.horizon}"
            )

        if self.target:
            lines.append(
                f"Target: {self.target}"
            )

        lines.extend(
            [
                "",
                f"Overall status: {self.overall_status}",
                f"Production ready: {self.production_ready}",
            ]
        )

        if self.overall_score is not None:
            lines.append(
                f"Evaluation score: "
                f"{self.overall_score:.4f}"
            )

        lines.append(
            ""
        )

        for section in self.sections.values():
            lines.append(
                f"[{section.status}] "
                f"{section.name}"
            )

        if self.critical_failures:
            lines.extend(
                [
                    "",
                    "Critical failures:",
                ]
            )

            lines.extend(
                f"- {item}"
                for item in self.critical_failures
            )

        if self.warnings:
            lines.extend(
                [
                    "",
                    "Warnings:",
                ]
            )

            lines.extend(
                f"- {item}"
                for item in self.warnings
            )

        return "\n".join(
            lines
        )


# ----------------------------------------------------------------------
# Builder
# ----------------------------------------------------------------------


class EvaluationReportBuilder:
    """
    Builder for constructing a unified evaluation report.

    The builder accepts dictionaries, dataclasses, or result objects
    from the existing evaluation modules.

    It intentionally performs conservative extraction: unknown fields
    are preserved under metadata rather than being interpreted as
    successful evaluation gates.
    """

    def __init__(
        self,
        *,
        experiment_id: str,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        horizon: Optional[int] = None,
        target: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> None:
        if not experiment_id.strip():
            raise ValueError(
                "experiment_id cannot be empty."
            )

        self.report = EvaluationReport(
            experiment_id=experiment_id,
            symbol=symbol,
            timeframe=timeframe,
            horizon=horizon,
            target=target,
            model_id=model_id,
        )

    # ------------------------------------------------------------------
    # Generic section
    # ------------------------------------------------------------------

    def add_section(
        self,
        name: str,
        *,
        status: str,
        metrics: Optional[
            Mapping[str, Any]
        ] = None,
        warnings: Optional[
            list[str]
        ] = None,
        errors: Optional[
            list[str]
        ] = None,
        notes: Optional[
            list[str]
        ] = None,
    ) -> "EvaluationReportBuilder":

        normalized_status = (
            str(status)
            .upper()
        )

        valid_statuses = {
            "PASS",
            "FAIL",
            "WARNING",
            "NOT_EVALUATED",
            "NOT_AVAILABLE",
        }

        if (
            normalized_status
            not in valid_statuses
        ):
            raise ValueError(
                f"Invalid evaluation status: {status}"
            )

        section = EvaluationSection(
            name=name,
            status=normalized_status,
            metrics=dict(
                metrics or {}
            ),
            warnings=list(
                warnings or []
            ),
            errors=list(
                errors or []
            ),
            notes=list(
                notes or []
            ),
        )

        self.report.add_section(
            section
        )

        return self

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def add_validation(
        self,
        result: Any,
    ) -> "EvaluationReportBuilder":
        """
        Add chronological/walk-forward validation results.
        """

        metrics = _extract_metrics(
            result
        )

        status = _infer_status(
            result,
            preferred_keys=[
                "passed",
                "overall_passed",
                "validation_passed",
            ],
        )

        if status == "NOT_EVALUATED":
            accuracy = _find_numeric(
                metrics,
                [
                    "mean_accuracy",
                    "accuracy",
                    "validation_accuracy",
                ],
            )

            if accuracy is not None:
                status = (
                    "PASS"
                    if accuracy >= 0.95
                    else "FAIL"
                )

        self.add_section(
            "validation",
            status=status,
            metrics=metrics,
            notes=[
                "Validation accuracy is only one "
                "research gate; it is not sufficient "
                "for production approval."
            ],
        )

        return self

    # ------------------------------------------------------------------
    # Final holdout
    # ------------------------------------------------------------------

    def add_final_holdout(
        self,
        result: Any,
    ) -> "EvaluationReportBuilder":
        metrics = _extract_metrics(
            result
        )

        accuracy = _find_numeric(
            metrics,
            [
                "accuracy",
                "test_accuracy",
                "final_holdout_accuracy",
            ],
        )

        status = "NOT_EVALUATED"

        if accuracy is not None:
            status = (
                "PASS"
                if accuracy >= 0.95
                else "FAIL"
            )

        self.add_section(
            "final_holdout",
            status=status,
            metrics=metrics,
            notes=[
                "The final holdout must remain untouched "
                "until model selection and tuning are frozen."
            ],
        )

        return self

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def add_calibration(
        self,
        result: Any,
    ) -> "EvaluationReportBuilder":
        metrics = _extract_metrics(
            result
        )

        brier = _find_numeric(
            metrics,
            [
                "brier",
                "brier_score",
                "calibrated_brier",
            ],
        )

        ece = _find_numeric(
            metrics,
            [
                "ece",
                "expected_calibration_error",
                "calibrated_ece",
            ],
        )

        passed = _find_boolean(
            result,
            [
                "passed",
                "calibration_passed",
            ],
        )

        if passed is not None:
            status = (
                "PASS"
                if passed
                else "FAIL"
            )
        elif (
            brier is not None
            and ece is not None
        ):
            status = (
                "PASS"
                if brier <= 0.25
                and ece <= 0.15
                else "FAIL"
            )
        else:
            status = "NOT_EVALUATED"

        self.add_section(
            "calibration",
            status=status,
            metrics=metrics,
        )

        return self

    # ------------------------------------------------------------------
    # Range validation
    # ------------------------------------------------------------------

    def add_range_validation(
        self,
        result: Any,
    ) -> "EvaluationReportBuilder":
        metrics = _extract_metrics(
            result
        )

        coverage = _find_numeric(
            metrics,
            [
                "coverage",
                "interval_coverage",
                "mean_coverage",
            ],
        )

        if coverage is None:
            status = "NOT_EVALUATED"
        else:
            status = (
                "PASS"
                if 0.70 <= coverage <= 0.95
                else "FAIL"
            )

        self.add_section(
            "range_validation",
            status=status,
            metrics=metrics,
            notes=[
                "Target-price range coverage must be "
                "evaluated out-of-sample."
            ],
        )

        return self

    # ------------------------------------------------------------------
    # Regime stability
    # ------------------------------------------------------------------

    def add_regime_validation(
        self,
        result: Any,
    ) -> "EvaluationReportBuilder":
        metrics = _extract_metrics(
            result
        )

        score = _find_numeric(
            metrics,
            [
                "stability_score",
                "regime_stability",
                "overall_stability",
            ],
        )

        if score is None:
            status = "NOT_EVALUATED"
        else:
            status = (
                "PASS"
                if score >= 0.60
                else "FAIL"
            )

        self.add_section(
            "regime_stability",
            status=status,
            metrics=metrics,
        )

        return self

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    def add_backtest(
        self,
        result: Any,
    ) -> "EvaluationReportBuilder":
        metrics = _extract_metrics(
            result
        )

        profit_factor = _find_numeric(
            metrics,
            [
                "profit_factor",
                "pf",
            ],
        )

        sharpe = _find_numeric(
            metrics,
            [
                "sharpe_ratio",
                "sharpe",
            ],
        )

        trades = _find_numeric(
            metrics,
            [
                "trades",
                "trade_count",
                "number_of_trades",
            ],
        )

        max_drawdown = _find_numeric(
            metrics,
            [
                "max_drawdown",
                "maximum_drawdown",
            ],
        )

        if all(
            value is not None
            for value in (
                profit_factor,
                sharpe,
                trades,
                max_drawdown,
            )
        ):
            drawdown_ok = (
                abs(max_drawdown)
                <= 0.30
            )

            status = (
                "PASS"
                if profit_factor >= 1.20
                and sharpe >= 0.80
                and trades >= 30
                and drawdown_ok
                else "FAIL"
            )
        else:
            status = "NOT_EVALUATED"

        self.add_section(
            "backtest",
            status=status,
            metrics=metrics,
            notes=[
                "Backtest results must be based on "
                "out-of-sample predictions rather than "
                "in-sample model predictions."
            ],
        )

        return self

    # ------------------------------------------------------------------
    # Robustness
    # ------------------------------------------------------------------

    def add_robustness(
        self,
        result: Any,
    ) -> "EvaluationReportBuilder":
        metrics = _extract_metrics(
            result
        )

        score = _find_numeric(
            metrics,
            [
                "robustness_score",
                "score",
            ],
        )

        passed = _find_boolean(
            result,
            [
                "passed",
                "overall_passed",
                "robustness_passed",
            ],
        )

        if passed is not None:
            status = (
                "PASS"
                if passed
                else "FAIL"
            )
        elif score is not None:
            status = (
                "PASS"
                if score >= 0.60
                else "FAIL"
            )
        else:
            status = "NOT_EVALUATED"

        self.add_section(
            "robustness",
            status=status,
            metrics=metrics,
        )

        return self

    # ------------------------------------------------------------------
    # Approval
    # ------------------------------------------------------------------

    def add_approval(
        self,
        result: Any,
    ) -> "EvaluationReportBuilder":
        metrics = _extract_metrics(
            result
        )

        status_value = _extract_value(
            result,
            [
                "status",
                "approval_status",
                "overall_status",
            ],
        )

        if status_value is None:
            status = "NOT_EVALUATED"
        else:
            normalized = str(
                status_value
            ).upper()

            if normalized in {
                "APPROVED",
                "PASS",
            }:
                status = "PASS"
            elif normalized in {
                "REJECTED",
                "FAIL",
            }:
                status = "FAIL"
            else:
                status = "WARNING"

        self.add_section(
            "production_approval",
            status=status,
            metrics=metrics,
        )

        return self

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def add_metadata(
        self,
        values: Mapping[str, Any],
    ) -> "EvaluationReportBuilder":
        self.report.metadata.update(
            dict(values)
        )

        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
    ) -> EvaluationReport:
        self._finalize()

        return self.report

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    def _finalize(
        self,
    ) -> None:
        sections = list(
            self.report.sections.values()
        )

        if not sections:
            self.report.overall_status = (
                "RESEARCH"
            )
            self.report.production_ready = False
            self.report.overall_score = None
            return

        evaluated = [
            section
            for section in sections
            if section.evaluated()
        ]

        failed = [
            section
            for section in sections
            if section.failed()
        ]

        self.report.critical_failures = []

        for section in failed:
            self.report.critical_failures.append(
                f"{section.name}: evaluation failed"
            )

            self.report.critical_failures.extend(
                section.errors
            )

        # Collect warnings without duplicating them.
        warning_values: list[str] = []

        for section in sections:
            warning_values.extend(
                section.warnings
            )

        self.report.warnings = list(
            dict.fromkeys(
                warning_values
            )
        )

        if failed:
            self.report.overall_status = (
                "REJECTED"
            )
            self.report.production_ready = False

        elif not evaluated:
            self.report.overall_status = (
                "RESEARCH"
            )
            self.report.production_ready = False

        elif len(evaluated) < len(sections):
            self.report.overall_status = (
                "RESEARCH"
            )
            self.report.production_ready = False

        else:
            self.report.overall_status = (
                "PASSED_RESEARCH_GATES"
            )

            # IMPORTANT:
            # This does not independently approve a production model.
            # Production approval remains the responsibility of the
            # dedicated approval engine.
            self.report.production_ready = False

        self.report.overall_score = (
            self._calculate_score(
                sections
            )
        )

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_score(
        sections: list[
            EvaluationSection
        ],
    ) -> Optional[float]:
        evaluated = [
            section
            for section in sections
            if section.evaluated()
        ]

        if not evaluated:
            return None

        scores = []

        for section in evaluated:
            if section.status == "PASS":
                scores.append(
                    1.0
                )
            elif section.status == "WARNING":
                scores.append(
                    0.5
                )
            elif section.status == "FAIL":
                scores.append(
                    0.0
                )

        if not scores:
            return None

        return float(
            np.mean(
                scores
            )
        )


# ----------------------------------------------------------------------
# Result extraction helpers
# ----------------------------------------------------------------------


def _extract_value(
    result: Any,
    keys: list[str],
) -> Any:
    if result is None:
        return None

    if isinstance(
        result,
        Mapping,
    ):
        for key in keys:
            if key in result:
                return result[key]

    for key in keys:
        if hasattr(
            result,
            key,
        ):
            return getattr(
                result,
                key,
            )

    return None


def _extract_metrics(
    result: Any,
) -> dict[str, Any]:
    """
    Extract a flat metric dictionary without mutating the source.
    """

    if result is None:
        return {}

    if isinstance(
        result,
        Mapping,
    ):
        if isinstance(
            result.get(
                "metrics"
            ),
            Mapping,
        ):
            return _make_json_safe_dict(
                result["metrics"]
            )

        return _make_json_safe_dict(
            dict(result)
        )

    if hasattr(
        result,
        "metrics",
    ):
        metrics = getattr(
            result,
            "metrics",
        )

        if isinstance(
            metrics,
            Mapping,
        ):
            return _make_json_safe_dict(
                dict(metrics)
            )

    # Dataclass/object fallback.
    try:
        values = asdict(
            result
        )

        if isinstance(
            values,
            dict,
        ):
            return _make_json_safe_dict(
                values
            )

    except Exception:
        pass

    if hasattr(
        result,
        "__dict__",
    ):
        return _make_json_safe_dict(
            dict(
                result.__dict__
            )
        )

    return {}


def _find_numeric(
    metrics: Mapping[str, Any],
    keys: list[str],
) -> Optional[float]:
    normalized = {
        str(key).lower(): value
        for key, value
        in metrics.items()
    }

    for key in keys:
        value = normalized.get(
            key.lower()
        )

        if value is None:
            continue

        try:
            numeric = float(
                value
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if np.isfinite(
            numeric
        ):
            return numeric

    return None


def _find_boolean(
    result: Any,
    keys: list[str],
) -> Optional[bool]:
    value = _extract_value(
        result,
        keys,
    )

    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        (np.bool_,),
    ):
        return bool(
            value
        )

    return None


def _infer_status(
    result: Any,
    *,
    preferred_keys: list[str],
) -> str:
    value = _find_boolean(
        result,
        preferred_keys,
    )

    if value is None:
        return "NOT_EVALUATED"

    return (
        "PASS"
        if value
        else "FAIL"
    )


# ----------------------------------------------------------------------
# JSON safety
# ----------------------------------------------------------------------


def _make_json_safe_dict(
    values: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        str(key): _json_safe(
            value
        )
        for key, value
        in values.items()
    }


def _json_safe(
    value: Any,
) -> Any:
    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        if isinstance(
            value,
            float,
        ) and not np.isfinite(
            value
        ):
            return None

        return value

    if isinstance(
        value,
        (
            np.integer,
        ),
    ):
        return int(
            value
        )

    if isinstance(
        value,
        (
            np.floating,
        ),
    ):
        value = float(
            value
        )

        return (
            value
            if np.isfinite(
                value
            )
            else None
        )

    if isinstance(
        value,
        (
            np.bool_,
        ),
    ):
        return bool(
            value
        )

    if isinstance(
        value,
        (
            pd.Timestamp,
            datetime,
        ),
    ):
        return value.isoformat()

    if isinstance(
        value,
        pd.Series,
    ):
        return [
            _json_safe(
                item
            )
            for item in value.tolist()
        ]

    if isinstance(
        value,
        pd.Index,
    ):
        return [
            _json_safe(
                item
            )
            for item in value.tolist()
        ]

    if isinstance(
        value,
        pd.DataFrame,
    ):
        return [
            {
                str(key): _json_safe(
                    item
                )
                for key, item in row.items()
            }
            for row in value.to_dict(
                orient="records"
            )
        ]

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key): _json_safe(
                item
            )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [
            _json_safe(
                item
            )
            for item in value
        ]

    if hasattr(
        value,
        "value",
    ):
        try:
            return _json_safe(
                value.value
            )
        except Exception:
            pass

    return str(
        value
    )


def _json_default(
    value: Any,
) -> Any:
    return _json_safe(
        value
    )


# ----------------------------------------------------------------------
# Convenience API
# ----------------------------------------------------------------------


def build_evaluation_report(
    *,
    experiment_id: str,
    validation: Any = None,
    final_holdout: Any = None,
    calibration: Any = None,
    range_validation: Any = None,
    regime_validation: Any = None,
    backtest: Any = None,
    robustness: Any = None,
    approval: Any = None,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    horizon: Optional[int] = None,
    target: Optional[str] = None,
    model_id: Optional[str] = None,
    metadata: Optional[
        Mapping[str, Any]
    ] = None,
) -> EvaluationReport:
    """
    Build a complete evaluation report from available results.
    """

    builder = EvaluationReportBuilder(
        experiment_id=experiment_id,
        symbol=symbol,
        timeframe=timeframe,
        horizon=horizon,
        target=target,
        model_id=model_id,
    )

    if validation is not None:
        builder.add_validation(
            validation
        )

    if final_holdout is not None:
        builder.add_final_holdout(
            final_holdout
        )

    if calibration is not None:
        builder.add_calibration(
            calibration
        )

    if range_validation is not None:
        builder.add_range_validation(
            range_validation
        )

    if regime_validation is not None:
        builder.add_regime_validation(
            regime_validation
        )

    if backtest is not None:
        builder.add_backtest(
            backtest
        )

    if robustness is not None:
        builder.add_robustness(
            robustness
        )

    if approval is not None:
        builder.add_approval(
            approval
        )

    if metadata is not None:
        builder.add_metadata(
            metadata
        )

    return builder.build()


__all__ = [
    "EvaluationSection",
    "EvaluationReport",
    "EvaluationReportBuilder",
    "build_evaluation_report",
]
