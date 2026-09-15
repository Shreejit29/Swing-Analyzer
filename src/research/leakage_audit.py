"""
AI Swing Analyser - Research Leakage Audit.

Provides explicit pre-training checks for data leakage.

The audit is intentionally conservative.

A model should never be considered valid merely because its accuracy
is high. If information from the future has entered the feature matrix,
the experiment must be rejected.

Checks include:

    - chronological ordering
    - duplicate timestamps
    - feature/target overlap
    - suspicious future-looking feature names
    - non-numeric feature values
    - infinite values
    - target availability
    - temporal separation between development and holdout data
    - feature mutation checks

This module does not train models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Result types
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class LeakageFinding:
    """
    One leakage-audit finding.
    """

    check: str
    status: str
    message: str
    severity: str = "ERROR"


@dataclass
class LeakageAuditReport:
    """
    Complete leakage-audit result.
    """

    passed: bool

    findings: list[LeakageFinding] = field(
        default_factory=list
    )

    checked_rows: int = 0

    checked_features: int = 0

    checked_targets: int = 0

    development_end: pd.Timestamp | None = None

    holdout_start: pd.Timestamp | None = None

    metadata: dict[str, object] = field(
        default_factory=dict
    )

    @property
    def errors(self) -> list[LeakageFinding]:
        return [
            finding
            for finding in self.findings
            if finding.severity.upper()
            == "ERROR"
            and finding.status.upper()
            == "FAIL"
        ]

    @property
    def warnings(self) -> list[LeakageFinding]:
        return [
            finding
            for finding in self.findings
            if finding.severity.upper()
            == "WARNING"
        ]

    def summary(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "rows": self.checked_rows,
            "features": self.checked_features,
            "targets": self.checked_targets,
            "errors": len(
                self.errors
            ),
            "warnings": len(
                self.warnings
            ),
            "development_end": self.development_end,
            "holdout_start": self.holdout_start,
        }


# ----------------------------------------------------------------------
# Audit configuration
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class LeakageAuditConfig:
    """
    Configuration for leakage detection.
    """

    suspicious_tokens: tuple[str, ...] = (
        "future",
        "target",
        "forward",
        "next_day",
        "nextday",
        "next_week",
        "nextweek",
        "lead",
    )

    allow_feature_names: tuple[str, ...] = (
        "Target_RSI",
    )

    minimum_feature_count: int = 1

    require_datetime_index: bool = True

    require_chronological_order: bool = True

    reject_duplicate_timestamps: bool = True

    reject_non_numeric_features: bool = True

    reject_infinite_features: bool = True

    require_targets: bool = True


# ----------------------------------------------------------------------
# Leakage auditor
# ----------------------------------------------------------------------


class LeakageAuditor:
    """
    Performs conservative leakage checks on a research dataset.
    """

    def __init__(
        self,
        config: LeakageAuditConfig | None = None,
    ) -> None:

        self.config = (
            config
            or LeakageAuditConfig()
        )

    # ------------------------------------------------------------------
    # Complete audit
    # ------------------------------------------------------------------

    def audit(
        self,
        dataframe: pd.DataFrame,
        feature_columns: Sequence[str],
        target_columns: Sequence[str],
    ) -> LeakageAuditReport:
        """
        Run all static leakage checks.
        """

        findings: list[
            LeakageFinding
        ] = []

        self._check_dataframe(
            dataframe,
            findings,
        )

        self._check_feature_columns(
            dataframe,
            feature_columns,
            findings,
        )

        self._check_target_columns(
            dataframe,
            target_columns,
            findings,
        )

        self._check_feature_target_overlap(
            feature_columns,
            target_columns,
            findings,
        )

        self._check_feature_names(
            feature_columns,
            findings,
        )

        self._check_feature_values(
            dataframe,
            feature_columns,
            findings,
        )

        failures = [
            finding
            for finding in findings
            if finding.status.upper()
            == "FAIL"
            and finding.severity.upper()
            == "ERROR"
        ]

        return LeakageAuditReport(
            passed=len(
                failures
            )
            == 0,
            findings=findings,
            checked_rows=len(
                dataframe
            ),
            checked_features=len(
                feature_columns
            ),
            checked_targets=len(
                target_columns
            ),
            metadata={
                "audit_version": "1.0",
                "suspicious_tokens": list(
                    self.config.suspicious_tokens
                ),
            },
        )

    # ------------------------------------------------------------------
    # Dataframe checks
    # ------------------------------------------------------------------

    def _check_dataframe(
        self,
        dataframe: pd.DataFrame,
        findings: list[LeakageFinding],
    ) -> None:

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            findings.append(
                LeakageFinding(
                    check="dataframe_type",
                    status="FAIL",
                    message=(
                        "Research dataset is not a pandas DataFrame."
                    ),
                )
            )
            return

        if dataframe.empty:
            findings.append(
                LeakageFinding(
                    check="empty_dataset",
                    status="FAIL",
                    message=(
                        "Research dataset is empty."
                    ),
                )
            )
            return

        if self.config.require_datetime_index:

            if not isinstance(
                dataframe.index,
                pd.DatetimeIndex,
            ):
                findings.append(
                    LeakageFinding(
                        check="datetime_index",
                        status="FAIL",
                        message=(
                            "Research dataset must use a DatetimeIndex."
                        ),
                    )
                )
            else:
                findings.append(
                    LeakageFinding(
                        check="datetime_index",
                        status="PASS",
                        message=(
                            "Dataset uses a DatetimeIndex."
                        ),
                        severity="INFO",
                    )
                )

        if (
            self.config.reject_duplicate_timestamps
            and dataframe.index.has_duplicates
        ):
            findings.append(
                LeakageFinding(
                    check="duplicate_timestamps",
                    status="FAIL",
                    message=(
                        "Duplicate timestamps detected."
                    ),
                )
            )
        else:
            findings.append(
                LeakageFinding(
                    check="duplicate_timestamps",
                    status="PASS",
                    message=(
                        "No duplicate timestamps detected."
                    ),
                    severity="INFO",
                )
            )

        if (
            self.config.require_chronological_order
            and not dataframe.index.is_monotonic_increasing
        ):
            findings.append(
                LeakageFinding(
                    check="chronological_order",
                    status="FAIL",
                    message=(
                        "Dataset timestamps are not monotonically increasing."
                    ),
                )
            )
        else:
            findings.append(
                LeakageFinding(
                    check="chronological_order",
                    status="PASS",
                    message=(
                        "Dataset is chronologically ordered."
                    ),
                    severity="INFO",
                )
            )

    # ------------------------------------------------------------------
    # Feature checks
    # ------------------------------------------------------------------

    def _check_feature_columns(
        self,
        dataframe: pd.DataFrame,
        feature_columns: Sequence[str],
        findings: list[LeakageFinding],
    ) -> None:

        if (
            len(feature_columns)
            < self.config.minimum_feature_count
        ):
            findings.append(
                LeakageFinding(
                    check="feature_count",
                    status="FAIL",
                    message=(
                        "Insufficient feature columns."
                    ),
                )
            )
            return

        missing = [
            column
            for column in feature_columns
            if column not in dataframe.columns
        ]

        if missing:
            findings.append(
                LeakageFinding(
                    check="feature_presence",
                    status="FAIL",
                    message=(
                        "Feature columns missing from dataset: "
                        f"{sorted(missing)}"
                    ),
                )
            )
        else:
            findings.append(
                LeakageFinding(
                    check="feature_presence",
                    status="PASS",
                    message=(
                        "All feature columns exist in the dataset."
                    ),
                    severity="INFO",
                )
            )

    # ------------------------------------------------------------------
    # Target checks
    # ------------------------------------------------------------------

    def _check_target_columns(
        self,
        dataframe: pd.DataFrame,
        target_columns: Sequence[str],
        findings: list[LeakageFinding],
    ) -> None:

        if (
            self.config.require_targets
            and not target_columns
        ):
            findings.append(
                LeakageFinding(
                    check="target_presence",
                    status="FAIL",
                    message=(
                        "No target columns were supplied."
                    ),
                )
            )
            return

        missing = [
            column
            for column in target_columns
            if column not in dataframe.columns
        ]

        if missing:
            findings.append(
                LeakageFinding(
                    check="target_presence",
                    status="FAIL",
                    message=(
                        "Target columns missing from dataset: "
                        f"{sorted(missing)}"
                    ),
                )
            )
        else:
            findings.append(
                LeakageFinding(
                    check="target_presence",
                    status="PASS",
                    message=(
                        "All target columns exist in the dataset."
                    ),
                    severity="INFO",
                )
            )

    # ------------------------------------------------------------------
    # Feature / target overlap
    # ------------------------------------------------------------------

    @staticmethod
    def _check_feature_target_overlap(
        feature_columns: Sequence[str],
        target_columns: Sequence[str],
        findings: list[LeakageFinding],
    ) -> None:

        overlap = (
            set(feature_columns)
            & set(target_columns)
        )

        if overlap:
            findings.append(
                LeakageFinding(
                    check="feature_target_overlap",
                    status="FAIL",
                    message=(
                        "Target columns are present in the feature set: "
                        f"{sorted(overlap)}"
                    ),
                )
            )
        else:
            findings.append(
                LeakageFinding(
                    check="feature_target_overlap",
                    status="PASS",
                    message=(
                        "Feature and target columns are separated."
                    ),
                    severity="INFO",
                )
            )

    # ------------------------------------------------------------------
    # Feature-name leakage
    # ------------------------------------------------------------------

    def _check_feature_names(
        self,
        feature_columns: Sequence[str],
        findings: list[LeakageFinding],
    ) -> None:

        suspicious = []

        allowed = {
            value.lower()
            for value in self.config.allow_feature_names
        }

        for column in feature_columns:

            name = str(
                column
            )

            lower = name.lower()

            if lower in allowed:
                continue

            if any(
                token in lower
                for token in self.config.suspicious_tokens
            ):
                suspicious.append(
                    name
                )

        if suspicious:
            findings.append(
                LeakageFinding(
                    check="future_feature_names",
                    status="FAIL",
                    message=(
                        "Potential forward-looking feature names detected: "
                        f"{sorted(suspicious)}"
                    ),
                )
            )
        else:
            findings.append(
                LeakageFinding(
                    check="future_feature_names",
                    status="PASS",
                    message=(
                        "No suspicious future-looking feature names detected."
                    ),
                    severity="INFO",
                )
            )

    # ------------------------------------------------------------------
    # Feature values
    # ------------------------------------------------------------------

    def _check_feature_values(
        self,
        dataframe: pd.DataFrame,
        feature_columns: Sequence[str],
        findings: list[LeakageFinding],
    ) -> None:

        for column in feature_columns:

            if column not in dataframe.columns:
                continue

            series = dataframe[
                column
            ]

            if (
                self.config.reject_non_numeric_features
                and not pd.api.types.is_numeric_dtype(
                    series
                )
            ):
                findings.append(
                    LeakageFinding(
                        check="numeric_feature",
                        status="FAIL",
                        message=(
                            f"Feature '{column}' is not numeric."
                        ),
                    )
                )

            if (
                self.config.reject_infinite_features
                and pd.api.types.is_numeric_dtype(
                    series
                )
            ):
                values = series.to_numpy(
                    dtype=float
                )

                if np.isinf(
                    values
                ).any():
                    findings.append(
                        LeakageFinding(
                            check="infinite_feature",
                            status="FAIL",
                            message=(
                                f"Feature '{column}' contains infinite values."
                            ),
                        )
                    )

        if not any(
            finding.check
            in {
                "numeric_feature",
                "infinite_feature",
            }
            and finding.status == "FAIL"
            for finding in findings
        ):
            findings.append(
                LeakageFinding(
                    check="feature_values",
                    status="PASS",
                    message=(
                        "Feature values passed numeric and finite-value checks."
                    ),
                    severity="INFO",
                )
            )

    # ------------------------------------------------------------------
    # Temporal holdout audit
    # ------------------------------------------------------------------

    @staticmethod
    def audit_temporal_separation(
        development_data: pd.DataFrame,
        holdout_data: pd.DataFrame,
        *,
        embargo: int = 0,
    ) -> LeakageAuditReport:
        """
        Verify that development data cannot contain observations from
        the final holdout period.

        This does not replace purged walk-forward validation.

        It is an additional safety check.
        """

        findings: list[
            LeakageFinding
        ] = []

        if (
            not isinstance(
                development_data.index,
                pd.DatetimeIndex,
            )
            or not isinstance(
                holdout_data.index,
                pd.DatetimeIndex,
            )
        ):
            findings.append(
                LeakageFinding(
                    check="temporal_index",
                    status="FAIL",
                    message=(
                        "Development and holdout datasets must use "
                        "DatetimeIndex."
                    ),
                )
            )

            return LeakageAuditReport(
                passed=False,
                findings=findings,
            )

        if development_data.empty:
            findings.append(
                LeakageFinding(
                    check="development_data",
                    status="FAIL",
                    message=(
                        "Development dataset is empty."
                    ),
                )
            )

        if holdout_data.empty:
            findings.append(
                LeakageFinding(
                    check="holdout_data",
                    status="FAIL",
                    message=(
                        "Holdout dataset is empty."
                    ),
                )
            )

        if not findings:

            development_end = (
                development_data.index.max()
            )

            holdout_start = (
                holdout_data.index.min()
            )

            if (
                development_end
                >= holdout_start
            ):
                findings.append(
                    LeakageFinding(
                        check="temporal_separation",
                        status="FAIL",
                        message=(
                            "Development data overlaps or extends into "
                            "the holdout period."
                        ),
                    )
                )
            else:
                findings.append(
                    LeakageFinding(
                        check="temporal_separation",
                        status="PASS",
                        message=(
                            "Development data ends before the holdout begins."
                        ),
                        severity="INFO",
                    )
                )

            if embargo > 0:

                ordered_holdout = (
                    holdout_data.sort_index()
                )

                if len(
                    ordered_holdout
                ) > 0:

                    embargo_position = min(
                        embargo,
                        len(
                            ordered_holdout
                        ),
                    )

                    embargo_start = (
                        ordered_holdout.index[
                            embargo_position - 1
                        ]
                    )

                    if (
                        development_end
                        >= embargo_start
                    ):
                        findings.append(
                            LeakageFinding(
                                check="embargo_separation",
                                status="WARNING",
                                message=(
                                    "Development data is close to the "
                                    "requested embargo boundary."
                                ),
                                severity="WARNING",
                            )
                        )
                    else:
                        findings.append(
                            LeakageFinding(
                                check="embargo_separation",
                                status="PASS",
                                message=(
                                    "Development data respects the "
                                    "requested temporal embargo."
                                ),
                                severity="INFO",
                            )
                        )

        failures = [
            finding
            for finding in findings
            if finding.status == "FAIL"
            and finding.severity == "ERROR"
        ]

        return LeakageAuditReport(
            passed=len(
                failures
            )
            == 0,
            findings=findings,
            checked_rows=(
                len(
                    development_data
                )
                + len(
                    holdout_data
                )
            ),
            development_end=(
                development_data.index.max()
                if not development_data.empty
                else None
            ),
            holdout_start=(
                holdout_data.index.min()
                if not holdout_data.empty
                else None
            ),
        )

    # ------------------------------------------------------------------
    # Feature mutation test
    # ------------------------------------------------------------------

    @staticmethod
    def feature_mutation_audit(
        original: pd.DataFrame,
        mutated: pd.DataFrame,
        feature_columns: Sequence[str],
        *,
        cutoff: pd.Timestamp,
    ) -> LeakageAuditReport:
        """
        Detect whether changing future observations changes historical
        feature values.

        This is a powerful adversarial leakage test.

        Example:

            Original:
                data through 2025

            Mutated:
                identical historical data
                radically altered 2025+ values

        Features before the cutoff should remain unchanged if the
        feature engineering process is causal.
        """

        findings: list[
            LeakageFinding
        ] = []

        if not isinstance(
            original.index,
            pd.DatetimeIndex,
        ):
            findings.append(
                LeakageFinding(
                    check="original_index",
                    status="FAIL",
                    message=(
                        "Original dataset must use DatetimeIndex."
                    ),
                )
            )

        if not isinstance(
            mutated.index,
            pd.DatetimeIndex,
        ):
            findings.append(
                LeakageFinding(
                    check="mutated_index",
                    status="FAIL",
                    message=(
                        "Mutated dataset must use DatetimeIndex."
                    ),
                )
            )

        if findings:
            return LeakageAuditReport(
                passed=False,
                findings=findings,
            )

        common_index = (
            original.index
            .intersection(
                mutated.index
            )
        )

        historical_index = common_index[
            common_index
            <= pd.Timestamp(
                cutoff
            )
        ]

        changed_features = []

        for column in feature_columns:

            if (
                column not in original.columns
                or column not in mutated.columns
            ):
                findings.append(
                    LeakageFinding(
                        check="feature_presence",
                        status="FAIL",
                        message=(
                            f"Feature '{column}' is missing from one dataset."
                        ),
                    )
                )
                continue

            left = original.loc[
                historical_index,
                column,
            ]

            right = mutated.loc[
                historical_index,
                column,
            ]

            equal = np.isclose(
                left.to_numpy(
                    dtype=float
                ),
                right.to_numpy(
                    dtype=float
                ),
                equal_nan=True,
            )

            if not np.all(
                equal
            ):
                changed_features.append(
                    column
                )

        if changed_features:
            findings.append(
                LeakageFinding(
                    check="future_mutation",
                    status="FAIL",
                    message=(
                        "Historical feature values changed after future "
                        "data was mutated: "
                        f"{sorted(changed_features)}"
                    ),
                )
            )
        else:
            findings.append(
                LeakageFinding(
                    check="future_mutation",
                    status="PASS",
                    message=(
                        "Historical feature values remained unchanged "
                        "after future-data mutation."
                    ),
                    severity="INFO",
                )
            )

        failures = [
            finding
            for finding in findings
            if finding.status == "FAIL"
            and finding.severity == "ERROR"
        ]

        return LeakageAuditReport(
            passed=len(
                failures
            )
            == 0,
            findings=findings,
            checked_rows=len(
                historical_index
            ),
            checked_features=len(
                feature_columns
            ),
        )


# ----------------------------------------------------------------------
# Convenience functions
# ----------------------------------------------------------------------


def audit_research_dataset(
    dataframe: pd.DataFrame,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    *,
    config: LeakageAuditConfig | None = None,
) -> LeakageAuditReport:
    """
    Convenience wrapper for static leakage auditing.
    """

    auditor = LeakageAuditor(
        config
    )

    return auditor.audit(
        dataframe,
        feature_columns,
        target_columns,
    )


def assert_leakage_free(
    dataframe: pd.DataFrame,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    *,
    config: LeakageAuditConfig | None = None,
) -> LeakageAuditReport:
    """
    Run the audit and raise ValueError if leakage checks fail.
    """

    report = audit_research_dataset(
        dataframe,
        feature_columns,
        target_columns,
        config=config,
    )

    if not report.passed:
        messages = [
            finding.message
            for finding in report.errors
        ]

        raise ValueError(
            "Research dataset failed leakage audit: "
            + " | ".join(
                messages
            )
        )

    return report


def compare_future_mutation(
    original: pd.DataFrame,
    mutated: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    cutoff: pd.Timestamp,
) -> LeakageAuditReport:
    """
    Convenience wrapper for adversarial future-mutation testing.
    """

    return LeakageAuditor.feature_mutation_audit(
        original,
        mutated,
        feature_columns,
        cutoff=cutoff,
    )


__all__ = [
    "LeakageFinding",
    "LeakageAuditConfig",
    "LeakageAuditReport",
    "LeakageAuditor",
    "audit_research_dataset",
    "assert_leakage_free",
    "compare_future_mutation",
]
