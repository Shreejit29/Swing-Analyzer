"""
End-to-end probability calibration pipeline.

Purpose
-------
Convert raw classifier probabilities into probabilities that better
represent observed frequencies.

Example:

    Raw probability:        0.82
    Calibrated probability: 0.71

The calibrated value should only be interpreted as a probability after
calibration quality has been demonstrated.

Temporal safety
---------------
Calibration data must occur after the model-training period and before
the final untouched holdout.

The final holdout must never be used to fit calibration parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .calibration import (
    CalibrationConfig,
    ProbabilityCalibrator,
)
from .metrics import classification_metrics
from .preprocessing import (
    PreprocessorConfig,
    SafePreprocessor,
)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass
class CalibrationPipelineConfig:
    """
    Configuration for temporal probability calibration.
    """

    method: str = "sigmoid"

    minimum_samples: int = 50

    minimum_calibration_accuracy: float = 0.0

    maximum_brier_score: float = 0.25

    maximum_ece: float = 0.15

    minimum_high_confidence_precision: float = 0.0

    high_confidence_threshold: float = 0.70

    n_bins: int = 10

    random_state: int = 42

    def __post_init__(self) -> None:
        if self.minimum_samples < 1:
            raise ValueError(
                "minimum_samples must be positive."
            )

        if not 0.0 <= self.minimum_calibration_accuracy <= 1.0:
            raise ValueError(
                "minimum_calibration_accuracy must be between 0 and 1."
            )

        if not 0.0 <= self.maximum_brier_score <= 1.0:
            raise ValueError(
                "maximum_brier_score must be between 0 and 1."
            )

        if not 0.0 <= self.maximum_ece <= 1.0:
            raise ValueError(
                "maximum_ece must be between 0 and 1."
            )

        if not 0.0 <= self.minimum_high_confidence_precision <= 1.0:
            raise ValueError(
                "minimum_high_confidence_precision must be between 0 and 1."
            )

        if not 0.0 < self.high_confidence_threshold < 1.0:
            raise ValueError(
                "high_confidence_threshold must be between 0 and 1."
            )

        if self.n_bins < 2:
            raise ValueError(
                "n_bins must be at least 2."
            )


# ----------------------------------------------------------------------
# Result containers
# ----------------------------------------------------------------------


@dataclass
class CalibrationMetrics:
    """
    Calibration diagnostics.
    """

    brier_score: float

    log_loss: float

    ece: float

    high_confidence_precision: float

    high_confidence_coverage: float

    sample_count: int

    passed: bool

    notes: list[str] = field(
        default_factory=list
    )


@dataclass
class CalibrationPipelineResult:
    """
    Complete calibration pipeline result.
    """

    calibrator: ProbabilityCalibrator

    raw_probabilities: np.ndarray

    calibrated_probabilities: np.ndarray

    y_true: np.ndarray

    raw_metrics: CalibrationMetrics

    calibrated_metrics: CalibrationMetrics

    reliability_table: pd.DataFrame

    target_column: str

    feature_names: Sequence[str]

    training_end: Optional[pd.Timestamp]

    calibration_start: Optional[pd.Timestamp]

    calibration_end: Optional[pd.Timestamp]

    notes: list[str] = field(
        default_factory=list
    )

    @property
    def passed(self) -> bool:
        return self.calibrated_metrics.passed

    def transform_probability(
        self,
        probabilities: Sequence[float],
    ) -> np.ndarray:
        """
        Apply the fitted calibrator to new raw probabilities.
        """

        values = np.asarray(
            probabilities,
            dtype=float,
        )

        return self.calibrator.transform(
            values
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "target_column": self.target_column,
            "sample_count": int(
                len(self.y_true)
            ),
            "method": self.calibrator.config.method,
            "raw_brier_score": (
                self.raw_metrics.brier_score
            ),
            "calibrated_brier_score": (
                self.calibrated_metrics.brier_score
            ),
            "raw_ece": (
                self.raw_metrics.ece
            ),
            "calibrated_ece": (
                self.calibrated_metrics.ece
            ),
            "high_confidence_precision": (
                self.calibrated_metrics
                .high_confidence_precision
            ),
            "passed": self.passed,
        }


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------


class CalibrationPipeline:
    """
    Temporal calibration pipeline.

    Expected workflow:

        training data
              ↓
        fit model
              ↓
        generate predictions on later calibration data
              ↓
        fit calibrator
              ↓
        evaluate calibration
              ↓
        freeze calibrator
              ↓
        final untouched holdout
    """

    def __init__(
        self,
        config: Optional[
            CalibrationPipelineConfig
        ] = None,
    ) -> None:
        self.config = (
            config
            or CalibrationPipelineConfig()
        )

        self.calibrator = (
            ProbabilityCalibrator(
                CalibrationConfig(
                    method=self.config.method,
                    n_bins=self.config.n_bins,
                    confidence_threshold=(
                        self.config
                        .high_confidence_threshold
                    ),
                )
            )
        )

        self._fitted = False

    # ------------------------------------------------------------------
    # Fit from predictions
    # ------------------------------------------------------------------

    def fit(
        self,
        raw_probabilities: Sequence[float],
        y_true: Sequence[int],
        *,
        timestamps: Optional[
            Sequence[pd.Timestamp]
        ] = None,
        training_end: Optional[
            pd.Timestamp
        ] = None,
        target_column: str = "Direction",
        feature_names: Optional[
            Sequence[str]
        ] = None,
    ) -> CalibrationPipelineResult:
        """
        Fit the calibrator on a temporally later calibration set.

        ``raw_probabilities`` must come from a model that was trained
        without using this calibration data.
        """

        probabilities = self._validate_probabilities(
            raw_probabilities
        )

        target = self._validate_target(
            y_true
        )

        if len(probabilities) != len(target):
            raise ValueError(
                "Probability and target lengths do not match."
            )

        if len(target) < (
            self.config.minimum_samples
        ):
            raise ValueError(
                "Insufficient calibration samples. "
                f"Required at least "
                f"{self.config.minimum_samples}, "
                f"received {len(target)}."
            )

        calibration_timestamps = (
            self._validate_timestamps(
                timestamps,
                len(target),
                training_end,
            )
        )

        calibration_start = None
        calibration_end = None

        if calibration_timestamps is not None:
            calibration_start = (
                calibration_timestamps.min()
            )

            calibration_end = (
                calibration_timestamps.max()
            )

        # --------------------------------------------------------------
        # Raw metrics
        # --------------------------------------------------------------

        raw_metrics = self._evaluate(
            target,
            probabilities,
        )

        # --------------------------------------------------------------
        # Fit calibrator
        # --------------------------------------------------------------

        self.calibrator.fit(
            probabilities,
            target,
        )

        self._fitted = True

        calibrated = (
            self.calibrator.transform(
                probabilities
            )
        )

        # --------------------------------------------------------------
        # Calibrated metrics
        # --------------------------------------------------------------

        calibrated_metrics = self._evaluate(
            target,
            calibrated,
        )

        reliability = (
            self._reliability_table(
                target,
                calibrated,
            )
        )

        notes = [
            (
                "Calibration was fitted only on the supplied "
                "calibration observations."
            ),
            (
                "The final untouched holdout must remain excluded "
                "from calibration fitting."
            ),
        ]

        if calibrated_metrics.brier_score < (
            raw_metrics.brier_score
        ):
            notes.append(
                "Calibration improved Brier score."
            )
        else:
            notes.append(
                "Calibration did not improve Brier score; "
                "investigate before production approval."
            )

        if calibrated_metrics.ece < (
            raw_metrics.ece
        ):
            notes.append(
                "Calibration improved expected calibration error."
            )
        else:
            notes.append(
                "Calibration did not improve ECE."
            )

        return CalibrationPipelineResult(
            calibrator=self.calibrator,
            raw_probabilities=probabilities,
            calibrated_probabilities=calibrated,
            y_true=target,
            raw_metrics=raw_metrics,
            calibrated_metrics=calibrated_metrics,
            reliability_table=reliability,
            target_column=target_column,
            feature_names=(
                list(feature_names)
                if feature_names is not None
                else []
            ),
            training_end=training_end,
            calibration_start=calibration_start,
            calibration_end=calibration_end,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Fit from model
    # ------------------------------------------------------------------

    def fit_from_model(
        self,
        model: Any,
        X_calibration: pd.DataFrame,
        y_calibration: Sequence[int],
        *,
        preprocessor: Optional[
            SafePreprocessor
        ] = None,
        timestamps: Optional[
            Sequence[pd.Timestamp]
        ] = None,
        training_end: Optional[
            pd.Timestamp
        ] = None,
        target_column: str = "Direction",
    ) -> CalibrationPipelineResult:
        """
        Generate raw probabilities from a fitted model and calibrate them.

        The supplied model must already be trained before the calibration
        period begins.
        """

        if not hasattr(
            model,
            "predict_proba",
        ):
            raise TypeError(
                "Model must expose predict_proba() "
                "for probability calibration."
            )

        if preprocessor is not None:
            X = preprocessor.transform(
                X_calibration
            )
        else:
            X = X_calibration

        raw_probability = (
            self._extract_positive_probability(
                model.predict_proba(X)
            )
        )

        return self.fit(
            raw_probability,
            y_calibration,
            timestamps=timestamps,
            training_end=training_end,
            target_column=target_column,
            feature_names=list(
                X_calibration.columns
            ),
        )

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(
        self,
        raw_probabilities: Sequence[float],
    ) -> np.ndarray:
        """
        Calibrate new model probabilities.
        """

        if not self._fitted:
            raise RuntimeError(
                "CalibrationPipeline has not been fitted."
            )

        values = self._validate_probabilities(
            raw_probabilities
        )

        return self.calibrator.transform(
            values
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
    ) -> CalibrationMetrics:
        probabilities = np.clip(
            probabilities,
            1e-7,
            1.0 - 1e-7,
        )

        predictions = (
            probabilities >= 0.50
        ).astype(int)

        metrics = classification_metrics(
            y_true,
            predictions,
            probabilities,
        )

        values = self._metrics_to_dict(
            metrics
        )

        brier = float(
            values.get(
                "brier_score",
                np.mean(
                    (
                        probabilities
                        - y_true
                    )
                    ** 2
                ),
            )
        )

        log_loss = float(
            values.get(
                "log_loss",
                self._binary_log_loss(
                    y_true,
                    probabilities,
                ),
            )
        )

        ece = self._expected_calibration_error(
            y_true,
            probabilities,
        )

        high_conf_precision, high_conf_coverage = (
            self._high_confidence_metrics(
                y_true,
                probabilities,
            )
        )

        passed = (
            brier
            <= self.config.maximum_brier_score
            and ece
            <= self.config.maximum_ece
            and high_conf_precision
            >= self.config.minimum_high_confidence_precision
        )

        notes = []

        if brier <= self.config.maximum_brier_score:
            notes.append(
                "Brier score passed."
            )
        else:
            notes.append(
                "Brier score failed."
            )

        if ece <= self.config.maximum_ece:
            notes.append(
                "ECE passed."
            )
        else:
            notes.append(
                "ECE failed."
            )

        return CalibrationMetrics(
            brier_score=brier,
            log_loss=log_loss,
            ece=ece,
            high_confidence_precision=(
                high_conf_precision
            ),
            high_confidence_coverage=(
                high_conf_coverage
            ),
            sample_count=len(y_true),
            passed=passed,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Reliability
    # ------------------------------------------------------------------

    def _reliability_table(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
    ) -> pd.DataFrame:
        bins = np.linspace(
            0.0,
            1.0,
            self.config.n_bins + 1,
        )

        rows = []

        for i in range(
            self.config.n_bins
        ):
            lower = bins[i]
            upper = bins[i + 1]

            if i == self.config.n_bins - 1:
                mask = (
                    (probabilities >= lower)
                    & (probabilities <= upper)
                )
            else:
                mask = (
                    (probabilities >= lower)
                    & (probabilities < upper)
                )

            count = int(
                mask.sum()
            )

            if count == 0:
                continue

            rows.append(
                {
                    "bin": i + 1,
                    "lower": lower,
                    "upper": upper,
                    "sample_count": count,
                    "mean_predicted_probability": float(
                        probabilities[mask].mean()
                    ),
                    "observed_positive_rate": float(
                        y_true[mask].mean()
                    ),
                    "absolute_gap": float(
                        abs(
                            probabilities[mask].mean()
                            - y_true[mask].mean()
                        )
                    ),
                }
            )

        return pd.DataFrame(rows)

    def _expected_calibration_error(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
    ) -> float:
        table = self._reliability_table(
            y_true,
            probabilities,
        )

        if table.empty:
            return 1.0

        total = len(y_true)

        return float(
            (
                table["sample_count"]
                * table["absolute_gap"]
            ).sum()
            / total
        )

    def _high_confidence_metrics(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
    ) -> tuple[float, float]:
        threshold = (
            self.config
            .high_confidence_threshold
        )

        mask = (
            probabilities >= threshold
        )

        if not mask.any():
            return 0.0, 0.0

        precision = float(
            y_true[mask].mean()
        )

        coverage = float(
            mask.mean()
        )

        return precision, coverage

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_probabilities(
        probabilities: Sequence[float],
    ) -> np.ndarray:
        values = np.asarray(
            probabilities,
            dtype=float,
        ).reshape(-1)

        if values.size == 0:
            raise ValueError(
                "No probabilities supplied."
            )

        if not np.isfinite(values).all():
            raise ValueError(
                "Probabilities contain NaN or infinite values."
            )

        if (
            (values < 0.0).any()
            or (values > 1.0).any()
        ):
            raise ValueError(
                "Probabilities must be between 0 and 1."
            )

        return values

    @staticmethod
    def _validate_target(
        target: Sequence[int],
    ) -> np.ndarray:
        values = np.asarray(
            target
        ).reshape(-1)

        if values.size == 0:
            raise ValueError(
                "No calibration targets supplied."
            )

        if pd.isna(values).any():
            raise ValueError(
                "Calibration target contains missing values."
            )

        unique = set(
            np.unique(values).tolist()
        )

        if not unique.issubset(
            {0, 1}
        ):
            raise ValueError(
                "Calibration target must contain only 0 and 1."
            )

        if len(unique) < 2:
            raise ValueError(
                "Calibration target must contain both classes."
            )

        return values.astype(int)

    @staticmethod
    def _validate_timestamps(
        timestamps: Optional[
            Sequence[pd.Timestamp]
        ],
        expected_length: int,
        training_end: Optional[
            pd.Timestamp
        ],
    ) -> Optional[pd.DatetimeIndex]:
        if timestamps is None:
            return None

        index = pd.DatetimeIndex(
            timestamps
        )

        if len(index) != expected_length:
            raise ValueError(
                "Timestamp length does not match calibration data."
            )

        if index.has_duplicates:
            raise ValueError(
                "Calibration timestamps contain duplicates."
            )

        if not index.is_monotonic_increasing:
            raise ValueError(
                "Calibration timestamps must be chronological."
            )

        if training_end is not None:
            training_end = pd.Timestamp(
                training_end
            )

            if index.min() <= training_end:
                raise ValueError(
                    "Calibration data overlaps or precedes "
                    "the model training period."
                )

        return index

    @staticmethod
    def _extract_positive_probability(
        probability_matrix: Any,
    ) -> np.ndarray:
        matrix = np.asarray(
            probability_matrix,
            dtype=float,
        )

        if matrix.ndim != 2:
            raise ValueError(
                "predict_proba() must return a 2D array."
            )

        if matrix.shape[1] < 2:
            raise ValueError(
                "Binary classifier must provide probabilities "
                "for both classes."
            )

        return matrix[:, 1]

    @staticmethod
    def _binary_log_loss(
        y_true: np.ndarray,
        probability: np.ndarray,
    ) -> float:
        p = np.clip(
            probability,
            1e-7,
            1.0 - 1e-7,
        )

        return float(
            -np.mean(
                (
                    y_true * np.log(p)
                )
                + (
                    (1 - y_true)
                    * np.log(1 - p)
                )
            )
        )

    @staticmethod
    def _metrics_to_dict(
        metrics: Any,
    ) -> Dict[str, float]:
        if isinstance(
            metrics,
            dict,
        ):
            source = metrics

        elif hasattr(
            metrics,
            "__dataclass_fields__",
        ):
            source = {
                name: getattr(
                    metrics,
                    name,
                )
                for name in metrics.__dataclass_fields__
            }

        else:
            source = vars(
                metrics
            )

        result = {}

        for key, value in source.items():
            if isinstance(
                value,
                (
                    int,
                    float,
                    np.integer,
                    np.floating,
                ),
            ):
                if np.isfinite(value):
                    result[key] = float(
                        value
                    )

        return result


# ----------------------------------------------------------------------
# Convenience functions
# ----------------------------------------------------------------------


def calibrate_predictions(
    raw_probabilities: Sequence[float],
    y_true: Sequence[int],
    *,
    config: Optional[
        CalibrationPipelineConfig
    ] = None,
    timestamps: Optional[
        Sequence[pd.Timestamp]
    ] = None,
    training_end: Optional[
        pd.Timestamp
    ] = None,
    target_column: str = "Direction",
) -> CalibrationPipelineResult:
    """
    Convenience function for calibrating raw predictions.
    """

    pipeline = CalibrationPipeline(
        config=config
    )

    return pipeline.fit(
        raw_probabilities,
        y_true,
        timestamps=timestamps,
        training_end=training_end,
        target_column=target_column,
    )


def calibration_summary(
    result: CalibrationPipelineResult,
) -> Dict[str, Any]:
    """Return a compact calibration summary."""

    return result.summary()


__all__ = [
    "CalibrationPipelineConfig",
    "CalibrationMetrics",
    "CalibrationPipelineResult",
    "CalibrationPipeline",
    "calibrate_predictions",
    "calibration_summary",
]
