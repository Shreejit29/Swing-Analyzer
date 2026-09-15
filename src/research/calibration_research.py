"""
AI Swing Analyser — Probability Calibration Research.

Calibrates model probabilities using a temporally separated calibration
dataset.

The final holdout must remain untouched.

Research flow:

    Training
       ↓
    Model
       ↓
    Calibration period
       ↓
    Calibrated probability
       ↓
    Final holdout evaluation

Calibration is never fitted on the final holdout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.models.calibration import (
    CalibrationConfig,
    ProbabilityCalibrator,
)


# ---------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------


@dataclass
class CalibrationResearchResult:
    """
    Result of probability calibration research.
    """

    method: str

    raw_brier_score: float
    calibrated_brier_score: float

    raw_log_loss: float
    calibrated_log_loss: float

    raw_ece: float
    calibrated_ece: float

    calibration_samples: int

    calibration_start: pd.Timestamp
    calibration_end: pd.Timestamp

    high_confidence_threshold: float

    raw_high_confidence_precision: float | None
    calibrated_high_confidence_precision: float | None

    raw_high_confidence_coverage: float
    calibrated_high_confidence_coverage: float

    calibrator: ProbabilityCalibrator

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    warnings: list[str] = field(
        default_factory=list
    )

    @property
    def brier_improved(self) -> bool:
        return (
            self.calibrated_brier_score
            < self.raw_brier_score
        )

    @property
    def ece_improved(self) -> bool:
        return (
            self.calibrated_ece
            < self.raw_ece
        )

    @property
    def calibration_improved(self) -> bool:
        return (
            self.brier_improved
            or self.ece_improved
        )

    def summary(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "raw_brier_score": self.raw_brier_score,
            "calibrated_brier_score": (
                self.calibrated_brier_score
            ),
            "raw_log_loss": self.raw_log_loss,
            "calibrated_log_loss": (
                self.calibrated_log_loss
            ),
            "raw_ece": self.raw_ece,
            "calibrated_ece": self.calibrated_ece,
            "brier_improved": self.brier_improved,
            "ece_improved": self.ece_improved,
            "calibration_improved": (
                self.calibration_improved
            ),
            "calibration_samples": (
                self.calibration_samples
            ),
            "calibration_start": (
                self.calibration_start
            ),
            "calibration_end": (
                self.calibration_end
            ),
            "high_confidence_threshold": (
                self.high_confidence_threshold
            ),
            "raw_high_confidence_precision": (
                self.raw_high_confidence_precision
            ),
            "calibrated_high_confidence_precision": (
                self.calibrated_high_confidence_precision
            ),
            "raw_high_confidence_coverage": (
                self.raw_high_confidence_coverage
            ),
            "calibrated_high_confidence_coverage": (
                self.calibrated_high_confidence_coverage
            ),
            "metadata": dict(self.metadata),
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------
# Research calibrator
# ---------------------------------------------------------------------


class CalibrationResearchEngine:
    """
    Research-stage probability calibration engine.

    It uses only a dedicated calibration period and never accesses the
    final holdout.
    """

    def __init__(
        self,
        config: CalibrationConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else CalibrationConfig()
        )

        if not (
            0.0
            < self.config.high_confidence_threshold
            < 1.0
        ):
            raise ValueError(
                "high_confidence_threshold must be between 0 and 1."
            )

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        probabilities: np.ndarray,
        actuals: np.ndarray,
        timestamps: pd.Index,
    ) -> None:

        probabilities = np.asarray(
            probabilities,
            dtype=float,
        )

        actuals = np.asarray(
            actuals
        )

        if len(probabilities) != len(
            actuals
        ):
            raise ValueError(
                "Probabilities and actuals must have the same length."
            )

        if len(probabilities) != len(
            timestamps
        ):
            raise ValueError(
                "Timestamps and probabilities must have the same length."
            )

        if len(probabilities) == 0:
            raise ValueError(
                "Calibration dataset is empty."
            )

        if not np.isfinite(
            probabilities
        ).all():
            raise ValueError(
                "Calibration probabilities contain non-finite values."
            )

        if (
            (probabilities < 0).any()
            or (probabilities > 1).any()
        ):
            raise ValueError(
                "Calibration probabilities must be in [0, 1]."
            )

        if len(
            np.unique(actuals)
        ) < 2:
            raise ValueError(
                "Calibration targets must contain both classes."
            )

        if not isinstance(
            timestamps,
            pd.DatetimeIndex,
        ):
            raise TypeError(
                "Calibration timestamps must be a DatetimeIndex."
            )

        if timestamps.has_duplicates:
            raise ValueError(
                "Calibration timestamps contain duplicates."
            )

        if not timestamps.is_monotonic_increasing:
            raise ValueError(
                "Calibration timestamps must be chronological."
            )

    # -----------------------------------------------------------------
    # Calibration split
    # -----------------------------------------------------------------

    @staticmethod
    def create_calibration_split(
        development: pd.DataFrame,
        calibration_fraction: float = 0.20,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
    ]:

        if not isinstance(
            development,
            pd.DataFrame,
        ):
            raise TypeError(
                "development must be a DataFrame."
            )

        if development.empty:
            raise ValueError(
                "development must not be empty."
            )

        if not (
            0.0
            < calibration_fraction
            < 1.0
        ):
            raise ValueError(
                "calibration_fraction must be between 0 and 1."
            )

        if not isinstance(
            development.index,
            pd.DatetimeIndex,
        ):
            raise TypeError(
                "development must use DatetimeIndex."
            )

        if not development.index.is_monotonic_increasing:
            raise ValueError(
                "development must be chronological."
            )

        calibration_size = max(
            1,
            int(
                len(development)
                * calibration_fraction
            ),
        )

        train_end = (
            len(development)
            - calibration_size
        )

        if train_end <= 0:
            raise ValueError(
                "Training portion would be empty."
            )

        training = development.iloc[
            :train_end
        ].copy()

        calibration = development.iloc[
            train_end:
        ].copy()

        if (
            training.index.max()
            >= calibration.index.min()
        ):
            raise ValueError(
                "Training and calibration periods overlap."
            )

        return (
            training,
            calibration,
        )

    # -----------------------------------------------------------------
    # Metric helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _brier(
        probabilities: np.ndarray,
        actuals: np.ndarray,
    ) -> float:

        return float(
            np.mean(
                (
                    probabilities
                    - actuals
                )
                ** 2
            )
        )

    @staticmethod
    def _log_loss(
        probabilities: np.ndarray,
        actuals: np.ndarray,
    ) -> float:

        eps = 1e-15

        p = np.clip(
            probabilities,
            eps,
            1.0 - eps,
        )

        return float(
            -np.mean(
                actuals * np.log(p)
                + (
                    1 - actuals
                )
                * np.log(
                    1 - p
                )
            )
        )

    @staticmethod
    def _ece(
        probabilities: np.ndarray,
        actuals: np.ndarray,
        bins: int = 10,
    ) -> float:

        probabilities = np.asarray(
            probabilities,
            dtype=float,
        )

        actuals = np.asarray(
            actuals,
            dtype=float,
        )

        edges = np.linspace(
            0.0,
            1.0,
            bins + 1,
        )

        total = len(
            probabilities
        )

        ece = 0.0

        for lower, upper in zip(
            edges[:-1],
            edges[1:],
        ):

            mask = (
                (
                    probabilities
                    >= lower
                )
                & (
                    probabilities
                    < upper
                )
            )

            if upper == 1.0:
                mask |= (
                    probabilities
                    == upper
                )

            if not mask.any():
                continue

            confidence = float(
                probabilities[
                    mask
                ].mean()
            )

            accuracy = float(
                actuals[
                    mask
                ].mean()
            )

            ece += (
                mask.sum()
                / total
            ) * abs(
                confidence
                - accuracy
            )

        return float(ece)

    @staticmethod
    def _high_confidence_stats(
        probabilities: np.ndarray,
        actuals: np.ndarray,
        threshold: float,
    ) -> tuple[
        float | None,
        float,
    ]:

        mask = (
            probabilities
            >= threshold
        )

        coverage = float(
            mask.mean()
        )

        if not mask.any():
            return (
                None,
                coverage,
            )

        precision = float(
            actuals[
                mask
            ].mean()
        )

        return (
            precision,
            coverage,
        )

    # -----------------------------------------------------------------
    # Fit calibration
    # -----------------------------------------------------------------

    def fit(
        self,
        probabilities: np.ndarray,
        actuals: np.ndarray,
        timestamps: pd.DatetimeIndex,
        training_end: pd.Timestamp | None = None,
    ) -> CalibrationResearchResult:

        probabilities = np.asarray(
            probabilities,
            dtype=float,
        )

        actuals = np.asarray(
            actuals,
            dtype=int,
        )

        if not isinstance(timestamps, pd.DatetimeIndex):
            raise TypeError(
                "Calibration timestamps must be a DatetimeIndex."
            )

        self._validate_inputs(
            probabilities,
            actuals,
            timestamps,
        )

        if (
            training_end is not None
            and timestamps.min()
            <= pd.Timestamp(
                training_end
            )
        ):
            raise ValueError(
                "Calibration data must occur strictly after "
                "the training period."
            )

        minimum_samples = getattr(
            self.config,
            "min_samples",
            50,
        )

        if len(probabilities) < minimum_samples:
            raise ValueError(
                "Calibration dataset contains fewer than "
                f"{minimum_samples} observations."
            )

        # -------------------------------------------------------------
        # Raw metrics
        # -------------------------------------------------------------

        raw_brier = self._brier(
            probabilities,
            actuals,
        )

        raw_log_loss = self._log_loss(
            probabilities,
            actuals,
        )

        raw_ece = self._ece(
            probabilities,
            actuals,
        )

        threshold = (
            self.config.high_confidence_threshold
        )

        raw_precision, raw_coverage = (
            self._high_confidence_stats(
                probabilities,
                actuals,
                threshold,
            )
        )

        # -------------------------------------------------------------
        # Fit calibrator ONLY on calibration observations.
        # -------------------------------------------------------------

        calibrator = ProbabilityCalibrator(
            config=CalibrationConfig(
                method=self.config.method,
                clip_min=getattr(self.config, "clip_min", 0.001),
                clip_max=getattr(self.config, "clip_max", 0.999),
            )
        )

        calibrator.fit(
            probabilities,
            actuals,
        )

        calibrated = np.asarray(
            calibrator.transform(
                probabilities
            ),
            dtype=float,
        )

        calibrated_brier = self._brier(
            calibrated,
            actuals,
        )

        calibrated_log_loss = self._log_loss(
            calibrated,
            actuals,
        )

        calibrated_ece = self._ece(
            calibrated,
            actuals,
        )

        calibrated_precision, calibrated_coverage = (
            self._high_confidence_stats(
                calibrated,
                actuals,
                threshold,
            )
        )

        warnings: list[str] = []

        if calibrated_brier > raw_brier:
            warnings.append(
                "Calibration increased Brier score."
            )

        if calibrated_ece > raw_ece:
            warnings.append(
                "Calibration increased expected calibration error."
            )

        if calibrated_precision is None:
            warnings.append(
                "No calibrated observations reached the "
                "high-confidence threshold."
            )

        return CalibrationResearchResult(
            method=self.config.method,
            raw_brier_score=raw_brier,
            calibrated_brier_score=calibrated_brier,
            raw_log_loss=raw_log_loss,
            calibrated_log_loss=calibrated_log_loss,
            raw_ece=raw_ece,
            calibrated_ece=calibrated_ece,
            calibration_samples=len(
                probabilities
            ),
            calibration_start=timestamps.min(),
            calibration_end=timestamps.max(),
            high_confidence_threshold=threshold,
            raw_high_confidence_precision=raw_precision,
            calibrated_high_confidence_precision=(
                calibrated_precision
            ),
            raw_high_confidence_coverage=raw_coverage,
            calibrated_high_confidence_coverage=(
                calibrated_coverage
            ),
            calibrator=calibrator,
            metadata={
                "calibration_only": True,
                "final_holdout_used": False,
                "calibrator_fitted_on_holdout": False,
                "training_end": training_end,
                "method": self.config.method,
                "research_only": True,
            },
            warnings=warnings,
        )

    # -----------------------------------------------------------------
    # Transform
    # -----------------------------------------------------------------

    def transform(
        self,
        result: CalibrationResearchResult,
        probabilities: np.ndarray,
    ) -> np.ndarray:

        if result is None:
            raise ValueError(
                "Calibration result is required."
            )

        probabilities = np.asarray(
            probabilities,
            dtype=float,
        )

        if not np.isfinite(
            probabilities
        ).all():
            raise ValueError(
                "Probabilities contain non-finite values."
            )

        if (
            (probabilities < 0).any()
            or (probabilities > 1).any()
        ):
            raise ValueError(
                "Probabilities must be in [0, 1]."
            )

        return np.asarray(
            result.calibrator.transform(
                probabilities
            ),
            dtype=float,
        )


# ---------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------


def calibrate_research_probabilities(
    probabilities: np.ndarray,
    actuals: np.ndarray,
    timestamps: pd.DatetimeIndex,
    training_end: pd.Timestamp | None = None,
    config: CalibrationConfig | None = None,
) -> CalibrationResearchResult:

    engine = CalibrationResearchEngine(
        config=config
    )

    return engine.fit(
        probabilities=probabilities,
        actuals=actuals,
        timestamps=timestamps,
        training_end=training_end,
    )


__all__ = [
    "CalibrationResearchResult",
    "CalibrationResearchEngine",
    "calibrate_research_probabilities",
]
