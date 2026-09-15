"""
Safe production inference engine for AI Swing Analyser.

Purpose
-------
Load an approved model artifact and generate predictions on new data.

Production inference must NOT:
    - retrain models
    - fit preprocessing
    - change feature order
    - select new features
    - tune hyperparameters
    - use future information

Required lifecycle:

    Research
       ↓
    Validation
       ↓
    Approval
       ↓
    Artifact
       ↓
    Inference
       ↓
    Prediction

Only APPROVED artifacts are permitted for production inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .artifacts import (
    ModelArtifact,
    load_artifact,
)
from .model_registry import (
    ModelRegistry,
)
from .predictor import (
    DirectionPrediction,
    PredictionConfig,
    PredictionEngine,
    UnifiedPrediction,
)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass
class InferenceConfig:
    """
    Production inference configuration.
    """

    require_approved_artifact: bool = True

    reject_missing_features: bool = True

    reject_extra_features: bool = False

    require_finite_features: bool = True

    maximum_missing_fraction: float = 0.40

    allow_nan_after_preprocessing: bool = False

    random_state: int = 42

    def __post_init__(self) -> None:
        if not (
            0.0
            <= self.maximum_missing_fraction
            < 1.0
        ):
            raise ValueError(
                "maximum_missing_fraction must be between 0 and 1."
            )


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------


@dataclass
class InferenceValidation:
    """
    Validation report for inference input.
    """

    passed: bool

    missing_features: list[str]

    extra_features: list[str]

    non_numeric_features: list[str]

    high_missing_features: list[str]

    non_finite_features: list[str]

    row_count: int

    notes: list[str] = field(
        default_factory=list
    )


# ----------------------------------------------------------------------
# Prediction result
# ----------------------------------------------------------------------


@dataclass
class InferenceResult:
    """
    Production inference output.
    """

    model_id: str

    symbol: Optional[str]

    timeframe: Optional[str]

    horizon: int

    timestamp: pd.Timestamp

    prediction: UnifiedPrediction

    validation: InferenceValidation

    approved: bool

    notes: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "model_id": self.model_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "horizon": self.horizon,
            "timestamp": self.timestamp.isoformat(),
            "approved": self.approved,
            "validation_passed": (
                self.validation.passed
            ),
            "notes": self.notes,
        }

        prediction = self.prediction

        if hasattr(
            prediction,
            "__dataclass_fields__",
        ):
            result["prediction"] = {
                name: self._serialize(
                    getattr(
                        prediction,
                        name,
                    )
                )
                for name in (
                    prediction.__dataclass_fields__
                )
            }
        else:
            result["prediction"] = str(
                prediction
            )

        return result

    @staticmethod
    def _serialize(
        value: Any,
    ) -> Any:
        if isinstance(
            value,
            np.ndarray,
        ):
            return value.tolist()

        if isinstance(
            value,
            (
                np.integer,
                np.floating,
            ),
        ):
            return value.item()

        if isinstance(
            value,
            pd.Timestamp,
        ):
            return value.isoformat()

        if isinstance(
            value,
            dict,
        ):
            return {
                key: InferenceResult._serialize(
                    item
                )
                for key, item in value.items()
            }

        if isinstance(
            value,
            list,
        ):
            return [
                InferenceResult._serialize(
                    item
                )
                for item in value
            ]

        return value


# ----------------------------------------------------------------------
# Safe inference engine
# ----------------------------------------------------------------------


class SafeInferenceEngine:
    """
    Strict inference engine.

    The artifact contains the fitted model and preprocessing pipeline.

    No fitting is performed by this class.
    """

    def __init__(
        self,
        artifact: ModelArtifact,
        *,
        model_id: Optional[str] = None,
        config: Optional[
            InferenceConfig
        ] = None,
        prediction_config: Optional[
            PredictionConfig
        ] = None,
    ) -> None:
        self.artifact = artifact

        self.config = (
            config
            or InferenceConfig()
        )

        self.model_id = (
            model_id
            or f"{artifact.metadata.experiment_id}:{artifact.metadata.model_name}"
        )

        self.prediction_engine = (
            PredictionEngine(
                config=(
                    prediction_config
                    or PredictionConfig()
                )
            )
        )

        self._register_artifact_model()

    # ------------------------------------------------------------------
    # Artifact loading
    # ------------------------------------------------------------------

    @classmethod
    def from_artifact(
        cls,
        artifact_path: str,
        *,
        config: Optional[
            InferenceConfig
        ] = None,
        prediction_config: Optional[
            PredictionConfig
        ] = None,
    ) -> "SafeInferenceEngine":
        """
        Load a model artifact.

        The artifact itself must contain an approved production status
        when strict approval is enabled.
        """

        artifact = load_artifact(
            artifact_path
        )

        if (
            config is not None
            and config.require_approved_artifact
        ):
            cls._assert_artifact_approved(
                artifact
            )

        return cls(
            artifact,
            config=config,
            prediction_config=prediction_config,
        )

    @classmethod
    def from_registry(
        cls,
        registry: ModelRegistry,
        model_id: str,
        *,
        config: Optional[
            InferenceConfig
        ] = None,
        prediction_config: Optional[
            PredictionConfig
        ] = None,
    ) -> "SafeInferenceEngine":
        """
        Load an approved model through the model registry.
        """

        entry = registry.assert_approved(
            model_id
        )

        artifact_path = getattr(
            entry,
            "artifact_path",
            None,
        )

        if not artifact_path:
            raise ValueError(
                "Approved registry entry has no artifact path."
            )

        return cls.from_artifact(
            artifact_path,
            config=config,
            prediction_config=prediction_config,
        )

    # ------------------------------------------------------------------
    # Main prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        data: pd.DataFrame,
        *,
        timestamp: Optional[
            pd.Timestamp
        ] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> InferenceResult:
        """
        Generate a production prediction.

        The input must already contain all required engineered features.

        The latest row is used unless a specific timestamp is supplied.
        """

        validation = (
            self.validate_input(
                data
            )
        )

        if not validation.passed:
            raise ValueError(
                self._format_validation_error(
                    validation
                )
            )

        frame = data.copy()

        if timestamp is None:
            row = frame.iloc[
                [-1]
            ]

            prediction_timestamp = (
                frame.index[-1]
            )

        else:
            prediction_timestamp = (
                pd.Timestamp(
                    timestamp
                )
            )

            if prediction_timestamp not in (
                frame.index
            ):
                raise ValueError(
                    "Requested timestamp is not present in input data."
                )

            row = frame.loc[
                [prediction_timestamp]
            ]

        features = self._feature_frame(
            row
        )

        transformed = (
            self._transform_features(
                features
            )
        )

        prediction = (
            self._predict_transformed(
                transformed
            )
        )

        notes = [
            (
                "Prediction generated from an approved artifact."
            ),
            (
                "No model fitting or feature fitting was performed "
                "during inference."
            ),
        ]

        return InferenceResult(
            model_id=self.model_id,
            symbol=symbol,
            timeframe=timeframe,
            horizon=int(
                self.artifact.metadata.horizon
            ),
            timestamp=prediction_timestamp,
            prediction=prediction,
            validation=validation,
            approved=True,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate predictions for every valid row.

        This is intended primarily for research diagnostics.

        For live inference, ``predict`` should normally be called only
        for the latest completed observation.
        """

        validation = self.validate_input(
            data
        )

        if not validation.passed:
            raise ValueError(
                self._format_validation_error(
                    validation
                )
            )

        features = self._feature_frame(
            data
        )

        transformed = (
            self._transform_features(
                features
            )
        )

        prediction = (
            self._predict_transformed(
                transformed
            )
        )

        return self._prediction_to_frame(
            prediction,
            data.index,
        )

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def validate_input(
        self,
        data: pd.DataFrame,
    ) -> InferenceValidation:
        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise TypeError(
                "Inference input must be a pandas DataFrame."
            )

        if not isinstance(
            data.index,
            pd.DatetimeIndex,
        ):
            raise TypeError(
                "Inference input must have a DatetimeIndex."
            )

        required_features = list(
            self.artifact.metadata
            .feature_names
        )

        columns = list(
            data.columns
        )

        missing = [
            feature
            for feature in required_features
            if feature not in columns
        ]

        extra = [
            column
            for column in columns
            if column not in required_features
        ]

        non_numeric = [
            feature
            for feature in required_features
            if feature in data.columns
            and not pd.api.types.is_numeric_dtype(
                data[feature]
            )
        ]

        high_missing = []

        non_finite = []

        for feature in required_features:
            if feature not in data.columns:
                continue

            series = data[
                feature
            ]

            missing_fraction = float(
                series.isna().mean()
            )

            if (
                missing_fraction
                > self.config
                .maximum_missing_fraction
            ):
                high_missing.append(
                    feature
                )

            if pd.api.types.is_numeric_dtype(
                series
            ):
                numeric = pd.to_numeric(
                    series,
                    errors="coerce",
                )

                if not np.isfinite(
                    numeric.fillna(0.0)
                ).all():
                    non_finite.append(
                        feature
                    )

        passed = True

        if missing and (
            self.config
            .reject_missing_features
        ):
            passed = False

        if non_numeric:
            passed = False

        if high_missing:
            passed = False

        if (
            non_finite
            and self.config
            .require_finite_features
        ):
            passed = False

        if (
            extra
            and self.config
            .reject_extra_features
        ):
            passed = False

        notes = []

        if extra:
            notes.append(
                "Extra input columns were ignored."
            )

        if not extra:
            notes.append(
                "Input contains no extra columns."
            )

        return InferenceValidation(
            passed=passed,
            missing_features=missing,
            extra_features=extra,
            non_numeric_features=non_numeric,
            high_missing_features=high_missing,
            non_finite_features=non_finite,
            row_count=len(data),
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Artifact registration
    # ------------------------------------------------------------------

    def _register_artifact_model(
        self,
    ) -> None:
        """
        Register the artifact's model with the prediction engine.

        The engine receives already-transformed data.
        """

        direction_model = getattr(
            self.artifact,
            "model",
            None,
        )

        if direction_model is None:
            raise ValueError(
                "Artifact does not contain a trained model."
            )

        self.prediction_engine.register_direction_model(
            "approved_direction_model",
            direction_model,
        )

    # ------------------------------------------------------------------
    # Artifact approval
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_artifact_approved(
        artifact: ModelArtifact,
    ) -> None:
        metadata = artifact.metadata

        status = str(
            getattr(
                metadata,
                "production_approved",
                False,
            )
        ).upper()

        approved = (
            status is True
        )

        if not approved:
            raise PermissionError(
                "Production inference is blocked because "
                "the model artifact is not APPROVED."
            )

    # ------------------------------------------------------------------
    # Feature transformation
    # ------------------------------------------------------------------

    def _feature_frame(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        features = list(
            self.artifact.metadata
            .feature_names
        )

        frame = data[
            features
        ].copy()

        # Exact feature order is critical.
        return frame[
            features
        ]

    def _transform_features(
        self,
        features: pd.DataFrame,
    ):
        preprocessor = getattr(
            self.artifact,
            "preprocessor",
            None,
        )

        if preprocessor is None:
            raise ValueError(
                "Artifact does not contain a fitted preprocessor."
            )

        if not hasattr(
            preprocessor,
            "transform",
        ):
            raise TypeError(
                "Artifact preprocessor does not support transform()."
            )

        transformed = (
            preprocessor.transform(
                features
            )
        )

        if not self.config.allow_nan_after_preprocessing:
            array = np.asarray(
                transformed
            )

            if not np.isfinite(
                array
            ).all():
                raise ValueError(
                    "Preprocessed inference features contain "
                    "NaN or infinite values."
                )

        return transformed

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _predict_transformed(
        self,
        transformed: Any,
    ) -> UnifiedPrediction:
        model = getattr(
            self.artifact,
            "model",
            None,
        )

        if model is None:
            raise ValueError(
                "Artifact does not contain a trained model."
            )

        if not hasattr(
            model,
            "predict_proba",
        ):
            raise TypeError(
                "Approved production classifier must expose predict_proba()."
            )

        probabilities = np.asarray(
            model.predict_proba(
                transformed
            ),
            dtype=float,
        )

        if probabilities.ndim != 2:
            raise ValueError(
                "Model probabilities must be a 2D array."
            )

        if probabilities.shape[1] < 2:
            raise ValueError(
                "Binary classifier must provide two probabilities."
            )

        probability_up = (
            probabilities[:, 1]
        )

        probability_down = (
            probabilities[:, 0]
        )

        direction = np.where(
            probability_up >= 0.50,
            "UP",
            "DOWN",
        )

        direction_prediction = (
            DirectionPrediction(
                up_probability=(
                    probability_up
                ),
                down_probability=(
                    probability_down
                ),
                predicted_direction=direction,
                confidence=np.maximum(
                    probability_up,
                    probability_down,
                ),
                model_disagreement=np.zeros(
                    len(probability_up)
                ),
                model_agreement=np.ones(
                    len(probability_up)
                ),
            )
        )

        # Build a UnifiedPrediction-compatible result.
        return UnifiedPrediction(
            direction=direction_prediction,
            return_prediction=None,
            range_prediction=None,
        )

    # ------------------------------------------------------------------
    # Batch output
    # ------------------------------------------------------------------

    @staticmethod
    def _prediction_to_frame(
        prediction: UnifiedPrediction,
        index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        direction = prediction.direction

        frame = pd.DataFrame(
            index=index
        )

        frame[
            "Probability_Up"
        ] = np.asarray(
            direction.probabilities_up
        )

        frame[
            "Probability_Down"
        ] = np.asarray(
            direction.probabilities_down
        )

        frame[
            "Predicted_Direction"
        ] = np.asarray(
            direction.predicted_direction
        )

        frame[
            "Confidence"
        ] = np.asarray(
            direction.confidence
        )

        return frame

    # ------------------------------------------------------------------
    # Error formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_validation_error(
        validation: InferenceValidation,
    ) -> str:
        parts = [
            "Inference input validation failed."
        ]

        if validation.missing_features:
            parts.append(
                "Missing features: "
                + ", ".join(
                    validation.missing_features
                )
            )

        if validation.non_numeric_features:
            parts.append(
                "Non-numeric features: "
                + ", ".join(
                    validation.non_numeric_features
                )
            )

        if validation.high_missing_features:
            parts.append(
                "High-missing features: "
                + ", ".join(
                    validation.high_missing_features
                )
            )

        if validation.non_finite_features:
            parts.append(
                "Non-finite features: "
                + ", ".join(
                    validation.non_finite_features
                )
            )

        if validation.extra_features:
            parts.append(
                "Extra features ignored: "
                + ", ".join(
                    validation.extra_features
                )
            )

        return " ".join(parts)


# ----------------------------------------------------------------------
# Convenience functions
# ----------------------------------------------------------------------


def load_approved_inference_engine(
    artifact_path: str,
    *,
    config: Optional[
        InferenceConfig
    ] = None,
) -> SafeInferenceEngine:
    """
    Load an approved artifact for production inference.
    """

    effective_config = (
        config
        or InferenceConfig(
            require_approved_artifact=True
        )
    )

    return SafeInferenceEngine.from_artifact(
        artifact_path,
        config=effective_config,
    )


def predict_with_approved_model(
    artifact_path: str,
    data: pd.DataFrame,
    *,
    timestamp: Optional[
        pd.Timestamp
    ] = None,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> InferenceResult:
    """
    Convenience function for one production prediction.
    """

    engine = load_approved_inference_engine(
        artifact_path
    )

    return engine.predict(
        data,
        timestamp=timestamp,
        symbol=symbol,
        timeframe=timeframe,
    )


__all__ = [
    "InferenceConfig",
    "InferenceValidation",
    "InferenceResult",
    "SafeInferenceEngine",
    "load_approved_inference_engine",
    "predict_with_approved_model",
]
