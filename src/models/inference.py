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

This module is deliberately independent from model training.
It consumes an already fitted model and an already fitted preprocessor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import math

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
    RangePrediction,
    ReturnPrediction,
    UnifiedPrediction,
)


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass
class InferenceConfig:
    """
    Configuration for safe production inference.
    """

    require_approved_artifact: bool = True

    reject_missing_features: bool = True

    reject_extra_features: bool = False

    require_finite_features: bool = True

    maximum_missing_fraction: float = 0.40

    allow_nan_after_preprocessing: bool = False

    validate_probability_output: bool = True

    probability_sum_tolerance: float = 1e-6

    clip_probability_output: bool = False

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

        if self.probability_sum_tolerance <= 0:
            raise ValueError(
                "probability_sum_tolerance must be positive."
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
    Complete production inference output.
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

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """
        Convert inference result into a JSON-friendly dictionary.
        """

        result: Dict[str, Any] = {
            "model_id": self.model_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "horizon": self.horizon,
            "timestamp": self.timestamp.isoformat(),
            "approved": self.approved,
            "validation_passed": (
                self.validation.passed
            ),
            "notes": list(
                self.notes
            ),
        }

        result["validation"] = {
            "passed": self.validation.passed,
            "missing_features": list(
                self.validation.missing_features
            ),
            "extra_features": list(
                self.validation.extra_features
            ),
            "non_numeric_features": list(
                self.validation.non_numeric_features
            ),
            "high_missing_features": list(
                self.validation.high_missing_features
            ),
            "non_finite_features": list(
                self.validation.non_finite_features
            ),
            "row_count": self.validation.row_count,
            "notes": list(
                self.validation.notes
            ),
        }

        result["prediction"] = (
            self._serialize(
                self.prediction
            )
        )

        return result

    @staticmethod
    def _serialize(
        value: Any,
    ) -> Any:
        """
        Recursively serialize common NumPy/Pandas/dataclass values.
        """

        if hasattr(
            value,
            "__dataclass_fields__",
        ):
            return {
                name: InferenceResult._serialize(
                    getattr(
                        value,
                        name,
                    )
                )
                for name in value.__dataclass_fields__
            }

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
            (
                np.bool_,
            ),
        ):
            return bool(
                value
            )

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
            (
                list,
                tuple,
            ),
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
    Strict production inference engine.

    The model artifact contains:

        - fitted model
        - fitted preprocessing object
        - feature schema
        - training/validation metadata

    This class never calls ``fit`` on the model or preprocessor.
    """

    def __init__(
        self,
        artifact: ModelArtifact,
        *,
        model_id: Optional[str] = None,
        config: Optional[
            InferenceConfig
        ] = None,
    ) -> None:

        if artifact is None:
            raise ValueError(
                "A valid model artifact is required."
            )

        self.artifact = artifact

        self.config = (
            config
            or InferenceConfig()
        )

        self.model_id = (
            model_id
            or self._default_model_id()
        )

        self._validate_artifact()

    # ------------------------------------------------------------------
    # Artifact metadata
    # ------------------------------------------------------------------

    def _default_model_id(
        self,
    ) -> str:

        metadata = self.artifact.metadata

        experiment_id = str(
            getattr(
                metadata,
                "experiment_id",
                "unknown_experiment",
            )
        )

        model_name = str(
            getattr(
                metadata,
                "model_name",
                "unknown_model",
            )
        )

        return (
            f"{experiment_id}:{model_name}"
        )

    def _validate_artifact(
        self,
    ) -> None:

        metadata = getattr(
            self.artifact,
            "metadata",
            None,
        )

        if metadata is None:
            raise ValueError(
                "Artifact does not contain metadata."
            )

        model = getattr(
            self.artifact,
            "model",
            None,
        )

        if model is None:
            raise ValueError(
                "Artifact does not contain a trained model."
            )

        preprocessor = getattr(
            self.artifact,
            "preprocessor",
            None,
        )

        if preprocessor is None:
            raise ValueError(
                "Artifact does not contain a fitted preprocessor."
            )

        feature_names = list(
            getattr(
                metadata,
                "feature_names",
                [],
            )
        )

        if not feature_names:
            raise ValueError(
                "Artifact contains no feature schema."
            )

        if len(feature_names) != len(
            set(feature_names)
        ):
            raise ValueError(
                "Artifact feature schema contains duplicate feature names."
            )

        horizon = int(
            getattr(
                metadata,
                "horizon",
                0,
            )
        )

        if horizon <= 0:
            raise ValueError(
                "Artifact horizon must be positive."
            )

        if not hasattr(
            preprocessor,
            "transform",
        ):
            raise TypeError(
                "Artifact preprocessor must provide transform()."
            )

        if not hasattr(
            model,
            "predict_proba",
        ):
            raise TypeError(
                "Production classifier must provide predict_proba()."
            )

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
        model_id: Optional[str] = None,
    ) -> "SafeInferenceEngine":
        """
        Load a model artifact from disk.

        By default the artifact must have production approval.
        """

        artifact = load_artifact(
            artifact_path
        )

        effective_config = (
            config
            or InferenceConfig()
        )

        if (
            effective_config
            .require_approved_artifact
        ):
            cls._assert_artifact_approved(
                artifact
            )

        return cls(
            artifact,
            model_id=model_id,
            config=effective_config,
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
    ) -> "SafeInferenceEngine":
        """
        Load an approved model through the model registry.
        """

        if registry is None:
            raise ValueError(
                "A model registry is required."
            )

        if not model_id:
            raise ValueError(
                "model_id cannot be empty."
            )

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

        effective_config = (
            config
            or InferenceConfig()
        )

        return cls.from_artifact(
            artifact_path,
            config=effective_config,
            model_id=model_id,
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

        If timestamp is omitted, the latest row is used.

        No training or fitting occurs.
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

        if data.empty:
            raise ValueError(
                "Inference input contains no rows."
            )

        row, prediction_timestamp = (
            self._select_prediction_row(
                data,
                timestamp,
            )
        )

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
                transformed,
                current_price=self._extract_current_price(
                    row
                ),
            )
        )

        notes = [
            (
                "Prediction generated from an approved "
                "model artifact."
            ),
            (
                "No model fitting or feature fitting "
                "was performed during inference."
            ),
        ]

        metadata = getattr(
            self.artifact,
            "metadata",
            None,
        )

        if metadata is not None:

            if getattr(
                metadata,
                "validation_passed",
                False,
            ):
                notes.append(
                    "Artifact contains a passed validation status."
                )

            if getattr(
                metadata,
                "final_holdout_passed",
                False,
            ):
                notes.append(
                    "Artifact contains a passed final holdout status."
                )

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
        Generate predictions for every input row.

        This method performs transformation and prediction only.
        It does not fit anything.

        It is useful for diagnostics and historical analysis.
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

        if data.empty:
            return pd.DataFrame(
                index=data.index
            )

        features = self._feature_frame(
            data
        )

        transformed = (
            self._transform_features(
                features
            )
        )

        current_price = (
            self._extract_current_price(
                data
            )
        )

        prediction = (
            self._predict_transformed(
                transformed,
                current_price=current_price,
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
        """
        Validate inference data against the artifact schema.
        """

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
            self.artifact.metadata.feature_names
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

        non_numeric: list[str] = []

        high_missing: list[str] = []

        non_finite: list[str] = []

        for feature in required_features:

            if feature not in data.columns:
                continue

            series = data[
                feature
            ]

            if not pd.api.types.is_numeric_dtype(
                series
            ):
                non_numeric.append(
                    feature
                )
                continue

            missing_fraction = float(
                series.isna().mean()
            )

            if (
                missing_fraction
                > self.config.maximum_missing_fraction
            ):
                high_missing.append(
                    feature
                )

            numeric = pd.to_numeric(
                series,
                errors="coerce",
            )

            finite_mask = np.isfinite(
                numeric.to_numpy(
                    dtype=float,
                    na_value=np.nan,
                )
            )

            # NaN is treated separately as missing.
            # Infinity/-Infinity are non-finite.
            infinite_mask = (
                ~finite_mask
                & ~numeric.isna().to_numpy()
            )

            if infinite_mask.any():
                non_finite.append(
                    feature
                )

        passed = True

        if (
            missing
            and self.config.reject_missing_features
        ):
            passed = False

        if non_numeric:
            passed = False

        if high_missing:
            passed = False

        if (
            non_finite
            and self.config.require_finite_features
        ):
            passed = False

        if (
            extra
            and self.config.reject_extra_features
        ):
            passed = False

        notes: list[str] = []

        if extra:
            if self.config.reject_extra_features:
                notes.append(
                    "Extra input columns caused validation failure."
                )
            else:
                notes.append(
                    "Extra input columns will be ignored."
                )

        if not extra:
            notes.append(
                "Input contains no extra columns."
            )

        if missing:
            notes.append(
                "Required features are missing."
            )

        if non_numeric:
            notes.append(
                "One or more required features are non-numeric."
            )

        if high_missing:
            notes.append(
                "One or more required features exceed "
                "the missing-value threshold."
            )

        if non_finite:
            notes.append(
                "One or more required features contain "
                "infinite values."
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
    # Prediction-row selection
    # ------------------------------------------------------------------

    @staticmethod
    def _select_prediction_row(
        data: pd.DataFrame,
        timestamp: Optional[
            pd.Timestamp
        ],
    ) -> tuple[
        pd.DataFrame,
        pd.Timestamp,
    ]:

        if timestamp is None:

            row = data.iloc[
                [-1]
            ]

            prediction_timestamp = pd.Timestamp(
                data.index[-1]
            )

            return (
                row,
                prediction_timestamp,
            )

        prediction_timestamp = pd.Timestamp(
            timestamp
        )

        if prediction_timestamp not in data.index:
            raise ValueError(
                "Requested timestamp is not present in input data."
            )

        row = data.loc[
            [prediction_timestamp]
        ]

        return (
            row,
            prediction_timestamp,
        )

    # ------------------------------------------------------------------
    # Artifact approval
    # ------------------------------------------------------------------

    @staticmethod
    def _assert_artifact_approved(
        artifact: ModelArtifact,
    ) -> None:
        """
        Fail closed unless metadata.production_approved is exactly True.
        """

        metadata = getattr(
            artifact,
            "metadata",
            None,
        )

        if metadata is None:
            raise PermissionError(
                "Production inference is blocked because "
                "the artifact has no metadata."
            )

        approved = getattr(
            metadata,
            "production_approved",
            False,
        )

        # Important:
        #
        # Do NOT use:
        #
        #     str(approved).upper() == "TRUE"
        #
        # because malformed metadata should not accidentally
        # become deployable.
        if approved is not True:
            raise PermissionError(
                "Production inference is blocked because "
                "the model artifact is not APPROVED."
            )

    # ------------------------------------------------------------------
    # Feature handling
    # ------------------------------------------------------------------

    def _feature_frame(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:

        features = list(
            self.artifact.metadata.feature_names
        )

        # Selecting in this exact order protects against
        # accidental column reordering.
        return data[
            features
        ].copy()

    def _transform_features(
        self,
        features: pd.DataFrame,
    ) -> Any:
        """
        Apply the fitted artifact preprocessor.

        ``fit`` is intentionally never called.
        """

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

        if (
            not self.config
            .allow_nan_after_preprocessing
        ):

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
    # Current price extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_current_price(
        data: pd.DataFrame,
    ) -> Optional[float]:
        """
        Try to extract the current price from common OHLC column names.

        This value is metadata only and is not injected into the model.
        """

        candidates = (
            "Close",
            "close",
            "Adj Close",
            "adj_close",
            "Current_Price",
            "current_price",
        )

        for column in candidates:

            if column not in data.columns:
                continue

            values = pd.to_numeric(
                data[column],
                errors="coerce",
            )

            if values.empty:
                continue

            value = values.iloc[
                -1
            ]

            if pd.isna(value):
                continue

            value = float(
                value
            )

            if value > 0 and math.isfinite(
                value
            ):
                return value

        return None

    # ------------------------------------------------------------------
    # Model prediction
    # ------------------------------------------------------------------

    def _predict_transformed(
        self,
        transformed: Any,
        *,
        current_price: Optional[float] = None,
    ) -> UnifiedPrediction:
        """
        Generate a UnifiedPrediction from the approved model.

        The artifact currently contains one direction model.
        Optional return/range outputs are detected when available.
        """

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

        if probabilities.shape[1] != 2:
            raise ValueError(
                "Production direction classifier must "
                "provide exactly two class probabilities."
            )

        probability_down = (
            probabilities[:, 0]
        )

        probability_up = (
            probabilities[:, 1]
        )

        if self.config.validate_probability_output:

            self._validate_probability_array(
                probability_down,
                "down_probability",
            )

            self._validate_probability_array(
                probability_up,
                "up_probability",
            )

            probability_sum = (
                probability_down
                + probability_up
            )

            if not np.allclose(
                probability_sum,
                1.0,
                atol=self.config.probability_sum_tolerance,
            ):

                if self.config.clip_probability_output:

                    total = np.where(
                        probability_sum == 0.0,
                        1.0,
                        probability_sum,
                    )

                    probability_down = (
                        probability_down
                        / total
                    )

                    probability_up = (
                        probability_up
                        / total
                    )

                else:

                    raise ValueError(
                        "Model probability outputs do not sum to 1 "
                        "within the configured tolerance."
                    )

        predicted_direction = np.where(
            probability_up >= 0.50,
            "UP",
            "DOWN",
        )

        confidence = np.maximum(
            probability_up,
            probability_down,
        )

        model_probabilities = {
            "approved_direction_model": float(
                probability_up[0]
            )
        }

        direction_prediction = (
            DirectionPrediction(
                up_probability=float(
                    probability_up[0]
                ),
                down_probability=float(
                    probability_down[0]
                ),
                predicted_direction=str(
                    predicted_direction[0]
                ),
                confidence=float(
                    confidence[0]
                ),
                model_probabilities=(
                    model_probabilities
                ),
                disagreement=0.0,
                agreement=1.0,
            )
        )

        returns = (
            self._extract_return_prediction(
                model,
                transformed,
            )
        )

        price_range = (
            self._extract_range_prediction(
                model,
                transformed,
            )
        )

        return UnifiedPrediction(
            horizon=int(
                self.artifact.metadata.horizon
            ),
            direction=direction_prediction,
            returns=returns,
            price_range=price_range,
            current_price=current_price,
            metadata={
                "inference_engine": (
                    "SafeInferenceEngine"
                ),
                "model_id": self.model_id,
                "production_approved": True,
                "artifact_experiment_id": (
                    self.artifact.metadata.experiment_id
                ),
                "artifact_model_name": (
                    self.artifact.metadata.model_name
                ),
            },
        )

    # ------------------------------------------------------------------
    # Probability validation
    # ------------------------------------------------------------------

    def _validate_probability_array(
        self,
        values: np.ndarray,
        name: str,
    ) -> None:

        if values.size == 0:
            raise ValueError(
                f"{name} output is empty."
            )

        if not np.isfinite(
            values
        ).all():

            raise ValueError(
                f"{name} contains NaN or infinite values."
            )

        if (
            np.any(values < 0.0)
            or np.any(values > 1.0)
        ):

            raise ValueError(
                f"{name} must be between 0 and 1."
            )

    # ------------------------------------------------------------------
    # Optional return prediction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_return_prediction(
        model: Any,
        transformed: Any,
    ) -> Optional[ReturnPrediction]:
        """
        Extract an expected-return prediction when the model explicitly
        provides one.

        A normal sklearn classifier will return None here.

        This keeps inference compatible with the current direction-only
        artifact while allowing future multi-output artifacts.
        """

        predictor = getattr(
            model,
            "predict_return",
            None,
        )

        if predictor is None:
            predictor = getattr(
                model,
                "predict_expected_return",
                None,
            )

        if predictor is None:
            return None

        predicted = predictor(
            transformed
        )

        values = np.asarray(
            predicted,
            dtype=float,
        ).reshape(
            -1
        )

        if values.size == 0:
            return None

        if not np.isfinite(
            values
        ).all():

            raise ValueError(
                "Expected-return prediction contains "
                "NaN or infinite values."
            )

        return ReturnPrediction(
            predicted_return=float(
                values[0]
            ),
            model_returns={
                "approved_return_model": float(
                    values[0]
                )
            },
            disagreement=0.0,
        )

    # ------------------------------------------------------------------
    # Optional range prediction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_range_prediction(
        model: Any,
        transformed: Any,
    ) -> Optional[RangePrediction]:
        """
        Extract a return interval when the model explicitly provides one.

        Supported optional model methods:

            predict_range()
            predict_interval()

        Expected output can be:

            [lower, median, upper]

        or:

            [[lower, median, upper], ...]

        A standard binary classifier returns None.
        """

        predictor = getattr(
            model,
            "predict_range",
            None,
        )

        if predictor is None:
            predictor = getattr(
                model,
                "predict_interval",
                None,
            )

        if predictor is None:
            return None

        predicted = predictor(
            transformed
        )

        array = np.asarray(
            predicted,
            dtype=float,
        )

        if array.ndim == 1:

            if array.size != 3:
                raise ValueError(
                    "Range prediction must contain "
                    "lower, median and upper values."
                )

            lower = float(
                array[0]
            )

            median = float(
                array[1]
            )

            upper = float(
                array[2]
            )

        elif array.ndim == 2:

            if array.shape[1] != 3:
                raise ValueError(
                    "Range prediction must contain "
                    "three columns: lower, median, upper."
                )

            lower = float(
                array[0, 0]
            )

            median = float(
                array[0, 1]
            )

            upper = float(
                array[0, 2]
            )

        else:

            raise ValueError(
                "Unsupported range prediction shape."
            )

        if not all(
            math.isfinite(
                value
            )
            for value in (
                lower,
                median,
                upper,
            )
        ):
            raise ValueError(
                "Range prediction contains "
                "NaN or infinite values."
            )

        if not (
            lower
            <= median
            <= upper
        ):
            raise ValueError(
                "Range prediction must satisfy "
                "lower <= median <= upper."
            )

        return RangePrediction(
            lower_return=lower,
            median_return=median,
            upper_return=upper,
            model_ranges={
                "approved_range_model": {
                    "lower": lower,
                    "median": median,
                    "upper": upper,
                }
            },
            width=float(
                upper - lower
            ),
        )

    # ------------------------------------------------------------------
    # Batch output
    # ------------------------------------------------------------------

    @staticmethod
    def _prediction_to_frame(
        prediction: UnifiedPrediction,
        index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Convert a batch UnifiedPrediction into a DataFrame.

        The current artifact is a single-row direction model, but the
        function is also robust to vector-valued prediction objects.
        """

        direction = prediction.direction

        n = len(index)

        up_probability = np.asarray(
            direction.up_probability
        ).reshape(
            -1
        )

        down_probability = np.asarray(
            direction.down_probability
        ).reshape(
            -1
        )

        predicted_direction = np.asarray(
            direction.predicted_direction
        ).reshape(
            -1
        )

        confidence = np.asarray(
            direction.confidence
        ).reshape(
            -1
        )

        if len(up_probability) == 1 and n > 1:
            up_probability = np.repeat(
                up_probability,
                n,
            )

            down_probability = np.repeat(
                down_probability,
                n,
            )

            predicted_direction = np.repeat(
                predicted_direction,
                n,
            )

            confidence = np.repeat(
                confidence,
                n,
            )

        if len(up_probability) != n:
            raise ValueError(
                "Prediction output row count does not "
                "match the input row count."
            )

        frame = pd.DataFrame(
            index=index
        )

        frame[
            "Probability_Up"
        ] = up_probability

        frame[
            "Probability_Down"
        ] = down_probability

        frame[
            "Predicted_Direction"
        ] = predicted_direction

        frame[
            "Confidence"
        ] = confidence

        if prediction.returns is not None:

            predicted_return = np.asarray(
                prediction.returns.predicted_return
            ).reshape(
                -1
            )

            if len(predicted_return) == 1 and n > 1:

                predicted_return = np.repeat(
                    predicted_return,
                    n,
                )

            if len(predicted_return) == n:

                frame[
                    "Predicted_Return"
                ] = predicted_return

        if prediction.price_range is not None:

            range_prediction = (
                prediction.price_range
            )

            lower = np.asarray(
                range_prediction.lower_return
            ).reshape(
                -1
            )

            median = np.asarray(
                range_prediction.median_return
            ).reshape(
                -1
            )

            upper = np.asarray(
                range_prediction.upper_return
            ).reshape(
                -1
            )

            if len(lower) == 1 and n > 1:

                lower = np.repeat(
                    lower,
                    n,
                )

                median = np.repeat(
                    median,
                    n,
                )

                upper = np.repeat(
                    upper,
                    n,
                )

            if len(lower) == n:

                frame[
                    "Lower_Return"
                ] = lower

                frame[
                    "Median_Return"
                ] = median

                frame[
                    "Upper_Return"
                ] = upper

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
                "Extra features: "
                + ", ".join(
                    validation.extra_features
                )
            )

        return " ".join(
            parts
        )


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
