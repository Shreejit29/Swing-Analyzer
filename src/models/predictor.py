"""
Unified prediction engine for AI Swing Analyser.

This module combines:
    - Direction classifiers
    - Return regressors
    - Price-range models
    - Model ensembles

It produces a single structured prediction that can later be passed to:
    - calibration
    - decision/gating
    - swing setup construction
    - backtesting
    - Streamlit dashboard

This module does NOT perform training, validation, or live trading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence

import math
import numpy as np

from .classifier import DirectionClassifier
from .ensemble import ModelEnsemble
from .range_model import PriceRangeModel
from .regressor import ReturnRegressor


@dataclass
class PredictionConfig:
    """Configuration for the unified prediction engine."""

    horizon: int = 5

    minimum_probability: float = 0.50

    require_range_prediction: bool = True

    clip_probability: bool = True

    probability_sum_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")

        if not 0.0 <= self.minimum_probability <= 1.0:
            raise ValueError(
                "minimum_probability must be between 0 and 1."
            )

        if self.probability_sum_tolerance <= 0:
            raise ValueError(
                "probability_sum_tolerance must be positive."
            )


@dataclass
class DirectionPrediction:
    """Directional prediction."""

    up_probability: float
    down_probability: float
    predicted_direction: str
    confidence: float

    model_probabilities: Dict[str, float] = field(
        default_factory=dict
    )

    disagreement: float = 0.0
    agreement: float = 1.0


@dataclass
class ReturnPrediction:
    """Expected return prediction."""

    predicted_return: float

    model_returns: Dict[str, float] = field(
        default_factory=dict
    )

    disagreement: float = 0.0


@dataclass
class RangePrediction:
    """Predicted future-return interval."""

    lower_return: float
    median_return: float
    upper_return: float

    model_ranges: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )

    width: float = 0.0


@dataclass
class UnifiedPrediction:
    """
    Complete model prediction for one stock and one horizon.
    """

    horizon: int

    direction: DirectionPrediction

    returns: Optional[ReturnPrediction]

    price_range: Optional[RangePrediction]

    current_price: Optional[float]

    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def predicted_direction(self) -> str:
        return self.direction.predicted_direction

    @property
    def up_probability(self) -> float:
        return self.direction.up_probability

    @property
    def down_probability(self) -> float:
        return self.direction.down_probability

    @property
    def confidence(self) -> float:
        return self.direction.confidence

    @property
    def predicted_return(self) -> Optional[float]:
        if self.returns is None:
            return None

        return self.returns.predicted_return

    @property
    def lower_return(self) -> Optional[float]:
        if self.price_range is None:
            return None

        return self.price_range.lower_return

    @property
    def median_return(self) -> Optional[float]:
        if self.price_range is None:
            return None

        return self.price_range.median_return

    @property
    def upper_return(self) -> Optional[float]:
        if self.price_range is None:
            return None

        return self.price_range.upper_return


class UnifiedPredictor:
    """
    Unified prediction service.

    Models are expected to already be trained.

    The predictor intentionally accepts pre-fitted models rather than
    training anything internally. This prevents accidental mixing of
    training and inference data.
    """

    def __init__(
        self,
        config: Optional[PredictionConfig] = None,
    ) -> None:
        self.config = config or PredictionConfig()

        self.classifiers: Dict[str, DirectionClassifier] = {}
        self.regressors: Dict[str, ReturnRegressor] = {}
        self.range_models: Dict[str, PriceRangeModel] = {}

        self.classifier_ensemble: Optional[ModelEnsemble] = None
        self.regression_ensemble: Optional[ModelEnsemble] = None
        self.range_ensemble: Optional[ModelEnsemble] = None

    # ------------------------------------------------------------------
    # Model registration
    # ------------------------------------------------------------------

    def add_classifier(
        self,
        name: str,
        model: DirectionClassifier,
    ) -> None:
        """Register a trained direction classifier."""

        self._validate_name(name)

        if not isinstance(model, DirectionClassifier):
            raise TypeError(
                "model must be a DirectionClassifier."
            )

        self.classifiers[name] = model

    def add_regressor(
        self,
        name: str,
        model: ReturnRegressor,
    ) -> None:
        """Register a trained return regressor."""

        self._validate_name(name)

        if not isinstance(model, ReturnRegressor):
            raise TypeError(
                "model must be a ReturnRegressor."
            )

        self.regressors[name] = model

    def add_range_model(
        self,
        name: str,
        model: PriceRangeModel,
    ) -> None:
        """Register a trained price-range model."""

        self._validate_name(name)

        if not isinstance(model, PriceRangeModel):
            raise TypeError(
                "model must be a PriceRangeModel."
            )

        self.range_models[name] = model

    def set_classifier_ensemble(
        self,
        ensemble: ModelEnsemble,
    ) -> None:
        self.classifier_ensemble = ensemble

    def set_regression_ensemble(
        self,
        ensemble: ModelEnsemble,
    ) -> None:
        self.regression_ensemble = ensemble

    def set_range_ensemble(
        self,
        ensemble: ModelEnsemble,
    ) -> None:
        self.range_ensemble = ensemble

    # ------------------------------------------------------------------
    # Main prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        features,
        current_price: Optional[float] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> UnifiedPrediction:
        """
        Generate a unified prediction.

        Parameters
        ----------
        features:
            Feature matrix containing the columns expected by the
            already-trained models.

        current_price:
            Current/latest price. Used only for metadata at this stage.

        metadata:
            Optional metadata such as symbol, timeframe, regime, etc.
        """

        if current_price is not None:
            if current_price <= 0:
                raise ValueError(
                    "current_price must be positive."
                )

        direction = self._predict_direction(features)

        returns = self._predict_returns(features)

        price_range = self._predict_range(features)

        if (
            self.config.require_range_prediction
            and price_range is None
        ):
            raise RuntimeError(
                "Range prediction is required but no trained "
                "range model is available."
            )

        return UnifiedPrediction(
            horizon=self.config.horizon,
            direction=direction,
            returns=returns,
            price_range=price_range,
            current_price=current_price,
            metadata=dict(metadata or {}),
        )

    # ------------------------------------------------------------------
    # Direction
    # ------------------------------------------------------------------

    def _predict_direction(
        self,
        features,
    ) -> DirectionPrediction:
        if not self.classifiers:
            raise RuntimeError(
                "At least one direction classifier is required."
            )

        probabilities: Dict[str, float] = {}

        for name, model in self.classifiers.items():
            probability = self._extract_up_probability(
                model.predict_up_probability(features)
            )

            probabilities[name] = probability

        if self.classifier_ensemble is not None:
            up_probability = float(
                self.classifier_ensemble.probability_average(
                    probabilities
                )
            )
        else:
            up_probability = float(
                np.mean(list(probabilities.values()))
            )

        up_probability = self._sanitize_probability(
            up_probability
        )

        down_probability = 1.0 - up_probability

        predicted_direction = self._direction_from_probability(
            up_probability
        )

        confidence = max(
            up_probability,
            down_probability,
        )

        disagreement = self._calculate_disagreement(
            probabilities.values()
        )

        agreement = 1.0 - disagreement

        return DirectionPrediction(
            up_probability=up_probability,
            down_probability=down_probability,
            predicted_direction=predicted_direction,
            confidence=confidence,
            model_probabilities=probabilities,
            disagreement=disagreement,
            agreement=agreement,
        )

    # ------------------------------------------------------------------
    # Return prediction
    # ------------------------------------------------------------------

    def _predict_returns(
        self,
        features,
    ) -> Optional[ReturnPrediction]:
        if not self.regressors:
            return None

        predictions: Dict[str, float] = {}

        for name, model in self.regressors.items():
            prediction = model.predict(features)

            prediction_value = self._extract_scalar(
                prediction
            )

            if not math.isfinite(prediction_value):
                raise ValueError(
                    f"Regressor '{name}' produced a non-finite prediction."
                )

            predictions[name] = prediction_value

        if self.regression_ensemble is not None:
            predicted_return = float(
                self.regression_ensemble.return_average(
                    predictions
                )
            )
        else:
            predicted_return = float(
                np.mean(list(predictions.values()))
            )

        disagreement = self._calculate_numeric_disagreement(
            predictions.values()
        )

        return ReturnPrediction(
            predicted_return=predicted_return,
            model_returns=predictions,
            disagreement=disagreement,
        )

    # ------------------------------------------------------------------
    # Range prediction
    # ------------------------------------------------------------------

    def _predict_range(
        self,
        features,
    ) -> Optional[RangePrediction]:
        if not self.range_models:
            return None

        ranges: Dict[str, Dict[str, float]] = {}

        for name, model in self.range_models.items():
            prediction = model.predict_returns(features)

            lower, median, upper = self._extract_range(
                prediction
            )

            if lower > upper:
                lower, upper = upper, lower

            median = min(
                max(median, lower),
                upper,
            )

            ranges[name] = {
                "lower": lower,
                "median": median,
                "upper": upper,
            }

        if self.range_ensemble is not None:
            combined = self.range_ensemble.range_average(
                ranges
            )

            lower = float(combined["lower"])
            median = float(combined["median"])
            upper = float(combined["upper"])

        else:
            lower = float(
                np.mean(
                    [
                        value["lower"]
                        for value in ranges.values()
                    ]
                )
            )

            median = float(
                np.mean(
                    [
                        value["median"]
                        for value in ranges.values()
                    ]
                )
            )

            upper = float(
                np.mean(
                    [
                        value["upper"]
                        for value in ranges.values()
                    ]
                )
            )

        if lower > upper:
            lower, upper = upper, lower

        median = min(
            max(median, lower),
            upper,
        )

        return RangePrediction(
            lower_return=lower,
            median_return=median,
            upper_return=upper,
            model_ranges=ranges,
            width=upper - lower,
        )

    # ------------------------------------------------------------------
    # Ensemble management
    # ------------------------------------------------------------------

    def build_equal_weight_ensembles(self) -> None:
        """
        Build equal-weight ensembles from currently registered models.
        """

        if self.classifiers:
            ensemble = ModelEnsemble()
            ensemble.equal_weights(
                list(self.classifiers.keys())
            )
            self.classifier_ensemble = ensemble

        if self.regressors:
            ensemble = ModelEnsemble()
            ensemble.equal_weights(
                list(self.regressors.keys())
            )
            self.regression_ensemble = ensemble

        if self.range_models:
            ensemble = ModelEnsemble()
            ensemble.equal_weights(
                list(self.range_models.keys())
            )
            self.range_ensemble = ensemble

    # ------------------------------------------------------------------
    # Probability helpers
    # ------------------------------------------------------------------

    def _direction_from_probability(
        self,
        up_probability: float,
    ) -> str:
        if (
            up_probability
            >= self.config.minimum_probability
        ):
            return "UP"

        if (
            up_probability
            <= 1.0 - self.config.minimum_probability
        ):
            return "DOWN"

        return "NEUTRAL"

    def _sanitize_probability(
        self,
        probability: float,
    ) -> float:
        if not math.isfinite(probability):
            raise ValueError(
                "Model produced a non-finite probability."
            )

        if self.config.clip_probability:
            return float(
                min(
                    max(probability, 0.0),
                    1.0,
                )
            )

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "Probability must be between 0 and 1."
            )

        return float(probability)

    # ------------------------------------------------------------------
    # Numeric extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_up_probability(
        prediction,
    ) -> float:
        """
        Extract a scalar probability from model output.

        Supports:
            scalar
            numpy scalar
            single-element array
            one-row Series/DataFrame-like output
        """

        array = np.asarray(prediction)

        if array.size == 0:
            raise ValueError(
                "Empty probability prediction."
            )

        if array.size > 1:
            # If a model returns [probability], use it.
            # If it returns a larger vector, this is ambiguous.
            if array.size != 1:
                raise ValueError(
                    "Expected one probability value for inference."
                )

        return float(array.reshape(-1)[0])

    @staticmethod
    def _extract_scalar(
        prediction,
    ) -> float:
        array = np.asarray(prediction)

        if array.size == 0:
            raise ValueError(
                "Empty regression prediction."
            )

        if array.size != 1:
            raise ValueError(
                "Expected one regression prediction."
            )

        return float(array.reshape(-1)[0])

    @staticmethod
    def _extract_range(
        prediction,
    ) -> tuple[float, float, float]:
        """
        Extract lower, median and upper range predictions.

        Supported structures:
            - dictionary with lower/median/upper
            - 3-element array
            - 3-element sequence
        """

        if isinstance(prediction, Mapping):
            required = (
                "lower",
                "median",
                "upper",
            )

            missing = [
                key
                for key in required
                if key not in prediction
            ]

            if missing:
                raise ValueError(
                    f"Range prediction missing keys: {missing}"
                )

            values = (
                float(prediction["lower"]),
                float(prediction["median"]),
                float(prediction["upper"]),
            )

        else:
            array = np.asarray(prediction).reshape(-1)

            if array.size != 3:
                raise ValueError(
                    "Range prediction must contain "
                    "lower, median and upper values."
                )

            values = (
                float(array[0]),
                float(array[1]),
                float(array[2]),
            )

        if not all(
            math.isfinite(value)
            for value in values
        ):
            raise ValueError(
                "Range prediction contains non-finite values."
            )

        return values

    # ------------------------------------------------------------------
    # Disagreement
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_disagreement(
        probabilities,
    ) -> float:
        values = np.asarray(
            list(probabilities),
            dtype=float,
        )

        if values.size <= 1:
            return 0.0

        standard_deviation = float(
            np.std(values)
        )

        # A standard deviation of 0.5 would represent extremely
        # contradictory binary predictions. Normalize to [0, 1].
        disagreement = min(
            standard_deviation / 0.5,
            1.0,
        )

        return float(disagreement)

    @staticmethod
    def _calculate_numeric_disagreement(
        values,
    ) -> float:
        array = np.asarray(
            list(values),
            dtype=float,
        )

        if array.size <= 1:
            return 0.0

        mean_abs = float(
            np.mean(np.abs(array))
        )

        if mean_abs <= 1e-12:
            return 0.0

        standard_deviation = float(
            np.std(array)
        )

        # Relative disagreement.
        disagreement = standard_deviation / (
            abs(mean_abs) + 1e-12
        )

        return float(
            min(disagreement, 1.0)
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_name(name: str) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "Model name must be a non-empty string."
            )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def model_summary(self) -> Dict[str, object]:
        """Return a summary of registered models."""

        return {
            "classifiers": list(
                self.classifiers.keys()
            ),
            "regressors": list(
                self.regressors.keys()
            ),
            "range_models": list(
                self.range_models.keys()
            ),
            "classifier_ensemble": (
                self.classifier_ensemble is not None
            ),
            "regression_ensemble": (
                self.regression_ensemble is not None
            ),
            "range_ensemble": (
                self.range_ensemble is not None
            ),
            "horizon": self.config.horizon,
        }


def prediction_to_dict(
    prediction: UnifiedPrediction,
) -> Dict[str, object]:
    """Convert a prediction into a dashboard/API-friendly dictionary."""

    result: Dict[str, object] = {
        "horizon": prediction.horizon,
        "predicted_direction": prediction.predicted_direction,
        "up_probability": prediction.up_probability,
        "down_probability": prediction.down_probability,
        "confidence": prediction.confidence,
        "current_price": prediction.current_price,
        "direction_disagreement": (
            prediction.direction.disagreement
        ),
        "direction_agreement": (
            prediction.direction.agreement
        ),
        "metadata": dict(prediction.metadata),
    }

    if prediction.returns is not None:
        result["predicted_return"] = (
            prediction.returns.predicted_return
        )
        result["return_disagreement"] = (
            prediction.returns.disagreement
        )
        result["model_returns"] = dict(
            prediction.returns.model_returns
        )
    else:
        result["predicted_return"] = None

    if prediction.price_range is not None:
        result["lower_return"] = (
            prediction.price_range.lower_return
        )
        result["median_return"] = (
            prediction.price_range.median_return
        )
        result["upper_return"] = (
            prediction.price_range.upper_return
        )
        result["range_width"] = (
            prediction.price_range.width
        )
        result["model_ranges"] = dict(
            prediction.price_range.model_ranges
        )
    else:
        result["lower_return"] = None
        result["median_return"] = None
        result["upper_return"] = None

    return result


def price_range_from_prediction(
    prediction: UnifiedPrediction,
    current_price: Optional[float] = None,
) -> Optional[Dict[str, float]]:
    """
    Convert predicted returns into predicted prices.

    Returns
    -------
    dict or None
        lower, median and upper predicted prices.
    """

    price = (
        current_price
        if current_price is not None
        else prediction.current_price
    )

    if price is None:
        return None

    if price <= 0:
        raise ValueError(
            "current_price must be positive."
        )

    if prediction.price_range is None:
        return None

    lower = price * (
        1.0 + prediction.price_range.lower_return
    )

    median = price * (
        1.0 + prediction.price_range.median_return
    )

    upper = price * (
        1.0 + prediction.price_range.upper_return
    )

    return {
        "lower_price": float(lower),
        "median_price": float(median),
        "upper_price": float(upper),
    }


__all__ = [
    "PredictionConfig",
    "DirectionPrediction",
    "ReturnPrediction",
    "RangePrediction",
    "UnifiedPrediction",
    "UnifiedPredictor",
    "prediction_to_dict",
    "price_range_from_prediction",
]


# Backward-compatible public name used by the inference layer.
PredictionEngine = UnifiedPredictor
