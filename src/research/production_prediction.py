"""
AI Swing Analyser — Production Prediction Gateway.

This module is the controlled boundary between research artifacts
and production predictions.

Rules:
- Only approved models may be used.
- Final holdout approval must already exist upstream.
- The model/preprocessor are never refitted here.
- Feature names must match the approved artifact.
- Missing/non-finite features fail closed.
- Prediction probabilities are treated as model outputs, not
  automatically as calibrated probabilities.
- The gateway may return WAIT / NO_HIGH_CONFIDENCE_SETUP.
- This module does not train, tune, calibrate, or approve models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.models.artifacts import ModelArtifact, load_artifact
from src.models.model_registry import ModelRegistry


class ProductionPredictionStatus(str, Enum):
    READY = "READY"
    WAIT = "WAIT"
    BLOCKED = "BLOCKED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class ProductionPredictionConfig:
    """Safety configuration for production prediction."""

    require_approved_model: bool = True
    minimum_probability: float = 0.60
    high_confidence_probability: float = 0.70
    maximum_missing_fraction: float = 0.40
    require_exact_feature_schema: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_probability <= 1.0:
            raise ValueError(
                "minimum_probability must be between 0 and 1."
            )

        if not 0.0 <= self.high_confidence_probability <= 1.0:
            raise ValueError(
                "high_confidence_probability must be between 0 and 1."
            )

        if self.high_confidence_probability < self.minimum_probability:
            raise ValueError(
                "high_confidence_probability cannot be below "
                "minimum_probability."
            )

        if not 0.0 <= self.maximum_missing_fraction < 1.0:
            raise ValueError(
                "maximum_missing_fraction must be in [0, 1)."
            )


@dataclass
class ProductionPrediction:
    """Safe production prediction result."""

    model_id: str
    status: ProductionPredictionStatus

    predicted_direction: str | None = None

    probability_up: float | None = None
    probability_down: float | None = None

    confidence: float | None = None

    high_confidence: bool = False
    trade_allowed: bool = False

    current_price: float | None = None

    feature_count: int = 0
    missing_feature_count: int = 0

    warnings: list[str] = field(
        default_factory=list
    )

    errors: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def is_ready(self) -> bool:
        return (
            self.status
            == ProductionPredictionStatus.READY
        )

    @property
    def should_wait(self) -> bool:
        return (
            self.status
            == ProductionPredictionStatus.WAIT
        )

    def summary(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "status": self.status.value,
            "predicted_direction": (
                self.predicted_direction
            ),
            "probability_up": (
                self.probability_up
            ),
            "probability_down": (
                self.probability_down
            ),
            "confidence": self.confidence,
            "high_confidence": (
                self.high_confidence
            ),
            "trade_allowed": (
                self.trade_allowed
            ),
            "current_price": (
                self.current_price
            ),
            "feature_count": self.feature_count,
            "missing_feature_count": (
                self.missing_feature_count
            ),
            "warning_count": len(
                self.warnings
            ),
            "error_count": len(
                self.errors
            ),
        }


class ProductionPredictionGateway:
    """
    Controlled production inference gateway.

    The gateway accepts a previously approved artifact and a
    feature row. It never fits or modifies the model.
    """

    def __init__(
        self,
        artifact: ModelArtifact,
        config: ProductionPredictionConfig | None = None,
    ) -> None:
        if not isinstance(
            artifact,
            ModelArtifact,
        ):
            raise TypeError(
                "artifact must be a ModelArtifact."
            )

        self.artifact = artifact

        self.config = (
            config
            if config is not None
            else ProductionPredictionConfig()
        )

        self._validate_artifact()

    @classmethod
    def from_artifact(
        cls,
        artifact_path: str,
        *,
        registry: ModelRegistry | None = None,
        config: ProductionPredictionConfig | None = None,
    ) -> "ProductionPredictionGateway":
        """
        Load a production artifact.

        When a registry is supplied, the model must be registered and
        approved before inference is allowed.
        """

        if registry is not None:
            entry = registry.assert_approved(
                artifact_path
            )

            if entry is None:
                raise RuntimeError(
                    "Model registry did not return an approved entry."
                )

        artifact = load_artifact(
            artifact_path
        )

        return cls(
            artifact=artifact,
            config=config,
        )

    def _validate_artifact(self) -> None:
        metadata = self.artifact.metadata

        approved = bool(
            getattr(
                metadata,
                "production_approved",
                False,
            )
        )

        if (
            self.config.require_approved_model
            and not approved
        ):
            raise RuntimeError(
                "Production inference is blocked because the "
                "artifact is not explicitly production approved."
            )

        feature_names = getattr(
            metadata,
            "feature_names",
            None,
        )

        if not feature_names:
            raise RuntimeError(
                "Approved artifact does not contain a feature schema."
            )

        if len(feature_names) != len(
            set(feature_names)
        ):
            raise RuntimeError(
                "Approved artifact contains duplicate feature names."
            )

    def expected_features(self) -> tuple[str, ...]:
        return tuple(
            self.artifact.metadata.feature_names
        )

    def validate_features(
        self,
        features: Mapping[str, Any] | pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        list[str],
    ]:
        """
        Validate and normalize a single production feature row.

        No fitting occurs here.
        """

        if isinstance(
            features,
            Mapping,
        ):
            frame = pd.DataFrame(
                [dict(features)]
            )

        elif isinstance(
            features,
            pd.DataFrame,
        ):
            frame = features.copy()

            if len(frame) != 1:
                raise ValueError(
                    "Production prediction requires exactly one row."
                )

        else:
            raise TypeError(
                "features must be a mapping or a one-row DataFrame."
            )

        if frame.empty:
            raise ValueError(
                "Production feature row cannot be empty."
            )

        expected = list(
            self.expected_features()
        )

        if self.config.require_exact_feature_schema:
            actual = list(frame.columns)

            if set(actual) != set(expected):
                missing = sorted(
                    set(expected) - set(actual)
                )
                extra = sorted(
                    set(actual) - set(expected)
                )

                raise ValueError(
                    "Production feature schema mismatch. "
                    f"Missing={missing}; Extra={extra}."
                )

        else:
            missing = [
                name
                for name in expected
                if name not in frame.columns
            ]

            if missing:
                raise ValueError(
                    "Required production features are missing: "
                    f"{missing}"
                )

        frame = frame.loc[:, expected]

        non_numeric = [
            column
            for column in frame.columns
            if not pd.api.types.is_numeric_dtype(
                frame[column]
            )
        ]

        if non_numeric:
            raise TypeError(
                "Production features must be numeric. "
                f"Invalid columns: {non_numeric}"
            )

        numeric = frame.astype(float)

        nonfinite = ~np.isfinite(
            numeric.to_numpy()
        )

        if nonfinite.any():
            raise ValueError(
                "Production features contain NaN or infinite values."
            )

        missing_fraction = float(
            numeric.isna()
            .mean()
            .mean()
        )

        if (
            missing_fraction
            > self.config.maximum_missing_fraction
        ):
            raise ValueError(
                "Production feature row exceeds the allowed "
                "missing-value fraction."
            )

        return (
            numeric,
            [],
        )

    def _transform(
        self,
        frame: pd.DataFrame,
    ) -> Any:
        preprocessor = self.artifact.preprocessor

        if preprocessor is None:
            raise RuntimeError(
                "Approved artifact does not contain a saved preprocessor."
            )

        transform = getattr(
            preprocessor,
            "transform",
            None,
        )

        if not callable(transform):
            raise RuntimeError(
                "Saved artifact preprocessor does not expose transform()."
            )

        return transform(frame)

    def _predict_probability(
        self,
        transformed: Any,
    ) -> tuple[float, float]:
        model = self.artifact.model

        predict_proba = getattr(
            model,
            "predict_proba",
            None,
        )

        if not callable(predict_proba):
            raise RuntimeError(
                "Production classifier must expose predict_proba()."
            )

        probabilities = np.asarray(
            predict_proba(
                transformed
            ),
            dtype=float,
        )

        if probabilities.ndim != 2:
            raise RuntimeError(
                "Classifier probability output must be 2-dimensional."
            )

        if probabilities.shape[0] != 1:
            raise RuntimeError(
                "Production prediction must produce exactly one row."
            )

        if probabilities.shape[1] != 2:
            raise RuntimeError(
                "Production classifier must expose exactly two "
                "class probabilities."
            )

        classes = getattr(
            model,
            "classes_",
            None,
        )

        if classes is None:
            raise RuntimeError(
                "Classifier classes_ are required for safe probability mapping."
            )

        classes = list(classes)

        if set(classes) != {0, 1}:
            raise RuntimeError(
                "Production classifier must use binary classes {0, 1}."
            )

        index_up = classes.index(1)
        index_down = classes.index(0)

        probability_up = float(
            probabilities[
                0,
                index_up,
            ]
        )

        probability_down = float(
            probabilities[
                0,
                index_down,
            ]
        )

        if not (
            0.0
            <= probability_up
            <= 1.0
        ):
            raise RuntimeError(
                "Invalid probability_up returned by model."
            )

        if not (
            0.0
            <= probability_down
            <= 1.0
        ):
            raise RuntimeError(
                "Invalid probability_down returned by model."
            )

        if not np.isclose(
            probability_up
            + probability_down,
            1.0,
            atol=1e-6,
        ):
            raise RuntimeError(
                "Classifier probabilities do not sum to one."
            )

        return (
            probability_up,
            probability_down,
        )

    def predict(
        self,
        features: Mapping[str, Any] | pd.DataFrame,
        *,
        current_price: float | None = None,
    ) -> ProductionPrediction:
        """
        Generate one production prediction.

        WAIT is returned when the model does not meet the configured
        minimum probability threshold.
        """

        try:
            frame, warnings = (
                self.validate_features(
                    features
                )
            )

            transformed = self._transform(
                frame
            )

            probability_up, probability_down = (
                self._predict_probability(
                    transformed
                )
            )

            confidence = max(
                probability_up,
                probability_down,
            )

            if (
                probability_up
                >= self.config.minimum_probability
                and probability_up
                > probability_down
            ):
                direction = "UP"

            elif (
                probability_down
                >= self.config.minimum_probability
                and probability_down
                > probability_up
            ):
                direction = "DOWN"

            else:
                direction = "NEUTRAL"

            high_confidence = (
                confidence
                >= self.config.high_confidence_probability
            )

            trade_allowed = (
                direction
                in {
                    "UP",
                    "DOWN",
                }
                and confidence
                >= self.config.minimum_probability
            )

            status = (
                ProductionPredictionStatus.READY
                if trade_allowed
                else ProductionPredictionStatus.WAIT
            )

            price = None

            if current_price is not None:
                price = float(
                    current_price
                )

                if not np.isfinite(price):
                    raise ValueError(
                        "current_price must be finite."
                    )

                if price <= 0:
                    raise ValueError(
                        "current_price must be positive."
                    )

            result = ProductionPrediction(
                model_id=self.artifact.metadata.model_id,
                status=status,
                predicted_direction=direction,
                probability_up=probability_up,
                probability_down=probability_down,
                confidence=confidence,
                high_confidence=high_confidence,
                trade_allowed=trade_allowed,
                current_price=price,
                feature_count=len(
                    frame.columns
                ),
                missing_feature_count=0,
                warnings=list(
                    warnings
                ),
                metadata={
                    "production_inference": True,
                    "model_fitted": False,
                    "preprocessor_fitted": False,
                    "final_holdout_fitted": False,
                    "research_only": False,
                    "artifact_version": getattr(
                        self.artifact.metadata,
                        "artifact_version",
                        None,
                    ),
                },
            )

            return result

        except Exception as exc:
            return ProductionPrediction(
                model_id=self.artifact.metadata.model_id,
                status=ProductionPredictionStatus.BLOCKED,
                trade_allowed=False,
                errors=[str(exc)],
                metadata={
                    "production_inference": True,
                    "fail_closed": True,
                    "research_only": False,
                    "production_approved": bool(
                        getattr(
                            self.artifact.metadata,
                            "production_approved",
                            False,
                        )
                    ),
                },
            )

    def predict_many(
        self,
        features: Sequence[
            Mapping[str, Any]
        ] | pd.DataFrame,
    ) -> list[ProductionPrediction]:
        """
        Predict multiple independent rows.

        Each row is processed independently. No fitting occurs.
        """

        if isinstance(
            features,
            pd.DataFrame,
        ):
            rows = [
                row.to_dict()
                for _, row in features.iterrows()
            ]
        else:
            rows = list(features)

        return [
            self.predict(row)
            for row in rows
        ]


def production_prediction_summary(
    prediction: ProductionPrediction,
) -> dict[str, Any]:
    """Return a serializable production prediction summary."""

    if not isinstance(
        prediction,
        ProductionPrediction,
    ):
        raise TypeError(
            "prediction must be a ProductionPrediction."
        )

    return prediction.summary()


__all__ = [
    "ProductionPredictionStatus",
    "ProductionPredictionConfig",
    "ProductionPrediction",
    "ProductionPredictionGateway",
    "production_prediction_summary",
]
