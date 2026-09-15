"""
AI Swing Analyser — Production Multi-Horizon Prediction.

Combines independently approved production models across multiple
forecast horizons.

Default swing horizons:
    1D
    3D
    5D
    10D
    20D

Important:
- Each horizon uses its own approved model/artifact.
- No model is trained here.
- No preprocessing is fitted here.
- No calibration is performed here.
- A failed horizon never becomes a forced trade.
- The aggregator may return WAIT.
- Agreement across horizons is diagnostic and does not replace
  calibrated probability or approval gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .production_prediction import (
    ProductionPrediction,
    ProductionPredictionConfig,
    ProductionPredictionGateway,
    ProductionPredictionStatus,
)


DEFAULT_HORIZONS = (
    1,
    3,
    5,
    10,
    20,
)


@dataclass(frozen=True)
class MultiHorizonConfig:
    """Configuration for multi-horizon production inference."""

    horizons: tuple[int, ...] = DEFAULT_HORIZONS

    minimum_agreement_fraction: float = 0.60
    minimum_ready_horizons: int = 1

    preferred_horizon: int = 5

    def __post_init__(self) -> None:
        if not self.horizons:
            raise ValueError(
                "At least one horizon is required."
            )

        if any(
            not isinstance(horizon, int)
            or isinstance(horizon, bool)
            or horizon <= 0
            for horizon in self.horizons
        ):
            raise ValueError(
                "Horizons must be positive integers."
            )

        if len(self.horizons) != len(
            set(self.horizons)
        ):
            raise ValueError(
                "Horizons must be unique."
            )

        if not 0.0 <= (
            self.minimum_agreement_fraction
        ) <= 1.0:
            raise ValueError(
                "minimum_agreement_fraction must be between 0 and 1."
            )

        if (
            self.minimum_ready_horizons < 1
        ):
            raise ValueError(
                "minimum_ready_horizons must be at least 1."
            )

        if (
            self.minimum_ready_horizons
            > len(self.horizons)
        ):
            raise ValueError(
                "minimum_ready_horizons cannot exceed the number of horizons."
            )

        if (
            self.preferred_horizon
            not in self.horizons
        ):
            raise ValueError(
                "preferred_horizon must be one of the configured horizons."
            )


@dataclass
class MultiHorizonPrediction:
    """Combined production prediction across horizons."""

    model_id: str

    status: ProductionPredictionStatus

    horizon_predictions: dict[
        int,
        ProductionPrediction,
    ] = field(
        default_factory=dict
    )

    consensus_direction: str = "NEUTRAL"

    consensus_probability: float | None = None

    agreement_fraction: float = 0.0

    ready_horizon_count: int = 0

    high_confidence_horizon_count: int = 0

    preferred_horizon: int | None = None

    trade_allowed: bool = False

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
            "consensus_direction": (
                self.consensus_direction
            ),
            "consensus_probability": (
                self.consensus_probability
            ),
            "agreement_fraction": (
                self.agreement_fraction
            ),
            "ready_horizon_count": (
                self.ready_horizon_count
            ),
            "high_confidence_horizon_count": (
                self.high_confidence_horizon_count
            ),
            "preferred_horizon": (
                self.preferred_horizon
            ),
            "trade_allowed": (
                self.trade_allowed
            ),
            "horizons": sorted(
                self.horizon_predictions.keys()
            ),
            "warning_count": len(
                self.warnings
            ),
            "error_count": len(
                self.errors
            ),
        }


class MultiHorizonPredictionEngine:
    """
    Production-safe multi-horizon prediction engine.

    Each horizon must have its own ProductionPredictionGateway.
    """

    def __init__(
        self,
        gateways: Mapping[
            int,
            ProductionPredictionGateway,
        ],
        config: MultiHorizonConfig | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else MultiHorizonConfig()
        )

        if not gateways:
            raise ValueError(
                "At least one production gateway is required."
            )

        self.gateways: dict[
            int,
            ProductionPredictionGateway,
        ] = {}

        for horizon, gateway in gateways.items():
            if (
                not isinstance(horizon, int)
                or isinstance(horizon, bool)
                or horizon <= 0
            ):
                raise ValueError(
                    "Gateway horizons must be positive integers."
                )

            if not isinstance(
                gateway,
                ProductionPredictionGateway,
            ):
                raise TypeError(
                    "Every gateway must be a "
                    "ProductionPredictionGateway."
                )

            self.gateways[horizon] = gateway

        missing = set(
            self.config.horizons
        ) - set(
            self.gateways
        )

        if missing:
            raise ValueError(
                "Missing production gateways for horizons: "
                f"{sorted(missing)}"
            )

    @property
    def horizons(self) -> tuple[int, ...]:
        return self.config.horizons

    def predict(
        self,
        features_by_horizon: Mapping[
            int,
            Mapping[str, Any],
        ],
        *,
        current_price: float | None = None,
    ) -> MultiHorizonPrediction:
        """
        Generate independent predictions for every configured horizon.

        Missing features for one horizon do not cause another horizon
        to be silently reused.
        """

        predictions: dict[
            int,
            ProductionPrediction,
        ] = {}

        warnings: list[str] = []
        errors: list[str] = []

        model_ids = set()

        for horizon in self.horizons:
            gateway = self.gateways[
                horizon
            ]

            if horizon not in features_by_horizon:
                prediction = ProductionPrediction(
                    model_id=(
                        gateway.artifact.metadata.model_id
                    ),
                    status=(
                        ProductionPredictionStatus.BLOCKED
                    ),
                    errors=[
                        "Features for this horizon are missing."
                    ],
                    metadata={
                        "horizon": horizon,
                        "fail_closed": True,
                    },
                )

                predictions[horizon] = prediction

                errors.append(
                    f"{horizon}D: missing features."
                )

                continue

            prediction = gateway.predict(
                features_by_horizon[
                    horizon
                ],
                current_price=current_price,
            )

            predictions[horizon] = prediction

            model_ids.add(
                prediction.model_id
            )

            for warning in prediction.warnings:
                warnings.append(
                    f"{horizon}D: {warning}"
                )

            for error in prediction.errors:
                errors.append(
                    f"{horizon}D: {error}"
                )

        ready_predictions = {
            horizon: prediction
            for horizon, prediction
            in predictions.items()
            if prediction.status
            in {
                ProductionPredictionStatus.READY,
                ProductionPredictionStatus.WAIT,
            }
        }

        trade_predictions = {
            horizon: prediction
            for horizon, prediction
            in ready_predictions.items()
            if prediction.trade_allowed
        }

        ready_count = len(
            ready_predictions
        )

        high_confidence_count = sum(
            prediction.high_confidence
            for prediction
            in ready_predictions.values()
        )

        if not ready_predictions:
            model_id = (
                next(iter(model_ids))
                if model_ids
                else "multi_horizon"
            )

            return MultiHorizonPrediction(
                model_id=model_id,
                status=(
                    ProductionPredictionStatus.BLOCKED
                ),
                horizon_predictions=predictions,
                ready_horizon_count=0,
                high_confidence_horizon_count=0,
                warnings=warnings,
                errors=errors
                or [
                    "No horizon produced a valid prediction."
                ],
                metadata={
                    "multi_horizon": True,
                    "fail_closed": True,
                },
            )

        directions = [
            prediction.predicted_direction
            for prediction
            in trade_predictions.values()
            if prediction.predicted_direction
            in {
                "UP",
                "DOWN",
            }
        ]

        if directions:
            up_count = directions.count(
                "UP"
            )

            down_count = directions.count(
                "DOWN"
            )

            if up_count > down_count:
                consensus_direction = "UP"
            elif down_count > up_count:
                consensus_direction = "DOWN"
            else:
                consensus_direction = "NEUTRAL"
        else:
            consensus_direction = "NEUTRAL"

        directional_count = len(
            directions
        )

        if directional_count:
            consensus_count = max(
                directions.count("UP"),
                directions.count("DOWN"),
            )

            agreement_fraction = (
                consensus_count
                / directional_count
            )
        else:
            agreement_fraction = 0.0

        probabilities: list[float] = []

        for prediction in trade_predictions.values():
            if (
                prediction.predicted_direction
                == "UP"
                and prediction.probability_up
                is not None
            ):
                probabilities.append(
                    prediction.probability_up
                )

            elif (
                prediction.predicted_direction
                == "DOWN"
                and prediction.probability_down
                is not None
            ):
                probabilities.append(
                    prediction.probability_down
                )

        consensus_probability = (
            float(
                np.mean(probabilities)
            )
            if probabilities
            else None
        )

        preferred = predictions.get(
            self.config.preferred_horizon
        )

        preferred_is_tradeable = (
            preferred is not None
            and preferred.trade_allowed
        )

        sufficient_horizons = (
            ready_count
            >= self.config.minimum_ready_horizons
        )

        sufficient_agreement = (
            agreement_fraction
            >= self.config.minimum_agreement_fraction
        )

        consensus_trade_allowed = (
            sufficient_horizons
            and sufficient_agreement
            and consensus_direction
            in {
                "UP",
                "DOWN",
            }
        )

        if consensus_trade_allowed:
            status = (
                ProductionPredictionStatus.READY
            )
        else:
            status = (
                ProductionPredictionStatus.WAIT
            )

        if not preferred_is_tradeable:
            warnings.append(
                "Preferred horizon does not currently "
                "meet its individual trade threshold."
            )

        model_id = (
            next(iter(model_ids))
            if len(model_ids) == 1
            else "multi_horizon"
        )

        return MultiHorizonPrediction(
            model_id=model_id,
            status=status,
            horizon_predictions=predictions,
            consensus_direction=(
                consensus_direction
            ),
            consensus_probability=(
                consensus_probability
            ),
            agreement_fraction=(
                agreement_fraction
            ),
            ready_horizon_count=ready_count,
            high_confidence_horizon_count=(
                high_confidence_count
            ),
            preferred_horizon=(
                self.config.preferred_horizon
            ),
            trade_allowed=(
                consensus_trade_allowed
            ),
            warnings=warnings,
            errors=errors,
            metadata={
                "multi_horizon": True,
                "configured_horizons": list(
                    self.horizons
                ),
                "preferred_horizon": (
                    self.config.preferred_horizon
                ),
                "model_fitted": False,
                "preprocessor_fitted": False,
                "calibration_fitted": False,
                "final_holdout_fitted": False,
                "production_approved": False,
                "research_only": False,
            },
        )

    def predict_same_features(
        self,
        features: Mapping[str, Any],
        *,
        current_price: float | None = None,
    ) -> MultiHorizonPrediction:
        """
        Convenience method for cases where all horizons intentionally
        use the same feature vector.

        This should only be used when the approved artifacts have
        identical feature schemas.
        """

        features_by_horizon = {
            horizon: dict(features)
            for horizon in self.horizons
        }

        return self.predict(
            features_by_horizon,
            current_price=current_price,
        )


def build_multi_horizon_engine(
    gateways: Mapping[
        int,
        ProductionPredictionGateway,
    ],
    *,
    config: MultiHorizonConfig | None = None,
) -> MultiHorizonPredictionEngine:
    """Create a production multi-horizon prediction engine."""

    return MultiHorizonPredictionEngine(
        gateways=gateways,
        config=config,
    )


def multi_horizon_summary(
    prediction: MultiHorizonPrediction,
) -> dict[str, Any]:
    """Return a serializable multi-horizon summary."""

    if not isinstance(
        prediction,
        MultiHorizonPrediction,
    ):
        raise TypeError(
            "prediction must be a MultiHorizonPrediction."
        )

    return prediction.summary()


__all__ = [
    "DEFAULT_HORIZONS",
    "MultiHorizonConfig",
    "MultiHorizonPrediction",
    "MultiHorizonPredictionEngine",
    "build_multi_horizon_engine",
    "multi_horizon_summary",
]
