"""
Production prediction integration for the AI Swing Analyser.

This module provides a strict boundary between research artifacts and
production inference.

Production inference is permitted only when:
    - the approval gate passed,
    - final holdout evaluation was completed,
    - the requested horizon was approved,
    - the artifact is explicitly marked production-approved.

This module does NOT train, tune, calibrate, or select models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProductionPredictionConfig:
    """Configuration for production inference."""

    allowed_horizons: tuple[int, ...] = (
        1,
        3,
        5,
        10,
        20,
    )

    minimum_probability: float = 0.95

    require_approval: bool = True
    require_final_holdout: bool = True

    allow_wait_signal: bool = True

    def __post_init__(self) -> None:
        horizons = tuple(
            int(x)
            for x in self.allowed_horizons
        )

        if not horizons:
            raise ValueError(
                "At least one allowed horizon is required."
            )

        if any(x <= 0 for x in horizons):
            raise ValueError(
                "Allowed horizons must be positive."
            )

        if len(set(horizons)) != len(horizons):
            raise ValueError(
                "Allowed horizons must be unique."
            )

        if not 0.0 <= self.minimum_probability <= 1.0:
            raise ValueError(
                "minimum_probability must be between 0 and 1."
            )

        object.__setattr__(
            self,
            "allowed_horizons",
            horizons,
        )


@dataclass
class ProductionPrediction:
    """A production-safe prediction."""

    symbol: str
    horizon: int

    probability_up: float | None = None
    probability_down: float | None = None

    predicted_direction: str = "WAIT"

    confidence: float | None = None

    target_low: float | None = None
    target_mid: float | None = None
    target_high: float | None = None

    current_price: float | None = None

    production_approved: bool = False
    final_holdout_used: bool = False

    executable: bool = False

    reasons: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class ProductionPredictionIntegration:
    """
    Strict production inference boundary.

    The object supplied as `approval_result` must already represent the
    final production approval decision. No approval is inferred from
    prediction confidence.
    """

    def __init__(
        self,
        config: ProductionPredictionConfig | None = None,
    ) -> None:
        self.config = (
            config
            or ProductionPredictionConfig()
        )

    @staticmethod
    def _extract(
        source: Any,
        names: tuple[str, ...],
    ) -> Any:
        if source is None:
            return None

        if isinstance(
            source,
            Mapping,
        ):
            for name in names:
                if name in source:
                    return source[name]

        for name in names:
            if hasattr(
                source,
                name,
            ):
                return getattr(
                    source,
                    name,
                )

        return None

    @classmethod
    def _horizon_result(
        cls,
        source: Any,
        horizon: int,
    ) -> Any:
        if source is None:
            return None

        results = cls._extract(
            source,
            (
                "horizon_results",
                "results",
            ),
        )

        if isinstance(
            results,
            Mapping,
        ):
            return results.get(horizon)

        getter = getattr(
            source,
            "get",
            None,
        )

        if callable(getter):
            try:
                return getter(horizon)
            except (
                KeyError,
                TypeError,
            ):
                return None

        return None

    @classmethod
    def _approved_for_horizon(
        cls,
        approval_result: Any,
        horizon: int,
    ) -> bool:
        item = cls._horizon_result(
            approval_result,
            horizon,
        )

        item_approved = cls._extract(
            item,
            (
                "approved",
                "production_approved",
            ),
        )

        if item_approved is True:
            return True

        approved_horizons = cls._extract(
            approval_result,
            (
                "approved_horizons",
            ),
        )

        if approved_horizons is not None:
            return horizon in approved_horizons

        return False

    @classmethod
    def _approval_flag(
        cls,
        approval_result: Any,
    ) -> bool:
        value = cls._extract(
            approval_result,
            (
                "approved",
                "production_approved",
                "production_ready",
            ),
        )

        return value is True

    @classmethod
    def _holdout_flag(
        cls,
        approval_result: Any,
    ) -> bool:
        value = cls._extract(
            approval_result,
            (
                "final_holdout_used",
            ),
        )

        if value is True:
            return True

        metadata = cls._extract(
            approval_result,
            (
                "metadata",
            ),
        )

        if isinstance(
            metadata,
            Mapping,
        ):
            return (
                metadata.get(
                    "final_holdout_used"
                )
                is True
            )

        return False

    @staticmethod
    def _validate_probability(
        value: Any,
        name: str,
    ) -> float:
        try:
            probability = float(value)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"{name} must be numeric."
            ) from exc

        if not np.isfinite(
            probability
        ):
            raise ValueError(
                f"{name} must be finite."
            )

        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"{name} must be between 0 and 1."
            )

        return probability

    @staticmethod
    def _validate_price(
        value: Any,
        name: str,
    ) -> float | None:
        if value is None:
            return None

        try:
            price = float(value)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"{name} must be numeric."
            ) from exc

        if not np.isfinite(price):
            raise ValueError(
                f"{name} must be finite."
            )

        if price <= 0:
            raise ValueError(
                f"{name} must be positive."
            )

        return price

    @staticmethod
    def _direction_from_probability(
        probability_up: float,
        minimum_probability: float,
        allow_wait: bool,
    ) -> tuple[str, float]:
        probability_down = (
            1.0 - probability_up
        )

        confidence = max(
            probability_up,
            probability_down,
        )

        if (
            probability_up
            >= minimum_probability
        ):
            return (
                "BUY",
                confidence,
            )

        if (
            probability_down
            >= minimum_probability
        ):
            return (
                "SELL",
                confidence,
            )

        if allow_wait:
            return (
                "WAIT",
                confidence,
            )

        raise ValueError(
            "Prediction does not meet the configured "
            "minimum probability threshold."
        )

    @staticmethod
    def _validate_prediction_frame(
        features: pd.DataFrame,
    ) -> None:
        if not isinstance(
            features,
            pd.DataFrame,
        ):
            raise TypeError(
                "features must be a pandas DataFrame."
            )

        if features.empty:
            raise ValueError(
                "features cannot be empty."
            )

        if features.isna().any().any():
            raise ValueError(
                "features contain missing values."
            )

        numeric_columns = features.select_dtypes(
            include=[np.number]
        ).columns

        if len(numeric_columns) == 0:
            raise ValueError(
                "features must contain numeric model features."
            )

        numeric_values = features[
            numeric_columns
        ].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            numeric_values
        ).all():
            raise ValueError(
                "features contain NaN or infinite numeric values."
            )

    def _validate_approval(
        self,
        approval_result: Any,
        horizon: int,
    ) -> list[str]:
        reasons: list[str] = []

        if self.config.require_approval:
            if not self._approval_flag(
                approval_result
            ):
                reasons.append(
                    "Production approval gate has not passed."
                )

            if not self._approved_for_horizon(
                approval_result,
                horizon,
            ):
                reasons.append(
                    f"{horizon}D horizon is not production-approved."
                )

        if self.config.require_final_holdout:
            if not self._holdout_flag(
                approval_result
            ):
                reasons.append(
                    "Final holdout evaluation has not been verified."
                )

        return reasons

    def predict(
        self,
        symbol: str,
        horizon: int,
        probability_up: Any,
        approval_result: Any,
        *,
        current_price: Any = None,
        target_low: Any = None,
        target_mid: Any = None,
        target_high: Any = None,
    ) -> ProductionPrediction:
        """
        Convert an already-generated model probability into a production
        prediction only if all governance requirements are satisfied.
        """

        if not isinstance(
            symbol,
            str,
        ) or not symbol.strip():
            raise ValueError(
                "symbol must be a non-empty string."
            )

        try:
            horizon = int(horizon)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "horizon must be an integer."
            ) from exc

        if horizon not in (
            self.config.allowed_horizons
        ):
            raise ValueError(
                f"Unsupported horizon={horizon}."
            )

        probability_up = (
            self._validate_probability(
                probability_up,
                "probability_up",
            )
        )

        probability_down = (
            1.0 - probability_up
        )

        current_price = (
            self._validate_price(
                current_price,
                "current_price",
            )
        )

        target_low = (
            self._validate_price(
                target_low,
                "target_low",
            )
        )

        target_mid = (
            self._validate_price(
                target_mid,
                "target_mid",
            )
        )

        target_high = (
            self._validate_price(
                target_high,
                "target_high",
            )
        )

        reasons = self._validate_approval(
            approval_result,
            horizon,
        )

        holdout_used = (
            self._holdout_flag(
                approval_result
            )
        )

        if reasons:
            return ProductionPrediction(
                symbol=symbol.upper(),
                horizon=horizon,
                probability_up=probability_up,
                probability_down=probability_down,
                predicted_direction="WAIT",
                confidence=max(
                    probability_up,
                    probability_down,
                ),
                current_price=current_price,
                target_low=target_low,
                target_mid=target_mid,
                target_high=target_high,
                production_approved=False,
                final_holdout_used=holdout_used,
                executable=False,
                reasons=reasons,
                warnings=[
                    "No production trade should be executed."
                ],
                metadata={
                    "research_only": True,
                    "production_approved": False,
                    "final_holdout_used": holdout_used,
                    "model_fitted_here": False,
                    "threshold_optimized_here": False,
                },
            )

        direction, confidence = (
            self._direction_from_probability(
                probability_up=probability_up,
                minimum_probability=(
                    self.config.minimum_probability
                ),
                allow_wait=(
                    self.config.allow_wait_signal
                ),
            )
        )

        warnings: list[str] = []

        if direction == "WAIT":
            warnings.append(
                "Prediction confidence is below "
                "the configured production threshold."
            )

        executable = bool(
            direction in {"BUY", "SELL"}
            and confidence
            >= self.config.minimum_probability
        )

        if (
            target_low is not None
            and target_mid is not None
            and target_high is not None
            and not (
                target_low
                <= target_mid
                <= target_high
            )
        ):
            warnings.append(
                "Target range is not monotonically ordered."
            )
            executable = False

        return ProductionPrediction(
            symbol=symbol.upper(),
            horizon=horizon,
            probability_up=probability_up,
            probability_down=probability_down,
            predicted_direction=direction,
            confidence=confidence,
            target_low=target_low,
            target_mid=target_mid,
            target_high=target_high,
            current_price=current_price,
            production_approved=True,
            final_holdout_used=holdout_used,
            executable=executable,
            reasons=[],
            warnings=warnings,
            metadata={
                "research_only": False,
                "production_approved": True,
                "final_holdout_used": holdout_used,
                "model_fitted_here": False,
                "threshold_optimized_here": False,
                "feature_selection_here": False,
                "calibration_here": False,
            },
        )


def generate_production_prediction(
    symbol: str,
    horizon: int,
    probability_up: Any,
    approval_result: Any,
    *,
    current_price: Any = None,
    target_low: Any = None,
    target_mid: Any = None,
    target_high: Any = None,
    config: ProductionPredictionConfig | None = None,
) -> ProductionPrediction:
    """Convenience wrapper for production prediction."""

    engine = ProductionPredictionIntegration(
        config=config
    )

    return engine.predict(
        symbol=symbol,
        horizon=horizon,
        probability_up=probability_up,
        approval_result=approval_result,
        current_price=current_price,
        target_low=target_low,
        target_mid=target_mid,
        target_high=target_high,
    )


def production_prediction_summary(
    prediction: ProductionPrediction,
) -> dict[str, Any]:
    """Return a dashboard-safe production prediction."""

    if not isinstance(
        prediction,
        ProductionPrediction,
    ):
        raise TypeError(
            "prediction must be a ProductionPrediction."
        )

    return {
        "symbol": prediction.symbol,
        "horizon": prediction.horizon,
        "probability_up": prediction.probability_up,
        "probability_down": prediction.probability_down,
        "direction": prediction.predicted_direction,
        "confidence": prediction.confidence,
        "current_price": prediction.current_price,
        "target_low": prediction.target_low,
        "target_mid": prediction.target_mid,
        "target_high": prediction.target_high,
        "production_approved": (
            prediction.production_approved
        ),
        "final_holdout_used": (
            prediction.final_holdout_used
        ),
        "executable": prediction.executable,
        "reasons": list(
            prediction.reasons
        ),
        "warnings": list(
            prediction.warnings
        ),
    }
