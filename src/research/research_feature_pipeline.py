"""
AI Swing Analyser — Research Feature Pipeline.

Pipeline:

Prepared market data
        ↓
Base / MTF / market-context features
        ↓
Feature validation
        ↓
Research feature matrix

This module does NOT:
- create future targets
- train models
- select models
- calibrate probabilities
- evaluate the final holdout
- approve production models
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data_preparation import (
    PreparedMarketData,
)
from .feature_integration import (
    UnifiedFeatureConfig,
    UnifiedFeatureResult,
    build_unified_features,
    feature_matrix,
)
from .market_data_context import (
    MarketContextData,
)


@dataclass(frozen=True)
class ResearchFeaturePipelineConfig:
    """Configuration for research feature construction."""

    unified: UnifiedFeatureConfig = (
        UnifiedFeatureConfig()
    )

    minimum_features: int = 10

    require_numeric_features: bool = True

    reject_future_features: bool = True

    def __post_init__(self) -> None:
        if not isinstance(
            self.unified,
            UnifiedFeatureConfig,
        ):
            raise TypeError(
                "unified must be UnifiedFeatureConfig."
            )

        if (
            not isinstance(
                self.minimum_features,
                int,
            )
            or isinstance(
                self.minimum_features,
                bool,
            )
            or self.minimum_features < 1
        ):
            raise ValueError(
                "minimum_features must be a positive integer."
            )


@dataclass
class ResearchFeaturePipelineResult:
    """Result of research feature construction."""

    prepared_data: PreparedMarketData

    features: UnifiedFeatureResult

    feature_matrix: pd.DataFrame

    success: bool

    warnings: list[str] = field(
        default_factory=list
    )

    errors: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, object] = field(
        default_factory=dict
    )

    @property
    def feature_count(self) -> int:
        return len(
            self.features.feature_names
        )

    @property
    def rows(self) -> int:
        return len(
            self.feature_matrix
        )

    @property
    def production_ready(self) -> bool:
        """Feature generation never grants production readiness."""

        return False

    def summary(self) -> dict[str, object]:
        return {
            "success": self.success,
            "rows": self.rows,
            "feature_count": self.feature_count,
            "production_ready": False,
            "warning_count": len(
                self.warnings
            ),
            "error_count": len(
                self.errors
            ),
            "research_only": True,
            "final_holdout_used": False,
        }


class ResearchFeaturePipeline:
    """
    Controlled prepared-data → feature-matrix pipeline.
    """

    def __init__(
        self,
        *,
        config: ResearchFeaturePipelineConfig
        | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else ResearchFeaturePipelineConfig()
        )

    def build(
        self,
        prepared: PreparedMarketData,
        *,
        market_context: MarketContextData
        | None = None,
        timeframe_data: dict[
            str,
            pd.DataFrame,
        ]
        | None = None,
    ) -> ResearchFeaturePipelineResult:
        """
        Construct and validate the research feature matrix.
        """

        if not isinstance(
            prepared,
            PreparedMarketData,
        ):
            raise TypeError(
                "prepared must be PreparedMarketData."
            )

        if not prepared.passed:
            raise ValueError(
                "Cannot build features from failed "
                "market-data preparation."
            )

        warnings = list(
            prepared.warnings
        )

        errors: list[str] = []

        try:
            unified = build_unified_features(
                prepared.data,
                market_context=(
                    market_context
                ),
                timeframe_data=(
                    timeframe_data
                ),
                config=self.config.unified,
            )

            warnings.extend(
                unified.warnings
            )

            matrix = feature_matrix(
                unified
            )

            self._validate_feature_matrix(
                matrix
            )

            success = True

        except Exception as exc:
            unified = None
            matrix = pd.DataFrame(
                index=prepared.data.index
            )

            errors.append(
                str(exc)
            )

            success = False

        if unified is None:
            raise ValueError(
                "Feature construction failed: "
                + "; ".join(errors)
            )

        metadata = {
            "research_only": True,
            "production_ready": False,
            "production_approved": False,
            "future_values_used": False,
            "targets_created": False,
            "model_fitted": False,
            "final_holdout_used": False,
            "calibration_fitted": False,
            "feature_selection_completed": False,
            "feature_count": len(
                unified.feature_names
            ),
        }

        return ResearchFeaturePipelineResult(
            prepared_data=prepared,
            features=unified,
            feature_matrix=matrix,
            success=success,
            warnings=warnings,
            errors=errors,
            metadata=metadata,
        )

    def _validate_feature_matrix(
        self,
        matrix: pd.DataFrame,
    ) -> None:
        """Apply strict feature-matrix validation."""

        if not isinstance(
            matrix,
            pd.DataFrame,
        ):
            raise TypeError(
                "Feature matrix must be a DataFrame."
            )

        if matrix.empty:
            raise ValueError(
                "Feature matrix is empty."
            )

        if len(matrix.columns) < (
            self.config.minimum_features
        ):
            raise ValueError(
                "Only "
                f"{len(matrix.columns)} features were generated; "
                f"minimum required is "
                f"{self.config.minimum_features}."
            )

        if matrix.columns.has_duplicates:
            raise ValueError(
                "Feature matrix contains duplicate columns."
            )

        if not matrix.index.is_monotonic_increasing:
            raise ValueError(
                "Feature matrix index is not chronological."
            )

        if matrix.index.has_duplicates:
            raise ValueError(
                "Feature matrix contains duplicate timestamps."
            )

        if self.config.require_numeric_features:
            non_numeric = [
                column
                for column in matrix.columns
                if not pd.api.types.is_numeric_dtype(
                    matrix[column]
                )
            ]

            if non_numeric:
                raise TypeError(
                    "Non-numeric features detected: "
                    f"{non_numeric}"
                )

        values = matrix.select_dtypes(
            include=[np.number]
        )

        if not np.isfinite(
            values.fillna(0.0).to_numpy()
        ).all():
            raise ValueError(
                "Feature matrix contains non-finite values."
            )

        if self.config.reject_future_features:
            suspicious_tokens = (
                "future_",
                "direction_",
                "target_",
                "target",
                "label",
            )

            suspicious = [
                column
                for column in matrix.columns
                if any(
                    token
                    in column.lower()
                    for token in suspicious_tokens
                )
            ]

            if suspicious:
                raise ValueError(
                    "Potential future/target features detected: "
                    f"{suspicious}"
                )

    def summary(
        self,
        result: ResearchFeaturePipelineResult,
    ) -> dict[str, object]:
        """Return a compact feature-pipeline summary."""

        if not isinstance(
            result,
            ResearchFeaturePipelineResult,
        ):
            raise TypeError(
                "result must be ResearchFeaturePipelineResult."
            )

        return result.summary()


def build_research_features(
    prepared: PreparedMarketData,
    *,
    market_context: MarketContextData
    | None = None,
    timeframe_data: dict[
        str,
        pd.DataFrame,
    ]
    | None = None,
    config: ResearchFeaturePipelineConfig
    | None = None,
) -> ResearchFeaturePipelineResult:
    """Convenience function for research feature construction."""

    pipeline = ResearchFeaturePipeline(
        config=config
    )

    return pipeline.build(
        prepared,
        market_context=market_context,
        timeframe_data=timeframe_data,
    )


def research_feature_pipeline_summary(
    result: ResearchFeaturePipelineResult,
) -> dict[str, object]:
    """Return a compact summary."""

    if not isinstance(
        result,
        ResearchFeaturePipelineResult,
    ):
        raise TypeError(
            "result must be ResearchFeaturePipelineResult."
        )

    return result.summary()


__all__ = [
    "ResearchFeaturePipelineConfig",
    "ResearchFeaturePipelineResult",
    "ResearchFeaturePipeline",
    "build_research_features",
    "research_feature_pipeline_summary",
]
