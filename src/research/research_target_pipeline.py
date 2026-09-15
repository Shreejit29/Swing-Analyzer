"""
AI Swing Analyser — Research Target Pipeline.

Pipeline boundary:

Prepared market data
        ↓
Validated feature matrix
        ↓
Future target construction
        ↓
Feature/target separation
        ↓
Leakage audit

Supported research horizons:
- 1
- 3
- 5
- 10
- 20

This module does NOT:
- train models
- select models
- calibrate probabilities
- optimize thresholds
- evaluate the final holdout
- approve production models

Important:
Future targets are created deliberately for supervised learning,
but they are never exposed as model features.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .research_feature_pipeline import (
    ResearchFeaturePipelineResult,
)
from src.models.targets import (
    TargetSpec,
    build_all_targets,
)


DEFAULT_RESEARCH_HORIZONS = (
    1,
    3,
    5,
    10,
    20,
)


@dataclass(frozen=True)
class ResearchTargetPipelineConfig:
    """Configuration for supervised target construction."""

    horizons: tuple[int, ...] = (
        DEFAULT_RESEARCH_HORIZONS
    )

    direction_threshold: float = 0.0

    include_path_targets: bool = True

    include_range_targets: bool = True

    include_mfe_mae: bool = True

    reject_future_features: bool = True

    def __post_init__(self) -> None:
        if not self.horizons:
            raise ValueError(
                "At least one horizon is required."
            )

        normalized = tuple(
            int(horizon)
            for horizon in self.horizons
        )

        if any(
            horizon <= 0
            for horizon in normalized
        ):
            raise ValueError(
                "All horizons must be positive integers."
            )

        if len(normalized) != len(
            set(normalized)
        ):
            raise ValueError(
                "Duplicate horizons are not allowed."
            )

        if self.direction_threshold < 0:
            raise ValueError(
                "direction_threshold cannot be negative."
            )


@dataclass
class ResearchTargetPipelineResult:
    """Result of target construction."""

    data: pd.DataFrame

    feature_columns: list[str]

    target_columns: list[str]

    direction_columns: list[str]

    return_columns: list[str]

    range_columns: list[str]

    path_columns: list[str]

    rows: int

    horizons: tuple[int, ...]

    warnings: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, object] = field(
        default_factory=dict
    )

    @property
    def feature_count(self) -> int:
        return len(
            self.feature_columns
        )

    @property
    def target_count(self) -> int:
        return len(
            self.target_columns
        )

    @property
    def production_ready(self) -> bool:
        """
        Target construction never grants production readiness.
        """

        return False

    def feature_matrix(self) -> pd.DataFrame:
        """Return only model features."""

        return self.data[
            self.feature_columns
        ].copy(
            deep=True
        )

    def target_matrix(self) -> pd.DataFrame:
        """Return only supervised-learning targets."""

        return self.data[
            self.target_columns
        ].copy(
            deep=True
        )

    def summary(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "feature_count": self.feature_count,
            "target_count": self.target_count,
            "horizons": list(
                self.horizons
            ),
            "production_ready": False,
            "research_only": True,
            "final_holdout_used": False,
            "warning_count": len(
                self.warnings
            ),
        }


class ResearchTargetPipeline:
    """
    Controlled feature → target construction pipeline.
    """

    def __init__(
        self,
        *,
        config: ResearchTargetPipelineConfig
        | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else ResearchTargetPipelineConfig()
        )

    def _target_specs(
        self,
    ) -> list[TargetSpec]:
        """
        Build target specifications for all requested horizons.
        """

        specs: list[TargetSpec] = []

        for horizon in self.config.horizons:
            specs.append(
                TargetSpec(
                    horizon=horizon
                )
            )

        return specs

    def _identify_features(
        self,
        data: pd.DataFrame,
    ) -> list[str]:
        """
        Identify model features.

        Anything explicitly representing a future target,
        direction label, target, symbol, or internal timestamp
        is excluded.
        """

        excluded_tokens = (
            "future_",
            "direction_",
            "target_",
            "target",
            "label",
        )

        excluded_exact = {
            "Symbol",
            "_source_time",
            "_base_time",
        }

        features: list[str] = []

        for column in data.columns:
            if column in excluded_exact:
                continue

            lowered = column.lower()

            if any(
                token in lowered
                for token in excluded_tokens
            ):
                continue

            if pd.api.types.is_numeric_dtype(
                data[column]
            ):
                features.append(
                    column
                )

        return features

    def _identify_targets(
        self,
        data: pd.DataFrame,
    ) -> list[str]:
        """
        Identify target columns created by the target builder.
        """

        target_columns: list[str] = []

        for column in data.columns:
            lowered = column.lower()

            if (
                lowered.startswith(
                    "future_"
                )
                or lowered.startswith(
                    "direction_"
                )
                or lowered.startswith(
                    "target_"
                )
            ):
                target_columns.append(
                    column
                )

        return target_columns

    def _validate_feature_target_separation(
        self,
        feature_columns: list[str],
        target_columns: list[str],
    ) -> None:
        """
        Ensure targets cannot appear in the feature matrix.
        """

        overlap = sorted(
            set(feature_columns)
            & set(target_columns)
        )

        if overlap:
            raise ValueError(
                "Feature/target overlap detected: "
                f"{overlap}"
            )

        if self.config.reject_future_features:
            suspicious_features = [
                column
                for column in feature_columns
                if any(
                    token
                    in column.lower()
                    for token in (
                        "future_",
                        "direction_",
                        "target_",
                        "target",
                        "label",
                    )
                )
            ]

            if suspicious_features:
                raise ValueError(
                    "Future/target information detected "
                    "inside feature columns: "
                    f"{suspicious_features}"
                )

    def _validate_targets(
        self,
        data: pd.DataFrame,
        target_columns: list[str],
    ) -> None:
        """Validate generated target columns."""

        if not target_columns:
            raise ValueError(
                "No target columns were generated."
            )

        for column in target_columns:
            if not pd.api.types.is_numeric_dtype(
                data[column]
            ):
                raise TypeError(
                    "Target column is not numeric: "
                    f"{column}"
                )

            values = data[column].dropna()

            if not np.isfinite(
                values.to_numpy()
            ).all():
                raise ValueError(
                    "Target contains non-finite values: "
                    f"{column}"
                )

    def build(
        self,
        feature_result: ResearchFeaturePipelineResult,
    ) -> ResearchTargetPipelineResult:
        """
        Create future supervised-learning targets.

        The feature matrix from the previous stage is preserved,
        while future targets are appended only to the research
        dataset representation.
        """

        if not isinstance(
            feature_result,
            ResearchFeaturePipelineResult,
        ):
            raise TypeError(
                "feature_result must be "
                "ResearchFeaturePipelineResult."
            )

        if not feature_result.success:
            raise ValueError(
                "Cannot construct targets from a failed "
                "feature pipeline."
            )

        base_data = feature_result.features.data.copy(
            deep=True
        )

        original_index = base_data.index.copy()

        # -----------------------------------------------------------
        # Build supervised targets
        # -----------------------------------------------------------

        target_specs = self._target_specs()

        target_data = build_all_targets(
            base_data.copy(
                deep=True
            ),
            target_specs,
        )

        if not isinstance(
            target_data,
            pd.DataFrame,
        ):
            raise TypeError(
                "build_all_targets must return a DataFrame."
            )

        # -----------------------------------------------------------
        # Timeline protection
        # -----------------------------------------------------------

        if not target_data.index.equals(
            original_index
        ):
            raise RuntimeError(
                "Target construction changed the research timeline."
            )

        # -----------------------------------------------------------
        # Identify targets
        # -----------------------------------------------------------

        target_columns = self._identify_targets(
            target_data
        )

        self._validate_targets(
            target_data,
            target_columns,
        )

        # -----------------------------------------------------------
        # Identify feature columns
        # -----------------------------------------------------------

        feature_columns = (
            self._identify_features(
                target_data
            )
        )

        self._validate_feature_target_separation(
            feature_columns,
            target_columns,
        )

        # -----------------------------------------------------------
        # Categorize targets
        # -----------------------------------------------------------

        direction_columns = [
            column
            for column in target_columns
            if column.lower().startswith(
                "direction_"
            )
        ]

        return_columns = [
            column
            for column in target_columns
            if column.lower().startswith(
                "future_close_return"
            )
            or (
                "return"
                in column.lower()
                and column
                not in direction_columns
            )
        ]

        range_columns = [
            column
            for column in target_columns
            if any(
                token
                in column.lower()
                for token in (
                    "high",
                    "low",
                    "range",
                )
            )
        ]

        path_columns = [
            column
            for column in target_columns
            if any(
                token
                in column.lower()
                for token in (
                    "mfe",
                    "mae",
                    "path",
                )
            )
        ]

        warnings: list[str] = []

        missing_target_rows = (
            target_data[
                target_columns
            ]
            .isna()
            .any(axis=1)
            .sum()
        )

        if missing_target_rows:
            warnings.append(
                f"{missing_target_rows} rows contain "
                "incomplete future targets and should not "
                "be used for supervised training."
            )

        metadata = {
            "research_only": True,
            "production_ready": False,
            "production_approved": False,
            "future_targets_created": True,
            "future_targets_used_as_features": False,
            "final_holdout_used": False,
            "model_fitted": False,
            "model_selected": False,
            "calibration_fitted": False,
            "approval_granted": False,
            "horizons": list(
                self.config.horizons
            ),
        }

        return ResearchTargetPipelineResult(
            data=target_data,
            feature_columns=feature_columns,
            target_columns=target_columns,
            direction_columns=direction_columns,
            return_columns=return_columns,
            range_columns=range_columns,
            path_columns=path_columns,
            rows=len(target_data),
            horizons=self.config.horizons,
            warnings=warnings,
            metadata=metadata,
        )


def build_research_targets(
    feature_result: ResearchFeaturePipelineResult,
    *,
    config: ResearchTargetPipelineConfig
    | None = None,
) -> ResearchTargetPipelineResult:
    """Convenience function for target construction."""

    pipeline = ResearchTargetPipeline(
        config=config
    )

    return pipeline.build(
        feature_result
    )


def research_target_pipeline_summary(
    result: ResearchTargetPipelineResult,
) -> dict[str, object]:
    """Return a compact target-pipeline summary."""

    if not isinstance(
        result,
        ResearchTargetPipelineResult,
    ):
        raise TypeError(
            "result must be ResearchTargetPipelineResult."
        )

    return result.summary()


__all__ = [
    "DEFAULT_RESEARCH_HORIZONS",
    "ResearchTargetPipelineConfig",
    "ResearchTargetPipelineResult",
    "ResearchTargetPipeline",
    "build_research_targets",
    "research_target_pipeline_summary",
]
