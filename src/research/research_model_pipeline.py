"""
AI Swing Analyser — Research Model-Ready Dataset Pipeline.

Pipeline:

Raw OHLCV
    ↓
Data preparation
    ↓
Feature engineering
    ↓
Target construction
    ↓
Feature / target alignment
    ↓
Model-ready research dataset

This module does NOT:
- fit models
- tune hyperparameters
- select models
- calibrate probabilities
- optimize trading thresholds
- evaluate the final holdout
- approve production models

Important:
The final rows with unavailable future targets are retained in the
research dataset but are excluded from supervised model fitting by
the training stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data_preparation import (
    PreparedMarketData,
)
from .research_feature_pipeline import (
    ResearchFeaturePipelineResult,
)
from .research_target_pipeline import (
    ResearchTargetPipelineConfig,
    ResearchTargetPipelineResult,
    build_research_targets,
)


@dataclass(frozen=True)
class ResearchModelPipelineConfig:
    """Configuration for model-ready dataset construction."""

    target: ResearchTargetPipelineConfig = (
        ResearchTargetPipelineConfig()
    )

    minimum_rows: int = 100

    drop_rows_with_missing_targets: bool = False

    drop_rows_with_missing_features: bool = False

    reject_future_features: bool = True

    def __post_init__(self) -> None:
        if not isinstance(
            self.target,
            ResearchTargetPipelineConfig,
        ):
            raise TypeError(
                "target must be ResearchTargetPipelineConfig."
            )

        if (
            not isinstance(
                self.minimum_rows,
                int,
            )
            or isinstance(
                self.minimum_rows,
                bool,
            )
            or self.minimum_rows < 1
        ):
            raise ValueError(
                "minimum_rows must be a positive integer."
            )


@dataclass
class ResearchModelDataset:
    """Model-ready research dataset."""

    data: pd.DataFrame

    feature_columns: list[str]

    target_columns: list[str]

    direction_columns: list[str]

    return_columns: list[str]

    range_columns: list[str]

    path_columns: list[str]

    rows: int

    complete_target_rows: int

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
        return False

    def feature_matrix(
        self,
        *,
        complete_targets_only: bool = False,
    ) -> pd.DataFrame:
        """Return the model feature matrix."""

        result = self.data[
            self.feature_columns
        ].copy(
            deep=True
        )

        if complete_targets_only:
            mask = (
                self.data[
                    self.target_columns
                ]
                .notna()
                .all(axis=1)
            )

            result = result.loc[
                mask
            ].copy()

        return result

    def target_matrix(
        self,
        *,
        complete_only: bool = False,
    ) -> pd.DataFrame:
        """Return the supervised target matrix."""

        result = self.data[
            self.target_columns
        ].copy(
            deep=True
        )

        if complete_only:
            result = result.dropna(
                how="any"
            )

        return result

    def supervised_dataset(
        self,
        *,
        target_columns: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return X/y for supervised learning.

        Rows with missing selected targets are removed.

        No fitting occurs here.
        """

        selected_targets = (
            list(target_columns)
            if target_columns is not None
            else list(self.target_columns)
        )

        if not selected_targets:
            raise ValueError(
                "At least one target column is required."
            )

        unknown = sorted(
            set(selected_targets)
            - set(self.target_columns)
        )

        if unknown:
            raise ValueError(
                "Unknown target columns: "
                f"{unknown}"
            )

        subset = self.data[
            self.feature_columns
            + selected_targets
        ].dropna(
            subset=selected_targets
        )

        if subset.empty:
            raise ValueError(
                "No complete supervised-learning rows remain."
            )

        X = subset[
            self.feature_columns
        ].copy(
            deep=True
        )

        y = subset[
            selected_targets
        ].copy(
            deep=True
        )

        return X, y

    def summary(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "complete_target_rows": (
                self.complete_target_rows
            ),
            "feature_count": self.feature_count,
            "target_count": self.target_count,
            "horizons": list(
                self.horizons
            ),
            "production_ready": False,
            "research_only": True,
            "final_holdout_used": False,
            "model_fitted": False,
            "warning_count": len(
                self.warnings
            ),
        }


class ResearchModelPipeline:
    """
    Connect feature engineering and target construction.

    This creates the dataset that the model-development stage will
    consume, but does not perform model fitting.
    """

    def __init__(
        self,
        *,
        config: ResearchModelPipelineConfig
        | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else ResearchModelPipelineConfig()
        )

    def _validate_feature_result(
        self,
        feature_result: ResearchFeaturePipelineResult,
    ) -> None:
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
                "Cannot construct model dataset from "
                "a failed feature pipeline."
            )

        if feature_result.feature_matrix.empty:
            raise ValueError(
                "Feature matrix is empty."
            )

    def _validate_target_result(
        self,
        target_result: ResearchTargetPipelineResult,
    ) -> None:
        if not isinstance(
            target_result,
            ResearchTargetPipelineResult,
        ):
            raise TypeError(
                "target_result must be "
                "ResearchTargetPipelineResult."
            )

        if not target_result.target_columns:
            raise ValueError(
                "No target columns were generated."
            )

    def _validate_feature_target_alignment(
        self,
        features: ResearchFeaturePipelineResult,
        targets: ResearchTargetPipelineResult,
    ) -> None:
        if not features.feature_matrix.index.equals(
            targets.data.index
        ):
            raise ValueError(
                "Feature and target timelines do not match."
            )

        overlap = sorted(
            set(
                features.features.feature_names
            )
            & set(
                targets.target_columns
            )
        )

        if overlap:
            raise ValueError(
                "Feature/target overlap detected: "
                f"{overlap}"
            )

    def _validate_no_future_features(
        self,
        feature_columns: list[str],
    ) -> None:
        if not self.config.reject_future_features:
            return

        suspicious_tokens = (
            "future_",
            "direction_",
            "target_",
            "target",
            "label",
        )

        suspicious = [
            column
            for column in feature_columns
            if any(
                token in column.lower()
                for token in suspicious_tokens
            )
        ]

        if suspicious:
            raise ValueError(
                "Future/target information detected "
                "inside model features: "
                f"{suspicious}"
            )

    def build(
        self,
        feature_result: ResearchFeaturePipelineResult,
    ) -> ResearchModelDataset:
        """
        Build a complete model-ready research dataset.
        """

        self._validate_feature_result(
            feature_result
        )

        # -----------------------------------------------------------
        # Target construction
        # -----------------------------------------------------------

        target_result = build_research_targets(
            feature_result,
            config=self.config.target,
        )

        self._validate_target_result(
            target_result
        )

        self._validate_feature_target_alignment(
            feature_result,
            target_result,
        )

        feature_columns = list(
            feature_result.features.feature_names
        )

        target_columns = list(
            target_result.target_columns
        )

        self._validate_no_future_features(
            feature_columns
        )

        # -----------------------------------------------------------
        # Build final dataframe
        # -----------------------------------------------------------

        data = target_result.data[
            feature_columns
            + target_columns
        ].copy(
            deep=True
        )

        if not data.index.is_monotonic_increasing:
            raise RuntimeError(
                "Model dataset is not chronologically ordered."
            )

        if data.index.has_duplicates:
            raise RuntimeError(
                "Model dataset contains duplicate timestamps."
            )

        # -----------------------------------------------------------
        # Feature missingness
        # -----------------------------------------------------------

        feature_missing = (
            data[
                feature_columns
            ]
            .isna()
            .any(axis=1)
        )

        if (
            self.config.drop_rows_with_missing_features
            and feature_missing.any()
        ):
            data = data.loc[
                ~feature_missing
            ].copy()

        # -----------------------------------------------------------
        # Target missingness
        # -----------------------------------------------------------

        target_missing = (
            data[
                target_columns
            ]
            .isna()
            .any(axis=1)
        )

        complete_target_rows = int(
            (~target_missing).sum()
        )

        warnings: list[str] = list(
            target_result.warnings
        )

        if target_missing.any():
            warnings.append(
                f"{int(target_missing.sum())} rows contain "
                "unavailable future targets."
            )

        # -----------------------------------------------------------
        # Numeric validation
        # -----------------------------------------------------------

        for column in (
            feature_columns
            + target_columns
        ):
            if not pd.api.types.is_numeric_dtype(
                data[column]
            ):
                raise TypeError(
                    "Non-numeric model dataset column: "
                    f"{column}"
                )

        numeric_values = data[
            feature_columns
            + target_columns
        ].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        if (
            numeric_values[
                feature_columns
            ]
            .isna()
            .all(axis=0)
            .any()
        ):
            raise ValueError(
                "At least one feature contains no usable values."
            )

        # -----------------------------------------------------------
        # Minimum rows
        # -----------------------------------------------------------

        if len(data) < (
            self.config.minimum_rows
        ):
            warnings.append(
                f"Only {len(data)} rows remain; "
                f"minimum recommended rows are "
                f"{self.config.minimum_rows}."
            )

        # -----------------------------------------------------------
        # Metadata
        # -----------------------------------------------------------

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
            "threshold_optimization_completed": False,
            "approval_granted": False,
            "feature_count": len(
                feature_columns
            ),
            "target_count": len(
                target_columns
            ),
            "complete_target_rows": (
                complete_target_rows
            ),
        }

        return ResearchModelDataset(
            data=data,
            feature_columns=feature_columns,
            target_columns=target_columns,
            direction_columns=list(
                target_result.direction_columns
            ),
            return_columns=list(
                target_result.return_columns
            ),
            range_columns=list(
                target_result.range_columns
            ),
            path_columns=list(
                target_result.path_columns
            ),
            rows=len(data),
            complete_target_rows=(
                complete_target_rows
            ),
            horizons=(
                target_result.horizons
            ),
            warnings=warnings,
            metadata=metadata,
        )

    def build_for_horizon(
        self,
        feature_result: ResearchFeaturePipelineResult,
        horizon: int,
    ) -> ResearchModelDataset:
        """
        Build a dataset configured for one prediction horizon.
        """

        if (
            not isinstance(
                horizon,
                int,
            )
            or isinstance(
                horizon,
                bool,
            )
            or horizon <= 0
        ):
            raise ValueError(
                "horizon must be a positive integer."
            )

        config = ResearchModelPipelineConfig(
            target=ResearchTargetPipelineConfig(
                horizons=(horizon,),
                direction_threshold=(
                    self.config.target.direction_threshold
                ),
                include_path_targets=(
                    self.config.target.include_path_targets
                ),
                include_range_targets=(
                    self.config.target.include_range_targets
                ),
                include_mfe_mae=(
                    self.config.target.include_mfe_mae
                ),
                reject_future_features=(
                    self.config.target.reject_future_features
                ),
            ),
            minimum_rows=(
                self.config.minimum_rows
            ),
            drop_rows_with_missing_targets=(
                self.config.drop_rows_with_missing_targets
            ),
            drop_rows_with_missing_features=(
                self.config.drop_rows_with_missing_features
            ),
            reject_future_features=(
                self.config.reject_future_features
            ),
        )

        return ResearchModelPipeline(
            config=config
        ).build(
            feature_result
        )


def build_model_dataset(
    feature_result: ResearchFeaturePipelineResult,
    *,
    config: ResearchModelPipelineConfig
    | None = None,
) -> ResearchModelDataset:
    """Convenience function for model-ready dataset construction."""

    pipeline = ResearchModelPipeline(
        config=config
    )

    return pipeline.build(
        feature_result
    )


def build_horizon_dataset(
    feature_result: ResearchFeaturePipelineResult,
    horizon: int,
    *,
    config: ResearchModelPipelineConfig
    | None = None,
) -> ResearchModelDataset:
    """Build a model dataset for one horizon."""

    pipeline = ResearchModelPipeline(
        config=config
    )

    return pipeline.build_for_horizon(
        feature_result,
        horizon,
    )


def research_model_dataset_summary(
    result: ResearchModelDataset,
) -> dict[str, object]:
    """Return a compact model-dataset summary."""

    if not isinstance(
        result,
        ResearchModelDataset,
    ):
        raise TypeError(
            "result must be ResearchModelDataset."
        )

    return result.summary()


__all__ = [
    "ResearchModelPipelineConfig",
    "ResearchModelDataset",
    "ResearchModelPipeline",
    "build_model_dataset",
    "build_horizon_dataset",
    "research_model_dataset_summary",
]
