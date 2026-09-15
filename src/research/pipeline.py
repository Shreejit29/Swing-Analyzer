"""
AI Swing Analyser - Research Pipeline.

Central orchestration layer for reproducible model research.

The pipeline is deliberately staged:

    1. Configuration
    2. Data acquisition
    3. Data quality
    4. Feature engineering
    5. Target construction
    6. Dataset validation
    7. Temporal partitioning
    8. Model development
    9. Validation
    10. Calibration
    11. Range validation
    12. Regime validation
    13. Walk-forward backtesting
    14. Robustness
    15. Production approval
    16. Artifact / registry management

This module is an orchestration layer.

Individual algorithms remain inside their dedicated packages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

import pandas as pd

from src.data.pipeline import (
    PipelineResult,
    build_historical_dataset,
)

from src.features.engine import (
    FeatureSet,
    engineer_features,
)

from src.models.targets import (
    TargetSpec,
    build_all_targets,
)

from .config import (
    ResearchPipelineConfig,
)


# ----------------------------------------------------------------------
# Pipeline state
# ----------------------------------------------------------------------


class ResearchStage(str, Enum):
    """
    Current stage of a research run.
    """

    CREATED = "CREATED"
    DATA = "DATA"
    FEATURES = "FEATURES"
    TARGETS = "TARGETS"
    DATASET_READY = "DATASET_READY"
    MODEL_DEVELOPMENT = "MODEL_DEVELOPMENT"
    VALIDATION = "VALIDATION"
    CALIBRATION = "CALIBRATION"
    RANGE_VALIDATION = "RANGE_VALIDATION"
    REGIME_VALIDATION = "REGIME_VALIDATION"
    BACKTEST = "BACKTEST"
    ROBUSTNESS = "ROBUSTNESS"
    APPROVAL = "APPROVAL"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


# ----------------------------------------------------------------------
# Research result
# ----------------------------------------------------------------------


@dataclass
class ResearchPipelineResult:
    """
    Container for everything produced by one research run.
    """

    config: ResearchPipelineConfig

    stage: ResearchStage = ResearchStage.CREATED

    symbol: str = ""

    timeframe: str = ""

    raw_data: pd.DataFrame | None = None

    prepared_data: pd.DataFrame | None = None

    feature_set: FeatureSet | None = None

    dataset: pd.DataFrame | None = None

    target_columns: list[str] = field(
        default_factory=list
    )

    feature_columns: list[str] = field(
        default_factory=list
    )

    quality_report: Any | None = None

    validation_result: Any | None = None

    calibration_result: Any | None = None

    range_validation_result: Any | None = None

    regime_validation_result: Any | None = None

    backtest_result: Any | None = None

    robustness_result: Any | None = None

    approval_report: Any | None = None

    experiment_id: str | None = None

    model_id: str | None = None

    errors: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def successful(self) -> bool:
        """
        True only when the pipeline reached COMPLETE.
        """

        return (
            self.stage
            == ResearchStage.COMPLETE
        )

    @property
    def production_ready(self) -> bool:
        """
        Conservative production-readiness flag.

        This is intentionally false unless the dedicated approval layer
        explicitly reports approval.
        """

        if self.approval_report is None:
            return False

        status = getattr(
            self.approval_report,
            "status",
            None,
        )

        if status is None:
            return False

        return str(
            status
        ).upper().endswith(
            "APPROVED"
        )

    def summary(self) -> dict[str, Any]:
        """
        Return a compact research summary.
        """

        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "stage": self.stage.value,
            "successful": self.successful,
            "production_ready": self.production_ready,
            "rows": (
                0
                if self.dataset is None
                else len(
                    self.dataset
                )
            ),
            "features": len(
                self.feature_columns
            ),
            "targets": len(
                self.target_columns
            ),
            "experiment_id": self.experiment_id,
            "model_id": self.model_id,
            "errors": len(
                self.errors
            ),
            "warnings": len(
                self.warnings
            ),
        }


# ----------------------------------------------------------------------
# Research pipeline
# ----------------------------------------------------------------------


class ResearchPipeline:
    """
    Controlled orchestration of the AI Swing Analyser research process.

    The pipeline is intentionally dependency-injected where possible.
    This makes individual stages easier to test without requiring
    network access.
    """

    def __init__(
        self,
        config: ResearchPipelineConfig,
        *,
        data_builder: Callable[..., Any] | None = None,
        feature_builder: Callable[..., Any] | None = None,
        target_builder: Callable[..., Any] | None = None,
    ) -> None:
        if not isinstance(
            config,
            ResearchPipelineConfig,
        ):
            raise TypeError(
                "config must be a ResearchPipelineConfig."
            )

        self.config = config

        self.data_builder = (
            data_builder
            or build_historical_dataset
        )

        self.feature_builder = (
            feature_builder
            or engineer_features
        )

        self.target_builder = (
            target_builder
            or build_all_targets
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_data_stage(
        self,
        result: ResearchPipelineResult,
    ) -> ResearchPipelineResult:
        """
        Download and prepare historical data.

        No modelling occurs here.
        """

        self._require_stage(
            result,
            {
                ResearchStage.CREATED,
            },
        )

        try:
            pipeline_result = (
                self._build_data()
            )

            result.raw_data = self._extract_dataframe(
                pipeline_result
            )

            result.prepared_data = (
                result.raw_data.copy()
                if result.raw_data is not None
                else None
            )

            result.quality_report = getattr(
                pipeline_result,
                "quality",
                getattr(
                    pipeline_result,
                    "quality_report",
                    None,
                ),
            )

            if result.prepared_data is None:
                raise ValueError(
                    "Data builder returned no dataframe."
                )

            if result.prepared_data.empty:
                raise ValueError(
                    "Historical dataset is empty."
                )

            result.stage = ResearchStage.DATA

            return result

        except Exception as exc:
            return self._fail(
                result,
                "Data stage failed",
                exc,
            )

    def run_feature_stage(
        self,
        result: ResearchPipelineResult,
    ) -> ResearchPipelineResult:
        """
        Engineer technical, price-action, volume and regime features.
        """

        self._require_stage(
            result,
            {
                ResearchStage.DATA,
            },
        )

        try:
            if result.prepared_data is None:
                raise ValueError(
                    "prepared_data is missing."
                )

            feature_output = (
                self.feature_builder(
                    result.prepared_data
                )
            )

            if isinstance(
                feature_output,
                FeatureSet,
            ):
                result.feature_set = (
                    feature_output
                )

                result.dataset = (
                    feature_output.data.copy()
                )

            elif isinstance(
                feature_output,
                pd.DataFrame,
            ):
                result.dataset = (
                    feature_output.copy()
                )

                result.feature_set = None

            else:
                raise TypeError(
                    "Feature builder must return a FeatureSet "
                    "or pandas DataFrame."
                )

            if result.dataset.empty:
                raise ValueError(
                    "Feature dataset is empty."
                )

            result.feature_columns = (
                self._identify_feature_columns(
                    result.dataset
                )
            )

            result.stage = (
                ResearchStage.FEATURES
            )

            return result

        except Exception as exc:
            return self._fail(
                result,
                "Feature stage failed",
                exc,
            )

    def run_target_stage(
        self,
        result: ResearchPipelineResult,
    ) -> ResearchPipelineResult:
        """
        Construct future prediction targets.

        Target construction happens after feature engineering.

        Targets must never be allowed to enter the feature matrix.
        """

        self._require_stage(
            result,
            {
                ResearchStage.FEATURES,
            },
        )

        try:
            if result.dataset is None:
                raise ValueError(
                    "dataset is missing."
                )

            target_frames: list[
                pd.DataFrame
            ] = []

            for horizon in self.config.horizons:
                target_spec = TargetSpec(
                    horizon=int(
                        horizon
                    )
                )

                target_frame = (
                    self._build_targets(
                        result.dataset,
                        target_spec,
                    )
                )

                if not isinstance(
                    target_frame,
                    pd.DataFrame,
                ):
                    raise TypeError(
                        "Target builder must return a DataFrame."
                    )

                target_frames.append(
                    target_frame
                )

            result.dataset = (
                self._merge_target_frames(
                    result.dataset,
                    target_frames,
                )
            )

            result.target_columns = (
                self._identify_target_columns(
                    result.dataset
                )
            )

            self._assert_no_target_features(
                result.dataset,
                result.feature_columns,
                result.target_columns,
            )

            result.stage = (
                ResearchStage.TARGETS
            )

            return result

        except Exception as exc:
            return self._fail(
                result,
                "Target stage failed",
                exc,
            )

    def prepare_dataset(
        self,
        result: ResearchPipelineResult,
    ) -> ResearchPipelineResult:
        """
        Perform final structural checks before model development.
        """

        self._require_stage(
            result,
            {
                ResearchStage.TARGETS,
            },
        )

        try:
            if result.dataset is None:
                raise ValueError(
                    "dataset is missing."
                )

            dataset = result.dataset.copy()

            dataset = self._normalize_index(
                dataset
            )

            self._assert_sorted_index(
                dataset
            )

            self._assert_no_duplicate_index(
                dataset
            )

            self._assert_numeric_features(
                dataset,
                result.feature_columns,
            )

            self._assert_targets_present(
                dataset,
                result.target_columns,
            )

            self._assert_no_future_feature_names(
                result.feature_columns
            )

            result.dataset = dataset

            result.metadata.update(
                {
                    "rows": len(
                        dataset
                    ),
                    "features": len(
                        result.feature_columns
                    ),
                    "targets": len(
                        result.target_columns
                    ),
                    "horizons": list(
                        self.config.horizons
                    ),
                    "timeframe": self.config.timeframe,
                    "symbol": self.config.symbol,
                }
            )

            result.stage = (
                ResearchStage.DATASET_READY
            )

            return result

        except Exception as exc:
            return self._fail(
                result,
                "Dataset preparation failed",
                exc,
            )

    def run(
        self,
        *,
        stop_after: ResearchStage | None = None,
    ) -> ResearchPipelineResult:
        """
        Run the currently implemented research stages.

        By default the pipeline stops after DATASET_READY because the
        downstream modelling stages are deliberately connected later
        after the individual APIs have been integration-tested.

        stop_after is useful for development and testing.
        """

        result = ResearchPipelineResult(
            config=self.config,
            symbol=self.config.symbol,
            timeframe=self.config.timeframe,
        )

        stages = [
            self.run_data_stage,
            self.run_feature_stage,
            self.run_target_stage,
            self.prepare_dataset,
        ]

        for stage_function in stages:
            result = stage_function(
                result
            )

            if result.stage == ResearchStage.FAILED:
                return result

            if (
                stop_after is not None
                and result.stage == stop_after
            ):
                return result

        return result

    # ------------------------------------------------------------------
    # Data construction
    # ------------------------------------------------------------------

    def _build_data(self) -> Any:
        """
        Call the configured data builder.

        The existing data pipeline has a symbol/timeframe-oriented API,
        so this adapter keeps that dependency isolated.
        """

        try:
            return self.data_builder(
                self.config.symbol,
                timeframe=self.config.timeframe,
            )

        except TypeError:
            return self.data_builder(
                self.config.symbol
            )

    # ------------------------------------------------------------------
    # Target construction
    # ------------------------------------------------------------------

    def _build_targets(
        self,
        dataframe: pd.DataFrame,
        target_spec: TargetSpec,
    ) -> pd.DataFrame:
        """
        Build targets for exactly one prediction horizon.

        The canonical target API is `build_all_targets`.

        The research pipeline intentionally calls it with a
        single-element horizon tuple so that each target frame
        corresponds to exactly one horizon.

        This keeps target construction deterministic and avoids
        ambiguous compatibility calls.
        """

        if not isinstance(
            target_spec,
            TargetSpec,
        ):
            raise TypeError(
                "target_spec must be a TargetSpec."
            )

        horizon = int(
            target_spec.horizon
        )

        direction_threshold = float(
            getattr(
                target_spec,
                "direction_threshold",
                0.0,
            )
        )

        try:
            return build_all_targets(
                data=dataframe,
                horizons=(horizon,),
                direction_threshold=direction_threshold,
            )

        except TypeError:
            # Compatibility fallback for implementations whose
            # canonical API uses positional arguments.
            return build_all_targets(
                dataframe,
                horizons=(horizon,),
                direction_threshold=direction_threshold,
            )

    # ------------------------------------------------------------------
    # Dataframe extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_dataframe(
        pipeline_result: Any,
    ) -> pd.DataFrame | None:
        """
        Extract a dataframe from a data-pipeline result.

        Several names are supported because the data layer is still
        undergoing integration.
        """

        if isinstance(
            pipeline_result,
            pd.DataFrame,
        ):
            return pipeline_result

        candidate_names = (
            "data",
            "dataset",
            "dataframe",
            "daily",
            "prepared_data",
        )

        for name in candidate_names:
            value = getattr(
                pipeline_result,
                name,
                None,
            )

            if isinstance(
                value,
                pd.DataFrame,
            ):
                return value

        return None

    # ------------------------------------------------------------------
    # Target merging
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_target_frames(
        base: pd.DataFrame,
        target_frames: list[
            pd.DataFrame
        ],
    ) -> pd.DataFrame:
        """
        Merge target frames without overwriting source features.
        """

        result = base.copy()

        for frame in target_frames:
            if frame.empty:
                continue

            overlapping = (
                set(
                    frame.columns
                )
                & set(
                    result.columns
                )
            )

            allowed_overlap = {
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            }

            unexpected_overlap = (
                overlapping
                - allowed_overlap
            )

            if unexpected_overlap:
                frame = frame.drop(
                    columns=list(
                        unexpected_overlap
                    )
                )

            result = result.join(
                frame,
                how="left",
                rsuffix="_target",
            )

        return result

    # ------------------------------------------------------------------
    # Feature identification
    # ------------------------------------------------------------------

    @staticmethod
    def _identify_feature_columns(
        dataframe: pd.DataFrame,
    ) -> list[str]:
        excluded_prefixes = (
            "Future_",
            "Direction_",
            "Target_",
        )

        excluded_exact = {
            "Symbol",
        }

        feature_columns = []

        for column in dataframe.columns:
            name = str(
                column
            )

            if name in excluded_exact:
                continue

            if any(
                name.startswith(
                    prefix
                )
                for prefix in excluded_prefixes
            ):
                continue

            feature_columns.append(
                name
            )

        return feature_columns

    # ------------------------------------------------------------------
    # Target identification
    # ------------------------------------------------------------------

    @staticmethod
    def _identify_target_columns(
        dataframe: pd.DataFrame,
    ) -> list[str]:
        target_columns = []

        for column in dataframe.columns:
            name = str(
                column
            )

            if (
                name.startswith(
                    "Future_"
                )
                or name.startswith(
                    "Direction_"
                )
                or name.startswith(
                    "Target_"
                )
            ):
                target_columns.append(
                    name
                )

        return target_columns

    # ------------------------------------------------------------------
    # Structural checks
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_index(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        result = dataframe.copy()

        if not isinstance(
            result.index,
            pd.DatetimeIndex,
        ):
            result.index = pd.to_datetime(
                result.index,
                errors="coerce",
            )

        if result.index.isna().any():
            raise ValueError(
                "Dataset contains invalid timestamps."
            )

        if result.index.tz is not None:
            result.index = (
                result.index.tz_convert(
                    "Asia/Kolkata"
                ).tz_localize(
                    None
                )
            )

        return result

    @staticmethod
    def _assert_sorted_index(
        dataframe: pd.DataFrame,
    ) -> None:
        if not dataframe.index.is_monotonic_increasing:
            raise ValueError(
                "Dataset timestamps are not chronological."
            )

    @staticmethod
    def _assert_no_duplicate_index(
        dataframe: pd.DataFrame,
    ) -> None:
        if dataframe.index.has_duplicates:
            raise ValueError(
                "Dataset contains duplicate timestamps."
            )

    @staticmethod
    def _assert_numeric_features(
        dataframe: pd.DataFrame,
        feature_columns: list[str],
    ) -> None:
        missing = [
            column
            for column in feature_columns
            if column not in dataframe.columns
        ]

        if missing:
            raise ValueError(
                f"Missing feature columns: {missing}"
            )

        non_numeric = [
            column
            for column in feature_columns
            if not pd.api.types.is_numeric_dtype(
                dataframe[column]
            )
        ]

        if non_numeric:
            raise TypeError(
                "Non-numeric feature columns found: "
                f"{non_numeric}"
            )

    @staticmethod
    def _assert_targets_present(
        dataframe: pd.DataFrame,
        target_columns: list[str],
    ) -> None:
        if not target_columns:
            raise ValueError(
                "No target columns were generated."
            )

        missing = [
            column
            for column in target_columns
            if column not in dataframe.columns
        ]

        if missing:
            raise ValueError(
                f"Missing target columns: {missing}"
            )

    @staticmethod
    def _assert_no_future_feature_names(
        feature_columns: list[str],
    ) -> None:
        suspicious = []

        for column in feature_columns:
            lower = str(
                column
            ).lower()

            if (
                "future" in lower
                or "target" in lower
                or "forward" in lower
            ):
                suspicious.append(
                    column
                )

        if suspicious:
            raise ValueError(
                "Potential future/target leakage in feature columns: "
                f"{suspicious}"
            )

    @staticmethod
    def _assert_no_target_features(
        dataframe: pd.DataFrame,
        feature_columns: list[str],
        target_columns: list[str],
    ) -> None:
        overlap = (
            set(
                feature_columns
            )
            & set(
                target_columns
            )
        )

        if overlap:
            raise ValueError(
                "Target columns leaked into feature columns: "
                f"{sorted(overlap)}"
            )

    # ------------------------------------------------------------------
    # Stage management
    # ------------------------------------------------------------------

    @staticmethod
    def _require_stage(
        result: ResearchPipelineResult,
        allowed: set[ResearchStage],
    ) -> None:
        if result.stage not in allowed:
            allowed_text = ", ".join(
                stage.value
                for stage in sorted(
                    allowed,
                    key=lambda item: item.value,
                )
            )

            raise RuntimeError(
                f"Invalid pipeline stage "
                f"{result.stage.value}. "
                f"Expected one of: {allowed_text}"
            )

    @staticmethod
    def _fail(
        result: ResearchPipelineResult,
        message: str,
        exc: Exception,
    ) -> ResearchPipelineResult:
        result.errors.append(
            f"{message}: {exc}"
        )

        result.stage = (
            ResearchStage.FAILED
        )

        return result

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def configuration_summary(
        self,
    ) -> dict[str, Any]:
        """
        Return the active research configuration.
        """

        return self.config.summary()

    def dataset_summary(
        self,
        result: ResearchPipelineResult,
    ) -> dict[str, Any]:
        """
        Return dataset diagnostics.
        """

        if result.dataset is None:
            return {
                "available": False
            }

        return {
            "available": True,
            "rows": len(
                result.dataset
            ),
            "columns": len(
                result.dataset.columns
            ),
            "features": len(
                result.feature_columns
            ),
            "targets": len(
                result.target_columns
            ),
            "start": (
                result.dataset.index.min()
            ),
            "end": (
                result.dataset.index.max()
            ),
            "missing_fraction": float(
                result.dataset.isna()
                .mean()
                .mean()
            ),
        }


# ----------------------------------------------------------------------
# Convenience function
# ----------------------------------------------------------------------


def build_research_dataset(
    symbol: str,
    *,
    timeframe: str = "1D",
    horizons: tuple[int, ...] = (
        1,
        3,
        5,
        10,
        20,
    ),
    config: Mapping[str, Any] | None = None,
) -> ResearchPipelineResult:
    """
    Convenience function for building a research-ready dataset.

    This function stops after DATASET_READY.

    It intentionally does not train or approve a model.
    """

    pipeline_config = (
        ResearchPipelineConfig.from_mapping(
            symbol,
            config,
            timeframe=timeframe,
            horizons=horizons,
        )
    )

    pipeline = ResearchPipeline(
        pipeline_config
    )

    return pipeline.run(
        stop_after=ResearchStage.DATASET_READY
    )


__all__ = [
    "ResearchStage",
    "ResearchPipelineResult",
    "ResearchPipeline",
    "build_research_dataset",
]
