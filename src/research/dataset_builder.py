"""
Research dataset construction.

This module creates the dataset used by the research pipeline while
enforcing strict separation between:

    raw market data
        -> engineered features
        -> future targets

The dataset builder is intentionally conservative. A dataset that fails
the leakage audit must not proceed to model development.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from src.data.quality import QualityReport, assert_quality, validate_ohlcv
from src.features.engine import engineer_features
from src.models.targets import TargetSpec, build_all_targets
from src.research.leakage_audit import (
    LeakageAuditConfig,
    LeakageAuditReport,
    assert_leakage_free,
    audit_research_dataset,
)


@dataclass(frozen=True)
class ResearchDatasetResult:
    """
    Result produced by the research dataset builder.
    """

    data: pd.DataFrame
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    symbol: str
    timeframe: str
    horizons: tuple[int, ...]

    quality_report: QualityReport | None = None
    leakage_report: LeakageAuditReport | None = None

    warnings: tuple[str, ...] = field(
        default_factory=tuple
    )
    metadata: dict = field(
        default_factory=dict
    )

    @property
    def rows(self) -> int:
        return len(self.data)

    @property
    def passed(self) -> bool:
        if self.quality_report is not None:
            if not self.quality_report.passed:
                return False

        if self.leakage_report is not None:
            if not self.leakage_report.passed:
                return False

        return True

    @property
    def production_safe(self) -> bool:
        """
        Dataset readiness is not the same thing as model approval.

        This property therefore remains False. A clean dataset is only
        a prerequisite for research.
        """

        return False

    def summary(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "rows": self.rows,
            "features": len(
                self.feature_columns
            ),
            "targets": len(
                self.target_columns
            ),
            "horizons": list(
                self.horizons
            ),
            "passed": self.passed,
            "production_safe": self.production_safe,
            "warnings": list(
                self.warnings
            ),
            "metadata": dict(
                self.metadata
            ),
        }


class ResearchDatasetBuilder:
    """
    Build a leakage-audited research dataset.

    The builder is dependency-injectable so unit tests can replace the
    data, feature, and target construction functions.
    """

    def __init__(
        self,
        data_builder: Callable | None = None,
        feature_builder: Callable | None = None,
        target_builder: Callable | None = None,
        leakage_config: LeakageAuditConfig | None = None,
        minimum_rows: int = 500,
        maximum_feature_ratio: float = 0.25,
        maximum_missing_fraction: float = 0.40,
    ) -> None:
        self.data_builder = (
            data_builder
            if data_builder is not None
            else self._default_data_builder
        )

        self.feature_builder = (
            feature_builder
            if feature_builder is not None
            else engineer_features
        )

        self.target_builder = (
            target_builder
            if target_builder is not None
            else build_all_targets
        )

        self.leakage_config = (
            leakage_config
            if leakage_config is not None
            else LeakageAuditConfig()
        )

        if minimum_rows < 1:
            raise ValueError(
                "minimum_rows must be positive."
            )

        if not (
            0.0
            < maximum_feature_ratio
            <= 1.0
        ):
            raise ValueError(
                "maximum_feature_ratio must be in (0, 1]."
            )

        if not (
            0.0
            <= maximum_missing_fraction
            < 1.0
        ):
            raise ValueError(
                "maximum_missing_fraction must be in [0, 1)."
            )

        self.minimum_rows = minimum_rows
        self.maximum_feature_ratio = (
            maximum_feature_ratio
        )
        self.maximum_missing_fraction = (
            maximum_missing_fraction
        )

    # ------------------------------------------------------------------
    # Default builders
    # ------------------------------------------------------------------

    @staticmethod
    def _default_data_builder(
        symbol: str,
        timeframe: str = "1D",
    ) -> pd.DataFrame:
        """
        Default raw-data loader.

        Imports are intentionally local so that importing this module
        does not automatically trigger market-data dependencies.
        """

        from src.data.pipeline import (
            build_historical_dataset,
        )

        result = build_historical_dataset(
            symbol=symbol,
            timeframe=timeframe,
        )

        if hasattr(result, "data"):
            return result.data

        if isinstance(result, pd.DataFrame):
            return result

        raise TypeError(
            "Default data builder did not return a DataFrame."
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_basic_dataframe(
        data: pd.DataFrame,
    ) -> None:
        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise TypeError(
                "Research data must be a pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                "Research data is empty."
            )

        if not isinstance(
            data.index,
            pd.DatetimeIndex,
        ):
            raise TypeError(
                "Research data must use a DatetimeIndex."
            )

        if data.index.has_duplicates:
            raise ValueError(
                "Research data contains duplicate timestamps."
            )

        if not data.index.is_monotonic_increasing:
            raise ValueError(
                "Research data must be chronologically sorted."
            )

    @staticmethod
    def _identify_features(
        data: pd.DataFrame,
    ) -> list[str]:
        """
        Identify candidate model features.

        Target columns and obvious metadata columns are excluded.
        """

        excluded = {
            "Symbol",
            "Target",
            "_source_time",
            "_base_time",
        }

        features: list[str] = []

        for column in data.columns:
            name = str(column)

            if name in excluded:
                continue

            if name.startswith(
                "Future_"
            ):
                continue

            if name.startswith(
                "Direction_"
            ):
                continue

            if name.startswith(
                "Target_"
            ):
                continue

            features.append(name)

        return features

    @staticmethod
    def _identify_targets(
        data: pd.DataFrame,
    ) -> list[str]:
        """
        Identify future prediction targets.
        """

        targets: list[str] = []

        for column in data.columns:
            name = str(column)

            if (
                name.startswith("Future_")
                or name.startswith("Direction_")
            ):
                targets.append(name)

        return targets

    @staticmethod
    def _validate_feature_target_separation(
        features: Sequence[str],
        targets: Sequence[str],
    ) -> None:
        overlap = sorted(
            set(features).intersection(
                targets
            )
        )

        if overlap:
            raise ValueError(
                "Feature/target overlap detected: "
                f"{overlap}"
            )

    @staticmethod
    def _validate_numeric_features(
        data: pd.DataFrame,
        features: Sequence[str],
    ) -> None:
        for column in features:
            if not pd.api.types.is_numeric_dtype(
                data[column]
            ):
                raise TypeError(
                    f"Feature '{column}' is not numeric."
                )

    @staticmethod
    def _validate_finite_features(
        data: pd.DataFrame,
        features: Sequence[str],
    ) -> None:
        values = data.loc[
            :,
            list(features),
        ]

        if np.isinf(
            values.to_numpy(
                dtype=float,
                na_value=np.nan,
            )
        ).any():
            raise ValueError(
                "Feature matrix contains infinite values."
            )

    # ------------------------------------------------------------------
    # Target construction
    # ------------------------------------------------------------------

    def _build_targets(
        self,
        data: pd.DataFrame,
        horizons: Sequence[int],
    ) -> pd.DataFrame:
        """
        Build all requested future targets.

        The target builder is isolated from feature engineering so that
        future information can never accidentally become an input.
        """

        target_specs = [
            TargetSpec(
                horizon=int(horizon)
            )
            for horizon in horizons
        ]

        try:
            result = self.target_builder(
                data,
                target_specs,
            )
        except TypeError:
            # Compatibility with target builders that accept a horizon
            # sequence directly.
            result = self.target_builder(
                data,
                horizons,
            )

        if not isinstance(
            result,
            pd.DataFrame,
        ):
            raise TypeError(
                "Target builder must return a DataFrame."
            )

        return result

    # ------------------------------------------------------------------
    # Warnings
    # ------------------------------------------------------------------

    def _generate_warnings(
        self,
        data: pd.DataFrame,
        features: Sequence[str],
    ) -> list[str]:
        warnings: list[str] = []

        if len(data) < self.minimum_rows:
            warnings.append(
                "Dataset contains fewer than "
                f"{self.minimum_rows} observations. "
                "Statistical conclusions may be unstable."
            )

        if len(features) > 0:
            feature_ratio = (
                len(features)
                / max(len(data), 1)
            )

            if (
                feature_ratio
                > self.maximum_feature_ratio
            ):
                warnings.append(
                    "Feature-to-observation ratio is high "
                    f"({feature_ratio:.3f}). "
                    "Consider feature selection and regularization."
                )

        missing_fraction = (
            data.loc[
                :,
                list(features),
            ]
            .isna()
            .mean()
            .max()
            if features
            else 0.0
        )

        if (
            missing_fraction
            > self.maximum_missing_fraction
        ):
            warnings.append(
                "At least one feature has a high missing-value "
                f"fraction ({missing_fraction:.1%})."
            )

        return warnings

    # ------------------------------------------------------------------
    # Main build method
    # ------------------------------------------------------------------

    def build(
        self,
        symbol: str,
        timeframe: str = "1D",
        horizons: Sequence[int] = (
            1,
            3,
            5,
            10,
            20,
        ),
        raw_data: pd.DataFrame | None = None,
    ) -> ResearchDatasetResult:
        """
        Build and formally audit a research dataset.

        A failed leakage audit raises ValueError. This prevents
        contaminated datasets from silently reaching model training.
        """

        symbol = str(symbol).strip().upper()
        timeframe = str(
            timeframe
        ).strip().upper()

        if not symbol:
            raise ValueError(
                "symbol must not be empty."
            )

        if not timeframe:
            raise ValueError(
                "timeframe must not be empty."
            )

        horizons = tuple(
            sorted(
                {
                    int(horizon)
                    for horizon in horizons
                }
            )
        )

        if not horizons:
            raise ValueError(
                "At least one prediction horizon is required."
            )

        if any(
            horizon <= 0
            for horizon in horizons
        ):
            raise ValueError(
                "Prediction horizons must be positive."
            )

        # --------------------------------------------------------------
        # Stage 1: raw data
        # --------------------------------------------------------------

        data = (
            raw_data.copy()
            if raw_data is not None
            else self.data_builder(
                symbol=symbol,
                timeframe=timeframe,
            )
        )

        self._validate_basic_dataframe(
            data
        )

        quality_report = validate_ohlcv(
            data
        )

        assert_quality(
            quality_report
        )

        # --------------------------------------------------------------
        # Stage 2: feature engineering
        # --------------------------------------------------------------

        feature_result = self.feature_builder(
            data
        )

        if isinstance(
            feature_result,
            pd.DataFrame,
        ):
            featured = feature_result
        elif hasattr(
            feature_result,
            "data",
        ):
            featured = feature_result.data
        else:
            raise TypeError(
                "Feature builder must return a DataFrame "
                "or an object containing .data."
            )

        self._validate_basic_dataframe(
            featured
        )

        # --------------------------------------------------------------
        # Stage 3: target construction
        # --------------------------------------------------------------

        targets = self._build_targets(
            featured,
            horizons,
        )

        if not featured.index.equals(
            targets.index
        ):
            targets = targets.reindex(
                featured.index
            )

        combined = featured.join(
            targets,
            how="left",
            rsuffix="_target",
        )

        self._validate_basic_dataframe(
            combined
        )

        # --------------------------------------------------------------
        # Stage 4: feature/target separation
        # --------------------------------------------------------------

        feature_columns = self._identify_features(
            combined
        )

        target_columns = self._identify_targets(
            combined
        )

        if not feature_columns:
            raise ValueError(
                "No model features were identified."
            )

        if not target_columns:
            raise ValueError(
                "No prediction targets were identified."
            )

        self._validate_feature_target_separation(
            feature_columns,
            target_columns,
        )

        self._validate_numeric_features(
            combined,
            feature_columns,
        )

        self._validate_finite_features(
            combined,
            feature_columns,
        )

        # --------------------------------------------------------------
        # Stage 5: FORMAL LEAKAGE AUDIT
        # --------------------------------------------------------------

        leakage_report = audit_research_dataset(
            combined,
            feature_columns,
            target_columns,
            config=self.leakage_config,
        )

        if not leakage_report.passed:
            raise ValueError(
                "Research dataset failed the formal leakage audit. "
                "Model training is blocked."
            )

        # --------------------------------------------------------------
        # Stage 6: research warnings
        # --------------------------------------------------------------

        warnings = self._generate_warnings(
            combined,
            feature_columns,
        )

        metadata = {
            "builder": (
                "ResearchDatasetBuilder"
            ),
            "leakage_audit_passed": True,
            "quality_passed": (
                quality_report.passed
            ),
            "feature_count": len(
                feature_columns
            ),
            "target_count": len(
                target_columns
            ),
            "minimum_rows": (
                self.minimum_rows
            ),
            "research_only": True,
        }

        return ResearchDatasetResult(
            data=combined,
            feature_columns=tuple(
                feature_columns
            ),
            target_columns=tuple(
                target_columns
            ),
            symbol=symbol,
            timeframe=timeframe,
            horizons=horizons,
            quality_report=quality_report,
            leakage_report=leakage_report,
            warnings=tuple(
                warnings
            ),
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Convenience aliases
    # ------------------------------------------------------------------

    def run(
        self,
        symbol: str,
        timeframe: str = "1D",
        horizons: Sequence[int] = (
            1,
            3,
            5,
            10,
            20,
        ),
        raw_data: pd.DataFrame | None = None,
    ) -> ResearchDatasetResult:
        """
        Alias for build().
        """

        return self.build(
            symbol=symbol,
            timeframe=timeframe,
            horizons=horizons,
            raw_data=raw_data,
        )


def build_research_dataset(
    symbol: str,
    timeframe: str = "1D",
    horizons: Sequence[int] = (
        1,
        3,
        5,
        10,
        20,
    ),
    raw_data: pd.DataFrame | None = None,
    data_builder: Callable | None = None,
    feature_builder: Callable | None = None,
    target_builder: Callable | None = None,
    leakage_config: LeakageAuditConfig | None = None,
) -> ResearchDatasetResult:
    """
    Convenience function for building a research dataset.
    """

    builder = ResearchDatasetBuilder(
        data_builder=data_builder,
        feature_builder=feature_builder,
        target_builder=target_builder,
        leakage_config=leakage_config,
    )

    return builder.build(
        symbol=symbol,
        timeframe=timeframe,
        horizons=horizons,
        raw_data=raw_data,
    )


__all__ = [
    "ResearchDatasetResult",
    "ResearchDatasetBuilder",
    "build_research_dataset",
]
