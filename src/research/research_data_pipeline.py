"""
AI Swing Analyser — Research Data Pipeline.

Pipeline boundary:

Raw OHLCV
    ↓
Data preparation / quality
    ↓
Feature construction
    ↓
Target construction
    ↓
Leakage audit
    ↓
Research dataset

This module does NOT:
- train models
- select models
- calibrate models
- evaluate the final holdout
- approve production models
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .data_preparation import (
    DataPreparationConfig,
    PreparedMarketData,
    prepare_market_data,
)
from .dataset_builder import (
    ResearchDatasetBuilder,
    ResearchDatasetResult,
)
from .config import ResearchPipelineConfig


@dataclass(frozen=True)
class ResearchDataPipelineConfig:
    """Configuration for the research data pipeline."""

    preparation: DataPreparationConfig = (
        DataPreparationConfig()
    )

    require_research_dataset: bool = True

    def __post_init__(self) -> None:
        if not isinstance(
            self.preparation,
            DataPreparationConfig,
        ):
            raise TypeError(
                "preparation must be DataPreparationConfig."
            )


@dataclass
class ResearchDataPipelineResult:
    """Output of the complete data-to-research-dataset stage."""

    prepared_data: PreparedMarketData

    dataset: ResearchDatasetResult | None

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
    def production_ready(self) -> bool:
        """
        Data preparation never grants production readiness.
        """

        return False

    @property
    def research_ready(self) -> bool:
        return bool(
            self.success
            and self.dataset is not None
        )

    def summary(self) -> dict[str, object]:
        return {
            "success": self.success,
            "research_ready": self.research_ready,
            "production_ready": False,
            "prepared_rows": (
                self.prepared_data.rows
            ),
            "dataset_rows": (
                len(self.dataset.data)
                if self.dataset is not None
                else 0
            ),
            "warning_count": len(
                self.warnings
            ),
            "error_count": len(
                self.errors
            ),
            "research_only": True,
        }


class ResearchDataPipeline:
    """
    Controlled raw-data → research-dataset pipeline.
    """

    def __init__(
        self,
        *,
        config: ResearchDataPipelineConfig
        | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else ResearchDataPipelineConfig()
        )

    def prepare(
        self,
        data: pd.DataFrame,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
    ) -> PreparedMarketData:
        """
        Run only the data-quality/preparation stage.
        """

        return prepare_market_data(
            data,
            symbol=symbol,
            timeframe=timeframe,
            config=self.config.preparation,
        )

    def build_dataset(
        self,
        prepared: PreparedMarketData,
        *,
        symbol: str,
        timeframe: str,
        horizons: list[int] | tuple[int, ...],
    ) -> ResearchDatasetResult:
        """
        Build the formal research dataset from prepared data.
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
                "Cannot build research dataset from "
                "failed data-quality result."
            )

        if not symbol:
            raise ValueError(
                "symbol must be provided."
            )

        if not timeframe:
            raise ValueError(
                "timeframe must be provided."
            )

        if not horizons:
            raise ValueError(
                "At least one horizon is required."
            )

        builder = ResearchDatasetBuilder()

        return builder.build(
            prepared.data.copy(
                deep=True
            ),
            symbol=symbol,
            timeframe=timeframe,
            horizons=horizons,
        )

    def run(
        self,
        data: pd.DataFrame,
        *,
        symbol: str,
        timeframe: str,
        horizons: list[int] | tuple[int, ...],
    ) -> ResearchDataPipelineResult:
        """
        Run preparation followed by research dataset construction.
        """

        warnings: list[str] = []
        errors: list[str] = []

        prepared = self.prepare(
            data,
            symbol=symbol,
            timeframe=timeframe,
        )

        warnings.extend(
            prepared.warnings
        )

        dataset: ResearchDatasetResult | None = None

        try:
            dataset = self.build_dataset(
                prepared,
                symbol=symbol,
                timeframe=timeframe,
                horizons=horizons,
            )
        except Exception as exc:
            errors.append(
                str(exc)
            )

        success = (
            prepared.passed
            and dataset is not None
            and not errors
        )

        metadata = {
            "research_only": True,
            "production_ready": False,
            "production_approved": False,
            "future_values_used": False,
            "final_holdout_used": False,
            "model_fitted": False,
            "model_selected": False,
            "calibration_fitted": False,
            "approval_granted": False,
            "symbol": symbol,
            "timeframe": timeframe,
            "horizons": list(
                horizons
            ),
        }

        return ResearchDataPipelineResult(
            prepared_data=prepared,
            dataset=dataset,
            success=success,
            warnings=warnings,
            errors=errors,
            metadata=metadata,
        )


def run_research_data_pipeline(
    data: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    horizons: list[int] | tuple[int, ...],
    config: ResearchDataPipelineConfig
    | None = None,
) -> ResearchDataPipelineResult:
    """
    Convenience function for the complete data pipeline.
    """

    pipeline = ResearchDataPipeline(
        config=config
    )

    return pipeline.run(
        data,
        symbol=symbol,
        timeframe=timeframe,
        horizons=horizons,
    )


def research_data_pipeline_summary(
    result: ResearchDataPipelineResult,
) -> dict[str, object]:
    """Return a compact pipeline summary."""

    if not isinstance(
        result,
        ResearchDataPipelineResult,
    ):
        raise TypeError(
            "result must be ResearchDataPipelineResult."
        )

    return result.summary()


__all__ = [
    "ResearchDataPipelineConfig",
    "ResearchDataPipelineResult",
    "ResearchDataPipeline",
    "run_research_data_pipeline",
    "research_data_pipeline_summary",
]
