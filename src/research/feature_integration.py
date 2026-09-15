"""
AI Swing Analyser — Unified Feature Integration.

Combines:
- Technical indicators
- Price-action features
- Volume features
- Regime features
- Multi-timeframe context
- Indian market context
- Stock-vs-market relative strength

Design principles:
- Feature engineering remains causal.
- No target columns are allowed.
- No future columns are allowed.
- Original OHLCV data is not modified.
- Final holdout data is never used for fitting.
- This module only constructs research features.

It does NOT:
- train models
- select models
- calibrate probabilities
- generate BUY/SELL decisions
- approve production models
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.features.engine import (
    FeatureSet,
    engineer_features,
)
from src.features.multi_timeframe import (
    build_multi_timeframe_dataset,
)
from .market_context import (
    add_relative_strength,
)
from .market_data_context import (
    MarketContextData,
)
from .market_integration import (
    MarketIntegrationConfig,
    integrate_market_context,
)


@dataclass(frozen=True)
class UnifiedFeatureConfig:
    """Configuration for unified feature construction."""

    include_market_context: bool = True
    include_relative_strength: bool = True
    include_multi_timeframe: bool = True

    benchmark: str = "NIFTY50"
    relative_strength_window: int = 20

    max_missing_fraction: float = 0.40
    reject_future_columns: bool = True

    def __post_init__(self) -> None:
        if not isinstance(
            self.benchmark,
            str,
        ) or not self.benchmark.strip():
            raise ValueError(
                "benchmark must be a non-empty string."
            )

        if (
            not isinstance(
                self.relative_strength_window,
                int,
            )
            or isinstance(
                self.relative_strength_window,
                bool,
            )
            or self.relative_strength_window < 2
        ):
            raise ValueError(
                "relative_strength_window must be >= 2."
            )

        if not (
            0.0
            <= self.max_missing_fraction
            <= 1.0
        ):
            raise ValueError(
                "max_missing_fraction must be between 0 and 1."
            )


@dataclass
class UnifiedFeatureResult:
    """Complete research feature-set result."""

    data: pd.DataFrame

    feature_names: list[str]

    base_feature_names: list[str]

    market_feature_names: list[str]

    multi_timeframe_feature_names: list[str]

    rows: int

    warnings: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, object] = field(
        default_factory=dict
    )

    @property
    def feature_count(self) -> int:
        return len(
            self.feature_names
        )

    def summary(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "feature_count": self.feature_count,
            "base_features": len(
                self.base_feature_names
            ),
            "market_features": len(
                self.market_feature_names
            ),
            "multi_timeframe_features": len(
                self.multi_timeframe_feature_names
            ),
            "warning_count": len(
                self.warnings
            ),
            "research_only": self.metadata.get(
                "research_only",
                True,
            ),
        }


def _validate_ohlcv(
    data: pd.DataFrame,
) -> None:
    """Validate the primary stock dataframe."""

    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    if data.empty:
        raise ValueError(
            "data cannot be empty."
        )

    if not isinstance(
        data.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "data must have a DatetimeIndex."
        )

    if data.index.has_duplicates:
        raise ValueError(
            "data contains duplicate timestamps."
        )

    if not data.index.is_monotonic_increasing:
        raise ValueError(
            "data must be chronologically sorted."
        )

    required = {
        "Open",
        "High",
        "Low",
        "Close",
    }

    missing = required.difference(
        data.columns
    )

    if missing:
        raise ValueError(
            "Missing required OHLCV columns: "
            f"{sorted(missing)}"
        )

    for column in required:
        values = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        if values.isna().all():
            raise ValueError(
                f"{column} contains no valid numeric values."
            )

        finite = values.dropna()

        if not np.isfinite(
            finite.to_numpy()
        ).all():
            raise ValueError(
                f"{column} contains non-finite values."
            )


def _validate_feature_names(
    feature_names: list[str],
    *,
    reject_future_columns: bool,
) -> None:
    """Reject suspicious target/future columns."""

    if len(feature_names) != len(
        set(feature_names)
    ):
        raise ValueError(
            "Feature names contain duplicates."
        )

    if not reject_future_columns:
        return

    suspicious_tokens = (
        "Future_",
        "Direction_",
        "Target_",
        "Target",
        "Label",
    )

    suspicious = [
        column
        for column in feature_names
        if any(
            token.lower()
            in column.lower()
            for token in suspicious_tokens
        )
    ]

    if suspicious:
        raise ValueError(
            "Potential target/future columns detected "
            f"in features: {suspicious}"
        )


def _build_base_features(
    stock_data: pd.DataFrame,
) -> FeatureSet:
    """Build the existing base feature stack."""

    result = engineer_features(
        stock_data.copy(
            deep=True
        )
    )

    if not isinstance(
        result,
        FeatureSet,
    ):
        raise TypeError(
            "engineer_features must return FeatureSet."
        )

    return result


def _integrate_multi_timeframe(
    stock_data: pd.DataFrame,
    timeframe_data: dict[
        str,
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    """Build leakage-safe multi-timeframe features."""

    if not timeframe_data:
        return stock_data.copy(
            deep=True
        )

    frames = {
        key: value.copy(
            deep=True
        )
        for key, value
        in timeframe_data.items()
    }

    if "4H" not in frames:
        frames["4H"] = stock_data.copy(
            deep=True
        )

    return build_multi_timeframe_dataset(
        frames
    )


def _identify_added_columns(
    original_columns: list[str],
    combined_columns: list[str],
) -> list[str]:
    """Identify columns added to the original dataframe."""

    original = set(
        original_columns
    )

    return [
        column
        for column in combined_columns
        if column not in original
    ]


def _remove_high_missing(
    data: pd.DataFrame,
    feature_names: list[str],
    threshold: float,
) -> tuple[
    pd.DataFrame,
    list[str],
    list[str],
]:
    """Remove features exceeding the missing-value threshold."""

    if not feature_names:
        return (
            data.copy(deep=True),
            [],
            [],
        )

    missing_fraction = data[
        feature_names
    ].isna().mean()

    removed = [
        column
        for column in feature_names
        if missing_fraction[column]
        > threshold
    ]

    kept = [
        column
        for column in feature_names
        if column not in removed
    ]

    return (
        data.copy(deep=True),
        kept,
        removed,
    )


def _collect_feature_names(
    base_result: FeatureSet,
    data: pd.DataFrame,
    original_columns: set[str],
    mtf_feature_names: list[str],
    market_feature_names: list[str],
    *,
    reject_future_columns: bool,
) -> list[str]:
    """
    Build the final deterministic feature-name list.

    The FeatureSet produced by the existing feature engine is
    authoritative for base features.
    """

    excluded = {
        "Symbol",
        "_source_time",
        "_base_time",
    }

    feature_names: list[str] = []

    for column in base_result.feature_names:
        if (
            column in data.columns
            and column not in excluded
            and column not in feature_names
        ):
            feature_names.append(
                column
            )

    for column in mtf_feature_names:
        if (
            column in data.columns
            and column not in original_columns
            and column not in feature_names
        ):
            feature_names.append(
                column
            )

    for column in market_feature_names:
        if (
            column in data.columns
            and column not in original_columns
            and column not in feature_names
        ):
            feature_names.append(
                column
            )

    _validate_feature_names(
        feature_names,
        reject_future_columns=(
            reject_future_columns
        ),
    )

    return feature_names


def build_unified_features(
    stock_data: pd.DataFrame,
    *,
    market_context: MarketContextData | None = None,
    timeframe_data: dict[
        str,
        pd.DataFrame,
    ]
    | None = None,
    config: UnifiedFeatureConfig | None = None,
) -> UnifiedFeatureResult:
    """
    Construct the unified research feature dataframe.
    """

    _validate_ohlcv(
        stock_data
    )

    cfg = (
        config
        if config is not None
        else UnifiedFeatureConfig()
    )

    original = stock_data.copy(
        deep=True
    )

    original_columns = set(
        original.columns
    )

    warnings: list[str] = []

    # ---------------------------------------------------------------
    # 1. Base features
    # ---------------------------------------------------------------

    base_result = _build_base_features(
        original
    )

    combined = base_result.data.copy(
        deep=True
    )

    base_feature_names = list(
        base_result.feature_names
    )

    # ---------------------------------------------------------------
    # 2. Multi-timeframe features
    # ---------------------------------------------------------------

    mtf_feature_names: list[str] = []

    if (
        cfg.include_multi_timeframe
        and timeframe_data is not None
    ):
        mtf_data = _integrate_multi_timeframe(
            original,
            timeframe_data,
        )

        mtf_added = _identify_added_columns(
            list(original.columns),
            list(mtf_data.columns),
        )

        for column in mtf_added:
            if column not in combined.columns:
                combined[column] = (
                    mtf_data[column]
                    .reindex(
                        combined.index
                    )
                )

        mtf_feature_names = [
            column
            for column in mtf_added
            if column in combined.columns
            and column not in base_feature_names
        ]

    # ---------------------------------------------------------------
    # 3. Market context
    # ---------------------------------------------------------------

    market_feature_names: list[str] = []

    if (
        cfg.include_market_context
        and market_context is not None
    ):
        market_result = integrate_market_context(
            combined,
            market_context,
            config=MarketIntegrationConfig(
                benchmark=cfg.benchmark,
                relative_strength_window=(
                    cfg.relative_strength_window
                ),
                allow_market_missing=True,
            ),
        )

        combined = market_result.data.copy(
            deep=True
        )

        market_feature_names = list(
            market_result.market_columns
        )

        if cfg.include_relative_strength:
            market_feature_names.extend(
                market_result.relative_strength_columns
            )
        else:
            combined = combined.drop(
                columns=(
                    market_result
                    .relative_strength_columns
                ),
                errors="ignore",
            )

        warnings.extend(
            market_result.warnings
        )

    elif cfg.include_market_context:
        warnings.append(
            "Market context was requested but "
            "no market_context was supplied."
        )

    # ---------------------------------------------------------------
    # 4. Final feature names
    # ---------------------------------------------------------------

    feature_names = _collect_feature_names(
        base_result,
        combined,
        original_columns,
        mtf_feature_names,
        market_feature_names,
        reject_future_columns=(
            cfg.reject_future_columns
        ),
    )

    if not feature_names:
        raise ValueError(
            "No usable features were generated."
        )

    # ---------------------------------------------------------------
    # 5. Missing feature filtering
    # ---------------------------------------------------------------

    (
        combined,
        feature_names,
        removed_features,
    ) = _remove_high_missing(
        combined,
        feature_names,
        cfg.max_missing_fraction,
    )

    if removed_features:
        warnings.append(
            "Removed high-missing features: "
            + ", ".join(
                removed_features
            )
        )

    if not feature_names:
        raise ValueError(
            "No features remain after missing-value filtering."
        )

    # ---------------------------------------------------------------
    # 6. Numeric validation
    # ---------------------------------------------------------------

    numeric_features = []

    for column in feature_names:
        if not pd.api.types.is_numeric_dtype(
            combined[column]
        ):
            raise TypeError(
                "Non-numeric model feature detected: "
                f"{column}"
            )

        numeric_features.append(
            column
        )

    numeric_values = combined[
        numeric_features
    ]

    finite_values = np.isfinite(
        numeric_values.fillna(0.0).to_numpy()
    )

    if not finite_values.all():
        raise ValueError(
            "Feature dataframe contains non-finite values."
        )

    # ---------------------------------------------------------------
    # 7. Timeline integrity
    # ---------------------------------------------------------------

    if not combined.index.equals(
        original.index
    ):
        raise RuntimeError(
            "Feature integration changed the stock timeline."
        )

    # ---------------------------------------------------------------
    # 8. Keep feature-category lists consistent
    # ---------------------------------------------------------------

    final_base_features = [
        column
        for column in base_feature_names
        if column in feature_names
    ]

    final_mtf_features = [
        column
        for column in mtf_feature_names
        if column in feature_names
    ]

    final_market_features = [
        column
        for column in market_feature_names
        if column in feature_names
    ]

    # ---------------------------------------------------------------
    # 9. Final metadata
    # ---------------------------------------------------------------

    metadata = {
        "research_only": True,
        "production_approved": False,
        "future_values_used": False,
        "target_columns_included": False,
        "final_holdout_used": False,
        "base_feature_count": len(
            final_base_features
        ),
        "multi_timeframe_feature_count": len(
            final_mtf_features
        ),
        "market_feature_count": len(
            final_market_features
        ),
        "removed_high_missing_features": (
            removed_features
        ),
    }

    return UnifiedFeatureResult(
        data=combined,
        feature_names=list(
            feature_names
        ),
        base_feature_names=(
            final_base_features
        ),
        market_feature_names=(
            final_market_features
        ),
        multi_timeframe_feature_names=(
            final_mtf_features
        ),
        rows=len(combined),
        warnings=warnings,
        metadata=metadata,
    )


def feature_matrix(
    result: UnifiedFeatureResult,
) -> pd.DataFrame:
    """Return the model-ready feature matrix."""

    if not isinstance(
        result,
        UnifiedFeatureResult,
    ):
        raise TypeError(
            "result must be UnifiedFeatureResult."
        )

    return result.data[
        result.feature_names
    ].copy(
        deep=True
    )


def unified_feature_names(
    result: UnifiedFeatureResult,
) -> list[str]:
    """Return a copy of the final feature list."""

    if not isinstance(
        result,
        UnifiedFeatureResult,
    ):
        raise TypeError(
            "result must be UnifiedFeatureResult."
        )

    return list(
        result.feature_names
    )


def unified_feature_summary(
    result: UnifiedFeatureResult,
) -> dict[str, object]:
    """Return a compact feature summary."""

    if not isinstance(
        result,
        UnifiedFeatureResult,
    ):
        raise TypeError(
            "result must be UnifiedFeatureResult."
        )

    return result.summary()


__all__ = [
    "UnifiedFeatureConfig",
    "UnifiedFeatureResult",
    "build_unified_features",
    "feature_matrix",
    "unified_feature_names",
    "unified_feature_summary",
]
