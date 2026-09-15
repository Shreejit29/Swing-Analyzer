"""
AI Swing Analyser — Stock + Market Context Integration.

Combines stock OHLCV data with benchmark market-context features.

Design goals:
- Keep stock data as the primary index.
- Align market data using backward/as-of alignment.
- Never forward-fill future information.
- Add relative-strength features.
- Preserve the original stock dataframe.
- Remain research-only.

This module does NOT:
- train models
- optimize thresholds
- generate live BUY/SELL decisions
- approve production models
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .market_data_context import (
    MarketContextData,
    MarketContextDataLoader,
)


@dataclass(frozen=True)
class MarketIntegrationConfig:
    """Configuration for stock/market feature integration."""

    benchmark: str = "NIFTY50"

    relative_strength_window: int = 20

    allow_market_missing: bool = False

    max_market_missing_fraction: float = 0.40

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
                "relative_strength_window must be an integer >= 2."
            )

        if not (
            0.0
            <= self.max_market_missing_fraction
            <= 1.0
        ):
            raise ValueError(
                "max_market_missing_fraction must be between 0 and 1."
            )


@dataclass
class MarketIntegrationResult:
    """Result of integrating stock and market information."""

    data: pd.DataFrame

    stock_columns: list[str]

    market_columns: list[str]

    relative_strength_columns: list[str]

    rows: int

    market_missing_fraction: float

    warnings: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, object] = field(
        default_factory=dict
    )

    @property
    def feature_columns(self) -> list[str]:
        """Return newly added market-related columns."""

        return [
            *self.market_columns,
            *self.relative_strength_columns,
        ]

    def summary(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "stock_columns": len(
                self.stock_columns
            ),
            "market_columns": len(
                self.market_columns
            ),
            "relative_strength_columns": len(
                self.relative_strength_columns
            ),
            "market_missing_fraction": (
                self.market_missing_fraction
            ),
            "warning_count": len(
                self.warnings
            ),
            "research_only": self.metadata.get(
                "research_only",
                True,
            ),
        }


def _validate_stock_data(
    stock_data: pd.DataFrame,
) -> None:
    """Validate the stock dataframe."""

    if not isinstance(
        stock_data,
        pd.DataFrame,
    ):
        raise TypeError(
            "stock_data must be a pandas DataFrame."
        )

    if stock_data.empty:
        raise ValueError(
            "stock_data cannot be empty."
        )

    if not isinstance(
        stock_data.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "stock_data must have a DatetimeIndex."
        )

    if stock_data.index.has_duplicates:
        raise ValueError(
            "stock_data index contains duplicate timestamps."
        )

    if not stock_data.index.is_monotonic_increasing:
        raise ValueError(
            "stock_data index must be sorted chronologically."
        )

    required = {
        "Open",
        "High",
        "Low",
        "Close",
    }

    missing = required.difference(
        stock_data.columns
    )

    if missing:
        raise ValueError(
            "stock_data is missing required "
            f"columns: {sorted(missing)}"
        )

    close = pd.to_numeric(
        stock_data["Close"],
        errors="coerce",
    )

    if close.isna().all():
        raise ValueError(
            "stock_data contains no valid Close prices."
        )

    if not np.isfinite(
        close.dropna().to_numpy()
    ).all():
        raise ValueError(
            "stock_data Close contains non-finite values."
        )


def _normalise_index(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return a sorted, unique DatetimeIndex copy."""

    result = frame.copy()

    index = pd.to_datetime(
        result.index,
        errors="coerce",
    )

    if index.isna().any():
        raise ValueError(
            "Data contains invalid timestamps."
        )

    if getattr(
        index,
        "tz",
        None,
    ) is not None:
        index = index.tz_convert(
            "Asia/Kolkata"
        ).tz_localize(None)

    result.index = index

    result = (
        result.sort_index()
        .loc[
            ~result.index.duplicated(
                keep="last"
            )
        ]
    )

    return result


def _asof_align(
    stock_data: pd.DataFrame,
    market_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align market observations to stock timestamps.

    Only observations at or before the stock timestamp are used.
    No future market observation can be attached to a stock row.
    """

    stock = _normalise_index(
        stock_data
    )

    market = _normalise_index(
        market_data
    )

    stock_reset = stock.reset_index(
        names="_stock_timestamp"
    )

    market_reset = market.reset_index(
        names="_market_timestamp"
    )

    aligned = pd.merge_asof(
        stock_reset.sort_values(
            "_stock_timestamp"
        ),
        market_reset.sort_values(
            "_market_timestamp"
        ),
        left_on="_stock_timestamp",
        right_on="_market_timestamp",
        direction="backward",
        allow_exact_matches=True,
        suffixes=(
            "",
            "_market",
        ),
    )

    aligned = aligned.set_index(
        "_stock_timestamp"
    )

    aligned.index.name = stock.index.name

    if "_market_timestamp" in aligned.columns:
        aligned = aligned.drop(
            columns=[
                "_market_timestamp"
            ]
        )

    return aligned


def _market_feature_columns(
    stock_columns: list[str],
    integrated_columns: list[str],
) -> list[str]:
    """
    Identify columns added by market-context integration.

    Original stock columns are excluded.
    """

    stock_set = set(
        stock_columns
    )

    return [
        column
        for column in integrated_columns
        if column not in stock_set
    ]


def _market_missing_fraction(
    data: pd.DataFrame,
    market_columns: list[str],
) -> float:
    """Calculate missingness across market features."""

    if not market_columns:
        return 1.0

    values = data[
        market_columns
    ]

    if values.empty:
        return 1.0

    return float(
        values.isna().mean().mean()
    )


def integrate_market_context(
    stock_data: pd.DataFrame,
    market_context: MarketContextData,
    *,
    config: MarketIntegrationConfig | None = None,
) -> MarketIntegrationResult:
    """
    Integrate benchmark market features into stock data.
    """

    _validate_stock_data(
        stock_data
    )

    if not isinstance(
        market_context,
        MarketContextData,
    ):
        raise TypeError(
            "market_context must be MarketContextData."
        )

    cfg = (
        config
        if config is not None
        else MarketIntegrationConfig()
    )

    if cfg.benchmark not in (
        market_context.raw_data
    ):
        raise ValueError(
            f"Benchmark {cfg.benchmark} "
            "is not available in market context."
        )

    original = stock_data.copy(
        deep=True
    )

    stock_index = original.index

    context_frame = (
        market_context.context.data
        .copy(deep=True)
    )

    if context_frame.empty:
        raise ValueError(
            "Market context contains no data."
        )

    aligned = _asof_align(
        original,
        context_frame,
    )

    stock_columns = list(
        original.columns
    )

    market_columns = (
        _market_feature_columns(
            stock_columns,
            list(aligned.columns),
        )
    )

    benchmark_close = pd.Series(
        market_context.raw_data[
            cfg.benchmark
        ]["Close"],
        copy=True,
    )

    relative_strength = (
        MarketContextDataLoader(
            provider=None,
            context_config=(
                market_context.context
                and getattr(
                    market_context,
                    "config",
                    None,
                )
            ),
        )
        if False
        else None
    )

    stock_close = pd.to_numeric(
        original["Close"],
        errors="coerce",
    )

    # Calculate relative strength directly through the existing
    # market-context helper. This keeps the calculation consistent
    # with the rest of the research layer.
    from .market_context import (
        add_relative_strength,
    )

    relative_strength = add_relative_strength(
        stock_close,
        benchmark_close,
        window=cfg.relative_strength_window,
        prefix=(
            f"{cfg.benchmark}"
            "_Relative_Strength"
        ),
    )

    relative_strength = (
        relative_strength.reindex(
            stock_index
        )
    )

    result = aligned.copy()

    for column in relative_strength.columns:
        result[column] = relative_strength[
            column
        ]

    # Restore exact original stock index.
    result = result.reindex(
        stock_index
    )

    relative_strength_columns = [
        column
        for column in relative_strength.columns
        if column not in stock_columns
    ]

    # Only columns originating from the benchmark context
    # are considered market columns.
    market_columns = [
        column
        for column in market_columns
        if column not in relative_strength_columns
    ]

    missing_fraction = (
        _market_missing_fraction(
            result,
            market_columns,
        )
    )

    warnings = list(
        market_context.warnings
    )

    if (
        missing_fraction
        > cfg.max_market_missing_fraction
    ):
        message = (
            "Market-context missingness is "
            f"{missing_fraction:.2%}, exceeding the "
            f"configured limit of "
            f"{cfg.max_market_missing_fraction:.2%}."
        )

        if cfg.allow_market_missing:
            warnings.append(
                message
            )
        else:
            raise ValueError(
                message
            )

    if missing_fraction > 0:
        warnings.append(
            "Some market-context features are "
            "unavailable at stock timestamps."
        )

    # Explicitly guarantee that no rows were added or removed.
    if not result.index.equals(
        stock_index
    ):
        raise RuntimeError(
            "Market integration changed the stock index."
        )

    metadata = {
        "research_only": True,
        "production_approved": False,
        "future_values_used": False,
        "alignment": "backward_asof",
        "forward_fill_used": False,
        "benchmark": cfg.benchmark,
        "relative_strength_window": (
            cfg.relative_strength_window
        ),
        "market_missing_fraction": (
            missing_fraction
        ),
    }

    return MarketIntegrationResult(
        data=result,
        stock_columns=stock_columns,
        market_columns=market_columns,
        relative_strength_columns=(
            relative_strength_columns
        ),
        rows=len(result),
        market_missing_fraction=(
            missing_fraction
        ),
        warnings=warnings,
        metadata=metadata,
    )


def add_market_context_to_stock(
    stock_data: pd.DataFrame,
    market_context: MarketContextData,
    *,
    benchmark: str = "NIFTY50",
    relative_strength_window: int = 20,
    allow_market_missing: bool = False,
) -> pd.DataFrame:
    """
    Convenience function returning only the integrated dataframe.
    """

    config = MarketIntegrationConfig(
        benchmark=benchmark,
        relative_strength_window=(
            relative_strength_window
        ),
        allow_market_missing=(
            allow_market_missing
        ),
    )

    result = integrate_market_context(
        stock_data,
        market_context,
        config=config,
    )

    return result.data.copy(
        deep=True
    )


def market_integration_summary(
    result: MarketIntegrationResult,
) -> dict[str, object]:
    """Return a compact integration summary."""

    if not isinstance(
        result,
        MarketIntegrationResult,
    ):
        raise TypeError(
            "result must be MarketIntegrationResult."
        )

    return result.summary()


__all__ = [
    "MarketIntegrationConfig",
    "MarketIntegrationResult",
    "integrate_market_context",
    "add_market_context_to_stock",
    "market_integration_summary",
]
