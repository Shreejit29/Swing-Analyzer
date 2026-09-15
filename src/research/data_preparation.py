"""
AI Swing Analyser — Research Data Preparation.

Purpose
-------
Prepare validated chronological market data for research.

This layer intentionally does NOT:
- engineer predictive features
- create future targets
- train models
- select models
- calibrate probabilities
- approve models
- generate trading signals

It is the boundary between market-data ingestion and research.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.quality import (
    QualityReport,
    clean_ohlcv,
    validate_ohlcv,
)


@dataclass(frozen=True)
class DataPreparationConfig:
    """Configuration for research data preparation."""

    minimum_rows: int = 250

    require_volume: bool = True

    remove_zero_volume: bool = False

    drop_invalid_rows: bool = True

    reject_negative_prices: bool = True

    reject_high_missing_columns: bool = True

    max_missing_fraction: float = 0.40

    def __post_init__(self) -> None:
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

        if not (
            0.0
            <= self.max_missing_fraction
            <= 1.0
        ):
            raise ValueError(
                "max_missing_fraction must be between 0 and 1."
            )


@dataclass
class PreparedMarketData:
    """Validated market data ready for research."""

    data: pd.DataFrame

    symbol: str | None

    timeframe: str | None

    rows: int

    start: pd.Timestamp

    end: pd.Timestamp

    quality_report: QualityReport

    warnings: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, object] = field(
        default_factory=dict
    )

    @property
    def passed(self) -> bool:
        return bool(
            self.quality_report.passed
        )

    def summary(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "rows": self.rows,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "quality_passed": self.passed,
            "warning_count": len(
                self.warnings
            ),
            "research_only": self.metadata.get(
                "research_only",
                True,
            ),
        }


def _validate_dataframe(
    data: pd.DataFrame,
) -> None:
    """Validate basic dataframe structure."""

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
            "Missing required columns: "
            f"{sorted(missing)}"
        )


def _validate_prices(
    data: pd.DataFrame,
    *,
    reject_negative: bool,
) -> None:
    """Validate price values and OHLC relationships."""

    price_columns = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    for column in price_columns:
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

        if reject_negative and (
            finite <= 0
        ).any():
            raise ValueError(
                f"{column} contains non-positive prices."
            )

    high = pd.to_numeric(
        data["High"],
        errors="coerce",
    )

    low = pd.to_numeric(
        data["Low"],
        errors="coerce",
    )

    if (
        high < low
    ).any():
        raise ValueError(
            "Found rows where High < Low."
        )

    open_price = pd.to_numeric(
        data["Open"],
        errors="coerce",
    )

    close = pd.to_numeric(
        data["Close"],
        errors="coerce",
    )

    invalid_high = (
        high
        < pd.concat(
            [
                open_price,
                close,
            ],
            axis=1,
        ).max(axis=1)
    )

    invalid_low = (
        low
        > pd.concat(
            [
                open_price,
                close,
            ],
            axis=1,
        ).min(axis=1)
    )

    if invalid_high.any():
        raise ValueError(
            "Found rows where High is below Open or Close."
        )

    if invalid_low.any():
        raise ValueError(
            "Found rows where Low is above Open or Close."
        )


def _validate_volume(
    data: pd.DataFrame,
    *,
    require_volume: bool,
    remove_zero_volume: bool,
) -> pd.DataFrame:
    """Validate and optionally remove invalid volume rows."""

    result = data.copy(
        deep=True
    )

    if "Volume" not in result.columns:
        if require_volume:
            raise ValueError(
                "Volume column is required."
            )

        return result

    volume = pd.to_numeric(
        result["Volume"],
        errors="coerce",
    )

    if volume.isna().all():
        raise ValueError(
            "Volume contains no valid numeric values."
        )

    if (
        volume.dropna()
        < 0
    ).any():
        raise ValueError(
            "Volume contains negative values."
        )

    result["Volume"] = volume

    if remove_zero_volume:
        result = result.loc[
            result["Volume"] > 0
        ].copy()

    return result


def _check_missing_columns(
    data: pd.DataFrame,
    *,
    threshold: float,
) -> list[str]:
    """Return columns exceeding the missing-value threshold."""

    missing_fraction = (
        data.isna().mean()
    )

    return [
        column
        for column in data.columns
        if missing_fraction[column]
        > threshold
    ]


def prepare_market_data(
    data: pd.DataFrame,
    *,
    symbol: str | None = None,
    timeframe: str | None = None,
    config: DataPreparationConfig | None = None,
) -> PreparedMarketData:
    """
    Prepare market data for downstream research.

    The returned dataframe contains only historical information
    available at each timestamp.
    """

    cfg = (
        config
        if config is not None
        else DataPreparationConfig()
    )

    _validate_dataframe(
        data
    )

    prepared = data.copy(
        deep=True
    )

    # ---------------------------------------------------------------
    # Normalize column values
    # ---------------------------------------------------------------

    for column in [
        "Open",
        "High",
        "Low",
        "Close",
    ]:
        prepared[column] = pd.to_numeric(
            prepared[column],
            errors="coerce",
        )

    prepared = (
        prepared.sort_index()
        .loc[
            ~prepared.index.duplicated(
                keep="last"
            )
        ]
        .copy()
    )

    # ---------------------------------------------------------------
    # Optional cleaning
    # ---------------------------------------------------------------

    if cfg.drop_invalid_rows:
        prepared = clean_ohlcv(
            prepared
        )

    if prepared.empty:
        raise ValueError(
            "No valid market-data rows remain after cleaning."
        )

    # ---------------------------------------------------------------
    # Volume
    # ---------------------------------------------------------------

    prepared = _validate_volume(
        prepared,
        require_volume=cfg.require_volume,
        remove_zero_volume=(
            cfg.remove_zero_volume
        ),
    )

    if prepared.empty:
        raise ValueError(
            "No rows remain after volume filtering."
        )

    # ---------------------------------------------------------------
    # Price integrity
    # ---------------------------------------------------------------

    _validate_prices(
        prepared,
        reject_negative=(
            cfg.reject_negative_prices
        ),
    )

    # ---------------------------------------------------------------
    # Missing columns
    # ---------------------------------------------------------------

    high_missing = _check_missing_columns(
        prepared,
        threshold=cfg.max_missing_fraction,
    )

    warnings: list[str] = []

    if high_missing:
        message = (
            "Columns exceed the configured missing-value "
            "threshold: "
            + ", ".join(high_missing)
        )

        if cfg.reject_high_missing_columns:
            raise ValueError(
                message
            )

        warnings.append(
            message
        )

    # ---------------------------------------------------------------
    # Final chronological checks
    # ---------------------------------------------------------------

    if not prepared.index.is_monotonic_increasing:
        raise RuntimeError(
            "Prepared data is not chronologically ordered."
        )

    if prepared.index.has_duplicates:
        raise RuntimeError(
            "Prepared data contains duplicate timestamps."
        )

    if len(prepared) < cfg.minimum_rows:
        warnings.append(
            f"Only {len(prepared)} rows are available; "
            f"minimum recommended rows are "
            f"{cfg.minimum_rows}."
        )

    # ---------------------------------------------------------------
    # Formal quality report
    # ---------------------------------------------------------------

    quality_report = validate_ohlcv(
        prepared
    )

    if not quality_report.passed:
        raise ValueError(
            "Prepared market data failed OHLCV quality validation: "
            f"{quality_report.errors}"
        )

    # ---------------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------------

    metadata = {
        "research_only": True,
        "production_approved": False,
        "future_values_used": False,
        "targets_created": False,
        "features_created": False,
        "model_fitted": False,
        "final_holdout_used": False,
        "minimum_rows": cfg.minimum_rows,
    }

    return PreparedMarketData(
        data=prepared,
        symbol=symbol,
        timeframe=timeframe,
        rows=len(prepared),
        start=pd.Timestamp(
            prepared.index[0]
        ),
        end=pd.Timestamp(
            prepared.index[-1]
        ),
        quality_report=quality_report,
        warnings=warnings,
        metadata=metadata,
    )


def preparation_summary(
    result: PreparedMarketData,
) -> dict[str, object]:
    """Return a compact preparation summary."""

    if not isinstance(
        result,
        PreparedMarketData,
    ):
        raise TypeError(
            "result must be PreparedMarketData."
        )

    return result.summary()


__all__ = [
    "DataPreparationConfig",
    "PreparedMarketData",
    "prepare_market_data",
    "preparation_summary",
]
