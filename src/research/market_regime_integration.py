"""
AI Swing Analyser — Market & Regime Integration.

Combines stock features with broader Indian market context,
sector context, and market-regime information.

Research rules
--------------
- Never use future market observations.
- Never forward-fill future information.
- Preserve the stock timeline.
- Do not use final holdout data.
- Do not approve models.
- Do not generate live trading signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .market_context import (
    MarketContextConfig,
    MarketContextResult,
    add_relative_strength,
    build_market_context,
    market_strength_score,
)
from .market_integration import (
    MarketIntegrationConfig,
    MarketIntegrationResult,
    integrate_market_context,
)


DEFAULT_MARKET_REGIMES = (
    "BULL",
    "BEAR",
    "NEUTRAL",
    "UNKNOWN",
)

DEFAULT_VOLATILITY_REGIMES = (
    "LOW",
    "NORMAL",
    "HIGH",
    "UNKNOWN",
)


@dataclass(frozen=True)
class MarketRegimeIntegrationConfig:
    """Configuration for market and regime integration."""

    market_context: MarketContextConfig = (
        MarketContextConfig()
    )

    market_integration: MarketIntegrationConfig = (
        MarketIntegrationConfig()
    )

    minimum_market_features: int = 5

    include_trend_regime: bool = True

    include_volatility_regime: bool = True

    include_market_strength: bool = True

    include_relative_strength: bool = True

    include_sector_features: bool = True

    minimum_observations: int = 50

    def __post_init__(self) -> None:
        if not isinstance(
            self.market_context,
            MarketContextConfig,
        ):
            raise TypeError(
                "market_context must be MarketContextConfig."
            )

        if not isinstance(
            self.market_integration,
            MarketIntegrationConfig,
        ):
            raise TypeError(
                "market_integration must be "
                "MarketIntegrationConfig."
            )

        if (
            self.minimum_market_features
            < 1
        ):
            raise ValueError(
                "minimum_market_features must be positive."
            )

        if (
            self.minimum_observations
            < 1
        ):
            raise ValueError(
                "minimum_observations must be positive."
            )


@dataclass
class MarketRegimeIntegrationResult:
    """Result of market/regime feature integration."""

    data: pd.DataFrame

    stock_columns: list[str]

    market_columns: list[str]

    regime_columns: list[str]

    relative_strength_columns: list[str]

    sector_columns: list[str]

    market_strength: pd.Series | None

    market_regime: pd.Series | None

    volatility_regime: pd.Series | None

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
    def success(self) -> bool:
        return (
            not self.errors
            and not self.data.empty
        )

    @property
    def production_ready(self) -> bool:
        return False

    @property
    def final_holdout_used(self) -> bool:
        return bool(
            self.metadata.get(
                "final_holdout_used",
                False,
            )
        )

    def summary(self) -> dict[str, object]:
        return {
            "success": self.success,
            "rows": len(self.data),
            "stock_features": len(
                self.stock_columns
            ),
            "market_features": len(
                self.market_columns
            ),
            "regime_features": len(
                self.regime_columns
            ),
            "relative_strength_features": len(
                self.relative_strength_columns
            ),
            "sector_features": len(
                self.sector_columns
            ),
            "warnings": list(
                self.warnings
            ),
            "errors": list(
                self.errors
            ),
            "research_only": True,
            "production_ready": False,
            "final_holdout_used": (
                self.final_holdout_used
            ),
        }


class MarketRegimeIntegration:
    """
    Integrate broader Indian market context with stock data.

    This class is intentionally a feature/evidence layer.
    It does not train or approve a model.
    """

    def __init__(
        self,
        *,
        config: MarketRegimeIntegrationConfig
        | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else MarketRegimeIntegrationConfig()
        )

    @staticmethod
    def _validate_stock_data(
        stock_data: pd.DataFrame,
    ) -> None:
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
                "stock_data must use a DatetimeIndex."
            )

        if stock_data.index.has_duplicates:
            raise ValueError(
                "stock_data contains duplicate timestamps."
            )

        if not stock_data.index.is_monotonic_increasing:
            raise ValueError(
                "stock_data must be chronological."
            )

        required = {
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        }

        missing = (
            required
            - set(stock_data.columns)
        )

        if missing:
            raise ValueError(
                f"Missing OHLCV columns: "
                f"{sorted(missing)}"
            )

        if (
            stock_data["Close"]
            .isna()
            .any()
        ):
            raise ValueError(
                "Close contains missing values."
            )

        if (
            stock_data["Close"]
            <= 0
        ).any():
            raise ValueError(
                "Close contains non-positive prices."
            )

    @staticmethod
    def _numeric_columns(
        frame: pd.DataFrame,
        candidates: list[str],
    ) -> list[str]:
        columns = []

        for column in candidates:
            if column not in frame.columns:
                continue

            if pd.api.types.is_numeric_dtype(
                frame[column]
            ):
                columns.append(column)

        return columns

    @staticmethod
    def _add_stock_market_returns(
        data: pd.DataFrame,
        market_close: pd.Series,
    ) -> pd.DataFrame:
        """
        Add market return features using only information available
        at the current timestamp.

        The caller must provide an already time-aligned market series.
        """

        result = data.copy()

        aligned = (
            market_close
            .reindex(result.index)
        )

        for window in (
            1,
            3,
            5,
            10,
            20,
        ):
            result[
                f"Market_Return_{window}"
            ] = aligned.pct_change(
                window
            )

        return result

    @staticmethod
    def _trend_regime(
        close: pd.Series,
        fast_window: int = 50,
        slow_window: int = 200,
    ) -> pd.Series:
        fast = (
            close.rolling(
                fast_window,
                min_periods=fast_window,
            )
            .mean()
        )

        slow = (
            close.rolling(
                slow_window,
                min_periods=slow_window,
            )
            .mean()
        )

        regime = pd.Series(
            "UNKNOWN",
            index=close.index,
            dtype="object",
        )

        valid = (
            fast.notna()
            & slow.notna()
        )

        regime.loc[
            valid
            & (fast > slow)
        ] = "BULL"

        regime.loc[
            valid
            & (fast < slow)
        ] = "BEAR"

        regime.loc[
            valid
            & (fast == slow)
        ] = "NEUTRAL"

        return regime

    @staticmethod
    def _volatility_regime(
        close: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        returns = (
            close
            .pct_change()
        )

        volatility = (
            returns.rolling(
                window,
                min_periods=window,
            )
            .std()
        )

        history = (
            volatility
            .expanding(
                min_periods=window,
            )
        )

        low_threshold = (
            history.quantile(0.33)
        )

        high_threshold = (
            history.quantile(0.67)
        )

        regime = pd.Series(
            "UNKNOWN",
            index=close.index,
            dtype="object",
        )

        valid = volatility.notna()

        regime.loc[
            valid
            & (
                volatility
                <= low_threshold
            )
        ] = "LOW"

        regime.loc[
            valid
            & (
                volatility
                >= high_threshold
            )
        ] = "HIGH"

        regime.loc[
            valid
            & (
                volatility > low_threshold
            )
            & (
                volatility < high_threshold
            )
        ] = "NORMAL"

        return regime

    @staticmethod
    def _market_strength_series(
        context: MarketContextResult,
    ) -> pd.Series | None:
        """
        Build a point-in-time market-strength series.

        Only numeric trend-score columns are used.
        """

        frame = context.context.copy()

        score_columns = [
            column
            for column in frame.columns
            if (
                "Trend_Score"
                in column
                and pd.api.types.is_numeric_dtype(
                    frame[column]
                )
            )
        ]

        if not score_columns:
            return None

        score = frame[
            score_columns
        ].mean(axis=1)

        return score.rename(
            "Market_Strength_Score"
        )

    @staticmethod
    def _market_regime_from_context(
        context: MarketContextResult,
    ) -> pd.Series | None:
        regime_columns = [
            column
            for column in context.context.columns
            if (
                column.endswith(
                    "_Regime"
                )
                and "Volatility"
                not in column
            )
        ]

        if not regime_columns:
            return None

        regimes = context.context[
            regime_columns
        ]

        result = pd.Series(
            "UNKNOWN",
            index=regimes.index,
            dtype="object",
        )

        bull_votes = (
            regimes == "BULL"
        ).sum(axis=1)

        bear_votes = (
            regimes == "BEAR"
        ).sum(axis=1)

        result.loc[
            bull_votes > bear_votes
        ] = "BULL"

        result.loc[
            bear_votes > bull_votes
        ] = "BEAR"

        result.loc[
            bull_votes.eq(
                bear_votes
            )
            & bull_votes.gt(0)
        ] = "NEUTRAL"

        return result.rename(
            "Market_Regime"
        )

    def integrate(
        self,
        stock_data: pd.DataFrame,
        market_data: dict[
            str,
            pd.Series,
        ],
        *,
        sector_data: dict[
            str,
            pd.Series,
        ]
        | None = None,
    ) -> MarketRegimeIntegrationResult:
        """
        Integrate stock, market, and optional sector data.

        Parameters
        ----------
        stock_data:
            Chronological stock OHLCV data.

        market_data:
            Mapping of benchmark symbol/name to Close series.

        sector_data:
            Optional mapping of sector name to Close series.
        """

        self._validate_stock_data(
            stock_data
        )

        if not isinstance(
            market_data,
            dict,
        ):
            raise TypeError(
                "market_data must be a dictionary."
            )

        if sector_data is not None and not isinstance(
            sector_data,
            dict,
        ):
            raise TypeError(
                "sector_data must be a dictionary."
            )

        original = stock_data.copy(
            deep=True
        )

        warnings: list[str] = []
        errors: list[str] = []

        try:
            market_close = {}

            for name, series in (
                market_data.items()
            ):
                if not isinstance(
                    series,
                    pd.Series,
                ):
                    raise TypeError(
                        f"Market series '{name}' "
                        "must be pandas Series."
                    )

                if not isinstance(
                    series.index,
                    pd.DatetimeIndex,
                ):
                    raise TypeError(
                        f"Market series '{name}' "
                        "must use DatetimeIndex."
                    )

                if series.index.has_duplicates:
                    raise ValueError(
                        f"Market series '{name}' "
                        "contains duplicate timestamps."
                    )

                if not series.index.is_monotonic_increasing:
                    raise ValueError(
                        f"Market series '{name}' "
                        "must be chronological."
                    )

                numeric = pd.to_numeric(
                    series,
                    errors="coerce",
                )

                if numeric.isna().all():
                    raise ValueError(
                        f"Market series '{name}' "
                        "contains no numeric values."
                    )

                market_close[
                    str(name)
                ] = numeric.astype(float)

            if not market_close:
                raise ValueError(
                    "At least one market series is required."
                )

            market_frame = pd.DataFrame(
                market_close
            )

            # Backward as-of alignment ensures that a stock timestamp
            # only receives market information that existed at or
            # before that timestamp.
            market_frame = (
                market_frame
                .sort_index()
            )

            stock_frame = (
                original
                .sort_index()
            )

            aligned = pd.merge_asof(
                stock_frame.reset_index(),
                market_frame.reset_index(),
                on="index",
                direction="backward",
            )

            aligned = aligned.set_index(
                "index"
            )

            result = aligned

            market_columns = []

            for name in market_close:
                safe_name = (
                    str(name)
                    .replace(" ", "_")
                    .replace(".", "_")
                    .replace("-", "_")
                )

                source_column = name

                if source_column not in result.columns:
                    continue

                result[
                    f"Market_{safe_name}_Close"
                ] = result[
                    source_column
                ]

                result[
                    f"Market_{safe_name}_Return_1"
                ] = result[
                    source_column
                ].pct_change(1)

                result[
                    f"Market_{safe_name}_Return_5"
                ] = result[
                    source_column
                ].pct_change(5)

                result[
                    f"Market_{safe_name}_Return_20"
                ] = result[
                    source_column
                ].pct_change(20)

                market_columns.extend(
                    [
                        f"Market_{safe_name}_Close",
                        f"Market_{safe_name}_Return_1",
                        f"Market_{safe_name}_Return_5",
                        f"Market_{safe_name}_Return_20",
                    ]
                )

            benchmark_frame = pd.DataFrame(
                market_close
            )

            context = build_market_context(
                benchmark_frame,
                config=self.config.market_context,
            )

            context_frame = (
                context.context
                .copy()
                .sort_index()
            )

            # Context features are aligned backward. No future
            # benchmark candle can be attached to a stock row.
            result = pd.merge_asof(
                result.sort_index(),
                context_frame.reset_index(),
                on="index",
                direction="backward",
                suffixes=(
                    "",
                    "_market_context",
                ),
            ).set_index(
                "index"
            )

            context_columns = [
                column
                for column in context_frame.columns
                if column in result.columns
            ]

            numeric_context_columns = (
                self._numeric_columns(
                    result,
                    context_columns,
                )
            )

            market_columns.extend(
                numeric_context_columns
            )

            if (
                self.config.include_market_strength
            ):
                strength = (
                    self._market_strength_series(
                        context
                    )
                )

                if strength is not None:
                    strength = strength.sort_index()

                    result = pd.merge_asof(
                        result.sort_index(),
                        strength.rename(
                            "Market_Strength_Score"
                        ).reset_index(),
                        on="index",
                        direction="backward",
                    ).set_index(
                        "index"
                    )

                    market_columns.append(
                        "Market_Strength_Score"
                    )

            market_regime = None

            if (
                self.config.include_trend_regime
            ):
                market_regime = (
                    self._market_regime_from_context(
                        context
                    )
                )

                if market_regime is not None:
                    result = pd.merge_asof(
                        result.sort_index(),
                        market_regime.reset_index(),
                        on="index",
                        direction="backward",
                    ).set_index(
                        "index"
                    )

            if (
                self.config.include_trend_regime
                and market_regime is None
            ):
                close_for_regime = (
                    next(
                        iter(
                            market_close.values()
                        )
                    )
                )

                market_regime = (
                    self._trend_regime(
                        close_for_regime
                    )
                )

            if market_regime is not None:
                result = pd.merge_asof(
                    result.sort_index(),
                    market_regime.reset_index(),
                    on="index",
                    direction="backward",
                    suffixes=(
                        "",
                        "_regime",
                    ),
                ).set_index(
                    "index"
                )

            regime_columns = []

            if (
                "Market_Regime"
                in result.columns
            ):
                regime_columns.append(
                    "Market_Regime"
                )

            volatility_regime = None

            if (
                self.config.include_volatility_regime
            ):
                first_market = next(
                    iter(
                        market_close.values()
                    )
                )

                volatility_regime = (
                    self._volatility_regime(
                        first_market
                    )
                )

                result = pd.merge_asof(
                    result.sort_index(),
                    volatility_regime.rename(
                        "Market_Volatility_Regime"
                    ).reset_index(),
                    on="index",
                    direction="backward",
                ).set_index(
                    "index"
                )

                regime_columns.append(
                    "Market_Volatility_Regime"
                )

            relative_strength_columns = []

            if (
                self.config.include_relative_strength
            ):
                first_market = next(
                    iter(
                        market_close.values()
                    )
                )

                aligned_market = (
                    pd.merge_asof(
                        stock_frame[
                            ["Close"]
                        ].reset_index(),
                        first_market.rename(
                            "Market_Close"
                        ).reset_index(),
                        on="index",
                        direction="backward",
                    ).set_index(
                        "index"
                    )
                )

                stock_returns = (
                    aligned_market[
                        "Close"
                    ].pct_change(20)
                )

                market_returns = (
                    aligned_market[
                        "Market_Close"
                    ].pct_change(20)
                )

                result[
                    "Relative_Strength_20"
                ] = (
                    stock_returns
                    - market_returns
                )

                result[
                    "Relative_Strength_Ratio_20"
                ] = (
                    (
                        1.0
                        + stock_returns
                    )
                    /
                    (
                        1.0
                        + market_returns
                    )
                )

                relative_strength_columns.extend(
                    [
                        "Relative_Strength_20",
                        "Relative_Strength_Ratio_20",
                    ]
                )

            sector_columns = []

            if (
                self.config.include_sector_features
                and sector_data
            ):
                for name, series in (
                    sector_data.items()
                ):
                    if not isinstance(
                        series,
                        pd.Series,
                    ):
                        raise TypeError(
                            f"Sector series '{name}' "
                            "must be pandas Series."
                        )

                    if not isinstance(
                        series.index,
                        pd.DatetimeIndex,
                    ):
                        raise TypeError(
                            f"Sector series '{name}' "
                            "must use DatetimeIndex."
                        )

                    numeric = pd.to_numeric(
                        series,
                        errors="coerce",
                    )

                    if numeric.isna().all():
                        warnings.append(
                            f"Sector '{name}' has no usable "
                            "numeric observations."
                        )
                        continue

                    safe_name = (
                        str(name)
                        .replace(" ", "_")
                        .replace(".", "_")
                        .replace("-", "_")
                    )

                    sector_frame = pd.DataFrame(
                        {
                            f"Sector_{safe_name}_Close":
                                numeric.astype(float)
                        }
                    ).sort_index()

                    result = pd.merge_asof(
                        result.sort_index(),
                        sector_frame.reset_index(),
                        on="index",
                        direction="backward",
                    ).set_index(
                        "index"
                    )

                    close_column = (
                        f"Sector_{safe_name}_Close"
                    )

                    return_column = (
                        f"Sector_{safe_name}_Return_20"
                    )

                    result[
                        return_column
                    ] = result[
                        close_column
                    ].pct_change(20)

                    sector_columns.extend(
                        [
                            close_column,
                            return_column,
                        ]
                    )

            result = result.sort_index()

            if not result.index.equals(
                original.index
            ):
                raise RuntimeError(
                    "Market/regime integration changed "
                    "the stock timeline."
                )

            if len(
                market_columns
            ) < self.config.minimum_market_features:
                warnings.append(
                    "Fewer market features were produced "
                    "than the configured minimum."
                )

            if len(result) < (
                self.config.minimum_observations
            ):
                warnings.append(
                    "Market/regime integration contains "
                    "fewer observations than recommended."
                )

            numeric_market = (
                self._numeric_columns(
                    result,
                    market_columns
                    + relative_strength_columns
                    + sector_columns,
                )
            )

            if numeric_market:
                invalid = (
                    ~np.isfinite(
                        result[
                            numeric_market
                        ].fillna(0.0)
                        .to_numpy()
                    )
                )

                if invalid.any():
                    raise ValueError(
                        "Market feature matrix contains "
                        "non-finite values."
                    )

            stock_columns = [
                column
                for column in original.columns
                if column in result.columns
            ]

            metadata = {
                "research_only": True,
                "production_ready": False,
                "production_approved": False,
                "final_holdout_used": False,
                "future_market_data_used": False,
                "forward_fill_used": False,
                "asof_alignment": True,
                "market_context_enabled": True,
                "sector_context_enabled": bool(
                    sector_data
                ),
                "regime_features_enabled": (
                    self.config.include_trend_regime
                    or self.config.include_volatility_regime
                ),
            }

            return MarketRegimeIntegrationResult(
                data=result,
                stock_columns=stock_columns,
                market_columns=list(
                    dict.fromkeys(
                        market_columns
                    )
                ),
                regime_columns=regime_columns,
                relative_strength_columns=(
                    relative_strength_columns
                ),
                sector_columns=sector_columns,
                market_strength=(
                    result.get(
                        "Market_Strength_Score"
                    )
                ),
                market_regime=(
                    result.get(
                        "Market_Regime"
                    )
                ),
                volatility_regime=(
                    result.get(
                        "Market_Volatility_Regime"
                    )
                ),
                warnings=warnings,
                errors=errors,
                metadata=metadata,
            )

        except Exception as exc:
            errors.append(
                f"Market/regime integration failed: "
                f"{type(exc).__name__}: {exc}"
            )

            return MarketRegimeIntegrationResult(
                data=original,
                stock_columns=list(
                    original.columns
                ),
                market_columns=[],
                regime_columns=[],
                relative_strength_columns=[],
                sector_columns=[],
                market_strength=None,
                market_regime=None,
                volatility_regime=None,
                warnings=warnings,
                errors=errors,
                metadata={
                    "research_only": True,
                    "production_ready": False,
                    "production_approved": False,
                    "final_holdout_used": False,
                    "future_market_data_used": False,
                    "forward_fill_used": False,
                },
            )


def integrate_market_regime(
    stock_data: pd.DataFrame,
    market_data: dict[
        str,
        pd.Series,
    ],
    *,
    sector_data: dict[
        str,
        pd.Series,
    ]
    | None = None,
    config: MarketRegimeIntegrationConfig
    | None = None,
) -> MarketRegimeIntegrationResult:
    """Convenience API for market/regime integration."""

    pipeline = MarketRegimeIntegration(
        config=config
    )

    return pipeline.integrate(
        stock_data,
        market_data,
        sector_data=sector_data,
    )


def market_regime_integration_summary(
    result: MarketRegimeIntegrationResult,
) -> dict[str, object]:
    """Return a compact integration summary."""

    if not isinstance(
        result,
        MarketRegimeIntegrationResult,
    ):
        raise TypeError(
            "result must be MarketRegimeIntegrationResult."
        )

    return result.summary()


__all__ = [
    "DEFAULT_MARKET_REGIMES",
    "DEFAULT_VOLATILITY_REGIMES",
    "MarketRegimeIntegrationConfig",
    "MarketRegimeIntegrationResult",
    "MarketRegimeIntegration",
    "integrate_market_regime",
    "market_regime_integration_summary",
]
