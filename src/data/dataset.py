"""
Research dataset builder.

This module combines:

    Stock data
        +
    Market context
        +
    Sector context
        +
    Historical targets

The resulting dataset is the only dataset that should be passed
to the feature-selection, validation, and machine-learning layers.

IMPORTANT:
This module must never use future information when constructing
features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .market import build_market_features
from .sector import build_sector_features


@dataclass
class ResearchDataset:
    """
    Container for a model-ready research dataset.
    """

    symbol: str
    data: pd.DataFrame
    feature_columns: list[str]
    target_columns: list[str]


def _ensure_datetime_index(
    data: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    """
    Ensure a dataframe has a clean chronological DatetimeIndex.
    """

    if data is None or data.empty:
        raise ValueError(
            f"{name} cannot be empty."
        )

    result = data.copy()

    result.index = pd.to_datetime(
        result.index
    )

    if result.index.tz is not None:
        result.index = (
            result.index
            .tz_convert("Asia/Kolkata")
            .tz_localize(None)
        )

    result = result.sort_index()

    result = result[
        ~result.index.duplicated(
            keep="first"
        )
    ]

    return result


def add_stock_returns(
    data: pd.DataFrame,
    periods: Iterable[int] = (
        1,
        3,
        5,
        10,
        20,
    ),
) -> pd.DataFrame:
    """
    Add historical stock returns.

    These features use only current and past prices.
    """

    result = data.copy()

    for period in periods:
        result[
            f"Stock_Return_{period}"
        ] = (
            result["Close"]
            .pct_change(period)
        )

    return result


def add_volatility_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add historical volatility features.

    No future prices are used.
    """

    result = data.copy()

    daily_return = (
        result["Close"]
        .pct_change()
    )

    result["Volatility_5"] = (
        daily_return
        .rolling(5)
        .std()
    )

    result["Volatility_10"] = (
        daily_return
        .rolling(10)
        .std()
    )

    result["Volatility_20"] = (
        daily_return
        .rolling(20)
        .std()
    )

    result["Volatility_60"] = (
        daily_return
        .rolling(60)
        .std()
    )

    return result


def add_price_structure_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add historical price-structure features.

    These describe the stock relative to its own recent history.
    """

    result = data.copy()

    result["Distance_From_20D_High"] = (
        result["Close"]
        / result["High"]
        .rolling(20)
        .max()
        - 1
    )

    result["Distance_From_20D_Low"] = (
        result["Close"]
        / result["Low"]
        .rolling(20)
        .min()
        - 1
    )

    result["Range_Percent"] = (
        result["High"]
        - result["Low"]
    ) / result["Close"]

    result["Close_Position_In_Day"] = (
        result["Close"]
        - result["Low"]
    ) / (
        result["High"]
        - result["Low"]
    ).replace(0, np.nan)

    return result


def add_future_targets(
    data: pd.DataFrame,
    horizons: Iterable[int] = (
        1,
        3,
        5,
        10,
        20,
    ),
    neutral_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Create future prediction targets.

    Targets are deliberately stored separately from explanatory
    features.

    For each horizon H:

        Future_Return_H
        Direction_H
        Future_High_Return_H
        Future_Low_Return_H

    Direction:

        1 = future return above threshold
        0 = otherwise

    IMPORTANT:
    These columns contain future information and must NEVER be used
    as model features.
    """

    result = data.copy()

    for horizon in horizons:

        future_close = (
            result["Close"]
            .shift(-horizon)
        )

        future_return = (
            future_close
            / result["Close"]
            - 1
        )

        result[
            f"Future_Return_{horizon}"
        ] = future_return

        result[
            f"Direction_{horizon}"
        ] = (
            future_return
            > neutral_threshold
        ).astype("Int64")

        # Highest future high reached during the horizon.
        future_high = pd.concat(
            [
                result["High"].shift(-i)
                for i in range(
                    1,
                    horizon + 1,
                )
            ],
            axis=1,
        ).max(axis=1)

        result[
            f"Future_High_Return_{horizon}"
        ] = (
            future_high
            / result["Close"]
            - 1
        )

        # Lowest future low reached during the horizon.
        future_low = pd.concat(
            [
                result["Low"].shift(-i)
                for i in range(
                    1,
                    horizon + 1,
                )
            ],
            axis=1,
        ).min(axis=1)

        result[
            f"Future_Low_Return_{horizon}"
        ] = (
            future_low
            / result["Close"]
            - 1
        )

    return result


def identify_feature_columns(
    data: pd.DataFrame,
) -> list[str]:
    """
    Automatically identify explanatory features.

    Any column containing future information is excluded.

    This is an additional safety layer. Future columns are excluded
    by name rather than relying on the caller to remember them.
    """

    future_prefixes = (
        "Future_",
        "Direction_",
    )

    excluded = {
        "Target",
        "Symbol",
    }

    features = []

    for column in data.columns:

        if column in excluded:
            continue

        if any(
            column.startswith(prefix)
            for prefix in future_prefixes
        ):
            continue

        # Raw benchmark closing prices are allowed because they represent
        # information known at the observation timestamp.
        features.append(column)

    return features


def identify_target_columns(
    data: pd.DataFrame,
) -> list[str]:
    """
    Identify all future target columns.
    """

    targets = []

    for column in data.columns:

        if column.startswith("Future_"):
            targets.append(column)

        elif column.startswith("Direction_"):
            targets.append(column)

    return targets


def remove_invalid_feature_rows(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Remove rows where explanatory features are unavailable.

    We do NOT forward-fill technical indicators here.

    Missing values caused by indicator warm-up periods are handled
    by dropping the affected rows.
    """

    result = data.copy()

    result = result.dropna(
        subset=feature_columns
    )

    return result


def build_research_dataset(
    symbol: str,
    stock_data: pd.DataFrame,
    market_data: Optional[
        Dict[str, pd.DataFrame]
    ] = None,
    sector_data: Optional[
        Dict[str, pd.DataFrame]
    ] = None,
    horizons: Iterable[int] = (
        1,
        3,
        5,
        10,
        20,
    ),
) -> ResearchDataset:
    """
    Build the complete research dataset.

    Processing order:

        1. Clean stock index
        2. Add stock returns
        3. Add volatility
        4. Add price structure
        5. Add market context
        6. Add sector context
        7. Add future targets
        8. Identify features
        9. Remove feature warm-up rows

    The future target columns remain in the returned dataframe for
    evaluation, but are explicitly excluded from feature_columns.
    """

    if not symbol or not symbol.strip():
        raise ValueError(
            "A valid symbol is required."
        )

    symbol = symbol.strip().upper()

    result = _ensure_datetime_index(
        stock_data,
        "stock_data",
    )

    # ---------------------------------------------------------
    # Stock features
    # ---------------------------------------------------------

    result = add_stock_returns(result)

    result = add_volatility_features(
        result
    )

    result = add_price_structure_features(
        result
    )

    # ---------------------------------------------------------
    # Market context
    # ---------------------------------------------------------

    if market_data:
        result = build_market_features(
            stock_data=result,
            market_data=market_data,
        )

    # ---------------------------------------------------------
    # Sector context
    # ---------------------------------------------------------

    if sector_data:
        result = build_sector_features(
            stock_data=result,
            sector_data=sector_data,
        )

    # ---------------------------------------------------------
    # Future targets
    # ---------------------------------------------------------

    result = add_future_targets(
        result,
        horizons=horizons,
    )

    # ---------------------------------------------------------
    # Feature/target separation
    # ---------------------------------------------------------

    features = identify_feature_columns(
        result
    )

    targets = identify_target_columns(
        result
    )

    result = remove_invalid_feature_rows(
        result,
        feature_columns=features,
    )

    return ResearchDataset(
        symbol=symbol,
        data=result,
        feature_columns=features,
        target_columns=targets,
    )
