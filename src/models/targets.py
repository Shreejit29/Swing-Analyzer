"""
Target construction for AI Swing Analyser.

This module creates the variables the ML models will learn to
predict.

Targets are deliberately separated from explanatory features.

Supported prediction horizons:
    1D
    3D
    5D
    10D
    20D

Target types:
    - Binary Direction
    - Three-class Direction
    - Future return
    - Maximum favourable excursion
    - Maximum adverse excursion
    - Future high/low range

IMPORTANT:
Every target contains future information by definition.
Target columns must NEVER be supplied as model features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TargetSpec:
    """
    Definition of a single prediction target.
    """

    horizon: int
    direction_threshold: float = 0.0

    @property
    def direction_column(self) -> str:
        return f"Direction_{self.horizon}"

    @property
    def three_class_direction_column(self) -> str:
        return f"Direction3C_{self.horizon}"

    @property
    def return_column(self) -> str:
        return f"Future_Return_{self.horizon}"

    @property
    def high_return_column(self) -> str:
        return f"Future_High_Return_{self.horizon}"

    @property
    def low_return_column(self) -> str:
        return f"Future_Low_Return_{self.horizon}"

    @property
    def mfe_column(self) -> str:
        return f"MFE_{self.horizon}"

    @property
    def mae_column(self) -> str:
        return f"MAE_{self.horizon}"

    @property
    def range_upper_column(self) -> str:
        return f"Target_Upper_Return_{self.horizon}"

    @property
    def range_lower_column(self) -> str:
        return f"Target_Lower_Return_{self.horizon}"


DEFAULT_HORIZONS = (
    1,
    3,
    5,
    10,
    20,
)


def validate_horizons(
    horizons: Iterable[int],
) -> tuple[int, ...]:
    """
    Validate and normalize prediction horizons.
    """

    normalized = tuple(
        sorted(
            set(
                int(horizon)
                for horizon in horizons
            )
        )
    )

    if not normalized:
        raise ValueError(
            "At least one prediction horizon is required."
        )

    if any(
        horizon <= 0
        for horizon in normalized
    ):
        raise ValueError(
            "Prediction horizons must be positive integers."
        )

    return normalized


def add_direction_targets(
    data: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Create binary direction targets.

    Direction definition:

        1 -> future return > threshold
        0 -> future return <= threshold

    The final `horizon` rows do not have a known future return,
    therefore their target is kept as NaN.
    """

    if "Close" not in data.columns:
        raise ValueError(
            "Close column is required."
        )

    if threshold < 0:
        raise ValueError(
            "threshold cannot be negative."
        )

    result = data.copy()

    horizons = validate_horizons(
        horizons
    )

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
            future_return > threshold
        ).astype("float")

        # Rows where the future value does not exist
        # must remain missing rather than becoming
        # a valid negative class.
        result.loc[
            future_return.isna(),
            f"Direction_{horizon}",
        ] = np.nan

    return result


def add_noise_aware_direction_targets(
    data: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    threshold: float = 0.01,
) -> pd.DataFrame:
    """
    Create three-class direction targets.

    This is a research-oriented alternative to the existing
    binary Direction_* target.

    Classes
    -------
    -1 -> BEARISH
     0 -> NEUTRAL / MARKET NOISE
     1 -> BULLISH

    Example with threshold=0.01:

        future return > +1%  -> +1
        future return < -1%  -> -1
        otherwise             ->  0

    The existing Direction_* and Future_Return_* columns are
    preserved.

    IMPORTANT:
    This function does NOT replace the existing binary target.
    It adds a separate target so that both approaches can be
    compared during research.
    """

    if "Close" not in data.columns:
        raise ValueError(
            "Close column is required."
        )

    if threshold < 0:
        raise ValueError(
            "threshold cannot be negative."
        )

    result = data.copy()

    horizons = validate_horizons(
        horizons
    )

    for horizon in horizons:

        future_return = (
            result["Close"].shift(-horizon)
            / result["Close"]
            - 1.0
        )

        column = (
            f"Direction3C_{horizon}"
        )

        # Start with missing values.
        result[column] = np.nan

        valid = future_return.notna()

        # Bullish
        result.loc[
            valid
            & (
                future_return > threshold
            ),
            column,
        ] = 1.0

        # Bearish
        result.loc[
            valid
            & (
                future_return < -threshold
            ),
            column,
        ] = -1.0

        # Neutral / noise
        result.loc[
            valid
            & (
                future_return >= -threshold
            )
            & (
                future_return <= threshold
            ),
            column,
        ] = 0.0

    return result


def direction_class_name(
    value: int | float,
) -> str:
    """
    Convert a three-class direction value into a
    human-readable label.

    Returns
    -------
    str
        BEARISH, NEUTRAL, BULLISH or UNKNOWN.
    """

    if pd.isna(value):
        return "UNKNOWN"

    value = int(value)

    mapping = {
        -1: "BEARISH",
        0: "NEUTRAL",
        1: "BULLISH",
    }

    return mapping.get(
        value,
        "UNKNOWN",
    )


def direction_class_distribution(
    labels: pd.Series | np.ndarray,
) -> dict[str, float]:
    """
    Calculate the proportion of each direction class.

    This is useful for detecting class imbalance.

    Example output:

        {
            "BEARISH": 0.31,
            "NEUTRAL": 0.24,
            "BULLISH": 0.45
        }
    """

    series = pd.Series(
        labels
    ).dropna()

    if series.empty:
        return {}

    distribution = (
        series.value_counts(
            normalize=True
        )
    )

    result: dict[str, float] = {}

    for value, proportion in distribution.items():

        result[
            direction_class_name(value)
        ] = float(proportion)

    return result


def add_path_targets(
    data: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """
    Create future high/low path targets.

    For each horizon, the future path is defined using
    the next `horizon` observations.
    """

    if "Close" not in data.columns:
        raise ValueError(
            "Close column is required."
        )

    if "High" not in data.columns:
        raise ValueError(
            "High column is required."
        )

    if "Low" not in data.columns:
        raise ValueError(
            "Low column is required."
        )

    result = data.copy()

    horizons = validate_horizons(
        horizons
    )

    for horizon in horizons:

        future_high = (
            result["High"]
            .shift(-1)
            .rolling(
                window=horizon,
                min_periods=horizon,
            )
            .max()
            .shift(-(horizon - 1))
        )

        future_low = (
            result["Low"]
            .shift(-1)
            .rolling(
                window=horizon,
                min_periods=horizon,
            )
            .min()
            .shift(-(horizon - 1))
        )

        future_high_return = (
            future_high
            / result["Close"]
            - 1
        )

        future_low_return = (
            future_low
            / result["Close"]
            - 1
        )

        result[
            f"Future_High_Return_{horizon}"
        ] = future_high_return

        result[
            f"Future_Low_Return_{horizon}"
        ] = future_low_return

        # MFE represents the best favourable movement.
        result[
            f"MFE_{horizon}"
        ] = future_high_return.clip(
            lower=0
        )

        # MAE represents adverse movement.
        # Stored as a positive magnitude.
        result[
            f"MAE_{horizon}"
        ] = (
            -future_low_return
        ).clip(
            lower=0
        )

    return result


def add_range_targets(
    data: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """
    Create target-range variables.

    These are deliberately expressed as returns rather than
    absolute prices.

    The eventual prediction layer can convert:

        predicted return
            ->
        predicted price range

    using the latest available price.
    """

    result = data.copy()

    horizons = validate_horizons(
        horizons
    )

    # Path targets must exist first.
    required_targets = []

    for horizon in horizons:
        required_targets.extend(
            [
                f"Future_High_Return_{horizon}",
                f"Future_Low_Return_{horizon}",
            ]
        )

    missing = [
        column
        for column in required_targets
        if column not in result.columns
    ]

    if missing:
        result = add_path_targets(
            result,
            horizons=horizons,
        )

    for horizon in horizons:

        high_return = result[
            f"Future_High_Return_{horizon}"
        ]

        low_return = result[
            f"Future_Low_Return_{horizon}"
        ]

        result[
            f"Target_Upper_Return_{horizon}"
        ] = high_return

        result[
            f"Target_Lower_Return_{horizon}"
        ] = low_return

    return result


def build_all_targets(
    data: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    direction_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Build the complete existing target set.

    Processing order:

        Direction
            ↓
        Future path
            ↓
        Target ranges

    The three-class target is intentionally NOT included here,
    because the existing binary target API is used throughout
    the current research pipeline.
    """

    horizons = validate_horizons(
        horizons
    )

    result = add_direction_targets(
        data,
        horizons=horizons,
        threshold=direction_threshold,
    )

    result = add_path_targets(
        result,
        horizons=horizons,
    )

    result = add_range_targets(
        result,
        horizons=horizons,
    )

    return result


def build_targets(
    data: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    direction_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Backward-compatible target builder.

    Older research-pipeline modules use the name
    `build_targets`. The canonical implementation is
    `build_all_targets`.

    This wrapper intentionally delegates to the canonical
    implementation so both APIs produce exactly the same
    target set.
    """

    return build_all_targets(
        data=data,
        horizons=horizons,
        direction_threshold=direction_threshold,
    )


def target_columns(
    data: pd.DataFrame,
) -> list[str]:
    """
    Return every target column.

    This function is used as a safety barrier before modelling.

    Both binary Direction_* and three-class Direction3C_*
    targets are treated as targets.
    """

    target_prefixes = (
        "Future_",
        "Direction_",
        "Direction3C_",
        "MFE_",
        "MAE_",
        "Target_",
    )

    return [
        column
        for column in data.columns
        if column.startswith(
            target_prefixes
        )
    ]


def direction_target(
    data: pd.DataFrame,
    horizon: int,
) -> pd.Series:
    """Return one binary direction target."""

    column = (
        f"Direction_{horizon}"
    )

    if column not in data.columns:
        raise KeyError(
            f"Target not found: {column}"
        )

    return data[column]


def three_class_direction_target(
    data: pd.DataFrame,
    horizon: int,
) -> pd.Series:
    """
    Return one three-class direction target.

    Values:

        -1 -> BEARISH
         0 -> NEUTRAL
         1 -> BULLISH
    """

    column = (
        f"Direction3C_{horizon}"
    )

    if column not in data.columns:
        raise KeyError(
            f"Target not found: {column}"
        )

    return data[column]


def return_target(
    data: pd.DataFrame,
    horizon: int,
) -> pd.Series:
    """Return one future-return target."""

    column = (
        f"Future_Return_{horizon}"
    )

    if column not in data.columns:
        raise KeyError(
            f"Target not found: {column}"
        )

    return data[column]


def range_targets(
    data: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """
    Return upper and lower future-range targets.
    """

    upper = (
        f"Target_Upper_Return_{horizon}"
    )

    lower = (
        f"Target_Lower_Return_{horizon}"
    )

    missing = [
        column
        for column in (
            upper,
            lower,
        )
        if column not in data.columns
    ]

    if missing:
        raise KeyError(
            f"Range targets not found: {missing}"
        )

    return data[
        [
            lower,
            upper,
        ]
    ].copy()


def price_range_from_returns(
    current_price: float,
    lower_return: float,
    upper_return: float,
) -> tuple[float, float]:
    """
    Convert predicted return bounds into price bounds.
    """

    current_price = float(
        current_price
    )

    lower_return = float(
        lower_return
    )

    upper_return = float(
        upper_return
    )

    if not np.isfinite(current_price):
        raise ValueError(
            "current_price must be finite."
        )

    if current_price <= 0:
        raise ValueError(
            "current_price must be positive."
        )

    if not np.isfinite(lower_return):
        raise ValueError(
            "lower_return must be finite."
        )

    if not np.isfinite(upper_return):
        raise ValueError(
            "upper_return must be finite."
        )

    if lower_return > upper_return:
        raise ValueError(
            "lower_return cannot exceed upper_return."
        )

    lower_price = (
        current_price
        * (1 + lower_return)
    )

    upper_price = (
        current_price
        * (1 + upper_return)
    )

    return (
        float(lower_price),
        float(upper_price),
    )


def validate_target_columns(
    data: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> None:
    """
    Validate that all expected target columns exist.
    """

    horizons = validate_horizons(
        horizons
    )

    required = []

    for horizon in horizons:
        required.extend(
            [
                f"Direction_{horizon}",
                f"Future_Return_{horizon}",
                f"Future_High_Return_{horizon}",
                f"Future_Low_Return_{horizon}",
                f"MFE_{horizon}",
                f"MAE_{horizon}",
                f"Target_Upper_Return_{horizon}",
                f"Target_Lower_Return_{horizon}",
            ]
        )

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise KeyError(
            "Missing target columns: "
            + ", ".join(missing)
        )


__all__ = [
    "TargetSpec",
    "DEFAULT_HORIZONS",
    "validate_horizons",
    "add_direction_targets",
    "add_noise_aware_direction_targets",
    "direction_class_name",
    "direction_class_distribution",
    "add_path_targets",
    "add_range_targets",
    "build_all_targets",
    "build_targets",
    "target_columns",
    "direction_target",
    "three_class_direction_target",
    "return_target",
    "range_targets",
    "price_range_from_returns",
    "validate_target_columns",
]
