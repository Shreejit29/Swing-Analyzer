from __future__ import annotations
import pandas as pd
from .technical import add_technical_features
from .price_action import add_price_action_features
from .volume import add_volume_features


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    x = add_technical_features(df)
    x = add_price_action_features(x)
    x = add_volume_features(x)
    return x.replace([float("inf"), float("-inf")], pd.NA)


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"open", "high", "low", "close", "volume", "target", "future_return"}
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
