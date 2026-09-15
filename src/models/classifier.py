"""
Gradient Boosting classifier for swing direction.

Robust to OHLCV column capitalization and Yahoo Finance naming.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


def _get_close(df: pd.DataFrame) -> pd.Series:
    """Find Close regardless of common column naming."""
    candidates = ["Close", "close", "Adj Close", "adj close"]

    for name in candidates:
        if name in df.columns:
            value = df[name]
            if isinstance(value, pd.DataFrame):
                value = value.iloc[:, 0]
            return pd.to_numeric(value, errors="coerce")

    for col in df.columns:
        text = str(col).lower()
        if (
            text.startswith("close_")
            or text.endswith("_close")
            or text.startswith("adj close_")
            or text.endswith("_adj close")
        ):
            value = df[col]
            if isinstance(value, pd.DataFrame):
                value = value.iloc[:, 0]
            return pd.to_numeric(value, errors="coerce")

    raise KeyError(
        "'close' column not found. "
        f"Available columns: {list(df.columns)[:20]}"
    )


class GradientBoostingSwingClassifier:
    """Train a Gradient Boosting model to predict future positive returns."""

    def __init__(self, horizon: int = 5, random_state: int = 42):
        self.horizon = int(horizon)
        self.random_state = random_state
        self.model = None
        self.feature_columns: list[str] = []
        self.class_balance = {}
        self.trained_rows = 0

    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        x = features.copy()

        # MultiIndex columns can occur when Yahoo data is passed through.
        if isinstance(x.columns, pd.MultiIndex):
            x.columns = [
                "_".join(str(v) for v in col if str(v).lower() != "nan")
                for col in x.columns
            ]

        # Remove raw market columns from the model feature matrix.
        excluded = {
            "open", "high", "low", "close", "adj close", "volume",
            "Open", "High", "Low", "Close", "Adj Close", "Volume",
            "target", "future_return",
        }

        numeric = x.select_dtypes(include=[np.number]).copy()
        keep = [c for c in numeric.columns if str(c) not in excluded]
        numeric = numeric[keep]

        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        numeric = numeric.ffill().bfill()

        return numeric

    def fit(self, features: pd.DataFrame):
        if not isinstance(features, pd.DataFrame) or features.empty:
            raise ValueError("No feature data available for model training.")

        close = _get_close(features)

        # Future return. The final horizon rows intentionally remain NaN.
        future_return = close.shift(-self.horizon) / close - 1.0
        target = pd.Series(np.nan, index=features.index, dtype=float)
        valid_target = future_return.notna() & np.isfinite(future_return)
        target.loc[valid_target] = (future_return.loc[valid_target] > 0).astype(int)

        x = self._prepare_features(features)

        valid = target.notna()
        x = x.loc[valid]
        y = target.loc[valid].astype(int)

        if len(x) < 40:
            raise ValueError(
                f"Not enough valid observations for training ({len(x)})."
            )

        # Remove rows that still have missing feature values.
        complete = x.notna().all(axis=1)
        x = x.loc[complete]
        y = y.loc[complete]

        if len(x) < 40:
            raise ValueError(
                f"Not enough complete observations for training ({len(x)})."
            )

        if y.nunique() < 2:
            raise ValueError(
                "Training data contains only one target class. "
                "Try a different stock or prediction horizon."
            )

        self.feature_columns = list(x.columns)
        self.trained_rows = len(x)
        self.class_balance = y.value_counts(normalize=True).to_dict()

        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.04,
            max_depth=2,
            min_samples_leaf=8,
            subsample=0.85,
            random_state=self.random_state,
        )

        self.model.fit(x, y)
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained.")

        x = self._prepare_features(features)

        # Recreate missing columns if necessary and preserve training order.
        for col in self.feature_columns:
            if col not in x.columns:
                x[col] = 0.0

        x = x[self.feature_columns]
        x = x.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

        return self.model.predict_proba(x)

    def predict_latest(self, features: pd.DataFrame) -> tuple[int, float]:
        probabilities = self.predict_proba(features)

        if len(probabilities) == 0:
            raise ValueError("No rows available for prediction.")

        latest = probabilities[-1]
        down_probability = float(latest[0])
        up_probability = float(latest[1])

        prediction = int(up_probability >= down_probability)
        return prediction, up_probability
