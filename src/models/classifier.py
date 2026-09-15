from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.features.engine import build_features, feature_columns


@dataclass
class SwingClassifier:
    """
    Gradient Boosting classifier for short-term swing direction.

    Target:
        1 -> closing price is higher after `horizon` trading days
        0 -> closing price is not higher after `horizon` trading days
    """

    horizon: int = 5
    probability_threshold: float = 0.60
    random_state: int = 42
    min_samples: int = 150

    def __post_init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.columns: list[str] = []
        self.trained_rows: int = 0
        self.training_accuracy: Optional[float] = None
        self.class_balance: Optional[dict] = None

    def _make_target(self, features: pd.DataFrame) -> pd.Series:
        """Create the future-direction classification target."""
        future_close = features["close"].shift(-self.horizon)

        future_return = (
            future_close / features["close"]
        ) - 1.0

        # Do not convert the final horizon rows into class 0.
        # They have no future observation and must remain NaN.
        target = pd.Series(
            np.nan,
            index=features.index,
            dtype="float64",
        )

        valid = future_return.notna()
        target.loc[valid] = (
            future_return.loc[valid] > 0
        ).astype(int)

        return target

    def fit(self, df: pd.DataFrame) -> "SwingClassifier":
        """Build features and train the Gradient Boosting model."""
        if df is None or df.empty:
            raise ValueError("No market data supplied to the AI model.")

        if self.horizon < 1:
            raise ValueError("Prediction horizon must be at least 1 day.")

        features = build_features(df)

        if features.empty:
            raise ValueError("Unable to generate features for model training.")

        self.columns = feature_columns(features)

        if not self.columns:
            raise ValueError("No usable model features were generated.")

        target = self._make_target(features)

        X = features[self.columns].copy()

        # Replace infinite values before imputation.
        X = X.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        valid = target.notna()

        X = X.loc[valid]
        y = target.loc[valid].astype(int)

        if len(X) < self.min_samples:
            raise ValueError(
                f"Not enough historical rows. "
                f"Need at least {self.min_samples}, got {len(X)}."
            )

        if y.nunique() < 2:
            raise ValueError(
                "Training data contains only one target class. "
                "Try a longer historical period."
            )

        # Store class balance as a useful diagnostic.
        counts = y.value_counts().to_dict()

        self.class_balance = {
            "down": int(counts.get(0, 0)),
            "up": int(counts.get(1, 0)),
        }

        self.pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.04,
                        max_depth=2,
                        min_samples_leaf=8,
                        subsample=0.85,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

        self.pipeline.fit(X, y)

        self.trained_rows = len(X)

        # Diagnostic only. This is deliberately labelled as training
        # accuracy in the UI and must not be interpreted as live performance.
        self.training_accuracy = float(
            self.pipeline.score(X, y)
        )

        return self

    def predict_proba(self, df: pd.DataFrame) -> float:
        """Return probability of an upward move for the latest candle."""
        if self.pipeline is None:
            raise RuntimeError(
                "Model has not been fitted. Call fit() first."
            )

        if df is None or df.empty:
            raise ValueError("No market data supplied for prediction.")

        features = build_features(df)

        if features.empty:
            raise ValueError(
                "Unable to generate features for prediction."
            )

        missing = [
            column
            for column in self.columns
            if column not in features.columns
        ]

        if missing:
            raise ValueError(
                "Prediction features are missing: "
                + ", ".join(missing)
            )

        row = features[self.columns].iloc[[-1]].copy()

        row = row.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        probability = float(
            self.pipeline.predict_proba(row)[0, 1]
        )

        return float(
            np.clip(probability, 0.0, 1.0)
        )
