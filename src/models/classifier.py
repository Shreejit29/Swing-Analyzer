from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.engine import build_features, feature_columns


@dataclass
class SwingClassifier:
    horizon: int = 5
    probability_threshold: float = 0.60
    random_state: int = 42
    min_samples: int = 150

    def __post_init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.columns: list[str] = []
        self.trained_rows = 0
        self.training_accuracy: Optional[float] = None

    def _make_target(self, features: pd.DataFrame) -> pd.Series:
        future_return = features["close"].shift(-self.horizon) / features["close"] - 1
        return (future_return > 0).astype(float).where(future_return.notna())

    def fit(self, df: pd.DataFrame) -> "SwingClassifier":
        features = build_features(df)
        self.columns = feature_columns(features)
        target = self._make_target(features)
        data = features[self.columns].copy()
        valid = target.notna()
        X, y = data.loc[valid], target.loc[valid].astype(int)
        if len(X) < self.min_samples:
            raise ValueError(f"Not enough historical rows. Need at least {self.min_samples}, got {len(X)}.")
        if y.nunique() < 2:
            raise ValueError("Training data contains only one target class.")
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", GradientBoostingClassifier(
                n_estimators=180,
                learning_rate=0.04,
                max_depth=2,
                min_samples_leaf=8,
                random_state=self.random_state,
            )),
        ])
        self.pipeline.fit(X, y)
        self.trained_rows = len(X)
        self.training_accuracy = float(self.pipeline.score(X, y))
        return self

    def predict_proba(self, df: pd.DataFrame) -> float:
        if self.pipeline is None:
            raise RuntimeError("Model has not been fitted.")
        features = build_features(df)
        row = features[self.columns].iloc[[-1]]
        return float(self.pipeline.predict_proba(row)[0, 1])
