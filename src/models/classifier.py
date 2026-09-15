"""
Gradient Boosting classifier for swing-trading direction.

Target
------
1 = price is higher after the prediction horizon
0 = price is not higher after the prediction horizon

Important:
- Feature engineering is handled by src.features.engine.
- Raw OHLCV columns are excluded from model features.
- Future information is excluded from X.
- NaN and infinite values are handled safely.
- predict_proba() always returns probabilities.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


class GradientBoostingSwingClassifier:
    """
    Gradient Boosting model for swing-direction prediction.
    """

    def __init__(
        self,
        horizon: int = 5,
        probability_threshold: float = 0.60,
        random_state: int = 42,
        min_samples: int = 30,
        **kwargs: Any,
    ) -> None:

        self.horizon = int(horizon)

        self.probability_threshold = float(
            probability_threshold
        )

        self.random_state = int(
            random_state
        )

        self.min_samples = int(
            min_samples
        )

        # Store additional arguments for compatibility.
        self.extra_params = kwargs

        self.model: Pipeline | None = None

        self.feature_names_: list[str] = []

        self.training_accuracy: float | None = None

        self.is_fitted: bool = False

    # ========================================================
    # DATA HELPERS
    # ========================================================

    @staticmethod
    def _get_close(
        df: pd.DataFrame,
    ) -> pd.Series:
        """Return close price using case-insensitive matching."""

        if "close" in df.columns:

            value = df["close"]

            if isinstance(
                value,
                pd.DataFrame,
            ):
                value = value.iloc[:, 0]

            return pd.to_numeric(
                value,
                errors="coerce",
            )

        for column in df.columns:

            if str(column).strip().lower() in {
                "close",
                "close_price",
                "adj close",
                "adj_close",
            }:

                value = df[column]

                if isinstance(
                    value,
                    pd.DataFrame,
                ):
                    value = value.iloc[:, 0]

                return pd.to_numeric(
                    value,
                    errors="coerce",
                )

        raise ValueError(
            "A close column is required."
        )

    @staticmethod
    def _clean_features(
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Clean feature matrix.

        Non-numeric columns are removed.
        Infinite values become NaN.
        """

        if not isinstance(
            X,
            pd.DataFrame,
        ):
            X = pd.DataFrame(X)

        X = X.copy()

        # Convert boolean values to integers.
        for column in X.columns:

            if pd.api.types.is_bool_dtype(
                X[column]
            ):
                X[column] = X[
                    column
                ].astype(int)

        # Keep numerical columns only.
        numeric_columns = (
            X.select_dtypes(
                include=[np.number]
            ).columns
        )

        X = X[
            numeric_columns
        ].copy()

        X = X.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        return X

    @staticmethod
    def _remove_leakage_columns(
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Remove columns that contain future information,
        targets or raw OHLCV.
        """

        excluded = {
            "open",
            "high",
            "low",
            "close",
            "volume",

            "target",
            "label",
            "future_return",
            "future_close",

            "date",
            "datetime",
            "timestamp",
        }

        keep = []

        for column in X.columns:

            name = (
                str(column)
                .strip()
                .lower()
            )

            if name in excluded:
                continue

            # Avoid obvious future-looking feature names.
            if (
                "future" in name
                or "forward" in name
                or "target" in name
            ):
                continue

            keep.append(
                column
            )

        return X[
            keep
        ].copy()

    # ========================================================
    # FEATURE PREPARATION
    # ========================================================

    def _prepare_features(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert a feature-engineered dataframe into X.
        """

        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise TypeError(
                "Training data must be a pandas DataFrame."
            )

        X = self._clean_features(
            data
        )

        X = self._remove_leakage_columns(
            X
        )

        if X.empty:

            raise ValueError(
                "No usable numerical features were found."
            )

        return X

    # ========================================================
    # TARGET
    # ========================================================

    def _make_target(
        self,
        data: pd.DataFrame,
    ) -> pd.Series:
        """
        Create the future-direction target.

        target = 1 when future close > current close.
        """

        close = self._get_close(
            data
        )

        future_close = (
            close.shift(
                -self.horizon
            )
        )

        target = (
            future_close > close
        ).astype(int)

        # Last horizon rows have no future observation.
        target = target.astype(
            "float"
        )

        target[
            future_close.isna()
        ] = np.nan

        return target

    # ========================================================
    # FIT
    # ========================================================

    def fit(
        self,
        data: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> "GradientBoostingSwingClassifier":
        """
        Fit the Gradient Boosting classifier.

        Parameters
        ----------
        data:
            Feature-engineered dataframe.

        y:
            Optional target. If omitted, target is created from
            future closing prices.
        """

        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                "Cannot train on an empty dataframe."
            )

        # ----------------------------------------------------
        # Build target.
        # ----------------------------------------------------

        if y is None:

            target = self._make_target(
                data
            )

        else:

            target = pd.Series(
                y,
                index=data.index,
            )

            target = pd.to_numeric(
                target,
                errors="coerce",
            )

        # ----------------------------------------------------
        # Prepare features.
        # ----------------------------------------------------

        X = self._prepare_features(
            data
        )

        # Align X and y.
        combined = X.copy()

        combined[
            "__target__"
        ] = target

        combined = combined.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        combined = combined.dropna(
            subset=["__target__"]
        )

        if combined.empty:

            raise ValueError(
                "No valid training rows remain after "
                "target construction."
            )

        y_clean = combined[
            "__target__"
        ].astype(int)

        X_clean = combined.drop(
            columns=["__target__"]
        )

        # ----------------------------------------------------
        # Remove completely empty columns.
        # ----------------------------------------------------

        valid_columns = [
            column
            for column in X_clean.columns
            if not X_clean[
                column
            ].isna().all()
        ]

        X_clean = X_clean[
            valid_columns
        ]

        if X_clean.empty:

            raise ValueError(
                "All model features are empty."
            )

        # ----------------------------------------------------
        # Check sample size.
        # ----------------------------------------------------

        if len(X_clean) < self.min_samples:

            raise ValueError(
                f"Not enough training samples. "
                f"Required at least {self.min_samples}, "
                f"received {len(X_clean)}."
            )

        # ----------------------------------------------------
        # Check class diversity.
        # ----------------------------------------------------

        unique_classes = (
            np.unique(
                y_clean
            )
        )

        if len(unique_classes) < 2:

            raise ValueError(
                "Training target contains only one class. "
                "Both bullish and non-bullish samples are "
                "required."
            )

        # ----------------------------------------------------
        # Save feature names.
        # ----------------------------------------------------

        self.feature_names_ = list(
            X_clean.columns
        )

        # ----------------------------------------------------
        # Gradient Boosting model.
        # ----------------------------------------------------

        gb = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.85,
            random_state=self.random_state,
        )

        self.model = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        keep_empty_features=True,
                    ),
                ),
                (
                    "classifier",
                    gb,
                ),
            ]
        )

        # ----------------------------------------------------
        # Train.
        # ----------------------------------------------------

        self.model.fit(
            X_clean,
            y_clean,
        )

        # ----------------------------------------------------
        # Training accuracy.
        #
        # This is descriptive only and should NOT be treated
        # as out-of-sample performance.
        # ----------------------------------------------------

        try:

            training_prediction = (
                self.model.predict(
                    X_clean
                )
            )

            self.training_accuracy = float(
                accuracy_score(
                    y_clean,
                    training_prediction,
                )
            )

        except Exception:

            self.training_accuracy = None

        self.is_fitted = True

        return self

    # ========================================================
    # FEATURE ALIGNMENT
    # ========================================================

    def _align_features(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Align prediction data to the exact training feature set.
        """

        if not self.feature_names_:

            raise ValueError(
                "The model has no stored feature names."
            )

        X = self._prepare_features(
            data
        )

        # Add missing training columns.
        for column in self.feature_names_:

            if column not in X.columns:
                X[column] = np.nan

        # Remove columns not seen during training.
        X = X[
            self.feature_names_
        ].copy()

        X = X.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        return X

    # ========================================================
    # PREDICT PROBABILITY
    # ========================================================

    def predict_proba(
        self,
        data: pd.DataFrame,
    ) -> np.ndarray:
        """
        Return class probabilities.

        Shape:
            (n_samples, 2)

        Column 0:
            probability of class 0

        Column 1:
            probability of class 1
        """

        if not self.is_fitted:
            raise ValueError(
                "Model must be fitted before prediction."
            )

        if self.model is None:
            raise ValueError(
                "Internal model is not available."
            )

        X = self._align_features(
            data
        )

        probabilities = (
            self.model.predict_proba(
                X
            )
        )

        return np.asarray(
            probabilities,
            dtype=float,
        )

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        data: pd.DataFrame,
    ) -> np.ndarray:
        """
        Return binary predictions using the configured
        probability threshold.
        """

        probabilities = (
            self.predict_proba(
                data
            )
        )

        if probabilities.shape[1] < 2:

            return (
                probabilities[:, 0]
                >= self.probability_threshold
            ).astype(int)

        p_up = probabilities[
            :,
            1,
        ]

        return (
            p_up
            >= self.probability_threshold
        ).astype(int)

    # ========================================================
    # LATEST PROBABILITY
    # ========================================================

    def latest_probability(
        self,
        data: pd.DataFrame,
    ) -> float:
        """
        Return latest bullish probability.
        """

        probabilities = (
            self.predict_proba(
                data
            )
        )

        if probabilities.shape[1] >= 2:

            return float(
                probabilities[
                    -1,
                    1,
                ]
            )

        return float(
            probabilities[
                -1,
                0,
            ]
        )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    def feature_importance(
        self,
    ) -> pd.DataFrame:
        """
        Return Gradient Boosting feature importance.
        """

        if not self.is_fitted:
            raise ValueError(
                "Model must be fitted first."
            )

        if self.model is None:
            raise ValueError(
                "Internal model is not available."
            )

        classifier = self.model.named_steps[
            "classifier"
        ]

        importance = (
            classifier.feature_importances_
        )

        result = pd.DataFrame(
            {
                "feature": self.feature_names_,
                "importance": importance,
            }
        )

        result = result.sort_values(
            "importance",
            ascending=False,
        ).reset_index(
            drop=True
        )

        return result

    # ========================================================
    # MODEL SUMMARY
    # ========================================================

    def summary(self) -> dict[str, Any]:
        """
        Return a compact model summary.
        """

        return {
            "model": (
                "GradientBoostingClassifier"
            ),
            "horizon": self.horizon,
            "probability_threshold": (
                self.probability_threshold
            ),
            "n_features": len(
                self.feature_names_
            ),
            "training_accuracy": (
                self.training_accuracy
            ),
            "is_fitted": self.is_fitted,
        }


# ============================================================
# BACKWARD-COMPATIBLE ALIAS
# ============================================================

SwingClassifier = (
    GradientBoostingSwingClassifier
)


__all__ = [
    "GradientBoostingSwingClassifier",
    "SwingClassifier",
]
