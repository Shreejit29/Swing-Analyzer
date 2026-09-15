"""
Future price-range prediction models.

The range model estimates future return bounds rather than one
exact future price.

For example:

    Current price = 1000

    Predicted lower return = -0.03
    Predicted upper return = +0.07

    Predicted range = 970 to 1070

This is generally more appropriate for swing-trading analysis
than pretending the future closing price is exactly predictable.

IMPORTANT:
    Targets contain future information and must be constructed
    separately from explanatory features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)


@dataclass
class RangeModelConfig:
    """Configuration for range prediction."""

    model_type: str = "gradient_boosting"

    random_state: int = 42

    # Gradient boosting
    n_estimators: int = 150
    learning_rate: float = 0.03
    max_depth: int = 2
    min_samples_leaf: int = 10

    # Quantile regression.
    lower_alpha: float = 0.10
    upper_alpha: float = 0.90

    # Random forest
    random_forest_estimators: int = 300
    random_forest_max_depth: int = 8
    random_forest_min_samples_leaf: int = 10
    random_forest_max_features: str = "sqrt"


class PriceRangeModel:
    """
    Predict future return ranges.

    Two quantile models are trained:

        lower model -> lower return bound
        upper model -> upper return bound

    A separate median model estimates the central expected
    outcome.

    Quantile regression is useful because the objective is not
    merely predicting the average future return.
    """

    SUPPORTED_MODELS = {
        "gradient_boosting",
        "random_forest",
    }

    def __init__(
        self,
        config: Optional[
            RangeModelConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else RangeModelConfig()
        )

        if (
            self.config.model_type
            not in self.SUPPORTED_MODELS
        ):
            raise ValueError(
                "Unsupported range model: "
                f"{self.config.model_type}"
            )

        if not (
            0 < self.config.lower_alpha
            < 0.5
        ):
            raise ValueError(
                "lower_alpha must be between 0 and 0.5."
            )

        if not (
            0.5 < self.config.upper_alpha
            < 1
        ):
            raise ValueError(
                "upper_alpha must be between 0.5 and 1."
            )

        if (
            self.config.lower_alpha
            >= self.config.upper_alpha
        ):
            raise ValueError(
                "lower_alpha must be below upper_alpha."
            )

        self.lower_model = (
            self._build_model(
                self.config.lower_alpha
            )
        )

        self.median_model = (
            self._build_model(
                0.50
            )
        )

        self.upper_model = (
            self._build_model(
                self.config.upper_alpha
            )
        )

        self.feature_names_: list[str] = []

        self.is_fitted_: bool = False

    def _build_model(
        self,
        alpha: float,
    ):
        """Construct one quantile model."""

        config = self.config

        if config.model_type == (
            "gradient_boosting"
        ):

            return GradientBoostingRegressor(
                loss="quantile",
                alpha=alpha,
                n_estimators=(
                    config.n_estimators
                ),
                learning_rate=(
                    config.learning_rate
                ),
                max_depth=(
                    config.max_depth
                ),
                min_samples_leaf=(
                    config.min_samples_leaf
                ),
                random_state=(
                    config.random_state
                ),
            )

        return RandomForestRegressor(
            n_estimators=(
                config.random_forest_estimators
            ),
            max_depth=(
                config.random_forest_max_depth
            ),
            min_samples_leaf=(
                config.random_forest_min_samples_leaf
            ),
            max_features=(
                config.random_forest_max_features
            ),
            random_state=(
                config.random_state
            ),
            n_jobs=-1,
        )

    def _validate_X(
        self,
        X: pd.DataFrame,
    ) -> None:
        """Validate feature matrix."""

        if not isinstance(
            X,
            pd.DataFrame,
        ):
            raise TypeError(
                "X must be a pandas DataFrame."
            )

        if X.empty:
            raise ValueError(
                "X cannot be empty."
            )

        if X.columns.duplicated().any():
            raise ValueError(
                "Duplicate feature columns detected."
            )

        non_numeric = [
            column
            for column in X.columns
            if not pd.api.types.is_numeric_dtype(
                X[column]
            )
        ]

        if non_numeric:
            raise TypeError(
                "All features must be numeric. "
                f"Non-numeric: {non_numeric}"
            )

    def _validate_y(
        self,
        y: pd.Series,
    ) -> pd.Series:
        """Validate range target."""

        if isinstance(
            y,
            pd.DataFrame,
        ):

            if y.shape[1] != 1:
                raise ValueError(
                    "Target must contain one column."
                )

            y = y.iloc[:, 0]

        if not isinstance(
            y,
            pd.Series,
        ):
            y = pd.Series(y)

        if y.isna().any():
            raise ValueError(
                "Range target contains missing values."
            )

        values = y.to_numpy(
            dtype=float
        )

        if not np.isfinite(
            values
        ).all():
            raise ValueError(
                "Range target contains infinite values."
            )

        return y.astype(float)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "PriceRangeModel":
        """
        Fit lower, median and upper quantile models.

        X must already be transformed using preprocessing fitted
        exclusively on training data.
        """

        self._validate_X(X)

        y = self._validate_y(y)

        if len(X) != len(y):
            raise ValueError(
                "X and y must have equal length."
            )

        if len(X) < 50:
            raise ValueError(
                "At least 50 observations are recommended for "
                "range-model training."
            )

        valid_rows = ~X.isna().any(
            axis=1
        )

        X_fit = X.loc[
            valid_rows
        ]

        y_fit = y.loc[
            valid_rows
        ]

        if len(X_fit) < 50:
            raise ValueError(
                "Too few usable observations remain after removing "
                "missing features."
            )

        self.lower_model.fit(
            X_fit,
            y_fit,
        )

        self.median_model.fit(
            X_fit,
            y_fit,
        )

        self.upper_model.fit(
            X_fit,
            y_fit,
        )

        self.feature_names_ = list(
            X.columns
        )

        self.is_fitted_ = True

        return self

    def _validate_prediction_X(
        self,
        X: pd.DataFrame,
    ) -> None:
        """Validate prediction matrix."""

        if not self.is_fitted_:
            raise RuntimeError(
                "Range model has not been fitted."
            )

        self._validate_X(X)

        if list(X.columns) != (
            self.feature_names_
        ):
            raise ValueError(
                "Prediction feature columns do not match "
                "training features."
            )

    def predict_returns(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Predict lower, median and upper future returns.
        """

        self._validate_prediction_X(
            X
        )

        lower = np.asarray(
            self.lower_model.predict(X),
            dtype=float,
        )

        median = np.asarray(
            self.median_model.predict(X),
            dtype=float,
        )

        upper = np.asarray(
            self.upper_model.predict(X),
            dtype=float,
        )

        # Numerical models can occasionally cross quantiles.
        # Enforce an ordered interval.
        lower_bound = np.minimum(
            lower,
            upper,
        )

        upper_bound = np.maximum(
            lower,
            upper,
        )

        median = np.clip(
            median,
            lower_bound,
            upper_bound,
        )

        return pd.DataFrame(
            {
                "Predicted_Lower_Return": (
                    lower_bound
                ),
                "Predicted_Median_Return": (
                    median
                ),
                "Predicted_Upper_Return": (
                    upper_bound
                ),
            },
            index=X.index,
        )

    def predict_prices(
        self,
        X: pd.DataFrame,
        current_prices: pd.Series | np.ndarray,
    ) -> pd.DataFrame:
        """
        Convert predicted returns into predicted price ranges.
        """

        predictions = (
            self.predict_returns(X)
        )

        prices = np.asarray(
            current_prices,
            dtype=float,
        )

        if len(prices) != len(
            predictions
        ):
            raise ValueError(
                "current_prices must have the same length as X."
            )

        if not np.isfinite(
            prices
        ).all():
            raise ValueError(
                "current_prices contains non-finite values."
            )

        if (prices <= 0).any():
            raise ValueError(
                "current_prices must be positive."
            )

        predictions[
            "Predicted_Lower_Price"
        ] = (
            prices
            * (
                1
                + predictions[
                    "Predicted_Lower_Return"
                ].to_numpy()
            )
        )

        predictions[
            "Predicted_Median_Price"
        ] = (
            prices
            * (
                1
                + predictions[
                    "Predicted_Median_Return"
                ].to_numpy()
            )
        )

        predictions[
            "Predicted_Upper_Price"
        ] = (
            prices
            * (
                1
                + predictions[
                    "Predicted_Upper_Return"
                ].to_numpy()
            )
        )

        return predictions

    def feature_importance(
        self,
    ) -> pd.DataFrame:
        """
        Return average feature importance across the three
        quantile models.
        """

        if not self.is_fitted_:
            raise RuntimeError(
                "Range model has not been fitted."
            )

        models = [
            self.lower_model,
            self.median_model,
            self.upper_model,
        ]

        importances = []

        for model in models:

            if not hasattr(
                model,
                "feature_importances_",
            ):
                raise RuntimeError(
                    "Feature importance is unavailable."
                )

            importances.append(
                model.feature_importances_
            )

        average_importance = (
            np.mean(
                importances,
                axis=0,
            )
        )

        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names_,
                    "importance": average_importance,
                }
            )
            .sort_values(
                "importance",
                ascending=False,
            )
            .reset_index(
                drop=True,
            )
        )

    def get_model_name(
        self,
    ) -> str:
        """Return configured model name."""

        return self.config.model_type
