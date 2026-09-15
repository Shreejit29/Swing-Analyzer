"""
Return regression models for AI Swing Analyser.

The regression layer estimates future returns for a selected
prediction horizon.

Supported models:
    - Ridge Regression
    - Random Forest Regressor
    - Gradient Boosting Regressor

The predicted return can later be converted into a price:

    predicted_price = current_price * (1 + predicted_return)

IMPORTANT:
This module does not perform train/test splitting or preprocessing.
Those operations must happen outside the model.
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
from sklearn.linear_model import Ridge


@dataclass
class RegressorConfig:
    """Configuration for a return prediction model."""

    model_type: str = "ridge"

    random_state: int = 42

    # Ridge
    ridge_alpha: float = 10.0

    # Random Forest
    random_forest_estimators: int = 300
    random_forest_max_depth: int = 8
    random_forest_min_samples_leaf: int = 10
    random_forest_max_features: str = "sqrt"

    # Gradient Boosting
    gradient_estimators: int = 150
    gradient_learning_rate: float = 0.03
    gradient_max_depth: int = 2
    gradient_min_samples_leaf: int = 10
    gradient_loss: str = "huber"


class ReturnRegressor:
    """
    Common interface for future-return regression models.
    """

    SUPPORTED_MODELS = {
        "ridge",
        "random_forest",
        "gradient_boosting",
    }

    def __init__(
        self,
        config: Optional[
            RegressorConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else RegressorConfig()
        )

        if (
            self.config.model_type
            not in self.SUPPORTED_MODELS
        ):
            raise ValueError(
                "Unsupported model_type: "
                f"{self.config.model_type}. "
                f"Choose from {sorted(self.SUPPORTED_MODELS)}."
            )

        self.model = self._build_model()

        self.feature_names_: list[str] = []

        self.is_fitted_: bool = False

    def _build_model(self):
        """Construct the selected regression estimator."""

        config = self.config

        if config.model_type == "ridge":

            return Ridge(
                alpha=config.ridge_alpha,
            )

        if config.model_type == "random_forest":

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
                random_state=config.random_state,
                n_jobs=-1,
            )

        if config.model_type == "gradient_boosting":

            return GradientBoostingRegressor(
                n_estimators=(
                    config.gradient_estimators
                ),
                learning_rate=(
                    config.gradient_learning_rate
                ),
                max_depth=(
                    config.gradient_max_depth
                ),
                min_samples_leaf=(
                    config.gradient_min_samples_leaf
                ),
                loss=config.gradient_loss,
                random_state=config.random_state,
            )

        raise RuntimeError(
            "Regression model construction failed."
        )

    def _validate_training_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        """Validate regression training data."""

        if not isinstance(
            X,
            pd.DataFrame,
        ):
            raise TypeError(
                "X must be a pandas DataFrame."
            )

        if not isinstance(
            y,
            (
                pd.Series,
                pd.DataFrame,
            ),
        ):
            raise TypeError(
                "y must be a pandas Series or DataFrame."
            )

        if isinstance(
            y,
            pd.DataFrame,
        ):

            if y.shape[1] != 1:
                raise ValueError(
                    "y must contain exactly one target column."
                )

            y = y.iloc[:, 0]

        if len(X) != len(y):
            raise ValueError(
                "X and y must have the same number of rows."
            )

        if len(X) < 30:
            raise ValueError(
                "At least 30 observations are required for training."
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

        if y.isna().any():
            raise ValueError(
                "Regression target contains missing values."
            )

        if not np.isfinite(
            y.to_numpy(
                dtype=float
            )
        ).all():
            raise ValueError(
                "Regression target contains infinite values."
            )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "ReturnRegressor":
        """
        Fit the regression model.

        X must already be processed using a preprocessing pipeline
        fitted exclusively on the training data.
        """

        self._validate_training_data(
            X,
            y,
        )

        y_series = (
            y.iloc[:, 0]
            if isinstance(
                y,
                pd.DataFrame,
            )
            else y
        )

        valid_rows = (
            ~X.isna().any(axis=1)
        )

        X_fit = X.loc[
            valid_rows
        ]

        y_fit = y_series.loc[
            valid_rows
        ]

        if len(X_fit) < 30:
            raise ValueError(
                "Fewer than 30 usable observations remain after "
                "removing missing feature rows."
            )

        self.model.fit(
            X_fit,
            y_fit.astype(float),
        )

        self.feature_names_ = list(
            X.columns
        )

        self.is_fitted_ = True

        return self

    def _validate_prediction_data(
        self,
        X: pd.DataFrame,
    ) -> None:
        """Validate prediction features."""

        if not self.is_fitted_:
            raise RuntimeError(
                "Regressor has not been fitted."
            )

        if not isinstance(
            X,
            pd.DataFrame,
        ):
            raise TypeError(
                "X must be a pandas DataFrame."
            )

        if list(X.columns) != (
            self.feature_names_
        ):
            raise ValueError(
                "Prediction feature columns do not match "
                "training feature columns."
            )

    def predict(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """Predict future returns."""

        self._validate_prediction_data(
            X
        )

        predictions = (
            self.model.predict(X)
        )

        return np.asarray(
            predictions,
            dtype=float,
        )

    def feature_importance(
        self,
    ) -> pd.DataFrame:
        """
        Return model feature importance.

        For Ridge:
            absolute coefficient magnitude.

        For tree models:
            native feature importance.
        """

        if not self.is_fitted_:
            raise RuntimeError(
                "Regressor has not been fitted."
            )

        if isinstance(
            self.model,
            Ridge,
        ):

            importance = np.abs(
                self.model.coef_
            )

        elif hasattr(
            self.model,
            "feature_importances_",
        ):

            importance = (
                self.model.feature_importances_
            )

        else:

            raise RuntimeError(
                "Feature importance is not available."
            )

        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names_,
                    "importance": importance,
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
        """Return model name."""

        return self.config.model_type


def regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    """
    Calculate basic regression diagnostics.

    Metrics:

        MAE
        RMSE
        Mean Error
        Directional Accuracy

    These metrics are descriptive only.

    Trading performance will be evaluated separately with a
    proper backtesting engine.
    """

    actual = np.asarray(
        y_true,
        dtype=float,
    )

    predicted = np.asarray(
        y_pred,
        dtype=float,
    )

    if len(actual) != len(predicted):
        raise ValueError(
            "y_true and y_pred must have equal length."
        )

    valid = (
        np.isfinite(actual)
        & np.isfinite(predicted)
    )

    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        raise ValueError(
            "No valid observations available."
        )

    errors = (
        actual
        - predicted
    )

    mae = float(
        np.mean(
            np.abs(errors)
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                errors ** 2
            )
        )
    )

    mean_error = float(
        np.mean(errors)
    )

    directional_accuracy = float(
        np.mean(
            np.sign(actual)
            == np.sign(predicted)
        )
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "mean_error": mean_error,
        "directional_accuracy": (
            directional_accuracy
        ),
    }
