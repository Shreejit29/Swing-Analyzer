"""
Direction classification models.

The classifier predicts the probability that future return is
positive over a selected horizon.

Supported models:
    - Logistic Regression
    - Random Forest
    - Gradient Boosting

IMPORTANT:
This module does not perform train/test splitting.

Splitting, preprocessing and validation must happen outside the
model so that the same model can be evaluated correctly under
walk-forward validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression


@dataclass
class ClassifierConfig:
    """
    Configuration for a direction classifier.
    """

    model_type: str = "logistic"

    random_state: int = 42

    class_weight: Optional[str] = "balanced"

    # Logistic Regression
    logistic_c: float = 0.10
    logistic_max_iter: int = 2000

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


class DirectionClassifier:
    """
    Wrapper around classification algorithms.

    The wrapper provides a common interface so different model
    families can be compared fairly during validation.
    """

    SUPPORTED_MODELS = {
        "logistic",
        "random_forest",
        "gradient_boosting",
    }

    def __init__(
        self,
        config: Optional[
            ClassifierConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else ClassifierConfig()
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
        """Construct the selected estimator."""

        config = self.config

        if config.model_type == "logistic":

            return LogisticRegression(
                C=config.logistic_c,
                max_iter=config.logistic_max_iter,
                class_weight=config.class_weight,
                random_state=config.random_state,
                solver="lbfgs",
            )

        if config.model_type == "random_forest":

            return RandomForestClassifier(
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
                class_weight=config.class_weight,
                random_state=config.random_state,
                n_jobs=-1,
            )

        if config.model_type == "gradient_boosting":

            return GradientBoostingClassifier(
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
                random_state=config.random_state,
            )

        raise RuntimeError(
            "Model construction failed."
        )

    def _validate_training_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        """Validate training data."""

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
                "Training target contains missing values."
            )

        unique_classes = (
            pd.Series(y)
            .dropna()
            .unique()
        )

        if len(unique_classes) < 2:
            raise ValueError(
                "Training target contains only one class."
            )

        if not set(
            unique_classes
        ).issubset(
            {
                0,
                1,
            }
        ):
            raise ValueError(
                "Direction target must contain only 0 and 1."
            )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "DirectionClassifier":
        """
        Fit the classifier.

        IMPORTANT:
        X must already have been transformed using a preprocessing
        pipeline fitted exclusively on the training period.
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

        # Remove rows where features are unavailable.
        valid_rows = ~X.isna().any(
            axis=1
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

        unique_classes = (
            y_fit.unique()
        )

        if len(unique_classes) < 2:
            raise ValueError(
                "Training data contains only one class after filtering."
            )

        self.model.fit(
            X_fit,
            y_fit.astype(int),
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
                "Classifier has not been fitted."
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
                "the training feature columns."
            )

    def predict(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """Predict binary direction."""

        self._validate_prediction_data(
            X
        )

        return self.model.predict(
            X
        )

    def predict_probability(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Predict probability of each class.

        Returns:
            Array with columns:

                [P(class 0), P(class 1)]
        """

        self._validate_prediction_data(
            X
        )

        return self.model.predict_proba(
            X
        )

    def predict_up_probability(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Return probability of a positive future direction.
        """

        probabilities = (
            self.predict_probability(
                X
            )
        )

        classes = (
            self.model.classes_
        )

        if 1 not in classes:
            raise RuntimeError(
                "Positive class is unavailable."
            )

        positive_index = int(
            np.where(
                classes == 1
            )[0][0]
        )

        return probabilities[
            :,
            positive_index,
        ]

    def predict_down_probability(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Return probability of a negative future direction.
        """

        probabilities = (
            self.predict_probability(
                X
            )
        )

        classes = (
            self.model.classes_
        )

        if 0 not in classes:
            raise RuntimeError(
                "Negative class is unavailable."
            )

        negative_index = int(
            np.where(
                classes == 0
            )[0][0]
        )

        return probabilities[
            :,
            negative_index,
        ]

    def feature_importance(
        self,
    ) -> pd.DataFrame:
        """
        Return model feature importance.

        Logistic Regression:
            absolute coefficients.

        Tree models:
            native feature importance.
        """

        if not self.is_fitted_:
            raise RuntimeError(
                "Classifier has not been fitted."
            )

        if isinstance(
            self.model,
            LogisticRegression,
        ):

            importance = np.abs(
                self.model.coef_[0]
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
                "Feature importance is not available for this model."
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
                drop=True
            )
        )

    def get_model_name(
        self,
    ) -> str:
        """Return the configured model name."""

        return self.config.model_type

    def get_classes(
        self,
    ) -> np.ndarray:
        """Return learned target classes."""

        if not self.is_fitted_:
            raise RuntimeError(
                "Classifier has not been fitted."
            )

        return self.model.classes_
