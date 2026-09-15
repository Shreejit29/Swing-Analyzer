"""
Leakage-safe preprocessing for machine-learning models.

The preprocessing pipeline handles:

    - numeric feature selection
    - missing-value imputation
    - robust scaling
    - feature-name preservation

IMPORTANT:

Preprocessing must be fitted ONLY on training data.

Never fit an imputer/scaler on the complete dataset before
time-series cross-validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


@dataclass
class PreprocessorConfig:
    """
    Configuration for model preprocessing.
    """

    imputation_strategy: str = "median"

    scaling_quantile_range: tuple[
        float,
        float,
    ] = (10.0, 90.0)

    add_missing_indicators: bool = True


class SafePreprocessor:
    """
    Leakage-safe preprocessing wrapper.

    The object itself contains no fitted statistics until
    `fit()` is called.

    Typical workflow:

        processor = SafePreprocessor()

        processor.fit(X_train)

        X_train_processed = processor.transform(X_train)
        X_valid_processed = processor.transform(X_valid)
        X_test_processed = processor.transform(X_test)

    The validation and test sets are NEVER used by fit().
    """

    def __init__(
        self,
        config: Optional[
            PreprocessorConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else PreprocessorConfig()
        )

        self.pipeline: Optional[
            ColumnTransformer
        ] = None

        self.feature_names_: list[str] = []

        self.numeric_columns_: list[str] = []

        self.is_fitted_: bool = False

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
            duplicated = (
                X.columns[
                    X.columns.duplicated()
                ]
                .tolist()
            )

            raise ValueError(
                "Duplicate feature columns: "
                f"{duplicated}"
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
                "All model features must be numeric. "
                f"Non-numeric columns: {non_numeric}"
            )

    def fit(
        self,
        X: pd.DataFrame,
    ) -> "SafePreprocessor":
        """
        Fit preprocessing statistics on training data ONLY.
        """

        self._validate_X(X)

        self.numeric_columns_ = list(
            X.columns
        )

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy=(
                            self.config
                            .imputation_strategy
                        ),
                        add_indicator=(
                            self.config
                            .add_missing_indicators
                        ),
                    ),
                ),
                (
                    "scaler",
                    RobustScaler(
                        quantile_range=(
                            self.config
                            .scaling_quantile_range
                        ),
                    ),
                ),
            ]
        )

        self.pipeline = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_pipeline,
                    self.numeric_columns_,
                )
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        self.pipeline.fit(X)

        self.feature_names_ = list(
            self.pipeline
            .get_feature_names_out()
        )

        self.is_fitted_ = True

        return self

    def transform(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Transform new data using statistics learned from
        the training dataset.
        """

        if not self.is_fitted_:
            raise RuntimeError(
                "Preprocessor has not been fitted. "
                "Call fit() using training data first."
            )

        self._validate_X(X)

        missing_columns = [
            column
            for column in self.numeric_columns_
            if column not in X.columns
        ]

        if missing_columns:
            raise ValueError(
                "Input is missing features used during fitting: "
                f"{missing_columns}"
            )

        X_ordered = X[
            self.numeric_columns_
        ].copy()

        return self.pipeline.transform(
            X_ordered
        )

    def fit_transform(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Fit on X and immediately transform X.

        Use this ONLY for a training dataset.

        Do NOT use this method on validation or test data.
        """

        self.fit(X)

        return self.transform(X)

    def get_feature_names(
        self,
    ) -> list[str]:
        """Return transformed feature names."""

        if not self.is_fitted_:
            raise RuntimeError(
                "Preprocessor has not been fitted."
            )

        return list(
            self.feature_names_
        )

    def transform_dataframe(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform X and return a dataframe preserving the index.
        """

        transformed = self.transform(
            X
        )

        return pd.DataFrame(
            transformed,
            index=X.index,
            columns=self.get_feature_names(),
        )

    def reset(self) -> None:
        """Clear all fitted state."""

        self.pipeline = None

        self.feature_names_ = []

        self.numeric_columns_ = []

        self.is_fitted_ = False


def validate_train_validation_test_order(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    """
    Verify chronological ordering of train/validation/test sets.

    This prevents accidental temporal reversal.
    """

    if X_train.empty:
        raise ValueError(
            "Training data is empty."
        )

    if X_validation.empty:
        raise ValueError(
            "Validation data is empty."
        )

    if X_test.empty:
        raise ValueError(
            "Test data is empty."
        )

    train_end = X_train.index.max()

    validation_start = (
        X_validation.index.min()
    )

    validation_end = (
        X_validation.index.max()
    )

    test_start = X_test.index.min()

    if train_end >= validation_start:
        raise ValueError(
            "Training data overlaps or occurs after "
            "validation data."
        )

    if validation_end >= test_start:
        raise ValueError(
            "Validation data overlaps or occurs after "
            "test data."
        )


def ensure_numeric_features(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert feature values to numeric form.

    This function does NOT impute missing values.

    Missing values are intentionally preserved so that the
    training-only preprocessing pipeline can handle them.
    """

    result = X.copy()

    for column in result.columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    return result


def remove_infinite_values(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Replace positive/negative infinity with NaN.

    NaN values are subsequently handled by the training-only
    imputer.
    """

    result = X.copy()

    result = result.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    return result


def prepare_raw_features(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare raw features before fold-specific preprocessing.

    This operation is safe to apply before splitting because it
    does not learn statistics from the dataset.
    """

    result = ensure_numeric_features(
        X
    )

    result = remove_infinite_values(
        result
    )

    return result
