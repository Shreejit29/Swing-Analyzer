"""
Ensemble Swing Trading Classifier

A robust multi-model ensemble for directional swing prediction.

Models:
- Gradient Boosting
- Random Forest
- Extra Trees
- HistGradientBoosting
- Logistic Regression

The ensemble combines probabilities rather than hard labels.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


class GradientBoostingSwingClassifier:
    """
    Backward-compatible class name.

    Internally this is now a soft-voting ensemble rather than
    a single Gradient Boosting classifier.
    """

    def __init__(
        self,
        horizon: int = 5,
        probability_threshold: float = 0.60,
        random_state: int = 42,
        min_samples: int = 80,
        **kwargs: Any,
    ):
        self.horizon = int(horizon)
        self.probability_threshold = float(probability_threshold)
        self.random_state = int(random_state)
        self.min_samples = int(min_samples)

        self.feature_names_: list[str] = []
        self.model_ = None
        self.is_fitted = False

        self.training_accuracy: Optional[float] = None
        self.validation_accuracy: Optional[float] = None
        self.validation_precision: Optional[float] = None
        self.validation_recall: Optional[float] = None
        self.validation_f1: Optional[float] = None
        self.validation_roc_auc: Optional[float] = None

        self.n_training_samples: int = 0
        self.n_validation_samples: int = 0

        self.component_names = [
            "Gradient Boosting",
            "Random Forest",
            "Extra Trees",
            "Histogram Gradient Boosting",
            "Logistic Regression",
        ]

    # ------------------------------------------------------------------
    # DATA PREPARATION
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Flatten MultiIndex columns safely."""
        out = df.copy()

        if isinstance(out.columns, pd.MultiIndex):
            out.columns = [
                "_".join(
                    str(part).strip()
                    for part in col
                    if str(part).strip().lower() != "nan"
                ).strip("_")
                for col in out.columns
            ]

        out.columns = [str(c).strip() for c in out.columns]
        return out

    @staticmethod
    def _numeric_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Keep numeric features only and remove obvious leakage columns.
        """
        out = df.copy()
        out = GradientBoostingSwingClassifier._flatten_columns(out)

        # Case-insensitive leakage detection.
        leakage_keywords = {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "target",
            "future",
            "label",
            "date",
            "datetime",
            "timestamp",
            "time",
        }

        keep_columns = []

        for col in out.columns:
            name = str(col).strip().lower()

            # Remove raw OHLCV and target/future information.
            if name in leakage_keywords:
                continue

            if (
                name.startswith("future_")
                or name.startswith("target_")
                or name.startswith("label_")
            ):
                continue

            keep_columns.append(col)

        out = out[keep_columns]

        # Convert numeric-looking columns.
        for col in out.columns:
            if not pd.api.types.is_numeric_dtype(out[col]):
                out[col] = pd.to_numeric(out[col], errors="coerce")

        # Keep only numeric columns.
        out = out.select_dtypes(include=[np.number])

        # Remove infinities.
        out = out.replace([np.inf, -np.inf], np.nan)

        return out

    def _prepare_training_data(
        self,
        df: pd.DataFrame,
    ):
        """
        Build X/y using a strictly future-based target.

        Target:
            1 = future close > current close
            0 = future close <= current close
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        data = self._flatten_columns(df)

        close_candidates = [
            c for c in data.columns
            if str(c).strip().lower() == "close"
        ]

        if not close_candidates:
            raise ValueError(
                "A 'close' column is required to train the classifier."
            )

        close_col = close_candidates[0]

        close = pd.to_numeric(data[close_col], errors="coerce")

        future_close = close.shift(-self.horizon)

        valid_target = future_close.notna() & close.notna()

        y = (
            future_close[valid_target]
            > close[valid_target]
        ).astype(int)

        X_raw = data.loc[valid_target].copy()

        # Remove raw price columns and leakage columns.
        X = self._numeric_features(X_raw)

        if X.empty:
            raise ValueError(
                "No usable numeric model features were found."
            )

        # Align indexes exactly.
        X = X.loc[y.index]

        # Remove rows where every feature is missing.
        valid_features = ~X.isna().all(axis=1)

        X = X.loc[valid_features]
        y = y.loc[valid_features]

        if len(X) < self.min_samples:
            raise ValueError(
                f"Not enough training samples. "
                f"Need at least {self.min_samples}, got {len(X)}."
            )

        if y.nunique() < 2:
            raise ValueError(
                "Training target contains only one class. "
                "More historical data is required."
            )

        return X, y

    # ------------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------------

    def _build_model(self):
        """Create the diverse soft-voting ensemble."""

        gradient_boosting = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
                ),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=3,
                        min_samples_split=10,
                        min_samples_leaf=5,
                        subsample=0.85,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

        random_forest = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_depth=8,
                        min_samples_split=10,
                        min_samples_leaf=5,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        random_state=self.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        extra_trees = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
                ),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=250,
                        max_depth=10,
                        min_samples_split=8,
                        min_samples_leaf=4,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=self.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        histogram_gradient_boosting = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
                ),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=200,
                        learning_rate=0.05,
                        max_leaf_nodes=15,
                        min_samples_leaf=10,
                        l2_regularization=1.0,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

        logistic_regression = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        C=0.5,
                        max_iter=1500,
                        class_weight="balanced",
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

        return VotingClassifier(
            estimators=[
                ("gb", gradient_boosting),
                ("rf", random_forest),
                ("et", extra_trees),
                ("hgb", histogram_gradient_boosting),
                ("lr", logistic_regression),
            ],
            voting="soft",
            weights=[1.25, 1.0, 1.0, 1.15, 0.75],
            flatten_transform=True,
            n_jobs=-1,
        )

    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame):
        """
        Train the ensemble.

        Uses a chronological holdout for validation before fitting
        the final production model on all available historical data.
        """

        X, y = self._prepare_training_data(df)

        self.feature_names_ = list(X.columns)

        n = len(X)

        # Chronological validation.
        # Never randomly shuffle financial time-series data.
        validation_size = max(20, int(n * 0.20))

        if n - validation_size < self.min_samples:
            validation_size = max(
                10,
                n - self.min_samples,
            )

        split_index = n - validation_size

        if split_index <= 0:
            raise ValueError(
                "Insufficient data for chronological validation."
            )

        X_train = X.iloc[:split_index].copy()
        y_train = y.iloc[:split_index].copy()

        X_valid = X.iloc[split_index:].copy()
        y_valid = y.iloc[split_index:].copy()

        if y_train.nunique() < 2:
            raise ValueError(
                "Training portion contains only one target class."
            )

        if y_valid.nunique() < 2:
            # Validation metrics such as ROC-AUC cannot be calculated
            # reliably when the validation set contains one class.
            validation_possible = False
        else:
            validation_possible = True

        # --------------------------------------------------------------
        # VALIDATION MODEL
        # --------------------------------------------------------------

        validation_model = self._build_model()

        validation_model.fit(X_train, y_train)

        valid_probability = validation_model.predict_proba(X_valid)[:, 1]
        valid_prediction = (
            valid_probability >= self.probability_threshold
        ).astype(int)

        self.validation_accuracy = float(
            accuracy_score(y_valid, valid_prediction)
        )

        self.validation_precision = float(
            precision_score(
                y_valid,
                valid_prediction,
                zero_division=0,
            )
        )

        self.validation_recall = float(
            recall_score(
                y_valid,
                valid_prediction,
                zero_division=0,
            )
        )

        self.validation_f1 = float(
            f1_score(
                y_valid,
                valid_prediction,
                zero_division=0,
            )
        )

        if validation_possible:
            self.validation_roc_auc = float(
                roc_auc_score(
                    y_valid,
                    valid_probability,
                )
            )
        else:
            self.validation_roc_auc = None

        self.n_validation_samples = len(X_valid)

        # --------------------------------------------------------------
        # FINAL PRODUCTION MODEL
        # --------------------------------------------------------------

        self.model_ = self._build_model()
        self.model_.fit(X, y)

        self.n_training_samples = len(X)

        # Training accuracy is retained for backward compatibility,
        # but validation metrics are more important for reliability.
        train_probability = self.model_.predict_proba(X)[:, 1]
        train_prediction = (
            train_probability >= self.probability_threshold
        ).astype(int)

        self.training_accuracy = float(
            accuracy_score(y, train_prediction)
        )

        self.is_fitted = True

        return self

    # ------------------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------------------

    def _prepare_prediction_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        X = self._numeric_features(df)

        if X.empty:
            raise ValueError(
                "No usable numeric features found for prediction."
            )

        # Match training feature set.
        for col in self.feature_names_:
            if col not in X.columns:
                X[col] = np.nan

        X = X[self.feature_names_]

        return X

    def predict_proba(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Return probability array:

        column 0 = probability of DOWN
        column 1 = probability of UP
        """

        if not self.is_fitted or self.model_ is None:
            raise RuntimeError(
                "Model is not fitted. Call fit() first."
            )

        X = self._prepare_prediction_features(df)

        return self.model_.predict_proba(X)

    def predict(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """Return directional predictions."""

        probabilities = self.predict_proba(df)

        return (
            probabilities[:, 1]
            >= self.probability_threshold
        ).astype(int)

    def latest_probability(
        self,
        df: pd.DataFrame,
    ) -> float:
        """Return latest probability of upward movement."""

        probabilities = self.predict_proba(df)

        if len(probabilities) == 0:
            return float("nan")

        return float(probabilities[-1, 1])

    # ------------------------------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------------------------------

    def feature_importance(
        self,
    ) -> pd.DataFrame:
        """
        Aggregate feature importance across ensemble components.

        Tree-based models contribute impurity-based importance.
        Logistic regression contributes absolute coefficients.

        The result is normalized so that the total importance is 1.
        """

        if not self.is_fitted or self.model_ is None:
            return pd.DataFrame(
                columns=["feature", "importance"]
            )

        importance_values = []

        for name, estimator_pipeline in self.model_.named_estimators_.items():

            try:
                model = estimator_pipeline.named_steps["model"]

                values = None

                if hasattr(model, "feature_importances_"):
                    values = np.asarray(
                        model.feature_importances_,
                        dtype=float,
                    )

                elif hasattr(model, "coef_"):
                    coef = np.asarray(
                        model.coef_,
                        dtype=float,
                    )

                    if coef.ndim == 2:
                        values = np.mean(
                            np.abs(coef),
                            axis=0,
                        )
                    else:
                        values = np.abs(coef)

                if values is None:
                    continue

                # Imputer add_indicator=True can increase the number
                # of model features. Map only the original features
                # where possible.
                if len(values) > len(self.feature_names_):
                    values = values[:len(self.feature_names_)]

                if len(values) != len(self.feature_names_):
                    continue

                values = np.nan_to_num(
                    values,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

                total = values.sum()

                if total > 0:
                    values = values / total

                importance_values.append(values)

            except Exception:
                continue

        if not importance_values:
            return pd.DataFrame(
                columns=["feature", "importance"]
            )

        importance = np.mean(
            np.vstack(importance_values),
            axis=0,
        )

        total = importance.sum()

        if total > 0:
            importance = importance / total

        result = pd.DataFrame(
            {
                "feature": self.feature_names_,
                "importance": importance,
            }
        )

        return result.sort_values(
            "importance",
            ascending=False,
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # COMPONENT PROBABILITIES
    # ------------------------------------------------------------------

    def component_probabilities(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Return the latest probability from each ensemble component.

        Useful for understanding model agreement/disagreement.
        """

        if not self.is_fitted or self.model_ is None:
            raise RuntimeError(
                "Model is not fitted. Call fit() first."
            )

        X = self._prepare_prediction_features(df)

        if len(X) == 0:
            return {}

        latest = X.iloc[[-1]]

        result = {}

        for name, estimator_pipeline in self.model_.named_estimators_.items():

            try:
                probability = estimator_pipeline.predict_proba(
                    latest
                )[0, 1]

                result[name] = float(probability)

            except Exception:
                result[name] = float("nan")

        return result

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return model diagnostics."""

        return {
            "model": "Soft Voting Ensemble",
            "model_type": "Probability-based ensemble",
            "components": self.component_names,
            "horizon": self.horizon,
            "probability_threshold": self.probability_threshold,
            "n_features": len(self.feature_names_),
            "n_training_samples": self.n_training_samples,
            "n_validation_samples": self.n_validation_samples,
            "training_accuracy": self.training_accuracy,
            "validation_accuracy": self.validation_accuracy,
            "validation_precision": self.validation_precision,
            "validation_recall": self.validation_recall,
            "validation_f1": self.validation_f1,
            "validation_roc_auc": self.validation_roc_auc,
            "is_fitted": self.is_fitted,
        }


# ----------------------------------------------------------------------
# BACKWARD COMPATIBILITY
# ----------------------------------------------------------------------

EnsembleSwingClassifier = GradientBoostingSwingClassifier
SwingClassifier = GradientBoostingSwingClassifier


__all__ = [
    "GradientBoostingSwingClassifier",
    "EnsembleSwingClassifier",
    "SwingClassifier",
]
