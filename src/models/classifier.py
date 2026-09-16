"""
Calibrated Ensemble Swing Trading Classifier

Models:
- Gradient Boosting
- Random Forest
- Extra Trees
- HistGradientBoosting
- Logistic Regression

Features:
- Soft probability voting
- Time-series-aware probability calibration
- Chronological validation
- Model agreement / disagreement
- Confidence and uncertainty metrics
- Aggregated feature importance
- Backward-compatible SwingClassifier API
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
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
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit


class GradientBoostingSwingClassifier:
    """
    Backward-compatible class name.

    Internally this is a calibrated probability-based ensemble.
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
        self.base_model_ = None
        self.is_fitted = False

        self.training_accuracy: Optional[float] = None

        self.validation_accuracy: Optional[float] = None
        self.validation_precision: Optional[float] = None
        self.validation_recall: Optional[float] = None
        self.validation_f1: Optional[float] = None
        self.validation_roc_auc: Optional[float] = None
        self.validation_brier: Optional[float] = None

        self.n_training_samples = 0
        self.n_validation_samples = 0

        self.component_names = [
            "Gradient Boosting",
            "Random Forest",
            "Extra Trees",
            "Histogram Gradient Boosting",
            "Logistic Regression",
        ]

    # ================================================================
    # DATA PREPARATION
    # ================================================================

    @staticmethod
    def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
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
        out = df.copy()
        out = GradientBoostingSwingClassifier._flatten_columns(out)

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

        keep = []

        for col in out.columns:
            name = str(col).strip().lower()

            if name in leakage_keywords:
                continue

            if (
                name.startswith("future_")
                or name.startswith("target_")
                or name.startswith("label_")
            ):
                continue

            keep.append(col)

        out = out[keep]

        for col in out.columns:
            if not pd.api.types.is_numeric_dtype(out[col]):
                out[col] = pd.to_numeric(
                    out[col],
                    errors="coerce",
                )

        out = out.select_dtypes(include=[np.number])

        out = out.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        return out

    def _prepare_training_data(
        self,
        df: pd.DataFrame,
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "Input must be a pandas DataFrame."
            )

        data = self._flatten_columns(df)

        close_candidates = [
            c
            for c in data.columns
            if str(c).strip().lower() == "close"
        ]

        if not close_candidates:
            raise ValueError(
                "A 'close' column is required."
            )

        close = pd.to_numeric(
            data[close_candidates[0]],
            errors="coerce",
        )

        future_close = close.shift(-self.horizon)

        valid_target = (
            future_close.notna()
            & close.notna()
        )

        y = (
            future_close[valid_target]
            > close[valid_target]
        ).astype(int)

        X_raw = data.loc[valid_target].copy()

        X = self._numeric_features(X_raw)

        if X.empty:
            raise ValueError(
                "No usable numeric model features were found."
            )

        X = X.loc[y.index]

        valid_features = ~X.isna().all(axis=1)

        X = X.loc[valid_features]
        y = y.loc[valid_features]

        if len(X) < self.min_samples:
            raise ValueError(
                f"Not enough training samples. "
                f"Need at least {self.min_samples}, "
                f"got {len(X)}."
            )

        if y.nunique() < 2:
            raise ValueError(
                "Training target contains only one class."
            )

        return X, y

    # ================================================================
    # BASE MODELS
    # ================================================================

    def _build_model(self):

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
            weights=[
                1.25,
                1.00,
                1.00,
                1.15,
                0.75,
            ],
            flatten_transform=True,
            n_jobs=-1,
        )

    # ================================================================
    # CALIBRATION
    # ================================================================

    def _build_calibrated_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ):
        """
        Calibrate the ensemble using chronological folds.

        Calibration is performed only when enough samples/classes
        exist for TimeSeriesSplit.
        """

        base_model = self._build_model()

        n_samples = len(X_train)

        # Conservative number of folds.
        if n_samples >= 120:
            n_splits = 3
        elif n_samples >= 80:
            n_splits = 2
        else:
            return base_model

        try:
            tscv = TimeSeriesSplit(
                n_splits=n_splits
            )

            calibrated = CalibratedClassifierCV(
                estimator=base_model,
                method="sigmoid",
                cv=tscv,
                n_jobs=-1,
            )

            calibrated.fit(
                X_train,
                y_train,
            )

            return calibrated

        except Exception:
            # Calibration must never make the application unusable.
            base_model.fit(
                X_train,
                y_train,
            )

            return base_model

    # ================================================================
    # FIT
    # ================================================================

    def fit(self, df: pd.DataFrame):

        X, y = self._prepare_training_data(df)

        self.feature_names_ = list(X.columns)

        n = len(X)

        validation_size = max(
            20,
            int(n * 0.20),
        )

        if n - validation_size < self.min_samples:
            validation_size = max(
                10,
                n - self.min_samples,
            )

        split_index = n - validation_size

        if split_index <= 0:
            raise ValueError(
                "Insufficient data for validation."
            )

        X_train = X.iloc[:split_index].copy()
        y_train = y.iloc[:split_index].copy()

        X_valid = X.iloc[split_index:].copy()
        y_valid = y.iloc[split_index:].copy()

        if y_train.nunique() < 2:
            raise ValueError(
                "Training portion contains only one class."
            )

        # ------------------------------------------------------------
        # VALIDATION MODEL
        # ------------------------------------------------------------

        validation_model = self._build_calibrated_model(
            X_train,
            y_train,
        )

        valid_probability = (
            validation_model
            .predict_proba(X_valid)[:, 1]
        )

        valid_prediction = (
            valid_probability
            >= self.probability_threshold
        ).astype(int)

        self.validation_accuracy = float(
            accuracy_score(
                y_valid,
                valid_prediction,
            )
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

        if y_valid.nunique() >= 2:
            self.validation_roc_auc = float(
                roc_auc_score(
                    y_valid,
                    valid_probability,
                )
            )
        else:
            self.validation_roc_auc = None

        self.validation_brier = float(
            brier_score_loss(
                y_valid,
                valid_probability,
            )
        )

        self.n_validation_samples = len(
            X_valid
        )

        # ------------------------------------------------------------
        # FINAL PRODUCTION MODEL
        # ------------------------------------------------------------

        self.base_model_ = self._build_model()

        self.model_ = self._build_calibrated_model(
            X,
            y,
        )

        # If calibration returned an uncalibrated model,
        # it is already fitted.
        self.n_training_samples = len(X)

        train_probability = (
            self.model_
            .predict_proba(X)[:, 1]
        )

        train_prediction = (
            train_probability
            >= self.probability_threshold
        ).astype(int)

        self.training_accuracy = float(
            accuracy_score(
                y,
                train_prediction,
            )
        )

        self.is_fitted = True

        return self

    # ================================================================
    # PREDICTION FEATURES
    # ================================================================

    def _prepare_prediction_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        X = self._numeric_features(df)

        if X.empty:
            raise ValueError(
                "No usable numeric features found."
            )

        for col in self.feature_names_:
            if col not in X.columns:
                X[col] = np.nan

        X = X[self.feature_names_]

        return X

    # ================================================================
    # PREDICTION
    # ================================================================

    def predict_proba(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:

        if (
            not self.is_fitted
            or self.model_ is None
        ):
            raise RuntimeError(
                "Model is not fitted. "
                "Call fit() first."
            )

        X = self._prepare_prediction_features(
            df
        )

        return self.model_.predict_proba(X)

    def predict(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:

        probabilities = self.predict_proba(df)

        return (
            probabilities[:, 1]
            >= self.probability_threshold
        ).astype(int)

    def latest_probability(
        self,
        df: pd.DataFrame,
    ) -> float:

        probabilities = self.predict_proba(df)

        if len(probabilities) == 0:
            return float("nan")

        return float(
            probabilities[-1, 1]
        )

    # ================================================================
    # MODEL AGREEMENT
    # ================================================================

    def component_probabilities(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, float]:

        if (
            not self.is_fitted
            or self.model_ is None
        ):
            raise RuntimeError(
                "Model is not fitted."
            )

        X = self._prepare_prediction_features(
            df
        )

        if len(X) == 0:
            return {}

        latest = X.iloc[[-1]]

        # CalibratedClassifierCV stores fitted
        # calibrated classifiers rather than a direct
        # named_estimators_ object.
        if hasattr(
            self.model_,
            "calibrated_classifiers_",
        ):
            # Use the calibrated ensemble probability
            # as the authoritative production probability.
            probability = float(
                self.model_
                .predict_proba(latest)[0, 1]
            )

            return {
                "Calibrated Ensemble": probability
            }

        result = {}

        if hasattr(
            self.model_,
            "named_estimators_",
        ):
            for (
                name,
                estimator,
            ) in self.model_.named_estimators_.items():

                try:
                    p = estimator.predict_proba(
                        latest
                    )[0, 1]

                    result[name] = float(p)

                except Exception:
                    result[name] = float("nan")

        return result

    def model_agreement(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Calculate agreement between ensemble components.

        Agreement:
            fraction of models predicting the same direction.

        Disagreement:
            1 - agreement.

        This is intended as a risk-control signal rather
        than a guarantee of future performance.
        """

        probabilities = self.component_probabilities(
            df
        )

        if not probabilities:
            return {
                "agreement": 0.0,
                "disagreement": 1.0,
                "bullish_models": 0,
                "bearish_models": 0,
                "total_models": 0,
                "status": "UNKNOWN",
            }

        values = [
            p
            for p in probabilities.values()
            if np.isfinite(p)
        ]

        if not values:
            return {
                "agreement": 0.0,
                "disagreement": 1.0,
                "bullish_models": 0,
                "bearish_models": 0,
                "total_models": 0,
                "status": "UNKNOWN",
            }

        bullish = sum(
            p >= 0.50
            for p in values
        )

        bearish = len(values) - bullish

        majority = max(
            bullish,
            bearish,
        )

        agreement = majority / len(values)

        if agreement >= 0.80:
            status = "STRONG AGREEMENT"
        elif agreement >= 0.60:
            status = "MODERATE AGREEMENT"
        else:
            status = "HIGH DISAGREEMENT"

        return {
            "agreement": float(agreement),
            "disagreement": float(
                1.0 - agreement
            ),
            "bullish_models": int(bullish),
            "bearish_models": int(bearish),
            "total_models": len(values),
            "status": status,
        }

    # ================================================================
    # CONFIDENCE
    # ================================================================

    def confidence_metrics(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, float]:

        p_up = self.latest_probability(df)

        if not np.isfinite(p_up):
            return {
                "probability_up": float("nan"),
                "probability_down": float("nan"),
                "confidence": float("nan"),
                "uncertainty": float("nan"),
            }

        p_down = 1.0 - p_up

        confidence = abs(
            p_up - p_down
        )

        uncertainty = (
            1.0 - confidence
        )

        return {
            "probability_up": float(p_up),
            "probability_down": float(p_down),
            "confidence": float(confidence),
            "uncertainty": float(uncertainty),
        }

    # ================================================================
    # FEATURE IMPORTANCE
    # ================================================================

    def feature_importance(
        self,
    ) -> pd.DataFrame:

        if (
            not self.is_fitted
            or self.model_ is None
        ):
            return pd.DataFrame(
                columns=[
                    "feature",
                    "importance",
                ]
            )

        all_importances = []

        # ------------------------------------------------------------
        # Direct VotingClassifier
        # ------------------------------------------------------------

        if hasattr(
            self.model_,
            "named_estimators_",
        ):

            for (
                name,
                pipeline,
            ) in self.model_.named_estimators_.items():

                try:
                    model = pipeline.named_steps[
                        "model"
                    ]

                    values = None

                    if hasattr(
                        model,
                        "feature_importances_",
                    ):
                        values = np.asarray(
                            model.feature_importances_,
                            dtype=float,
                        )

                    elif hasattr(
                        model,
                        "coef_",
                    ):
                        coef = np.asarray(
                            model.coef_,
                            dtype=float,
                        )

                        values = np.mean(
                            np.abs(coef),
                            axis=0,
                        )

                    if values is None:
                        continue

                    if len(values) > len(
                        self.feature_names_
                    ):
                        values = values[
                            :len(self.feature_names_)
                        ]

                    if len(values) != len(
                        self.feature_names_
                    ):
                        continue

                    values = np.nan_to_num(
                        values
                    )

                    total = values.sum()

                    if total > 0:
                        values = (
                            values / total
                        )

                    all_importances.append(
                        values
                    )

                except Exception:
                    continue

        # ------------------------------------------------------------
        # CalibratedClassifierCV
        # ------------------------------------------------------------

        elif hasattr(
            self.model_,
            "calibrated_classifiers_",
        ):

            for calibrated_model in (
                self.model_
                .calibrated_classifiers_
            ):

                try:
                    estimator = (
                        calibrated_model
                        .estimator
                    )

                    if not hasattr(
                        estimator,
                        "named_estimators_",
                    ):
                        continue

                    for pipeline in (
                        estimator
                        .named_estimators_.values()
                    ):

                        model = (
                            pipeline
                            .named_steps[
                                "model"
                            ]
                        )

                        values = None

                        if hasattr(
                            model,
                            "feature_importances_",
                        ):
                            values = np.asarray(
                                model.feature_importances_,
                                dtype=float,
                            )

                        elif hasattr(
                            model,
                            "coef_",
                        ):
                            coef = np.asarray(
                                model.coef_,
                                dtype=float,
                            )

                            values = np.mean(
                                np.abs(coef),
                                axis=0,
                            )

                        if values is None:
                            continue

                        if len(values) > len(
                            self.feature_names_
                        ):
                            values = values[
                                :len(
                                    self.feature_names_
                                )
                            ]

                        if len(values) != len(
                            self.feature_names_
                        ):
                            continue

                        values = np.nan_to_num(
                            values
                        )

                        total = values.sum()

                        if total > 0:
                            values = (
                                values / total
                            )

                        all_importances.append(
                            values
                        )

                except Exception:
                    continue

        if not all_importances:
            return pd.DataFrame(
                columns=[
                    "feature",
                    "importance",
                ]
            )

        importance = np.mean(
            np.vstack(
                all_importances
            ),
            axis=0,
        )

        total = importance.sum()

        if total > 0:
            importance = (
                importance / total
            )

        result = pd.DataFrame(
            {
                "feature": self.feature_names_,
                "importance": importance,
            }
        )

        return (
            result
            .sort_values(
                "importance",
                ascending=False,
            )
            .reset_index(drop=True)
        )

    # ================================================================
    # SUMMARY
    # ================================================================

    def summary(self) -> Dict[str, Any]:

        return {
            "model": (
                "Calibrated Soft Voting Ensemble"
            ),
            "model_type": (
                "Probability-based ensemble"
            ),
            "components": self.component_names,
            "horizon": self.horizon,
            "probability_threshold": (
                self.probability_threshold
            ),
            "n_features": len(
                self.feature_names_
            ),
            "n_training_samples": (
                self.n_training_samples
            ),
            "n_validation_samples": (
                self.n_validation_samples
            ),
            "training_accuracy": (
                self.training_accuracy
            ),
            "validation_accuracy": (
                self.validation_accuracy
            ),
            "validation_precision": (
                self.validation_precision
            ),
            "validation_recall": (
                self.validation_recall
            ),
            "validation_f1": (
                self.validation_f1
            ),
            "validation_roc_auc": (
                self.validation_roc_auc
            ),
            "validation_brier": (
                self.validation_brier
            ),
            "is_fitted": self.is_fitted,
        }


# ================================================================
# BACKWARD COMPATIBILITY
# ================================================================

EnsembleSwingClassifier = (
    GradientBoostingSwingClassifier
)

SwingClassifier = (
    GradientBoostingSwingClassifier
)


__all__ = [
    "GradientBoostingSwingClassifier",
    "EnsembleSwingClassifier",
    "SwingClassifier",
]
