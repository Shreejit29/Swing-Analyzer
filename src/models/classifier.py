"""
Ensemble Swing Trading Classifier

Models:
    - Gradient Boosting
    - Random Forest
    - Extra Trees
    - HistGradientBoosting
    - Logistic Regression

Architecture:
    OHLCV/features
        ↓
    chronological target
        ↓
    chronological validation
        ↓
    soft-voting ensemble
        ↓
    probability calibration
        ↓
    final probability
        ↓
    BUY / SELL / WAIT

Important:
The final `horizon` rows are excluded from training because their
future outcome is not yet known.
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

from sklearn.linear_model import (
    LogisticRegression,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
)

from sklearn.pipeline import Pipeline


# =====================================================================
# CLASS
# =====================================================================


class EnsembleSwingClassifier:
    """
    Diverse soft-voting ensemble for swing-direction prediction.
    """

    def __init__(
        self,
        horizon: int = 5,
        probability_threshold: float = 0.60,
        random_state: int = 42,
        min_samples: int = 80,
        validation_fraction: float = 0.20,
        **kwargs: Any,
    ):

        self.horizon = int(
            horizon
        )

        self.probability_threshold = float(
            probability_threshold
        )

        self.random_state = int(
            random_state
        )

        self.min_samples = int(
            min_samples
        )

        self.validation_fraction = float(
            validation_fraction
        )

        self.model = None

        self.feature_names_ = []

        self.is_fitted = False

        self.training_accuracy = None

        self.validation_accuracy = None
        self.validation_precision = None
        self.validation_recall = None
        self.validation_f1 = None
        self.validation_roc_auc = None
        self.validation_brier = None

        self.validation_predictions_ = None
        self.validation_probabilities_ = None
        self.validation_actual_ = None

        self.model_component_names_ = []

        self.component_weights_ = {}

        self.calibration_enabled = False

        self.calibration_a_ = 1.0
        self.calibration_b_ = 0.0


    # =================================================================
    # FEATURE CLEANING
    # =================================================================


    def _clean_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        if not isinstance(
            df,
            pd.DataFrame,
        ):

            raise TypeError(
                "Input must be a pandas DataFrame."
            )

        data = df.copy()

        # Remove duplicate columns.
        data = data.loc[
            :,
            ~data.columns.duplicated(
                keep="first"
            ),
        ]

        # Convert column names to strings.
        data.columns = [
            str(c)
            for c in data.columns
        ]

        # Columns that must never be model inputs.
        forbidden = {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adj_close",
            "target",
            "future",
            "future_close",
            "date",
            "datetime",
            "time",
        }

        keep_columns = []

        for column in data.columns:

            lower = column.lower().strip()

            if lower in forbidden:
                continue

            if lower.startswith(
                "future_"
            ):
                continue

            if lower.startswith(
                "target_"
            ):
                continue

            keep_columns.append(
                column
            )

        data = data[
            keep_columns
        ]

        # Numeric features only.
        data = data.select_dtypes(
            include=[
                np.number
            ]
        )

        # Replace invalid values.
        data = data.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        return data


    # =================================================================
    # TARGET
    # =================================================================


    def _create_target(
        self,
        df: pd.DataFrame,
    ):

        if "close" not in df.columns:

            raise ValueError(
                "Input data must contain "
                "'close' for target creation."
            )

        close = pd.to_numeric(
            df["close"],
            errors="coerce",
        )

        future_close = (
            close.shift(
                -self.horizon
            )
        )

        target = (
            future_close
            > close
        ).astype(float)

        valid = (
            close.notna()
            & future_close.notna()
        )

        return (
            target.loc[valid].astype(int),
            valid,
        )


    # =================================================================
    # BUILD DATASET
    # =================================================================


    def _prepare_dataset(
        self,
        df: pd.DataFrame,
    ):

        target, valid = (
            self._create_target(df)
        )

        features = self._clean_features(
            df
        )

        features = features.loc[
            valid
        ]

        target = target.loc[
            features.index
        ]

        if len(features) < self.min_samples:

            raise ValueError(
                "Not enough valid observations "
                f"for training. Need at least "
                f"{self.min_samples}, got "
                f"{len(features)}."
            )

        if target.nunique() < 2:

            raise ValueError(
                "Training target contains only "
                "one class."
            )

        # Remove columns containing no usable data.
        all_nan = [
            column
            for column in features.columns
            if features[column].notna().sum() == 0
        ]

        if all_nan:

            features = features.drop(
                columns=all_nan
            )

        if features.empty:

            raise ValueError(
                "No numeric model features available."
            )

        return (
            features,
            target,
        )


    # =================================================================
    # MODEL FACTORIES
    # =================================================================


    def _make_models(self):

        imputer = SimpleImputer(
            strategy="median",
            add_indicator=True,
        )

        gradient_boosting = Pipeline(
            [
                (
                    "imputer",
                    imputer,
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
                        random_state=(
                            self.random_state
                        ),
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
                        class_weight=(
                            "balanced_subsample"
                        ),
                        random_state=(
                            self.random_state
                        ),
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
                        random_state=(
                            self.random_state
                        ),
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        hist_gradient_boosting = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
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
                        random_state=(
                            self.random_state
                        ),
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
                        solver="lbfgs",
                        random_state=(
                            self.random_state
                        ),
                    ),
                ),
            ]
        )

        return [
            (
                "GradientBoosting",
                gradient_boosting,
            ),
            (
                "RandomForest",
                random_forest,
            ),
            (
                "ExtraTrees",
                extra_trees,
            ),
            (
                "HistGradientBoosting",
                hist_gradient_boosting,
            ),
            (
                "LogisticRegression",
                logistic_regression,
            ),
        ]


    # =================================================================
    # CALIBRATION
    # =================================================================


    def _fit_calibration(
        self,
        raw_probability: np.ndarray,
        y_true: np.ndarray,
    ) -> None:
        """
        Fit simple logistic/Platt-style calibration.

        Uses log-odds of the raw ensemble probability and fits a
        one-dimensional logistic regression.

        If calibration cannot be estimated reliably, identity
        calibration is retained.
        """

        probability = np.asarray(
            raw_probability,
            dtype=float,
        )

        y = np.asarray(
            y_true,
            dtype=int,
        )

        valid = (
            np.isfinite(probability)
            & np.isfinite(y)
        )

        probability = probability[
            valid
        ]

        y = y[
            valid
        ]

        if (
            len(probability) < 20
            or len(np.unique(y)) < 2
        ):

            self.calibration_enabled = False

            self.calibration_a_ = 1.0
            self.calibration_b_ = 0.0

            return

        probability = np.clip(
            probability,
            1e-6,
            1.0 - 1e-6,
        )

        logit = np.log(
            probability
            / (
                1.0
                - probability
            )
        ).reshape(
            -1,
            1,
        )

        calibrator = LogisticRegression(
            C=1.0,
            max_iter=1000,
        )

        try:

            calibrator.fit(
                logit,
                y,
            )

            self.calibration_a_ = float(
                calibrator.coef_[0][0]
            )

            self.calibration_b_ = float(
                calibrator.intercept_[0]
            )

            self.calibration_enabled = True

        except Exception:

            self.calibration_enabled = False

            self.calibration_a_ = 1.0
            self.calibration_b_ = 0.0


    def _calibrate_probability(
        self,
        probability: float,
    ) -> float:

        probability = float(
            np.clip(
                probability,
                1e-6,
                1.0 - 1e-6,
            )
        )

        if not self.calibration_enabled:

            return probability

        logit = np.log(
            probability
            / (
                1.0
                - probability
            )
        )

        calibrated_logit = (
            self.calibration_a_
            * logit
            + self.calibration_b_
        )

        calibrated = (
            1.0
            / (
                1.0
                + np.exp(
                    -np.clip(
                        calibrated_logit,
                        -30,
                        30,
                    )
                )
            )
        )

        return float(
            np.clip(
                calibrated,
                0.0,
                1.0,
            )
        )


    # =================================================================
    # FIT
    # =================================================================


    def fit(
        self,
        df: pd.DataFrame,
    ):

        features, target = (
            self._prepare_dataset(df)
        )

        self.feature_names_ = list(
            features.columns
        )

        X = features
        y = target

        n = len(X)

        validation_size = max(
            20,
            int(
                n
                * self.validation_fraction
            ),
        )

        # Never allow validation to consume
        # the complete training dataset.
        validation_size = min(
            validation_size,
            n - max(
                self.min_samples // 2,
                40,
            ),
        )

        if validation_size < 10:

            validation_size = 10

        split = (
            n
            - validation_size
        )

        X_train = X.iloc[
            :split
        ]

        y_train = y.iloc[
            :split
        ]

        X_validation = X.iloc[
            split:
        ]

        y_validation = y.iloc[
            split:
        ]

        if y_train.nunique() < 2:

            raise ValueError(
                "Training portion contains only "
                "one target class."
            )

        if y_validation.nunique() < 2:

            # Validation still proceeds for predictions,
            # but ROC-AUC may be unavailable.
            pass

        # =============================================================
        # VALIDATION ENSEMBLE
        # =============================================================

        validation_models = self._make_models()

        validation_ensemble = VotingClassifier(
            estimators=validation_models,
            voting="soft",
            weights=[
                1.20,
                1.00,
                1.00,
                1.10,
                0.80,
            ],
            flatten_transform=True,
        )

        validation_ensemble.fit(
            X_train,
            y_train,
        )

        raw_validation_probability = (
            validation_ensemble
            .predict_proba(
                X_validation
            )[:, 1]
        )

        validation_prediction = (
            raw_validation_probability
            >= self.probability_threshold
        ).astype(int)

        # =============================================================
        # CALIBRATION
        # =============================================================

        self._fit_calibration(
            raw_validation_probability,
            y_validation.to_numpy(),
        )

        calibrated_validation_probability = (
            np.array(
                [
                    self._calibrate_probability(
                        p
                    )
                    for p in raw_validation_probability
                ]
            )
        )

        calibrated_validation_prediction = (
            calibrated_validation_probability
            >= self.probability_threshold
        ).astype(int)

        # =============================================================
        # VALIDATION METRICS
        # =============================================================

        self.validation_actual_ = (
            y_validation.to_numpy()
        )

        self.validation_probabilities_ = (
            calibrated_validation_probability
        )

        self.validation_predictions_ = (
            calibrated_validation_prediction
        )

        self.validation_accuracy = float(
            accuracy_score(
                y_validation,
                calibrated_validation_prediction,
            )
        )

        self.validation_precision = float(
            precision_score(
                y_validation,
                calibrated_validation_prediction,
                zero_division=0,
            )
        )

        self.validation_recall = float(
            recall_score(
                y_validation,
                calibrated_validation_prediction,
                zero_division=0,
            )
        )

        self.validation_f1 = float(
            f1_score(
                y_validation,
                calibrated_validation_prediction,
                zero_division=0,
            )
        )

        if y_validation.nunique() >= 2:

            try:

                self.validation_roc_auc = float(
                    roc_auc_score(
                        y_validation,
                        calibrated_validation_probability,
                    )
                )

            except Exception:

                self.validation_roc_auc = None

        else:

            self.validation_roc_auc = None

        try:

            self.validation_brier = float(
                brier_score_loss(
                    y_validation,
                    calibrated_validation_probability,
                )
            )

        except Exception:

            self.validation_brier = None

        # =============================================================
        # FINAL PRODUCTION ENSEMBLE
        # =============================================================

        final_models = self._make_models()

        self.model_component_names_ = [
            name
            for name, _ in final_models
        ]

        self.component_weights_ = {
            "GradientBoosting": 1.20,
            "RandomForest": 1.00,
            "ExtraTrees": 1.00,
            "HistGradientBoosting": 1.10,
            "LogisticRegression": 0.80,
        }

        self.model = VotingClassifier(
            estimators=final_models,
            voting="soft",
            weights=[
                1.20,
                1.00,
                1.00,
                1.10,
                0.80,
            ],
            flatten_transform=True,
        )

        self.model.fit(
            X,
            y,
        )

        # =============================================================
        # TRAINING ACCURACY
        # =============================================================

        training_probability = (
            self.model
            .predict_proba(
                X
            )[:, 1]
        )

        training_prediction = (
            training_probability
            >= self.probability_threshold
        ).astype(int)

        self.training_accuracy = float(
            accuracy_score(
                y,
                training_prediction,
            )
        )

        self.is_fitted = True

        return self


    # =================================================================
    # ALIGN FEATURES
    # =================================================================


    def _align_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        data = self._clean_features(
            df
        )

        for column in self.feature_names_:

            if column not in data.columns:

                data[column] = np.nan

        data = data[
            self.feature_names_
        ]

        return data


    # =================================================================
    # COMPONENT PROBABILITIES
    # =================================================================


    def component_probabilities(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, float]:

        if not self.is_fitted:

            raise RuntimeError(
                "Model is not fitted."
            )

        X = self._align_features(
            df
        )

        probabilities = {}

        try:

            named_estimators = (
                self.model.named_estimators_
            )

        except Exception:

            return probabilities

        for name, estimator in (
            named_estimators.items()
        ):

            try:

                probability = float(
                    estimator.predict_proba(
                        X
                    )[-1, 1]
                )

                if np.isfinite(
                    probability
                ):

                    probabilities[
                        name
                    ] = probability

            except Exception:

                continue

        return probabilities


    # =================================================================
    # MODEL AGREEMENT
    # =================================================================


    def model_agreement(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:

        probabilities = (
            self.component_probabilities(
                df
            )
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

        bullish = sum(
            probability >= 0.50
            for probability
            in probabilities.values()
        )

        bearish = (
            len(probabilities)
            - bullish
        )

        total = len(
            probabilities
        )

        agreement = max(
            bullish,
            bearish,
        ) / total

        disagreement = (
            1.0
            - agreement
        )

        if agreement >= 0.80:

            status = "STRONG AGREEMENT"

        elif agreement >= 0.60:

            status = "MODERATE AGREEMENT"

        else:

            status = "HIGH DISAGREEMENT"

        return {
            "agreement": float(
                agreement
            ),
            "disagreement": float(
                disagreement
            ),
            "bullish_models": int(
                bullish
            ),
            "bearish_models": int(
                bearish
            ),
            "total_models": int(
                total
            ),
            "status": status,
        }


    # =================================================================
    # PREDICT PROBABILITY
    # =================================================================


    def predict_proba(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:

        if not self.is_fitted:

            raise RuntimeError(
                "Model must be fitted before prediction."
            )

        X = self._align_features(
            df
        )

        raw = (
            self.model
            .predict_proba(
                X
            )[:, 1]
        )

        calibrated = np.array(
            [
                self._calibrate_probability(
                    probability
                )
                for probability in raw
            ]
        )

        calibrated = np.clip(
            calibrated,
            0.0,
            1.0,
        )

        return np.column_stack(
            [
                1.0 - calibrated,
                calibrated,
            ]
        )


    # =================================================================
    # PREDICT
    # =================================================================


    def predict(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:

        probabilities = (
            self.predict_proba(
                df
            )[:, 1]
        )

        return (
            probabilities
            >= self.probability_threshold
        ).astype(int)


    # =================================================================
    # LATEST PROBABILITY
    # =================================================================


    def latest_probability(
        self,
        df: pd.DataFrame,
    ) -> float:

        probabilities = (
            self.predict_proba(
                df
            )
        )

        if len(probabilities) == 0:

            return 0.5

        return float(
            probabilities[-1, 1]
        )


    # =================================================================
    # FEATURE IMPORTANCE
    # =================================================================


    def feature_importance(
        self,
    ) -> pd.DataFrame:

        if not self.is_fitted:

            raise RuntimeError(
                "Model is not fitted."
            )

        importance_arrays = []
        importance_weights = []

        named_estimators = (
            self.model.named_estimators_
        )

        for name, estimator in (
            named_estimators.items()
        ):

            try:

                inner_model = (
                    estimator.named_steps[
                        "model"
                    ]
                )

            except Exception:

                inner_model = estimator

            importance = None

            # Tree-based models.
            if hasattr(
                inner_model,
                "feature_importances_",
            ):

                importance = (
                    np.asarray(
                        inner_model.feature_importances_,
                        dtype=float,
                    )
                )

            # Logistic regression.
            elif hasattr(
                inner_model,
                "coef_",
            ):

                coefficient = (
                    np.asarray(
                        inner_model.coef_,
                        dtype=float,
                    )
                )

                if coefficient.ndim == 2:

                    importance = np.abs(
                        coefficient[0]
                    )

                else:

                    importance = np.abs(
                        coefficient
                    )

            if importance is None:
                continue

            # Imputer indicators can create additional
            # dimensions. Keep original feature dimension
            # by averaging/truncating if necessary.
            n_features = len(
                self.feature_names_
            )

            if len(importance) > n_features:

                importance = importance[
                    :n_features
                ]

            elif len(importance) < n_features:

                importance = np.pad(
                    importance,
                    (
                        0,
                        n_features
                        - len(importance),
                    ),
                    constant_values=0.0,
                )

            weight = _safe_float(
                self.component_weights_.get(
                    name,
                    1.0,
                ),
                1.0,
            )

            importance_arrays.append(
                importance
            )

            importance_weights.append(
                weight
            )

        if not importance_arrays:

            return pd.DataFrame(
                columns=[
                    "feature",
                    "importance",
                ]
            )

        matrix = np.vstack(
            importance_arrays
        )

        weights = np.asarray(
            importance_weights,
            dtype=float,
        )

        weighted = (
            matrix
            * weights.reshape(
                -1,
                1,
            )
        )

        combined = (
            weighted.sum(
                axis=0
            )
            / max(
                weights.sum(),
                1e-12,
            )
        )

        total = combined.sum()

        if total > 0:

            combined = (
                combined
                / total
            )

        result = pd.DataFrame(
            {
                "feature": (
                    self.feature_names_
                ),
                "importance": combined,
            }
        )

        result = result.sort_values(
            "importance",
            ascending=False,
        ).reset_index(
            drop=True
        )

        return result


    # =================================================================
    # SUMMARY
    # =================================================================


    def summary(
        self,
    ) -> Dict[str, Any]:

        return {
            "model_name": (
                "Calibrated Soft Voting Ensemble"
            ),

            "model_type": (
                "Diverse Ensemble"
            ),

            "components": (
                list(
                    self.model_component_names_
                )
            ),

            "weights": (
                dict(
                    self.component_weights_
                )
            ),

            "horizon": int(
                self.horizon
            ),

            "probability_threshold": float(
                self.probability_threshold
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

            "calibration_enabled": bool(
                self.calibration_enabled
            ),

            "n_features": int(
                len(
                    self.feature_names_
                )
            ),

            "n_components": int(
                len(
                    self.model_component_names_
                )
            ),

            "is_fitted": bool(
                self.is_fitted
            ),
        }


# =====================================================================
# COMPATIBILITY ALIASES
# =====================================================================


GradientBoostingSwingClassifier = (
    EnsembleSwingClassifier
)

SwingClassifier = (
    EnsembleSwingClassifier
)


# =====================================================================
# INTERNAL HELPER
# =====================================================================


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:

        result = float(value)

        if np.isfinite(result):

            return result

    except Exception:
        pass

    return float(default)


__all__ = [
    "EnsembleSwingClassifier",
    "GradientBoostingSwingClassifier",
    "SwingClassifier",
]
