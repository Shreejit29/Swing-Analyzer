
"""
Ensemble Swing Classifier
--------------------------

A leakage-conscious, sklearn-only ensemble for short-horizon swing direction.

Design:
- Gradient Boosting
- Random Forest
- Extra Trees
- HistGradientBoosting
- Logistic Regression
- chronological out-of-fold validation
- OOF probability calibration
- data-driven ensemble weights from OOF log loss
- final production refit on all available labelled observations

The public SwingClassifier name is retained for compatibility.
"""

from __future__ import annotations

from typing import Optional
import hashlib

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit

from src.features.engine import feature_columns


class EnsembleSwingClassifier:
    """
    Diverse ensemble classifier for future positive-price direction.

    Target:
        1 -> close after `horizon` trading days is higher
        0 -> otherwise

    Important:
        Validation probabilities are generated out-of-fold in chronological
        order. They are not predictions from a model trained on the same
        observations.
    """

    def __init__(
        self,
        horizon: int = 5,
        probability_threshold: float = 0.60,
        random_state: int = 42,
        min_samples: int = 120,
        n_splits: int = 5,
        min_train_size: int = 60,
        max_training_rows: Optional[int] = 1200,
    ):
        self.horizon = int(horizon)
        self.probability_threshold = float(probability_threshold)
        self.random_state = int(random_state)
        self.min_samples = int(min_samples)
        self.n_splits = int(n_splits)
        self.min_train_size = int(min_train_size)
        self.max_training_rows = (
            int(max_training_rows) if max_training_rows is not None else None
        )
        if self.max_training_rows is not None and self.max_training_rows < self.min_samples:
            raise ValueError("max_training_rows must be >= min_samples.")

        self.models = {}
        self.model = None
        self.feature_columns: list[str] = []
        self.feature_names_: list[str] = []
        self.trained_rows = 0
        self.class_balance = {}
        self.training_start_ = None
        self.training_end_ = None
        self.training_window_ = 0
        self.training_data_fingerprint_ = None
        self.model_fingerprint_ = None

        self.component_names: list[str] = []
        self.component_weights = {}
        self.base_weights = {
            "GradientBoosting": 1.20,
            "RandomForest": 1.00,
            "ExtraTrees": 1.00,
            "HistGradientBoosting": 1.10,
            "LogisticRegression": 0.80,
        }

        self.calibrator: Optional[LogisticRegression] = None
        self.calibration_enabled = False

        self.training_accuracy = None
        self.validation_accuracy = None
        self.validation_precision = None
        self.validation_recall = None
        self.validation_f1 = None
        self.validation_roc_auc = None
        self.validation_brier = None
        self.validation_log_loss = None

        self.validation_probabilities_ = None
        self.validation_actuals_ = None
        self.validation_predictions_ = None
        self.validation_component_probabilities_ = None
        self.validation_component_scores_ = None

        self.oof_folds_ = []
        self.component_validation_metrics_ = {}

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(features, pd.DataFrame) or features.empty:
            raise ValueError("No feature data available for model training.")

        x = features.copy()

        if isinstance(x.columns, pd.MultiIndex):
            x.columns = [
                "_".join(
                    str(v)
                    for v in col
                    if str(v).lower() != "nan"
                )
                for col in x.columns
            ]

        # Prefer the central feature-engineering contract.
        try:
            columns = feature_columns(x)
        except Exception:
            columns = []

        if columns:
            keep = [c for c in columns if c in x.columns]
            x = x[keep].copy()
        else:
            excluded = {
                "open", "high", "low", "close",
                "adj close", "volume",
                "Open", "High", "Low", "Close",
                "Adj Close", "Volume",
                "target", "future_return",
            }
            numeric = x.select_dtypes(include=[np.number]).copy()
            x = numeric[
                [c for c in numeric.columns if str(c) not in excluded]
            ]

        x = x.select_dtypes(include=[np.number]).copy()
        x = x.replace([np.inf, -np.inf], np.nan)

        # Do not fill using future rows. The imputer inside every model is
        # fitted only on each training fold.
        return x

    def _make_target(self, features: pd.DataFrame) -> pd.Series:
        if "close" not in features.columns:
            candidates = [
                c for c in features.columns
                if str(c).lower() in {"close", "adj close"}
            ]
            if not candidates:
                raise KeyError("A close price column is required.")
            close = pd.to_numeric(
                features[candidates[0]],
                errors="coerce",
            )
        else:
            close = pd.to_numeric(
                features["close"],
                errors="coerce",
            )

        future_close = close.shift(-self.horizon)
        future_return = future_close / close - 1.0

        target = pd.Series(
            np.nan,
            index=features.index,
            dtype="float64",
        )

        valid = future_return.notna() & np.isfinite(future_return)
        target.loc[valid] = (
            future_return.loc[valid] > 0
        ).astype(int)

        return target

    # ------------------------------------------------------------------
    # Model factory
    # ------------------------------------------------------------------

    def _build_models(self):
        rs = self.random_state

        return {
            "GradientBoosting": Pipeline(
                [
                    ("imputer", SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    )),
                    ("model", GradientBoostingClassifier(
                        n_estimators=220,
                        learning_rate=0.035,
                        max_depth=2,
                        min_samples_leaf=8,
                        subsample=0.85,
                        random_state=rs,
                    )),
                ]
            ),
            "RandomForest": Pipeline(
                [
                    ("imputer", SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    )),
                    ("model", RandomForestClassifier(
                        n_estimators=300,
                        max_depth=7,
                        min_samples_leaf=5,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=rs,
                    )),
                ]
            ),
            "ExtraTrees": Pipeline(
                [
                    ("imputer", SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    )),
                    ("model", ExtraTreesClassifier(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=4,
                        max_features="sqrt",
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=rs,
                    )),
                ]
            ),
            "HistGradientBoosting": Pipeline(
                [
                    ("imputer", SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    )),
                    ("model", HistGradientBoostingClassifier(
                        max_iter=220,
                        learning_rate=0.035,
                        max_leaf_nodes=15,
                        min_samples_leaf=12,
                        l2_regularization=1.0,
                        random_state=rs,
                    )),
                ]
            ),
            "LogisticRegression": Pipeline(
                [
                    ("imputer", SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    )),
                    ("model", LogisticRegression(
                        C=0.25,
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=rs,
                    )),
                ]
            ),
        }

    # ------------------------------------------------------------------
    # Chronological folds
    # ------------------------------------------------------------------

    def _make_splits(self, n_rows: int):
        if n_rows < max(self.min_train_size + 20, 80):
            return []

        requested = max(2, min(self.n_splits, 5))
        splitter = TimeSeriesSplit(
            n_splits=requested
        )

        splits = []
        for fold, (train_idx, valid_idx) in enumerate(
            splitter.split(np.arange(n_rows))
        ):
            if len(train_idx) < self.min_train_size:
                continue
            if len(valid_idx) < 5:
                continue

            splits.append(
                {
                    "fold": fold + 1,
                    "train": train_idx,
                    "validation": valid_idx,
                }
            )

        return splits

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _fit_calibrator(self, probabilities, actuals):
        probabilities = np.asarray(
            probabilities,
            dtype=float,
        )
        actuals = np.asarray(
            actuals,
            dtype=int,
        )

        if len(probabilities) < 30:
            return

        if len(np.unique(actuals)) < 2:
            return

        # Work in logit space. This is equivalent to a one-dimensional
        # logistic calibration model while preserving monotonicity.
        p = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
        logits = np.log(p / (1.0 - p)).reshape(-1, 1)

        calibrator = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=self.random_state,
        )

        try:
            calibrator.fit(logits, actuals)
            self.calibrator = calibrator
            self.calibration_enabled = True
        except Exception:
            self.calibrator = None
            self.calibration_enabled = False

    def _calibrate(self, probabilities):
        p = np.asarray(
            probabilities,
            dtype=float,
        )

        if not self.calibration_enabled or self.calibrator is None:
            return np.clip(p, 0.0, 1.0)

        clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
        logits = np.log(
            clipped / (1.0 - clipped)
        ).reshape(-1, 1)

        try:
            calibrated = self.calibrator.predict_proba(
                logits
            )[:, 1]
            return np.clip(
                calibrated,
                0.0,
                1.0,
            )
        except Exception:
            return np.clip(p, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, features: pd.DataFrame):
        if not isinstance(features, pd.DataFrame) or features.empty:
            raise ValueError("No feature data available for model training.")

        if self.horizon < 1:
            raise ValueError("Prediction horizon must be at least 1 day.")

        target = self._make_target(features)
        x = self._prepare_features(features)

        valid = target.notna()
        x = x.loc[valid].copy()
        y = target.loc[valid].astype(int)

        # Remove rows where every feature is unavailable.
        usable = ~x.isna().all(axis=1)
        x = x.loc[usable].copy()
        y = y.loc[usable].copy()

        if len(x) < self.min_samples:
            raise ValueError(
                f"Not enough historical rows. Need at least "
                f"{self.min_samples}, got {len(x)}."
            )

        if y.nunique() < 2:
            raise ValueError(
                "Training data contains only one target class. "
                "Try a longer historical period."
            )

        # Keep the training sample stable across repeated live refreshes.
        # The latest rows are retained, while the full feature frame remains
        # available to the caller for the final prediction.
        if (
            self.max_training_rows is not None
            and len(x) > self.max_training_rows
        ):
            x = x.iloc[-self.max_training_rows:].copy()
            y = y.loc[x.index].copy()

        # Enforce chronological order before validation/training.
        if not x.index.is_monotonic_increasing:
            order = np.argsort(np.asarray(x.index))
            x = x.iloc[order].copy()
            y = y.iloc[order].copy()

        self.feature_columns = list(x.columns)

        # Backward-compatible attribute expected by the predictor/UI.
        # Keep it synchronized with the exact feature order used by the
        # production ensemble.
        self.feature_names_ = list(self.feature_columns)

        self.trained_rows = len(x)
        self.training_window_ = len(x)
        if len(x):
            self.training_start_ = str(x.index[0])
            self.training_end_ = str(x.index[-1])

        # Fingerprint the exact labelled training snapshot and configuration.
        # This makes it possible to diagnose why a refreshed model changed.
        try:
            row_hash = pd.util.hash_pandas_object(
                pd.concat([x, y.rename("__target__")], axis=1),
                index=True,
            ).values.tobytes()
            config = repr((
                self.horizon,
                self.random_state,
                self.min_samples,
                self.n_splits,
                self.min_train_size,
                self.max_training_rows,
                tuple(self.feature_columns),
            )).encode("utf-8")
            self.training_data_fingerprint_ = hashlib.sha256(row_hash).hexdigest()[:16]
            self.model_fingerprint_ = hashlib.sha256(
                config + self.training_data_fingerprint_.encode("utf-8")
            ).hexdigest()[:16]
        except Exception:
            self.training_data_fingerprint_ = None
            self.model_fingerprint_ = None

        counts = y.value_counts().to_dict()
        self.class_balance = {
            "down": int(counts.get(0, 0)),
            "up": int(counts.get(1, 0)),
        }

        self.models = self._build_models()
        self.component_names = list(self.models.keys())

        # --------------------------------------------------------------
        # Chronological OOF predictions
        # --------------------------------------------------------------

        splits = self._make_splits(len(x))

        if not splits:
            # Safe fallback for shorter datasets: a single chronological
            # holdout is better than reporting training performance as OOS.
            split_point = max(
                self.min_train_size,
                int(len(x) * 0.80),
            )
            if split_point >= len(x):
                split_point = len(x) - 20

            splits = [
                {
                    "fold": 1,
                    "train": np.arange(0, split_point),
                    "validation": np.arange(
                        split_point,
                        len(x),
                    ),
                }
            ]

        oof_component = {
            name: np.full(
                len(x),
                np.nan,
                dtype=float,
            )
            for name in self.component_names
        }

        self.oof_folds_ = []

        for fold_info in splits:
            train_idx = fold_info["train"]
            valid_idx = fold_info["validation"]

            fold_actual = y.iloc[valid_idx].to_numpy()

            if len(np.unique(y.iloc[train_idx])) < 2:
                continue

            self.oof_folds_.append(
                {
                    "fold": int(fold_info["fold"]),
                    "train_rows": int(len(train_idx)),
                    "validation_rows": int(len(valid_idx)),
                }
            )

            for name, estimator in self.models.items():
                fitted = clone(estimator)

                try:
                    fitted.fit(
                        x.iloc[train_idx],
                        y.iloc[train_idx],
                    )
                    p = fitted.predict_proba(
                        x.iloc[valid_idx]
                    )[:, 1]

                    oof_component[name][valid_idx] = np.clip(
                        p,
                        0.0,
                        1.0,
                    )
                except Exception:
                    # A single component failure must not destroy the
                    # entire ensemble.
                    continue

        component_frame = pd.DataFrame(
            oof_component,
            index=x.index,
        )

        # Only rows where at least one model generated an OOF prediction.
        oof_valid = component_frame.notna().any(axis=1)

        if oof_valid.sum() < 20:
            raise ValueError(
                "Unable to generate enough chronological out-of-fold "
                "validation predictions."
            )

        component_frame = component_frame.loc[oof_valid]
        oof_y = y.loc[oof_valid].astype(int)

        # --------------------------------------------------------------
        # Data-driven ensemble weights
        # --------------------------------------------------------------

        metrics = {}

        for name in self.component_names:
            p = component_frame[name]

            if p.notna().sum() < 10:
                continue

            mask = p.notna()
            actual = oof_y.loc[mask].to_numpy()
            prob = np.clip(
                p.loc[mask].to_numpy(),
                1e-6,
                1.0 - 1e-6,
            )

            try:
                component_logloss = float(
                    log_loss(
                        actual,
                        np.column_stack(
                            [1.0 - prob, prob]
                        ),
                        labels=[0, 1],
                    )
                )
            except Exception:
                component_logloss = 0.693147

            try:
                component_auc = float(
                    roc_auc_score(
                        actual,
                        prob,
                    )
                )
            except Exception:
                component_auc = np.nan

            metrics[name] = {
                "oof_samples": int(mask.sum()),
                "log_loss": component_logloss,
                "roc_auc": component_auc,
            }

        self.component_validation_metrics_ = metrics

        raw_weights = {}

        for name in self.component_names:
            base = self.base_weights.get(name, 1.0)

            if name not in metrics:
                raw_weights[name] = 0.01
                continue

            ll = metrics[name]["log_loss"]

            # Lower OOF log-loss => higher weight.
            quality = 1.0 / max(ll, 0.05)

            # Keep weighting conservative so one model cannot dominate
            # merely because of a noisy validation split.
            quality = float(
                np.clip(
                    quality / 1.50,
                    0.50,
                    1.50,
                )
            )

            raw_weights[name] = base * quality

        weight_sum = sum(raw_weights.values())

        if weight_sum <= 0:
            self.component_weights = {
                name: 1.0 / len(self.component_names)
                for name in self.component_names
            }
        else:
            self.component_weights = {
                name: float(
                    raw_weights[name] / weight_sum
                )
                for name in self.component_names
            }

        # --------------------------------------------------------------
        # Weighted OOF ensemble probability
        # --------------------------------------------------------------

        weighted_oof = np.full(
            len(component_frame),
            np.nan,
            dtype=float,
        )

        for i, (_, row) in enumerate(
            component_frame.iterrows()
        ):
            values = []
            weights = []

            for name in self.component_names:
                value = row.get(name, np.nan)

                if pd.notna(value):
                    values.append(float(value))
                    weights.append(
                        self.component_weights.get(
                            name,
                            0.0,
                        )
                    )

            if values and sum(weights) > 0:
                weighted_oof[i] = float(
                    np.average(
                        values,
                        weights=weights,
                    )
                )

        valid_oof = np.isfinite(weighted_oof)

        if valid_oof.sum() < 20:
            raise ValueError(
                "Insufficient valid ensemble OOF predictions."
            )

        oof_probability = np.clip(
            weighted_oof[valid_oof],
            1e-6,
            1.0 - 1e-6,
        )
        oof_actual = oof_y.to_numpy()[valid_oof]

        # Calibrate only against OOF predictions.
        self._fit_calibrator(
            oof_probability,
            oof_actual,
        )

        calibrated_oof = self._calibrate(
            oof_probability
        )

        oof_predictions = (
            calibrated_oof >= 0.50
        ).astype(int)

        self.validation_probabilities_ = np.column_stack(
            [
                1.0 - calibrated_oof,
                calibrated_oof,
            ]
        )
        self.validation_actuals_ = oof_actual
        self.validation_predictions_ = oof_predictions

        self.validation_component_probabilities_ = (
            component_frame.loc[
                component_frame.index[
                    valid_oof
                ]
            ].to_numpy()
        )

        self.validation_component_scores_ = metrics

        self.validation_accuracy = float(
            accuracy_score(
                oof_actual,
                oof_predictions,
            )
        )
        self.validation_precision = float(
            precision_score(
                oof_actual,
                oof_predictions,
                zero_division=0,
            )
        )
        self.validation_recall = float(
            recall_score(
                oof_actual,
                oof_predictions,
                zero_division=0,
            )
        )
        self.validation_f1 = float(
            f1_score(
                oof_actual,
                oof_predictions,
                zero_division=0,
            )
        )

        try:
            self.validation_roc_auc = float(
                roc_auc_score(
                    oof_actual,
                    calibrated_oof,
                )
            )
        except Exception:
            self.validation_roc_auc = np.nan

        self.validation_brier = float(
            brier_score_loss(
                oof_actual,
                calibrated_oof,
            )
        )

        self.validation_log_loss = float(
            log_loss(
                oof_actual,
                np.column_stack(
                    [
                        1.0 - calibrated_oof,
                        calibrated_oof,
                    ]
                ),
                labels=[0, 1],
            )
        )

        # --------------------------------------------------------------
        # Production refit on all historical labelled observations
        # --------------------------------------------------------------

        self.models = self._build_models()

        for name, estimator in self.models.items():
            estimator.fit(x, y)

        # Compatibility: expose the production ensemble dictionary.
        self.model = self.models

        # Training accuracy is ensemble accuracy on training data only.
        # It is deliberately kept separate from OOF validation accuracy.
        production_probabilities = self._predict_raw_components(x)
        production_probability = self._weighted_average(
            production_probabilities
        )
        production_probability = self._calibrate(
            production_probability
        )
        production_prediction = (
            production_probability >= 0.50
        ).astype(int)

        self.training_accuracy = float(
            accuracy_score(
                y.to_numpy(),
                production_prediction,
            )
        )

        return self

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _align_features(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        x = self._prepare_features(features)

        for column in self.feature_columns:
            if column not in x.columns:
                x[column] = np.nan

        return x[self.feature_columns].copy()

    def _predict_raw_components(self, x):
        predictions = {}

        for name, model in self.models.items():
            try:
                predictions[name] = np.asarray(
                    model.predict_proba(x)[:, 1],
                    dtype=float,
                )
            except Exception:
                predictions[name] = np.full(
                    len(x),
                    np.nan,
                    dtype=float,
                )

        return predictions

    def _weighted_average(self, component_probabilities):
        if not component_probabilities:
            return np.full(0, 0.5)

        n = len(
            next(iter(component_probabilities.values()))
        )

        output = np.full(
            n,
            np.nan,
            dtype=float,
        )

        for i in range(n):
            values = []
            weights = []

            for name, probabilities in (
                component_probabilities.items()
            ):
                if i >= len(probabilities):
                    continue

                value = probabilities[i]

                if np.isfinite(value):
                    values.append(float(value))
                    weights.append(
                        self.component_weights.get(
                            name,
                            0.0,
                        )
                    )

            if values and sum(weights) > 0:
                output[i] = np.average(
                    values,
                    weights=weights,
                )

        return np.nan_to_num(
            output,
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        )

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if not self.models:
            raise RuntimeError(
                "Model has not been fitted. Call fit() first."
            )

        if features is None or features.empty:
            raise ValueError(
                "No feature data supplied for prediction."
            )

        x = self._align_features(features)

        raw_components = self._predict_raw_components(x)
        raw_probability = self._weighted_average(
            raw_components
        )

        calibrated = self._calibrate(
            raw_probability
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

    def predict_latest(
        self,
        features: pd.DataFrame,
    ) -> tuple[int, float]:
        probabilities = self.predict_proba(features)

        if len(probabilities) == 0:
            raise ValueError(
                "No rows available for prediction."
            )

        latest = probabilities[-1]

        down_probability = float(latest[0])
        up_probability = float(latest[1])

        prediction = int(
            up_probability >= down_probability
        )

        return prediction, up_probability

    # ------------------------------------------------------------------
    # Ensemble diagnostics
    # ------------------------------------------------------------------

    def component_probabilities(
        self,
        features: pd.DataFrame,
    ) -> dict:
        if not self.models:
            raise RuntimeError(
                "Model has not been fitted."
            )

        x = self._align_features(features)
        raw = self._predict_raw_components(x)

        latest = {}

        for name, probabilities in raw.items():
            if len(probabilities):
                latest[name] = float(
                    np.clip(
                        probabilities[-1],
                        0.0,
                        1.0,
                    )
                )

        return latest

    def model_agreement(
        self,
        features: pd.DataFrame,
    ) -> dict:
        probabilities = self.component_probabilities(
            features
        )

        if not probabilities:
            return {
                "agreement": 0.0,
                "disagreement": 1.0,
                "bullish_models": 0,
                "bearish_models": 0,
                "total_models": 0,
                "status": "NO DATA",
            }

        bullish = sum(
            p >= 0.50
            for p in probabilities.values()
        )
        bearish = len(probabilities) - bullish

        total = len(probabilities)

        agreement = max(
            bullish,
            bearish,
        ) / total

        disagreement = 1.0 - agreement

        if agreement >= 0.80:
            status = "STRONG AGREEMENT"
        elif agreement >= 0.60:
            status = "MODERATE AGREEMENT"
        else:
            status = "LOW AGREEMENT"

        return {
            "agreement": float(agreement),
            "disagreement": float(disagreement),
            "bullish_models": int(bullish),
            "bearish_models": int(bearish),
            "total_models": int(total),
            "status": status,
        }

    # Compatibility alias.
    model_agreement_score = model_agreement

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self) -> pd.DataFrame:
        if not self.models or not self.feature_columns:
            return pd.DataFrame(
                columns=["feature", "importance"]
            )

        accum = {
            feature: 0.0
            for feature in self.feature_columns
        }

        total_weight = 0.0

        for name, pipeline in self.models.items():
            estimator = pipeline.named_steps.get(
                "model"
            )
            weight = self.component_weights.get(
                name,
                0.0,
            )

            if weight <= 0:
                continue

            values = None

            if hasattr(
                estimator,
                "feature_importances_",
            ):
                values = np.asarray(
                    estimator.feature_importances_,
                    dtype=float,
                )

            elif hasattr(
                estimator,
                "coef_",
            ):
                coef = np.asarray(
                    estimator.coef_,
                    dtype=float,
                )

                if coef.ndim == 2:
                    values = np.abs(coef[0])
                else:
                    values = np.abs(coef)

            if values is None:
                continue

            # The imputer may add missingness indicators. The first
            # len(feature_columns) values correspond to the original
            # feature set.
            values = values[: len(self.feature_columns)]

            if len(values) != len(self.feature_columns):
                continue

            values = np.nan_to_num(
                values,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            if values.sum() > 0:
                values = values / values.sum()

            for feature, value in zip(
                self.feature_columns,
                values,
            ):
                accum[feature] += (
                    weight * float(value)
                )

            total_weight += weight

        if total_weight > 0:
            accum = {
                key: value / total_weight
                for key, value in accum.items()
            }

        frame = pd.DataFrame(
            {
                "feature": list(accum.keys()),
                "importance": list(accum.values()),
            }
        )

        return frame.sort_values(
            "importance",
            ascending=False,
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        return {
            "model_type": "Adaptive Weighted Ensemble",
            "components": list(
                self.component_names
            ),
            "component_weights": dict(
                self.component_weights
            ),
            "trained_rows": int(
                self.trained_rows
            ),
            "max_training_rows": (
                int(self.max_training_rows)
                if self.max_training_rows is not None
                else None
            ),
            "training_window_rows": int(self.training_window_),
            "training_start": self.training_start_,
            "training_end": self.training_end_,
            "training_data_fingerprint": self.training_data_fingerprint_,
            "model_fingerprint": self.model_fingerprint_,
            "feature_count": int(
                len(self.feature_columns)
            ),
            "horizon": int(
                self.horizon
            ),
            "probability_threshold": float(
                self.probability_threshold
            ),
            "calibration_enabled": bool(
                self.calibration_enabled
            ),
            "validation_method": (
                "Chronological out-of-fold validation"
            ),
            "validation_folds": int(
                len(self.oof_folds_)
            ),
            "validation_accuracy": (
                float(self.validation_accuracy)
                if self.validation_accuracy is not None
                else None
            ),
            "validation_precision": (
                float(self.validation_precision)
                if self.validation_precision is not None
                else None
            ),
            "validation_recall": (
                float(self.validation_recall)
                if self.validation_recall is not None
                else None
            ),
            "validation_f1": (
                float(self.validation_f1)
                if self.validation_f1 is not None
                else None
            ),
            "validation_roc_auc": (
                float(self.validation_roc_auc)
                if self.validation_roc_auc is not None
                and np.isfinite(self.validation_roc_auc)
                else None
            ),
            "validation_brier": (
                float(self.validation_brier)
                if self.validation_brier is not None
                else None
            ),
            "validation_log_loss": (
                float(self.validation_log_loss)
                if self.validation_log_loss is not None
                else None
            ),
            "training_accuracy": (
                float(self.training_accuracy)
                if self.training_accuracy is not None
                else None
            ),
            "class_balance": dict(
                self.class_balance
            ),
            "component_validation_metrics": dict(
                self.component_validation_metrics_
            ),
        }


# ----------------------------------------------------------------------
# Backward-compatible names used by the application
# ----------------------------------------------------------------------

GradientBoostingSwingClassifier = EnsembleSwingClassifier
SwingClassifier = EnsembleSwingClassifier

__all__ = [
    "EnsembleSwingClassifier",
    "GradientBoostingSwingClassifier",
    "SwingClassifier",
]
