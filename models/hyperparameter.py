"""
Controlled hyperparameter optimization for AI Swing Analyser.

The optimizer is designed for financial time-series research.

It deliberately avoids ordinary random K-fold cross-validation because
randomly mixing historical observations can leak temporal information.

Design:
    parameter candidates
          ↓
    walk-forward validation
          ↓
    out-of-sample fold metrics
          ↓
    stability-aware score
          ↓
    best configuration

IMPORTANT
---------
Hyperparameter optimization must happen before the final untouched
holdout is evaluated.

The final holdout must NEVER be used to select hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import classification_metrics
from .preprocessing import (
    PreprocessorConfig,
    SafePreprocessor,
)
from .splitter import purged_walk_forward_splits


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass
class HyperparameterSearchConfig:
    """
    Configuration for constrained time-series hyperparameter search.
    """

    n_splits: int = 5

    train_size: Optional[int] = None

    validation_size: Optional[int] = None

    gap: int = 0

    embargo: int = 0

    expanding: bool = True

    minimum_train_samples: int = 100

    minimum_validation_samples: int = 30

    maximum_trials: int = 50

    minimum_accuracy: float = 0.95

    maximum_accuracy_std: float = 0.10

    stability_weight: float = 0.25

    brier_weight: float = 0.15

    accuracy_weight: float = 0.60

    random_state: int = 42

    def __post_init__(self) -> None:
        if self.n_splits < 1:
            raise ValueError(
                "n_splits must be at least 1."
            )

        if self.train_size is not None and self.train_size < 1:
            raise ValueError(
                "train_size must be positive."
            )

        if self.validation_size is not None:
            if self.validation_size < 1:
                raise ValueError(
                    "validation_size must be positive."
                )

        if self.gap < 0:
            raise ValueError(
                "gap cannot be negative."
            )

        if self.embargo < 0:
            raise ValueError(
                "embargo cannot be negative."
            )

        if self.minimum_train_samples < 1:
            raise ValueError(
                "minimum_train_samples must be positive."
            )

        if self.minimum_validation_samples < 1:
            raise ValueError(
                "minimum_validation_samples must be positive."
            )

        if self.maximum_trials < 1:
            raise ValueError(
                "maximum_trials must be positive."
            )

        if not 0.0 <= self.minimum_accuracy <= 1.0:
            raise ValueError(
                "minimum_accuracy must be between 0 and 1."
            )

        if self.maximum_accuracy_std < 0:
            raise ValueError(
                "maximum_accuracy_std cannot be negative."
            )

        weights = (
            self.accuracy_weight,
            self.stability_weight,
            self.brier_weight,
        )

        if any(weight < 0 for weight in weights):
            raise ValueError(
                "Search weights cannot be negative."
            )

        if sum(weights) <= 0:
            raise ValueError(
                "At least one search weight must be positive."
            )


# ----------------------------------------------------------------------
# Trial result
# ----------------------------------------------------------------------


@dataclass
class HyperparameterTrial:
    """
    Result of one hyperparameter configuration.
    """

    trial_number: int

    parameters: Dict[str, Any]

    fold_accuracies: List[float]

    fold_brier_scores: List[float]

    mean_accuracy: float

    median_accuracy: float

    minimum_accuracy: float

    accuracy_std: float

    mean_brier_score: float

    stability_score: float

    objective_score: float

    passed_accuracy_gate: bool

    passed_stability_gate: bool

    passed: bool

    error: Optional[str] = None

    notes: List[str] = field(
        default_factory=list
    )

    @property
    def successful(self) -> bool:
        return self.error is None


@dataclass
class HyperparameterSearchResult:
    """
    Complete hyperparameter search result.
    """

    trials: List[HyperparameterTrial]

    best_parameters: Optional[Dict[str, Any]]

    best_trial_number: Optional[int]

    best_objective_score: Optional[float]

    search_config: HyperparameterSearchConfig

    feature_names: Sequence[str]

    target_column: str

    model_name: str

    notes: List[str] = field(
        default_factory=list
    )

    @property
    def successful_trials(self) -> List[HyperparameterTrial]:
        return [
            trial
            for trial in self.trials
            if trial.successful
        ]

    @property
    def passed_trials(self) -> List[HyperparameterTrial]:
        return [
            trial
            for trial in self.trials
            if trial.passed
        ]

    @property
    def found_valid_configuration(self) -> bool:
        return self.best_parameters is not None

    def leaderboard(self) -> pd.DataFrame:
        """Return all trials sorted by objective score."""

        records = []

        for trial in self.trials:
            records.append(
                {
                    "trial": trial.trial_number,
                    "parameters": str(
                        trial.parameters
                    ),
                    "mean_accuracy": (
                        trial.mean_accuracy
                    ),
                    "median_accuracy": (
                        trial.median_accuracy
                    ),
                    "minimum_accuracy": (
                        trial.minimum_accuracy
                    ),
                    "accuracy_std": (
                        trial.accuracy_std
                    ),
                    "mean_brier_score": (
                        trial.mean_brier_score
                    ),
                    "stability_score": (
                        trial.stability_score
                    ),
                    "objective_score": (
                        trial.objective_score
                    ),
                    "passed_accuracy_gate": (
                        trial.passed_accuracy_gate
                    ),
                    "passed_stability_gate": (
                        trial.passed_stability_gate
                    ),
                    "passed": trial.passed,
                    "error": trial.error,
                }
            )

        if not records:
            return pd.DataFrame()

        return (
            pd.DataFrame(records)
            .sort_values(
                "objective_score",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )


# ----------------------------------------------------------------------
# Main optimizer
# ----------------------------------------------------------------------


class TimeSeriesHyperparameterOptimizer:
    """
    Constrained hyperparameter optimizer using walk-forward validation.

    The optimizer accepts a model factory:

        model_factory(parameters) -> unfitted model

    and evaluates each configuration only on future observations relative
    to its training fold.
    """

    def __init__(
        self,
        config: Optional[
            HyperparameterSearchConfig
        ] = None,
        preprocessor_config: Optional[
            PreprocessorConfig
        ] = None,
    ) -> None:
        self.config = (
            config
            or HyperparameterSearchConfig()
        )

        self.preprocessor_config = (
            preprocessor_config
            or PreprocessorConfig()
        )

    # ------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------

    def search(
        self,
        data: pd.DataFrame,
        *,
        target_column: str,
        parameter_grid: Mapping[
            str,
            Sequence[Any],
        ],
        model_factory: Callable[
            [Mapping[str, Any]],
            Any,
        ],
        model_name: str = "model",
        feature_columns: Optional[
            Sequence[str]
        ] = None,
    ) -> HyperparameterSearchResult:
        """
        Run a constrained parameter search.

        The final holdout must not be included in ``data`` during this
        search.
        """

        frame = self._prepare_data(
            data=data,
            target_column=target_column,
            feature_columns=feature_columns,
        )

        features = self._resolve_features(
            frame,
            target_column,
            feature_columns,
        )

        candidates = self._build_parameter_candidates(
            parameter_grid
        )

        if len(candidates) > (
            self.config.maximum_trials
        ):
            candidates = candidates[
                : self.config.maximum_trials
            ]

        trials: List[
            HyperparameterTrial
        ] = []

        for trial_number, parameters in enumerate(
            candidates,
            start=1,
        ):
            try:
                trial = self._evaluate_trial(
                    frame=frame,
                    target_column=target_column,
                    features=features,
                    parameters=parameters,
                    model_factory=model_factory,
                    trial_number=trial_number,
                )

            except Exception as exc:
                trial = HyperparameterTrial(
                    trial_number=trial_number,
                    parameters=dict(
                        parameters
                    ),
                    fold_accuracies=[],
                    fold_brier_scores=[],
                    mean_accuracy=0.0,
                    median_accuracy=0.0,
                    minimum_accuracy=0.0,
                    accuracy_std=1.0,
                    mean_brier_score=1.0,
                    stability_score=0.0,
                    objective_score=-np.inf,
                    passed_accuracy_gate=False,
                    passed_stability_gate=False,
                    passed=False,
                    error=str(exc),
                    notes=[
                        "Trial failed during evaluation."
                    ],
                )

            trials.append(trial)

        successful = [
            trial
            for trial in trials
            if trial.successful
        ]

        valid = [
            trial
            for trial in successful
            if trial.passed
        ]

        # Prefer configurations that actually pass the research gates.
        candidates_for_selection = (
            valid
            if valid
            else successful
        )

        if candidates_for_selection:
            best = max(
                candidates_for_selection,
                key=lambda trial: (
                    trial.objective_score
                ),
            )

            best_parameters = dict(
                best.parameters
            )

            best_trial_number = (
                best.trial_number
            )

            best_objective_score = (
                best.objective_score
            )

        else:
            best_parameters = None
            best_trial_number = None
            best_objective_score = None

        notes = [
            (
                "Hyperparameters were evaluated using chronological "
                "walk-forward validation."
            ),
            (
                "The final untouched holdout must not be used during "
                "this search."
            ),
        ]

        if valid:
            notes.append(
                "At least one configuration passed the configured "
                "research gates."
            )
        elif successful:
            notes.append(
                "No configuration passed all configured gates; "
                "best successful configuration is reported."
            )
        else:
            notes.append(
                "No successful hyperparameter trials were completed."
            )

        return HyperparameterSearchResult(
            trials=trials,
            best_parameters=best_parameters,
            best_trial_number=best_trial_number,
            best_objective_score=best_objective_score,
            search_config=self.config,
            feature_names=features,
            target_column=target_column,
            model_name=model_name,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Trial evaluation
    # ------------------------------------------------------------------

    def _evaluate_trial(
        self,
        *,
        frame: pd.DataFrame,
        target_column: str,
        features: Sequence[str],
        parameters: Mapping[str, Any],
        model_factory: Callable[
            [Mapping[str, Any]],
            Any,
        ],
        trial_number: int,
    ) -> HyperparameterTrial:
        target = frame[
            target_column
        ]

        splits = self._create_splits(
            frame
        )

        fold_accuracies: List[float] = []
        fold_brier_scores: List[float] = []

        for train_indices, validation_indices in splits:
            train = frame.iloc[
                train_indices
            ]

            validation = frame.iloc[
                validation_indices
            ]

            # Strict temporal safety.
            if train.index.max() >= validation.index.min():
                raise RuntimeError(
                    "Temporal leakage detected in hyperparameter fold."
                )

            X_train = train[
                list(features)
            ]

            X_validation = validation[
                list(features)
            ]

            y_train = target.iloc[
                train_indices
            ]

            y_validation = target.iloc[
                validation_indices
            ]

            # ----------------------------------------------------------
            # Fresh preprocessor per fold
            # ----------------------------------------------------------

            preprocessor = SafePreprocessor(
                self.preprocessor_config
            )

            X_train_transformed = (
                preprocessor.fit_transform(
                    X_train
                )
            )

            X_validation_transformed = (
                preprocessor.transform(
                    X_validation
                )
            )

            # ----------------------------------------------------------
            # Fresh model per fold
            # ----------------------------------------------------------

            model = model_factory(
                parameters
            )

            if not hasattr(
                model,
                "fit",
            ):
                raise TypeError(
                    "model_factory must return an object with fit()."
                )

            model.fit(
                X_train_transformed,
                y_train,
            )

            y_pred = np.asarray(
                model.predict(
                    X_validation_transformed
                )
            )

            y_probability = (
                self._predict_probability(
                    model,
                    X_validation_transformed,
                )
            )

            metrics = classification_metrics(
                y_validation,
                y_pred,
                y_probability,
            )

            metrics_dict = (
                self._metrics_to_dict(
                    metrics
                )
            )

            fold_accuracies.append(
                float(
                    metrics_dict.get(
                        "accuracy",
                        0.0,
                    )
                )
            )

            fold_brier_scores.append(
                float(
                    metrics_dict.get(
                        "brier_score",
                        1.0,
                    )
                )
            )

        return self._build_trial(
            trial_number=trial_number,
            parameters=parameters,
            fold_accuracies=fold_accuracies,
            fold_brier_scores=fold_brier_scores,
        )

    # ------------------------------------------------------------------
    # Trial scoring
    # ------------------------------------------------------------------

    def _build_trial(
        self,
        *,
        trial_number: int,
        parameters: Mapping[str, Any],
        fold_accuracies: Sequence[float],
        fold_brier_scores: Sequence[float],
    ) -> HyperparameterTrial:
        if not fold_accuracies:
            raise ValueError(
                "No fold accuracies were produced."
            )

        accuracies = np.asarray(
            fold_accuracies,
            dtype=float,
        )

        briers = np.asarray(
            fold_brier_scores,
            dtype=float,
        )

        mean_accuracy = float(
            accuracies.mean()
        )

        median_accuracy = float(
            np.median(accuracies)
        )

        minimum_accuracy = float(
            accuracies.min()
        )

        accuracy_std = float(
            accuracies.std()
        )

        mean_brier = float(
            briers.mean()
        )

        # Stability score:
        #
        #   1.0 = perfectly stable
        #   0.0 = highly unstable
        #
        # This is a diagnostic score rather than a probability.
        stability_score = float(
            max(
                0.0,
                min(
                    1.0,
                    1.0
                    - (
                        accuracy_std
                        / 0.20
                    ),
                ),
            )
        )

        # Lower Brier score is better.
        brier_score = max(
            0.0,
            min(
                1.0,
                1.0 - mean_brier,
            ),
        )

        objective_score = float(
            (
                self.config.accuracy_weight
                * mean_accuracy
            )
            + (
                self.config.stability_weight
                * stability_score
            )
            + (
                self.config.brier_weight
                * brier_score
            )
        )

        passed_accuracy_gate = (
            mean_accuracy
            >= self.config.minimum_accuracy
        )

        passed_stability_gate = (
            accuracy_std
            <= self.config.maximum_accuracy_std
        )

        passed = (
            passed_accuracy_gate
            and passed_stability_gate
        )

        notes: List[str] = []

        if passed_accuracy_gate:
            notes.append(
                "Mean accuracy passed."
            )
        else:
            notes.append(
                "Mean accuracy failed."
            )

        if passed_stability_gate:
            notes.append(
                "Fold accuracy stability passed."
            )
        else:
            notes.append(
                "Fold accuracy stability failed."
            )

        return HyperparameterTrial(
            trial_number=trial_number,
            parameters=dict(
                parameters
            ),
            fold_accuracies=list(
                fold_accuracies
            ),
            fold_brier_scores=list(
                fold_brier_scores
            ),
            mean_accuracy=mean_accuracy,
            median_accuracy=median_accuracy,
            minimum_accuracy=minimum_accuracy,
            accuracy_std=accuracy_std,
            mean_brier_score=mean_brier,
            stability_score=stability_score,
            objective_score=objective_score,
            passed_accuracy_gate=(
                passed_accuracy_gate
            ),
            passed_stability_gate=(
                passed_stability_gate
            ),
            passed=passed,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Split creation
    # ------------------------------------------------------------------

    def _create_splits(
        self,
        frame: pd.DataFrame,
    ):
        splits = purged_walk_forward_splits(
            frame,
            n_splits=self.config.n_splits,
            train_size=self.config.train_size,
            test_size=self.config.validation_size,
            gap=self.config.gap,
            embargo=self.config.embargo,
            expanding=self.config.expanding,
        )

        validated = []

        for train_indices, validation_indices in splits:
            train_indices = np.asarray(
                train_indices,
                dtype=int,
            )

            validation_indices = np.asarray(
                validation_indices,
                dtype=int,
            )

            if len(train_indices) < (
                self.config.minimum_train_samples
            ):
                continue

            if len(validation_indices) < (
                self.config.minimum_validation_samples
            ):
                continue

            validated.append(
                (
                    train_indices,
                    validation_indices,
                )
            )

        if not validated:
            raise ValueError(
                "No valid walk-forward folds were produced."
            )

        return validated

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    @staticmethod
    def _build_parameter_candidates(
        parameter_grid: Mapping[
            str,
            Sequence[Any],
        ],
    ) -> List[Dict[str, Any]]:
        if not parameter_grid:
            raise ValueError(
                "parameter_grid cannot be empty."
            )

        names = list(
            parameter_grid.keys()
        )

        values = []

        for name in names:
            candidates = list(
                parameter_grid[name]
            )

            if not candidates:
                raise ValueError(
                    f"No candidates supplied for '{name}'."
                )

            values.append(
                candidates
            )

        combinations = []

        for combination in product(
            *values
        ):
            combinations.append(
                {
                    name: value
                    for name, value in zip(
                        names,
                        combination,
                    )
                }
            )

        return combinations

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_data(
        *,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[
            Sequence[str]
        ],
    ) -> pd.DataFrame:
        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        if target_column not in data.columns:
            raise ValueError(
                f"Target column '{target_column}' was not found."
            )

        frame = data.copy()

        if not isinstance(
            frame.index,
            pd.DatetimeIndex,
        ):
            raise TypeError(
                "Hyperparameter search requires a DatetimeIndex."
            )

        frame = frame.sort_index()

        if frame.index.has_duplicates:
            raise ValueError(
                "Duplicate timestamps detected."
            )

        if feature_columns is not None:
            suspicious = [
                column
                for column in feature_columns
                if (
                    "future" in column.lower()
                    or "target" in column.lower()
                    or "direction" in column.lower()
                )
            ]

            if suspicious:
                raise ValueError(
                    "Potential target leakage in feature columns: "
                    f"{suspicious}"
                )

        frame = frame.dropna(
            subset=[target_column]
        )

        return frame

    @staticmethod
    def _resolve_features(
        frame: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[
            Sequence[str]
        ],
    ) -> List[str]:
        if feature_columns is not None:
            features = list(
                feature_columns
            )

            missing = [
                feature
                for feature in features
                if feature not in frame.columns
            ]

            if missing:
                raise ValueError(
                    f"Missing feature columns: {missing}"
                )

            return features

        features = []

        for column in frame.columns:
            if column == target_column:
                continue

            lower = column.lower()

            if (
                "future" in lower
                or "target" in lower
                or "direction" in lower
            ):
                continue

            if pd.api.types.is_numeric_dtype(
                frame[column]
            ):
                features.append(column)

        if not features:
            raise ValueError(
                "No numeric features found."
            )

        return features

    # ------------------------------------------------------------------
    # Probability
    # ------------------------------------------------------------------

    @staticmethod
    def _predict_probability(
        model: Any,
        X,
    ) -> np.ndarray:
        if hasattr(
            model,
            "predict_proba",
        ):
            probability = np.asarray(
                model.predict_proba(X),
                dtype=float,
            )

            if probability.ndim == 2:
                if probability.shape[1] == 1:
                    return probability[:, 0]

                return probability[:, -1]

            return probability.reshape(-1)

        # Not a calibrated probability.
        # This is retained only for metric compatibility.
        return np.asarray(
            model.predict(X),
            dtype=float,
        ).reshape(-1)

    # ------------------------------------------------------------------
    # Metric conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _metrics_to_dict(
        metrics: Any,
    ) -> Dict[str, float]:
        if isinstance(
            metrics,
            dict,
        ):
            source = metrics

        elif hasattr(
            metrics,
            "__dataclass_fields__",
        ):
            source = {
                name: getattr(
                    metrics,
                    name,
                )
                for name in metrics.__dataclass_fields__
            }

        else:
            source = {
                key: value
                for key, value in vars(
                    metrics
                ).items()
                if not key.startswith("_")
            }

        result = {}

        for key, value in source.items():
            if value is None:
                continue

            if isinstance(
                value,
                (
                    int,
                    float,
                    np.integer,
                    np.floating,
                ),
            ):
                if np.isfinite(value):
                    result[key] = float(
                        value
                    )

        return result


# ----------------------------------------------------------------------
# Convenience helpers
# ----------------------------------------------------------------------


def grid_search_time_series(
    data: pd.DataFrame,
    *,
    target_column: str,
    parameter_grid: Mapping[
        str,
        Sequence[Any],
    ],
    model_factory: Callable[
        [Mapping[str, Any]],
        Any,
    ],
    config: Optional[
        HyperparameterSearchConfig
    ] = None,
    preprocessor_config: Optional[
        PreprocessorConfig
    ] = None,
    model_name: str = "model",
    feature_columns: Optional[
        Sequence[str]
    ] = None,
) -> HyperparameterSearchResult:
    """
    Convenience wrapper for time-series hyperparameter search.
    """

    optimizer = TimeSeriesHyperparameterOptimizer(
        config=config,
        preprocessor_config=preprocessor_config,
    )

    return optimizer.search(
        data,
        target_column=target_column,
        parameter_grid=parameter_grid,
        model_factory=model_factory,
        model_name=model_name,
        feature_columns=feature_columns,
    )


def search_summary(
    result: HyperparameterSearchResult,
) -> Dict[str, Any]:
    """Return a compact search summary."""

    return {
        "model": result.model_name,
        "target": result.target_column,
        "trials": len(result.trials),
        "successful_trials": len(
            result.successful_trials
        ),
        "passed_trials": len(
            result.passed_trials
        ),
        "best_parameters": (
            result.best_parameters
        ),
        "best_trial": (
            result.best_trial_number
        ),
        "best_objective_score": (
            result.best_objective_score
        ),
        "found_valid_configuration": (
            result.found_valid_configuration
        ),
    }


__all__ = [
    "HyperparameterSearchConfig",
    "HyperparameterTrial",
    "HyperparameterSearchResult",
    "TimeSeriesHyperparameterOptimizer",
    "grid_search_time_series",
    "search_summary",
]
