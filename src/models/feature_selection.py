"""
Feature selection and feature-stability analysis.

The purpose of this module is to reduce overfitting by identifying
features that are:

    - stable across time
    - repeatedly useful across validation folds
    - sufficiently non-redundant
    - not dominated by missing values
    - not constant or near-constant

IMPORTANT
---------
Feature selection must be performed using training/validation data only
during model development.

The final untouched holdout must never influence feature selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass
class FeatureSelectionConfig:
    """
    Configuration for feature selection.

    Defaults are deliberately conservative.
    """

    minimum_variance: float = 1e-12

    maximum_missing_fraction: float = 0.40

    maximum_pairwise_correlation: float = 0.95

    minimum_importance: float = 0.0

    minimum_fold_frequency: float = 0.60

    maximum_features: Optional[int] = None

    minimum_features: int = 10

    use_absolute_correlation: bool = True

    prefer_stable_features: bool = True

    def __post_init__(self) -> None:
        if self.minimum_variance < 0:
            raise ValueError(
                "minimum_variance cannot be negative."
            )

        if not 0.0 <= self.maximum_missing_fraction <= 1.0:
            raise ValueError(
                "maximum_missing_fraction must be between 0 and 1."
            )

        if not 0.0 <= self.maximum_pairwise_correlation <= 1.0:
            raise ValueError(
                "maximum_pairwise_correlation must be between 0 and 1."
            )

        if self.minimum_importance < 0:
            raise ValueError(
                "minimum_importance cannot be negative."
            )

        if not 0.0 <= self.minimum_fold_frequency <= 1.0:
            raise ValueError(
                "minimum_fold_frequency must be between 0 and 1."
            )

        if self.maximum_features is not None:
            if self.maximum_features < 1:
                raise ValueError(
                    "maximum_features must be positive."
                )

        if self.minimum_features < 1:
            raise ValueError(
                "minimum_features must be positive."
            )


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------


@dataclass
class FeatureStatistics:
    """Descriptive statistics for one feature."""

    feature: str

    variance: float

    missing_fraction: float

    unique_values: int

    mean: float

    std: float

    minimum: float

    maximum: float

    valid: bool

    reasons: List[str] = field(
        default_factory=list
    )


@dataclass
class FeatureStability:
    """
    Stability information for one feature across model folds.
    """

    feature: str

    folds_present: int

    total_folds: int

    fold_frequency: float

    mean_importance: float

    median_importance: float

    importance_std: float

    importance_cv: float

    importance_rank_mean: float

    importance_rank_std: float

    stable: bool


@dataclass
class FeatureSelectionResult:
    """
    Complete feature-selection result.
    """

    selected_features: List[str]

    rejected_features: List[str]

    statistics: Dict[str, FeatureStatistics]

    stability: Dict[str, FeatureStability]

    correlation_pairs: List[tuple[str, str, float]]

    selection_scores: Dict[str, float]

    feature_rank: List[str]

    notes: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, object] = field(
        default_factory=dict
    )

    def summary(self) -> Dict[str, object]:
        return {
            "total_features": len(
                self.statistics
            ),
            "selected_features": len(
                self.selected_features
            ),
            "rejected_features": len(
                self.rejected_features
            ),
            "stable_features": sum(
                stability.stable
                for stability in self.stability.values()
            ),
            "correlated_pairs": len(
                self.correlation_pairs
            ),
        }

    def selected_frame(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return only selected feature columns."""

        missing = [
            feature
            for feature in self.selected_features
            if feature not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Selected features missing from data: {missing}"
            )

        return data[
            self.selected_features
        ].copy()


# ----------------------------------------------------------------------
# Main selector
# ----------------------------------------------------------------------


class FeatureSelector:
    """
    Feature-selection and stability engine.

    This class does not train a model itself.

    It can consume importance values from multiple walk-forward folds
    and determine which features remain stable over time.
    """

    def __init__(
        self,
        config: Optional[
            FeatureSelectionConfig
        ] = None,
    ) -> None:
        self.config = (
            config
            or FeatureSelectionConfig()
        )

    # ------------------------------------------------------------------
    # Basic statistics
    # ------------------------------------------------------------------

    def calculate_statistics(
        self,
        data: pd.DataFrame,
        *,
        feature_columns: Optional[
            Sequence[str]
        ] = None,
    ) -> Dict[str, FeatureStatistics]:
        """
        Calculate descriptive statistics for features.
        """

        features = self._resolve_features(
            data,
            feature_columns,
        )

        statistics: Dict[
            str,
            FeatureStatistics,
        ] = {}

        for feature in features:
            series = pd.to_numeric(
                data[feature],
                errors="coerce",
            )

            missing_fraction = float(
                series.isna().mean()
            )

            valid_values = series.dropna()

            if valid_values.empty:
                statistics[feature] = (
                    FeatureStatistics(
                        feature=feature,
                        variance=0.0,
                        missing_fraction=1.0,
                        unique_values=0,
                        mean=0.0,
                        std=0.0,
                        minimum=0.0,
                        maximum=0.0,
                        valid=False,
                        reasons=[
                            "No valid numeric observations."
                        ],
                    )
                )

                continue

            variance = float(
                valid_values.var(
                    ddof=0
                )
            )

            standard_deviation = float(
                valid_values.std(
                    ddof=0
                )
            )

            reasons: List[str] = []

            if (
                missing_fraction
                > self.config.maximum_missing_fraction
            ):
                reasons.append(
                    "Excessive missing values."
                )

            if (
                variance
                <= self.config.minimum_variance
            ):
                reasons.append(
                    "Near-zero variance."
                )

            valid = len(reasons) == 0

            statistics[feature] = (
                FeatureStatistics(
                    feature=feature,
                    variance=variance,
                    missing_fraction=missing_fraction,
                    unique_values=int(
                        valid_values.nunique()
                    ),
                    mean=float(
                        valid_values.mean()
                    ),
                    std=standard_deviation,
                    minimum=float(
                        valid_values.min()
                    ),
                    maximum=float(
                        valid_values.max()
                    ),
                    valid=valid,
                    reasons=reasons,
                )
            )

        return statistics

    # ------------------------------------------------------------------
    # Correlation
    # ------------------------------------------------------------------

    def find_correlated_features(
        self,
        data: pd.DataFrame,
        *,
        feature_columns: Optional[
            Sequence[str]
        ] = None,
    ) -> List[tuple[str, str, float]]:
        """
        Identify highly correlated feature pairs.

        This is used to reduce redundant information.

        Correlation does not imply that one feature should automatically
        be deleted. The final selection considers feature stability and
        importance as well.
        """

        features = self._resolve_features(
            data,
            feature_columns,
        )

        if len(features) < 2:
            return []

        numeric = data[
            features
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        correlation = numeric.corr(
            method="spearman"
        )

        pairs: List[
            tuple[str, str, float]
        ] = []

        for i, first in enumerate(
            features
        ):
            for second in features[
                i + 1:
            :]:
                value = correlation.loc[
                    first,
                    second,
                ]

                if pd.isna(value):
                    continue

                comparison = (
                    abs(value)
                    if self.config.use_absolute_correlation
                    else value
                )

                if (
                    comparison
                    >= self.config.maximum_pairwise_correlation
                ):
                    pairs.append(
                        (
                            first,
                            second,
                            float(value),
                        )
                    )

        return pairs

    # ------------------------------------------------------------------
    # Fold importance stability
    # ------------------------------------------------------------------

    def calculate_stability(
        self,
        fold_importances: Sequence[
            Mapping[str, float]
        ],
    ) -> Dict[str, FeatureStability]:
        """
        Calculate feature importance stability across folds.

        Parameters
        ----------
        fold_importances:
            One mapping per walk-forward fold:

                {
                    "RSI_14": 0.15,
                    "EMA_20": 0.11,
                    ...
                }

        Features absent from a fold are treated as zero importance.
        """

        if not fold_importances:
            return {}

        all_features = sorted(
            {
                feature
                for importance in fold_importances
                for feature in importance
            }
        )

        total_folds = len(
            fold_importances
        )

        stability: Dict[
            str,
            FeatureStability,
        ] = {}

        for feature in all_features:
            values = np.asarray(
                [
                    self._safe_importance(
                        importance.get(
                            feature,
                            0.0,
                        )
                    )
                    for importance in fold_importances
                ],
                dtype=float,
            )

            present = values > 0

            folds_present = int(
                present.sum()
            )

            frequency = (
                folds_present
                / total_folds
            )

            mean_importance = float(
                values.mean()
            )

            median_importance = float(
                np.median(values)
            )

            importance_std = float(
                values.std()
            )

            if abs(mean_importance) <= 1e-12:
                importance_cv = float(
                    "inf"
                )
            else:
                importance_cv = (
                    importance_std
                    / abs(mean_importance)
                )

            ranks = []

            for importance in fold_importances:
                ordered = sorted(
                    importance.items(),
                    key=lambda item: abs(
                        float(item[1])
                    ),
                    reverse=True,
                )

                rank_lookup = {
                    name: rank + 1
                    for rank, (
                        name,
                        _,
                    ) in enumerate(ordered)
                }

                ranks.append(
                    rank_lookup.get(
                        feature,
                        len(
                            ordered
                        ) + 1,
                    )
                )

            rank_array = np.asarray(
                ranks,
                dtype=float,
            )

            rank_mean = float(
                rank_array.mean()
            )

            rank_std = float(
                rank_array.std()
            )

            stable = (
                frequency
                >= self.config.minimum_fold_frequency
                and mean_importance
                >= self.config.minimum_importance
            )

            stability[feature] = (
                FeatureStability(
                    feature=feature,
                    folds_present=folds_present,
                    total_folds=total_folds,
                    fold_frequency=frequency,
                    mean_importance=mean_importance,
                    median_importance=median_importance,
                    importance_std=importance_std,
                    importance_cv=importance_cv,
                    importance_rank_mean=rank_mean,
                    importance_rank_std=rank_std,
                    stable=stable,
                )
            )

        return stability

    # ------------------------------------------------------------------
    # Complete selection
    # ------------------------------------------------------------------

    def select(
        self,
        data: pd.DataFrame,
        *,
        feature_columns: Optional[
            Sequence[str]
        ] = None,
        fold_importances: Optional[
            Sequence[Mapping[str, float]]
        ] = None,
    ) -> FeatureSelectionResult:
        """
        Perform complete feature selection.

        The selection process is:

            1. basic data-quality screening
            2. missingness screening
            3. variance screening
            4. correlation analysis
            5. fold-importance stability
            6. ranking
            7. final feature selection
        """

        features = self._resolve_features(
            data,
            feature_columns,
        )

        statistics = self.calculate_statistics(
            data,
            feature_columns=features,
        )

        correlation_pairs = (
            self.find_correlated_features(
                data,
                feature_columns=features,
            )
        )

        if fold_importances:
            stability = self.calculate_stability(
                fold_importances
            )
        else:
            stability = {}

        rejected: set[str] = set()

        # --------------------------------------------------------------
        # Basic-quality rejection
        # --------------------------------------------------------------

        for feature, stats in statistics.items():
            if not stats.valid:
                rejected.add(feature)

        # --------------------------------------------------------------
        # Correlation-based redundancy
        # --------------------------------------------------------------

        correlation_rejection = (
            self._select_correlation_representatives(
                correlation_pairs,
                stability,
            )
        )

        rejected.update(
            correlation_rejection
        )

        # --------------------------------------------------------------
        # Stability-based rejection
        # --------------------------------------------------------------

        if fold_importances:
            for feature in features:
                feature_stability = stability.get(
                    feature
                )

                if feature_stability is None:
                    rejected.add(feature)
                    continue

                if not feature_stability.stable:
                    rejected.add(feature)

        # --------------------------------------------------------------
        # Rank surviving features
        # --------------------------------------------------------------

        scores = self._calculate_selection_scores(
            features=features,
            statistics=statistics,
            stability=stability,
        )

        ranked = sorted(
            features,
            key=lambda feature: scores.get(
                feature,
                0.0,
            ),
            reverse=True,
        )

        selected = [
            feature
            for feature in ranked
            if feature not in rejected
        ]

        # --------------------------------------------------------------
        # Maximum feature count
        # --------------------------------------------------------------

        if (
            self.config.maximum_features
            is not None
        ):
            selected = selected[
                : self.config.maximum_features
            ]

        # Avoid accidentally returning an empty feature set when the
        # dataset has enough valid features. The caller can explicitly
        # reject the result if minimum_features cannot be satisfied.
        if len(selected) < self.config.minimum_features:
            notes = [
                (
                    "Fewer than the configured minimum number of "
                    "stable features survived selection."
                )
            ]
        else:
            notes = [
                (
                    "Feature selection completed using temporal "
                    "stability and redundancy controls."
                )
            ]

        if fold_importances:
            notes.append(
                "Fold importance stability was included."
            )
        else:
            notes.append(
                "No fold importances supplied; temporal stability "
                "could not be evaluated."
            )

        notes.append(
            "Final holdout data must not influence this selection."
        )

        return FeatureSelectionResult(
            selected_features=selected,
            rejected_features=sorted(
                rejected
            ),
            statistics=statistics,
            stability=stability,
            correlation_pairs=correlation_pairs,
            selection_scores=scores,
            feature_rank=ranked,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Selection scoring
    # ------------------------------------------------------------------

    def _calculate_selection_scores(
        self,
        *,
        features: Sequence[str],
        statistics: Mapping[
            str,
            FeatureStatistics,
        ],
        stability: Mapping[
            str,
            FeatureStability,
        ],
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}

        for feature in features:
            stats = statistics.get(
                feature
            )

            if stats is None:
                scores[feature] = 0.0
                continue

            quality_score = 1.0

            quality_score *= max(
                0.0,
                1.0
                - stats.missing_fraction,
            )

            if stats.variance <= (
                self.config.minimum_variance
            ):
                quality_score = 0.0

            importance_score = 0.0
            stability_score = 1.0

            if feature in stability:
                feature_stability = stability[
                    feature
                ]

                importance_score = max(
                    0.0,
                    feature_stability.mean_importance,
                )

                stability_score = (
                    feature_stability.fold_frequency
                )

            if (
                self.config.prefer_stable_features
            ):
                score = (
                    quality_score
                    * (
                        0.30
                        + 0.40
                        * importance_score
                        + 0.30
                        * stability_score
                    )
                )
            else:
                score = (
                    quality_score
                    * (
                        0.50
                        + 0.50
                        * importance_score
                    )
                )

            scores[feature] = float(
                max(
                    0.0,
                    score,
                )
            )

        return scores

    # ------------------------------------------------------------------
    # Correlation representative selection
    # ------------------------------------------------------------------

    def _select_correlation_representatives(
        self,
        pairs: Sequence[
            tuple[str, str, float]
        ],
        stability: Mapping[
            str,
            FeatureStability,
        ],
    ) -> set[str]:
        """
        Select which feature to reject from highly correlated pairs.

        Prefer the feature with stronger temporal importance stability.
        """

        rejected: set[str] = set()

        for first, second, _ in pairs:
            first_stability = stability.get(
                first
            )

            second_stability = stability.get(
                second
            )

            if (
                first_stability is not None
                and second_stability is not None
            ):
                first_score = (
                    first_stability.mean_importance
                    * first_stability.fold_frequency
                )

                second_score = (
                    second_stability.mean_importance
                    * second_stability.fold_frequency
                )

                if first_score >= second_score:
                    rejected.add(second)
                else:
                    rejected.add(first)

            elif first_stability is not None:
                rejected.add(second)

            elif second_stability is not None:
                rejected.add(first)

            else:
                # Without importance information, keep the first feature
                # and reject the second deterministically.
                rejected.add(second)

        return rejected

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_features(
        data: pd.DataFrame,
        feature_columns: Optional[
            Sequence[str]
        ],
    ) -> List[str]:
        if not isinstance(
            data,
            pd.DataFrame,
        ):
            raise TypeError(
                "data must be a pandas DataFrame."
            )

        if feature_columns is not None:
            features = list(
                feature_columns
            )

            missing = [
                feature
                for feature in features
                if feature not in data.columns
            ]

            if missing:
                raise ValueError(
                    f"Feature columns missing from data: {missing}"
                )

            return features

        features = []

        for column in data.columns:
            if pd.api.types.is_numeric_dtype(
                data[column]
            ):
                features.append(column)

        if not features:
            raise ValueError(
                "No numeric feature columns found."
            )

        return features

    @staticmethod
    def _safe_importance(
        value: object,
    ) -> float:
        try:
            number = float(value)
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        if not np.isfinite(number):
            return 0.0

        return max(
            0.0,
            number,
        )


# ----------------------------------------------------------------------
# Standalone functions
# ----------------------------------------------------------------------


def feature_statistics_table(
    statistics: Mapping[
        str,
        FeatureStatistics,
    ],
) -> pd.DataFrame:
    """Convert feature statistics to a DataFrame."""

    records = []

    for feature, stats in statistics.items():
        records.append(
            {
                "feature": feature,
                "variance": stats.variance,
                "missing_fraction": (
                    stats.missing_fraction
                ),
                "unique_values": (
                    stats.unique_values
                ),
                "mean": stats.mean,
                "std": stats.std,
                "minimum": stats.minimum,
                "maximum": stats.maximum,
                "valid": stats.valid,
                "reasons": "; ".join(
                    stats.reasons
                ),
            }
        )

    return pd.DataFrame(records)


def feature_stability_table(
    stability: Mapping[
        str,
        FeatureStability,
    ],
) -> pd.DataFrame:
    """Convert feature stability results to a DataFrame."""

    records = []

    for feature, result in stability.items():
        records.append(
            {
                "feature": feature,
                "folds_present": (
                    result.folds_present
                ),
                "total_folds": (
                    result.total_folds
                ),
                "fold_frequency": (
                    result.fold_frequency
                ),
                "mean_importance": (
                    result.mean_importance
                ),
                "median_importance": (
                    result.median_importance
                ),
                "importance_std": (
                    result.importance_std
                ),
                "importance_cv": (
                    result.importance_cv
                ),
                "importance_rank_mean": (
                    result.importance_rank_mean
                ),
                "importance_rank_std": (
                    result.importance_rank_std
                ),
                "stable": result.stable,
            }
        )

    return pd.DataFrame(records)


def select_stable_features(
    data: pd.DataFrame,
    *,
    feature_columns: Optional[
        Sequence[str]
    ] = None,
    fold_importances: Optional[
        Sequence[Mapping[str, float]]
    ] = None,
    config: Optional[
        FeatureSelectionConfig
    ] = None,
) -> FeatureSelectionResult:
    """
    Convenience wrapper for feature selection.
    """

    selector = FeatureSelector(
        config=config
    )

    return selector.select(
        data,
        feature_columns=feature_columns,
        fold_importances=fold_importances,
    )


def feature_selection_summary(
    result: FeatureSelectionResult,
) -> Dict[str, object]:
    """Return a compact selection summary."""

    return result.summary()


__all__ = [
    "FeatureSelectionConfig",
    "FeatureStatistics",
    "FeatureStability",
    "FeatureSelectionResult",
    "FeatureSelector",
    "feature_statistics_table",
    "feature_stability_table",
    "select_stable_features",
    "feature_selection_summary",
]
