"""
Tests for the core machine-learning model layer.

These tests verify:
    - preprocessing is fit/used correctly
    - classifiers and regressors train successfully
    - range predictions remain ordered
    - probability calibration works
    - ensembles combine models deterministically
    - artifacts preserve feature schemas
    - model registry approval rules remain strict
    - prediction engine produces valid outputs
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.artifacts import (
    ArtifactMetadata,
    ModelArtifact,
    load_artifact,
    save_artifact,
)
from src.models.calibration import (
    ProbabilityCalibrator,
)
from src.models.classifier import (
    ClassifierConfig,
    DirectionClassifier,
)
from src.models.ensemble import (
    EnsembleConfig,
    ModelEnsemble,
)
from src.models.metrics import (
    classification_metrics,
    regression_metrics,
)
from src.models.preprocessing import (
    PreprocessorConfig,
    SafePreprocessor,
)
from src.models.range_model import (
    PriceRangeModel,
    RangeModelConfig,
)
from src.models.regressor import (
    RegressorConfig,
    ReturnRegressor,
)
from src.models.model_registry import (
    ModelRegistry,
    ModelRegistryEntry,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def model_data() -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    """
    Deterministic synthetic classification dataset.
    """

    rng = np.random.default_rng(
        42
    )

    n = 400

    x1 = rng.normal(
        0,
        1,
        n,
    )

    x2 = rng.normal(
        0,
        1,
        n,
    )

    x3 = rng.normal(
        0,
        1,
        n,
    )

    x4 = rng.normal(
        0,
        1,
        n,
    )

    x5 = rng.normal(
        0,
        1,
        n,
    )

    score = (
        1.5 * x1
        + 0.8 * x2
        - 0.4 * x3
        + 0.2 * x4
        + rng.normal(
            0,
            0.4,
            n,
        )
    )

    y = (
        score > 0
    ).astype(int)

    X = pd.DataFrame(
        {
            "feature_1": x1,
            "feature_2": x2,
            "feature_3": x3,
            "feature_4": x4,
            "feature_5": x5,
        }
    )

    return X, pd.Series(
        y,
        name="Direction_5",
    )


@pytest.fixture
def regression_data() -> tuple[
    pd.DataFrame,
    pd.Series,
]:
    rng = np.random.default_rng(
        123
    )

    n = 400

    x1 = rng.normal(
        0,
        1,
        n,
    )

    x2 = rng.normal(
        0,
        1,
        n,
    )

    x3 = rng.normal(
        0,
        1,
        n,
    )

    y = (
        0.03 * x1
        - 0.02 * x2
        + 0.01 * x3
        + rng.normal(
            0,
            0.005,
            n,
        )
    )

    X = pd.DataFrame(
        {
            "feature_1": x1,
            "feature_2": x2,
            "feature_3": x3,
        }
    )

    return X, pd.Series(
        y,
        name="Future_Close_Return_5",
    )


@pytest.fixture
def range_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    rng = np.random.default_rng(
        77
    )

    n = 300

    X = pd.DataFrame(
        {
            "feature_1": rng.normal(
                0,
                1,
                n,
            ),
            "feature_2": rng.normal(
                0,
                1,
                n,
            ),
            "feature_3": rng.normal(
                0,
                1,
                n,
            ),
        }
    )

    center = (
        0.02 * X["feature_1"]
        - 0.01 * X["feature_2"]
    )

    noise = np.abs(
        rng.normal(
            0.01,
            0.003,
            n,
        )
    )

    targets = pd.DataFrame(
        {
            "lower": center - noise,
            "median": center,
            "upper": center + noise,
        }
    )

    return X, targets


# ----------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------


def test_safe_preprocessor_fit_transform(
    model_data: tuple[
        pd.DataFrame,
        pd.Series,
    ],
) -> None:
    X, _ = model_data

    X = X.copy()

    X.loc[
        0,
        "feature_1",
    ] = np.nan

    preprocessor = SafePreprocessor(
        PreprocessorConfig()
    )

    transformed = preprocessor.fit_transform(
        X
    )

    assert transformed.shape[0] == len(
        X
    )

    assert transformed.shape[1] >= len(
        X.columns
    )

    assert np.isfinite(
        transformed
    ).all()


def test_preprocessor_transform_does_not_refit(
    model_data: tuple[
        pd.DataFrame,
        pd.Series,
    ],
) -> None:
    X, _ = model_data

    train = X.iloc[
        :300
    ]

    test = X.iloc[
        300:
    ]

    preprocessor = SafePreprocessor(
        PreprocessorConfig()
    )

    preprocessor.fit(
        train
    )

    feature_names_before = (
        preprocessor.get_feature_names()
    )

    transformed = preprocessor.transform(
        test
    )

    feature_names_after = (
        preprocessor.get_feature_names()
    )

    assert transformed.shape[0] == len(
        test
    )

    assert (
        feature_names_before
        == feature_names_after
    )


def test_preprocessor_rejects_non_numeric() -> None:
    X = pd.DataFrame(
        {
            "numeric": [1.0, 2.0, 3.0],
            "text": [
                "a",
                "b",
                "c",
            ],
        }
    )

    preprocessor = SafePreprocessor(
        PreprocessorConfig()
    )

    with pytest.raises(
        ValueError
    ):
        preprocessor.fit(
            X
        )


# ----------------------------------------------------------------------
# Classifier
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "algorithm",
    [
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    ],
)
def test_direction_classifier_algorithms(
    algorithm: str,
    model_data: tuple[
        pd.DataFrame,
        pd.Series,
    ],
) -> None:
    X, y = model_data

    model = DirectionClassifier(
        ClassifierConfig(
            algorithm=algorithm,
        )
    )

    model.fit(
        X,
        y,
    )

    predictions = model.predict(
        X
    )

    probabilities = model.predict_proba(
        X
    )

    assert len(
        predictions
    ) == len(
        X
    )

    assert probabilities.shape == (
        len(X),
        2,
    )

    assert np.isfinite(
        probabilities
    ).all()

    assert np.allclose(
        probabilities.sum(
            axis=1
        ),
        1.0,
        atol=1e-6,
    )


def test_classifier_probability_range(
    model_data: tuple[
        pd.DataFrame,
        pd.Series,
    ],
) -> None:
    X, y = model_data

    model = DirectionClassifier()

    model.fit(
        X,
        y,
    )

    probabilities = model.predict_proba(
        X
    )

    assert (
        probabilities >= 0
    ).all()

    assert (
        probabilities <= 1
    ).all()


def test_classifier_rejects_one_class() -> None:
    X = pd.DataFrame(
        {
            "a": np.arange(
                50.0
            )
        }
    )

    y = pd.Series(
        np.ones(
            50,
            dtype=int,
        )
    )

    model = DirectionClassifier()

    with pytest.raises(
        ValueError
    ):
        model.fit(
            X,
            y,
        )


# ----------------------------------------------------------------------
# Regression
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "algorithm",
    [
        "ridge",
        "random_forest",
        "gradient_boosting",
    ],
)
def test_return_regressor_algorithms(
    algorithm: str,
    regression_data: tuple[
        pd.DataFrame,
        pd.Series,
    ],
) -> None:
    X, y = regression_data

    model = ReturnRegressor(
        RegressorConfig(
            algorithm=algorithm,
        )
    )

    model.fit(
        X,
        y,
    )

    predictions = model.predict(
        X
    )

    assert len(
        predictions
    ) == len(
        X
    )

    assert np.isfinite(
        predictions
    ).all()


def test_regression_metrics_are_valid(
    regression_data: tuple[
        pd.DataFrame,
        pd.Series,
    ],
) -> None:
    X, y = regression_data

    model = ReturnRegressor()

    model.fit(
        X,
        y,
    )

    predictions = model.predict(
        X
    )

    metrics = regression_metrics(
        y,
        predictions,
    )

    assert metrics.mae >= 0
    assert metrics.rmse >= 0


# ----------------------------------------------------------------------
# Range model
# ----------------------------------------------------------------------


def test_range_model_predictions_are_ordered(
    range_data: tuple[
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    X, targets = range_data

    model = PriceRangeModel(
        RangeModelConfig()
    )

    model.fit(
        X,
        targets["lower"],
        targets["median"],
        targets["upper"],
    )

    predictions = model.predict_returns(
        X
    )

    assert len(
        predictions
    ) == len(
        X
    )

    assert (
        predictions["lower"]
        <= predictions["median"]
    ).all()

    assert (
        predictions["median"]
        <= predictions["upper"]
    ).all()


def test_range_model_can_convert_returns_to_prices(
    range_data: tuple[
        pd.DataFrame,
        pd.DataFrame,
    ],
) -> None:
    X, targets = range_data

    model = PriceRangeModel(
        RangeModelConfig()
    )

    model.fit(
        X,
        targets["lower"],
        targets["median"],
        targets["upper"],
    )

    current_price = 100.0

    result = model.predict_prices(
        X,
        current_price,
    )

    assert {
        "lower",
        "median",
        "upper",
    }.issubset(
        result.columns
    )

    assert (
        result["lower"]
        <= result["median"]
    ).all()

    assert (
        result["median"]
        <= result["upper"]
    ).all()


# ----------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------


def test_probability_calibration_sigmoid() -> None:
    rng = np.random.default_rng(
        42
    )

    y = rng.integers(
        0,
        2,
        300,
    )

    probabilities = np.clip(
        0.5
        + (
            y
            - 0.5
        )
        * 0.5
        + rng.normal(
            0,
            0.05,
            len(y),
        ),
        0.01,
        0.99,
    )

    calibrator = ProbabilityCalibrator()

    calibrator.fit(
        probabilities,
        y,
    )

    calibrated = calibrator.transform(
        probabilities
    )

    assert len(
        calibrated
    ) == len(
        probabilities
    )

    assert (
        calibrated >= 0
    ).all()

    assert (
        calibrated <= 1
    ).all()


def test_calibration_preserves_probability_order() -> None:
    probabilities = np.linspace(
        0.05,
        0.95,
        200,
    )

    y = (
        probabilities
        > 0.5
    ).astype(int)

    calibrator = ProbabilityCalibrator()

    calibrator.fit(
        probabilities,
        y,
    )

    calibrated = calibrator.transform(
        probabilities
    )

    differences = np.diff(
        calibrated
    )

    # Calibration should normally preserve monotonic ordering.
    assert (
        differences >= -1e-8
    ).all()


# ----------------------------------------------------------------------
# Ensemble
# ----------------------------------------------------------------------


def test_ensemble_weighted_probability() -> None:
    ensemble = ModelEnsemble(
        EnsembleConfig(
            method="weighted",
        )
    )

    probability_matrix = np.array(
        [
            [0.80, 0.20],
            [0.60, 0.40],
            [0.30, 0.70],
        ]
    )

    result = ensemble.combine_probabilities(
        probability_matrix,
        weights=np.array(
            [
                0.75,
                0.25,
            ]
        ),
    )

    expected = (
        0.75
        * probability_matrix[
            :,
            0,
        ]
        + 0.25
        * probability_matrix[
            :,
            1,
        ]
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_ensemble_equal_weights() -> None:
    ensemble = ModelEnsemble(
        EnsembleConfig(
            method="equal",
        )
    )

    matrix = np.array(
        [
            [0.8, 0.6, 0.4],
            [0.2, 0.5, 0.9],
        ]
    )

    result = ensemble.combine_probabilities(
        matrix
    )

    expected = np.mean(
        matrix,
        axis=1,
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_ensemble_weights_sum_to_one() -> None:
    ensemble = ModelEnsemble(
        EnsembleConfig(
            method="weighted",
        )
    )

    weights = ensemble.effective_weights(
        np.array(
            [
                0.2,
                0.3,
                0.5,
            ]
        )
    )

    assert np.isclose(
        weights.sum(),
        1.0,
    )


# ----------------------------------------------------------------------
# Classification metrics
# ----------------------------------------------------------------------


def test_classification_metrics_accuracy() -> None:
    y_true = np.array(
        [
            0,
            0,
            1,
            1,
        ]
    )

    y_pred = np.array(
        [
            0,
            1,
            1,
            1,
        ]
    )

    probabilities = np.array(
        [
            0.10,
            0.60,
            0.80,
            0.90,
        ]
    )

    metrics = classification_metrics(
        y_true,
        y_pred,
        probabilities,
    )

    assert np.isclose(
        metrics.accuracy,
        0.75,
    )

    assert 0 <= metrics.brier_score <= 1


# ----------------------------------------------------------------------
# Artifact persistence
# ----------------------------------------------------------------------


def test_artifact_save_and_load(
    tmp_path: Path,
) -> None:
    X, y = (
        pd.DataFrame(
            {
                "a": np.arange(
                    100.0
                ),
                "b": np.arange(
                    100.0
                )[::-1],
            }
        ),
        pd.Series(
            [
                0,
                1,
            ]
            * 50
        ),
    )

    preprocessor = SafePreprocessor(
        PreprocessorConfig()
    )

    preprocessor.fit(
        X
    )

    transformed = preprocessor.transform(
        X
    )

    model = DirectionClassifier()

    model.fit(
        transformed,
        y,
    )

    metadata = ArtifactMetadata(
        model_id="test-model",
        version="1.0",
        symbol="TEST",
        timeframe="1D",
        horizon=5,
        target="Direction_5",
        feature_names=preprocessor.get_feature_names(),
        approval_status="RESEARCH",
    )

    artifact = ModelArtifact(
        model=model,
        preprocessor=preprocessor,
        metadata=metadata,
    )

    path = (
        tmp_path
        / "model.joblib"
    )

    save_artifact(
        artifact,
        path,
    )

    assert path.exists()

    loaded = load_artifact(
        path
    )

    assert (
        loaded.metadata.model_id
        == "test-model"
    )

    assert (
        loaded.metadata.feature_names
        == metadata.feature_names
    )


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------


def test_model_registry_research_entry(
    tmp_path: Path,
) -> None:
    registry_path = (
        tmp_path
        / "registry.json"
    )

    registry = ModelRegistry(
        registry_path
    )

    entry = ModelRegistryEntry(
        model_id="model-001",
        artifact_path="artifacts/model-001.joblib",
        experiment_id="experiment-001",
        version="1.0",
        symbol="RELIANCE.NS",
        timeframe="1D",
        horizon=5,
        target="Direction_5",
        status="RESEARCH",
    )

    registry.register(
        entry
    )

    stored = registry.get(
        "model-001"
    )

    assert stored is not None

    assert (
        stored.model_id
        == "model-001"
    )

    assert (
        stored.status
        == "RESEARCH"
    )


def test_registry_does_not_approve_without_all_gates(
    tmp_path: Path,
) -> None:
    registry_path = (
        tmp_path
        / "registry.json"
    )

    registry = ModelRegistry(
        registry_path
    )

    entry = ModelRegistryEntry(
        model_id="model-002",
        artifact_path=str(
            tmp_path
            / "model.joblib"
        ),
        experiment_id="experiment-002",
        version="1.0",
        symbol="TEST",
        timeframe="1D",
        horizon=5,
        target="Direction_5",
        status="RESEARCH",
    )

    registry.register(
        entry
    )

    with pytest.raises(
        Exception
    ):
        registry.assert_approved(
            "model-002"
        )


def test_registry_can_find_research_models(
    tmp_path: Path,
) -> None:
    registry_path = (
        tmp_path
        / "registry.json"
    )

    registry = ModelRegistry(
        registry_path
    )

    for number in range(
        3
    ):
        registry.register(
            ModelRegistryEntry(
                model_id=f"model-{number}",
                artifact_path=f"model-{number}.joblib",
                experiment_id=f"exp-{number}",
                version="1.0",
                symbol="TEST",
                timeframe="1D",
                horizon=5,
                target="Direction_5",
                status="RESEARCH",
            )
        )

    models = registry.research_models()

    assert len(
        models
    ) == 3
