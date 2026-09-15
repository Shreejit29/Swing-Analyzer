"""
Tests for final holdout evaluation.

The final holdout is the last untouched historical segment.

These tests verify that:

- holdout data is strictly newer than development data
- the evaluator does not fit the model
- the evaluator does not fit the preprocessor
- predictions retain exact holdout timestamps
- holdout metrics are calculated correctly
- holdout data is not used for feature selection
- changing only holdout data does not alter development data
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.classifier import (
    ClassifierConfig,
    DirectionClassifier,
)
from src.models.preprocessing import (
    PreprocessorConfig,
    SafePreprocessor,
)
from src.research.config import (
    ResearchPipelineConfig,
)
from src.research.holdout_evaluation import (
    FinalHoldoutEvaluator,
    HoldoutEvaluationResult,
    evaluate_final_holdout,
)
from src.research.model_development import (
    DevelopmentPartitions,
    ModelDevelopmentOrchestrator,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(42)

    index = pd.date_range(
        "2020-01-01",
        periods=600,
        freq="D",
    )

    f1 = rng.normal(
        0,
        1,
        len(index),
    )

    f2 = rng.normal(
        0,
        1,
        len(index),
    )

    signal = (
        f1
        + 0.25 * f2
        + rng.normal(
            0,
            0.25,
            len(index),
        )
    )

    target = (
        signal > 0
    ).astype(int)

    return pd.DataFrame(
        {
            "Feature_1": f1,
            "Feature_2": f2,
            "Close": 100 + f1,
            "Direction_5": target,
        },
        index=index,
    )


@pytest.fixture
def features() -> list[str]:
    return [
        "Feature_1",
        "Feature_2",
        "Close",
    ]


@pytest.fixture
def config() -> ResearchPipelineConfig:
    return ResearchPipelineConfig()


@pytest.fixture
def orchestrator(
    config,
) -> ModelDevelopmentOrchestrator:

    return ModelDevelopmentOrchestrator(
        config=config,
        preprocessor_config=PreprocessorConfig(),
        classifier_config=ClassifierConfig(
            model_type="logistic_regression"
        ),
    )


@pytest.fixture
def trained_objects(
    dataframe,
    features,
    orchestrator,
):
    """
    Train a model exclusively on the development/training period.
    """

    partitions = (
        orchestrator.split_development_holdout(
            dataframe
        )
    )

    training, _ = (
        orchestrator.split_training_validation(
            partitions.development
        )
    )

    preprocessor = SafePreprocessor(
        PreprocessorConfig()
    )

    X_train = training.loc[
        :,
        features,
    ]

    y_train = training[
        "Direction_5"
    ]

    X_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    model = DirectionClassifier(
        ClassifierConfig(
            model_type="logistic_regression"
        )
    )

    model.fit(
        X_processed,
        y_train,
    )

    return (
        partitions,
        preprocessor,
        model,
    )


# ---------------------------------------------------------------------
# Evaluator construction
# ---------------------------------------------------------------------


def test_evaluator_can_be_created():
    evaluator = (
        FinalHoldoutEvaluator()
    )

    assert evaluator is not None


def test_invalid_accuracy_threshold_fails():
    with pytest.raises(
        ValueError
    ):
        FinalHoldoutEvaluator(
            minimum_accuracy=1.5
        )


def test_negative_accuracy_threshold_fails():
    with pytest.raises(
        ValueError
    ):
        FinalHoldoutEvaluator(
            minimum_accuracy=-0.1
        )


# ---------------------------------------------------------------------
# Holdout evaluation
# ---------------------------------------------------------------------


def test_evaluation_returns_result(
    dataframe,
    features,
    trained_objects,
):
    partitions, preprocessor, model = (
        trained_objects
    )

    evaluator = FinalHoldoutEvaluator(
        minimum_accuracy=0.0
    )

    result = evaluator.evaluate(
        model=model,
        preprocessor=preprocessor,
        partitions=partitions,
        feature_columns=features,
        target_column="Direction_5",
        horizon=5,
        model_name="logistic_regression",
    )

    assert isinstance(
        result,
        HoldoutEvaluationResult,
    )


def test_holdout_accuracy_is_valid(
    dataframe,
    features,
    trained_objects,
):
    partitions, preprocessor, model = (
        trained_objects
    )

    result = evaluate_final_holdout(
        model=model,
        preprocessor=preprocessor,
        partitions=partitions,
        feature_columns=features,
        target_column="Direction_5",
        horizon=5,
        minimum_accuracy=0.0,
    )

    assert 0.0 <= (
        result.accuracy
    ) <= 1.0


def test_holdout_predictions_match_samples(
    dataframe,
    features,
    trained_objects,
):
    partitions, preprocessor, model = (
        trained_objects
    )

    result = evaluate_final_holdout(
        model=model,
        preprocessor=preprocessor,
        partitions=partitions,
        feature_columns=features,
        target_column="Direction_5",
        horizon=5,
        minimum_accuracy=0.0,
    )

    assert (
        len(result.predictions)
        == len(result.actuals)
    )

    assert (
        len(result.predictions)
        == result.evaluated_samples
    )


# ---------------------------------------------------------------------
# Exact timestamp preservation
# ---------------------------------------------------------------------


def test_holdout_timestamps_are_preserved(
    dataframe,
    features,
    trained_objects,
):
    partitions, preprocessor, model = (
        trained_objects
    )

    result = evaluate_final_holdout(
        model=model,
        preprocessor=preprocessor,
        partitions=partitions,
        feature_columns=features,
        target_column="Direction_5",
        horizon=5,
        minimum_accuracy=0.0,
    )

    expected_index = (
        partitions.holdout.loc[
            partitions.holdout[
                "Direction_5"
            ].notna()
        ].index
    )

    assert result.holdout_index.equals(
        expected_index
    )


def test_holdout_period_is_correct(
    dataframe,
    features,
    trained_objects,
):
    partitions, preprocessor, model = (
        trained_objects
    )

    result = evaluate_final_holdout(
        model=model,
        preprocessor=preprocessor,
        partitions=partitions,
        feature_columns=features,
        target_column="Direction_5",
        horizon=5,
        minimum_accuracy=0.0,
    )

    assert (
        result.holdout_start
        == result.holdout_index.min()
    )

    assert (
        result.holdout_end
        == result.holdout_index.max()
    )


# ---------------------------------------------------------------------
# Temporal safety
# ---------------------------------------------------------------------


def test_development_precedes_holdout(
    trained_objects,
):
    partitions, _, _ = (
        trained_objects
    )

    assert (
        partitions.development_end
        < partitions.holdout_start
    )


def test_overlapping_partitions_fail(
    dataframe,
    features,
    trained_objects,
):
    _, preprocessor, model = (
        trained_objects
    )

    development = dataframe.iloc[
        :450
    ].copy()

    holdout = dataframe.iloc[
        400:
    ].copy()

    overlapping = (
        DevelopmentPartitions(
            development=development,
            holdout=holdout,
            development_index=(
                development.index
            ),
            holdout_index=(
                holdout.index
            ),
            development_end=(
                development.index.max()
            ),
            holdout_start=(
                holdout.index.min()
            ),
        )
    )

    evaluator = FinalHoldoutEvaluator(
        minimum_accuracy=0.0
    )

    with pytest.raises(
        ValueError
    ):
        evaluator.evaluate(
            model=model,
            preprocessor=preprocessor,
            partitions=overlapping,
            feature_columns=features,
            target_column="Direction_5",
            horizon=5,
        )


# ---------------------------------------------------------------------
# Missing input protection
# ---------------------------------------------------------------------


def test_missing_model_fails(
    trained_objects,
    features,
):
    partitions, preprocessor, _ = (
        trained_objects
    )

    evaluator = FinalHoldoutEvaluator()

    with pytest.raises(
        ValueError
    ):
        evaluator.evaluate(
            model=None,
            preprocessor=preprocessor,
            partitions=partitions,
            feature_columns=features,
            target_column="Direction_5",
            horizon=5,
        )


def test_missing_preprocessor_fails(
    trained_objects,
    features,
):
    partitions, _, model = (
        trained_objects
    )

    evaluator = FinalHoldoutEvaluator()

    with pytest.raises(
        ValueError
    ):
        evaluator.evaluate(
            model=model,
            preprocessor=None,
            partitions=partitions,
            feature_columns=features,
            target_column="Direction_5",
            horizon=5,
        )


def test_missing_feature_fails(
    trained_objects,
):
    partitions, preprocessor, model = (
        trained_objects
    )

    evaluator = FinalHoldoutEvaluator()

    with pytest.raises(
        ValueError
    ):
        evaluator.evaluate(
            model=model,
            preprocessor=preprocessor,
            partitions=partitions,
            feature_columns=[
                "MissingFeature"
            ],
            target_column="Direction_5",
            horizon=5,
        )


def test_missing_target_fails(
    trained_objects,
    features,
):
    partitions, preprocessor, model = (
        trained_objects
    )

    evaluator = FinalHoldoutEvaluator()

    with pytest.raises(
        ValueError
    ):
        evaluator.evaluate(
            model=model,
            preprocessor=preprocessor,
            partitions=partitions,
            feature_columns=features,
            target_column="MissingTarget",
            horizon=5,
        )


# ---------------------------------------------------------------------
# Frozen-object safety
# ---------------------------------------------------------------------


def test_evaluation_does_not_replace_preprocessor(
    trained_objects,
    features,
):
    partitions, preprocessor, model = (
        trained_objects
    )

    original_feature_names = (
        preprocessor.feature_names_
    )

    evaluator = FinalHoldoutEvaluator(
        minimum_accuracy=0.0
    )

    evaluator.evaluate(
        model=model,
        preprocessor=preprocessor,
        partitions=partitions,
        feature_columns=features,
        target_column="Direction_5",
        horizon=5,
    )

    assert (
        preprocessor.feature_names_
        == original_feature_names
    )


def test_evaluation_does_not_refit_model(
    trained_objects,
    features,
):
    partitions, preprocessor, model = (
        trained_objects
    )

    before = model.predict(
        preprocessor.transform(
            partitions.holdout.loc[
                :,
                features,
            ]
        )
    )

    evaluator = FinalHoldoutEvaluator(
        minimum_accuracy=0.0
    )

    evaluator.evaluate(
        model=model,
        preprocessor=preprocessor,
        partitions=partitions,
        feature_columns=features,
        target_column="Direction_5",
        horizon=5,
    )

    after = model.predict(
        preprocessor.transform(
            partitions.holdout.loc[
                :,
                features,
            ]
        )
    )

    np.testing.assert_array_equal(
        before,
        after,
    )


# ---------------------------------------------------------------------
# Metadata safety
# ---------------------------------------------------------------------


def test_metadata_confirms_holdout_not_used_for_fitting(
    trained_objects,
    features,
):
    partitions, preprocessor, model = (
        trained_objects
    )

    result = evaluate_final_holdout(
        model=model,
        preprocessor=preprocessor,
        partitions=partitions,
        feature_columns=features,
        target_column="Direction_5",
        horizon=5,
        minimum_accuracy=0.0,
    )

    assert (
        result.metadata[
            "holdout_used_for_fitting"
        ]
        is False
    )

    assert (
        result.metadata[
            "holdout_used_for_feature_selection"
        ]
        is False
    )

    assert (
        result.metadata[
            "holdout_used_for_hyperparameter_search"
        ]
        is False
    )

    assert (
        result.metadata[
            "holdout_used_for_calibration"
        ]
        is False
    )

    assert (
        result.metadata[
            "preprocessor_refit_on_holdout"
        ]
        is False
    )

    assert (
        result.metadata[
            "model_refit_on_holdout"
        ]
        is False
    )


# ---------------------------------------------------------------------
# Gate behaviour
# ---------------------------------------------------------------------


def test_accuracy_gate_can_pass(
    trained_objects,
    features,
):
    partitions, preprocessor, model = (
        trained_objects
    )

    result = evaluate_final_holdout(
        model=model,
        preprocessor=preprocessor,
        partitions=partitions,
        feature_columns=features,
        target_column="Direction_5",
        horizon=5,
        minimum_accuracy=0.0,
    )

    assert (
        result.passed_accuracy_gate
        is True
    )


def test_impossible_high_threshold_fails(
    trained_objects,
    features,
):
    partitions, preprocessor, model = (
        trained_objects
    )

    result = evaluate_final_holdout(
        model=model,
        preprocessor=preprocessor,
        partitions=partitions,
        feature_columns=features,
        target_column="Direction_5",
        horizon=5,
        minimum_accuracy=1.0,
    )

    assert (
        result.passed_accuracy_gate
        == (
            result.accuracy
            >= 1.0
        )
    )


# ---------------------------------------------------------------------
# Holdout mutation attack
# ---------------------------------------------------------------------


def test_holdout_mutation_does_not_change_development(
    dataframe,
    orchestrator,
):
    partitions = (
        orchestrator.split_development_holdout(
            dataframe
        )
    )

    mutated = dataframe.copy()

    mutated.loc[
        partitions.holdout_index,
        "Feature_1",
    ] = 999_999

    mutated.loc[
        partitions.holdout_index,
        "Feature_2",
    ] = -999_999

    mutated_partitions = (
        orchestrator.split_development_holdout(
            mutated
        )
    )

    pd.testing.assert_frame_equal(
        partitions.development,
        mutated_partitions.development,
    )


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------


def test_summary_contains_key_fields(
    trained_objects,
    features,
):
    partitions, preprocessor, model = (
        trained_objects
    )

    result = evaluate_final_holdout(
        model=model,
        preprocessor=preprocessor,
        partitions=partitions,
        feature_columns=features,
        target_column="Direction_5",
        horizon=5,
        minimum_accuracy=0.0,
    )

    summary = result.summary()

    assert (
        "accuracy"
        in summary
    )

    assert (
        "evaluated_samples"
        in summary
    )

    assert (
        "holdout_start"
        in summary
    )

    assert (
        "holdout_end"
        in summary
    )

    assert (
        "passed_accuracy_gate"
        in summary
    )


# ---------------------------------------------------------------------
# Horizon validation
# ---------------------------------------------------------------------


def test_non_positive_horizon_fails(
    trained_objects,
    features,
):
    partitions, preprocessor, model = (
        trained_objects
    )

    evaluator = FinalHoldoutEvaluator(
        minimum_accuracy=0.0
    )

    with pytest.raises(
        ValueError
    ):
        evaluator.evaluate(
            model=model,
            preprocessor=preprocessor,
            partitions=partitions,
            feature_columns=features,
            target_column="Direction_5",
            horizon=0,
        )
