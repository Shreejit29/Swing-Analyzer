"""
Tests for the protected final holdout research stage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.research.holdout_gate import HoldoutGateInput
from src.research.holdout_stage import (
    FinalHoldoutStage,
    HoldoutStageResult,
    run_final_holdout_stage,
)


def make_dataset(
    n: int = 240,
    start: str = "2020-01-01",
) -> pd.DataFrame:
    index = pd.date_range(
        start=start,
        periods=n,
        freq="D",
    )

    x1 = np.sin(np.arange(n) / 10.0)
    x2 = np.cos(np.arange(n) / 15.0)

    target = (
        (x1 + x2) > 0
    ).astype(int)

    return pd.DataFrame(
        {
            "Feature_A": x1,
            "Feature_B": x2,
            "Target": target,
        },
        index=index,
    )


def make_partitions():
    data = make_dataset()

    development = data.iloc[:180].copy()
    holdout = data.iloc[180:].copy()

    return development, holdout


def make_frozen_model_and_preprocessor(
    development: pd.DataFrame,
):
    features = [
        "Feature_A",
        "Feature_B",
    ]

    scaler = StandardScaler()

    X_train = development[features]
    y_train = development["Target"]

    X_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
    )

    model.fit(
        X_scaled,
        y_train,
    )

    return model, scaler, features


def make_gate(
    model_id: str = "model_test",
    **overrides,
) -> HoldoutGateInput:
    values = {
        "model_id": model_id,
        "selection_completed": True,
        "walk_forward_completed": True,
        "feature_selection_frozen": True,
        "hyperparameters_frozen": True,
        "preprocessing_frozen": True,
        "calibration_fitted": False,
        "threshold_optimization_completed": False,
        "final_holdout_used_previously": False,
    }

    values.update(overrides)

    return HoldoutGateInput(**values)


def test_stage_construction():
    stage = FinalHoldoutStage()

    assert stage is not None
    assert stage.accuracy_threshold == 0.95


def test_custom_accuracy_threshold():
    stage = FinalHoldoutStage(
        accuracy_threshold=0.90
    )

    assert stage.accuracy_threshold == 0.90


def test_invalid_accuracy_threshold():
    with pytest.raises(ValueError):
        FinalHoldoutStage(
            accuracy_threshold=1.5
        )

    with pytest.raises(ValueError):
        FinalHoldoutStage(
            accuracy_threshold=-0.1
        )


def test_successful_holdout_stage():
    development, holdout = make_partitions()

    model, preprocessor, features = (
        make_frozen_model_and_preprocessor(
            development
        )
    )

    result = run_final_holdout_stage(
        model_id="model_test",
        model=model,
        preprocessor=preprocessor,
        development_data=development,
        holdout_data=holdout,
        feature_columns=features,
        target_column="Target",
        gate_input=make_gate(),
    )

    assert isinstance(
        result,
        HoldoutStageResult,
    )

    assert result.successful is True
    assert result.evaluated is True
    assert result.final_holdout_used is True


def test_gate_must_pass_before_evaluation():
    development, holdout = make_partitions()

    model, preprocessor, features = (
        make_frozen_model_and_preprocessor(
            development
        )
    )

    result = run_final_holdout_stage(
        model_id="model_test",
        model=model,
        preprocessor=preprocessor,
        development_data=development,
        holdout_data=holdout,
        feature_columns=features,
        target_column="Target",
        gate_input=make_gate(
            selection_completed=False
        ),
    )

    assert result.successful is False
    assert result.evaluated is False
    assert result.final_holdout_used is False
    assert result.evaluation is None


def test_previous_holdout_usage_blocks_evaluation():
    development, holdout = make_partitions()

    model, preprocessor, features = (
        make_frozen_model_and_preprocessor(
            development
        )
    )

    result = run_final_holdout_stage(
        model_id="model_test",
        model=model,
        preprocessor=preprocessor,
        development_data=development,
        holdout_data=holdout,
        feature_columns=features,
        target_column="Target",
        gate_input=make_gate(
            final_holdout_used_previously=True
        ),
    )

    assert result.successful is False
    assert result.evaluated is False
    assert result.final_holdout_used is False


def test_calibration_before_holdout_blocks_evaluation():
    development, holdout = make_partitions()

    model, preprocessor, features = (
        make_frozen_model_and_preprocessor(
            development
        )
    )

    result = run_final_holdout_stage(
        model_id="model_test",
        model=model,
        preprocessor=preprocessor,
        development_data=development,
        holdout_data=holdout,
        feature_columns=features,
        target_column="Target",
        gate_input=make_gate(
            calibration_fitted=True
        ),
    )

    assert result.successful is False
    assert result.evaluated is False


def test_threshold_optimization_blocks_evaluation():
    development, holdout = make_partitions()

    model, preprocessor, features = (
        make_frozen_model_and_preprocessor(
            development
        )
    )

    result = run_final_holdout_stage(
        model_id="model_test",
        model=model,
        preprocessor=preprocessor,
        development_data=development,
        holdout_data=holdout,
        feature_columns=features,
        target_column="Target",
        gate_input=make_gate(
            threshold_optimization_completed=True
        ),
    )

    assert result.successful is False
    assert result.evaluated is False


def test_model_id_must_match_gate():
    development, holdout = make_partitions()

    model, preprocessor, features = (
        make_frozen_model_and_preprocessor(
            development
        )
    )

    with pytest.raises(ValueError, match="model_id"):
        run_final_holdout_stage(
            model_id="model_A",
            model=model,
            preprocessor=preprocessor,
            development_data=development,
            holdout_data=holdout,
            feature_columns=features,
            target_column="Target",
            gate_input=make_gate(
                model_id="model_B"
            ),
        )


def test_empty_model_id_fails():
    development, holdout = make_partitions()

    model, preprocessor, features = (
        make_frozen_model_and_preprocessor(
            development
        )
    )

    with pytest.raises(ValueError):
        run_final_holdout_stage(
            model_id="",
            model=model,
            preprocessor=preprocessor,
            development_data=development,
            holdout_data=holdout,
            feature_columns=features,
            target_column="Target",
            gate_input=make_gate(
                model_id=""
            ),
        )


def test_empty_development_data_fails():
    _, holdout = make_partitions()

    data = make_dataset()

    model, preprocessor, features = (
        make_frozen_model_and_preprocessor(
            data.iloc[:180]
        )
    )

    with pytest.raises(ValueError):
        run_final_holdout_stage(
            model_id="model_test",
            model=model,
            preprocessor=preprocessor,
            development_data=data.iloc[:0],
            holdout_data=holdout,
            feature_columns=features,
            target_column="Target",
            gate_input=make_gate(),
        )


def test_empty_holdout_data_fails():
    development, _ = make_partitions()

    model, preprocessor, features = (
        make_frozen_model_and_preprocessor(
            development
        )
    )

    empty_holdout = development.iloc[:0].copy()

    with pytest.raises(ValueError):
        run_final_holdout_stage(
            model_id="model_test",
            model=model,
            preprocessor=preprocessor,
            development_data=development,
            holdout_data=empty_holdout,
            feature_columns=features,
            target_column="Target",
            gate_input=make_gate(),
        )


def test_unsorted_development_data_fails():
    development, holdout = make_partitions()

    model, preprocessor, features = (
        make_frozen_model_and_preprocessor(
            development
        )
    )

    shuffled = development.sample(
        frac=1,
        random_state=42,
    )

    with pytest.raises(ValueError):
        run_final_holdout_stage(
            model_id="model_test",
            model=model,
            preprocessor=preprocessor,
            development_data=shuffled,
            holdout_data=holdout,
            feature_columns=features,
            target_column="Target",
            gate_input=make_gate(),
        )


def test_duplicate_holdout_timestamps_fail():
    development, holdout = make_partitions()

    model, preprocessor, features = (
        make_frozen_model_and_preprocessor(
            development
        )
    )

    duplicate_holdout = holdout.copy()

    duplicate_holdout.index = (
        [holdout.index[0]]
        + list(holdout.index[:-1])
    )

    with pytest.raises(ValueError):
        run_final_holdout_stage(
            model_id="model_test",
            model=model,
            preprocessor=preprocessor,
            development_data=development,
            holdout_data=duplicate_holdout,
            feature_columns=features,
            target_column="Target",
            gate_input=make_gate(),
        )


def test_overlapping_partitions_fail():
    development, holdout = make_partitions()

    model, preprocessor, features = (
        make_frozen_model_and_preprocessor(
            development
        )
    )

    overlapping_holdout = development.iloc[
        -10:
    ].copy()

    with pytest.raises(
        ValueError,
        match="overlap",
    ):
        run_final_holdout_stage(
            model_id="model_test",
            model=model,
            preprocessor=preprocessor,
            development_data=development,
            holdout_data=overlapping_holdout,
            feature_columns=features,
            target_column="Target",
            gate_input=make_gate(),
        )


def test_missing_feature_fails():
    development, holdout = make_partitions()

    model, preprocessor, _ = (
        make_frozen_model_and_preprocessor(
            development
        )
    )

    with pytest.raises(ValueError, match="Features"):
        run_final_holdout_stage(
            model_id="model_test",
            model=model,
            preprocessor=preprocessor,
            development_data=development,
            holdout_data=holdout,
            feature_columns=[
                "Feature_A",
                "Missing_Feature",
            ],
            target_column="Target",
            gate_input=make_gate(),
        )


def test_missing_target_fails():
    development, holdout = make_partitions()

    model, preprocessor, features = (
        make_frozen_model_and_preprocessor(
            development
        )
    )

    with pytest.raises(ValueError, match="Target"):
        run_final_holdout_stage(
            model_id="model_test",
            model=model,
            preprocessor=preprocessor,
            development_data=development,
            holdout_data=holdout,
            feature_columns=features,
            target_column="Missing_Target",
            gate_input=make_gate(),
        )


def test_empty_feature_list_fails():
    development, holdout = make_partitions()

    model, preprocessor, _ = (
        make_frozen_model_and_preprocessor(
            development
        )
    )

    with pytest.raises(ValueError):
        run_final_holdout_stage(
            model_id="model_test",
            model=model,
            preprocessor=preprocessor,
            development_data=development,
            holdout_data=holdout,
            feature_columns=[],
            target_column="Target",
            gate_input=make_gate(),
        )


def test_none_model_fails():
    development, holdout = make_partitions()

    _, preprocessor, features = (
        make_frozen_model_and_preprocessor(
            development
        )
    )

    with pytest.raises(ValueError):
        run_final_holdout_stage(
            model_id="model_test",
            model=None,
            preprocessor=preprocessor,
            development_data=development,
            holdout_data=holdout,
            feature_columns=features,
            target_column="Target",
            gate_input=make_gate(),
        )


def test_none_preprocessor_fails():
    development, holdout = make_partitions()

    model, _, features = (
        make_frozen_model_and_preprocessor(
            development
        )
    )

    with pytest.raises(ValueError):
        run_final_holdout_stage(
            model_id="model_test",
            model=model,
            preprocessor=None,
            development_data=development,
            holdout_data=holdout,
            feature_columns=features,
            target_column="Target",
            gate_input=make_gate(),
        )


def test_non_datetime_index_fails():
    development, holdout = make_partitions()

    model, preprocessor, features = (
        make_frozen_model_and_preprocessor(
            development
        )
    )

    development_bad = development.copy()
    development_bad.index = range(
        len(development_bad)
    )

    with pytest.raises(TypeError):
        run_final_holdout_stage(
            model_id="model_test",
            model=model,
            preprocessor=preprocessor,
            development_data=development_bad,
            holdout_data=holdout,
            feature_columns=features,
            target_column="Target",
            gate_input=make_gate(),
        )


def test_holdout_is_not_used_for_preprocessing_fit():
    development, holdout = make_partitions()

    model, preprocessor, features = (
        make_frozen_model_and_preprocessor(
            development
        )
    )

    mean_before = preprocessor.mean_.copy()

    result = run_final_holdout_stage(
        model_id="model_test",
        model=model,
        preprocessor=preprocessor,
        development_data=development,
        holdout_data=holdout,
        feature_columns=features,
        target_column="Target",
        gate_input=make_gate(),
    )

    mean_after = preprocessor.mean_.copy()

    assert result.evaluated is True
    np.testing.assert_array_equal(
        mean_before,
        mean_after,
    )


def test_model_is_not_refitted():
    development, holdout = make_partitions()

    model, preprocessor, features = (
        make_frozen_model_and_preprocessor(
            development
        )
    )

    coefficients_before = (
        model.coef_.copy()
    )

    result = run_final_holdout_stage(
        model_id="model_test",
        model=model,
        preprocessor=preprocessor,
        development_data=development,
        holdout_data=holdout,
        feature_columns=features,
        target_column="Target",
        gate_input=make_gate(),
    )

    coefficients_after = (
        model.coef_.copy()
    )

    assert result.evaluated is True

    np.testing.assert_array_equal(
        coefficients_before,
        coefficients_after,
    )


def test_holdout_mutation_does_not_change_frozen_model():
    development, holdout = make_partitions()

    model, preprocessor, features = (
        make_frozen_model_and_preprocessor(
            development
        )
    )

    predictions_before = model.predict(
        preprocessor.transform(
            holdout[features]
        )
    )

    mutated = holdout.copy()

    mutated["Feature_A"] = (
        mutated["Feature_A"] * 1000.0
    )

    mutated["Feature_B"] = (
        mutated["Feature_B"] * -500.0
    )

    mutated["Target"] = 1 - mutated["Target"]

    run_final_holdout_stage(
        model_id="model_test",
        model=model,
        preprocessor=preprocessor,
        development_data=development,
        holdout_data=mutated,
        feature_columns=features,
        target_column="Target",
        gate_input=make_gate(),
    )

    predictions_after = model.predict(
        preprocessor.transform(
            holdout[features]
        )
    )

    np.testing.assert_array_equal(
        predictions_before,
        predictions_after,
    )


def test_result_contains_holdout_accuracy():
    development, holdout = make_partitions()

    model, preprocessor, features = (
        make_frozen_model_and_preprocessor(
            development
        )
    )

    result = run_final_holdout_stage(
        model_id="model_test",
        model=model,
        preprocessor=preprocessor,
        development_data=development,
        holdout_data=holdout,
        feature_columns=features,
        target_column="Target",
        gate_input=make_gate(),
    )

    assert result.holdout_accuracy is not None
    assert 0.0 <= result.holdout_accuracy <= 1.0


def test_result_records_accuracy_gate():
    development, holdout = make_partitions()

    model, preprocessor, features = (
        make_frozen_model_and_preprocessor(
            development
        )
    )

    result = run_final_holdout_stage(
        model_id="model_test",
        model=model,
        preprocessor=preprocessor,
        development_data=development,
        holdout_data=holdout,
        feature_columns=features,
        target_column="Target",
        gate_input=make_gate(),
    )

    expected = (
        result.holdout_accuracy >= 0.95
    )

    assert (
        result.passed_accuracy_gate
        == expected
    )


def test_summary_contains_stage_information():
    development, holdout = make_partitions()

    model, preprocessor, features = (
        make_frozen_model_and_preprocessor(
            development
        )
    )

    result = run_final_holdout_stage(
        model_id="model_test",
        model=model,
        preprocessor=preprocessor,
        development_data=development,
        holdout_data=holdout,
        feature_columns=features,
        target_column="Target",
        gate_input=make_gate(),
    )

    summary = result.summary()

    assert summary["model_id"] == "model_test"
    assert summary["evaluated"] is True
    assert summary["final_holdout_used"] is True
    assert "holdout_accuracy" in summary


def test_blocked_summary_records_no_holdout_usage():
    development, holdout = make_partitions()

    model, preprocessor, features = (
        make_frozen_model_and_preprocessor(
            development
        )
    )

    result = run_final_holdout_stage(
        model_id="model_test",
        model=model,
        preprocessor=preprocessor,
        development_data=development,
        holdout_data=holdout,
        feature_columns=features,
        target_column="Target",
        gate_input=make_gate(
            walk_forward_completed=False
        ),
    )

    summary = result.summary()

    assert summary["evaluated"] is False
    assert summary["final_holdout_used"] is False
    assert summary["successful"] is False


def test_assert_evaluated_passes_after_evaluation():
    development, holdout = make_partitions()

    model, preprocessor, features = (
        make_frozen_model_and_preprocessor(
            development
        )
    )

    stage = FinalHoldoutStage()

    result = stage.evaluate(
        model_id="model_test",
        model=model,
        preprocessor=preprocessor,
        development_data=development,
        holdout_data=holdout,
        feature_columns=features,
        target_column="Target",
        gate_input=make_gate(),
    )

    asserted = stage.assert_evaluated(
        result
    )

    assert asserted is result


def test_assert_evaluated_raises_when_gate_blocks():
    development, holdout = make_partitions()

    model, preprocessor, features = (
        make_frozen_model_and_preprocessor(
            development
        )
    )

    stage = FinalHoldoutStage()

    result = stage.evaluate(
        model_id="model_test",
        model=model,
        preprocessor=preprocessor,
        development_data=development,
        holdout_data=holdout,
        feature_columns=features,
        target_column="Target",
        gate_input=make_gate(
            selection_completed=False
        ),
    )

    with pytest.raises(RuntimeError):
        stage.assert_evaluated(result)


def test_assert_evaluated_requires_correct_type():
    stage = FinalHoldoutStage()

    with pytest.raises(TypeError):
        stage.assert_evaluated("invalid")


def test_research_only_metadata():
    development, holdout = make_partitions()

    model, preprocessor, features = (
        make_frozen_model_and_preprocessor(
            development
        )
    )

    result = run_final_holdout_stage(
        model_id="model_test",
        model=model,
        preprocessor=preprocessor,
        development_data=development,
        holdout_data=holdout,
        feature_columns=features,
        target_column="Target",
        gate_input=make_gate(),
    )

    assert result.metadata["research_only"] is True
    assert result.metadata["production_approved"] is False


def test_holdout_contamination_metadata_is_rejected():
    development, holdout = make_partitions()

    model, preprocessor, features = (
        make_frozen_model_and_preprocessor(
            development
        )
    )

    contaminated_gate = make_gate(
        metadata={
            "holdout_used_for_selection": True,
        }
    )

    with pytest.raises(
        ValueError,
        match="contamination",
    ):
        run_final_holdout_stage(
            model_id="model_test",
            model=model,
            preprocessor=preprocessor,
            development_data=development,
            holdout_data=holdout,
            feature_columns=features,
            target_column="Target",
            gate_input=contaminated_gate,
        )


def test_multiple_contamination_flags_are_rejected():
    development, holdout = make_partitions()

    model, preprocessor, features = (
        make_frozen_model_and_preprocessor(
            development
        )
    )

    contaminated_gate = make_gate(
        metadata={
            "holdout_used_for_selection": True,
            "holdout_used_for_calibration": True,
        }
    )

    with pytest.raises(
        ValueError,
        match="contamination",
    ):
        run_final_holdout_stage(
            model_id="model_test",
            model=model,
            preprocessor=preprocessor,
            development_data=development,
            holdout_data=holdout,
            feature_columns=features,
            target_column="Target",
            gate_input=contaminated_gate,
        )


def test_input_data_is_not_modified():
    development, holdout = make_partitions()

    development_before = development.copy(
        deep=True
    )
    holdout_before = holdout.copy(
        deep=True
    )

    model, preprocessor, features = (
        make_frozen_model_and_preprocessor(
            development
        )
    )

    run_final_holdout_stage(
        model_id="model_test",
        model=model,
        preprocessor=preprocessor,
        development_data=development,
        holdout_data=holdout,
        feature_columns=features,
        target_column="Target",
        gate_input=make_gate(),
    )

    pd.testing.assert_frame_equal(
        development,
        development_before,
    )

    pd.testing.assert_frame_equal(
        holdout,
        holdout_before,
    )
