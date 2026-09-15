"""
Tests for the protected final holdout pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.research.holdout_gate import HoldoutGateInput
from src.research.pipeline_holdout import (
    HoldoutPipelineResult,
    ProtectedHoldoutPipeline,
    run_protected_holdout,
)


def make_dataset(
    n: int = 240,
) -> pd.DataFrame:
    index = pd.date_range(
        "2020-01-01",
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


def make_model_and_preprocessor(
    development: pd.DataFrame,
):
    features = [
        "Feature_A",
        "Feature_B",
    ]

    preprocessor = StandardScaler()

    X = preprocessor.fit_transform(
        development[features]
    )

    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
    )

    model.fit(
        X,
        development["Target"],
    )

    return (
        model,
        preprocessor,
        features,
    )


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


def test_pipeline_construction():
    pipeline = ProtectedHoldoutPipeline()

    assert pipeline is not None
    assert pipeline.accuracy_threshold == 0.95


def test_custom_threshold():
    pipeline = ProtectedHoldoutPipeline(
        accuracy_threshold=0.90
    )

    assert pipeline.accuracy_threshold == 0.90


def test_invalid_threshold():
    with pytest.raises(ValueError):
        ProtectedHoldoutPipeline(
            accuracy_threshold=-0.1
        )

    with pytest.raises(ValueError):
        ProtectedHoldoutPipeline(
            accuracy_threshold=1.1
        )


def test_successful_pipeline_run():
    development, holdout = (
        make_partitions()
    )

    model, preprocessor, features = (
        make_model_and_preprocessor(
            development
        )
    )

    pipeline = ProtectedHoldoutPipeline()

    result = pipeline.run(
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
        HoldoutPipelineResult,
    )

    assert result.completed is True
    assert result.evaluated is True
    assert result.final_holdout_used is True


def test_convenience_api():
    development, holdout = (
        make_partitions()
    )

    model, preprocessor, features = (
        make_model_and_preprocessor(
            development
        )
    )

    result = run_protected_holdout(
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
        HoldoutPipelineResult,
    )

    assert result.completed is True


def test_blocked_gate_prevents_evaluation():
    development, holdout = (
        make_partitions()
    )

    model, preprocessor, features = (
        make_model_and_preprocessor(
            development
        )
    )

    pipeline = ProtectedHoldoutPipeline()

    result = pipeline.run(
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

    assert result.completed is False
    assert result.evaluated is False
    assert result.final_holdout_used is False
    assert result.stage.evaluation is None


def test_previous_holdout_usage_blocks_pipeline():
    development, holdout = (
        make_partitions()
    )

    model, preprocessor, features = (
        make_model_and_preprocessor(
            development
        )
    )

    result = run_protected_holdout(
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

    assert result.completed is False
    assert result.evaluated is False
    assert result.final_holdout_used is False


def test_calibration_before_holdout_blocks_pipeline():
    development, holdout = (
        make_partitions()
    )

    model, preprocessor, features = (
        make_model_and_preprocessor(
            development
        )
    )

    result = run_protected_holdout(
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

    assert result.completed is False
    assert result.evaluated is False


def test_threshold_optimization_blocks_pipeline():
    development, holdout = (
        make_partitions()
    )

    model, preprocessor, features = (
        make_model_and_preprocessor(
            development
        )
    )

    result = run_protected_holdout(
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

    assert result.completed is False
    assert result.evaluated is False


def test_model_identity_must_match():
    development, holdout = (
        make_partitions()
    )

    model, preprocessor, features = (
        make_model_and_preprocessor(
            development
        )
    )

    pipeline = ProtectedHoldoutPipeline()

    with pytest.raises(
        ValueError,
        match="model_id",
    ):
        pipeline.run(
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


def test_invalid_gate_type_fails():
    development, holdout = (
        make_partitions()
    )

    model, preprocessor, features = (
        make_model_and_preprocessor(
            development
        )
    )

    pipeline = ProtectedHoldoutPipeline()

    with pytest.raises(TypeError):
        pipeline.run(
            model_id="model_test",
            model=model,
            preprocessor=preprocessor,
            development_data=development,
            holdout_data=holdout,
            feature_columns=features,
            target_column="Target",
            gate_input="invalid",
        )


def test_invalid_model_id_type_fails():
    development, holdout = (
        make_partitions()
    )

    model, preprocessor, features = (
        make_model_and_preprocessor(
            development
        )
    )

    pipeline = ProtectedHoldoutPipeline()

    with pytest.raises(TypeError):
        pipeline.run(
            model_id=123,
            model=model,
            preprocessor=preprocessor,
            development_data=development,
            holdout_data=holdout,
            feature_columns=features,
            target_column="Target",
            gate_input=make_gate(),
        )


def test_empty_model_id_fails():
    development, holdout = (
        make_partitions()
    )

    model, preprocessor, features = (
        make_model_and_preprocessor(
            development
        )
    )

    pipeline = ProtectedHoldoutPipeline()

    with pytest.raises(ValueError):
        pipeline.run(
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
    development, holdout = (
        make_partitions()
    )

    model, preprocessor, features = (
        make_model_and_preprocessor(
            development
        )
    )

    pipeline = ProtectedHoldoutPipeline()

    with pytest.raises(ValueError):
        pipeline.run(
            model_id="model_test",
            model=model,
            preprocessor=preprocessor,
            development_data=development.iloc[:0],
            holdout_data=holdout,
            feature_columns=features,
            target_column="Target",
            gate_input=make_gate(),
        )


def test_empty_holdout_data_fails():
    development, _ = (
        make_partitions()
    )

    model, preprocessor, features = (
        make_model_and_preprocessor(
            development
        )
    )

    empty_holdout = development.iloc[:0].copy()

    pipeline = ProtectedHoldoutPipeline()

    with pytest.raises(ValueError):
        pipeline.run(
            model_id="model_test",
            model=model,
            preprocessor=preprocessor,
            development_data=development,
            holdout_data=empty_holdout,
            feature_columns=features,
            target_column="Target",
            gate_input=make_gate(),
        )


def test_non_datetime_development_index_fails():
    development, holdout = (
        make_partitions()
    )

    model, preprocessor, features = (
        make_model_and_preprocessor(
            development
        )
    )

    bad_development = development.copy()

    bad_development.index = range(
        len(bad_development)
    )

    pipeline = ProtectedHoldoutPipeline()

    with pytest.raises(TypeError):
        pipeline.run(
            model_id="model_test",
            model=model,
            preprocessor=preprocessor,
            development_data=bad_development,
            holdout_data=holdout,
            feature_columns=features,
            target_column="Target",
            gate_input=make_gate(),
        )


def test_non_datetime_holdout_index_fails():
    development, holdout = (
        make_partitions()
    )

    model, preprocessor, features = (
        make_model_and_preprocessor(
            development
        )
    )

    bad_holdout = holdout.copy()

    bad_holdout.index = range(
        len(bad_holdout)
    )

    pipeline = ProtectedHoldoutPipeline()

    with pytest.raises(TypeError):
        pipeline.run(
            model_id="model_test",
            model=model,
            preprocessor=preprocessor,
            development_data=development,
            holdout_data=bad_holdout,
            feature_columns=features,
            target_column="Target",
            gate_input=make_gate(),
        )


def test_missing_feature_fails():
    development, holdout = (
        make_partitions()
    )

    model, preprocessor, _ = (
        make_model_and_preprocessor(
            development
        )
    )

    pipeline = ProtectedHoldoutPipeline()

    with pytest.raises(
        ValueError,
        match="Features",
    ):
        pipeline.run(
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
    development, holdout = (
        make_partitions()
    )

    model, preprocessor, features = (
        make_model_and_preprocessor(
            development
        )
    )

    pipeline = ProtectedHoldoutPipeline()

    with pytest.raises(
        ValueError,
        match="Target",
    ):
        pipeline.run(
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
    development, holdout = (
        make_partitions()
    )

    model, preprocessor, _ = (
        make_model_and_preprocessor(
            development
        )
    )

    pipeline = ProtectedHoldoutPipeline()

    with pytest.raises(ValueError):
        pipeline.run(
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
    development, holdout = (
        make_partitions()
    )

    _, preprocessor, features = (
        make_model_and_preprocessor(
            development
        )
    )

    pipeline = ProtectedHoldoutPipeline()

    with pytest.raises(ValueError):
        pipeline.run(
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
    development, holdout = (
        make_partitions()
    )

    model, _, features = (
        make_model_and_preprocessor(
            development
        )
    )

    pipeline = ProtectedHoldoutPipeline()

    with pytest.raises(ValueError):
        pipeline.run(
            model_id="model_test",
            model=model,
            preprocessor=None,
            development_data=development,
            holdout_data=holdout,
            feature_columns=features,
            target_column="Target",
            gate_input=make_gate(),
        )


def test_overlapping_partitions_fail():
    development, _ = (
        make_partitions()
    )

    model, preprocessor, features = (
        make_model_and_preprocessor(
            development
        )
    )

    overlapping = development.iloc[
        -10:
    ].copy()

    pipeline = ProtectedHoldoutPipeline()

    with pytest.raises(
        ValueError,
        match="overlap",
    ):
        pipeline.run(
            model_id="model_test",
            model=model,
            preprocessor=preprocessor,
            development_data=development,
            holdout_data=overlapping,
            feature_columns=features,
            target_column="Target",
            gate_input=make_gate(),
        )


def test_model_is_not_modified():
    development, holdout = (
        make_partitions()
    )

    model, preprocessor, features = (
        make_model_and_preprocessor(
            development
        )
    )

    coefficients_before = (
        model.coef_.copy()
    )

    run_protected_holdout(
        model_id="model_test",
        model=model,
        preprocessor=preprocessor,
        development_data=development,
        holdout_data=holdout,
        feature_columns=features,
        target_column="Target",
        gate_input=make_gate(),
    )

    np.testing.assert_array_equal(
        coefficients_before,
        model.coef_,
    )


def test_preprocessor_is_not_modified():
    development, holdout = (
        make_partitions()
    )

    model, preprocessor, features = (
        make_model_and_preprocessor(
            development
        )
    )

    mean_before = preprocessor.mean_.copy()
    scale_before = (
        preprocessor.scale_.copy()
    )

    run_protected_holdout(
        model_id="model_test",
        model=model,
        preprocessor=preprocessor,
        development_data=development,
        holdout_data=holdout,
        feature_columns=features,
        target_column="Target",
        gate_input=make_gate(),
    )

    np.testing.assert_array_equal(
        mean_before,
        preprocessor.mean_,
    )

    np.testing.assert_array_equal(
        scale_before,
        preprocessor.scale_,
    )


def test_input_data_is_not_modified():
    development, holdout = (
        make_partitions()
    )

    development_before = (
        development.copy(deep=True)
    )

    holdout_before = (
        holdout.copy(deep=True)
    )

    model, preprocessor, features = (
        make_model_and_preprocessor(
            development
        )
    )

    run_protected_holdout(
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


def test_pipeline_never_grants_production_approval():
    development, holdout = (
        make_partitions()
    )

    model, preprocessor, features = (
        make_model_and_preprocessor(
            development
        )
    )

    result = run_protected_holdout(
        model_id="model_test",
        model=model,
        preprocessor=preprocessor,
        development_data=development,
        holdout_data=holdout,
        feature_columns=features,
        target_column="Target",
        gate_input=make_gate(),
    )

    assert result.production_approved is False
    assert result.metadata[
        "production_approved"
    ] is False
    assert result.metadata[
        "research_only"
    ] is True


def test_pipeline_records_holdout_usage_only_after_evaluation():
    development, holdout = (
        make_partitions()
    )

    model, preprocessor, features = (
        make_model_and_preprocessor(
            development
        )
    )

    blocked = run_protected_holdout(
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

    assert blocked.final_holdout_used is False

    allowed = run_protected_holdout(
        model_id="model_test",
        model=model,
        preprocessor=preprocessor,
        development_data=development,
        holdout_data=holdout,
        feature_columns=features,
        target_column="Target",
        gate_input=make_gate(),
    )

    assert allowed.final_holdout_used is True


def test_result_contains_accuracy():
    development, holdout = (
        make_partitions()
    )

    model, preprocessor, features = (
        make_model_and_preprocessor(
            development
        )
    )

    result = run_protected_holdout(
        model_id="model_test",
        model=model,
        preprocessor=preprocessor,
        development_data=development,
        holdout_data=holdout,
        feature_columns=features,
        target_column="Target",
        gate_input=make_gate(),
    )

    assert result.accuracy is not None
    assert 0.0 <= result.accuracy <= 1.0


def test_accuracy_gate_is_recorded():
    development, holdout = (
        make_partitions()
    )

    model, preprocessor, features = (
        make_model_and_preprocessor(
            development
        )
    )

    result = run_protected_holdout(
        model_id="model_test",
        model=model,
        preprocessor=preprocessor,
        development_data=development,
        holdout_data=holdout,
        feature_columns=features,
        target_column="Target",
        gate_input=make_gate(),
    )

    assert result.passed_accuracy_gate == (
        result.accuracy >= 0.95
    )


def test_summary_contains_key_fields():
    development, holdout = (
        make_partitions()
    )

    model, preprocessor, features = (
        make_model_and_preprocessor(
            development
        )
    )

    result = run_protected_holdout(
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
    assert "completed" in summary
    assert "eligible" in summary
    assert "evaluated" in summary
    assert "accuracy" in summary
    assert "production_approved" in summary


def test_blocked_result_summary_is_fail_closed():
    development, holdout = (
        make_partitions()
    )

    model, preprocessor, features = (
        make_model_and_preprocessor(
            development
        )
    )

    result = run_protected_holdout(
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

    assert summary["completed"] is False
    assert summary["evaluated"] is False
    assert summary[
        "final_holdout_used"
    ] is False
    assert summary[
        "production_approved"
    ] is False


def test_metadata_explicitly_blocks_holdout_tuning():
    development, holdout = (
        make_partitions()
    )

    model, preprocessor, features = (
        make_model_and_preprocessor(
            development
        )
    )

    result = run_protected_holdout(
        model_id="model_test",
        model=model,
        preprocessor=preprocessor,
        development_data=development,
        holdout_data=holdout,
        feature_columns=features,
        target_column="Target",
        gate_input=make_gate(),
    )

    metadata = result.metadata

    assert metadata[
        "holdout_used_for_selection"
    ] is False

    assert metadata[
        "holdout_used_for_feature_selection"
    ] is False

    assert metadata[
        "holdout_used_for_hyperparameter_tuning"
    ] is False

    assert metadata[
        "holdout_used_for_calibration"
    ] is False

    assert metadata[
        "holdout_used_for_threshold_optimization"
    ] is False


def test_metadata_confirms_no_model_refit():
    development, holdout = (
        make_partitions()
    )

    model, preprocessor, features = (
        make_model_and_preprocessor(
            development
        )
    )

    result = run_protected_holdout(
        model_id="model_test",
        model=model,
        preprocessor=preprocessor,
        development_data=development,
        holdout_data=holdout,
        feature_columns=features,
        target_column="Target",
        gate_input=make_gate(),
    )

    assert result.metadata[
        "model_fitted_on_holdout"
    ] is False

    assert result.metadata[
        "preprocessor_fitted_on_holdout"
    ] is False


def test_holdout_gate_status_is_recorded():
    development, holdout = (
        make_partitions()
    )

    model, preprocessor, features = (
        make_model_and_preprocessor(
            development
        )
    )

    result = run_protected_holdout(
        model_id="model_test",
        model=model,
        preprocessor=preprocessor,
        development_data=development,
        holdout_data=holdout,
        feature_columns=features,
        target_column="Target",
        gate_input=make_gate(),
    )

    assert result.metadata[
        "holdout_gate_status"
    ] == "READY"


def test_assertion_helper():
    development, holdout = (
        make_partitions()
    )

    model, preprocessor, features = (
        make_model_and_preprocessor(
            development
        )
    )

    pipeline = ProtectedHoldoutPipeline()

    result = pipeline.run(
        model_id="model_test",
        model=model,
        preprocessor=preprocessor,
        development_data=development,
        holdout_data=holdout,
        feature_columns=features,
        target_column="Target",
        gate_input=make_gate(),
    )

    assert result.evaluated is True
