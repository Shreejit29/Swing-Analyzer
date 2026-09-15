"""
Shared API contracts and compatibility helpers for the AI Swing Analyser.

This module standardizes communication between research stages.

The purpose is to prevent subtle integration failures caused by different
research modules using different conventions for:

    - passed
    - approved
    - production_ready
    - production_approved
    - final_holdout_used
    - horizon results
    - metadata

This module does not train models, modify predictions, or perform research.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass
class StageContract:
    """Normalized representation of a research-stage result."""

    stage_name: str

    executed: bool = False
    passed: bool = False

    production_ready: bool = False
    production_approved: bool = False

    final_holdout_used: bool = False
    research_only: bool = True

    horizons: tuple[int, ...] = ()

    errors: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class HorizonContract:
    """Normalized representation of one horizon result."""

    horizon: int

    evaluated: bool = False
    passed: bool = False
    approved: bool = False

    accuracy: float | None = None
    confidence: float | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


def extract_value(
    source: Any,
    names: Sequence[str],
    default: Any = None,
) -> Any:
    """
    Extract a value from a mapping or object.

    The first matching field wins.
    """

    if source is None:
        return default

    if isinstance(
        source,
        Mapping,
    ):
        for name in names:
            if name in source:
                return source[name]

    for name in names:
        if hasattr(
            source,
            name,
        ):
            return getattr(
                source,
                name,
            )

    return default


def extract_metadata(
    source: Any,
) -> dict[str, Any]:
    """Return metadata without exposing the original mutable dictionary."""

    metadata = extract_value(
        source,
        ("metadata",),
        default={},
    )

    if isinstance(
        metadata,
        Mapping,
    ):
        return dict(metadata)

    return {}


def extract_errors(
    source: Any,
) -> list[str]:
    """Normalize stage errors."""

    errors = extract_value(
        source,
        ("errors",),
        default=[],
    )

    if errors is None:
        return []

    if isinstance(
        errors,
        str,
    ):
        return [errors]

    try:
        return [
            str(error)
            for error in errors
        ]
    except TypeError:
        return [str(errors)]


def extract_warnings(
    source: Any,
) -> list[str]:
    """Normalize stage warnings."""

    warnings = extract_value(
        source,
        ("warnings",),
        default=[],
    )

    if warnings is None:
        return []

    if isinstance(
        warnings,
        str,
    ):
        return [warnings]

    try:
        return [
            str(warning)
            for warning in warnings
        ]
    except TypeError:
        return [str(warnings)]


def extract_horizons(
    source: Any,
) -> tuple[int, ...]:
    """Extract and normalize configured horizons."""

    horizons = extract_value(
        source,
        (
            "horizons",
            "required_horizons",
            "allowed_horizons",
            "evaluated_horizons",
            "passed_horizons",
        ),
        default=(),
    )

    if horizons is None:
        return ()

    if isinstance(
        horizons,
        int,
    ):
        horizons = (horizons,)

    normalized = []

    try:
        for horizon in horizons:
            normalized.append(
                int(horizon)
            )
    except (
        TypeError,
        ValueError,
    ):
        return ()

    return tuple(
        dict.fromkeys(
            normalized
        )
    )


def extract_results_mapping(
    source: Any,
) -> Mapping[Any, Any]:
    """Return the horizon-result mapping when available."""

    if source is None:
        return {}

    if isinstance(
        source,
        Mapping,
    ):
        # A mapping can itself be a horizon mapping.
        if all(
            isinstance(
                key,
                int,
            )
            for key in source.keys()
        ):
            return source

    results = extract_value(
        source,
        (
            "results",
            "horizon_results",
        ),
        default={},
    )

    if isinstance(
        results,
        Mapping,
    ):
        return results

    return {}


def get_horizon_result(
    source: Any,
    horizon: int,
) -> Any:
    """Return a normalized horizon result."""

    horizon = int(horizon)

    results = extract_results_mapping(
        source
    )

    if horizon in results:
        return results[horizon]

    # Some serialized dictionaries may use string keys.
    if str(horizon) in results:
        return results[str(horizon)]

    getter = getattr(
        source,
        "get",
        None,
    )

    if callable(getter):
        try:
            return getter(horizon)
        except (
            KeyError,
            TypeError,
        ):
            pass

    return None


def extract_passed(
    source: Any,
) -> bool:
    """
    Normalize a stage-level pass status.

    Important:
    missing evidence is NOT treated as success.
    """

    value = extract_value(
        source,
        (
            "passed",
            "candidate_passed",
        ),
        default=False,
    )

    return value is True


def extract_approved(
    source: Any,
) -> bool:
    """Normalize an approval status."""

    value = extract_value(
        source,
        (
            "approved",
            "production_approved",
        ),
        default=False,
    )

    return value is True


def extract_production_ready(
    source: Any,
) -> bool:
    """Normalize production readiness."""

    value = extract_value(
        source,
        (
            "production_ready",
            "production_approved",
            "approved",
        ),
        default=False,
    )

    return value is True


def extract_final_holdout_used(
    source: Any,
) -> bool:
    """Determine whether final holdout evaluation was actually used."""

    direct = extract_value(
        source,
        ("final_holdout_used",),
        default=False,
    )

    if direct is True:
        return True

    metadata = extract_metadata(
        source
    )

    return (
        metadata.get(
            "final_holdout_used",
            False,
        )
        is True
    )


def normalize_horizon_pass(
    source: Any,
    horizon: int,
) -> bool:
    """
    Determine whether a specific horizon passed.

    No missing horizon is considered a pass.
    """

    result = get_horizon_result(
        source,
        horizon,
    )

    if result is None:
        return False

    return extract_passed(
        result
    ) or extract_approved(
        result
    )


def normalize_horizon_contract(
    source: Any,
    horizon: int,
) -> HorizonContract:
    """Convert a horizon result into the shared contract."""

    result = get_horizon_result(
        source,
        horizon,
    )

    if result is None:
        return HorizonContract(
            horizon=int(horizon),
            evaluated=False,
            passed=False,
            approved=False,
        )

    accuracy = extract_value(
        result,
        (
            "accuracy",
            "holdout_accuracy",
        ),
        default=None,
    )

    confidence = extract_value(
        result,
        (
            "confidence",
            "prediction_confidence",
        ),
        default=None,
    )

    try:
        if accuracy is not None:
            accuracy = float(
                accuracy
            )
    except (
        TypeError,
        ValueError,
    ):
        accuracy = None

    try:
        if confidence is not None:
            confidence = float(
                confidence
            )
    except (
        TypeError,
        ValueError,
    ):
        confidence = None

    return HorizonContract(
        horizon=int(horizon),
        evaluated=bool(
            extract_value(
                result,
                ("evaluated",),
                default=True,
            )
        ),
        passed=extract_passed(
            result
        ),
        approved=extract_approved(
            result
        ),
        accuracy=accuracy,
        confidence=confidence,
        metadata=extract_metadata(
            result
        ),
    )


def normalize_stage_contract(
    stage_name: str,
    source: Any,
    *,
    horizons: Sequence[int] = (),
) -> StageContract:
    """
    Convert any compatible research-stage result into one common contract.
    """

    metadata = extract_metadata(
        source
    )

    normalized_horizons = (
        tuple(
            int(x)
            for x in horizons
        )
        if horizons
        else extract_horizons(
            source
        )
    )

    final_holdout_used = (
        extract_final_holdout_used(
            source
        )
    )

    production_approved = (
        extract_approved(
            source
        )
        or metadata.get(
            "production_approved",
            False,
        )
        is True
    )

    production_ready = (
        extract_production_ready(
            source
        )
        or production_approved
    )

    errors = extract_errors(
        source
    )

    warnings = extract_warnings(
        source
    )

    passed = extract_passed(
        source
    )

    executed = bool(
        extract_value(
            source,
            ("executed",),
            default=True,
        )
    )

    research_only = bool(
        metadata.get(
            "research_only",
            not production_approved,
        )
    )

    metadata.update(
        {
            "stage_name": stage_name,
            "final_holdout_used": (
                final_holdout_used
            ),
            "production_approved": (
                production_approved
            ),
            "production_ready": (
                production_ready
            ),
            "research_only": (
                research_only
            ),
        }
    )

    return StageContract(
        stage_name=stage_name,
        executed=executed,
        passed=passed,
        production_ready=production_ready,
        production_approved=production_approved,
        final_holdout_used=final_holdout_used,
        research_only=research_only,
        horizons=normalized_horizons,
        errors=errors,
        warnings=warnings,
        metadata=metadata,
    )


def require_stage_passed(
    stage_name: str,
    source: Any,
) -> None:
    """Raise if a required research stage did not pass."""

    contract = normalize_stage_contract(
        stage_name,
        source,
    )

    if not contract.executed:
        raise ValueError(
            f"Required stage '{stage_name}' was not executed."
        )

    if not contract.passed:
        details = "; ".join(
            contract.errors
        )

        if details:
            raise ValueError(
                f"Required stage '{stage_name}' failed: "
                f"{details}"
            )

        raise ValueError(
            f"Required stage '{stage_name}' failed."
        )


def require_final_holdout(
    source: Any,
) -> None:
    """Raise unless final holdout evidence is explicitly present."""

    if not extract_final_holdout_used(
        source
    ):
        raise ValueError(
            "Final holdout evidence is required "
            "before production approval."
        )


def require_production_approval(
    source: Any,
) -> None:
    """Raise unless explicit production approval exists."""

    if not extract_approved(
        source
    ):
        raise ValueError(
            "Explicit production approval is required."
        )


def validate_horizon_consistency(
    sources: Mapping[str, Any],
    horizons: Sequence[int],
) -> None:
    """
    Verify that required horizons exist consistently across stages.

    Missing horizon evidence is treated as an integration error.
    """

    required = {
        int(horizon)
        for horizon in horizons
    }

    if not required:
        raise ValueError(
            "At least one horizon is required."
        )

    problems: list[str] = []

    for stage_name, source in sources.items():
        available = set(
            extract_horizons(
                source
            )
        )

        if available:
            missing = (
                required - available
            )

            if missing:
                problems.append(
                    f"{stage_name} is missing horizons: "
                    f"{sorted(missing)}"
                )

    if problems:
        raise ValueError(
            "Horizon consistency check failed: "
            + " | ".join(problems)
        )


def validate_research_provenance(
    source: Any,
) -> None:
    """
    Ensure a research result has not already been marked as production
    approved before reaching the final approval gate.
    """

    metadata = extract_metadata(
        source
    )

    if (
        metadata.get(
            "production_approved"
        )
        is True
    ):
        raise ValueError(
            "Research result already contains production approval."
        )

    if (
        metadata.get(
            "model_fitted_here"
        )
        is True
    ):
        raise ValueError(
            "Integration result must not fit a model."
        )

    if (
        metadata.get(
            "threshold_optimized_here"
        )
        is True
    ):
        raise ValueError(
            "Integration result must not optimize thresholds."
        )

    if (
        metadata.get(
            "feature_selection_here"
        )
        is True
    ):
        raise ValueError(
            "Integration result must not perform feature selection."
        )


def build_stage_contracts(
    stages: Mapping[str, Any],
    horizons: Sequence[int] = (),
) -> dict[str, StageContract]:
    """Normalize several research-stage outputs at once."""

    return {
        name: normalize_stage_contract(
            name,
            output,
            horizons=horizons,
        )
        for name, output in stages.items()
    }


def research_pipeline_passed(
    stages: Mapping[str, Any],
    required_stages: Sequence[str],
) -> bool:
    """
    Return True only when every required stage explicitly passed.

    Missing stages always fail.
    """

    if not required_stages:
        return False

    for stage_name in required_stages:
        if stage_name not in stages:
            return False

        contract = normalize_stage_contract(
            stage_name,
            stages[stage_name],
        )

        if not contract.executed:
            return False

        if not contract.passed:
            return False

    return True


def production_pipeline_approved(
    *,
    approval_result: Any,
    final_holdout_result: Any,
    required_horizons: Sequence[int],
) -> bool:
    """
    Strict final production predicate.

    Approval requires:
        - final holdout evidence,
        - explicit production approval,
        - every required horizon approved.

    No probability or performance value is inferred here.
    """

    if not extract_final_holdout_used(
        final_holdout_result
    ):
        return False

    if not extract_approved(
        approval_result
    ):
        return False

    for horizon in required_horizons:
        if not normalize_horizon_pass(
            approval_result,
            int(horizon),
        ):
            return False

    return True


__all__ = [
    "StageContract",
    "HorizonContract",
    "extract_value",
    "extract_metadata",
    "extract_errors",
    "extract_warnings",
    "extract_horizons",
    "extract_results_mapping",
    "get_horizon_result",
    "extract_passed",
    "extract_approved",
    "extract_production_ready",
    "extract_final_holdout_used",
    "normalize_horizon_pass",
    "normalize_horizon_contract",
    "normalize_stage_contract",
    "require_stage_passed",
    "require_final_holdout",
    "require_production_approval",
    "validate_horizon_consistency",
    "validate_research_provenance",
    "build_stage_contracts",
    "research_pipeline_passed",
    "production_pipeline_approved",
]
