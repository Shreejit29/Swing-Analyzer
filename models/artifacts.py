"""
Model artifact management for AI Swing Analyser.

Stores and restores complete trained-model packages.

An artifact contains:

    - Model
    - Preprocessor
    - Feature names
    - Target information
    - Training metadata
    - Validation metadata
    - Experiment ID
    - Configuration

IMPORTANT:

A model artifact must never be considered deployable merely
because it can be loaded successfully.

Production approval must come from the research validation gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import hashlib
import json

import joblib
import numpy as np


ARTIFACT_VERSION = "1.0"


@dataclass
class ArtifactMetadata:
    """
    Metadata describing a trained model artifact.
    """

    artifact_version: str

    experiment_id: str

    symbol: str

    model_name: str

    target_name: str

    horizon: int

    feature_names: list[str]

    training_start: str

    training_end: str

    validation_start: str

    validation_end: str

    test_start: str

    test_end: str

    created_at: str

    random_state: int = 42

    production_approved: bool = False

    validation_passed: bool = False

    final_holdout_passed: bool = False

    notes: list[str] = field(
        default_factory=list
    )


@dataclass
class ModelArtifact:
    """
    Complete serialized model package.
    """

    metadata: ArtifactMetadata

    model: Any

    preprocessor: Any


def _hash_feature_names(
    feature_names: list[str],
) -> str:
    """
    Create deterministic feature-schema hash.
    """

    payload = json.dumps(
        feature_names,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def validate_metadata(
    metadata: ArtifactMetadata,
) -> None:
    """Validate artifact metadata."""

    if (
        metadata.artifact_version
        != ARTIFACT_VERSION
    ):
        raise ValueError(
            "Unsupported artifact version: "
            f"{metadata.artifact_version}"
        )

    required = {
        "experiment_id": metadata.experiment_id,
        "symbol": metadata.symbol,
        "model_name": metadata.model_name,
        "target_name": metadata.target_name,
        "training_start": metadata.training_start,
        "training_end": metadata.training_end,
        "validation_start": metadata.validation_start,
        "validation_end": metadata.validation_end,
        "test_start": metadata.test_start,
        "test_end": metadata.test_end,
    }

    missing = [
        name
        for name, value in required.items()
        if value is None
        or str(value).strip() == ""
    ]

    if missing:
        raise ValueError(
            "Missing artifact metadata: "
            f"{missing}"
        )

    if metadata.horizon <= 0:
        raise ValueError(
            "Artifact horizon must be positive."
        )

    if not metadata.feature_names:
        raise ValueError(
            "Artifact must contain feature names."
        )

    if len(
        metadata.feature_names
    ) != len(
        set(metadata.feature_names)
    ):
        raise ValueError(
            "Artifact contains duplicate feature names."
        )


def validate_artifact(
    artifact: ModelArtifact,
) -> None:
    """
    Validate complete model artifact.
    """

    validate_metadata(
        artifact.metadata
    )

    if artifact.model is None:
        raise ValueError(
            "Artifact model is missing."
        )

    if artifact.preprocessor is None:
        raise ValueError(
            "Artifact preprocessor is missing."
        )


def create_metadata(
    *,
    experiment_id: str,
    symbol: str,
    model_name: str,
    target_name: str,
    horizon: int,
    feature_names: list[str],
    training_start: str,
    training_end: str,
    validation_start: str,
    validation_end: str,
    test_start: str,
    test_end: str,
    random_state: int = 42,
    production_approved: bool = False,
    validation_passed: bool = False,
    final_holdout_passed: bool = False,
    notes: Optional[list[str]] = None,
) -> ArtifactMetadata:
    """
    Create artifact metadata.
    """

    return ArtifactMetadata(
        artifact_version=ARTIFACT_VERSION,
        experiment_id=experiment_id,
        symbol=symbol,
        model_name=model_name,
        target_name=target_name,
        horizon=horizon,
        feature_names=list(
            feature_names
        ),
        training_start=str(
            training_start
        ),
        training_end=str(
            training_end
        ),
        validation_start=str(
            validation_start
        ),
        validation_end=str(
            validation_end
        ),
        test_start=str(
            test_start
        ),
        test_end=str(
            test_end
        ),
        created_at=datetime.now(
            timezone.utc
        ).isoformat(),
        random_state=random_state,
        production_approved=(
            production_approved
        ),
        validation_passed=(
            validation_passed
        ),
        final_holdout_passed=(
            final_holdout_passed
        ),
        notes=(
            list(notes)
            if notes is not None
            else []
        ),
    )


def create_artifact(
    *,
    model: Any,
    preprocessor: Any,
    metadata: ArtifactMetadata,
) -> ModelArtifact:
    """
    Package model and preprocessing pipeline.
    """

    artifact = ModelArtifact(
        metadata=metadata,
        model=model,
        preprocessor=preprocessor,
    )

    validate_artifact(
        artifact
    )

    return artifact


def artifact_to_metadata_dict(
    artifact: ModelArtifact,
) -> Dict[str, Any]:
    """
    Return serializable artifact metadata.

    The actual model and preprocessor are intentionally not
    converted to JSON.
    """

    validate_artifact(
        artifact
    )

    metadata = asdict(
        artifact.metadata
    )

    metadata[
        "feature_schema_hash"
    ] = _hash_feature_names(
        artifact.metadata.feature_names
    )

    return metadata


def save_artifact(
    artifact: ModelArtifact,
    path: str | Path,
) -> Path:
    """
    Save complete artifact to disk.

    Uses joblib because trained sklearn objects can contain
    numpy arrays and fitted preprocessing transformers.
    """

    validate_artifact(
        artifact
    )

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        raise FileExistsError(
            "Artifact already exists: "
            f"{path}"
        )

    joblib.dump(
        artifact,
        path,
        compress=3,
    )

    return path


def load_artifact(
    path: str | Path,
) -> ModelArtifact:
    """
    Load and validate model artifact.
    """

    path = Path(
        path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {path}"
        )

    artifact = joblib.load(
        path
    )

    if not isinstance(
        artifact,
        ModelArtifact,
    ):
        raise TypeError(
            "Loaded object is not a ModelArtifact."
        )

    validate_artifact(
        artifact
    )

    return artifact


def verify_feature_schema(
    artifact: ModelArtifact,
    feature_names: list[str],
) -> None:
    """
    Verify that inference features exactly match the trained schema.

    Feature order matters.
    """

    expected = list(
        artifact.metadata.feature_names
    )

    received = list(
        feature_names
    )

    if expected != received:

        missing = [
            name
            for name in expected
            if name not in received
        ]

        extra = [
            name
            for name in received
            if name not in expected
        ]

        raise ValueError(
            "Feature schema mismatch.\n"
            f"Expected feature count: {len(expected)}\n"
            f"Received feature count: {len(received)}\n"
            f"Missing features: {missing}\n"
            f"Extra features: {extra}"
        )


def transform_for_inference(
    artifact: ModelArtifact,
    features,
):
    """
    Transform inference data using the exact preprocessing object
    saved with the model.
    """

    if hasattr(
        features,
        "columns",
    ):

        verify_feature_schema(
            artifact,
            list(
                features.columns
            ),
        )

    transformed = (
        artifact.preprocessor.transform(
            features
        )
    )

    return transformed


def predict_probability(
    artifact: ModelArtifact,
    features,
) -> np.ndarray:
    """
    Generate positive-class probabilities.

    The model must expose predict_proba().
    """

    transformed = (
        transform_for_inference(
            artifact,
            features,
        )
    )

    if not hasattr(
        artifact.model,
        "predict_proba",
    ):
        raise TypeError(
            "Stored model does not expose predict_proba()."
        )

    probability = (
        artifact.model.predict_proba(
            transformed
        )
    )

    probability = np.asarray(
        probability,
        dtype=float,
    )

    if probability.ndim != 2:
        raise ValueError(
            "Model probability output must be 2-dimensional."
        )

    if probability.shape[1] != 2:
        raise ValueError(
            "Binary classification model must return two probability columns."
        )

    positive_probability = (
        probability[:, 1]
    )

    if not np.isfinite(
        positive_probability
    ).all():
        raise ValueError(
            "Model produced non-finite probabilities."
        )

    return np.clip(
        positive_probability,
        0.0,
        1.0,
    )


def predict_return(
    artifact: ModelArtifact,
    features,
) -> np.ndarray:
    """
    Generate future-return predictions.
    """

    transformed = (
        transform_for_inference(
            artifact,
            features,
        )
    )

    if not hasattr(
        artifact.model,
        "predict",
    ):
        raise TypeError(
            "Stored model does not expose predict()."
        )

    prediction = np.asarray(
        artifact.model.predict(
            transformed
        ),
        dtype=float,
    ).reshape(-1)

    if not np.isfinite(
        prediction
    ).all():
        raise ValueError(
            "Model produced non-finite predictions."
        )

    return prediction


def artifact_summary(
    artifact: ModelArtifact,
) -> Dict[str, Any]:
    """
    Return compact artifact information for dashboards.
    """

    validate_artifact(
        artifact
    )

    metadata = artifact.metadata

    return {
        "artifact_version": (
            metadata.artifact_version
        ),
        "experiment_id": (
            metadata.experiment_id
        ),
        "symbol": (
            metadata.symbol
        ),
        "model": (
            metadata.model_name
        ),
        "target": (
            metadata.target_name
        ),
        "horizon": (
            metadata.horizon
        ),
        "feature_count": len(
            metadata.feature_names
        ),
        "feature_schema_hash": (
            _hash_feature_names(
                metadata.feature_names
            )
        ),
        "validation_passed": (
            metadata.validation_passed
        ),
        "final_holdout_passed": (
            metadata.final_holdout_passed
        ),
        "production_approved": (
            metadata.production_approved
        ),
        "created_at": (
            metadata.created_at
        ),
    }
