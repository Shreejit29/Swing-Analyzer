"""
Model registry for AI Swing Analyser.

The registry provides controlled management of trained model artifacts.

It prevents the application from blindly loading the newest model and
allows only explicitly approved models to be used for production-style
inference.

Important:
    A high accuracy score alone is NOT sufficient for approval.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import json


REGISTRY_VERSION = "1.0"


class RegistryError(RuntimeError):
    """Base exception for model registry errors."""


class ModelNotFoundError(RegistryError):
    """Raised when a requested model cannot be found."""


class ModelApprovalError(RegistryError):
    """Raised when an invalid approval operation is attempted."""


@dataclass
class ModelRegistryEntry:
    """
    Metadata describing one registered model artifact.
    """

    model_id: str

    artifact_path: str

    experiment_id: str

    version: str

    symbol: str

    timeframe: str

    horizon: int

    target: str

    created_at: str

    status: str = "research"

    validation_passed: bool = False
    final_holdout_passed: bool = False
    leakage_check_passed: bool = False
    calibration_passed: bool = False
    range_validation_passed: bool = False
    regime_validation_passed: bool = False
    backtest_passed: bool = False

    validation_accuracy: Optional[float] = None
    holdout_accuracy: Optional[float] = None

    feature_schema_hash: Optional[str] = None

    notes: str = ""

    metadata: Dict[str, object] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError(
                "model_id cannot be empty."
            )

        if not self.artifact_path.strip():
            raise ValueError(
                "artifact_path cannot be empty."
            )

        if not self.experiment_id.strip():
            raise ValueError(
                "experiment_id cannot be empty."
            )

        if self.horizon <= 0:
            raise ValueError(
                "horizon must be positive."
            )

        if self.validation_accuracy is not None:
            if not 0.0 <= self.validation_accuracy <= 1.0:
                raise ValueError(
                    "validation_accuracy must be between 0 and 1."
                )

        if self.holdout_accuracy is not None:
            if not 0.0 <= self.holdout_accuracy <= 1.0:
                raise ValueError(
                    "holdout_accuracy must be between 0 and 1."
                )

    @property
    def is_approved(self) -> bool:
        """Return whether the model is production-approved."""

        return self.status == "approved"

    @property
    def is_rejected(self) -> bool:
        """Return whether the model has been rejected."""

        return self.status == "rejected"

    @property
    def is_research(self) -> bool:
        """Return whether the model remains research-only."""

        return self.status == "research"

    def approval_requirements(self) -> Dict[str, bool]:
        """Return all approval requirements."""

        return {
            "validation_passed": self.validation_passed,
            "final_holdout_passed": self.final_holdout_passed,
            "leakage_check_passed": self.leakage_check_passed,
            "calibration_passed": self.calibration_passed,
            "range_validation_passed": self.range_validation_passed,
            "regime_validation_passed": self.regime_validation_passed,
            "backtest_passed": self.backtest_passed,
        }

    def approval_ready(self) -> bool:
        """Return True only if every required gate has passed."""

        requirements = self.approval_requirements()

        return all(requirements.values())

    def to_dict(self) -> Dict[str, object]:
        """Serialize the registry entry."""

        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, object],
    ) -> "ModelRegistryEntry":
        """Create an entry from serialized data."""

        return cls(**data)


class ModelRegistry:
    """
    Persistent registry for model artifacts.

    Registry storage is a JSON file so it remains human-readable and easy
    to version-control during research.

    Example:

        registry = ModelRegistry("models/registry.json")

        registry.register(entry)

        registry.mark_validation(
            model_id,
            passed=True,
            accuracy=0.96,
        )

        registry.approve(model_id)
    """

    def __init__(
        self,
        registry_path: str | Path = "models/registry.json",
    ) -> None:
        self.registry_path = Path(registry_path)

        self.registry_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._entries: Dict[str, ModelRegistryEntry] = {}

        self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load registry from disk."""

        if not self.registry_path.exists():
            self._entries = {}
            return

        try:
            with self.registry_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

        except (OSError, json.JSONDecodeError) as exc:
            raise RegistryError(
                f"Unable to load model registry: {exc}"
            ) from exc

        registry_version = data.get(
            "registry_version",
            "unknown",
        )

        if registry_version != REGISTRY_VERSION:
            raise RegistryError(
                "Unsupported registry version: "
                f"{registry_version}"
            )

        entries = data.get(
            "models",
            [],
        )

        self._entries = {}

        for item in entries:
            entry = ModelRegistryEntry.from_dict(item)
            self._entries[entry.model_id] = entry

    def save(self) -> None:
        """Persist registry to disk."""

        payload = {
            "registry_version": REGISTRY_VERSION,
            "updated_at": self._utc_now(),
            "models": [
                entry.to_dict()
                for entry in self._entries.values()
            ],
        }

        temporary_path = self.registry_path.with_suffix(
            ".tmp"
        )

        try:
            with temporary_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    payload,
                    file,
                    indent=2,
                    sort_keys=True,
                )

            temporary_path.replace(
                self.registry_path
            )

        except OSError as exc:
            raise RegistryError(
                f"Unable to save model registry: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        entry: ModelRegistryEntry,
        *,
        overwrite: bool = False,
    ) -> None:
        """
        Register a model.

        By default, an existing model ID cannot be overwritten.
        """

        if (
            entry.model_id in self._entries
            and not overwrite
        ):
            raise RegistryError(
                f"Model '{entry.model_id}' is already registered."
            )

        self._entries[entry.model_id] = entry

        self.save()

    def create_entry(
        self,
        *,
        model_id: str,
        artifact_path: str,
        experiment_id: str,
        version: str,
        symbol: str,
        timeframe: str,
        horizon: int,
        target: str,
        status: str = "research",
        notes: str = "",
        metadata: Optional[Dict[str, object]] = None,
    ) -> ModelRegistryEntry:
        """Create and register a new research model entry."""

        if model_id in self._entries:
            raise RegistryError(
                f"Model '{model_id}' already exists."
            )

        entry = ModelRegistryEntry(
            model_id=model_id,
            artifact_path=artifact_path,
            experiment_id=experiment_id,
            version=version,
            symbol=symbol,
            timeframe=timeframe,
            horizon=horizon,
            target=target,
            created_at=self._utc_now(),
            status=status,
            notes=notes,
            metadata=dict(metadata or {}),
        )

        self._entries[model_id] = entry
        self.save()

        return entry

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(
        self,
        model_id: str,
    ) -> ModelRegistryEntry:
        """Retrieve a model by ID."""

        if model_id not in self._entries:
            raise ModelNotFoundError(
                f"Model '{model_id}' was not found."
            )

        return self._entries[model_id]

    def exists(
        self,
        model_id: str,
    ) -> bool:
        """Check whether a model exists."""

        return model_id in self._entries

    def all_models(self) -> List[ModelRegistryEntry]:
        """Return all registered models."""

        return list(self._entries.values())

    def research_models(self) -> List[ModelRegistryEntry]:
        """Return models still in research status."""

        return [
            entry
            for entry in self._entries.values()
            if entry.status == "research"
        ]

    def approved_models(self) -> List[ModelRegistryEntry]:
        """Return approved models only."""

        return [
            entry
            for entry in self._entries.values()
            if entry.status == "approved"
        ]

    def rejected_models(self) -> List[ModelRegistryEntry]:
        """Return rejected models."""

        return [
            entry
            for entry in self._entries.values()
            if entry.status == "rejected"
        ]

    # ------------------------------------------------------------------
    # Validation state
    # ------------------------------------------------------------------

    def mark_validation(
        self,
        model_id: str,
        *,
        passed: bool,
        accuracy: Optional[float] = None,
    ) -> ModelRegistryEntry:
        """
        Record historical validation status.

        This does not approve the model.
        """

        entry = self.get(model_id)

        if accuracy is not None:
            if not 0.0 <= accuracy <= 1.0:
                raise ValueError(
                    "accuracy must be between 0 and 1."
                )

        entry.validation_passed = bool(passed)
        entry.validation_accuracy = accuracy

        self.save()

        return entry

    def mark_final_holdout(
        self,
        model_id: str,
        *,
        passed: bool,
        accuracy: Optional[float] = None,
    ) -> ModelRegistryEntry:
        """
        Record final untouched holdout evaluation.

        The holdout should only be evaluated after model selection has
        been frozen.
        """

        entry = self.get(model_id)

        if accuracy is not None:
            if not 0.0 <= accuracy <= 1.0:
                raise ValueError(
                    "accuracy must be between 0 and 1."
                )

        entry.final_holdout_passed = bool(passed)
        entry.holdout_accuracy = accuracy

        self.save()

        return entry

    def mark_leakage_check(
        self,
        model_id: str,
        *,
        passed: bool,
    ) -> ModelRegistryEntry:
        """Record data-leakage audit status."""

        entry = self.get(model_id)

        entry.leakage_check_passed = bool(passed)

        self.save()

        return entry

    def mark_calibration(
        self,
        model_id: str,
        *,
        passed: bool,
    ) -> ModelRegistryEntry:
        """Record probability-calibration status."""

        entry = self.get(model_id)

        entry.calibration_passed = bool(passed)

        self.save()

        return entry

    def mark_range_validation(
        self,
        model_id: str,
        *,
        passed: bool,
    ) -> ModelRegistryEntry:
        """Record predicted-range validation status."""

        entry = self.get(model_id)

        entry.range_validation_passed = bool(passed)

        self.save()

        return entry

    def mark_regime_validation(
        self,
        model_id: str,
        *,
        passed: bool,
    ) -> ModelRegistryEntry:
        """Record regime-stability validation status."""

        entry = self.get(model_id)

        entry.regime_validation_passed = bool(passed)

        self.save()

        return entry

    def mark_backtest(
        self,
        model_id: str,
        *,
        passed: bool,
    ) -> ModelRegistryEntry:
        """Record historical trading backtest status."""

        entry = self.get(model_id)

        entry.backtest_passed = bool(passed)

        self.save()

        return entry

    # ------------------------------------------------------------------
    # Approval
    # ------------------------------------------------------------------

    def approve(
        self,
        model_id: str,
    ) -> ModelRegistryEntry:
        """
        Approve a model for production-style inference.

        Every required validation gate must pass.

        Accuracy alone can never approve a model.
        """

        entry = self.get(model_id)

        if entry.status == "approved":
            return entry

        requirements = entry.approval_requirements()

        failed = [
            name
            for name, passed in requirements.items()
            if not passed
        ]

        if failed:
            raise ModelApprovalError(
                "Model cannot be approved. Failed requirements: "
                + ", ".join(failed)
            )

        entry.status = "approved"

        self.save()

        return entry

    def reject(
        self,
        model_id: str,
        *,
        reason: str = "",
    ) -> ModelRegistryEntry:
        """Reject a model."""

        entry = self.get(model_id)

        entry.status = "rejected"

        if reason:
            entry.notes = reason

        self.save()

        return entry

    def return_to_research(
        self,
        model_id: str,
        *,
        reason: str = "",
    ) -> ModelRegistryEntry:
        """
        Move a model back to research status.

        This invalidates its production approval.
        """

        entry = self.get(model_id)

        entry.status = "research"

        if reason:
            entry.notes = reason

        self.save()

        return entry

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def find(
        self,
        *,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        horizon: Optional[int] = None,
        target: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[ModelRegistryEntry]:
        """
        Find models matching supplied criteria.
        """

        results: Iterable[
            ModelRegistryEntry
        ] = self._entries.values()

        if symbol is not None:
            results = [
                entry
                for entry in results
                if entry.symbol == symbol
            ]

        if timeframe is not None:
            results = [
                entry
                for entry in results
                if entry.timeframe == timeframe
            ]

        if horizon is not None:
            results = [
                entry
                for entry in results
                if entry.horizon == horizon
            ]

        if target is not None:
            results = [
                entry
                for entry in results
                if entry.target == target
            ]

        if status is not None:
            results = [
                entry
                for entry in results
                if entry.status == status
            ]

        return list(results)

    def latest_approved(
        self,
        *,
        symbol: str,
        timeframe: str,
        horizon: int,
        target: str,
    ) -> Optional[ModelRegistryEntry]:
        """
        Return the most recently created approved model matching
        the requested configuration.
        """

        candidates = self.find(
            symbol=symbol,
            timeframe=timeframe,
            horizon=horizon,
            target=target,
            status="approved",
        )

        if not candidates:
            return None

        candidates.sort(
            key=lambda entry: entry.created_at,
            reverse=True,
        )

        return candidates[0]

    # ------------------------------------------------------------------
    # Artifact safety
    # ------------------------------------------------------------------

    def validate_artifact_path(
        self,
        model_id: str,
    ) -> Path:
        """
        Validate that a registered artifact exists.

        The registry path itself is not enough to approve a model.
        """

        entry = self.get(model_id)

        path = Path(entry.artifact_path)

        if not path.exists():
            raise RegistryError(
                f"Artifact for model '{model_id}' does not exist: "
                f"{path}"
            )

        if not path.is_file():
            raise RegistryError(
                f"Artifact path is not a file: {path}"
            )

        return path

    def assert_approved(
        self,
        model_id: str,
    ) -> ModelRegistryEntry:
        """
        Return a model only if it is explicitly approved.

        This method should be used by live/prediction code.
        """

        entry = self.get(model_id)

        if not entry.is_approved:
            raise ModelApprovalError(
                f"Model '{model_id}' is not approved. "
                f"Current status: {entry.status}"
            )

        self.validate_artifact_path(model_id)

        return entry

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, object]:
        """Return registry summary."""

        models = self.all_models()

        return {
            "registry_version": REGISTRY_VERSION,
            "registry_path": str(
                self.registry_path
            ),
            "total_models": len(models),
            "research_models": len(
                self.research_models()
            ),
            "approved_models": len(
                self.approved_models()
            ),
            "rejected_models": len(
                self.rejected_models()
            ),
        }

    def to_records(self) -> List[Dict[str, object]]:
        """Return all entries as serializable records."""

        return [
            entry.to_dict()
            for entry in self.all_models()
        ]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(
            timezone.utc
        ).isoformat()


def create_model_id(
    *,
    symbol: str,
    timeframe: str,
    horizon: int,
    version: str,
) -> str:
    """
    Create a deterministic human-readable model ID.
    """

    clean_symbol = symbol.replace(
        "^",
        "",
    ).replace(
        "/",
        "_",
    )

    return (
        f"{clean_symbol}_"
        f"{timeframe}_"
        f"H{horizon}_"
        f"V{version}"
    )


def registry_summary(
    registry: ModelRegistry,
) -> Dict[str, object]:
    """Convenience wrapper for registry summary."""

    return registry.summary()


__all__ = [
    "REGISTRY_VERSION",
    "RegistryError",
    "ModelNotFoundError",
    "ModelApprovalError",
    "ModelRegistryEntry",
    "ModelRegistry",
    "create_model_id",
    "registry_summary",
]
