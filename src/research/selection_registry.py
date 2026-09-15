"""
AI Swing Analyser — Research Selection Registry.

Stores controlled model-selection results as research evidence.

Important:
- Selection is not production approval.
- Final holdout results are not used to select models.
- Registry entries are immutable research records once written.
- Production approval remains the responsibility of the approval/registry layer.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from .selection_pipeline import SelectionPipelineResult


@dataclass(frozen=True)
class SelectionRecord:
    """Persistent record of one controlled model-selection decision."""

    selection_id: str
    created_at: str

    symbol: str
    timeframe: str
    horizon: int

    selected_model_id: Optional[str]
    selected_model_name: Optional[str]

    candidate_count: int
    successful: bool

    final_holdout_used: bool
    production_approved: bool

    validation_accuracy: Optional[float] = None
    walk_forward_mean_accuracy: Optional[float] = None
    walk_forward_min_accuracy: Optional[float] = None
    walk_forward_accuracy_std: Optional[float] = None
    walk_forward_brier: Optional[float] = None

    research_only: bool = True

    warnings: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)


class SelectionRegistry:
    """
    JSON-backed registry for controlled research model selections.
    """

    def __init__(
        self,
        root: str | Path = "experiments/selections",
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _selection_id(
        symbol: str,
        timeframe: str,
        horizon: int,
        selected_model_id: Optional[str],
        created_at: str,
    ) -> str:
        payload = "|".join(
            [
                symbol.upper(),
                timeframe.upper(),
                str(horizon),
                str(selected_model_id),
                created_at,
            ]
        )

        digest = hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()[:16]

        return f"SEL-{digest}"

    @staticmethod
    def _validate_identity(
        symbol: str,
        timeframe: str,
        horizon: int,
    ) -> None:
        if not symbol or not symbol.strip():
            raise ValueError("symbol cannot be empty.")

        if not timeframe or not timeframe.strip():
            raise ValueError("timeframe cannot be empty.")

        if horizon <= 0:
            raise ValueError("horizon must be positive.")

    @staticmethod
    def _selected_metrics(
        result: SelectionPipelineResult,
    ) -> dict[str, Optional[float]]:
        selected = result.selected_candidate

        if selected is None:
            return {
                "validation_accuracy": None,
                "walk_forward_mean_accuracy": None,
                "walk_forward_min_accuracy": None,
                "walk_forward_accuracy_std": None,
                "walk_forward_brier": None,
            }

        return {
            "validation_accuracy": selected.validation_accuracy,
            "walk_forward_mean_accuracy": (
                selected.walk_forward_mean_accuracy
            ),
            "walk_forward_min_accuracy": (
                selected.walk_forward_min_accuracy
            ),
            "walk_forward_accuracy_std": (
                selected.walk_forward_accuracy_std
            ),
            "walk_forward_brier": selected.walk_forward_brier,
        }

    def create_record(
        self,
        result: SelectionPipelineResult,
        symbol: str,
        timeframe: str,
    ) -> SelectionRecord:
        """
        Convert a selection result into a persistent research record.
        """

        self._validate_identity(
            symbol=symbol,
            timeframe=timeframe,
            horizon=result.horizon,
        )

        created_at = self._utc_now()

        selection_id = self._selection_id(
            symbol=symbol,
            timeframe=timeframe,
            horizon=result.horizon,
            selected_model_id=result.selected_model_id,
            created_at=created_at,
        )

        metrics = self._selected_metrics(result)

        metadata = dict(result.metadata)
        metadata.update(
            {
                "symbol": symbol.upper(),
                "timeframe": timeframe.upper(),
                "horizon": result.horizon,
                "candidate_count": len(result.candidates),
            }
        )

        return SelectionRecord(
            selection_id=selection_id,
            created_at=created_at,
            symbol=symbol.upper(),
            timeframe=timeframe.upper(),
            horizon=result.horizon,
            selected_model_id=result.selected_model_id,
            selected_model_name=result.selected_model_name,
            candidate_count=len(result.candidates),
            successful=result.successful,
            final_holdout_used=False,
            production_approved=False,
            validation_accuracy=metrics["validation_accuracy"],
            walk_forward_mean_accuracy=(
                metrics["walk_forward_mean_accuracy"]
            ),
            walk_forward_min_accuracy=(
                metrics["walk_forward_min_accuracy"]
            ),
            walk_forward_accuracy_std=(
                metrics["walk_forward_accuracy_std"]
            ),
            walk_forward_brier=metrics["walk_forward_brier"],
            research_only=True,
            warnings=tuple(result.warnings),
            metadata=metadata,
        )

    def save(
        self,
        record: SelectionRecord,
    ) -> Path:
        """
        Save a selection record as JSON.

        Existing records are never silently overwritten.
        """

        if not isinstance(record, SelectionRecord):
            raise TypeError("record must be a SelectionRecord.")

        path = self.root / f"{record.selection_id}.json"

        if path.exists():
            raise FileExistsError(
                f"Selection record already exists: {path}"
            )

        payload = asdict(record)

        payload["warnings"] = list(record.warnings)

        temporary_path = path.with_suffix(".tmp")

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
            )

        temporary_path.replace(path)

        return path

    def register(
        self,
        result: SelectionPipelineResult,
        symbol: str,
        timeframe: str,
    ) -> SelectionRecord:
        """
        Create and persist a selection record.
        """

        record = self.create_record(
            result=result,
            symbol=symbol,
            timeframe=timeframe,
        )

        self.save(record)

        return record

    @staticmethod
    def _record_from_dict(
        payload: dict[str, Any],
    ) -> SelectionRecord:
        required = {
            "selection_id",
            "created_at",
            "symbol",
            "timeframe",
            "horizon",
            "selected_model_id",
            "selected_model_name",
            "candidate_count",
            "successful",
            "final_holdout_used",
            "production_approved",
        }

        missing = sorted(required - set(payload))

        if missing:
            raise ValueError(
                f"Selection record is missing fields: {missing}"
            )

        if payload.get("final_holdout_used") is True:
            raise ValueError(
                "Invalid selection record: final holdout was used."
            )

        if payload.get("production_approved") is True:
            raise ValueError(
                "Selection registry cannot mark a model production-approved."
            )

        warnings = tuple(payload.get("warnings", []))

        return SelectionRecord(
            selection_id=payload["selection_id"],
            created_at=payload["created_at"],
            symbol=payload["symbol"],
            timeframe=payload["timeframe"],
            horizon=int(payload["horizon"]),
            selected_model_id=payload["selected_model_id"],
            selected_model_name=payload["selected_model_name"],
            candidate_count=int(payload["candidate_count"]),
            successful=bool(payload["successful"]),
            final_holdout_used=False,
            production_approved=False,
            validation_accuracy=payload.get("validation_accuracy"),
            walk_forward_mean_accuracy=payload.get(
                "walk_forward_mean_accuracy"
            ),
            walk_forward_min_accuracy=payload.get(
                "walk_forward_min_accuracy"
            ),
            walk_forward_accuracy_std=payload.get(
                "walk_forward_accuracy_std"
            ),
            walk_forward_brier=payload.get("walk_forward_brier"),
            research_only=bool(payload.get("research_only", True)),
            warnings=warnings,
            metadata=dict(payload.get("metadata", {})),
        )

    def load(
        self,
        selection_id: str,
    ) -> SelectionRecord:
        """Load one selection record."""

        if not selection_id:
            raise ValueError("selection_id cannot be empty.")

        path = self.root / f"{selection_id}.json"

        if not path.exists():
            raise FileNotFoundError(
                f"Selection record not found: {selection_id}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            payload = json.load(handle)

        return self._record_from_dict(payload)

    def list_records(self) -> list[SelectionRecord]:
        """Return all valid selection records."""

        records: list[SelectionRecord] = []

        for path in sorted(self.root.glob("SEL-*.json")):
            with path.open(
                "r",
                encoding="utf-8",
            ) as handle:
                payload = json.load(handle)

            records.append(
                self._record_from_dict(payload)
            )

        records.sort(
            key=lambda record: record.created_at,
            reverse=True,
        )

        return records

    def find(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        horizon: Optional[int] = None,
        selected_model_id: Optional[str] = None,
    ) -> list[SelectionRecord]:
        """
        Find selection records using optional filters.
        """

        records = self.list_records()

        if symbol is not None:
            symbol = symbol.upper()
            records = [
                record
                for record in records
                if record.symbol.upper() == symbol
            ]

        if timeframe is not None:
            timeframe = timeframe.upper()
            records = [
                record
                for record in records
                if record.timeframe.upper() == timeframe
            ]

        if horizon is not None:
            records = [
                record
                for record in records
                if record.horizon == horizon
            ]

        if selected_model_id is not None:
            records = [
                record
                for record in records
                if record.selected_model_id == selected_model_id
            ]

        return records

    def latest(
        self,
        symbol: str,
        timeframe: str,
        horizon: int,
    ) -> Optional[SelectionRecord]:
        """Return the newest selection for a symbol/timeframe/horizon."""

        records = self.find(
            symbol=symbol,
            timeframe=timeframe,
            horizon=horizon,
        )

        return records[0] if records else None

    def delete(
        self,
        selection_id: str,
    ) -> None:
        """
        Delete a research record.

        This is intended for local research cleanup only.
        """

        if not selection_id:
            raise ValueError("selection_id cannot be empty.")

        path = self.root / f"{selection_id}.json"

        if not path.exists():
            raise FileNotFoundError(
                f"Selection record not found: {selection_id}"
            )

        path.unlink()

    def summary(self) -> dict[str, Any]:
        """Return registry-level summary statistics."""

        records = self.list_records()

        selected = [
            record
            for record in records
            if record.successful
        ]

        return {
            "root": str(self.root),
            "total_records": len(records),
            "successful_selections": len(selected),
            "unsuccessful_selections": (
                len(records) - len(selected)
            ),
            "final_holdout_used": any(
                record.final_holdout_used
                for record in records
            ),
            "production_approved_records": sum(
                record.production_approved
                for record in records
            ),
            "research_only_records": sum(
                record.research_only
                for record in records
            ),
        }


def register_selection(
    result: SelectionPipelineResult,
    symbol: str,
    timeframe: str,
    root: str | Path = "experiments/selections",
) -> SelectionRecord:
    """Convenience API for registering a selection result."""

    registry = SelectionRegistry(root=root)

    return registry.register(
        result=result,
        symbol=symbol,
        timeframe=timeframe,
    )


__all__ = [
    "SelectionRecord",
    "SelectionRegistry",
    "register_selection",
]
