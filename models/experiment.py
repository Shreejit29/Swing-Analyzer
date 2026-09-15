"""
Experiment tracking for AI Swing Analyser.

Stores reproducible metadata and evaluation results for model
experiments.

The purpose is to prevent:

    - cherry-picking
    - undocumented model changes
    - accidental comparison of different datasets
    - loss of validation history

Experiments are research records, not live trading records.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import json
import hashlib

import pandas as pd


@dataclass
class ExperimentConfig:
    """Configuration describing one experiment."""

    experiment_name: str

    symbol: str

    model_name: str

    target: str

    horizon: int

    feature_count: int

    training_start: str

    training_end: str

    validation_start: str

    validation_end: str

    test_start: str

    test_end: str

    random_state: int = 42

    notes: str = ""


@dataclass
class ExperimentResult:
    """Metrics produced by an experiment."""

    accuracy: Optional[float] = None

    balanced_accuracy: Optional[float] = None

    precision: Optional[float] = None

    recall: Optional[float] = None

    f1: Optional[float] = None

    roc_auc: Optional[float] = None

    brier_score: Optional[float] = None

    mae: Optional[float] = None

    rmse: Optional[float] = None

    directional_accuracy: Optional[float] = None

    range_coverage: Optional[float] = None

    range_width: Optional[float] = None

    total_return: Optional[float] = None

    sharpe_ratio: Optional[float] = None

    maximum_drawdown: Optional[float] = None

    win_rate: Optional[float] = None

    profit_factor: Optional[float] = None

    trades: Optional[int] = None


@dataclass
class ExperimentRecord:
    """Complete experiment record."""

    experiment_id: str

    created_at: str

    config: ExperimentConfig

    result: ExperimentResult

    validation_passed: bool = False

    final_holdout_passed: bool = False

    production_approved: bool = False

    warnings: list[str] = field(
        default_factory=list
    )


def generate_experiment_id(
    config: ExperimentConfig,
) -> str:
    """
    Generate deterministic experiment identifier.

    The ID depends on the core experiment configuration.
    """

    payload = json.dumps(
        asdict(config),
        sort_keys=True,
        default=str,
    )

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()

    return (
        f"{config.symbol.upper()}-"
        f"{config.model_name.upper()}-"
        f"{digest[:12]}"
    )


def create_experiment(
    config: ExperimentConfig,
    result: Optional[
        ExperimentResult
    ] = None,
) -> ExperimentRecord:
    """Create a new experiment record."""

    experiment_id = (
        generate_experiment_id(
            config
        )
    )

    created_at = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    return ExperimentRecord(
        experiment_id=experiment_id,
        created_at=created_at,
        config=config,
        result=(
            result
            if result is not None
            else ExperimentResult()
        ),
    )


def validate_experiment_record(
    record: ExperimentRecord,
) -> None:
    """
    Validate experiment metadata.

    An experiment without clearly defined temporal partitions
    should not be considered reproducible.
    """

    config = record.config

    required_fields = {
        "symbol": config.symbol,
        "model_name": config.model_name,
        "target": config.target,
        "training_start": config.training_start,
        "training_end": config.training_end,
        "validation_start": config.validation_start,
        "validation_end": config.validation_end,
        "test_start": config.test_start,
        "test_end": config.test_end,
    }

    missing = [
        name
        for name, value
        in required_fields.items()
        if value is None
        or str(value).strip() == ""
    ]

    if missing:
        raise ValueError(
            "Experiment is missing required metadata: "
            f"{missing}"
        )

    if config.horizon <= 0:
        raise ValueError(
            "Experiment horizon must be positive."
        )

    if config.feature_count <= 0:
        raise ValueError(
            "feature_count must be positive."
        )

    if not record.experiment_id:
        raise ValueError(
            "experiment_id cannot be empty."
        )

    if not record.created_at:
        raise ValueError(
            "created_at cannot be empty."
        )


def approve_experiment(
    record: ExperimentRecord,
    *,
    validation_passed: bool,
    final_holdout_passed: bool,
    no_leakage: bool,
    calibration_passed: bool,
    range_validation_passed: bool,
    backtest_passed: bool,
    regime_stability_passed: bool,
) -> ExperimentRecord:
    """
    Apply the production approval gate.

    Production approval requires ALL major research gates.

    There is deliberately no way to approve a model solely because
    its accuracy exceeds 95%.
    """

    record.validation_passed = (
        bool(validation_passed)
    )

    record.final_holdout_passed = (
        bool(final_holdout_passed)
    )

    record.production_approved = False

    record.warnings = []

    gates = {
        "validation": validation_passed,
        "final_holdout": final_holdout_passed,
        "no_leakage": no_leakage,
        "calibration": calibration_passed,
        "range_validation": range_validation_passed,
        "backtest": backtest_passed,
        "regime_stability": regime_stability_passed,
    }

    failed = [
        name
        for name, passed
        in gates.items()
        if not passed
    ]

    if failed:

        record.warnings.append(
            "Production approval blocked. "
            "Failed gates: "
            + ", ".join(failed)
        )

    else:

        record.production_approved = True

    return record


def experiment_to_dict(
    record: ExperimentRecord,
) -> Dict[str, Any]:
    """Convert experiment record to serializable dictionary."""

    validate_experiment_record(
        record
    )

    return asdict(
        record
    )


def save_experiment(
    record: ExperimentRecord,
    directory: str | Path = "experiments",
) -> Path:
    """
    Save an experiment as JSON.

    Existing records with the same experiment ID are not silently
    overwritten.
    """

    validate_experiment_record(
        record
    )

    directory = Path(
        directory
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        directory
        / f"{record.experiment_id}.json"
    )

    if path.exists():
        raise FileExistsError(
            "Experiment already exists: "
            f"{path}"
        )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            experiment_to_dict(
                record
            ),
            file,
            indent=2,
            ensure_ascii=False,
        )

    return path


def load_experiment(
    path: str | Path,
) -> ExperimentRecord:
    """Load an experiment JSON record."""

    path = Path(
        path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Experiment not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        payload = json.load(
            file
        )

    config = ExperimentConfig(
        **payload["config"]
    )

    result = ExperimentResult(
        **payload["result"]
    )

    record = ExperimentRecord(
        experiment_id=payload[
            "experiment_id"
        ],
        created_at=payload[
            "created_at"
        ],
        config=config,
        result=result,
        validation_passed=payload.get(
            "validation_passed",
            False,
        ),
        final_holdout_passed=payload.get(
            "final_holdout_passed",
            False,
        ),
        production_approved=payload.get(
            "production_approved",
            False,
        ),
        warnings=payload.get(
            "warnings",
            [],
        ),
    )

    validate_experiment_record(
        record
    )

    return record


def list_experiments(
    directory: str | Path = "experiments",
) -> pd.DataFrame:
    """
    Load experiment metadata into a dataframe.

    Corrupt records are reported in the dataframe instead of
    silently ignored.
    """

    directory = Path(
        directory
    )

    if not directory.exists():
        return pd.DataFrame()

    rows = []

    for path in sorted(
        directory.glob("*.json")
    ):

        try:

            record = load_experiment(
                path
            )

            config = record.config
            result = record.result

            rows.append(
                {
                    "Experiment ID": (
                        record.experiment_id
                    ),
                    "Created At": (
                        record.created_at
                    ),
                    "Symbol": (
                        config.symbol
                    ),
                    "Model": (
                        config.model_name
                    ),
                    "Target": (
                        config.target
                    ),
                    "Horizon": (
                        config.horizon
                    ),
                    "Features": (
                        config.feature_count
                    ),
                    "Accuracy": (
                        result.accuracy
                    ),
                    "Balanced Accuracy": (
                        result.balanced_accuracy
                    ),
                    "F1": (
                        result.f1
                    ),
                    "Brier Score": (
                        result.brier_score
                    ),
                    "MAE": (
                        result.mae
                    ),
                    "Range Coverage": (
                        result.range_coverage
                    ),
                    "Total Return": (
                        result.total_return
                    ),
                    "Sharpe": (
                        result.sharpe_ratio
                    ),
                    "Max Drawdown": (
                        result.maximum_drawdown
                    ),
                    "Win Rate": (
                        result.win_rate
                    ),
                    "Profit Factor": (
                        result.profit_factor
                    ),
                    "Trades": (
                        result.trades
                    ),
                    "Validation Passed": (
                        record.validation_passed
                    ),
                    "Holdout Passed": (
                        record.final_holdout_passed
                    ),
                    "Production Approved": (
                        record.production_approved
                    ),
                    "Warnings": (
                        " | ".join(
                            record.warnings
                        )
                    ),
                }
            )

        except Exception as exc:

            rows.append(
                {
                    "Experiment ID": path.stem,
                    "Error": str(exc),
                }
            )

    return pd.DataFrame(
        rows
    )


def compare_experiments(
    directory: str | Path = "experiments",
) -> pd.DataFrame:
    """
    Return experiments sorted by research quality.

    Accuracy is NOT used as the sole ranking criterion.
    """

    dataframe = list_experiments(
        directory
    )

    if dataframe.empty:
        return dataframe

    if "Error" in dataframe.columns:
        dataframe = dataframe[
            dataframe["Error"].isna()
            if "Error" in dataframe.columns
            else True
        ]

    if dataframe.empty:
        return dataframe

    ranking_columns = [
        column
        for column in [
            "Validation Passed",
            "Holdout Passed",
            "Production Approved",
            "Sharpe",
            "Accuracy",
            "Range Coverage",
        ]
        if column in dataframe.columns
    ]

    if not ranking_columns:
        return dataframe

    sort_columns = []

    ascending = []

    for column in ranking_columns:

        sort_columns.append(
            column
        )

        if column in {
            "Validation Passed",
            "Holdout Passed",
            "Production Approved",
        }:

            ascending.append(False)

        else:

            ascending.append(False)

    return dataframe.sort_values(
        sort_columns,
        ascending=ascending,
        na_position="last",
    ).reset_index(
        drop=True
    )
