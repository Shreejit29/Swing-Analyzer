"""
Evaluation and research-reporting package.

This package is responsible for consolidating model evaluation results
into reproducible research reports.

The evaluation layer must remain separate from model training so that
performance measurement does not accidentally influence the training
process.
"""

from .report import (
    EvaluationReport,
    EvaluationReportBuilder,
    build_evaluation_report,
)

__all__ = [
    "EvaluationReport",
    "EvaluationReportBuilder",
    "build_evaluation_report",
]
