"""
Model package for the AI Swing Stock Analyzer.

Exports the Gradient Boosting swing classifier and its
backward-compatible alias.
"""

from .classifier import GradientBoostingSwingClassifier

# Backward-compatible name used by the rest of the application.
SwingClassifier = GradientBoostingSwingClassifier


__all__ = [
    "GradientBoostingSwingClassifier",
    "SwingClassifier",
]
