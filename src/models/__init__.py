"""
Model package exports.

Keep the public SwingClassifier name for backward compatibility with
existing imports while using the Gradient Boosting implementation.
"""

from .classifier import GradientBoostingSwingClassifier

# Backward-compatible name used by the existing application.
SwingClassifier = GradientBoostingSwingClassifier

__all__ = ["SwingClassifier", "GradientBoostingSwingClassifier"]
