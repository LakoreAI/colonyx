"""Shared sklearn-facing mixins for colonyx."""

from __future__ import annotations


class OptimizerMixin:
    """Lightweight mixin for optimization estimators.

    This does not change estimator behavior. It only provides a stable place
    for optimizer-specific helpers that can be shared by future estimators.
    """

    def get_optimization_params(self):
        """Return optimization parameters as a plain dictionary."""
        return self.get_params()

    def is_fitted(self) -> bool:
        """Return whether `fit()` has been called."""
        return bool(getattr(self, "_fitted", False))
