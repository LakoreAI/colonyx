"""Validation helpers for optimization problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np

try:
    from scipy import sparse as scipy_sparse
except Exception:  # pragma: no cover - scipy is optional at runtime
    scipy_sparse = None


class OptimizationError(ValueError):
    """Base class for optimization validation errors."""


class BoundsError(OptimizationError):
    """Raised when search bounds are malformed."""


class ObjectiveFunctionError(TypeError, OptimizationError):
    """Raised when an objective function is invalid."""


@dataclass(frozen=True, slots=True)
class Interval:
    """Numeric interval constraint."""

    lower: float | None = None
    upper: float | None = None
    inclusive: bool = True

    def validate(self, name: str, value: Any) -> None:
        if value is None:
            return
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(f"{name} must be numeric")
        numeric = float(value)
        if self.lower is not None:
            if self.inclusive:
                if numeric < self.lower:
                    raise ValueError(f"{name} must be >= {self.lower}")
            elif numeric <= self.lower:
                raise ValueError(f"{name} must be > {self.lower}")
        if self.upper is not None:
            if self.inclusive:
                if numeric > self.upper:
                    raise ValueError(f"{name} must be <= {self.upper}")
            elif numeric >= self.upper:
                raise ValueError(f"{name} must be < {self.upper}")


@dataclass(frozen=True, slots=True)
class StrOptions:
    """Enumerated string constraint."""

    options: tuple[str, ...]

    def __init__(self, options: Iterable[str]):
        object.__setattr__(self, "options", tuple(options))

    def validate(self, name: str, value: Any) -> None:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
        if value not in self.options:
            raise ValueError(f"{name} must be one of {list(self.options)}")


def check_bounds(bounds: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
    """Validate a sequence of ``(low, high)`` pairs and return split bounds."""
    arr = np.asarray(bounds, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise BoundsError(
            "bounds must be a sequence of (low, high) pairs, "
            f"e.g. [(-5, 5), (-5, 5)]; got shape {arr.shape}"
        )

    if arr.shape[0] < 1:
        raise BoundsError("bounds must have at least one dimension")

    lower = arr[:, 0].tolist()
    upper = arr[:, 1].tolist()
    if any(low > high for low, high in zip(lower, upper)):
        raise BoundsError("each bound must have low <= high")
    return lower, upper


def check_objective_function(
    objective: Callable[[list[float]], float],
    probe_point: Sequence[float] | None = None,
) -> Callable[[list[float]], float]:
    """Validate that an objective function is callable and numeric."""
    if not callable(objective):
        raise ObjectiveFunctionError("objective must be callable")

    probe = list(probe_point) if probe_point is not None else [0.0]
    try:
        value = objective(probe)
    except Exception as exc:  # pragma: no cover - validation path
        raise ObjectiveFunctionError(f"objective function raised an error: {exc}") from exc

    if not isinstance(value, (int, float, np.floating)):
        raise ObjectiveFunctionError("objective function must return a numeric value")
    return objective


def check_graph_adjacency(matrix: Any) -> np.ndarray:
    """Validate a square adjacency/distance matrix."""
    if scipy_sparse is not None and scipy_sparse.issparse(matrix):
        raise OptimizationError("sparse graph inputs are not supported")

    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise OptimizationError(f"expected a square matrix; got shape {array.shape}")
    return array


def check_optimization_problem(X: Any, y: Any | None = None) -> dict[str, Any]:
    """Classify an optimization problem and validate its basic dimensions."""
    if callable(X):
        return {
            "problem_type": "continuous",
            "dimensions": None,
            "is_discrete": False,
            "is_supervised": False,
        }

    if scipy_sparse is not None and scipy_sparse.issparse(X):
        raise BoundsError("Sparse input is not supported")

    array = np.asarray(X)
    if array.ndim == 2 and array.shape[0] == array.shape[1]:
        return {
            "problem_type": "discrete",
            "dimensions": int(array.shape[0]),
            "is_discrete": True,
            "is_supervised": y is not None,
        }

    if array.ndim == 2:
        if y is not None and np.asarray(y).shape[0] != array.shape[0]:
            raise BoundsError(
                "X and y must have the same number of samples; "
                f"got {array.shape[0]} and {np.asarray(y).shape[0]}"
            )
        return {
            "problem_type": "tabular",
            "dimensions": int(array.shape[1]),
            "is_discrete": False,
            "is_supervised": y is not None,
        }

    raise BoundsError(
        "Unable to infer optimization problem type; expected a callable, "
        "a square distance matrix, or a 2D tabular array"
    )
