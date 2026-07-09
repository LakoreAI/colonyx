"""Validation helpers for optimization problems."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

try:
    from scipy import sparse as scipy_sparse
except Exception:  # pragma: no cover - scipy is available in sklearn envs, but keep fallback safe.
    scipy_sparse = None


def check_bounds(bounds: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
    """Validate a sequence of ``(low, high)`` pairs and return split bounds."""
    arr = np.asarray(bounds, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            "bounds must be a sequence of (low, high) pairs, "
            f"e.g. [(-5, 5), (-5, 5)]; got shape {arr.shape}"
        )

    lower = arr[:, 0].tolist()
    upper = arr[:, 1].tolist()
    if not lower:
        raise ValueError("bounds must have at least one dimension")
    if any(low > high for low, high in zip(lower, upper)):
        raise ValueError("each bound must have low <= high")
    return lower, upper


def check_objective_function(
    objective: Callable[[list[float]], float],
    probe_point: Sequence[float] | None = None,
) -> Callable[[list[float]], float]:
    """Validate that an objective function is callable and numeric."""
    if not callable(objective):
        raise TypeError("objective must be callable")

    probe = list(probe_point) if probe_point is not None else [0.0]
    value = objective(probe)
    if not isinstance(value, (int, float, np.floating)):
        raise TypeError("objective function must return a numeric value")
    return objective


def check_optimization_problem(X: Any, y: Any | None = None) -> dict[str, Any]:
    """Classify an optimization problem and validate its basic dimensions."""
    if scipy_sparse is not None and scipy_sparse.issparse(X):
        raise ValueError("Sparse input is not supported")

    if callable(X):
        return {
            "problem_type": "continuous",
            "dimensions": None,
            "is_discrete": False,
            "is_supervised": False,
        }

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
            raise ValueError(
                "X and y must have the same number of samples; "
                f"got {array.shape[0]} and {np.asarray(y).shape[0]}"
            )
        return {
            "problem_type": "tabular",
            "dimensions": int(array.shape[1]),
            "is_discrete": False,
            "is_supervised": y is not None,
        }

    raise ValueError(
        "Unable to infer optimization problem type; expected a callable, "
        "a square distance matrix, or a 2D tabular array"
    )
