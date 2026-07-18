"""Shared base classes for colonyx estimators and optimization problems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils._tags import InputTags, Tags, TargetTags, TransformerTags

from .utils import (
    BoundsError,
    Interval,
    ObjectiveFunctionError,
    StrOptions,
    check_bounds,
    check_objective_function,
    check_optimization_problem,
)


class OptimizerMixin:
    """Lightweight mixin for optimization estimators."""

    def get_optimization_params(self):
        """Return optimization parameters as a plain dictionary."""
        return self.get_params()

    def is_fitted(self) -> bool:
        """Return whether ``fit()`` has been called."""
        return bool(getattr(self, "_fitted", False))


@dataclass(frozen=True, slots=True)
class BaseProblem(ABC):
    """Common metadata shared by optimization problem descriptors."""

    name: str
    description: str = ""
    dimensions: int | None = None
    optimal_value: float | None = None
    bounds: tuple[tuple[float, float], ...] | None = None
    is_discrete: bool = False

    @abstractmethod
    def evaluate(self, candidate: Sequence[float] | Sequence[int]) -> float:
        """Evaluate a candidate solution."""


@dataclass(frozen=True, slots=True)
class ContinuousProblem(BaseProblem):
    """Descriptor for a continuous minimization problem."""

    objective: Callable[[Sequence[float]], float] | None = None
    gradient: Callable[[Sequence[float]], Sequence[float]] | None = None
    is_discrete: bool = False

    def __post_init__(self) -> None:
        if self.objective is None:
            raise ValueError("continuous problems require an objective callable")
        if self.bounds is None:
            raise ValueError("continuous problems require bounds")
        check_bounds(self.bounds)
        if self.dimensions is None:
            object.__setattr__(self, "dimensions", len(self.bounds))
        check_objective_function(self.objective, probe_point=[0.0] * len(self.bounds))

    def evaluate(self, candidate: Sequence[float] | Sequence[int]) -> float:
        return float(self.objective(candidate))


@dataclass(frozen=True, slots=True)
class DiscreteProblem(BaseProblem):
    """Descriptor for a discrete graph/TSP-style problem."""

    distance_matrix: np.ndarray | None = None
    is_discrete: bool = True

    def __post_init__(self) -> None:
        if self.distance_matrix is None:
            raise ValueError("discrete problems require a distance matrix")
        matrix = np.asarray(self.distance_matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise BoundsError(
                "discrete problems require a square distance matrix; "
                f"got shape {matrix.shape}"
            )
        if self.dimensions is None:
            object.__setattr__(self, "dimensions", int(matrix.shape[0]))

    def evaluate(self, candidate: Sequence[float] | Sequence[int]) -> float:
        matrix = np.asarray(self.distance_matrix, dtype=float)
        tour = np.asarray(candidate, dtype=int)
        if tour.ndim != 1 or tour.size != matrix.shape[0]:
            raise ValueError("tour length must match the distance matrix size")
        if len(set(tour.tolist())) != tour.size:
            raise ValueError("tour must visit each node exactly once")
        distance = 0.0
        for index, next_index in zip(tour, np.roll(tour, -1)):
            distance += float(matrix[int(index), int(next_index)])
        return float(distance)


class BaseOptimizer(OptimizerMixin, BaseEstimator, ABC):
    """Abstract base class for colonyx optimizers."""

    _parameter_constraints = {
        "mode": (StrOptions(("auto", "aco", "pso", "abc", "gwo", "fa", "sa", "cs", "ba", "gso", "bfo", "de", "cmaes")),),
        "n_iterations": (Interval(1, None),),
        "random_state": (Interval(0, None),),
    }

    def __init__(self, mode: str = "auto", n_iterations: int = 100, random_state: int | None = None):
        self.mode = mode
        self.n_iterations = n_iterations
        self.random_state = random_state
        self._validate_params()

    def get_params(self, deep: bool = True):  # noqa: D401 - sklearn-compatible override
        """Return estimator parameters."""
        return super().get_params(deep=deep)

    def set_params(self, **params):  # noqa: D401 - sklearn-compatible override
        """Set estimator parameters."""
        result = super().set_params(**params)
        self._validate_params()
        return result

    def __repr__(self) -> str:
        params = ", ".join(
            f"{name}={value!r}" for name, value in sorted(self.get_params(deep=False).items())
        )
        return f"{self.__class__.__name__}({params})"

    def _more_tags(self) -> dict[str, Any]:
        return {
            "requires_y": False,
            "non_deterministic": False,
            "X_types": ["2darray", "object"],
        }

    def __sklearn_tags__(self):
        return Tags(
            estimator_type=None,
            target_tags=TargetTags(required=False),
            transformer_tags=TransformerTags(),
            requires_fit=True,
            non_deterministic=False,
            input_tags=InputTags(two_d_array=True),
        )

    def _check_is_fitted(self) -> None:
        if not getattr(self, "_fitted", False):
            raise NotFittedError("Must call fit() before using this estimator")

    def _validate_params(self) -> None:
        for name, constraints in self._parameter_constraints.items():
            if not hasattr(self, name):
                continue
            value = getattr(self, name)
            for constraint in constraints:
                constraint.validate(name, value)

        if hasattr(self, "mode"):
            valid_modes = getattr(self, "_valid_modes", None)
            if valid_modes is not None and self.mode not in valid_modes:
                raise ValueError(f"Invalid mode '{self.mode}'. Must be one of {list(valid_modes)}")

    @abstractmethod
    def fit(self, X, y=None, **fit_params):
        """Fit the estimator."""

    @abstractmethod
    def score(self, X=None, y=None):
        """Return the estimator score."""

    def validate_problem(self, X: Any, y: Any | None = None) -> dict[str, Any]:
        """Classify a candidate optimization problem."""
        return check_optimization_problem(X, y)
