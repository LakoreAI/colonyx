"""Tests for shared base classes and validation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from colonyx import BaseOptimizer, ContinuousProblem, DiscreteProblem
from colonyx.utils import BoundsError, Interval, ObjectiveFunctionError, OptimizationError, StrOptions, check_bounds, check_graph_adjacency, check_objective_function


class DummyOptimizer(BaseOptimizer):
    """Minimal concrete optimizer for testing the base class."""

    def fit(self, X, y=None, **fit_params):
        self._fitted = True
        self.best_value_ = 1.0
        return self

    def score(self, X=None, y=None):
        self._check_is_fitted()
        return self.best_value_


def test_base_optimizer_validation_and_repr():
    optimizer = DummyOptimizer(mode="auto", n_iterations=10, random_state=1)
    assert "DummyOptimizer" in repr(optimizer)
    assert optimizer.get_params()["n_iterations"] == 10

    with pytest.raises(ValueError):
        DummyOptimizer(mode="invalid", n_iterations=10, random_state=1)


def test_interval_and_stroptions_constraints():
    interval = Interval(1, 10)
    interval.validate("n_iterations", 5)
    with pytest.raises(ValueError):
        interval.validate("n_iterations", 0)

    options = StrOptions(("a", "b"))
    options.validate("mode", "a")
    with pytest.raises(ValueError):
        options.validate("mode", "c")


def test_problem_descriptors():
    continuous = ContinuousProblem(
        name="sphere",
        objective=lambda x: float(np.sum(np.asarray(x, dtype=float) ** 2)),
        bounds=((-5.0, 5.0), (-5.0, 5.0)),
    )
    assert continuous.evaluate([1.0, 2.0]) == 5.0

    matrix = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ]
    )
    discrete = DiscreteProblem(name="toy-tsp", distance_matrix=matrix)
    assert discrete.evaluate([0, 1, 2]) == 6.0


def test_validation_helpers_raise_useful_errors():
    lower, upper = check_bounds([(-1, 1), (-2, 2)])
    assert lower == [-1.0, -2.0]
    assert upper == [1.0, 2.0]

    with pytest.raises(OptimizationError):
        check_graph_adjacency(np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]]))

    with pytest.raises(ObjectiveFunctionError):
        check_objective_function(lambda x: "not-numeric")
