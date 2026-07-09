"""Tests for utility validation helpers."""

import numpy as np
import pytest

from colonyx import check_bounds, check_objective_function, check_optimization_problem


def sphere(x):
    return sum(value * value for value in x)


def test_check_bounds_splits_and_validates():
    lower, upper = check_bounds([(-5, 5), (-2, 3)])

    assert lower == [-5.0, -2.0]
    assert upper == [5.0, 3.0]


def test_check_bounds_rejects_invalid_shape():
    with pytest.raises(ValueError):
        check_bounds([1, 2, 3])


def test_check_objective_function_validates_numeric_output():
    assert check_objective_function(sphere) is sphere


def test_check_objective_function_rejects_non_numeric_output():
    with pytest.raises(TypeError):
        check_objective_function(lambda x: "not numeric")


def test_check_optimization_problem_detects_problem_types():
    assert check_optimization_problem(sphere)["problem_type"] == "continuous"
    assert check_optimization_problem(np.eye(3))["problem_type"] == "discrete"
    assert check_optimization_problem(np.ones((4, 2)), y=np.arange(4))["problem_type"] == "tabular"


def test_check_optimization_problem_rejects_dimension_mismatch():
    with pytest.raises(ValueError):
        check_optimization_problem(np.ones((4, 2)), y=np.arange(3))
