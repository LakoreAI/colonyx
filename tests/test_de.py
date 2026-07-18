"""End-to-end tests for Differential Evolution via the Python API."""

import numpy as np
import pytest

from colonyx import AutoColony
from colonyx._colonyx import DifferentialEvolution


def sphere(x):
    return sum(value * value for value in x)


# --- direct Rust binding -------------------------------------------------

def test_binding_minimizes_sphere():
    de = DifferentialEvolution(n_individuals=30, n_iterations=120, f=0.8, cr=0.9, random_state=42)
    de.fit(sphere, [-5, -5, -5], [5, 5, 5])
    assert de.score() < 1e-4
    assert np.allclose(de.predict(), [0, 0, 0], atol=1e-2)


def test_binding_respects_bounds():
    de = DifferentialEvolution(n_individuals=15, n_iterations=60, random_state=3)
    de.fit(sphere, [2, 2], [5, 5])
    pos = de.predict()
    assert all(2 <= v <= 5 for v in pos)


def test_binding_reproducible_with_seed():
    a = DifferentialEvolution(random_state=7)
    a.fit(sphere, [-5, -5], [5, 5])
    b = DifferentialEvolution(random_state=7)
    b.fit(sphere, [-5, -5], [5, 5])
    assert a.predict() == b.predict()


def test_binding_predict_before_fit_raises():
    with pytest.raises(ValueError):
        DifferentialEvolution().predict()


def test_binding_bad_objective_raises():
    with pytest.raises(TypeError):
        DifferentialEvolution().fit(lambda x: "not a number", [-1, -1], [1, 1])


def test_binding_mismatched_bounds_raises():
    with pytest.raises(ValueError):
        DifferentialEvolution().fit(sphere, [-1, -1], [1, 1, 1])


# --- AutoColony facade ---------------------------------------------------

def test_autocolony_de_minimizes_sphere():
    optimizer = AutoColony(mode="de", n_iterations=120, random_state=42, n_individuals=30, f=0.8, cr=0.9)
    optimizer.fit(sphere, bounds=[(-5, 5), (-5, 5), (-5, 5)])
    assert optimizer.best_score_ is not None
    assert optimizer.best_score_ < 1e-4
    assert len(optimizer.best_solution_) == 3
    assert optimizer.predict() == optimizer.best_solution_


def test_autocolony_de_requires_bounds():
    with pytest.raises(ValueError):
        AutoColony(mode="de").fit(sphere)


def test_autocolony_de_requires_callable():
    with pytest.raises(ValueError):
        AutoColony(mode="de").fit([[1, 2], [3, 4]], bounds=[(-1, 1), (-1, 1)])
