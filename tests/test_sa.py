"""End-to-end tests for Simulated Annealing via the Python API."""

import numpy as np
import pytest

from colonyx import AutoColony
from colonyx._colonyx import SimulatedAnnealing


def sphere(x):
    return sum(value * value for value in x)


# --- direct Rust binding -------------------------------------------------

def test_binding_minimizes_sphere():
    sa = SimulatedAnnealing(initial_temperature=10.0, cooling_rate=0.95, step_scale=0.1, n_iterations=200, random_state=42)
    sa.fit(sphere, [-5, -5, -5], [5, 5, 5])
    assert sa.score() < 2.0
    assert len(sa.predict()) == 3


def test_binding_respects_bounds():
    sa = SimulatedAnnealing(n_iterations=100, random_state=3)
    sa.fit(sphere, [2, 2], [5, 5])
    pos = sa.predict()
    assert all(2 <= v <= 5 for v in pos)


def test_binding_reproducible_with_seed():
    a = SimulatedAnnealing(random_state=7)
    a.fit(sphere, [-5, -5], [5, 5])
    b = SimulatedAnnealing(random_state=7)
    b.fit(sphere, [-5, -5], [5, 5])
    assert a.predict() == b.predict()


def test_binding_predict_before_fit_raises():
    with pytest.raises(ValueError):
        SimulatedAnnealing().predict()


def test_binding_bad_objective_raises():
    with pytest.raises(TypeError):
        SimulatedAnnealing().fit(lambda x: "not a number", [-1, -1], [1, 1])


def test_binding_mismatched_bounds_raises():
    with pytest.raises(ValueError):
        SimulatedAnnealing().fit(sphere, [-1, -1], [1, 1, 1])


# --- AutoColony facade ---------------------------------------------------

def test_autocolony_sa_minimizes_sphere():
    optimizer = AutoColony(mode="sa", n_iterations=200, random_state=42)
    optimizer.fit(sphere, bounds=[(-5, 5), (-5, 5), (-5, 5)])
    assert optimizer.best_score_ is not None
    assert optimizer.best_score_ < 2.0
    assert len(optimizer.best_solution_) == 3


def test_autocolony_sa_requires_bounds():
    with pytest.raises(ValueError):
        AutoColony(mode="sa").fit(sphere)


def test_autocolony_sa_requires_callable():
    with pytest.raises(ValueError):
        AutoColony(mode="sa").fit([[1, 2], [3, 4]], bounds=[(-1, 1), (-1, 1)])
