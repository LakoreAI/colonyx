"""End-to-end tests for CMA-ES via the Python API."""

import numpy as np
import pytest

from colonyx import AutoColony
from colonyx._colonyx import CmaEsOptimizer


def sphere(x):
    return sum(xi * xi for xi in x)


def rosenbrock(x):
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))


# --- direct Rust binding -------------------------------------------------

def test_binding_minimizes_sphere():
    opt = CmaEsOptimizer(n_individuals=20, n_iterations=150, sigma=0.5, random_state=42)
    opt.fit(sphere, [-5, -5, -5], [5, 5, 5])
    assert opt.score() < 1e-2
    assert np.allclose(opt.predict(), [0, 0, 0], atol=1e-1)


def test_binding_respects_bounds():
    opt = CmaEsOptimizer(n_individuals=10, n_iterations=50, sigma=0.3, random_state=3)
    opt.fit(sphere, [2, 2], [5, 5])  # optimum (origin) lies outside the box
    pos = opt.predict()
    assert all(2 <= v <= 5 for v in pos)


def test_binding_reproducible_with_seed():
    a = CmaEsOptimizer(random_state=7)
    a.fit(sphere, [-5, -5], [5, 5])
    b = CmaEsOptimizer(random_state=7)
    b.fit(sphere, [-5, -5], [5, 5])
    assert a.predict() == b.predict()


def test_binding_predict_before_fit_raises():
    with pytest.raises(ValueError):
        CmaEsOptimizer().predict()


def test_binding_bad_objective_raises():
    with pytest.raises(TypeError):
        CmaEsOptimizer().fit(lambda x: "not a number", [-1, -1], [1, 1])


def test_binding_mismatched_bounds_raises():
    with pytest.raises(ValueError):
        CmaEsOptimizer().fit(sphere, [-1, -1], [1, 1, 1])


# --- AutoColony facade ---------------------------------------------------

def test_autocolony_cmaes_minimizes_sphere():
    opt = AutoColony(mode="cmaes", n_iterations=150, random_state=42, n_individuals=20, cmaes_sigma=0.5)
    opt.fit(sphere, bounds=[(-5, 5), (-5, 5), (-5, 5)])
    assert opt.score() < 1e-2


def test_autocolony_cmaes_requires_bounds():
    with pytest.raises(ValueError):
        AutoColony(mode="cmaes").fit(sphere)


def test_autocolony_cmaes_requires_callable():
    with pytest.raises(ValueError):
        AutoColony(mode="cmaes").fit([[1, 2], [3, 4]], bounds=[(-1, 1), (-1, 1)])
