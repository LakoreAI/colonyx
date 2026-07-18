"""End-to-end tests for Firefly Algorithm via the Python API."""

import numpy as np
import pytest

from colonyx import AutoColony
from colonyx._colonyx import FireflyOptimizer


def sphere(x):
    return sum(value * value for value in x)


# --- direct Rust binding -------------------------------------------------

def test_binding_minimizes_sphere():
    fa = FireflyOptimizer(n_fireflies=20, n_iterations=120, beta0=1.0, gamma=1.0, alpha=0.2, random_state=42)
    fa.fit(sphere, [-5, -5, -5], [5, 5, 5])
    assert fa.score() < 1e-2
    assert np.allclose(fa.predict(), [0, 0, 0], atol=1e-1)


def test_binding_respects_bounds():
    fa = FireflyOptimizer(n_fireflies=10, n_iterations=60, random_state=3)
    fa.fit(sphere, [2, 2], [5, 5])
    pos = fa.predict()
    assert all(2 <= v <= 5 for v in pos)


def test_binding_reproducible_with_seed():
    a = FireflyOptimizer(random_state=7)
    a.fit(sphere, [-5, -5], [5, 5])
    b = FireflyOptimizer(random_state=7)
    b.fit(sphere, [-5, -5], [5, 5])
    assert a.predict() == b.predict()


def test_binding_predict_before_fit_raises():
    with pytest.raises(ValueError):
        FireflyOptimizer().predict()


def test_binding_bad_objective_raises():
    with pytest.raises(TypeError):
        FireflyOptimizer().fit(lambda x: "not a number", [-1, -1], [1, 1])


def test_binding_mismatched_bounds_raises():
    with pytest.raises(ValueError):
        FireflyOptimizer().fit(sphere, [-1, -1], [1, 1, 1])


# --- AutoColony facade ---------------------------------------------------

def test_autocolony_fa_minimizes_sphere():
    optimizer = AutoColony(mode="fa", n_iterations=120, random_state=42, n_fireflies=20)
    optimizer.fit(sphere, bounds=[(-5, 5), (-5, 5), (-5, 5)])
    assert optimizer.best_score_ is not None
    assert optimizer.best_score_ < 1e-2
    assert len(optimizer.best_solution_) == 3


def test_autocolony_fa_requires_bounds():
    with pytest.raises(ValueError):
        AutoColony(mode="fa").fit(sphere)


def test_autocolony_fa_requires_callable():
    with pytest.raises(ValueError):
        AutoColony(mode="fa").fit([[1, 2], [3, 4]], bounds=[(-1, 1), (-1, 1)])
