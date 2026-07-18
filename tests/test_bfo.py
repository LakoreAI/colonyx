"""End-to-end tests for Bacterial Foraging Optimization via the Python API."""

import numpy as np
import pytest

from colonyx import AutoColony
from colonyx._colonyx import BacterialForagingOptimizer


def sphere(x):
    return sum(xi * xi for xi in x)


# --- direct Rust binding -------------------------------------------------

def test_binding_minimizes_sphere():
    bfo = BacterialForagingOptimizer(
        n_bacteria=20,
        n_iterations=4,
        n_chemotactic_steps=40,
        n_reproduction_steps=4,
        step_scale=0.1,
        random_state=42,
    )
    bfo.fit(sphere, [-5, -5, -5], [5, 5, 5])
    assert bfo.score() < 1e-2
    assert np.allclose(bfo.predict(), [0, 0, 0], atol=1e-1)


def test_binding_respects_bounds():
    bfo = BacterialForagingOptimizer(
        n_bacteria=10, n_iterations=2, n_chemotactic_steps=20, random_state=3,
    )
    bfo.fit(sphere, [2, 2], [5, 5])  # optimum (origin) lies outside the box
    pos = bfo.predict()
    assert all(2 <= v <= 5 for v in pos)


def test_binding_reproducible_with_seed():
    a = BacterialForagingOptimizer(
        n_bacteria=10, n_iterations=2, n_chemotactic_steps=20, random_state=7,
    )
    a.fit(sphere, [-5, -5], [5, 5])
    b = BacterialForagingOptimizer(
        n_bacteria=10, n_iterations=2, n_chemotactic_steps=20, random_state=7,
    )
    b.fit(sphere, [-5, -5], [5, 5])
    assert a.predict() == b.predict()


def test_binding_predict_before_fit_raises():
    with pytest.raises(ValueError):
        BacterialForagingOptimizer().predict()


def test_binding_bad_objective_raises():
    with pytest.raises(TypeError):
        BacterialForagingOptimizer().fit(lambda x: "not a number", [-1, -1], [1, 1])


def test_binding_mismatched_bounds_raises():
    with pytest.raises(ValueError):
        BacterialForagingOptimizer().fit(sphere, [-1, -1], [1, 1, 1])


# --- AutoColony facade ---------------------------------------------------

def test_autocolony_bfo_minimizes_sphere():
    opt = AutoColony(
        mode="bfo", n_iterations=4, random_state=42,
        n_bacteria=20, n_chemotactic_steps=40, n_reproduction_steps=4,
    )
    opt.fit(sphere, bounds=[(-5, 5), (-5, 5), (-5, 5)])
    assert opt.score() < 1e-2


def test_autocolony_bfo_requires_bounds():
    with pytest.raises(ValueError):
        AutoColony(mode="bfo").fit(sphere)


def test_autocolony_bfo_requires_callable():
    with pytest.raises(ValueError):
        AutoColony(mode="bfo").fit([[1, 2], [3, 4]], bounds=[(-1, 1), (-1, 1)])
