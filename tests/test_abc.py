"""End-to-end tests for Artificial Bee Colony via the Python API."""

import numpy as np
import pytest

from colonyx import AutoColony
from colonyx._colonyx import BeeColony


def sphere(x):
    return sum(xi * xi for xi in x)


# --- direct Rust binding -------------------------------------------------

def test_binding_minimizes_sphere():
    abc = BeeColony(n_bees=40, n_iterations=200, limit=20, random_state=42)
    abc.fit(sphere, [-5, -5, -5], [5, 5, 5])
    assert abc.score() < 1e-2
    assert np.allclose(abc.predict(), [0, 0, 0], atol=1e-1)


def test_binding_respects_bounds():
    abc = BeeColony(n_bees=20, n_iterations=60, limit=10, random_state=3)
    abc.fit(sphere, [2, 2], [5, 5])  # optimum (origin) lies outside the box
    pos = abc.predict()
    assert all(2 <= v <= 5 for v in pos)


def test_binding_reproducible_with_seed():
    a = BeeColony(random_state=7)
    a.fit(sphere, [-5, -5], [5, 5])
    b = BeeColony(random_state=7)
    b.fit(sphere, [-5, -5], [5, 5])
    assert a.predict() == b.predict()


def test_binding_predict_before_fit_raises():
    with pytest.raises(ValueError):
        BeeColony().predict()


def test_binding_bad_objective_raises():
    with pytest.raises(TypeError):
        BeeColony().fit(lambda x: "not a number", [-1, -1], [1, 1])


# --- AutoColony facade ---------------------------------------------------

def test_autocolony_abc_minimizes_sphere():
    opt = AutoColony(mode="abc", n_iterations=200, random_state=42, n_bees=40, limit=20)
    opt.fit(sphere, bounds=[(-5, 5), (-5, 5), (-5, 5)])
    assert opt.score() < 1e-2


def test_autocolony_abc_requires_bounds():
    with pytest.raises(ValueError):
        AutoColony(mode="abc").fit(sphere)
