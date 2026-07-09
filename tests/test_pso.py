"""End-to-end tests for Particle Swarm Optimization via the Python API."""

import numpy as np
import pytest

from colonyx import AutoColony
from colonyx._colonyx import ParticleSwarm


def sphere(x):
    return sum(xi * xi for xi in x)


def rosenbrock(x):
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))


# --- direct Rust binding -------------------------------------------------

def test_binding_minimizes_sphere():
    pso = ParticleSwarm(n_particles=30, n_iterations=150, w=0.7, c1=1.5, c2=1.5, random_state=42)
    pso.fit(sphere, [-5, -5, -5], [5, 5, 5])
    assert pso.score() < 1e-4
    assert np.allclose(pso.predict(), [0, 0, 0], atol=1e-2)


def test_binding_minimizes_rosenbrock():
    pso = ParticleSwarm(n_particles=40, n_iterations=300, w=0.6, c1=1.5, c2=1.5, random_state=1)
    pso.fit(rosenbrock, [-2, -2], [2, 2])
    assert pso.score() < 1e-3
    assert np.allclose(pso.predict(), [1, 1], atol=1e-2)


def test_binding_respects_bounds():
    pso = ParticleSwarm(n_particles=20, n_iterations=50, random_state=3)
    pso.fit(sphere, [2, 2], [5, 5])  # optimum (origin) lies outside the box
    pos = pso.predict()
    assert all(2 <= v <= 5 for v in pos)


def test_binding_reproducible_with_seed():
    a = ParticleSwarm(random_state=7)
    a.fit(sphere, [-5, -5], [5, 5])
    b = ParticleSwarm(random_state=7)
    b.fit(sphere, [-5, -5], [5, 5])
    assert a.predict() == b.predict()


def test_binding_predict_before_fit_raises():
    with pytest.raises(ValueError):
        ParticleSwarm().predict()


def test_binding_bad_objective_raises():
    with pytest.raises(TypeError):
        ParticleSwarm().fit(lambda x: "not a number", [-1, -1], [1, 1])


def test_binding_mismatched_bounds_raises():
    with pytest.raises(ValueError):
        ParticleSwarm().fit(sphere, [-1, -1], [1, 1, 1])


# --- AutoColony facade ---------------------------------------------------

def test_autocolony_pso_minimizes_sphere():
    opt = AutoColony(mode="pso", n_iterations=150, random_state=42, n_particles=30, w=0.7, c1=1.5, c2=1.5)
    opt.fit(sphere, bounds=[(-5, 5), (-5, 5), (-5, 5)])
    assert opt.score() < 1e-4


def test_autocolony_auto_routes_callable_to_pso():
    opt = AutoColony(mode="auto", n_iterations=100, random_state=1)
    opt.fit(sphere, bounds=[(-5, 5), (-5, 5)])
    assert opt._algorithm_mode == "pso"


def test_autocolony_pso_requires_bounds():
    with pytest.raises(ValueError):
        AutoColony(mode="pso").fit(sphere)


def test_autocolony_pso_requires_callable():
    with pytest.raises(ValueError):
        AutoColony(mode="pso").fit([[1, 2], [3, 4]], bounds=[(-1, 1), (-1, 1)])
