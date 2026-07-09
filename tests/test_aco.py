"""End-to-end tests for Ant Colony Optimization via the Python API."""

import numpy as np
import pytest

from colonyx import AutoColony
from colonyx._colonyx import AntColony


def ring(n):
    """Ring graph on ``n`` cities: adjacent cost 1, else 10; optimum length n."""
    matrix = np.full((n, n), 10.0)
    for i in range(n):
        matrix[i, i] = 0.0
        matrix[i, (i + 1) % n] = 1.0
        matrix[i, (i - 1) % n] = 1.0
    return matrix


# --- direct Rust binding -------------------------------------------------

def test_binding_finds_optimal_ring():
    aco = AntColony(n_ants=15, n_iterations=30, beta=3.0, random_state=3)
    aco.fit(ring(5).tolist())
    assert sorted(aco.predict()) == list(range(5))
    assert aco.score() == 5.0


def test_binding_predict_before_fit_raises():
    with pytest.raises(ValueError):
        AntColony().predict()


def test_binding_rejects_non_square():
    with pytest.raises(ValueError):
        AntColony().fit([[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]])


# --- AutoColony facade ---------------------------------------------------

def test_autocolony_aco_finds_optimal_ring():
    opt = AutoColony(mode="aco", n_iterations=50, n_ants=20, beta=3.0, random_state=42)
    opt.fit(ring(6))
    assert sorted(opt.predict()) == list(range(6))
    assert opt.score() == 6.0


def test_autocolony_auto_routes_square_matrix_to_aco():
    opt = AutoColony(mode="auto", n_iterations=30, n_ants=15, beta=3.0, random_state=1)
    opt.fit(ring(5))
    assert sorted(opt.predict()) == list(range(5))


def test_autocolony_reproducible_with_seed():
    a = AutoColony(mode="aco", random_state=7, n_iterations=30).fit(ring(6))
    b = AutoColony(mode="aco", random_state=7, n_iterations=30).fit(ring(6))
    assert a.predict() == b.predict()
    assert a.score() == b.score()


def test_autocolony_predict_before_fit_raises():
    with pytest.raises(ValueError):
        AutoColony(mode="aco").predict()


def test_autocolony_rejects_non_square():
    with pytest.raises(ValueError):
        AutoColony(mode="aco").fit(np.zeros((3, 4)))
