"""End-to-end tests for Cuckoo Search, Bat Algorithm, and Glowworm via the Python API."""

import numpy as np
import pytest

from colonyx import AutoColony
from colonyx._colonyx import BatAlgorithm, CuckooSearch, GlowwormOptimizer


def sphere(x):
    return sum(value * value for value in x)


# --- Cuckoo Search: direct Rust binding ----------------------------------

def test_cs_binding_minimizes_sphere():
    cs = CuckooSearch(n_nests=20, n_iterations=120, pa=0.25, alpha=0.01, levy_scale=1.0, random_state=42)
    cs.fit(sphere, [-5, -5, -5], [5, 5, 5])
    assert cs.score() < 1.0
    assert len(cs.predict()) == 3


def test_cs_binding_respects_bounds():
    cs = CuckooSearch(n_nests=10, n_iterations=60, random_state=3)
    cs.fit(sphere, [2, 2], [5, 5])
    pos = cs.predict()
    assert all(2 <= v <= 5 for v in pos)


def test_cs_binding_reproducible_with_seed():
    a = CuckooSearch(random_state=7)
    a.fit(sphere, [-5, -5], [5, 5])
    b = CuckooSearch(random_state=7)
    b.fit(sphere, [-5, -5], [5, 5])
    assert a.predict() == b.predict()


def test_cs_binding_predict_before_fit_raises():
    with pytest.raises(ValueError):
        CuckooSearch().predict()


# --- Bat Algorithm: direct Rust binding ----------------------------------

def test_ba_binding_minimizes_sphere():
    ba = BatAlgorithm(n_bats=20, n_iterations=120, random_state=42)
    ba.fit(sphere, [-5, -5, -5], [5, 5, 5])
    assert ba.score() < 5.0
    assert len(ba.predict()) == 3


def test_ba_binding_respects_bounds():
    ba = BatAlgorithm(n_bats=10, n_iterations=60, random_state=3)
    ba.fit(sphere, [2, 2], [5, 5])
    pos = ba.predict()
    assert all(2 <= v <= 5 for v in pos)


def test_ba_binding_reproducible_with_seed():
    a = BatAlgorithm(random_state=7)
    a.fit(sphere, [-5, -5], [5, 5])
    b = BatAlgorithm(random_state=7)
    b.fit(sphere, [-5, -5], [5, 5])
    assert a.predict() == b.predict()


def test_ba_binding_predict_before_fit_raises():
    with pytest.raises(ValueError):
        BatAlgorithm().predict()


# --- Glowworm: direct Rust binding ----------------------------------------

def test_gso_binding_minimizes_sphere():
    gso = GlowwormOptimizer(n_worms=20, n_iterations=120, random_state=42)
    gso.fit(sphere, [-5, -5, -5], [5, 5, 5])
    assert gso.score() < 10.0
    assert len(gso.predict()) == 3


def test_gso_binding_respects_bounds():
    gso = GlowwormOptimizer(n_worms=10, n_iterations=60, random_state=3)
    gso.fit(sphere, [2, 2], [5, 5])
    pos = gso.predict()
    assert all(2 <= v <= 5 for v in pos)


def test_gso_binding_reproducible_with_seed():
    a = GlowwormOptimizer(random_state=7)
    a.fit(sphere, [-5, -5], [5, 5])
    b = GlowwormOptimizer(random_state=7)
    b.fit(sphere, [-5, -5], [5, 5])
    assert a.predict() == b.predict()


def test_gso_binding_predict_before_fit_raises():
    with pytest.raises(ValueError):
        GlowwormOptimizer().predict()


# --- AutoColony facade ---------------------------------------------------

def test_autocolony_cs_minimizes_sphere():
    optimizer = AutoColony(mode="cs", n_iterations=120, random_state=42, n_nests=20)
    optimizer.fit(sphere, bounds=[(-5, 5), (-5, 5), (-5, 5)])
    assert optimizer.best_score_ is not None
    assert optimizer.best_score_ < 1.0
    assert len(optimizer.best_solution_) == 3


def test_autocolony_ba_minimizes_sphere():
    optimizer = AutoColony(mode="ba", n_iterations=120, random_state=42, n_bats=20)
    optimizer.fit(sphere, bounds=[(-5, 5), (-5, 5), (-5, 5)])
    assert optimizer.best_score_ is not None
    assert optimizer.best_score_ < 5.0
    assert len(optimizer.best_solution_) == 3


def test_autocolony_gso_minimizes_sphere():
    optimizer = AutoColony(mode="gso", n_iterations=120, random_state=42, n_worms=20)
    optimizer.fit(sphere, bounds=[(-5, 5), (-5, 5), (-5, 5)])
    assert optimizer.best_score_ is not None
    assert optimizer.best_score_ < 10.0
    assert len(optimizer.best_solution_) == 3
