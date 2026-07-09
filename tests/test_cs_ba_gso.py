from __future__ import annotations

from colonyx import AutoColony


def sphere(x):
    return sum(value * value for value in x)


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
