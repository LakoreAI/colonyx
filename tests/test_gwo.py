from __future__ import annotations

from colonyx import AutoColony


def sphere(x):
    return sum(value * value for value in x)


def test_autocolony_gwo_minimizes_sphere():
    optimizer = AutoColony(mode="gwo", n_iterations=120, random_state=42, n_wolves=20)

    optimizer.fit(sphere, bounds=[(-5, 5), (-5, 5), (-5, 5)])

    assert optimizer.best_score_ is not None
    assert optimizer.best_score_ < 1e-3
    assert len(optimizer.best_solution_) == 3
    assert optimizer.predict() == optimizer.best_solution_

