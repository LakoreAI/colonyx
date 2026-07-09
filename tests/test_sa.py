from __future__ import annotations

from colonyx import AutoColony


def sphere(x):
    return sum(value * value for value in x)


def test_autocolony_sa_minimizes_sphere():
    optimizer = AutoColony(mode="sa", n_iterations=200, random_state=42)

    optimizer.fit(sphere, bounds=[(-5, 5), (-5, 5), (-5, 5)])

    assert optimizer.best_score_ is not None
    assert optimizer.best_score_ < 2.0
    assert len(optimizer.best_solution_) == 3

