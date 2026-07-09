from __future__ import annotations

from colonyx import AutoColony


def sphere(x):
    return sum(value * value for value in x)


def test_autocolony_fa_minimizes_sphere():
    optimizer = AutoColony(mode="fa", n_iterations=120, random_state=42, n_fireflies=20)

    optimizer.fit(sphere, bounds=[(-5, 5), (-5, 5), (-5, 5)])

    assert optimizer.best_score_ is not None
    assert optimizer.best_score_ < 1e-2
    assert len(optimizer.best_solution_) == 3

