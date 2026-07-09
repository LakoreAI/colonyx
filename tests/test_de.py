from __future__ import annotations

from colonyx import AutoColony


def sphere(x):
    return sum(value * value for value in x)


def test_autocolony_de_minimizes_sphere():
    optimizer = AutoColony(
        mode="de",
        n_iterations=120,
        random_state=42,
        n_individuals=30,
        f=0.8,
        cr=0.9,
    )

    optimizer.fit(sphere, bounds=[(-5, 5), (-5, 5), (-5, 5)])

    assert optimizer.best_score_ is not None
    assert optimizer.best_score_ < 1e-4
    assert len(optimizer.best_solution_) == 3
    assert optimizer.predict() == optimizer.best_solution_

