"""Tests for advanced optimizers."""

from __future__ import annotations

import numpy as np
import pytest

from colonyx import MopsoOptimizer, Nsga2Optimizer, PermutationGeneticOptimizer


def test_permutation_genetic_optimizer_finds_valid_tour():
    distance_matrix = np.array(
        [
            [0.0, 1.0, 9.0, 9.0, 1.0],
            [1.0, 0.0, 1.0, 9.0, 9.0],
            [9.0, 1.0, 0.0, 1.0, 9.0],
            [9.0, 9.0, 1.0, 0.0, 1.0],
            [1.0, 9.0, 9.0, 1.0, 0.0],
        ],
        dtype=float,
    )

    optimizer = PermutationGeneticOptimizer(
        n_individuals=20,
        n_iterations=40,
        mutation_rate=0.2,
        use_two_opt=True,
        random_state=3,
    )
    optimizer.fit(distance_matrix)

    tour = optimizer.predict()
    assert sorted(tour) == list(range(5))
    assert optimizer.score() < 20.0


def test_nsga2_optimizer_returns_pareto_front():
    def objectives(x):
        return [sum(value * value for value in x), sum((value - 1.0) ** 2 for value in x)]

    optimizer = Nsga2Optimizer(
        n_individuals=20,
        n_iterations=20,
        crossover_rate=0.9,
        mutation_rate=0.2,
        mutation_scale=0.1,
        archive_size=10,
        random_state=11,
    )
    optimizer.fit(objectives, lower=[0.0, 0.0], upper=[1.0, 1.0])

    front = optimizer.predict()
    assert front
    assert all(len(candidate) == 2 for candidate in front)
    assert np.isfinite(optimizer.score())


def test_mopso_optimizer_returns_pareto_archive():
    def objectives(x):
        return [sum(value * value for value in x), sum((value - 1.0) ** 2 for value in x)]

    optimizer = MopsoOptimizer(
        n_particles=20,
        n_iterations=20,
        w=0.7,
        c1=1.5,
        c2=1.5,
        mutation_scale=0.1,
        archive_size=10,
        random_state=17,
    )
    optimizer.fit(objectives, lower=[0.0, 0.0], upper=[1.0, 1.0])

    archive = optimizer.predict()
    assert archive
    assert all(len(candidate) == 2 for candidate in archive)
    assert np.isfinite(optimizer.score())


def test_nsga2_score_raises_for_a_single_objective_function():
    # Regression test: score() used to silently return 0.0 (a fabricated,
    # meaningless value) instead of erroring when fit() with an objective
    # returning fewer than the 2 values hypervolume needs was misused.
    def single_objective(x):
        return [sum(value * value for value in x)]

    optimizer = Nsga2Optimizer(
        n_individuals=10,
        n_iterations=5,
        crossover_rate=0.9,
        mutation_rate=0.2,
        mutation_scale=0.1,
        archive_size=10,
        random_state=1,
    )
    optimizer.fit(single_objective, lower=[0.0], upper=[1.0])

    with pytest.raises(ValueError, match="hypervolume"):
        optimizer.score()


def test_mopso_score_raises_for_a_single_objective_function():
    def single_objective(x):
        return [sum(value * value for value in x)]

    optimizer = MopsoOptimizer(
        n_particles=10,
        n_iterations=5,
        w=0.7,
        c1=1.5,
        c2=1.5,
        mutation_scale=0.1,
        archive_size=10,
        random_state=1,
    )
    optimizer.fit(single_objective, lower=[0.0], upper=[1.0])

    with pytest.raises(ValueError, match="hypervolume"):
        optimizer.score()
