<p align="center">
  <img src="colonyx-logo.svg" alt="Colonyx Logo" width="160" height="160">
</p>

<h1 align="center">colonyx</h1>

<p align="center">
  A Pythonic toolkit for swarm intelligence optimization — Rust-fast, scikit-learn compatible.
</p>

<p align="center">
  <a href="https://pypi.org/project/colonyx/"><img src="https://img.shields.io/pypi/v/colonyx" alt="PyPI version"></a>
  <a href="https://github.com/LakoreAI/colonyx/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT license"></a>
  <a href="https://github.com/LakoreAI/colonyx"><img src="https://img.shields.io/badge/repo-GitHub-181717?logo=github" alt="GitHub repo"></a>
</p>

`colonyx` gives you one unified interface — `AutoColony` — for 16 swarm
intelligence and metaheuristic optimizers, backed by a Rust core for speed
and a scikit-learn-compatible Python surface for everything else
(`fit`/`predict`/`score`, `Pipeline`, `GridSearchCV`, `RandomizedSearchCV`).

## Highlights

| | |
| --- | --- |
| **16 algorithms, one interface** | ACO, PSO, ABC, GWO, FA, SA, CS, BA, GSO, BFO, DE, CMA-ES, Binary PSO, permutation GA, NSGA-II, MOPSO — selected via `AutoColony(mode=...)` or `mode="auto"` |
| **Rust core** | Every optimization loop runs in Rust (PyO3); population-based algorithms parallelize independent fitness evaluations with [Rayon](benchmarking.md) where it's safe to |
| **scikit-learn native** | `AutoColony` is a real `BaseEstimator` — drop it into a `Pipeline`, tune it with `GridSearchCV`, cross-validate it |
| **Discrete & continuous** | Square distance matrices for TSP-style problems, or any callable objective over box bounds |
| **Multi-objective** | NSGA-II and MOPSO return a Pareto front/archive, both usable through a common `MultiObjectiveOptimizer` trait on the Rust side |
| **Built for comparison** | A standard benchmark suite (`sphere` … `michalewicz`), a TSPLIB loader, and paired/Wilcoxon significance tests for comparing runs |
| **CLI included** | `colonyx optimize` / `benchmark` / `report` for quick checks without writing a script |

## Install

```bash
pip install colonyx
```

## Quick start

=== "Continuous"

    ```python
    from colonyx import AutoColony

    def sphere(x):
        return sum(xi * xi for xi in x)  # minimum 0 at the origin

    optimizer = AutoColony(mode="pso", n_iterations=150, random_state=42)
    optimizer.fit(sphere, bounds=[(-5, 5), (-5, 5), (-5, 5)])

    optimizer.predict()  # best position, ~ [0, 0, 0]
    optimizer.score()    # objective value at that position, ~ 0
    ```

=== "Discrete (TSP)"

    ```python
    import numpy as np
    from colonyx import AutoColony

    distance_matrix = np.array([
        [0, 1, 9, 9, 1],
        [1, 0, 1, 9, 9],
        [9, 1, 0, 1, 9],
        [9, 9, 1, 0, 1],
        [1, 9, 9, 1, 0],
    ], dtype=float)

    optimizer = AutoColony(mode="aco", n_iterations=100, random_state=42)
    optimizer.fit(distance_matrix)

    optimizer.predict()  # best tour, e.g. [0, 1, 2, 3, 4]
    optimizer.score()    # tour length (lower is better)
    ```

=== "Auto"

    ```python
    from colonyx import AutoColony

    # Picks ACO for a square matrix, PSO/ABC for a continuous objective,
    # based on problem shape and dimensionality.
    optimizer = AutoColony(mode="auto", n_iterations=100, random_state=42)
    optimizer.fit(objective_or_distance_matrix, bounds=bounds_or_none)
    ```

!!! tip "sklearn compatibility"
    `AutoColony` also accepts tabular `X, y` data and behaves like a
    conventional estimator (`Pipeline`, `GridSearchCV`, `cross_val_score`) —
    see [AutoColony API](autocolony-api.md).

## Where to go next

<div class="grid cards" markdown>

- **[Getting Started](getting-started.md)**
  Install from source, build the Rust extension, run your first fit.

- **[Algorithms](algorithms.md)**
  Every optimizer, its parameters, and when to reach for it.

- **[AutoColony API](autocolony-api.md)**
  The unified interface: modes, parameters, `fit`/`predict`/`score`.

- **[Benchmarking & Metrics](benchmarking.md)**
  Standard test functions, TSPLIB, significance tests, run comparison.

- **[CLI](cli.md)**
  `colonyx optimize` / `benchmark` / `report` from the terminal.

- **[Rust Usage](rust.md)**
  Depend on the `colonyx` crate directly from Rust code.

</div>

## By problem type

**Discrete** — `ACO` (with `basic`/`acs`/`elitist`/`mmas` variants) and
`two_opt` for local tour refinement; `PermutationGeneticOptimizer` for
order-crossover-based permutation search.

**Continuous** — `PSO`, `ABC`, `GWO`, `FA`, `SA`, `CS`, `BA`, `GSO`, `BFO`,
`DE`, and `CMA-ES`, each accessed via `AutoColony(mode=...)` or directly as a
Rust-backed class from `colonyx._colonyx`.

**Multi-objective** — `Nsga2Optimizer` and `MopsoOptimizer` return a Pareto
front/archive rather than a single best point; `BinaryParticleSwarm` handles
bit-vector search spaces.

## License

MIT
