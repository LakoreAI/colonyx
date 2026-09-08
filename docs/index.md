---
title: "colonyx — Rust-powered Python swarm intelligence toolkit"
description: "colonyx is a Python library with a Rust core for swarm intelligence and metaheuristic optimization: PSO, ACO, ABC, GWO, DE, CMA-ES and more behind one scikit-learn-compatible interface."
---

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

`colonyx` is a Python library for optimizing functions that you can evaluate but cannot differentiate — it gives you 16 swarm intelligence and metaheuristic algorithms behind a single interface, `AutoColony`, with every optimization loop executed in Rust for speed and a scikit-learn-compatible Python surface (`fit`/`predict`/`score`, `Pipeline`, `GridSearchCV`) for everything else.

## What problem does colonyx solve?

Most optimization tooling you already know — `scipy.optimize.minimize`, gradient descent in a training loop, Newton's method — assumes your objective function is smooth and that you can compute or approximate its gradient. That assumption breaks down constantly in practice: your objective might call out to a simulator, a black-box scoring function, a discrete combinatorial structure like a delivery route, or a noisy, non-convex surface riddled with local minima where gradient-based methods get stuck immediately. Swarm intelligence and evolutionary metaheuristics — Particle Swarm Optimization, Ant Colony Optimization, Differential Evolution, and their relatives — sidestep the differentiability requirement entirely. They work by maintaining a population of candidate solutions (particles, ants, bees, wolves, bats — the "swarm") that explore the search space in parallel, share information about what's working, and iteratively bias the population toward better regions, guided only by the objective's output value, never its derivative. That makes them the right tool whenever your objective is a black box, expensive but not too expensive to call thousands of times, discrete (a tour, a permutation, a bit-vector), multi-modal (many local optima), or you simply don't have — or don't trust — a gradient. `colonyx` exists to make reaching for one of these algorithms as easy as reaching for `scipy.optimize`, without giving up the raw performance you'd otherwise only get by hand-rolling the inner loop in a compiled language.

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

`colonyx` ships as a prebuilt wheel with the Rust extension already compiled, so `pip install colonyx` is all most users need — no Rust toolchain required. See [Getting Started](getting-started.md) if you want to build the extension from source instead.

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

**Discrete** — [ACO](algorithms/aco.md) (with [`basic`/`acs`/`elitist`/`mmas` variants](algorithms/aco-variants.md)) and `two_opt` for local tour refinement; [`PermutationGeneticOptimizer`](algorithms/permutation-ga.md) for order-crossover-based permutation search.

**Continuous** — [PSO](algorithms/pso.md), [ABC](algorithms/abc.md), [GWO](algorithms/gwo.md), [FA](algorithms/fa.md), [SA](algorithms/sa.md), [CS](algorithms/cs.md), [BA](algorithms/ba.md), [GSO](algorithms/gso.md), [BFO](algorithms/bfo.md), [DE](algorithms/de.md), and [CMA-ES](algorithms/cmaes.md), each accessed via `AutoColony(mode=...)` or directly as a Rust-backed class from `colonyx._colonyx`. See the [full comparison table](algorithms.md) for how to pick between them.

**Multi-objective** — [`Nsga2Optimizer`](algorithms/nsga2.md) and [`MopsoOptimizer`](algorithms/mopso.md) return a Pareto front/archive rather than a single best point; [`BinaryParticleSwarm`](algorithms/binary-pso.md) handles bit-vector search spaces.

## Frequently asked questions

### Is colonyx faster than a pure-Python implementation of the same algorithm?

Every optimization loop in `colonyx` — the fitness evaluations, population updates, pheromone/velocity/mutation math — runs in compiled Rust behind a thin PyO3 binding, not in interpreted Python, and population-based algorithms additionally parallelize independent fitness evaluations across CPU cores with Rayon where doing so doesn't change the algorithm's result. The Python side only builds the initial arguments (bounds, distance matrix, callable) and calls into that Rust core once per `fit()`, so the per-iteration cost you'd normally pay for a pure-Python loop over a population is largely eliminated. See [Rust Usage](rust.md) and [Benchmarking & Metrics](benchmarking.md) if you want to measure this on your own objective rather than take that on faith — we deliberately don't publish a single "Nx faster" headline number here because the speedup depends heavily on how expensive your own objective function is relative to the optimizer's bookkeeping.

### Does colonyx work with scikit-learn pipelines?

Yes. `AutoColony` implements the scikit-learn estimator contract (`get_params`, `set_params`, `__sklearn_tags__`, `fit`/`predict`/`score`, `fit_transform`) so it can sit inside a `Pipeline`, be tuned with `GridSearchCV` or `RandomizedSearchCV` using `AutoColony.default_param_grids()` / `default_param_distributions()`, and be cross-validated with a compatible splitter from `AutoColony.optimization_cv_strategy(...)`. See [AutoColony API](autocolony-api.md) for the full contract, including the tabular-data compatibility fallback used when you pass conventional `X, y` instead of a bounds/objective pair.

### Which algorithm should I use for a discrete, combinatorial problem like TSP?

Use `AutoColony(mode="aco")` and pass a square distance matrix — Ant Colony Optimization is purpose-built for exactly this shape of problem, and `AutoColony` will even pick it automatically under `mode="auto"` when it sees a square matrix. For a from-scratch look at how the pheromone-update variants differ, see [ACO Variants](algorithms/aco-variants.md); for a related but distinct permutation search built on genetic operators instead of pheromones, see [Permutation GA](algorithms/permutation-ga.md).

### Which algorithm should I use for a continuous, black-box objective?

If you don't have a strong prior, start with `mode="pso"` for low-dimensional problems or `mode="abc"` once dimensionality climbs — this mirrors what `AutoColony(mode="auto")` itself does via `recommend_algorithm()`. If your surface is highly multi-modal or you want to hedge against getting stuck, [Simulated Annealing](algorithms/sa.md), [Differential Evolution](algorithms/de.md), and [CMA-ES](algorithms/cmaes.md) are strong alternatives; the full [algorithm comparison table](algorithms.md) breaks down when each one tends to win.

### Do I need a multi-objective algorithm, and what does colonyx offer for that?

If you're optimizing two or more competing objectives at once — cost versus latency, accuracy versus model size — a single-objective optimizer can only give you one point on the trade-off curve. [NSGA-II](algorithms/nsga2.md) and [MOPSO](algorithms/mopso.md) instead return a whole Pareto front (or archive) of non-dominated solutions, letting you choose the trade-off after the fact rather than baking a single weighting into your objective function up front.

### Can I use colonyx without installing a Rust toolchain?

Yes — `pip install colonyx` pulls a prebuilt wheel with the compiled `colonyx._colonyx` extension already inside it, so a plain `pip install` is enough for normal use. You only need Rust and [`maturin`](https://www.maturin.rs/) if you're building from a source checkout or contributing to the Rust core itself; see [Getting Started](getting-started.md) for that path.

## License

MIT
