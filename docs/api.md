---
title: API Reference
description: Module-level API reference for colonyx — every Python export, Rust-backed optimizer class, and Rust trait, plus how return values differ across single-, multi-, and discrete-objective algorithms.
---

# API reference

This page is the module-level map of everything `colonyx` exports; for the day-to-day unified interface (the one most users actually call), see [AutoColony API](autocolony-api.md), and for algorithm-specific tuning guidance see the [Algorithms](algorithms.md) section.

## `colonyx.AutoColony`

The scikit-learn-compatible entry point for optimization, and the recommended way to use colonyx unless you have a specific reason to reach for a backend class directly.

Key arguments:

- `mode` — algorithm selector (`"auto"`, `"aco"`, `"pso"`, `"abc"`, `"gwo"`, `"fa"`, `"sa"`, `"cs"`, `"ba"`, `"gso"`, `"bfo"`, `"de"`, `"cmaes"`)
- `n_iterations`
- `random_state`
- algorithm-specific parameters such as `n_particles`, `n_bees`, or `n_individuals` (full table in [AutoColony API](autocolony-api.md#algorithm-specific-parameters))

Methods:

- `fit(X, y=None, bounds=None)`
- `predict()`
- `score()`
- `get_params()` / `set_params()`
- `recommend_algorithm()`, `suggest_parameters()`, `optimization_metrics()`, `performance_metrics()` — see [AutoColony API](autocolony-api.md#introspection-metrics)

## Rust-backed exports

These are compiled Rust structs, exposed to Python via PyO3 and re-exported from the top-level `colonyx` package (they live in the compiled extension module `colonyx._colonyx`, but you import them the short way: `from colonyx import ParticleSwarm`). `AutoColony` builds and drives one of these internally for every `fit()` call; you only need to reach for them directly if you want a constructor signature `AutoColony` doesn't expose (for example, ACO's `variant` argument), or if you're calling from a context where sklearn compatibility doesn't matter.

| Class | Algorithm | Docs |
| --- | --- | --- |
| `AntColony` | Ant Colony Optimization | [ACO](algorithms/aco.md), [variants](algorithms/aco-variants.md) |
| `ParticleSwarm` | Particle Swarm Optimization | [PSO](algorithms/pso.md) |
| `BeeColony` | Artificial Bee Colony | [ABC](algorithms/abc.md) |
| `GreyWolfOptimizer` | Grey Wolf Optimizer | [GWO](algorithms/gwo.md) |
| `FireflyOptimizer` | Firefly Algorithm | [FA](algorithms/fa.md) |
| `SimulatedAnnealing` | Simulated Annealing | [SA](algorithms/sa.md) |
| `CuckooSearch` | Cuckoo Search | [CS](algorithms/cs.md) |
| `BatAlgorithm` | Bat Algorithm | [BA](algorithms/ba.md) |
| `GlowwormOptimizer` | Glowworm Swarm Optimization | [GSO](algorithms/gso.md) |
| `BacterialForagingOptimizer` | Bacterial Foraging Optimization | [BFO](algorithms/bfo.md) |
| `DifferentialEvolution` | Differential Evolution | [DE](algorithms/de.md) |
| `CmaEsOptimizer` | Covariance Matrix Adaptation Evolution Strategy | [CMA-ES](algorithms/cmaes.md) |
| `PermutationGeneticOptimizer` | Genetic Algorithm over permutations | [Permutation GA](algorithms/permutation-ga.md) |
| `BinaryParticleSwarm` | Binary/discrete PSO | [Binary PSO](algorithms/binary-pso.md) |
| `Nsga2Optimizer` | NSGA-II multi-objective GA | [NSGA-II](algorithms/nsga2.md) |
| `MopsoOptimizer` | Multi-Objective PSO | [MOPSO](algorithms/mopso.md) |
| `two_opt` | Standalone 2-opt tour local search | Used internally by ACO's `use_two_opt` |

## Return values

Because colonyx spans single-objective continuous search, discrete/combinatorial search, and multi-objective search, what `predict()`/`score()` hand back depends on which family the algorithm belongs to:

- **Continuous optimizers** (PSO, ABC, GWO, FA, SA, CS, BA, GSO, BFO, DE, CMA-ES) return the best position found — a list of floats matching your `bounds` — from `predict()`, and the objective value at that position from `score()`. Lower is better; colonyx always minimizes, so negate your objective first if you're maximizing.
- **ACO** returns the best tour found (a permutation of node indices) from `predict()`, and that tour's total length from `score()`.
- **Multi-objective optimizers** (`Nsga2Optimizer`, `MopsoOptimizer`) return a Pareto archive — the set of non-dominated solutions found — from `predict()`, rather than a single best point, since by definition a multi-objective problem doesn't have one scalar "best." Calling `score()` on one of these requires the objective to return at least two values (it computes a 2-objective hypervolume indicator over the archive) and raises `ValueError` if the objective is scalar-valued — hypervolume isn't defined for a single objective. See [NSGA-II](algorithms/nsga2.md) and [MOPSO](algorithms/mopso.md) for how to read a Pareto front once you have one.

## `colonyx.benchmarks`

Standard continuous test functions and a TSPLIB loader for exercising and comparing optimizers — full write-up, including what each function actually stresses, in [Benchmarking & Metrics](benchmarking.md#benchmark-functions).

- `sphere`, `rosenbrock`, `rastrigin`, `ackley`, `griewank`, `schwefel`, `levy`, `zakharov`, `michalewicz` — standard continuous test functions, each `f(x: array-like) -> float`
- `BenchmarkProblem` — a frozen dataclass bundling `name`, `objective`, `bounds`, `minimum`, and `optimum`
- `benchmark_suite()` — all nine functions above as `BenchmarkProblem` descriptors, keyed by name
- `load_tsplib(path_or_lines)` — parses a TSPLIB (`EUC_2D`) instance into a distance matrix ready for `AutoColony(mode="aco")`

## `colonyx.metrics`

Turns one or many optimization runs into comparable, statistically grounded numbers — see [Benchmarking & Metrics](benchmarking.md) for the full narrative reference with worked examples.

- `benchmark_optimizer(name, factory, *fit_args, repeats=3, ...)`, `benchmark_optimizers(factories, *fit_args, ...)` — run one or several optimizer factories repeatedly and aggregate
- `benchmark_report(results)`, `benchmark_visualization(results, metric="mean_score")`, `compare_benchmark_results(baseline, challenger)` — turn aggregated results into a table, a text bar chart, or a before/after delta
- `profile_optimization_run(optimizer, *fit_args, **fit_kwargs)`, `profile_callable(func, *args, **kwargs)` — wall-clock time and peak memory for one run
- `convergence_rate(history)`, `optimization_gap(best_score, optimum=0.0)`, `success_rate(scores, threshold=0.0, optimum=0.0)`, `computational_efficiency(history, elapsed_seconds=None)` — single-number diagnostics
- `distribution_analysis(scores)`, `robustness_analysis(scores)`, `aggregate_runs(scores, optimum=0.0, success_threshold=0.0)` — cross-run statistics (mean, std, quantiles, coefficient of variation)
- `paired_significance_test(scores_a, scores_b)` (paired t-test), `wilcoxon_signed_rank_test(scores_a, scores_b)` (its non-parametric alternative) — both require `scipy` and raise `ImportError` rather than a fabricated p-value if it's missing

## `colonyx.datasets`

- `list_benchmark_problems()`, `load_benchmark_problem(name)` — thin convenience wrappers over `benchmark_suite()` for discovering and loading a named problem by string, handy for CLI-style or config-driven scripts.

## `colonyx.utils`

- `check_bounds(bounds)`, `check_objective_function(func, probe_point)`, `check_optimization_problem(X, y=None)` — the same input-validation helpers `AutoColony.fit()` uses internally, exposed so you can validate your own inputs early (e.g. in a notebook) before wiring them into an optimizer.

## `colonyx.base`

- `BaseOptimizer` — the abstract sklearn-compatible base class every colonyx estimator (including `AutoColony`) derives from.
- `BaseProblem`, `ContinuousProblem`, `DiscreteProblem` — problem descriptors used internally for input validation; `ContinuousProblem` requires an `objective` callable and `bounds`, `DiscreteProblem` requires a square `distance_matrix`.
- `OptimizerMixin` — small mixin adding `get_optimization_params()` and `is_fitted()`.

## Rust-side traits

For consumers using colonyx as a Rust crate directly (see [Rust Usage](rust.md) for full examples) rather than through the Python bindings:

- `Optimizer` — the common `fit`/`predict`/`score`/`get_params` trait every single-objective algorithm implements. `BinaryParticleSwarm` implements it directly (in addition to its own `fit_with_objective` method).
- `MultiObjectiveOptimizer` — `fit`/`pareto_front` for algorithms that return a Pareto front instead of a single best point. `Nsga2Optimizer` and `MopsoOptimizer` both implement it, so they can be held polymorphically as `Box<dyn MultiObjectiveOptimizer>` — see the [Rust multi-objective example](rust.md#multi-objective-optimization).
- `Problem` / `MultiObjectiveProblem` — the scalar- and vector-valued objective traits that `Optimizer`/`MultiObjectiveOptimizer` are generic over.

## Where to next

- New to colonyx? Start with [Getting Started](getting-started.md).
- Deciding which algorithm to reach for? See the [Algorithms overview](algorithms.md) comparison table.
- Comparing algorithms rigorously, not just eyeballing one run? See [Benchmarking & Metrics](benchmarking.md).
- Calling colonyx from a terminal instead of a script? See [CLI](cli.md).
