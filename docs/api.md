# API

## `AutoColony`

Main entry point for optimization.

Key arguments:

- `mode`
- `n_iterations`
- `random_state`
- algorithm-specific parameters such as `n_particles`, `n_bees`, or `n_individuals`

Methods:

- `fit(X, y=None, bounds=None)`
- `predict()`
- `score()`
- `get_params()`
- `set_params()`

## Rust-backed exports

Available from `colonyx._colonyx`:

- `AntColony`
- `ParticleSwarm`
- `BeeColony`
- `GreyWolfOptimizer`
- `FireflyOptimizer`
- `SimulatedAnnealing`
- `CuckooSearch`
- `BatAlgorithm`
- `GlowwormOptimizer`
- `BacterialForagingOptimizer`
- `DifferentialEvolution`
- `CmaEsOptimizer`
- `PermutationGeneticOptimizer`
- `BinaryParticleSwarm`
- `Nsga2Optimizer`
- `MopsoOptimizer`
- `two_opt`

## Return values

- Continuous optimizers return the best position and best score.
- ACO returns the best tour and tour length.
- Multi-objective optimizers return a Pareto archive via `predict()`; calling
  `score()` on one requires the objective to return at least 2 values (it
  computes a 2-objective hypervolume) and raises `ValueError` otherwise.

## `colonyx.benchmarks`

- `sphere`, `rosenbrock`, `rastrigin`, `ackley`, `griewank`, `schwefel`,
  `levy`, `zakharov`, `michalewicz` — standard continuous test functions
- `benchmark_suite()` — all of the above as `BenchmarkProblem` descriptors
  (objective + bounds + known optimum)
- `load_tsplib(path_or_lines)` — TSPLIB (`EUC_2D`) instance to a distance matrix

See [Benchmarking & Metrics](benchmarking.md) for the full reference.

## `colonyx.metrics`

- `benchmark_optimizer`, `benchmark_optimizers`, `benchmark_report`, `benchmark_visualization`, `compare_benchmark_results`
- `profile_optimization_run`, `profile_callable`
- `convergence_rate`, `optimization_gap`, `success_rate`, `computational_efficiency`
- `distribution_analysis`, `robustness_analysis`, `aggregate_runs`
- `paired_significance_test` (paired t-test), `wilcoxon_signed_rank_test` (non-parametric alternative) — both require `scipy`

## Rust-side traits

- `Optimizer` — the common `fit`/`predict`/`score`/`get_params` trait for
  single-objective algorithms. `BinaryParticleSwarm` implements it directly
  (in addition to its own `fit_with_objective`).
- `MultiObjectiveOptimizer` — `fit`/`pareto_front` for algorithms that return
  a Pareto front. `Nsga2Optimizer` and `MopsoOptimizer` both implement it, so
  they can be held polymorphically as `Box<dyn MultiObjectiveOptimizer>`.
- `Problem` / `MultiObjectiveProblem` — the scalar and vector-valued
  objective traits `Optimizer`/`MultiObjectiveOptimizer` are generic over.

See [Rust Usage](rust.md) for examples.
