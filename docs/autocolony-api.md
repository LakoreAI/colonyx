---
title: AutoColony API Reference
description: Full reference for colonyx's AutoColony estimator — the scikit-learn-compatible interface that swaps between twelve Rust-backed swarm intelligence algorithms via a single mode parameter.
---

# AutoColony API reference

`AutoColony` is the one class you need to run any swarm intelligence algorithm in colonyx: `from colonyx import AutoColony`, pick a `mode` (or let it pick one for you), call `.fit()`, and read `.predict()`/`.score()` — the same three calls whether the algorithm underneath is Particle Swarm Optimization, Differential Evolution, or Ant Colony Optimization.

## Why a unified interface instead of twelve classes?

Every algorithm colonyx ships (`aco`, `pso`, `abc`, `gwo`, `fa`, `sa`, `cs`, `ba`, `gso`, `bfo`, `de`, `cmaes`) is implemented natively as its own Rust struct behind the scenes — see [Related objects](#related-objects) below — but almost nobody wants to memorize twelve different constructors and twelve different `fit()` signatures just to try a different metaheuristic on the same problem. `AutoColony` is the layer that makes swapping algorithms a one-line change: `AutoColony(mode="pso", ...)` becomes `AutoColony(mode="de", ...)` with everything else in your script untouched, in the same spirit as HuggingFace's `AutoModel` picking a concrete architecture behind one loader. Because `AutoColony` also subclasses `sklearn.base.BaseEstimator`/`TransformerMixin`, it participates in `get_params()`/`set_params()`/`clone()`, `sklearn.pipeline.Pipeline`, and `GridSearchCV`/`RandomizedSearchCV` out of the box — you can hyperparameter-search *across* algorithms, not just within one, using `AutoColony.default_param_grids()` (see below).

## A realistic walkthrough

Say you have a black-box objective function and a rough idea of its search space, but you're not sure which algorithm will converge fastest. A typical session looks like this:

```python
from colonyx import AutoColony

def sphere(x):
    return sum(xi * xi for xi in x)

bounds = [(-5, 5)] * 6

# 1. Ask AutoColony what it would pick, and why.
probe = AutoColony()
print(probe.recommend_algorithm(sphere, bounds=bounds))
# {'mode': 'abc', 'reason': 'higher-dimensional continuous objective (6 dims)', ...}

# 2. Fit with mode="auto" (does the same recommendation internally), or pin a mode explicitly.
optimizer = AutoColony(mode="auto", n_iterations=150, random_state=42)
optimizer.fit(sphere, bounds=bounds)

# 3. Read the result.
print(optimizer.predict())   # best position, ~ [0, 0, 0, 0, 0, 0]
print(optimizer.score())     # objective value at that position, ~ 0

# 4. Go beyond the raw score.
print(optimizer.optimization_metrics())
# {'best_score': ..., 'convergence_rate': ..., 'diversity': ..., 'robustness': ...}
```

`recommend_algorithm` and `mode="auto"` share the same heuristic (see [Behavior](#behavior) below), so you can inspect the decision before committing to it, then either accept it or override `mode` explicitly once you know your problem better.

## Constructor

```python
AutoColony(
    mode="auto",
    n_iterations=100,
    random_state=None,
    ...
)
```

### Common arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `mode` | `"auto"` | Backend selector: `"auto"`, `"aco"`, `"pso"`, `"abc"`, `"gwo"`, `"fa"`, `"sa"`, `"cs"`, `"ba"`, `"gso"`, `"bfo"`, `"de"`, or `"cmaes"`. |
| `n_iterations` | `100` | Iteration budget handed to the selected optimizer. |
| `random_state` | `None` | Seed forwarded to the Rust backend's RNG for reproducible runs. |

### Algorithm-specific parameters

Every non-common constructor argument belongs to exactly one algorithm and is silently ignored (and recorded, see [`resolve_parameter_conflicts`](#introspection-metrics) below) if you pass it while a different `mode` is active. The table below is generated from the same registry the code uses internally, so it can't drift out of sync with what `AutoColony` actually accepts — see the [algorithm pages](algorithms.md) for what each parameter does mechanically and how to tune it.

| Parameters | Applies to | Defaults |
| --- | --- | --- |
| `n_ants`, `alpha`, `beta`, `rho`, `q`, `use_two_opt` | [`aco`](algorithms/aco.md) | `50, 1.0, 2.0, 0.5, 1.0, True` |
| `n_particles`, `w`, `c1`, `c2` | [`pso`](algorithms/pso.md) | `30, 0.9, 2.0, 2.0` |
| `n_bees`, `limit` | [`abc`](algorithms/abc.md) | `50, 10` |
| `n_wolves` | [`gwo`](algorithms/gwo.md) | `30` |
| `n_fireflies`, `beta0`, `gamma`, `fa_alpha` | [`fa`](algorithms/fa.md) | `30, 1.0, 1.0, 0.2` |
| `initial_temperature`, `cooling_rate`, `step_scale` | [`sa`](algorithms/sa.md) | `10.0, 0.95, 0.1` |
| `n_nests`, `pa`, `cs_alpha`, `levy_scale` | [`cs`](algorithms/cs.md) | `25, 0.25, 0.01, 1.0` |
| `n_bats`, `fmin`, `fmax`, `bat_alpha`, `bat_gamma`, `loudness`, `pulse_rate` | [`ba`](algorithms/ba.md) | `30, 0.0, 2.0, 0.9, 0.9, 1.0, 0.5` |
| `n_worms`, `luciferin_decay`, `luciferin_enhancement`, `gso_step_size`, `neighborhood_radius` | [`gso`](algorithms/gso.md) | `30, 0.4, 0.6, 0.1, 1.0` |
| `n_bacteria`, `n_chemotactic_steps`, `n_reproduction_steps`, `elimination_probability`, `bfo_step_scale` | [`bfo`](algorithms/bfo.md) | `30, 10, 4, 0.25, 0.1` |
| `n_individuals`, `f`, `cr` | [`de`](algorithms/de.md) | `40, 0.8, 0.9` |
| `n_individuals`, `cmaes_sigma` | [`cmaes`](algorithms/cmaes.md) | `40, 0.5` |

A handful of names (`alpha`, `gamma`, `step_scale`) are reused with algorithm-specific meaning across ACO/FA/CS/BA/BFO — that's why the frontend attribute names above (`fa_alpha`, `cs_alpha`, `bat_alpha`, `bat_gamma`, `bfo_step_scale`, `gso_step_size`, `cmaes_sigma`) are qualified per algorithm even though the underlying Rust constructor keyword is the shorter, unqualified name (`alpha`, `gamma`, `step_scale`, `sigma`) — see [`parameter_mapping`](#introspection-metrics).

!!! note "Source of truth"
    All algorithm-specific parameter names, defaults, and their mapping to
    the underlying Rust constructor keywords live in one place:
    `_ALGORITHM_PARAM_SPECS` in `colonyx/auto.py`. If a parameter isn't
    listed above, check that dict — it's authoritative.

## The `fit()` contract

```python
fit(X, y=None, bounds=None)
```

`fit()` accepts three different shapes of `X`, and dispatches on which one you gave it plus the active `mode`:

- **A callable objective** `f(list[float]) -> float` for every continuous mode (`pso`, `abc`, `gwo`, `fa`, `sa`, `cs`, `ba`, `gso`, `bfo`, `de`, `cmaes`). This requires `bounds=[(low, high), ...]`, one pair per dimension — `AutoColony` has no other way to know the search space, so omitting `bounds` for a continuous mode raises `ValueError`.
- **A square distance matrix** for `mode="aco"` — a 2D array-like where `matrix[i][j]` is the cost of the edge from node `i` to node `j`. ACO ignores `bounds` entirely, since a tour has no per-dimension range.
- **Tabular `(X, y)` data**, array-like rather than callable, for sklearn compatibility. This routes to a deterministic fallback (`_fit_sklearn_compatibility`) that finds the training row with the best `y` value and returns it as a constant prediction — useful for keeping `AutoColony` a drop-in estimator inside pipelines and cross-validation that were built assuming plain tabular fit/predict, not as a real supervised learner. If you're doing real optimization, feed it a callable + bounds or a distance matrix, not `(X, y)`.

## `predict()` and `score()`

- `predict()` returns the best position (continuous modes) or best tour (`aco`) found during `fit()`. Under the tabular-compatibility fallback it instead returns a constant vector equal to the best observed score, sized to match `X`'s row count, so the estimator still behaves shape-correctly inside a pipeline.
- `score()` returns the best objective value (lower is better, since colonyx minimizes) or best tour length. Under the tabular-compatibility fallback it returns *negative* mean squared error instead, so higher-is-better sklearn scoring conventions still hold there.
- Both raise `sklearn.exceptions.NotFittedError` if called before `fit()`.

## Behavior

- `mode="auto"` calls the same heuristic as `recommend_algorithm()`: a square distance-matrix-shaped `X` routes to ACO; a callable continuous objective routes to PSO at four dimensions or fewer, and to ABC above that (empirically, larger candidate populations tend to help ABC's food-source exploration scale better than PSO's velocity dynamics as dimensionality grows); tabular `(X, y)` routes to the sklearn-compatibility fallback.
- The Rust extension (`colonyx._colonyx`, built via PyO3/maturin — see [Rust Usage](rust.md)) performs every actual optimization step; `AutoColony` itself is pure-Python glue that validates inputs, resolves which backend class and keyword arguments to build, and normalizes the result.
- `AntColony` (the ACO backend) additionally exposes a `variant="basic" | "acs" | "elitist" | "mmas"` constructor argument for the classic ACO variants — see [ACO Variants](algorithms/aco-variants.md). `AutoColony`'s `mode="aco"` always uses the basic variant; instantiate `colonyx.AntColony` directly if you need a variant.

## Introspection & metrics

These methods let you see *why* `AutoColony` would make a choice, or squeeze more signal out of a completed run, without having to reimplement any of colonyx's own heuristics:

- `recommend_algorithm(X, y=None, bounds=None)` — the same heuristic `mode="auto"` uses, returned as a dict with `mode`, `reason`, `problem_type`, and `dimension` so you can inspect *why* a backend would be picked before you commit to it.
- `suggest_parameters(X, y=None, bounds=None)` — reasonable starting parameters for the recommended (or explicitly set) mode, sized from the problem's dimensionality — a quick way to skip the "what should `n_particles` even be" question for a first run.
- `parameter_mapping(algorithm_mode=None)` / `parameter_help(algorithm_mode=None)` — the frontend-parameter-to-backend-keyword mapping and a one-line summary for a given mode.
- `resolve_parameter_conflicts(algorithm_mode)` — the active parameters for that mode, and records which of the *other* algorithms' parameters you passed but that don't apply, in `parameter_conflicts_`. Handy for catching a typo'd or leftover parameter from a previous `mode` silently doing nothing.
- After `fit()`: `optimization_metrics()` bundles `score()` with `convergence_rate_score()`, `diversity_score()`, and `robustness_score()`; `performance_metrics(optimum=0.0, success_threshold=0.0)` additionally computes `optimization_gap` and `success_rate` against a known optimum — see [Benchmarking & Metrics](benchmarking.md) for what each of those numbers actually measures and when to trust them.
- `AutoColony.default_param_grids()` / `default_param_distributions()` — ready-made `GridSearchCV`/`RandomizedSearchCV` search spaces, one entry per mode, so you can hyperparameter-search across algorithm families in a single sklearn search rather than hand-rolling twelve separate grids.
- `AutoColony.optimization_cv_strategy(X, y=None, n_splits=5, random_state=42)` — a `StratifiedKFold` or `KFold` splitter chosen automatically based on `y`'s cardinality, for use with the sklearn-compatibility fallback.
- Static run-comparison helpers, useful once you have scores from multiple repeated fits: `AutoColony.summarize_runs(scores, optimum=0.0, success_threshold=0.0)`, `AutoColony.compare_runs(scores_a, scores_b)` (paired significance test), `AutoColony.describe_run_distribution(scores)`, `AutoColony.robustness_report(scores)`, and `AutoColony.profile_run(optimizer, *fit_args, **fit_kwargs)` for timing/memory profiling a single fit.

## Related objects

Every mode above has a same-named Rust class you can import and use directly if you want to skip `AutoColony`'s dispatch layer (for example, to pass `use_two_opt` per-call rather than at construction, or to hold algorithm instances polymorphically in Rust — see [Rust Usage](rust.md)):

- `colonyx.AntColony` (aliased from `colonyx._colonyx.AntColony`)
- `colonyx.ParticleSwarm`
- `colonyx.BeeColony`
- `colonyx.GreyWolfOptimizer`
- `colonyx.FireflyOptimizer`
- `colonyx.SimulatedAnnealing`
- `colonyx.CuckooSearch`
- `colonyx.BatAlgorithm`
- `colonyx.GlowwormOptimizer`
- `colonyx.BacterialForagingOptimizer`
- `colonyx.DifferentialEvolution`
- `colonyx.CmaEsOptimizer`
- `colonyx.BinaryParticleSwarm` (see [Binary PSO](algorithms/binary-pso.md))
- `colonyx.PermutationGeneticOptimizer` (see [Permutation GA](algorithms/permutation-ga.md))
- `colonyx.Nsga2Optimizer` (see [NSGA-II](algorithms/nsga2.md))
- `colonyx.MopsoOptimizer` (see [MOPSO](algorithms/mopso.md))
- `colonyx.two_opt` (standalone 2-opt local search, used internally by ACO when `use_two_opt=True`)

See [API Reference](api.md) for the full module-level listing, or [Getting Started](getting-started.md) for a from-scratch installation and first-run walkthrough.
