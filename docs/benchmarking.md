# Benchmarking & Metrics

`colonyx.benchmarks` ships a standard suite of continuous test functions and
a minimal TSPLIB loader; `colonyx.metrics` turns repeated optimizer runs into
comparable, statistically grounded results.

## Benchmark functions

`colonyx.benchmarks` exposes each function directly (`sphere`, `rosenbrock`, ...)
and as a `BenchmarkProblem` descriptor via `benchmark_suite()`, which bundles
the objective with literature-standard bounds and the known optimum:

```python
from colonyx import AutoColony
from colonyx.benchmarks import benchmark_suite

suite = benchmark_suite()
problem = suite["rastrigin"]

optimizer = AutoColony(mode="de", n_iterations=200, random_state=7)
optimizer.fit(problem.objective, bounds=[problem.bounds[0]] * 5)
print(optimizer.score(), "vs known optimum", problem.minimum)
```

| Function | Minimum | Notes |
| --- | --- | --- |
| `sphere` | 0 at the origin | Smooth, unimodal — a sanity baseline |
| `rosenbrock` | 0 at `[1, ..., 1]` | Narrow curved valley; needs ≥2 dimensions |
| `rastrigin` | 0 at the origin | Highly multimodal |
| `ackley` | 0 at the origin | Flat outer region, sharp central well |
| `griewank` | 0 at the origin | Many regularly-spaced local minima |
| `schwefel` | 0 near `418.9829` per dimension | Deceptive — the global optimum is far from the origin |
| `levy` | 0 at `[1, ..., 1]` | Levy function N.13 |
| `zakharov` | 0 at the origin | No local minima other than the global one |
| `michalewicz` | ≈ -1.8013 for 2D | Steep multimodal "valleys"; `optimum`/`minimum` in `benchmark_suite()` are for the 2D case only, since higher dimensions have no simple closed form |

!!! note "Michalewicz steepness"
    `michalewicz(x, m=10)` takes a steepness parameter `m`, defaulting to the
    commonly used value of 10. Calling it through `benchmark_suite()` always
    uses that default.

## TSPLIB loader

`load_tsplib(path_or_lines)` parses a `TYPE: TSP`, `EDGE_WEIGHT_TYPE: EUC_2D`
TSPLIB instance (the common case — plain 2D coordinates with Euclidean edge
weights) into a square distance matrix ready for ACO:

```python
from colonyx import AutoColony
from colonyx.benchmarks import load_tsplib

distance_matrix = load_tsplib("berlin52.tsp")
optimizer = AutoColony(mode="aco", n_iterations=200, random_state=7)
optimizer.fit(distance_matrix)
```

`path_or_lines` also accepts an iterable of lines (e.g. a string's
`.splitlines()`) for in-memory instances. Other `EDGE_WEIGHT_TYPE` values
(`GEO`, `ATT`, explicit weight matrices, ...) raise a clear `ValueError`
rather than silently producing a wrong distance matrix.

## Profiling a single run

`profile_optimization_run(optimizer, *fit_args, **fit_kwargs)` fits an
optimizer once and returns a `ProfilingResult` with wall-clock time, peak
memory, and convergence-derived fields read from the optimizer's own
`score()`/`score_history_`:

```python
from colonyx import AutoColony
from colonyx.metrics import profile_optimization_run

optimizer = AutoColony(mode="pso", n_iterations=100, random_state=1)
result = profile_optimization_run(optimizer, lambda x: sum(v * v for v in x), bounds=[(-5, 5)] * 3)
print(result.elapsed_seconds, result.best_score, result.improvement_rate)
```

`profile_callable(func, *args, **kwargs)` is the generic counterpart for a
plain callable that isn't an optimizer — it only measures elapsed time and
peak memory; its optimization-specific fields (`best_score`,
`improvement_rate`, ...) are always `nan`/`0` since a bare callable has no
`score()`/`score_history_` to read.

## Comparing several optimizers

`benchmark_optimizer` repeats one optimizer factory and aggregates the runs;
`benchmark_optimizers` runs several factories and skips (rather than aborts
on) any that raise:

```python
from colonyx import AutoColony
from colonyx.metrics import benchmark_optimizers, benchmark_report, benchmark_visualization

factories = {
    "pso": lambda: AutoColony(mode="pso", n_iterations=100, random_state=1),
    "de": lambda: AutoColony(mode="de", n_iterations=100, random_state=1),
    "cmaes": lambda: AutoColony(mode="cmaes", n_iterations=100, random_state=1),
}

results = benchmark_optimizers(
    factories,
    lambda x: sum(v * v for v in x),
    bounds=[(-5, 5)] * 3,
    repeats=5,
)

print(benchmark_report(results))
print(benchmark_visualization(results, metric="mean_score"))
```

A factory/run that raises is excluded from `results`, reported via a
`RuntimeWarning`, and (if given) an `on_error(name, exc)` callback — a single
broken mode never aborts the whole comparison. The `colonyx benchmark`/
`colonyx report` CLI commands (see [CLI](cli.md)) wrap this same machinery.

## Statistical significance

Two paired-sample tests compare two sets of run scores from the *same*
problem (e.g. algorithm A vs. algorithm B, one score per matched run):

```python
from colonyx.metrics import paired_significance_test, wilcoxon_signed_rank_test

paired_significance_test(scores_a, scores_b)     # paired t-test
wilcoxon_signed_rank_test(scores_a, scores_b)     # Wilcoxon signed-rank test
```

- `paired_significance_test` assumes the score differences are roughly
  normally distributed.
- `wilcoxon_signed_rank_test` is the non-parametric alternative — use it when
  that assumption is shaky, which is common for optimizer benchmark scores.

Both require `scipy` and raise `ImportError` if it's missing, rather than
returning a fabricated p-value.

## Other metrics

- `convergence_rate(history)` — relative improvement from the first to the
  last recorded score.
- `optimization_gap(best_score, optimum=0.0)` — absolute distance to a known
  optimum.
- `success_rate(scores, threshold=0.0, optimum=0.0)` — fraction of runs that
  reached within `threshold` of `optimum`.
- `computational_efficiency(history, elapsed_seconds=None)` — improvement
  per unit of cost (wall-clock time, or iteration count if time isn't given).
- `distribution_analysis(scores)` / `robustness_analysis(scores)` — mean,
  std, quantiles, coefficient of variation, and a robustness score across
  repeated runs.
- `aggregate_runs(scores, optimum=0.0, success_threshold=0.0)` — convenience
  wrapper combining `distribution_analysis` with success rate and mean gap.

`AutoColony` exposes the run-local subset of these directly:
`optimizer.optimization_metrics()`, `optimizer.performance_metrics()`, and
the `AutoColony.summarize_runs()` / `AutoColony.compare_runs()` /
`AutoColony.describe_run_distribution()` / `AutoColony.robustness_report()`
static helpers — see [AutoColony API](autocolony-api.md).
