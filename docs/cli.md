---
title: colonyx CLI Reference
description: Reference and walkthrough for the colonyx command-line tool — run one optimizer, compare all of them on a built-in benchmark, or generate a JSON/CSV comparison report, without writing a script.
---

# CLI

`colonyx` installs a small command-line tool (`colonyx`, via `colonyx.cli:main`) for running an optimizer, comparing several, or generating a report against a built-in benchmark function — no Python script required. It's the fastest way to sanity-check that the package works after installation, to demo an algorithm, or to get a quick read on which mode fits a benchmark best before writing any application code around it.

## `colonyx optimize`

Runs a single optimizer against a built-in benchmark objective and prints the best solution as JSON.

```bash
colonyx optimize --mode pso --objective sphere --dimensions 3 --iterations 100
```

```json
{
  "mode": "pso",
  "objective": "sphere",
  "dimensions": 3,
  "best_solution": [0.0021, -0.0007, 0.0013],
  "best_score": 6.1e-06
}
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--mode` | `pso` | Algorithm to run — any of `pso`, `abc`, `gwo`, `fa`, `sa`, `cs`, `ba`, `gso`, `bfo`, `de`, `cmaes`, `aco`, `auto`. |
| `--objective` | `sphere` | One of the built-in [benchmark functions](benchmarking.md#benchmark-functions): `sphere`, `rosenbrock`, `rastrigin`, `ackley`, `griewank`, `schwefel`, `levy`, `zakharov`, `michalewicz`. |
| `--dimensions` | `3` | Search-space dimensionality; the objective's default bounds are repeated/truncated to match. |
| `--iterations` | `100` | Iteration budget passed through as `n_iterations`. |
| `--random-state` | `None` | Seed for reproducible runs. |

Note that `--repeats` is accepted by `optimize`'s argument parser but the command itself only ever runs once and ignores it — use `colonyx benchmark` or `colonyx report` (below) for repeated runs.

## `colonyx benchmark`

Runs *every* continuous mode (`pso`, `abc`, `gwo`, `fa`, `sa`, `cs`, `ba`, `gso`, `bfo`, `de`, `cmaes` — not `aco`, since it needs a distance matrix rather than a continuous objective) on the same benchmark objective, repeated `--repeats` times each, and prints the aggregated per-mode results as JSON.

```bash
colonyx benchmark --objective sphere --dimensions 3 --iterations 100 --repeats 3
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--objective` | `sphere` | Built-in benchmark function to run every mode against. |
| `--dimensions` | `3` | Search-space dimensionality. |
| `--iterations` | `100` | Iteration budget per run. |
| `--repeats` | `3` | Independent runs per mode, aggregated into mean/std/best. |
| `--random-state` | `None` | Seed, reused across modes for a fairer comparison. |
| `--plot` | off | Print each run's result as it finishes, plus a text bar chart at the end (see [Benchmarking & Metrics](benchmarking.md#comparing-several-optimizers)). |
| `--early-stopping-rounds` | `None` | Stop a mode's repeats early after this many consecutive non-improving runs. |

A mode that raises an exception during its runs is skipped rather than aborting the whole comparison — you'll see `[mode] failed and was skipped: <error>` on stderr, and that mode is simply absent from the JSON output, so one broken configuration never blocks results for the other eleven.

## `colonyx report`

Same underlying comparison as `benchmark`, reshaped into a compact report and written as JSON or CSV.

```bash
colonyx report --objective rastrigin --dimensions 10 --iterations 200 --repeats 5 --format csv
```

```csv
name,best_score,mean_score,std_score,mean_elapsed_seconds,mean_peak_memory_kib,convergence_rate,efficiency
abc,0.994959,3.482111,...
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--objective` | `sphere` | Built-in benchmark function. |
| `--dimensions` | `3` | Search-space dimensionality. |
| `--iterations` | `100` | Iteration budget per run. |
| `--repeats` | `3` | Independent runs per mode. |
| `--random-state` | `None` | Seed, reused across modes. |
| `--format` | `json` | `json` or `csv`. |
| `--output` | `None` | Write to this path instead of stdout; the file gets exactly one trailing newline. |
| `--plot` | off | Also print the text bar chart described above. |
| `--early-stopping-rounds` | `None` | Same early-stopping behavior as `benchmark`. |

The CSV columns are fixed: `name, best_score, mean_score, std_score, mean_elapsed_seconds, mean_peak_memory_kib, convergence_rate, efficiency` — one row per mode, sorted by name. These are exactly the fields `colonyx.metrics.benchmark_report()` returns per optimizer; see [Benchmarking & Metrics](benchmarking.md#other-metrics) for what each one measures.

## When to reach for the CLI vs. the Python API

The CLI only exercises the built-in benchmark functions against the continuous modes — it has no way to point at your own objective function or your own TSP distance matrix from the shell, and it can't touch the multi-objective (`Nsga2Optimizer`/`MopsoOptimizer`) or combinatorial (`PermutationGeneticOptimizer`) algorithms at all, since those need a Python objective or a permutation-shaped problem, not a scalar bounds tuple. Use it for smoke-testing an installation, demoing an algorithm, or getting a first-pass read on which mode looks promising on a *standard* function; once you're optimizing your own objective, move to the [AutoColony API](autocolony-api.md) and the patterns in [Benchmarking & Metrics](benchmarking.md).

## Exit codes

Every subcommand returns `0` on success. Argument errors (an unknown `--mode`, a missing required value) are handled by `argparse` and exit `2` with a usage message, the standard argparse convention.
