# CLI

`colonyx` ships with a small command-line interface for quick checks and benchmarks.

## Commands

### Optimize

```bash
colonyx optimize --mode pso --objective sphere --dimensions 3 --iterations 100
```

Runs one optimizer against a built-in benchmark objective and prints the best solution as JSON.

### Benchmark

```bash
colonyx benchmark --objective sphere --dimensions 3 --iterations 100 --repeats 3
```

Runs the built-in continuous optimizers on the same benchmark objective and prints aggregated results.

### Report

```bash
colonyx report --objective sphere --dimensions 3 --iterations 100 --repeats 3 --format csv
```

Generates a compact comparison report in JSON or CSV.

#### Options

- `--format json|csv`
- `--output path/to/file`
- `--plot` prints a text chart for the benchmark results
- `--early-stopping-rounds N` stops after N non-improving repeats

## Notes

- `optimize` supports `aco`, `pso`, `abc`, `gwo`, `fa`, `sa`, `cs`, `ba`, `gso`, `bfo`, `de`, `cmaes`, and `auto`.
- `benchmark` and `report` can print text plots with `--plot`.
- The CLI is intended for smoke testing, demos, and quick comparisons.
