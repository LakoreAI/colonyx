"""Command-line interface for colonyx."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from io import StringIO
from pathlib import Path
from typing import Callable

from . import AutoColony
from .benchmarks import benchmark_suite
from .metrics import BenchmarkResult, benchmark_optimizers, benchmark_report

# Modes usable with a continuous objective + bounds, as run by `benchmark`/`report`.
_CONTINUOUS_MODES = ("pso", "abc", "gwo", "fa", "sa", "cs", "ba", "gso", "bfo", "de", "cmaes")
# Every mode `optimize` accepts: the continuous ones above, plus discrete/auto.
_ALL_MODES = (*_CONTINUOUS_MODES, "aco", "auto")


def _continuous_objective(name: str) -> tuple[Callable[[list[float]], float], tuple[tuple[float, float], ...]]:
    suite = benchmark_suite()
    if name not in suite:
        raise ValueError(f"Unknown benchmark objective: {name}")
    problem = suite[name]
    return problem.objective, problem.bounds


def _expand_bounds(bounds: tuple[tuple[float, float], ...], dimensions: int) -> list[tuple[float, float]]:
    if dimensions < 1:
        raise ValueError("dimensions must be at least 1")
    if len(bounds) == dimensions:
        return list(bounds)
    if len(bounds) == 1:
        return [bounds[0]] * dimensions
    if len(bounds) > dimensions:
        return list(bounds[:dimensions])
    return list(bounds) + [bounds[-1]] * (dimensions - len(bounds))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="colonyx", description="Rust-backed swarm intelligence optimizer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    optimize = subparsers.add_parser("optimize", help="Run a single optimization")
    optimize.add_argument("--mode", default="pso", choices=list(_ALL_MODES))
    optimize.add_argument("--objective", default="sphere", choices=sorted(benchmark_suite().keys()))
    optimize.add_argument("--dimensions", type=int, default=3)
    optimize.add_argument("--iterations", type=int, default=100)
    optimize.add_argument("--repeats", type=int, default=1)
    optimize.add_argument("--random-state", type=int, default=None)

    benchmark = subparsers.add_parser("benchmark", help="Benchmark multiple optimizers on a built-in objective")
    benchmark.add_argument("--objective", default="sphere", choices=sorted(benchmark_suite().keys()))
    benchmark.add_argument("--dimensions", type=int, default=3)
    benchmark.add_argument("--iterations", type=int, default=100)
    benchmark.add_argument("--repeats", type=int, default=3)
    benchmark.add_argument("--random-state", type=int, default=None)
    benchmark.add_argument("--plot", action="store_true")
    benchmark.add_argument("--early-stopping-rounds", type=int, default=None)

    report = subparsers.add_parser("report", help="Generate a compact benchmark report")
    report.add_argument("--objective", default="sphere", choices=sorted(benchmark_suite().keys()))
    report.add_argument("--dimensions", type=int, default=3)
    report.add_argument("--iterations", type=int, default=100)
    report.add_argument("--repeats", type=int, default=3)
    report.add_argument("--random-state", type=int, default=None)
    report.add_argument("--format", choices=["json", "csv"], default="json")
    report.add_argument("--output", type=Path, default=None)
    report.add_argument("--plot", action="store_true")
    report.add_argument("--early-stopping-rounds", type=int, default=None)

    return parser


def _make_optimizer(mode: str, iterations: int, random_state: int | None) -> AutoColony:
    return AutoColony(mode=mode, n_iterations=iterations, random_state=random_state)


def _run_optimize(args: argparse.Namespace) -> int:
    objective, bounds = _continuous_objective(args.objective)
    search_bounds = _expand_bounds(bounds, args.dimensions)
    optimizer = _make_optimizer(args.mode, args.iterations, args.random_state)
    optimizer.fit(objective, bounds=search_bounds)
    result = {
        "mode": args.mode,
        "objective": args.objective,
        "dimensions": args.dimensions,
        "best_solution": optimizer.predict(),
        "best_score": optimizer.score(),
    }
    print(json.dumps(result, indent=2))
    return 0


def _run_benchmark_suite(args: argparse.Namespace) -> dict[str, BenchmarkResult]:
    """Shared setup for ``benchmark``/``report``: run every continuous mode
    on the chosen objective. A mode that raises is reported to stderr and
    excluded from the result instead of aborting the whole batch.
    """
    objective, bounds = _continuous_objective(args.objective)
    search_bounds = _expand_bounds(bounds, args.dimensions)
    progress = (
        (lambda name, index, result: print(f"[{name}] run {index + 1}: best={result.best_score:.6g}"))
        if args.plot
        else None
    )
    return benchmark_optimizers(
        {
            mode: (lambda mode=mode: _make_optimizer(mode, args.iterations, args.random_state))
            for mode in _CONTINUOUS_MODES
        },
        objective,
        bounds=search_bounds,
        repeats=args.repeats,
        callback=progress,
        early_stopping_rounds=args.early_stopping_rounds,
        on_error=lambda name, exc: print(f"[{name}] failed and was skipped: {exc}", file=sys.stderr),
    )


def _print_visualization(results: dict[str, BenchmarkResult]) -> None:
    from .metrics import benchmark_visualization

    print()
    print(benchmark_visualization(results))


def _run_benchmark(args: argparse.Namespace) -> int:
    results = _run_benchmark_suite(args)
    print(json.dumps({name: value.__dict__ for name, value in results.items()}, indent=2, default=list))
    if args.plot:
        _print_visualization(results)
    return 0


def _run_report(args: argparse.Namespace) -> int:
    results = _run_benchmark_suite(args)
    report = benchmark_report(results)

    if args.format == "json":
        payload = json.dumps(report, indent=2, sort_keys=True)
    else:
        buffer = StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(
            [
                "name",
                "best_score",
                "mean_score",
                "std_score",
                "mean_elapsed_seconds",
                "mean_peak_memory_kib",
                "convergence_rate",
                "efficiency",
            ]
        )
        for name, metrics in sorted(report.items()):
            writer.writerow(
                [
                    name,
                    metrics["best_score"],
                    metrics["mean_score"],
                    metrics["std_score"],
                    metrics["mean_elapsed_seconds"],
                    metrics["mean_peak_memory_kib"],
                    metrics["convergence_rate"],
                    metrics["efficiency"],
                ]
            )
        payload = buffer.getvalue().rstrip("\n")

    if args.output is not None:
        # Text files should end with exactly one trailing newline.
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    if args.plot:
        _print_visualization(results)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "optimize":
        return _run_optimize(args)
    if args.command == "benchmark":
        return _run_benchmark(args)
    if args.command == "report":
        return _run_report(args)
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
