"""Convenience accessors for built-in benchmark datasets."""

from __future__ import annotations

from .benchmarks import BenchmarkProblem, ackley, benchmark_suite, griewank, rastrigin, rosenbrock, schwefel, sphere


def list_benchmark_problems() -> list[str]:
    """Return the available built-in benchmark problem names."""
    return sorted(benchmark_suite().keys())


def load_benchmark_problem(name: str) -> BenchmarkProblem:
    """Return a built-in benchmark descriptor by name."""
    suite = benchmark_suite()
    if name not in suite:
        raise KeyError(f"Unknown benchmark problem: {name}")
    return suite[name]

