"""Performance and statistical helpers for optimization runs."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import tracemalloc
from math import sqrt
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - scipy is present in CI, keep fallback safe.
    scipy_stats = None


@dataclass(frozen=True)
class ProfilingResult:
    """Lightweight profiling summary for a single optimization run."""

    elapsed_seconds: float
    peak_memory_kib: float
    best_score: float
    score_history_length: int
    improvement_rate: float
    efficiency: float


@dataclass(frozen=True)
class BenchmarkResult:
    """Aggregated benchmark results for repeated optimizer runs."""

    name: str
    scores: tuple[float, ...]
    elapsed_seconds: tuple[float, ...]
    peak_memory_kib: tuple[float, ...]
    best_score: float
    mean_score: float
    std_score: float
    mean_elapsed_seconds: float
    mean_peak_memory_kib: float
    convergence_rate: float
    efficiency: float


def profile_callable(func, *args, **kwargs) -> tuple[object, ProfilingResult]:
    """Measure wall-clock and peak-memory cost of a callable."""
    tracemalloc.start()
    start_time = perf_counter()
    try:
        result = func(*args, **kwargs)
    finally:
        elapsed_seconds = perf_counter() - start_time
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    return result, ProfilingResult(
        elapsed_seconds=float(elapsed_seconds),
        peak_memory_kib=float(peak_memory / 1024.0),
        best_score=float(getattr(func, "__profile_best_score__", np.nan)),
        score_history_length=int(getattr(func, "__profile_history_length__", 0)),
        improvement_rate=float(getattr(func, "__profile_improvement_rate__", 0.0)),
        efficiency=float(getattr(func, "__profile_efficiency__", 0.0)),
    )


def profile_optimization_run(optimizer, *fit_args, **fit_kwargs) -> ProfilingResult:
    """Fit an optimizer and return a profiling summary."""
    tracemalloc.start()
    start_time = perf_counter()
    try:
        optimizer.fit(*fit_args, **fit_kwargs)
    finally:
        elapsed_seconds = perf_counter() - start_time
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    history = getattr(optimizer, "score_history_", [])
    best_score = optimizer.score() if hasattr(optimizer, "score") else np.nan
    improvement_rate = convergence_rate(history)
    efficiency = computational_efficiency(history, elapsed_seconds=elapsed_seconds)

    optimizer.__profile_best_score__ = best_score
    optimizer.__profile_history_length__ = len(history)
    optimizer.__profile_improvement_rate__ = improvement_rate
    optimizer.__profile_efficiency__ = efficiency

    return ProfilingResult(
        elapsed_seconds=float(elapsed_seconds),
        peak_memory_kib=float(peak_memory / 1024.0),
        best_score=float(best_score),
        score_history_length=int(len(history)),
        improvement_rate=float(improvement_rate),
        efficiency=float(efficiency),
    )


def benchmark_optimizer(
    name: str,
    optimizer_factory: Callable[[], object],
    *fit_args,
    repeats: int = 3,
    **fit_kwargs,
) -> BenchmarkResult:
    """Run one optimizer repeatedly and aggregate the results."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    scores: list[float] = []
    elapsed_seconds: list[float] = []
    peak_memory_kib: list[float] = []

    for _ in range(repeats):
        optimizer = optimizer_factory()
        profile = profile_optimization_run(optimizer, *fit_args, **fit_kwargs)
        scores.append(float(profile.best_score))
        elapsed_seconds.append(float(profile.elapsed_seconds))
        peak_memory_kib.append(float(profile.peak_memory_kib))

    score_array = np.asarray(scores, dtype=float)
    elapsed_array = np.asarray(elapsed_seconds, dtype=float)
    peak_array = np.asarray(peak_memory_kib, dtype=float)

    return BenchmarkResult(
        name=name,
        scores=tuple(scores),
        elapsed_seconds=tuple(elapsed_seconds),
        peak_memory_kib=tuple(peak_memory_kib),
        best_score=float(np.min(score_array)),
        mean_score=float(np.mean(score_array)),
        std_score=float(np.std(score_array)),
        mean_elapsed_seconds=float(np.mean(elapsed_array)),
        mean_peak_memory_kib=float(np.mean(peak_array)),
        convergence_rate=convergence_rate(scores),
        efficiency=computational_efficiency(scores, elapsed_seconds=float(np.sum(elapsed_array))),
    )


def benchmark_optimizers(
    optimizer_factories: Mapping[str, Callable[[], object]],
    *fit_args,
    repeats: int = 3,
    **fit_kwargs,
) -> dict[str, BenchmarkResult]:
    """Benchmark several optimizers and return per-optimizer summaries."""
    return {
        name: benchmark_optimizer(name, factory, *fit_args, repeats=repeats, **fit_kwargs)
        for name, factory in optimizer_factories.items()
    }


def benchmark_report(results: Mapping[str, BenchmarkResult]) -> dict[str, dict[str, float]]:
    """Convert benchmark results into a compact comparison report."""
    if not results:
        raise ValueError("results must not be empty")

    report: dict[str, dict[str, float]] = {}
    for name, result in results.items():
        report[name] = {
            "best_score": result.best_score,
            "mean_score": result.mean_score,
            "std_score": result.std_score,
            "mean_elapsed_seconds": result.mean_elapsed_seconds,
            "mean_peak_memory_kib": result.mean_peak_memory_kib,
            "convergence_rate": result.convergence_rate,
            "efficiency": result.efficiency,
        }
    return report


def benchmark_visualization(results: Mapping[str, BenchmarkResult], metric: str = "mean_score") -> str:
    """Render a simple text visualization for benchmark comparisons."""
    if not results:
        raise ValueError("results must not be empty")

    metric_values = []
    for name, result in results.items():
        if not hasattr(result, metric):
            raise ValueError(f"Unknown metric: {metric}")
        metric_values.append((name, float(getattr(result, metric))))

    values = [value for _, value in metric_values]
    minimum = min(values)
    maximum = max(values)
    span = max(maximum - minimum, 1e-12)

    lines = [f"Benchmark visualization for {metric}"]
    for name, value in sorted(metric_values, key=lambda item: item[1]):
        normalized = int(round(((value - minimum) / span) * 24))
        bar = "█" * normalized
        lines.append(f"{name:>12} | {bar:<24} {value:.6g}")
    return "\n".join(lines)


def compare_benchmark_results(
    baseline: BenchmarkResult,
    challenger: BenchmarkResult,
) -> dict[str, float]:
    """Compare two benchmark summaries."""
    return {
        "score_delta": float(challenger.mean_score - baseline.mean_score),
        "elapsed_delta": float(challenger.mean_elapsed_seconds - baseline.mean_elapsed_seconds),
        "memory_delta": float(challenger.mean_peak_memory_kib - baseline.mean_peak_memory_kib),
        "efficiency_delta": float(challenger.efficiency - baseline.efficiency),
    }


def convergence_rate(history: Sequence[float]) -> float:
    """Return relative improvement from the first to the last score."""
    values = np.asarray(history, dtype=float)
    if values.size < 2:
        return 0.0
    start_score = values[0]
    end_score = values[-1]
    denominator = max(abs(start_score), 1e-12)
    return float((start_score - end_score) / denominator)


def optimization_gap(best_score: float, optimum: float = 0.0) -> float:
    """Return the absolute gap to a known optimum."""
    return float(abs(best_score - optimum))


def success_rate(scores: Sequence[float], threshold: float = 0.0, optimum: float = 0.0) -> float:
    """Return the fraction of scores that meet the target threshold."""
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.mean(values <= (optimum + threshold)))


def computational_efficiency(history: Sequence[float], elapsed_seconds: float | None = None) -> float:
    """Return a simple improvement-per-cost metric."""
    values = np.asarray(history, dtype=float)
    if values.size < 2:
        return 0.0
    improvement = abs(values[0] - values[-1])
    cost = float(elapsed_seconds) if elapsed_seconds and elapsed_seconds > 0 else float(values.size - 1)
    return float(improvement / cost)


def distribution_analysis(scores: Sequence[float]) -> dict[str, float]:
    """Summarise the distribution of multiple run scores."""
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        raise ValueError("scores must not be empty")

    return {
        "count": float(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "q1": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q3": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
        "cv": float(np.std(values) / max(abs(np.mean(values)), 1e-12)),
    }


def robustness_analysis(scores: Sequence[float]) -> dict[str, float]:
    """Return robustness indicators for repeated optimization runs."""
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        raise ValueError("scores must not be empty")

    iqr = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "cv": float(np.std(values) / max(abs(np.mean(values)), 1e-12)),
        "iqr": iqr,
        "robustness": float(1.0 / (1.0 + (np.std(values) / max(abs(np.mean(values)), 1e-12)))),
    }


def paired_significance_test(scores_a: Sequence[float], scores_b: Sequence[float]) -> dict[str, float]:
    """Run a paired significance test between two sets of scores."""
    values_a = np.asarray(scores_a, dtype=float)
    values_b = np.asarray(scores_b, dtype=float)
    if values_a.size != values_b.size or values_a.size == 0:
        raise ValueError("scores_a and scores_b must be non-empty and have the same length")

    if scipy_stats is not None:
        statistic, pvalue = scipy_stats.ttest_rel(values_a, values_b)
        return {"statistic": float(statistic), "pvalue": float(pvalue)}

    differences = values_a - values_b
    mean_diff = float(np.mean(differences))
    std_diff = float(np.std(differences, ddof=1))
    if std_diff == 0.0:
        return {"statistic": 0.0, "pvalue": 1.0}
    statistic = mean_diff / (std_diff / sqrt(values_a.size))
    return {"statistic": float(statistic), "pvalue": 1.0}


def aggregate_runs(scores: Sequence[float], optimum: float = 0.0, success_threshold: float = 0.0) -> dict[str, float]:
    """Convenience aggregation for benchmark runs."""
    distribution = distribution_analysis(scores)
    return {
        **distribution,
        "success_rate": success_rate(scores, threshold=success_threshold, optimum=optimum),
        "gap_mean": float(np.mean(np.abs(np.asarray(scores, dtype=float) - optimum))),
    }
