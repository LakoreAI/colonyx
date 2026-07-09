from __future__ import annotations

from colonyx import AutoColony, benchmark_optimizer, benchmark_optimizers, benchmark_report, compare_benchmark_results


def sphere(x):
    return sum(value * value for value in x)


def test_benchmark_helpers_produce_summary_and_comparison():
    factories = {
        "pso": lambda: AutoColony(mode="pso", n_iterations=15, random_state=7),
        "gwo": lambda: AutoColony(mode="gwo", n_iterations=15, random_state=7),
    }

    results = benchmark_optimizers(factories, sphere, bounds=[(-5, 5), (-5, 5), (-5, 5)], repeats=2)
    report = benchmark_report(results)
    comparison = compare_benchmark_results(results["pso"], results["gwo"])

    assert set(results) == {"pso", "gwo"}
    assert set(report) == {"pso", "gwo"}
    assert "mean_score" in report["pso"]
    assert set(comparison) == {"score_delta", "elapsed_delta", "memory_delta", "efficiency_delta"}


def test_single_benchmark_result_is_well_formed():
    result = benchmark_optimizer(
        "pso",
        lambda: AutoColony(mode="pso", n_iterations=10, random_state=11),
        sphere,
        bounds=[(-5, 5), (-5, 5)],
        repeats=2,
    )

    assert result.name == "pso"
    assert len(result.scores) == 2
    assert result.best_score <= result.mean_score

