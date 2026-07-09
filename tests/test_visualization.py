from __future__ import annotations

from colonyx import AutoColony, benchmark_optimizers, benchmark_visualization


def sphere(x):
    return sum(value * value for value in x)


def test_benchmark_visualization_returns_text_chart():
    results = benchmark_optimizers(
        {
            "pso": lambda: AutoColony(mode="pso", n_iterations=10, random_state=7),
            "fa": lambda: AutoColony(mode="fa", n_iterations=10, random_state=7),
        },
        sphere,
        bounds=[(-5, 5), (-5, 5), (-5, 5)],
        repeats=1,
    )

    chart = benchmark_visualization(results)

    assert "Benchmark visualization" in chart
    assert "pso" in chart
    assert "fa" in chart

