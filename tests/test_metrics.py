"""Tests for metric and statistical helpers."""

import numpy as np

from colonyx import (
    AutoColony,
    aggregate_runs,
    computational_efficiency,
    convergence_rate,
    distribution_analysis,
    optimization_gap,
    paired_significance_test,
    profile_optimization_run,
    robustness_analysis,
    success_rate,
)


def test_scalar_metrics_are_computed():
    history = [10.0, 6.0, 4.0, 2.0]

    assert convergence_rate(history) == 0.8
    assert optimization_gap(1.5, optimum=1.0) == 0.5
    assert success_rate([0.1, 0.3, 0.9], threshold=0.5) == 2 / 3
    assert computational_efficiency(history) > 0


def test_distribution_and_robustness_reports():
    scores = [1.0, 2.0, 3.0, 4.0]

    distribution = distribution_analysis(scores)
    robustness = robustness_analysis(scores)

    assert distribution["count"] == 4.0
    assert np.isfinite(list(distribution.values())).all()
    assert np.isfinite(list(robustness.values())).all()


def test_paired_significance_and_aggregation_helpers():
    scores_a = [1.0, 1.2, 0.9]
    scores_b = [1.4, 1.1, 1.0]

    comparison = paired_significance_test(scores_a, scores_b)
    aggregate = aggregate_runs(scores_a, optimum=0.0, success_threshold=1.5)

    assert set(comparison) == {"statistic", "pvalue"}
    assert "success_rate" in aggregate


def test_autocolony_reports_metrics_for_multiple_runs():
    optimizer = AutoColony(mode="pso", n_iterations=25, random_state=9)
    optimizer.fit(lambda x: sum(value * value for value in x), bounds=[(-5, 5), (-5, 5)])

    performance = optimizer.performance_metrics(optimum=0.0)
    summary = AutoColony.summarize_runs([1.0, 2.0, 3.0], optimum=0.0)
    report = AutoColony.robustness_report([1.0, 2.0, 3.0])

    assert "optimization_gap" in performance
    assert "success_rate" in performance
    assert "mean" in summary
    assert "robustness" in report


def test_profile_optimization_run_captures_timing_and_memory():
    optimizer = AutoColony(mode="pso", n_iterations=15, random_state=12)

    profile = profile_optimization_run(
        optimizer,
        lambda x: sum(value * value for value in x),
        bounds=[(-5, 5), (-5, 5)],
    )

    assert profile.elapsed_seconds >= 0.0
    assert profile.peak_memory_kib >= 0.0
    assert profile.score_history_length >= 1
    assert np.isfinite([profile.best_score, profile.improvement_rate, profile.efficiency]).all()
