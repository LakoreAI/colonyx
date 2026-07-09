from __future__ import annotations

import math

import numpy as np

from colonyx import ackley, benchmark_suite, griewank, rastrigin, rosenbrock, schwefel, sphere


def test_benchmark_suite_exposes_named_problems():
    suite = benchmark_suite()

    assert {"sphere", "rosenbrock", "rastrigin", "ackley", "griewank", "schwefel"}.issubset(suite)
    assert suite["sphere"].objective([0.0, 0.0]) == 0.0
    assert suite["rosenbrock"].minimum == 0.0


def test_standard_benchmarks_have_expected_optima():
    assert sphere([0.0, 0.0, 0.0]) == 0.0
    assert rosenbrock([1.0, 1.0]) == 0.0
    assert rastrigin([0.0, 0.0]) == 0.0
    assert math.isclose(ackley([0.0, 0.0]), 0.0, abs_tol=1e-12)
    assert griewank([0.0, 0.0]) == 0.0
    assert schwefel([420.9687, 420.9687]) >= 0.0


def test_benchmarks_accept_numpy_inputs():
    vector = np.array([0.0, 0.0, 0.0])
    assert sphere(vector) == 0.0
