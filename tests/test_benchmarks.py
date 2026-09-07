from __future__ import annotations

import math

import numpy as np
import pytest

from colonyx import (
    ackley,
    benchmark_suite,
    griewank,
    levy,
    load_tsplib,
    michalewicz,
    rastrigin,
    rosenbrock,
    schwefel,
    sphere,
    zakharov,
)


def test_benchmark_suite_exposes_named_problems():
    suite = benchmark_suite()

    assert {
        "sphere",
        "rosenbrock",
        "rastrigin",
        "ackley",
        "griewank",
        "schwefel",
        "levy",
        "zakharov",
        "michalewicz",
    }.issubset(suite)
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


def test_levy_has_zero_minimum_at_ones():
    assert math.isclose(levy([1.0, 1.0, 1.0]), 0.0, abs_tol=1e-9)


def test_zakharov_has_zero_minimum_at_origin():
    assert zakharov([0.0, 0.0, 0.0]) == 0.0


def test_michalewicz_matches_known_2d_optimum():
    suite = benchmark_suite()
    optimum = suite["michalewicz"].optimum
    assert math.isclose(michalewicz(optimum), suite["michalewicz"].minimum, abs_tol=1e-3)


def test_load_tsplib_parses_euc_2d_square_instance():
    tsplib_text = """
    NAME: square4
    TYPE: TSP
    DIMENSION: 4
    EDGE_WEIGHT_TYPE: EUC_2D
    NODE_COORD_SECTION
    1 0.0 0.0
    2 1.0 0.0
    3 1.0 1.0
    4 0.0 1.0
    EOF
    """
    matrix = load_tsplib(tsplib_text.splitlines())

    assert matrix.shape == (4, 4)
    assert np.allclose(np.diag(matrix), 0.0)
    assert np.allclose(matrix, matrix.T)
    assert math.isclose(matrix[0, 1], 1.0)
    assert math.isclose(matrix[0, 2], math.sqrt(2.0))


def test_load_tsplib_rejects_unsupported_edge_weight_type():
    tsplib_text = """
    TYPE: TSP
    EDGE_WEIGHT_TYPE: GEO
    NODE_COORD_SECTION
    1 0.0 0.0
    2 1.0 0.0
    EOF
    """
    with pytest.raises(ValueError):
        load_tsplib(tsplib_text.splitlines())


def test_load_tsplib_requires_node_coord_section():
    with pytest.raises(ValueError):
        load_tsplib(["TYPE: TSP", "EDGE_WEIGHT_TYPE: EUC_2D", "EOF"])
