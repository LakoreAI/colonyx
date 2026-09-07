"""Standard continuous benchmark problems for optimization experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from math import e, pi
from typing import Callable, Iterable, Union

import numpy as np


def _as_vector(values: object) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError("benchmark objectives expect a one-dimensional input vector")
    if vector.size == 0:
        raise ValueError("benchmark objectives require at least one dimension")
    return vector


def sphere(x: object) -> float:
    """Sphere function with global minimum at the origin."""
    vector = _as_vector(x)
    return float(np.sum(vector**2))


def rosenbrock(x: object) -> float:
    """Rosenbrock valley with global minimum at ``[1, ..., 1]``."""
    vector = _as_vector(x)
    if vector.size < 2:
        raise ValueError("rosenbrock requires at least two dimensions")
    return float(np.sum(100.0 * (vector[1:] - vector[:-1] ** 2) ** 2 + (1.0 - vector[:-1]) ** 2))


def rastrigin(x: object) -> float:
    """Rastrigin function with many local minima and a global minimum at the origin."""
    vector = _as_vector(x)
    return float(10.0 * vector.size + np.sum(vector**2 - 10.0 * np.cos(2.0 * pi * vector)))


def ackley(x: object) -> float:
    """Ackley function with a global minimum at the origin."""
    vector = _as_vector(x)
    mean_square = float(np.mean(vector**2))
    mean_cosine = float(np.mean(np.cos(2.0 * pi * vector)))
    return float(-20.0 * np.exp(-0.2 * np.sqrt(mean_square)) - np.exp(mean_cosine) + 20.0 + e)


def griewank(x: object) -> float:
    """Griewank function with a global minimum at the origin."""
    vector = _as_vector(x)
    indices = np.sqrt(np.arange(1.0, vector.size + 1.0))
    return float(np.sum(vector**2) / 4000.0 - np.prod(np.cos(vector / indices)) + 1.0)


def schwefel(x: object) -> float:
    """Schwefel function with a global minimum near ``418.9829`` in each dimension."""
    vector = _as_vector(x)
    return float(418.9829 * vector.size - np.sum(vector * np.sin(np.sqrt(np.abs(vector)))))


def levy(x: object) -> float:
    """Levy function N.13 with a global minimum of 0 at ``[1, ..., 1]``."""
    vector = _as_vector(x)
    w = 1.0 + (vector - 1.0) / 4.0
    term1 = np.sin(pi * w[0]) ** 2
    term3 = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * pi * w[-1]) ** 2)
    middle = w[:-1]
    term2 = np.sum((middle - 1.0) ** 2 * (1.0 + 10.0 * np.sin(pi * middle + 1.0) ** 2))
    return float(term1 + term2 + term3)


def zakharov(x: object) -> float:
    """Zakharov function with a global minimum of 0 at the origin."""
    vector = _as_vector(x)
    indices = np.arange(1.0, vector.size + 1.0)
    linear_term = np.sum(0.5 * indices * vector)
    return float(np.sum(vector**2) + linear_term**2 + linear_term**4)


def michalewicz(x: object, m: int = 10) -> float:
    """Michalewicz function; global minimum is approximately -1.8013 for 2D.

    The steepness parameter ``m`` defaults to the commonly used value of 10.
    Unlike the other benchmarks here, its optimum has no simple closed form
    beyond low, specifically studied dimensions (2D is well known at
    approximately -1.8013), so the ``minimum``/``optimum`` recorded in
    ``benchmark_suite()`` are for the 2D case only.
    """
    vector = _as_vector(x)
    indices = np.arange(1.0, vector.size + 1.0)
    return float(-np.sum(np.sin(vector) * np.sin(indices * vector**2 / pi) ** (2 * m)))


@dataclass(frozen=True)
class BenchmarkProblem:
    """Descriptor for a named benchmark objective."""

    name: str
    objective: Callable[[object], float]
    bounds: tuple[tuple[float, float], ...]
    minimum: float
    optimum: tuple[float, ...]


def benchmark_suite() -> dict[str, BenchmarkProblem]:
    """Return a small suite of standard continuous benchmark problems."""
    return {
        "sphere": BenchmarkProblem(
            name="sphere",
            objective=sphere,
            bounds=((-5.12, 5.12),),
            minimum=0.0,
            optimum=(0.0,),
        ),
        "rosenbrock": BenchmarkProblem(
            name="rosenbrock",
            objective=rosenbrock,
            bounds=((-2.0, 2.0), (-2.0, 2.0)),
            minimum=0.0,
            optimum=(1.0, 1.0),
        ),
        "rastrigin": BenchmarkProblem(
            name="rastrigin",
            objective=rastrigin,
            bounds=((-5.12, 5.12),),
            minimum=0.0,
            optimum=(0.0,),
        ),
        "ackley": BenchmarkProblem(
            name="ackley",
            objective=ackley,
            bounds=((-32.768, 32.768),),
            minimum=0.0,
            optimum=(0.0,),
        ),
        "griewank": BenchmarkProblem(
            name="griewank",
            objective=griewank,
            bounds=((-600.0, 600.0),),
            minimum=0.0,
            optimum=(0.0,),
        ),
        "schwefel": BenchmarkProblem(
            name="schwefel",
            objective=schwefel,
            bounds=((-500.0, 500.0),),
            minimum=0.0,
            optimum=(420.9687,),
        ),
        "levy": BenchmarkProblem(
            name="levy",
            objective=levy,
            bounds=((-10.0, 10.0),),
            minimum=0.0,
            optimum=(1.0,),
        ),
        "zakharov": BenchmarkProblem(
            name="zakharov",
            objective=zakharov,
            bounds=((-5.0, 10.0),),
            minimum=0.0,
            optimum=(0.0,),
        ),
        "michalewicz": BenchmarkProblem(
            name="michalewicz",
            objective=michalewicz,
            bounds=((0.0, pi), (0.0, pi)),
            minimum=-1.8013,
            optimum=(2.20319, 1.57049),
        ),
    }


def load_tsplib(path_or_lines: Union[str, "os.PathLike[str]", Iterable[str]]) -> np.ndarray:
    """Parse a TSPLIB instance into a square Euclidean distance matrix.

    Accepts either a path to a ``.tsp`` file or an iterable of its lines
    (e.g. an in-memory string split with ``.splitlines()``). Only
    ``TYPE: TSP`` instances with ``EDGE_WEIGHT_TYPE: EUC_2D`` are supported —
    the most common TSPLIB format, where node coordinates are given directly
    in ``NODE_COORD_SECTION`` and edge weights are plain Euclidean distances.
    Other edge-weight types (e.g. ``GEO``, ``ATT``, explicit weight matrices)
    are not handled by this minimal loader.

    The resulting matrix can be passed directly to
    ``AutoColony(mode="aco").fit(distance_matrix)``.
    """
    if isinstance(path_or_lines, (str, os.PathLike)):
        with open(path_or_lines, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    else:
        lines = list(path_or_lines)

    edge_weight_type = None
    coords: dict[int, tuple[float, float]] = {}
    in_coord_section = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if line.upper().startswith("EDGE_WEIGHT_TYPE"):
            edge_weight_type = line.split(":", 1)[1].strip() if ":" in line else None
            continue

        if line.upper().startswith("NODE_COORD_SECTION"):
            in_coord_section = True
            continue

        if line == "EOF" or line == "-1":
            break

        if in_coord_section:
            parts = line.split()
            if len(parts) < 3:
                continue
            node_id = int(float(parts[0]))
            x, y = float(parts[1]), float(parts[2])
            coords[node_id] = (x, y)

    if not coords:
        raise ValueError("load_tsplib: no NODE_COORD_SECTION with coordinates found")

    if edge_weight_type is not None and edge_weight_type.upper() != "EUC_2D":
        raise ValueError(
            f"load_tsplib only supports EDGE_WEIGHT_TYPE 'EUC_2D', got {edge_weight_type!r}"
        )

    node_ids = sorted(coords)
    points = np.array([coords[node_id] for node_id in node_ids], dtype=float)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    distance_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    return distance_matrix

