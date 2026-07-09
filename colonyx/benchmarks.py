"""Standard continuous benchmark problems for optimization experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import e, pi
from typing import Callable

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


@dataclass(frozen=True, slots=True)
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
    }

