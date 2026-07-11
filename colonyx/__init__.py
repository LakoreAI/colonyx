"""
colonyx: Swarm Intelligence Optimization Library

A Python library for solving optimization problems using swarm intelligence algorithms
like Ant Colony Optimization (ACO), Particle Swarm Optimization (PSO), and 
Artificial Bee Colony (ABC) with a scikit-learn compatible interface.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("colonyx")
except PackageNotFoundError:  # pragma: no cover - local source checkout
    __version__ = "0.1.0"
__author__ = "Minh, Le Duc"
__email__ = "minh.leduc.0210@gmail.com"

# Import the compiled Rust core (built by maturin as colonyx._colonyx)
from . import _colonyx
from ._colonyx import (
    BacterialForagingOptimizer,
    BatAlgorithm,
    BeeColony,
    CuckooSearch,
    CmaEsOptimizer,
    DifferentialEvolution,
    FireflyOptimizer,
    GlowwormOptimizer,
    GreyWolfOptimizer,
    ParticleSwarm,
    SimulatedAnnealing,
    AntColony,
    two_opt,
)

# Import the main interface
from .auto import AutoColony
from .base import OptimizerMixin
from .benchmarks import (
    BenchmarkProblem,
    ackley,
    benchmark_suite,
    griewank,
    rastrigin,
    rosenbrock,
    schwefel,
    sphere,
)
from .metrics import (
    BenchmarkResult,
    aggregate_runs,
    benchmark_optimizer,
    benchmark_optimizers,
    benchmark_report,
    benchmark_visualization,
    computational_efficiency,
    convergence_rate,
    distribution_analysis,
    profile_callable,
    profile_optimization_run,
    ProfilingResult,
    compare_benchmark_results,
    optimization_gap,
    paired_significance_test,
    robustness_analysis,
    success_rate,
)
from .utils import check_bounds, check_objective_function, check_optimization_problem

# Import individual algorithms (will be implemented)
# from .algorithms import AntColonyOptimizer, ParticleSwarmOptimizer, ArtificialBeeColonyOptimizer

# Import utilities (will be implemented)
# from .utils import check_optimization_problem, check_bounds

# Import datasets (will be implemented)
# from .datasets import load_tsp_data, benchmark_functions

__all__ = [
    "AutoColony",
    "OptimizerMixin",
    "BenchmarkProblem",
    "benchmark_suite",
    "two_opt",
    "AntColony",
    "ParticleSwarm",
    "BeeColony",
    "GreyWolfOptimizer",
    "FireflyOptimizer",
    "SimulatedAnnealing",
    "CuckooSearch",
    "CmaEsOptimizer",
    "BatAlgorithm",
    "GlowwormOptimizer",
    "sphere",
    "rosenbrock",
    "rastrigin",
    "ackley",
    "griewank",
    "schwefel",
    "BenchmarkResult",
    "BacterialForagingOptimizer",
    "DifferentialEvolution",
    "convergence_rate",
    "optimization_gap",
    "success_rate",
    "computational_efficiency",
    "distribution_analysis",
    "robustness_analysis",
    "paired_significance_test",
    "aggregate_runs",
    "ProfilingResult",
    "profile_callable",
    "profile_optimization_run",
    "benchmark_optimizer",
    "benchmark_optimizers",
    "benchmark_report",
    "benchmark_visualization",
    "compare_benchmark_results",
    "check_bounds",
    "check_objective_function",
    "check_optimization_problem",
    # "AntColonyOptimizer", 
    # "ParticleSwarmOptimizer",
    # "ArtificialBeeColonyOptimizer",
    # "check_optimization_problem",
    # "check_bounds",
    # "load_tsp_data",
    # "benchmark_functions",
] 
