"""
colonyx: Swarm Intelligence Optimization Library

A Python library for solving optimization problems using swarm intelligence algorithms
like Ant Colony Optimization (ACO), Particle Swarm Optimization (PSO), and 
Artificial Bee Colony (ABC) with a scikit-learn compatible interface.
"""

__version__ = "0.1.0"
__author__ = "Minh, Le Duc"
__email__ = "minh.leduc.0210@gmail.com"

# Import the compiled Rust core (built by maturin as colonyx._colonyx)
from . import _colonyx

# Import the main interface
from .auto import AutoColony

# Import individual algorithms (will be implemented)
# from .algorithms import AntColonyOptimizer, ParticleSwarmOptimizer, ArtificialBeeColonyOptimizer

# Import utilities (will be implemented)
# from .utils import check_optimization_problem, check_bounds

# Import datasets (will be implemented)
# from .datasets import load_tsp_data, benchmark_functions

__all__ = [
    "AutoColony",
    # "AntColonyOptimizer", 
    # "ParticleSwarmOptimizer",
    # "ArtificialBeeColonyOptimizer",
    # "check_optimization_problem",
    # "check_bounds",
    # "load_tsp_data",
    # "benchmark_functions",
] 