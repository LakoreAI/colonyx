"""
AutoColony: Main interface for swarm intelligence optimization algorithms
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils._tags import InputTags, Tags, TargetTags, TransformerTags
from sklearn.utils.validation import validate_data

from .base import OptimizerMixin
from .metrics import (
    aggregate_runs,
    computational_efficiency,
    convergence_rate,
    distribution_analysis,
    profile_optimization_run,
    optimization_gap,
    paired_significance_test,
    robustness_analysis,
    success_rate,
)
from .utils import check_bounds, check_objective_function, check_optimization_problem


class AutoColony(OptimizerMixin, TransformerMixin, BaseEstimator):
    """
    Unified interface for swarm intelligence optimization algorithms.

    Similar to HuggingFace's AutoModel, this class provides a single interface
    for multiple optimization algorithms selected via the ``mode`` parameter.

    Parameters
    ----------
    mode : str, default='auto'
        Algorithm selection mode:
        - 'auto': Automatically select algorithm based on problem type
        - 'aco': Ant Colony Optimization
        - 'pso': Particle Swarm Optimization
        - 'abc': Artificial Bee Colony
        - 'gwo': Grey Wolf Optimizer
        - 'fa': Firefly Algorithm
        - 'sa': Simulated Annealing
        - 'cs': Cuckoo Search
        - 'ba': Bat Algorithm
        - 'gso': Glowworm Swarm Optimization
        - 'bfo': Bacterial Foraging Optimizer
        - 'de': Differential Evolution
    n_iterations : int, default=100
        Number of iterations to run.
    random_state : int, default=None
        Random seed for reproducibility.
    Algorithm-specific parameters
        ``n_ants``, ``alpha``, ``beta``, ``rho``, ``q`` for ACO;
        ``n_particles``, ``w``, ``c1``, ``c2`` for PSO;
        ``n_bees`` and ``limit`` for ABC;
        ``n_wolves`` for GWO;
        ``n_fireflies``, ``beta0``, ``gamma``, ``fa_alpha`` for FA;
        ``initial_temperature``, ``cooling_rate``, ``step_scale`` for SA;
        ``n_nests``, ``pa``, ``levy_scale`` for CS;
        ``n_bats``, ``fmin``, ``fmax``, ``bat_alpha``, ``bat_gamma``, ``loudness``, ``pulse_rate`` for BA;
        ``n_worms``, ``luciferin_decay``, ``luciferin_enhancement`` for GSO;
        ``n_bacteria``, ``n_chemotactic_steps``, ``n_reproduction_steps``, ``elimination_probability`` for BFO;
        ``n_individuals``, ``f``, ``cr`` for DE.
    """

    _valid_modes = ("auto", "aco", "pso", "abc", "gwo", "fa", "sa", "cs", "ba", "gso", "bfo", "de", "cmaes")

    def __init__(
        self,
        mode: str = "auto",
        n_iterations: int = 100,
        random_state: Optional[int] = None,
        n_ants: int = 50,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.5,
        q: float = 1.0,
        n_particles: int = 30,
        w: float = 0.9,
        c1: float = 2.0,
        c2: float = 2.0,
        n_bees: int = 50,
        limit: int = 10,
        n_wolves: int = 30,
        n_fireflies: int = 30,
        beta0: float = 1.0,
        gamma: float = 1.0,
        fa_alpha: float = 0.2,
        initial_temperature: float = 10.0,
        cooling_rate: float = 0.95,
        step_scale: float = 0.1,
        n_nests: int = 25,
        pa: float = 0.25,
        levy_scale: float = 1.0,
        n_bats: int = 30,
        fmin: float = 0.0,
        fmax: float = 2.0,
        bat_alpha: float = 0.9,
        bat_gamma: float = 0.9,
        loudness: float = 1.0,
        pulse_rate: float = 0.5,
        n_worms: int = 30,
        luciferin_decay: float = 0.4,
        luciferin_enhancement: float = 0.6,
        gso_step_size: float = 0.1,
        neighborhood_radius: float = 1.0,
        n_bacteria: int = 30,
        n_chemotactic_steps: int = 10,
        n_reproduction_steps: int = 4,
        elimination_probability: float = 0.25,
        bfo_step_scale: float = 0.1,
        use_two_opt: bool = True,
        n_individuals: int = 40,
        f: float = 0.8,
        cr: float = 0.9,
        cmaes_sigma: float = 0.5,
    ):
        self.mode = mode
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.n_bees = n_bees
        self.limit = limit
        self.n_wolves = n_wolves
        self.n_fireflies = n_fireflies
        self.beta0 = beta0
        self.gamma = gamma
        self.fa_alpha = fa_alpha
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.step_scale = step_scale
        self.n_nests = n_nests
        self.pa = pa
        self.levy_scale = levy_scale
        self.n_bats = n_bats
        self.fmin = fmin
        self.fmax = fmax
        self.bat_alpha = bat_alpha
        self.bat_gamma = bat_gamma
        self.loudness = loudness
        self.pulse_rate = pulse_rate
        self.n_worms = n_worms
        self.luciferin_decay = luciferin_decay
        self.luciferin_enhancement = luciferin_enhancement
        self.gso_step_size = gso_step_size
        self.neighborhood_radius = neighborhood_radius
        self.n_bacteria = n_bacteria
        self.n_chemotactic_steps = n_chemotactic_steps
        self.n_reproduction_steps = n_reproduction_steps
        self.elimination_probability = elimination_probability
        self.bfo_step_scale = bfo_step_scale
        self.use_two_opt = use_two_opt
        self.n_individuals = n_individuals
        self.f = f
        self.cr = cr
        self.cmaes_sigma = cmaes_sigma

    def _more_tags(self) -> Dict[str, Any]:
        return {
            "requires_y": False,
            "non_deterministic": False,
            "X_types": ["2darray", "object"],
        }

    def __sklearn_tags__(self):
        return Tags(
            estimator_type=None,
            target_tags=TargetTags(required=False),
            transformer_tags=TransformerTags(),
            requires_fit=True,
            non_deterministic=False,
            input_tags=InputTags(two_d_array=True),
        )

    def _check_is_fitted(self) -> None:
        if not getattr(self, "_fitted", False):
            raise NotFittedError("Must call fit() before using this estimator")

    def __sklearn_is_fitted__(self):
        return getattr(self, "_fitted", False)

    def _detect_problem_type(self, X, y=None):
        """Auto-detect problem type for algorithm selection."""
        problem = check_optimization_problem(X, y)
        if problem["problem_type"] == "discrete":
            return "aco"
        if problem["problem_type"] == "tabular":
            return "sklearn"
        return "pso"

    def _infer_problem_dimension(self, X, bounds=None):
        """Infer dimensionality for heuristic algorithm selection."""
        if bounds is not None:
            return len(bounds)
        if callable(X):
            return None

        arr = np.asarray(X)
        if arr.ndim != 2:
            return None
        if arr.shape[0] == arr.shape[1]:
            return int(arr.shape[0])
        return int(arr.shape[1])

    def recommend_algorithm(self, X, y=None, bounds=None):
        """Recommend an algorithm mode with a short rationale."""
        problem = check_optimization_problem(X, y)
        dimension = self._infer_problem_dimension(X, bounds)

        if problem["problem_type"] == "discrete":
            return {
                "mode": "aco",
                "reason": "square distance matrix detected",
                "problem_type": "discrete",
                "dimension": dimension,
            }

        if problem["problem_type"] == "tabular":
            return {
                "mode": "sklearn",
                "reason": "supervised tabular data detected",
                "problem_type": "tabular",
                "dimension": dimension,
            }

        if dimension is None:
            mode = "pso"
            reason = "continuous objective detected; defaulting to PSO"
        elif dimension <= 4:
            mode = "pso"
            reason = f"low-dimensional continuous objective ({dimension} dims)"
        else:
            mode = "abc"
            reason = f"higher-dimensional continuous objective ({dimension} dims)"

        return {
            "mode": mode,
            "reason": reason,
            "problem_type": "continuous",
            "dimension": dimension,
        }

    def parameter_mapping(self, algorithm_mode: Optional[str] = None):
        """Map unified parameter names to backend algorithm parameters."""
        mode = algorithm_mode or self.mode
        mappings = {
            "aco": {
                "n_iterations": "n_iterations",
                "random_state": "random_state",
                "n_ants": "n_ants",
                "alpha": "alpha",
                "beta": "beta",
                "rho": "rho",
                "q": "q",
            },
            "pso": {
                "n_iterations": "n_iterations",
                "random_state": "random_state",
                "n_particles": "n_particles",
                "w": "w",
                "c1": "c1",
                "c2": "c2",
            },
            "abc": {
                "n_iterations": "n_iterations",
                "random_state": "random_state",
                "n_bees": "n_bees",
                "limit": "limit",
            },
            "gwo": {
                "n_iterations": "n_iterations",
                "random_state": "random_state",
                "n_wolves": "n_wolves",
            },
            "fa": {
                "n_iterations": "n_iterations",
                "random_state": "random_state",
                "n_fireflies": "n_fireflies",
                "beta0": "beta0",
                "gamma": "gamma",
                "fa_alpha": "fa_alpha",
            },
            "sa": {
                "n_iterations": "n_iterations",
                "random_state": "random_state",
                "initial_temperature": "initial_temperature",
                "cooling_rate": "cooling_rate",
                "step_scale": "step_scale",
            },
            "cs": {
                "n_iterations": "n_iterations",
                "random_state": "random_state",
                "n_nests": "n_nests",
                "pa": "pa",
                "levy_scale": "levy_scale",
            },
            "ba": {
                "n_iterations": "n_iterations",
                "random_state": "random_state",
                "n_bats": "n_bats",
                "fmin": "fmin",
                "fmax": "fmax",
                "bat_alpha": "bat_alpha",
                "bat_gamma": "bat_gamma",
                "loudness": "loudness",
                "pulse_rate": "pulse_rate",
            },
            "gso": {
                "n_iterations": "n_iterations",
                "random_state": "random_state",
                "n_worms": "n_worms",
                "luciferin_decay": "luciferin_decay",
                "luciferin_enhancement": "luciferin_enhancement",
                "gso_step_size": "gso_step_size",
                "neighborhood_radius": "neighborhood_radius",
            },
            "bfo": {
                "n_iterations": "n_iterations",
                "random_state": "random_state",
                "n_bacteria": "n_bacteria",
                "n_chemotactic_steps": "n_chemotactic_steps",
                "n_reproduction_steps": "n_reproduction_steps",
                "elimination_probability": "elimination_probability",
                "bfo_step_scale": "bfo_step_scale",
            },
            "de": {
                "n_iterations": "n_iterations",
                "random_state": "random_state",
                "n_individuals": "n_individuals",
                "f": "f",
                "cr": "cr",
            },
            "cmaes": {
                "n_iterations": "n_iterations",
                "random_state": "random_state",
                "n_individuals": "n_individuals",
                "cmaes_sigma": "cmaes_sigma",
            },
        }
        return mappings.get(mode, {})

    def suggest_parameters(self, X, y=None, bounds=None):
        """Suggest mode-specific parameters for the current problem."""
        recommendation = self.recommend_algorithm(X, y=y, bounds=bounds)
        mode = self.mode if self.mode != "auto" else recommendation["mode"]
        dimension = recommendation["dimension"] or self._infer_problem_dimension(X, bounds) or 1

        if mode == "aco":
            size = dimension if dimension is not None else 50
            return {
                "mode": "aco",
                "n_iterations": max(self.n_iterations, 50),
                "n_ants": min(max(20, size), 80),
                "alpha": 1.0,
                "beta": 2.0 if size < 40 else 3.0,
                "rho": 0.5,
                "q": 1.0,
            }

        if mode == "abc":
            size = dimension if dimension is not None else 10
            return {
                "mode": "abc",
                "n_iterations": max(self.n_iterations, 80),
                "n_bees": min(max(30, size * 4), 80),
                "limit": max(10, size * 2),
            }

        if mode == "gwo":
            size = dimension if dimension is not None else 10
            return {
                "mode": "gwo",
                "n_iterations": max(self.n_iterations, 80),
                "n_wolves": min(max(10, size * 3), 60),
            }

        if mode == "fa":
            size = dimension if dimension is not None else 10
            return {
                "mode": "fa",
                "n_iterations": max(self.n_iterations, 80),
                "n_fireflies": min(max(10, size * 3), 60),
                "beta0": 1.0,
                "gamma": 1.0,
                "fa_alpha": 0.2,
            }

        if mode == "sa":
            return {
                "mode": "sa",
                "n_iterations": max(self.n_iterations, 100),
                "initial_temperature": 10.0,
                "cooling_rate": 0.95,
                "step_scale": 0.1,
            }

        if mode == "cs":
            size = dimension if dimension is not None else 10
            return {
                "mode": "cs",
                "n_iterations": max(self.n_iterations, 80),
                "n_nests": min(max(10, size * 3), 50),
                "pa": 0.25,
                "levy_scale": 1.0,
            }

        if mode == "ba":
            size = dimension if dimension is not None else 10
            return {
                "mode": "ba",
                "n_iterations": max(self.n_iterations, 80),
                "n_bats": min(max(10, size * 3), 60),
                "fmin": 0.0,
                "fmax": 2.0,
                "bat_alpha": 0.9,
                "bat_gamma": 0.9,
                "loudness": 1.0,
                "pulse_rate": 0.5,
            }

        if mode == "gso":
            size = dimension if dimension is not None else 10
            return {
                "mode": "gso",
                "n_iterations": max(self.n_iterations, 80),
                "n_worms": min(max(10, size * 3), 60),
                "luciferin_decay": 0.4,
                "luciferin_enhancement": 0.6,
                "gso_step_size": 0.1,
                "neighborhood_radius": 1.0,
            }

        if mode == "bfo":
            size = dimension if dimension is not None else 10
            return {
                "mode": "bfo",
                "n_iterations": max(self.n_iterations, 80),
                "n_bacteria": min(max(10, size * 3), 60),
                "n_chemotactic_steps": 8,
                "n_reproduction_steps": 4,
                "elimination_probability": 0.25,
                "bfo_step_scale": 0.1,
            }

        if mode == "de":
            size = dimension if dimension is not None else 10
            return {
                "mode": "de",
                "n_iterations": max(self.n_iterations, 80),
                "n_individuals": min(max(20, size * 4), 80),
                "f": 0.8,
                "cr": 0.9,
            }

        if mode == "cmaes":
            size = dimension if dimension is not None else 10
            return {
                "mode": "cmaes",
                "n_iterations": max(self.n_iterations, 80),
                "n_individuals": min(max(10, size * 4), 60),
                "cmaes_sigma": 0.5,
            }

        return {
            "mode": "pso",
            "n_iterations": max(self.n_iterations, 50),
            "n_particles": min(max(20, (dimension or 3) * 5), 60),
            "w": 0.7,
            "c1": 1.5,
            "c2": 1.5,
        }

    def parameter_help(self, algorithm_mode: Optional[str] = None):
        """Return concise parameter help text for a specific algorithm mode."""
        mode = algorithm_mode or self.mode
        help_map = {
            "aco": "ACO params: n_ants, alpha, beta, rho, q",
            "pso": "PSO params: n_particles, w, c1, c2",
            "abc": "ABC params: n_bees, limit",
            "gwo": "GWO params: n_wolves",
            "fa": "FA params: n_fireflies, beta0, gamma, fa_alpha",
            "sa": "SA params: initial_temperature, cooling_rate, step_scale",
            "cs": "CS params: n_nests, pa, levy_scale",
            "ba": "BA params: n_bats, fmin, fmax, bat_alpha, bat_gamma, loudness, pulse_rate",
            "gso": "GSO params: n_worms, luciferin_decay, luciferin_enhancement, gso_step_size",
            "bfo": "BFO params: n_bacteria, n_chemotactic_steps, n_reproduction_steps, elimination_probability",
            "de": "DE params: n_individuals, f, cr",
            "cmaes": "CMA-ES params: n_individuals, cmaes_sigma",
            "auto": "Auto mode selects a backend from the input shape and bounds",
        }
        return help_map.get(mode, "Unknown mode")

    def _filter_params(self, algorithm_mode: str) -> Dict[str, Any]:
        """Filter parameters relevant to the specific algorithm."""
        base_params = {
            "n_iterations": self.n_iterations,
            "random_state": self.random_state,
        }

        if algorithm_mode == "aco":
            return {
                **base_params,
                "n_ants": self.n_ants,
                "alpha": self.alpha,
                "beta": self.beta,
                "rho": self.rho,
                "q": self.q,
                "use_two_opt": self.use_two_opt,
            }

        if algorithm_mode == "pso":
            return {
                **base_params,
                "n_particles": self.n_particles,
                "w": self.w,
                "c1": self.c1,
                "c2": self.c2,
            }

        if algorithm_mode == "abc":
            return {
                **base_params,
                "n_bees": self.n_bees,
                "limit": self.limit,
            }

        if algorithm_mode == "gwo":
            return {
                **base_params,
                "n_wolves": self.n_wolves,
            }

        if algorithm_mode == "fa":
            return {
                **base_params,
                "n_fireflies": self.n_fireflies,
                "beta0": self.beta0,
                "gamma": self.gamma,
                "fa_alpha": self.fa_alpha,
            }

        if algorithm_mode == "sa":
            return {
                **base_params,
                "initial_temperature": self.initial_temperature,
                "cooling_rate": self.cooling_rate,
                "step_scale": self.step_scale,
            }

        if algorithm_mode == "cs":
            return {
                **base_params,
                "n_nests": self.n_nests,
                "pa": self.pa,
                "levy_scale": self.levy_scale,
            }

        if algorithm_mode == "ba":
            return {
                **base_params,
                "n_bats": self.n_bats,
                "fmin": self.fmin,
                "fmax": self.fmax,
                "bat_alpha": self.bat_alpha,
                "bat_gamma": self.bat_gamma,
                "loudness": self.loudness,
                "pulse_rate": self.pulse_rate,
            }

        if algorithm_mode == "gso":
            return {
                **base_params,
                "n_worms": self.n_worms,
                "luciferin_decay": self.luciferin_decay,
                "luciferin_enhancement": self.luciferin_enhancement,
                "gso_step_size": self.gso_step_size,
                "neighborhood_radius": self.neighborhood_radius,
            }

        if algorithm_mode == "bfo":
            return {
                **base_params,
                "n_bacteria": self.n_bacteria,
                "n_chemotactic_steps": self.n_chemotactic_steps,
                "n_reproduction_steps": self.n_reproduction_steps,
                "elimination_probability": self.elimination_probability,
                "bfo_step_scale": self.bfo_step_scale,
            }

        if algorithm_mode == "de":
            return {
                **base_params,
                "n_individuals": self.n_individuals,
                "f": self.f,
                "cr": self.cr,
            }

        if algorithm_mode == "cmaes":
            return {
                **base_params,
                "n_individuals": self.n_individuals,
                "cmaes_sigma": self.cmaes_sigma,
            }

        return base_params

    def resolve_parameter_conflicts(self, algorithm_mode: str):
        """Resolve mode-specific parameters and record ignored parameters."""
        active = self._filter_params(algorithm_mode)
        allowed = set(active)
        ignored = [
            key
            for key in self.get_params()
            if key not in allowed and key not in {"mode", "n_iterations", "random_state"}
        ]
        self.parameter_conflicts_ = ignored
        return active

    def _create_algorithm(self, algorithm_mode: str):
        """Create the appropriate algorithm instance."""
        if algorithm_mode == "aco":
            from ._colonyx import AntColony

            params = self.resolve_parameter_conflicts("aco")
            return AntColony(
                n_ants=int(params["n_ants"]),
                n_iterations=int(params["n_iterations"]),
                alpha=float(params["alpha"]),
                beta=float(params["beta"]),
                rho=float(params["rho"]),
                q=float(params["q"]),
                use_two_opt=bool(params["use_two_opt"]),
                random_state=params["random_state"],
            )

        if algorithm_mode == "pso":
            from ._colonyx import ParticleSwarm

            params = self.resolve_parameter_conflicts("pso")
            return ParticleSwarm(
                n_particles=int(params["n_particles"]),
                n_iterations=int(params["n_iterations"]),
                w=float(params["w"]),
                c1=float(params["c1"]),
                c2=float(params["c2"]),
                random_state=params["random_state"],
            )

        if algorithm_mode == "abc":
            from ._colonyx import BeeColony

            params = self.resolve_parameter_conflicts("abc")
            return BeeColony(
                n_bees=int(params["n_bees"]),
                n_iterations=int(params["n_iterations"]),
                limit=int(params["limit"]),
                random_state=params["random_state"],
            )

        if algorithm_mode == "gwo":
            from ._colonyx import GreyWolfOptimizer

            params = self.resolve_parameter_conflicts("gwo")
            return GreyWolfOptimizer(
                n_wolves=int(params["n_wolves"]),
                n_iterations=int(params["n_iterations"]),
                random_state=params["random_state"],
            )

        if algorithm_mode == "fa":
            from ._colonyx import FireflyOptimizer

            params = self.resolve_parameter_conflicts("fa")
            return FireflyOptimizer(
                n_fireflies=int(params["n_fireflies"]),
                n_iterations=int(params["n_iterations"]),
                beta0=float(params["beta0"]),
                gamma=float(params["gamma"]),
                alpha=float(params["fa_alpha"]),
                random_state=params["random_state"],
            )

        if algorithm_mode == "sa":
            from ._colonyx import SimulatedAnnealing

            params = self.resolve_parameter_conflicts("sa")
            return SimulatedAnnealing(
                initial_temperature=float(params["initial_temperature"]),
                cooling_rate=float(params["cooling_rate"]),
                step_scale=float(params["step_scale"]),
                n_iterations=int(params["n_iterations"]),
                random_state=params["random_state"],
            )

        if algorithm_mode == "cs":
            from ._colonyx import CuckooSearch

            params = self.resolve_parameter_conflicts("cs")
            return CuckooSearch(
                n_nests=int(params["n_nests"]),
                n_iterations=int(params["n_iterations"]),
                pa=float(params["pa"]),
                alpha=float(params["levy_scale"]),
                levy_scale=float(params["levy_scale"]),
                random_state=params["random_state"],
            )

        if algorithm_mode == "ba":
            from ._colonyx import BatAlgorithm

            params = self.resolve_parameter_conflicts("ba")
            return BatAlgorithm(
                n_bats=int(params["n_bats"]),
                n_iterations=int(params["n_iterations"]),
                fmin=float(params["fmin"]),
                fmax=float(params["fmax"]),
                alpha=float(params["bat_alpha"]),
                gamma=float(params["bat_gamma"]),
                loudness=float(params["loudness"]),
                pulse_rate=float(params["pulse_rate"]),
                random_state=params["random_state"],
            )

        if algorithm_mode == "gso":
            from ._colonyx import GlowwormOptimizer

            params = self.resolve_parameter_conflicts("gso")
            return GlowwormOptimizer(
                n_worms=int(params["n_worms"]),
                n_iterations=int(params["n_iterations"]),
                luciferin_decay=float(params["luciferin_decay"]),
                luciferin_enhancement=float(params["luciferin_enhancement"]),
                step_size=float(params["gso_step_size"]),
                neighborhood_radius=float(params["neighborhood_radius"]),
                random_state=params["random_state"],
            )

        if algorithm_mode == "bfo":
            from ._colonyx import BacterialForagingOptimizer

            params = self.resolve_parameter_conflicts("bfo")
            return BacterialForagingOptimizer(
                n_bacteria=int(params["n_bacteria"]),
                n_iterations=int(params["n_iterations"]),
                n_chemotactic_steps=int(params["n_chemotactic_steps"]),
                n_reproduction_steps=int(params["n_reproduction_steps"]),
                elimination_probability=float(params["elimination_probability"]),
                step_scale=float(params["bfo_step_scale"]),
                random_state=params["random_state"],
            )

        if algorithm_mode == "de":
            from ._colonyx import DifferentialEvolution

            params = self.resolve_parameter_conflicts("de")
            return DifferentialEvolution(
                n_individuals=int(params["n_individuals"]),
                n_iterations=int(params["n_iterations"]),
                f=float(params["f"]),
                cr=float(params["cr"]),
                random_state=params["random_state"],
            )

        if algorithm_mode == "cmaes":
            from ._colonyx import CmaEsOptimizer

            params = self.resolve_parameter_conflicts("cmaes")
            return CmaEsOptimizer(
                n_individuals=int(params["n_individuals"]),
                n_iterations=int(params["n_iterations"]),
                sigma=float(params["cmaes_sigma"]),
                random_state=params["random_state"],
            )

        raise ValueError(f"Unknown algorithm mode: {algorithm_mode}")

    def _as_distance_matrix(self, X):
        """Validate and convert X into a square distance matrix."""
        arr = np.asarray(X, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(
                "ACO (mode='aco') expects a square distance matrix; "
                f"got array of shape {arr.shape}"
            )
        return arr.tolist()

    def _split_bounds(self, bounds):
        """Split a sequence of (low, high) pairs into lower/upper lists."""
        return check_bounds(bounds)

    def _fit_sklearn_compatibility(self, X, y):
        """Fallback path for sklearn-style tabular inputs."""
        features, targets = validate_data(
            self,
            X,
            y,
            reset=True,
            ensure_2d=True,
            dtype="numeric",
        )
        targets = np.asarray(targets, dtype=float).reshape(-1)

        best_index = int(np.argmin(targets))
        best_row = features[best_index].copy()

        self._compatibility_mode = True
        self._best_solution = best_row.tolist()
        self._best_score = float(targets[best_index])
        self.score_history_ = [self._best_score]
        self.population_ = features.tolist()
        self.best_solution_ = self._best_solution
        self.best_score_ = self._best_score
        self.n_features_in_ = features.shape[1]
        self._fitted = True
        return self

    def fit(self, X, y=None, bounds=None):
        """
        Fit the optimizer to the problem.

        Parameters
        ----------
        X : array-like or callable
            Problem data: a square distance matrix for ACO, or an objective
            function ``f(list[float]) -> float`` to minimize for PSO/ABC.
        y : array-like, optional
            Target values for sklearn-style tabular compatibility.
        bounds : sequence of (low, high), optional
            Per-dimension search-space bounds. Required for PSO and ABC.
        """
        self._fitted = False
        self._compatibility_mode = False
        self._algorithm = None
        self._algorithm_mode = None
        self._best_solution = None
        self._best_score = None
        self.score_history_ = []
        self.population_ = None

        if self.mode not in self._valid_modes:
            raise ValueError(f"Invalid mode '{self.mode}'. Must be one of {list(self._valid_modes)}")

        if self.mode == "auto":
            algorithm_mode = self.recommend_algorithm(X, y=y, bounds=bounds)["mode"]
        else:
            algorithm_mode = self.mode

        if algorithm_mode == "sklearn":
            return self._fit_sklearn_compatibility(X, y)

        if algorithm_mode == "aco":
            if not callable(X):
                distance_matrix = self._as_distance_matrix(X)
            else:
                raise ValueError("mode='aco' expects a square distance matrix, not a callable")

            self._algorithm = self._create_algorithm("aco")
            self._algorithm_mode = "aco"
            self._algorithm.fit(distance_matrix)

        elif algorithm_mode in ("pso", "abc", "gwo", "fa", "sa", "cs", "ba", "gso", "bfo", "de", "cmaes"):
            if callable(X):
                check_objective_function(X, probe_point=[0.0])
                if bounds is None:
                    raise ValueError(
                        f"mode='{algorithm_mode}' requires bounds=[(low, high), ...]"
                    )
                lower, upper = self._split_bounds(bounds)
                self._algorithm = self._create_algorithm(algorithm_mode)
                self._algorithm_mode = algorithm_mode
                self._algorithm.fit(X, lower, upper)
            else:
                if y is None:
                    raise ValueError(
                        f"mode='{algorithm_mode}' expects a callable objective function "
                        "or tabular data with y for sklearn compatibility"
                    )
                check_optimization_problem(X, y)
                return self._fit_sklearn_compatibility(X, y)
        else:
            raise ValueError(f"Unknown algorithm mode: {algorithm_mode}")

        self._best_solution = self._algorithm.predict()
        self._best_score = self._algorithm.score()
        self.score_history_ = list(getattr(self._algorithm, "history_", []))
        self.population_ = getattr(self._algorithm, "population_", None)
        self.best_solution_ = self._best_solution
        self.best_score_ = self._best_score
        self._fitted = True

        return self

    def predict(self, X=None):
        """
        Get the best solution found.

        When fitted on tabular sklearn-style inputs, ``X`` may be provided and
        the method returns a deterministic constant vector so the estimator can
        participate in sklearn checks and pipelines.
        """
        self._check_is_fitted()

        if self._compatibility_mode:
            if X is None:
                return self._best_solution
            features = validate_data(self, X, reset=False, ensure_2d=True, dtype=None)
            return np.full(features.shape[0], self._best_score, dtype=float)

        return self._best_solution

    def score(self, X=None, y=None):
        """
        Get the best score/fitness value.

        For optimization modes this returns the objective value directly.
        For sklearn-style compatibility mode it returns a negative MSE so the
        estimator behaves like a standard sklearn scorer.
        """
        self._check_is_fitted()

        if not self._compatibility_mode:
            return self._best_score

        if y is None:
            return -self._best_score

        if X is not None:
            validate_data(self, X, reset=False, ensure_2d=True, dtype=None)

        targets = np.asarray(y, dtype=float).reshape(-1)
        predictions = np.full(targets.shape[0], self._best_score, dtype=float)
        return -float(np.mean((predictions - targets) ** 2))

    def convergence_rate_score(self):
        """Measure relative improvement over the recorded score history."""
        self._check_is_fitted()
        history = np.asarray(getattr(self, "score_history_", []), dtype=float)
        if history.size < 2:
            return 0.0
        start_score = history[0]
        end_score = history[-1]
        denominator = max(abs(start_score), 1e-12)
        return float((start_score - end_score) / denominator)

    def diversity_score(self):
        """Measure the spread of the final population or fitted data."""
        self._check_is_fitted()
        population = getattr(self, "population_", None)
        if population is None:
            if self._compatibility_mode and self._best_solution is not None:
                return 0.0
            return 0.0

        population_array = np.asarray(population, dtype=float)
        if population_array.ndim != 2 or population_array.shape[0] < 2:
            return 0.0
        return float(np.mean(np.std(population_array, axis=0)))

    def robustness_score(self):
        """Estimate run stability from the tail of the score history."""
        self._check_is_fitted()
        history = np.asarray(getattr(self, "score_history_", []), dtype=float)
        if history.size == 0:
            return 0.0
        tail = history[-min(5, history.size):]
        mean_value = float(np.mean(tail))
        if abs(mean_value) < 1e-12:
            return 1.0 / (1.0 + float(np.std(tail)))
        coefficient_of_variation = float(np.std(tail) / abs(mean_value))
        return float(1.0 / (1.0 + coefficient_of_variation))

    def optimization_metrics(self):
        """Return a bundle of optimization-specific metrics."""
        return {
            "best_score": self.score(),
            "convergence_rate": self.convergence_rate_score(),
            "diversity": self.diversity_score(),
            "robustness": self.robustness_score(),
        }

    def performance_metrics(self, optimum: float = 0.0, success_threshold: float = 0.0):
        """Return a fuller performance metric bundle."""
        self._check_is_fitted()
        return {
            "best_score": self.score(),
            "optimization_gap": optimization_gap(self.score(), optimum=optimum),
            "success_rate": success_rate([self.score()], threshold=success_threshold, optimum=optimum),
            "convergence_rate": convergence_rate(getattr(self, "score_history_", [])),
            "computational_efficiency": computational_efficiency(getattr(self, "score_history_", [])),
        }

    @staticmethod
    def summarize_runs(scores, optimum: float = 0.0, success_threshold: float = 0.0):
        """Summarize multiple optimization runs."""
        return aggregate_runs(scores, optimum=optimum, success_threshold=success_threshold)

    @staticmethod
    def compare_runs(scores_a, scores_b):
        """Compare two run distributions with a paired significance test."""
        return paired_significance_test(scores_a, scores_b)

    @staticmethod
    def describe_run_distribution(scores):
        """Describe the distribution of run scores."""
        return distribution_analysis(scores)

    @staticmethod
    def robustness_report(scores):
        """Return robustness statistics for a collection of runs."""
        return robustness_analysis(scores)

    @staticmethod
    def profile_run(optimizer, *fit_args, **fit_kwargs):
        """Profile a single optimization run."""
        return profile_optimization_run(optimizer, *fit_args, **fit_kwargs)

    @classmethod
    def default_param_grids(cls):
        """Return mode-specific parameter grids for GridSearchCV."""
        return [
            {"mode": ["aco"], "n_ants": [20, 50], "alpha": [0.5, 1.0, 2.0], "beta": [1.5, 2.0]},
            {"mode": ["pso"], "n_particles": [20, 50], "w": [0.5, 0.9], "c1": [1.5, 2.0], "c2": [1.5, 2.0]},
            {"mode": ["abc"], "n_bees": [20, 50], "limit": [5, 10]},
            {"mode": ["gwo"], "n_wolves": [10, 30], "n_iterations": [50, 100]},
            {"mode": ["fa"], "n_fireflies": [10, 30], "beta0": [0.5, 1.0], "gamma": [0.5, 1.0]},
            {"mode": ["sa"], "initial_temperature": [5.0, 10.0], "cooling_rate": [0.9, 0.95]},
            {"mode": ["cs"], "n_nests": [10, 25], "pa": [0.1, 0.25]},
            {"mode": ["ba"], "n_bats": [10, 30], "fmin": [0.0, 1.0], "fmax": [1.5, 2.0]},
            {"mode": ["gso"], "n_worms": [10, 30], "neighborhood_radius": [0.5, 1.0]},
            {"mode": ["bfo"], "n_bacteria": [10, 30], "n_chemotactic_steps": [5, 10]},
            {"mode": ["de"], "n_individuals": [20, 40], "f": [0.5, 0.8], "cr": [0.7, 0.9]},
        ]

    @classmethod
    def default_param_distributions(cls):
        """Return lightweight parameter distributions for randomized search."""
        return {
            "aco": {"n_ants": [20, 50, 80], "alpha": [0.5, 1.0, 2.0], "beta": [1.5, 2.0, 3.0]},
            "pso": {"n_particles": [20, 30, 50], "w": [0.4, 0.7, 0.9], "c1": [1.0, 1.5, 2.0], "c2": [1.0, 1.5, 2.0]},
            "abc": {"n_bees": [20, 30, 50], "limit": [5, 10, 20]},
            "gwo": {"n_wolves": [10, 20, 30], "n_iterations": [50, 100, 150]},
            "fa": {"n_fireflies": [10, 20, 30], "beta0": [0.5, 1.0], "gamma": [0.5, 1.0], "fa_alpha": [0.1, 0.2, 0.3]},
            "sa": {"initial_temperature": [5.0, 10.0], "cooling_rate": [0.9, 0.95], "step_scale": [0.05, 0.1]},
            "cs": {"n_nests": [10, 25], "pa": [0.1, 0.25], "levy_scale": [0.5, 1.0]},
            "ba": {"n_bats": [10, 20, 30], "fmin": [0.0, 0.5], "fmax": [1.5, 2.0], "bat_alpha": [0.8, 0.9], "bat_gamma": [0.8, 0.9]},
            "gso": {"n_worms": [10, 20, 30], "luciferin_decay": [0.3, 0.4], "luciferin_enhancement": [0.5, 0.6]},
            "bfo": {"n_bacteria": [10, 20, 30], "n_chemotactic_steps": [5, 10], "n_reproduction_steps": [2, 4]},
            "de": {"n_individuals": [20, 30, 40], "f": [0.5, 0.8, 1.0], "cr": [0.6, 0.8, 0.9]},
        }

    @classmethod
    def optimization_cv_strategy(cls, X, y=None, n_splits: int = 5, random_state: Optional[int] = 42):
        """Return a stable CV splitter suited to optimization-oriented searches."""
        if y is not None:
            y_array = np.asarray(y)
            unique_values = np.unique(y_array)
            if y_array.ndim == 1 and 1 < unique_values.size <= max(10, y_array.shape[0] // 2):
                splits = min(n_splits, unique_values.size, y_array.shape[0])
                if splits >= 2:
                    return StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)

        if hasattr(X, "shape"):
            n_samples = int(X.shape[0])
        else:
            n_samples = len(y) if y is not None else len(X)

        splits = min(n_splits, n_samples)
        if splits < 2:
            raise ValueError("Need at least 2 samples to build a cross-validation strategy")
        return KFold(n_splits=splits, shuffle=True, random_state=random_state)

    def transform(self, X):
        """Identity transform for sklearn pipeline compatibility."""
        self._check_is_fitted()
        return validate_data(self, X, reset=False, ensure_2d=True, dtype=None)

    def get_feature_names_out(self, input_features=None):
        """Return output feature names for compose/pipeline compatibility."""
        self._check_is_fitted()

        if input_features is not None:
            return np.asarray(input_features, dtype=object)

        if hasattr(self, "feature_names_in_"):
            return np.asarray(self.feature_names_in_, dtype=object)

        n_features = getattr(self, "n_features_in_", None)
        if n_features is None:
            if self._compatibility_mode and self._best_solution is not None:
                n_features = len(self._best_solution)
            else:
                raise AttributeError("feature names are unavailable before fitting")

        return np.asarray([f"x{i}" for i in range(n_features)], dtype=object)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit and then return an identity transform of ``X``."""
        return self.fit(X, y=y, **fit_params).transform(X)

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = {
            "mode": self.mode,
            "n_iterations": self.n_iterations,
            "random_state": self.random_state,
            "n_ants": self.n_ants,
            "alpha": self.alpha,
            "beta": self.beta,
            "rho": self.rho,
            "q": self.q,
            "n_particles": self.n_particles,
            "w": self.w,
            "c1": self.c1,
            "c2": self.c2,
            "n_bees": self.n_bees,
            "limit": self.limit,
            "n_wolves": self.n_wolves,
            "n_fireflies": self.n_fireflies,
            "beta0": self.beta0,
            "gamma": self.gamma,
            "fa_alpha": self.fa_alpha,
            "initial_temperature": self.initial_temperature,
            "cooling_rate": self.cooling_rate,
            "step_scale": self.step_scale,
            "n_nests": self.n_nests,
            "pa": self.pa,
            "levy_scale": self.levy_scale,
            "n_bats": self.n_bats,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "bat_alpha": self.bat_alpha,
            "bat_gamma": self.bat_gamma,
            "loudness": self.loudness,
            "pulse_rate": self.pulse_rate,
            "n_worms": self.n_worms,
            "luciferin_decay": self.luciferin_decay,
            "luciferin_enhancement": self.luciferin_enhancement,
            "gso_step_size": self.gso_step_size,
            "neighborhood_radius": self.neighborhood_radius,
            "n_bacteria": self.n_bacteria,
            "n_chemotactic_steps": self.n_chemotactic_steps,
            "n_reproduction_steps": self.n_reproduction_steps,
            "elimination_probability": self.elimination_probability,
            "bfo_step_scale": self.bfo_step_scale,
            "use_two_opt": self.use_two_opt,
            "n_individuals": self.n_individuals,
            "f": self.f,
            "cr": self.cr,
        }
        return params

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            if key in {
                "mode",
                "n_iterations",
                "random_state",
                "n_ants",
                "alpha",
                "beta",
                "rho",
                "q",
                "n_particles",
                "w",
                "c1",
                "c2",
                "n_bees",
                "limit",
                "n_wolves",
                "n_fireflies",
                "beta0",
                "gamma",
                "fa_alpha",
                "initial_temperature",
                "cooling_rate",
                "step_scale",
                "n_nests",
                "pa",
                "levy_scale",
                "n_bats",
                "fmin",
                "fmax",
                "bat_alpha",
                "bat_gamma",
                "loudness",
                "pulse_rate",
                "n_worms",
                "luciferin_decay",
                "luciferin_enhancement",
                "gso_step_size",
                "neighborhood_radius",
                "n_bacteria",
                "n_chemotactic_steps",
                "n_reproduction_steps",
                "elimination_probability",
                "bfo_step_scale",
                "use_two_opt",
                "n_individuals",
                "f",
                "cr",
            }:
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_algorithm"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__dict__.setdefault("_fitted", False)
        self.__dict__.setdefault("_compatibility_mode", False)
        self.__dict__.setdefault("_algorithm", None)
        self.__dict__.setdefault("_algorithm_mode", None)
        self.__dict__.setdefault("_best_solution", None)
        self.__dict__.setdefault("_best_score", None)
