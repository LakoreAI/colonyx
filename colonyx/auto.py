"""
AutoColony: Main interface for swarm intelligence optimization algorithms
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils._tags import InputTags, Tags, TargetTags, TransformerTags
from sklearn.utils.validation import validate_data

from .base import BaseOptimizer
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

# Single source of truth for every algorithm-specific parameter that
# ``AutoColony`` exposes. Each entry maps the unified (frontend) attribute
# name to a ``(default, backend_kwarg_name, cast)`` tuple, where
# ``backend_kwarg_name`` is the keyword argument accepted by the
# corresponding Rust pyo3 class in ``._colonyx`` and ``cast`` converts the
# stored attribute to the type that constructor expects. This dict drives
# ``get_params``, ``set_params``, ``_filter_params``, ``parameter_mapping``,
# ``parameter_help``, and ``_create_algorithm`` so that adding or renaming a
# parameter only requires touching this one place instead of eight.
_ALGORITHM_PARAM_SPECS: Dict[str, Dict[str, tuple]] = {
    "aco": {
        "n_ants": (50, "n_ants", int),
        "alpha": (1.0, "alpha", float),
        "beta": (2.0, "beta", float),
        "rho": (0.5, "rho", float),
        "q": (1.0, "q", float),
        "use_two_opt": (True, "use_two_opt", bool),
    },
    "pso": {
        "n_particles": (30, "n_particles", int),
        "w": (0.9, "w", float),
        "c1": (2.0, "c1", float),
        "c2": (2.0, "c2", float),
    },
    "abc": {
        "n_bees": (50, "n_bees", int),
        "limit": (10, "limit", int),
    },
    "gwo": {
        "n_wolves": (30, "n_wolves", int),
    },
    "fa": {
        "n_fireflies": (30, "n_fireflies", int),
        "beta0": (1.0, "beta0", float),
        "gamma": (1.0, "gamma", float),
        "fa_alpha": (0.2, "alpha", float),
    },
    "sa": {
        "initial_temperature": (10.0, "initial_temperature", float),
        "cooling_rate": (0.95, "cooling_rate", float),
        "step_scale": (0.1, "step_scale", float),
    },
    "cs": {
        "n_nests": (25, "n_nests", int),
        "pa": (0.25, "pa", float),
        "cs_alpha": (0.01, "alpha", float),
        "levy_scale": (1.0, "levy_scale", float),
    },
    "ba": {
        "n_bats": (30, "n_bats", int),
        "fmin": (0.0, "fmin", float),
        "fmax": (2.0, "fmax", float),
        "bat_alpha": (0.9, "alpha", float),
        "bat_gamma": (0.9, "gamma", float),
        "loudness": (1.0, "loudness", float),
        "pulse_rate": (0.5, "pulse_rate", float),
    },
    "gso": {
        "n_worms": (30, "n_worms", int),
        "luciferin_decay": (0.4, "luciferin_decay", float),
        "luciferin_enhancement": (0.6, "luciferin_enhancement", float),
        "gso_step_size": (0.1, "step_size", float),
        "neighborhood_radius": (1.0, "neighborhood_radius", float),
    },
    "bfo": {
        "n_bacteria": (30, "n_bacteria", int),
        "n_chemotactic_steps": (10, "n_chemotactic_steps", int),
        "n_reproduction_steps": (4, "n_reproduction_steps", int),
        "elimination_probability": (0.25, "elimination_probability", float),
        "bfo_step_scale": (0.1, "step_scale", float),
    },
    "de": {
        "n_individuals": (40, "n_individuals", int),
        "f": (0.8, "f", float),
        "cr": (0.9, "cr", float),
    },
    "cmaes": {
        "n_individuals": (40, "n_individuals", int),
        "cmaes_sigma": (0.5, "sigma", float),
    },
}

# Backend pyo3 class (in ``._colonyx``) for each mode, keyed the same way as
# ``_ALGORITHM_PARAM_SPECS``.
_BACKEND_CLASSES: Dict[str, str] = {
    "aco": "AntColony",
    "pso": "ParticleSwarm",
    "abc": "BeeColony",
    "gwo": "GreyWolfOptimizer",
    "fa": "FireflyOptimizer",
    "sa": "SimulatedAnnealing",
    "cs": "CuckooSearch",
    "ba": "BatAlgorithm",
    "gso": "GlowwormOptimizer",
    "bfo": "BacterialForagingOptimizer",
    "de": "DifferentialEvolution",
    "cmaes": "CmaEsOptimizer",
}

# Flattened attribute defaults, used only to keep AutoColony.__init__'s
# keyword defaults in sync with the registry above. __init__ itself must
# stay an explicit, introspectable, **kwargs-free signature that assigns
# every argument to an identically named attribute — that is a hard
# requirement of the scikit-learn estimator contract (clone()/get_params()
# rely on it), so it cannot itself be generated from a loop.
_DEFAULTS: Dict[str, Any] = {
    attr: spec[0] for specs in _ALGORITHM_PARAM_SPECS.values() for attr, spec in specs.items()
}

# All algorithm-specific attribute names across every mode, used by
# set_params() to validate incoming keys.
_ALL_PARAM_NAMES = {"mode", "n_iterations", "random_state"} | {
    attr for specs in _ALGORITHM_PARAM_SPECS.values() for attr in specs
}


class AutoColony(BaseOptimizer, TransformerMixin):
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
        n_ants: int = _DEFAULTS["n_ants"],
        alpha: float = _DEFAULTS["alpha"],
        beta: float = _DEFAULTS["beta"],
        rho: float = _DEFAULTS["rho"],
        q: float = _DEFAULTS["q"],
        n_particles: int = _DEFAULTS["n_particles"],
        w: float = _DEFAULTS["w"],
        c1: float = _DEFAULTS["c1"],
        c2: float = _DEFAULTS["c2"],
        n_bees: int = _DEFAULTS["n_bees"],
        limit: int = _DEFAULTS["limit"],
        n_wolves: int = _DEFAULTS["n_wolves"],
        n_fireflies: int = _DEFAULTS["n_fireflies"],
        beta0: float = _DEFAULTS["beta0"],
        gamma: float = _DEFAULTS["gamma"],
        fa_alpha: float = _DEFAULTS["fa_alpha"],
        initial_temperature: float = _DEFAULTS["initial_temperature"],
        cooling_rate: float = _DEFAULTS["cooling_rate"],
        step_scale: float = _DEFAULTS["step_scale"],
        n_nests: int = _DEFAULTS["n_nests"],
        pa: float = _DEFAULTS["pa"],
        cs_alpha: float = _DEFAULTS["cs_alpha"],
        levy_scale: float = _DEFAULTS["levy_scale"],
        n_bats: int = _DEFAULTS["n_bats"],
        fmin: float = _DEFAULTS["fmin"],
        fmax: float = _DEFAULTS["fmax"],
        bat_alpha: float = _DEFAULTS["bat_alpha"],
        bat_gamma: float = _DEFAULTS["bat_gamma"],
        loudness: float = _DEFAULTS["loudness"],
        pulse_rate: float = _DEFAULTS["pulse_rate"],
        n_worms: int = _DEFAULTS["n_worms"],
        luciferin_decay: float = _DEFAULTS["luciferin_decay"],
        luciferin_enhancement: float = _DEFAULTS["luciferin_enhancement"],
        gso_step_size: float = _DEFAULTS["gso_step_size"],
        neighborhood_radius: float = _DEFAULTS["neighborhood_radius"],
        n_bacteria: int = _DEFAULTS["n_bacteria"],
        n_chemotactic_steps: int = _DEFAULTS["n_chemotactic_steps"],
        n_reproduction_steps: int = _DEFAULTS["n_reproduction_steps"],
        elimination_probability: float = _DEFAULTS["elimination_probability"],
        bfo_step_scale: float = _DEFAULTS["bfo_step_scale"],
        use_two_opt: bool = _DEFAULTS["use_two_opt"],
        n_individuals: int = _DEFAULTS["n_individuals"],
        f: float = _DEFAULTS["f"],
        cr: float = _DEFAULTS["cr"],
        cmaes_sigma: float = _DEFAULTS["cmaes_sigma"],
    ):
        super().__init__(mode=mode, n_iterations=n_iterations, random_state=random_state)
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
        self.cs_alpha = cs_alpha
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
        specs = _ALGORITHM_PARAM_SPECS.get(mode)
        if specs is None:
            return {}
        mapping = {"n_iterations": "n_iterations", "random_state": "random_state"}
        mapping.update({attr: spec[1] for attr, spec in specs.items()})
        return mapping

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
                "cs_alpha": 0.01,
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
        if mode == "auto":
            return "Auto mode selects a backend from the input shape and bounds"
        specs = _ALGORITHM_PARAM_SPECS.get(mode)
        if specs is None:
            return "Unknown mode"
        label = "CMA-ES" if mode == "cmaes" else mode.upper()
        return f"{label} params: " + ", ".join(specs)

    def _filter_params(self, algorithm_mode: str) -> Dict[str, Any]:
        """Filter parameters relevant to the specific algorithm."""
        base_params = {
            "n_iterations": self.n_iterations,
            "random_state": self.random_state,
        }
        specs = _ALGORITHM_PARAM_SPECS.get(algorithm_mode)
        if specs is None:
            return base_params
        return {**base_params, **{attr: getattr(self, attr) for attr in specs}}

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
        """Create the appropriate algorithm instance from the parameter registry."""
        specs = _ALGORITHM_PARAM_SPECS.get(algorithm_mode)
        backend_name = _BACKEND_CLASSES.get(algorithm_mode)
        if specs is None or backend_name is None:
            raise ValueError(f"Unknown algorithm mode: {algorithm_mode}")

        from . import _colonyx

        backend_cls = getattr(_colonyx, backend_name)
        params = self.resolve_parameter_conflicts(algorithm_mode)

        kwargs: Dict[str, Any] = {
            "n_iterations": int(params["n_iterations"]),
            "random_state": params["random_state"],
        }
        for attr, (_default, backend_kwarg, cast) in specs.items():
            kwargs[backend_kwarg] = cast(params[attr])

        return backend_cls(**kwargs)

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
        self._compat_fit_targets = targets
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
        estimator behaves like a standard sklearn scorer (higher is better),
        defaulting to the targets seen during ``fit()`` when ``y`` is omitted.
        """
        self._check_is_fitted()

        if not self._compatibility_mode:
            return self._best_score

        if y is None:
            y = getattr(self, "_compat_fit_targets", None)
            if y is None:
                return 0.0

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
            {"mode": ["cs"], "n_nests": [10, 25], "pa": [0.1, 0.25], "cs_alpha": [0.005, 0.01, 0.05]},
            {"mode": ["ba"], "n_bats": [10, 30], "fmin": [0.0, 1.0], "fmax": [1.5, 2.0]},
            {"mode": ["gso"], "n_worms": [10, 30], "neighborhood_radius": [0.5, 1.0]},
            {"mode": ["bfo"], "n_bacteria": [10, 30], "n_chemotactic_steps": [5, 10]},
            {"mode": ["de"], "n_individuals": [20, 40], "f": [0.5, 0.8], "cr": [0.7, 0.9]},
            {"mode": ["cmaes"], "n_individuals": [10, 20], "cmaes_sigma": [0.3, 0.5]},
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
            "cs": {"n_nests": [10, 25], "pa": [0.1, 0.25], "cs_alpha": [0.005, 0.01, 0.05], "levy_scale": [0.5, 1.0]},
            "ba": {"n_bats": [10, 20, 30], "fmin": [0.0, 0.5], "fmax": [1.5, 2.0], "bat_alpha": [0.8, 0.9], "bat_gamma": [0.8, 0.9]},
            "gso": {"n_worms": [10, 20, 30], "luciferin_decay": [0.3, 0.4], "luciferin_enhancement": [0.5, 0.6]},
            "bfo": {"n_bacteria": [10, 20, 30], "n_chemotactic_steps": [5, 10], "n_reproduction_steps": [2, 4]},
            "de": {"n_individuals": [20, 30, 40], "f": [0.5, 0.8, 1.0], "cr": [0.6, 0.8, 0.9]},
            "cmaes": {"n_individuals": [10, 20, 30], "cmaes_sigma": [0.3, 0.5, 0.8]},
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
        params: Dict[str, Any] = {
            "mode": self.mode,
            "n_iterations": self.n_iterations,
            "random_state": self.random_state,
        }
        for name in _ALL_PARAM_NAMES - set(params):
            params[name] = getattr(self, name)
        return params

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            if key in _ALL_PARAM_NAMES:
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
