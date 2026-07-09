"""
AutoColony: Main interface for swarm intelligence optimization algorithms
"""

from typing import Optional, Union, Dict, Any, Callable
import numpy as np
from sklearn.base import BaseEstimator


class AutoColony(BaseEstimator):
    """
    Unified interface for swarm intelligence optimization algorithms.
    
    Similar to HuggingFace's AutoModel, this class provides a single interface
    for multiple optimization algorithms selected via the 'mode' parameter.
    
    Parameters
    ----------
    mode : str, default='auto'
        Algorithm selection mode:
        - 'auto': Automatically select algorithm based on problem type
        - 'aco': Ant Colony Optimization
        - 'pso': Particle Swarm Optimization  
        - 'abc': Artificial Bee Colony
        
    n_iterations : int, default=100
        Number of iterations to run
        
    random_state : int, default=None
        Random seed for reproducibility
        
    **kwargs : dict
        Algorithm-specific parameters
    """
    
    def __init__(
        self,
        mode: str = 'auto',
        n_iterations: int = 100,
        random_state: Optional[int] = None,
        **kwargs
    ):
        self.mode = mode
        self.n_iterations = n_iterations
        self.random_state = random_state
        
        # Algorithm-specific parameters
        self.kwargs = kwargs
        
        # Internal state
        self._fitted = False
        self._best_solution = None
        self._best_score = None
        self._algorithm = None
        
        # Validate mode
        valid_modes = ['auto', 'aco', 'pso', 'abc']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}")
    
    def _detect_problem_type(self, X, y=None):
        """Auto-detect problem type for algorithm selection"""
        if y is not None:
            # Supervised learning problem - use PSO
            return 'pso'
        elif hasattr(X, 'shape') and len(X.shape) == 2:
            # Distance matrix (TSP-like) - use ACO
            if X.shape[0] == X.shape[1]:
                return 'aco'
            else:
                return 'pso'
        else:
            # Default to PSO for continuous problems
            return 'pso'
    
    def _create_algorithm(self, algorithm_mode: str):
        """Create the appropriate algorithm instance"""
        
        if algorithm_mode == 'aco':
            from ._colonyx import AntColony
            params = self._filter_params('aco')
            return AntColony(
                n_ants=int(params['n_ants']),
                n_iterations=int(params['n_iterations']),
                alpha=float(params['alpha']),
                beta=float(params['beta']),
                rho=float(params['rho']),
                q=float(params['q']),
                random_state=params['random_state'],
            )
            
        elif algorithm_mode == 'pso':
            from ._colonyx import ParticleSwarm
            params = self._filter_params('pso')
            return ParticleSwarm(
                n_particles=int(params['n_particles']),
                n_iterations=int(params['n_iterations']),
                w=float(params['w']),
                c1=float(params['c1']),
                c2=float(params['c2']),
                random_state=params['random_state'],
            )

        elif algorithm_mode == 'abc':
            from ._colonyx import BeeColony
            params = self._filter_params('abc')
            return BeeColony(
                n_bees=int(params['n_bees']),
                n_iterations=int(params['n_iterations']),
                limit=int(params['limit']),
                random_state=params['random_state'],
            )

        else:
            raise ValueError(f"Unknown algorithm mode: {algorithm_mode}")
    
    def _filter_params(self, algorithm_mode: str) -> Dict[str, Any]:
        """Filter parameters relevant to the specific algorithm"""
        base_params = {
            'n_iterations': self.n_iterations,
            'random_state': self.random_state,
        }
        
        if algorithm_mode == 'aco':
            aco_params = {
                'n_ants': self.kwargs.get('n_ants', 50),
                'alpha': self.kwargs.get('alpha', 1.0),
                'beta': self.kwargs.get('beta', 2.0),
                'rho': self.kwargs.get('rho', 0.5),
                'q': self.kwargs.get('q', 1.0),
            }
            return {**base_params, **aco_params}
            
        elif algorithm_mode == 'pso':
            pso_params = {
                'n_particles': self.kwargs.get('n_particles', 30),
                'w': self.kwargs.get('w', 0.9),
                'c1': self.kwargs.get('c1', 2.0),
                'c2': self.kwargs.get('c2', 2.0),
            }
            return {**base_params, **pso_params}
            
        elif algorithm_mode == 'abc':
            abc_params = {
                'n_bees': self.kwargs.get('n_bees', 50),
                'limit': self.kwargs.get('limit', 10),
            }
            return {**base_params, **abc_params}

        return base_params

    def _as_distance_matrix(self, X):
        """Validate and convert X into a square distance matrix (list of lists)."""
        arr = np.asarray(X, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(
                "ACO (mode='aco') expects a square distance matrix; "
                f"got array of shape {arr.shape}"
            )
        return arr.tolist()

    def _split_bounds(self, bounds):
        """Split a sequence of (low, high) pairs into lower/upper lists."""
        arr = np.asarray(bounds, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                "bounds must be a sequence of (low, high) pairs, "
                f"e.g. [(-5, 5), (-5, 5)]; got shape {arr.shape}"
            )
        lower = arr[:, 0].tolist()
        upper = arr[:, 1].tolist()
        if any(lo > hi for lo, hi in zip(lower, upper)):
            raise ValueError("each bound must have low <= high")
        return lower, upper

    def fit(self, X, y=None, bounds=None):
        """
        Fit the optimizer to the problem

        Parameters
        ----------
        X : array-like or callable
            Problem data: a square distance matrix for ACO, or an objective
            function ``f(list[float]) -> float`` to minimize for PSO/ABC.
        y : array-like, optional
            Target values for supervised problems
        bounds : sequence of (low, high), optional
            Per-dimension search-space bounds. Required for PSO and ABC.

        Returns
        -------
        self : AutoColony
            Returns self for method chaining
        """
        # Determine algorithm mode
        if self.mode == 'auto':
            algorithm_mode = self._detect_problem_type(X, y)
        else:
            algorithm_mode = self.mode

        # Create algorithm instance
        self._algorithm = self._create_algorithm(algorithm_mode)
        self._algorithm_mode = algorithm_mode

        # Fit the algorithm on the appropriately-shaped input
        if algorithm_mode == 'aco':
            distance_matrix = self._as_distance_matrix(X)
            self._algorithm.fit(distance_matrix)
        elif algorithm_mode in ('pso', 'abc'):
            if not callable(X):
                raise ValueError(
                    f"mode='{algorithm_mode}' expects an objective function "
                    "(callable) as X"
                )
            if bounds is None:
                raise ValueError(
                    f"mode='{algorithm_mode}' requires bounds=[(low, high), ...]"
                )
            lower, upper = self._split_bounds(bounds)
            self._algorithm.fit(X, lower, upper)
        else:
            raise NotImplementedError(
                f"Fitting for mode '{algorithm_mode}' is not yet implemented"
            )

        self._best_solution = self._algorithm.predict()
        self._best_score = self._algorithm.score()
        self._fitted = True

        return self
    
    def predict(self):
        """
        Get the best solution found
        
        Returns
        -------
        solution : array-like
            Best solution found by the algorithm
        """
        if not self._fitted:
            raise ValueError("Must call fit() before predict()")
        
        # TODO: Return actual best solution
        return self._best_solution
    
    def score(self, X=None, y=None):
        """
        Get the best score/fitness value
        
        Returns
        -------
        score : float
            Best score found by the algorithm
        """
        if not self._fitted:
            raise ValueError("Must call fit() before score()")
        
        # TODO: Return actual best score
        return self._best_score
    
    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        params = {
            'mode': self.mode,
            'n_iterations': self.n_iterations,
            'random_state': self.random_state,
        }
        params.update(self.kwargs)
        return params
    
    def set_params(self, **params):
        """Set parameters for this estimator"""
        valid_params = self.get_params()
        
        for key, value in params.items():
            if key in ['mode', 'n_iterations', 'random_state']:
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        
        return self 