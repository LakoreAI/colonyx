//! Python bindings for the Rust optimization core.
//!
//! Each `#[pyclass]` here is a thin wrapper around a pure-Rust type from
//! `crate::algorithms` / `crate::core`, so the core stays usable without pyo3.

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::algorithms::{AntColony, BeeColony, Optimizer, ParticleSwarm};
use crate::core::{Bounds, ContinuousProblem, DiscreteProblem};

/// Build `Bounds` from Python lower/upper lists, surfacing errors as `ValueError`.
fn build_bounds(lower: Vec<f64>, upper: Vec<f64>) -> PyResult<Bounds> {
    Bounds::new(lower, upper).map_err(PyValueError::new_err)
}

/// Wrap a Python callable `f(list[float]) -> float` as a Rust objective closure.
///
/// The callable is validated once at `probe_point` so a broken objective raises a
/// clear Python exception up front; during the search any per-call error maps to
/// `+inf` (an infeasible point for minimization) rather than unwinding across FFI.
fn make_objective(
    py: Python<'_>,
    func: PyObject,
    probe_point: &[f64],
) -> PyResult<Box<dyn Fn(&[f64]) -> f64>> {
    let probe = func.call1(py, (probe_point.to_vec(),))?;
    probe.extract::<f64>(py)?;

    Ok(Box::new(move |x: &[f64]| -> f64 {
        Python::with_gil(|py| match func.call1(py, (x.to_vec(),)) {
            Ok(value) => value.extract::<f64>(py).unwrap_or(f64::INFINITY),
            Err(_) => f64::INFINITY,
        })
    }))
}

/// Ant Colony Optimization for discrete problems (e.g. TSP).
///
/// Exposed to Python as `colonyx._colonyx.AntColony`.
#[pyclass(name = "AntColony")]
pub struct PyAntColony {
    inner: AntColony,
    best_tour: Option<Vec<usize>>,
    best_length: Option<f64>,
}

#[pymethods]
impl PyAntColony {
    #[new]
    #[pyo3(signature = (
        n_ants = 50,
        n_iterations = 100,
        alpha = 1.0,
        beta = 2.0,
        rho = 0.5,
        q = 1.0,
        random_state = None,
    ))]
    fn new(
        n_ants: usize,
        n_iterations: usize,
        alpha: f64,
        beta: f64,
        rho: f64,
        q: f64,
        random_state: Option<u64>,
    ) -> Self {
        let mut inner = AntColony::new(n_ants, n_iterations, alpha, beta, rho, q);
        inner.set_random_seed(random_state);
        Self {
            inner,
            best_tour: None,
            best_length: None,
        }
    }

    /// Fit the colony to a square distance matrix and search for a short tour.
    fn fit(&mut self, distance_matrix: Vec<Vec<f64>>) -> PyResult<()> {
        let n = distance_matrix.len();
        if n == 0 {
            return Err(PyValueError::new_err("distance_matrix must be non-empty"));
        }
        for (i, row) in distance_matrix.iter().enumerate() {
            if row.len() != n {
                return Err(PyValueError::new_err(format!(
                    "distance_matrix must be square: row {} has length {}, expected {}",
                    i,
                    row.len(),
                    n
                )));
            }
        }

        let problem = DiscreteProblem {
            name: "tsp".to_string(),
            distance_matrix,
        };

        self.inner
            .fit(&problem)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        if let Some(solution) = self.inner.predict() {
            self.best_tour = Some(solution.variables.iter().map(|&x| x as usize).collect());
            self.best_length = solution.fitness;
        }

        Ok(())
    }

    /// Return the best tour found as a list of city indices.
    fn predict(&self) -> PyResult<Vec<usize>> {
        self.best_tour
            .clone()
            .ok_or_else(|| PyValueError::new_err("must call fit() before predict()"))
    }

    /// Return the length of the best tour found.
    fn score(&self) -> PyResult<f64> {
        self.best_length
            .ok_or_else(|| PyValueError::new_err("must call fit() before score()"))
    }

    /// Return the algorithm's hyperparameters.
    fn get_params(&self) -> HashMap<String, f64> {
        self.inner.get_params()
    }

    fn __repr__(&self) -> String {
        format!(
            "AntColony(n_ants={}, n_iterations={}, alpha={}, beta={}, rho={}, q={})",
            self.inner.n_ants,
            self.inner.n_iterations,
            self.inner.alpha,
            self.inner.beta,
            self.inner.rho,
            self.inner.q,
        )
    }
}

/// Particle Swarm Optimization for continuous minimization.
///
/// Exposed to Python as `colonyx._colonyx.ParticleSwarm`.
#[pyclass(name = "ParticleSwarm")]
pub struct PyParticleSwarm {
    n_particles: usize,
    n_iterations: usize,
    w: f64,
    c1: f64,
    c2: f64,
    random_state: Option<u64>,
    best_position: Option<Vec<f64>>,
    best_score: Option<f64>,
}

#[pymethods]
impl PyParticleSwarm {
    #[new]
    #[pyo3(signature = (
        n_particles = 30,
        n_iterations = 100,
        w = 0.9,
        c1 = 2.0,
        c2 = 2.0,
        random_state = None,
    ))]
    fn new(
        n_particles: usize,
        n_iterations: usize,
        w: f64,
        c1: f64,
        c2: f64,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            n_particles,
            n_iterations,
            w,
            c1,
            c2,
            random_state,
            best_position: None,
            best_score: None,
        }
    }

    /// Minimize `objective(list[float]) -> float` over the box `[lower, upper]`.
    #[pyo3(signature = (objective, lower, upper))]
    fn fit(
        &mut self,
        py: Python<'_>,
        objective: PyObject,
        lower: Vec<f64>,
        upper: Vec<f64>,
    ) -> PyResult<()> {
        let bounds = build_bounds(lower, upper)?;
        let dimensions = bounds.lower.len();
        let objective_function = make_objective(py, objective, &bounds.midpoint())?;

        let problem = ContinuousProblem {
            name: "objective".to_string(),
            dimensions,
            objective_function,
        };

        let mut pso = ParticleSwarm::new(
            self.n_particles,
            self.n_iterations,
            self.w,
            self.c1,
            self.c2,
            bounds,
        );
        pso.set_random_seed(self.random_state);
        pso.fit(&problem)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        if let Some(solution) = pso.predict() {
            self.best_position = Some(solution.variables);
            self.best_score = solution.fitness;
        }

        Ok(())
    }

    /// Return the best position found.
    fn predict(&self) -> PyResult<Vec<f64>> {
        self.best_position
            .clone()
            .ok_or_else(|| PyValueError::new_err("must call fit() before predict()"))
    }

    /// Return the objective value at the best position.
    fn score(&self) -> PyResult<f64> {
        self.best_score
            .ok_or_else(|| PyValueError::new_err("must call fit() before score()"))
    }

    fn get_params(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("n_particles".to_string(), self.n_particles as f64);
        params.insert("n_iterations".to_string(), self.n_iterations as f64);
        params.insert("w".to_string(), self.w);
        params.insert("c1".to_string(), self.c1);
        params.insert("c2".to_string(), self.c2);
        params
    }

    fn __repr__(&self) -> String {
        format!(
            "ParticleSwarm(n_particles={}, n_iterations={}, w={}, c1={}, c2={})",
            self.n_particles, self.n_iterations, self.w, self.c1, self.c2,
        )
    }
}

/// Artificial Bee Colony for continuous minimization.
///
/// Exposed to Python as `colonyx._colonyx.BeeColony`.
#[pyclass(name = "BeeColony")]
pub struct PyBeeColony {
    n_bees: usize,
    n_iterations: usize,
    limit: usize,
    random_state: Option<u64>,
    best_position: Option<Vec<f64>>,
    best_score: Option<f64>,
}

#[pymethods]
impl PyBeeColony {
    #[new]
    #[pyo3(signature = (
        n_bees = 50,
        n_iterations = 100,
        limit = 10,
        random_state = None,
    ))]
    fn new(n_bees: usize, n_iterations: usize, limit: usize, random_state: Option<u64>) -> Self {
        Self {
            n_bees,
            n_iterations,
            limit,
            random_state,
            best_position: None,
            best_score: None,
        }
    }

    /// Minimize `objective(list[float]) -> float` over the box `[lower, upper]`.
    #[pyo3(signature = (objective, lower, upper))]
    fn fit(
        &mut self,
        py: Python<'_>,
        objective: PyObject,
        lower: Vec<f64>,
        upper: Vec<f64>,
    ) -> PyResult<()> {
        let bounds = build_bounds(lower, upper)?;
        let dimensions = bounds.lower.len();
        let objective_function = make_objective(py, objective, &bounds.midpoint())?;

        let problem = ContinuousProblem {
            name: "objective".to_string(),
            dimensions,
            objective_function,
        };

        let mut abc = BeeColony::new(self.n_bees, self.n_iterations, self.limit, bounds);
        abc.set_random_seed(self.random_state);
        abc.fit(&problem)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        if let Some(solution) = abc.predict() {
            self.best_position = Some(solution.variables);
            self.best_score = solution.fitness;
        }

        Ok(())
    }

    /// Return the best position found.
    fn predict(&self) -> PyResult<Vec<f64>> {
        self.best_position
            .clone()
            .ok_or_else(|| PyValueError::new_err("must call fit() before predict()"))
    }

    /// Return the objective value at the best position.
    fn score(&self) -> PyResult<f64> {
        self.best_score
            .ok_or_else(|| PyValueError::new_err("must call fit() before score()"))
    }

    fn get_params(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("n_bees".to_string(), self.n_bees as f64);
        params.insert("n_iterations".to_string(), self.n_iterations as f64);
        params.insert("limit".to_string(), self.limit as f64);
        params
    }

    fn __repr__(&self) -> String {
        format!(
            "BeeColony(n_bees={}, n_iterations={}, limit={})",
            self.n_bees, self.n_iterations, self.limit,
        )
    }
}
