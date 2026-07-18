//! Python bindings for the Rust optimization core.
//!
//! Each `#[pyclass]` here is a thin wrapper around a pure-Rust type from
//! `crate::algorithms` / `crate::core`, so the core stays usable without pyo3.

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::algorithms::{
    two_opt, AntColony, BacterialForagingOptimizer, BatAlgorithm, BeeColony, CmaEsOptimizer,
    CuckooSearch, DifferentialEvolution, FireflyOptimizer, GlowwormOptimizer, GreyWolfOptimizer,
    MopsoOptimizer, Nsga2Optimizer, Optimizer, ParticleSwarm, PermutationGeneticOptimizer,
    SimulatedAnnealing,
};
use crate::algorithms::advanced::BinaryParticleSwarm;
use crate::algorithms::aco::AcoVariant;
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

fn make_multi_objective(
    py: Python<'_>,
    func: PyObject,
    probe_point: &[f64],
) -> PyResult<Box<dyn Fn(&[f64]) -> Vec<f64>>> {
    let probe = func.call1(py, (probe_point.to_vec(),))?;
    probe.extract::<Vec<f64>>(py)?;

    Ok(Box::new(move |x: &[f64]| -> Vec<f64> {
        Python::with_gil(|py| match func.call1(py, (x.to_vec(),)) {
            Ok(value) => value.extract::<Vec<f64>>(py).unwrap_or_else(|_| vec![f64::INFINITY]),
            Err(_) => vec![f64::INFINITY],
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
    #[pyo3(get)]
    history_: Vec<f64>,
    #[pyo3(get)]
    population_: Vec<Vec<f64>>,
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
        use_two_opt = true,
        variant = "basic",
        q0 = 0.9,
        elitist_weight = 2.0,
        tau_min = 1e-4,
        tau_max = 10.0,
        random_state = None,
    ))]
    fn new(
        n_ants: usize,
        n_iterations: usize,
        alpha: f64,
        beta: f64,
        rho: f64,
        q: f64,
        use_two_opt: bool,
        variant: &str,
        q0: f64,
        elitist_weight: f64,
        tau_min: f64,
        tau_max: f64,
        random_state: Option<u64>,
    ) -> Self {
        let variant = match variant {
            "acs" => AcoVariant::Acs,
            "elitist" => AcoVariant::Elitist,
            "mmas" => AcoVariant::Mmas,
            _ => AcoVariant::Basic,
        };
        let mut inner = AntColony::new(
            n_ants,
            n_iterations,
            alpha,
            beta,
            rho,
            q,
            use_two_opt,
            variant,
            q0,
            elitist_weight,
            tau_min,
            tau_max,
        );
        inner.set_random_seed(random_state);
        Self {
            inner,
            best_tour: None,
            best_length: None,
            history_: Vec::new(),
            population_: Vec::new(),
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
            self.history_ = self.best_length.into_iter().collect();
            self.population_ = self
                .best_tour
                .clone()
                .map(|tour| vec![tour.into_iter().map(|city| city as f64).collect()])
                .unwrap_or_default();
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
    #[pyo3(get)]
    history_: Vec<f64>,
    #[pyo3(get)]
    population_: Vec<Vec<f64>>,
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
            history_: Vec::new(),
            population_: Vec::new(),
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
            self.history_ = self.best_score.into_iter().collect();
            self.population_ = self
                .best_position
                .clone()
                .map(|position| vec![position])
                .unwrap_or_default();
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
    #[pyo3(get)]
    history_: Vec<f64>,
    #[pyo3(get)]
    population_: Vec<Vec<f64>>,
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
            history_: Vec::new(),
            population_: Vec::new(),
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
            self.history_ = self.best_score.into_iter().collect();
            self.population_ = self
                .best_position
                .clone()
                .map(|position| vec![position])
                .unwrap_or_default();
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

/// Grey Wolf Optimizer for continuous minimization.
#[pyclass(name = "GreyWolfOptimizer")]
pub struct PyGreyWolfOptimizer {
    n_wolves: usize,
    n_iterations: usize,
    random_state: Option<u64>,
    best_position: Option<Vec<f64>>,
    best_score: Option<f64>,
    #[pyo3(get)]
    history_: Vec<f64>,
    #[pyo3(get)]
    population_: Vec<Vec<f64>>,
}

#[pymethods]
impl PyGreyWolfOptimizer {
    #[new]
    #[pyo3(signature = (n_wolves = 30, n_iterations = 100, random_state = None))]
    fn new(n_wolves: usize, n_iterations: usize, random_state: Option<u64>) -> Self {
        Self {
            n_wolves,
            n_iterations,
            random_state,
            best_position: None,
            best_score: None,
            history_: Vec::new(),
            population_: Vec::new(),
        }
    }

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

        let mut optimizer = GreyWolfOptimizer::new(self.n_wolves, self.n_iterations, bounds);
        optimizer.set_random_seed(self.random_state);
        optimizer
            .fit(&problem)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        if let Some(solution) = optimizer.predict() {
            self.best_position = Some(solution.variables);
            self.best_score = solution.fitness;
        }
        self.history_ = optimizer.history;
        self.population_ = optimizer.population;
        Ok(())
    }

    fn predict(&self) -> PyResult<Vec<f64>> {
        self.best_position
            .clone()
            .ok_or_else(|| PyValueError::new_err("must call fit() before predict()"))
    }

    fn score(&self) -> PyResult<f64> {
        self.best_score
            .ok_or_else(|| PyValueError::new_err("must call fit() before score()"))
    }

    fn get_params(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("n_wolves".to_string(), self.n_wolves as f64);
        params.insert("n_iterations".to_string(), self.n_iterations as f64);
        params
    }
}

/// Firefly algorithm for continuous minimization.
#[pyclass(name = "FireflyOptimizer")]
pub struct PyFireflyOptimizer {
    n_fireflies: usize,
    n_iterations: usize,
    beta0: f64,
    gamma: f64,
    alpha: f64,
    random_state: Option<u64>,
    best_position: Option<Vec<f64>>,
    best_score: Option<f64>,
    #[pyo3(get)]
    history_: Vec<f64>,
    #[pyo3(get)]
    population_: Vec<Vec<f64>>,
}

#[pymethods]
impl PyFireflyOptimizer {
    #[new]
    #[pyo3(signature = (
        n_fireflies = 30,
        n_iterations = 100,
        beta0 = 1.0,
        gamma = 1.0,
        alpha = 0.2,
        random_state = None,
    ))]
    fn new(
        n_fireflies: usize,
        n_iterations: usize,
        beta0: f64,
        gamma: f64,
        alpha: f64,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            n_fireflies,
            n_iterations,
            beta0,
            gamma,
            alpha,
            random_state,
            best_position: None,
            best_score: None,
            history_: Vec::new(),
            population_: Vec::new(),
        }
    }

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

        let mut optimizer = FireflyOptimizer::new(
            self.n_fireflies,
            self.n_iterations,
            self.beta0,
            self.gamma,
            self.alpha,
            bounds,
        );
        optimizer.set_random_seed(self.random_state);
        optimizer
            .fit(&problem)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        if let Some(solution) = optimizer.predict() {
            self.best_position = Some(solution.variables);
            self.best_score = solution.fitness;
        }
        self.history_ = optimizer.history;
        self.population_ = optimizer.population;
        Ok(())
    }

    fn predict(&self) -> PyResult<Vec<f64>> {
        self.best_position
            .clone()
            .ok_or_else(|| PyValueError::new_err("must call fit() before predict()"))
    }

    fn score(&self) -> PyResult<f64> {
        self.best_score
            .ok_or_else(|| PyValueError::new_err("must call fit() before score()"))
    }

    fn get_params(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("n_fireflies".to_string(), self.n_fireflies as f64);
        params.insert("n_iterations".to_string(), self.n_iterations as f64);
        params.insert("beta0".to_string(), self.beta0);
        params.insert("gamma".to_string(), self.gamma);
        params.insert("alpha".to_string(), self.alpha);
        params
    }
}

/// Simulated annealing for continuous minimization.
#[pyclass(name = "SimulatedAnnealing")]
pub struct PySimulatedAnnealing {
    initial_temperature: f64,
    cooling_rate: f64,
    step_scale: f64,
    n_iterations: usize,
    random_state: Option<u64>,
    best_position: Option<Vec<f64>>,
    best_score: Option<f64>,
    #[pyo3(get)]
    history_: Vec<f64>,
    #[pyo3(get)]
    population_: Vec<Vec<f64>>,
}

#[pymethods]
impl PySimulatedAnnealing {
    #[new]
    #[pyo3(signature = (
        initial_temperature = 10.0,
        cooling_rate = 0.95,
        step_scale = 0.1,
        n_iterations = 100,
        random_state = None,
    ))]
    fn new(
        initial_temperature: f64,
        cooling_rate: f64,
        step_scale: f64,
        n_iterations: usize,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            initial_temperature,
            cooling_rate,
            step_scale,
            n_iterations,
            random_state,
            best_position: None,
            best_score: None,
            history_: Vec::new(),
            population_: Vec::new(),
        }
    }

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

        let mut optimizer = SimulatedAnnealing::new(
            self.initial_temperature,
            self.cooling_rate,
            self.step_scale,
            self.n_iterations,
            bounds,
        );
        optimizer.set_random_seed(self.random_state);
        optimizer
            .fit(&problem)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        if let Some(solution) = optimizer.predict() {
            self.best_position = Some(solution.variables);
            self.best_score = solution.fitness;
        }
        self.history_ = optimizer.history;
        self.population_ = optimizer.population;
        Ok(())
    }

    fn predict(&self) -> PyResult<Vec<f64>> {
        self.best_position
            .clone()
            .ok_or_else(|| PyValueError::new_err("must call fit() before predict()"))
    }

    fn score(&self) -> PyResult<f64> {
        self.best_score
            .ok_or_else(|| PyValueError::new_err("must call fit() before score()"))
    }

    fn get_params(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("initial_temperature".to_string(), self.initial_temperature);
        params.insert("cooling_rate".to_string(), self.cooling_rate);
        params.insert("step_scale".to_string(), self.step_scale);
        params.insert("n_iterations".to_string(), self.n_iterations as f64);
        params
    }
}

/// Cuckoo search for continuous minimization.
#[pyclass(name = "CuckooSearch")]
pub struct PyCuckooSearch {
    n_nests: usize,
    n_iterations: usize,
    pa: f64,
    alpha: f64,
    levy_scale: f64,
    random_state: Option<u64>,
    best_position: Option<Vec<f64>>,
    best_score: Option<f64>,
    #[pyo3(get)]
    history_: Vec<f64>,
    #[pyo3(get)]
    population_: Vec<Vec<f64>>,
}

#[pymethods]
impl PyCuckooSearch {
    #[new]
    #[pyo3(signature = (
        n_nests = 25,
        n_iterations = 100,
        pa = 0.25,
        alpha = 0.01,
        levy_scale = 1.0,
        random_state = None,
    ))]
    fn new(
        n_nests: usize,
        n_iterations: usize,
        pa: f64,
        alpha: f64,
        levy_scale: f64,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            n_nests,
            n_iterations,
            pa,
            alpha,
            levy_scale,
            random_state,
            best_position: None,
            best_score: None,
            history_: Vec::new(),
            population_: Vec::new(),
        }
    }

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

        let mut optimizer = CuckooSearch::new(
            self.n_nests,
            self.n_iterations,
            self.pa,
            self.alpha,
            self.levy_scale,
            bounds,
        );
        optimizer.set_random_seed(self.random_state);
        optimizer
            .fit(&problem)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        if let Some(solution) = optimizer.predict() {
            self.best_position = Some(solution.variables);
            self.best_score = solution.fitness;
        }
        self.history_ = optimizer.history;
        self.population_ = optimizer.population;
        Ok(())
    }

    fn predict(&self) -> PyResult<Vec<f64>> {
        self.best_position
            .clone()
            .ok_or_else(|| PyValueError::new_err("must call fit() before predict()"))
    }

    fn score(&self) -> PyResult<f64> {
        self.best_score
            .ok_or_else(|| PyValueError::new_err("must call fit() before score()"))
    }

    fn get_params(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("n_nests".to_string(), self.n_nests as f64);
        params.insert("n_iterations".to_string(), self.n_iterations as f64);
        params.insert("pa".to_string(), self.pa);
        params.insert("alpha".to_string(), self.alpha);
        params.insert("levy_scale".to_string(), self.levy_scale);
        params
    }
}

/// Bat algorithm for continuous minimization.
#[pyclass(name = "BatAlgorithm")]
pub struct PyBatAlgorithm {
    n_bats: usize,
    n_iterations: usize,
    fmin: f64,
    fmax: f64,
    alpha: f64,
    gamma: f64,
    loudness: f64,
    pulse_rate: f64,
    random_state: Option<u64>,
    best_position: Option<Vec<f64>>,
    best_score: Option<f64>,
    #[pyo3(get)]
    history_: Vec<f64>,
    #[pyo3(get)]
    population_: Vec<Vec<f64>>,
}

#[pymethods]
impl PyBatAlgorithm {
    #[new]
    #[pyo3(signature = (
        n_bats = 30,
        n_iterations = 100,
        fmin = 0.0,
        fmax = 2.0,
        alpha = 0.9,
        gamma = 0.9,
        loudness = 1.0,
        pulse_rate = 0.5,
        random_state = None,
    ))]
    fn new(
        n_bats: usize,
        n_iterations: usize,
        fmin: f64,
        fmax: f64,
        alpha: f64,
        gamma: f64,
        loudness: f64,
        pulse_rate: f64,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            n_bats,
            n_iterations,
            fmin,
            fmax,
            alpha,
            gamma,
            loudness,
            pulse_rate,
            random_state,
            best_position: None,
            best_score: None,
            history_: Vec::new(),
            population_: Vec::new(),
        }
    }

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

        let mut optimizer = BatAlgorithm::new(
            self.n_bats,
            self.n_iterations,
            self.fmin,
            self.fmax,
            self.alpha,
            self.gamma,
            self.loudness,
            self.pulse_rate,
            bounds,
        );
        optimizer.set_random_seed(self.random_state);
        optimizer
            .fit(&problem)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        if let Some(solution) = optimizer.predict() {
            self.best_position = Some(solution.variables);
            self.best_score = solution.fitness;
        }
        self.history_ = optimizer.history;
        self.population_ = optimizer.population;
        Ok(())
    }

    fn predict(&self) -> PyResult<Vec<f64>> {
        self.best_position
            .clone()
            .ok_or_else(|| PyValueError::new_err("must call fit() before predict()"))
    }

    fn score(&self) -> PyResult<f64> {
        self.best_score
            .ok_or_else(|| PyValueError::new_err("must call fit() before score()"))
    }

    fn get_params(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("n_bats".to_string(), self.n_bats as f64);
        params.insert("n_iterations".to_string(), self.n_iterations as f64);
        params.insert("fmin".to_string(), self.fmin);
        params.insert("fmax".to_string(), self.fmax);
        params.insert("alpha".to_string(), self.alpha);
        params.insert("gamma".to_string(), self.gamma);
        params.insert("loudness".to_string(), self.loudness);
        params.insert("pulse_rate".to_string(), self.pulse_rate);
        params
    }
}

/// Glowworm swarm optimizer for continuous minimization.
#[pyclass(name = "GlowwormOptimizer")]
pub struct PyGlowwormOptimizer {
    n_worms: usize,
    n_iterations: usize,
    luciferin_decay: f64,
    luciferin_enhancement: f64,
    step_size: f64,
    neighborhood_radius: f64,
    random_state: Option<u64>,
    best_position: Option<Vec<f64>>,
    best_score: Option<f64>,
    #[pyo3(get)]
    history_: Vec<f64>,
    #[pyo3(get)]
    population_: Vec<Vec<f64>>,
}

#[pymethods]
impl PyGlowwormOptimizer {
    #[new]
    #[pyo3(signature = (
        n_worms = 30,
        n_iterations = 100,
        luciferin_decay = 0.4,
        luciferin_enhancement = 0.6,
        step_size = 0.1,
        neighborhood_radius = 1.0,
        random_state = None,
    ))]
    fn new(
        n_worms: usize,
        n_iterations: usize,
        luciferin_decay: f64,
        luciferin_enhancement: f64,
        step_size: f64,
        neighborhood_radius: f64,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            n_worms,
            n_iterations,
            luciferin_decay,
            luciferin_enhancement,
            step_size,
            neighborhood_radius,
            random_state,
            best_position: None,
            best_score: None,
            history_: Vec::new(),
            population_: Vec::new(),
        }
    }

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

        let mut optimizer = GlowwormOptimizer::new(
            self.n_worms,
            self.n_iterations,
            self.luciferin_decay,
            self.luciferin_enhancement,
            self.step_size,
            self.neighborhood_radius,
            bounds,
        );
        optimizer.set_random_seed(self.random_state);
        optimizer
            .fit(&problem)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        if let Some(solution) = optimizer.predict() {
            self.best_position = Some(solution.variables);
            self.best_score = solution.fitness;
        }
        self.history_ = optimizer.history;
        self.population_ = optimizer.population;
        Ok(())
    }

    fn predict(&self) -> PyResult<Vec<f64>> {
        self.best_position
            .clone()
            .ok_or_else(|| PyValueError::new_err("must call fit() before predict()"))
    }

    fn score(&self) -> PyResult<f64> {
        self.best_score
            .ok_or_else(|| PyValueError::new_err("must call fit() before score()"))
    }

    fn get_params(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("n_worms".to_string(), self.n_worms as f64);
        params.insert("n_iterations".to_string(), self.n_iterations as f64);
        params.insert("luciferin_decay".to_string(), self.luciferin_decay);
        params.insert(
            "luciferin_enhancement".to_string(),
            self.luciferin_enhancement,
        );
        params.insert("step_size".to_string(), self.step_size);
        params.insert("neighborhood_radius".to_string(), self.neighborhood_radius);
        params
    }
}

/// Bacterial foraging optimizer for continuous minimization.
#[pyclass(name = "BacterialForagingOptimizer")]
pub struct PyBacterialForagingOptimizer {
    n_bacteria: usize,
    n_iterations: usize,
    n_chemotactic_steps: usize,
    n_reproduction_steps: usize,
    elimination_probability: f64,
    step_scale: f64,
    random_state: Option<u64>,
    best_position: Option<Vec<f64>>,
    best_score: Option<f64>,
    #[pyo3(get)]
    history_: Vec<f64>,
    #[pyo3(get)]
    population_: Vec<Vec<f64>>,
}

#[pymethods]
impl PyBacterialForagingOptimizer {
    #[new]
    #[pyo3(signature = (
        n_bacteria = 30,
        n_iterations = 100,
        n_chemotactic_steps = 10,
        n_reproduction_steps = 4,
        elimination_probability = 0.25,
        step_scale = 0.1,
        random_state = None,
    ))]
    fn new(
        n_bacteria: usize,
        n_iterations: usize,
        n_chemotactic_steps: usize,
        n_reproduction_steps: usize,
        elimination_probability: f64,
        step_scale: f64,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            n_bacteria,
            n_iterations,
            n_chemotactic_steps,
            n_reproduction_steps,
            elimination_probability,
            step_scale,
            random_state,
            best_position: None,
            best_score: None,
            history_: Vec::new(),
            population_: Vec::new(),
        }
    }

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

        let mut optimizer = BacterialForagingOptimizer::new(
            self.n_bacteria,
            self.n_iterations,
            self.n_chemotactic_steps,
            self.n_reproduction_steps,
            self.elimination_probability,
            self.step_scale,
            bounds,
        );
        optimizer.set_random_seed(self.random_state);
        optimizer
            .fit(&problem)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        if let Some(solution) = optimizer.predict() {
            self.best_position = Some(solution.variables);
            self.best_score = solution.fitness;
        }
        self.history_ = optimizer.history;
        self.population_ = optimizer.population;
        Ok(())
    }

    fn predict(&self) -> PyResult<Vec<f64>> {
        self.best_position
            .clone()
            .ok_or_else(|| PyValueError::new_err("must call fit() before predict()"))
    }

    fn score(&self) -> PyResult<f64> {
        self.best_score
            .ok_or_else(|| PyValueError::new_err("must call fit() before score()"))
    }

    fn get_params(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("n_bacteria".to_string(), self.n_bacteria as f64);
        params.insert("n_iterations".to_string(), self.n_iterations as f64);
        params.insert(
            "n_chemotactic_steps".to_string(),
            self.n_chemotactic_steps as f64,
        );
        params.insert(
            "n_reproduction_steps".to_string(),
            self.n_reproduction_steps as f64,
        );
        params.insert(
            "elimination_probability".to_string(),
            self.elimination_probability,
        );
        params.insert("step_scale".to_string(), self.step_scale);
        params
    }
}

/// Differential evolution for continuous minimization.
#[pyclass(name = "DifferentialEvolution")]
pub struct PyDifferentialEvolution {
    n_individuals: usize,
    n_iterations: usize,
    f: f64,
    cr: f64,
    random_state: Option<u64>,
    best_position: Option<Vec<f64>>,
    best_score: Option<f64>,
    #[pyo3(get)]
    history_: Vec<f64>,
    #[pyo3(get)]
    population_: Vec<Vec<f64>>,
}

#[pymethods]
impl PyDifferentialEvolution {
    #[new]
    #[pyo3(signature = (
        n_individuals = 40,
        n_iterations = 100,
        f = 0.8,
        cr = 0.9,
        random_state = None,
    ))]
    fn new(
        n_individuals: usize,
        n_iterations: usize,
        f: f64,
        cr: f64,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            n_individuals,
            n_iterations,
            f,
            cr,
            random_state,
            best_position: None,
            best_score: None,
            history_: Vec::new(),
            population_: Vec::new(),
        }
    }

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

        let mut optimizer = DifferentialEvolution::new(
            self.n_individuals,
            self.n_iterations,
            self.f,
            self.cr,
            bounds,
        );
        optimizer.set_random_seed(self.random_state);
        optimizer
            .fit(&problem)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        if let Some(solution) = optimizer.predict() {
            self.best_position = Some(solution.variables);
            self.best_score = solution.fitness;
        }
        self.history_ = optimizer.history;
        self.population_ = optimizer.population;
        Ok(())
    }

    fn predict(&self) -> PyResult<Vec<f64>> {
        self.best_position
            .clone()
            .ok_or_else(|| PyValueError::new_err("must call fit() before predict()"))
    }

    fn score(&self) -> PyResult<f64> {
        self.best_score
            .ok_or_else(|| PyValueError::new_err("must call fit() before score()"))
    }

    fn get_params(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("n_individuals".to_string(), self.n_individuals as f64);
        params.insert("n_iterations".to_string(), self.n_iterations as f64);
        params.insert("f".to_string(), self.f);
        params.insert("cr".to_string(), self.cr);
        params
    }
}

/// CMA-ES-style optimizer for continuous minimization.
#[pyclass(name = "CmaEsOptimizer")]
pub struct PyCmaEsOptimizer {
    n_individuals: usize,
    n_iterations: usize,
    sigma: f64,
    random_state: Option<u64>,
    best_position: Option<Vec<f64>>,
    best_score: Option<f64>,
    #[pyo3(get)]
    history_: Vec<f64>,
    #[pyo3(get)]
    population_: Vec<Vec<f64>>,
}

#[pymethods]
impl PyCmaEsOptimizer {
    #[new]
    #[pyo3(signature = (
        n_individuals = 20,
        n_iterations = 100,
        sigma = 0.5,
        random_state = None,
    ))]
    fn new(
        n_individuals: usize,
        n_iterations: usize,
        sigma: f64,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            n_individuals,
            n_iterations,
            sigma,
            random_state,
            best_position: None,
            best_score: None,
            history_: Vec::new(),
            population_: Vec::new(),
        }
    }

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

        let mut optimizer =
            CmaEsOptimizer::new(self.n_individuals, self.n_iterations, self.sigma, bounds);
        optimizer.set_random_seed(self.random_state);
        optimizer
            .fit(&problem)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        if let Some(solution) = optimizer.predict() {
            self.best_position = Some(solution.variables);
            self.best_score = solution.fitness;
        }
        self.history_ = optimizer.history;
        self.population_ = optimizer.population;
        Ok(())
    }

    fn predict(&self) -> PyResult<Vec<f64>> {
        self.best_position
            .clone()
            .ok_or_else(|| PyValueError::new_err("must call fit() before predict()"))
    }

    fn score(&self) -> PyResult<f64> {
        self.best_score
            .ok_or_else(|| PyValueError::new_err("must call fit() before score()"))
    }

    fn get_params(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("n_individuals".to_string(), self.n_individuals as f64);
        params.insert("n_iterations".to_string(), self.n_iterations as f64);
        params.insert("sigma".to_string(), self.sigma);
        params
    }
}

/// Permutation genetic optimizer for TSP-like discrete problems.
#[pyclass(name = "PermutationGeneticOptimizer")]
pub struct PyPermutationGeneticOptimizer {
    n_individuals: usize,
    n_iterations: usize,
    mutation_rate: f64,
    use_two_opt: bool,
    random_state: Option<u64>,
    best_tour: Option<Vec<usize>>,
    best_length: Option<f64>,
    #[pyo3(get)]
    history_: Vec<f64>,
    #[pyo3(get)]
    population_: Vec<Vec<f64>>,
}

/// Binary PSO for bit-vector objectives.
#[pyclass(name = "BinaryParticleSwarm")]
pub struct PyBinaryParticleSwarm {
    n_particles: usize,
    n_iterations: usize,
    w: f64,
    c1: f64,
    c2: f64,
    random_state: Option<u64>,
    best_position: Option<Vec<f64>>,
    best_score: Option<f64>,
    #[pyo3(get)]
    history_: Vec<f64>,
    #[pyo3(get)]
    population_: Vec<Vec<f64>>,
}

#[pymethods]
impl PyBinaryParticleSwarm {
    #[new]
    #[pyo3(signature = (
        n_particles = 30,
        n_iterations = 100,
        w = 0.7,
        c1 = 1.5,
        c2 = 1.5,
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
            history_: Vec::new(),
            population_: Vec::new(),
        }
    }

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
        let objective_function = make_objective(py, objective, &vec![0.0; dimensions])?;

        let mut optimizer = BinaryParticleSwarm::new(
            self.n_particles,
            self.n_iterations,
            self.w,
            self.c1,
            self.c2,
        );
        optimizer.set_random_seed(self.random_state);
        optimizer
            .fit_with_objective(&objective_function, dimensions)
            .map_err(|e: crate::algorithms::base::OptimizationError| PyValueError::new_err(e.to_string()))?;

        if let Some(solution) = optimizer.best_solution {
            self.best_position = Some(solution.variables);
            self.best_score = solution.fitness;
        }
        self.history_ = optimizer.history;
        self.population_ = optimizer.population;
        Ok(())
    }

    fn predict(&self) -> PyResult<Vec<f64>> {
        self.best_position
            .clone()
            .ok_or_else(|| PyValueError::new_err("must call fit() before predict()"))
    }

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
}

#[pymethods]
impl PyPermutationGeneticOptimizer {
    #[new]
    #[pyo3(signature = (
        n_individuals = 40,
        n_iterations = 100,
        mutation_rate = 0.1,
        use_two_opt = true,
        random_state = None,
    ))]
    fn new(
        n_individuals: usize,
        n_iterations: usize,
        mutation_rate: f64,
        use_two_opt: bool,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            n_individuals,
            n_iterations,
            mutation_rate,
            use_two_opt,
            random_state,
            best_tour: None,
            best_length: None,
            history_: Vec::new(),
            population_: Vec::new(),
        }
    }

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

        let mut optimizer = PermutationGeneticOptimizer::new(
            self.n_individuals,
            self.n_iterations,
            self.mutation_rate,
            self.use_two_opt,
        );
        optimizer.set_random_seed(self.random_state);
        optimizer
            .fit(&problem)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        if let Some(solution) = optimizer.predict() {
            self.best_tour = Some(solution.variables.iter().map(|value| *value as usize).collect());
            self.best_length = solution.fitness;
            self.history_ = optimizer.history;
            self.population_ = optimizer.population;
        }
        Ok(())
    }

    fn predict(&self) -> PyResult<Vec<usize>> {
        self.best_tour
            .clone()
            .ok_or_else(|| PyValueError::new_err("must call fit() before predict()"))
    }

    fn score(&self) -> PyResult<f64> {
        self.best_length
            .ok_or_else(|| PyValueError::new_err("must call fit() before score()"))
    }

    fn get_params(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("n_individuals".to_string(), self.n_individuals as f64);
        params.insert("n_iterations".to_string(), self.n_iterations as f64);
        params.insert("mutation_rate".to_string(), self.mutation_rate);
        params.insert("use_two_opt".to_string(), if self.use_two_opt { 1.0 } else { 0.0 });
        params
    }
}

/// NSGA-II-style multi-objective optimizer.
#[pyclass(name = "Nsga2Optimizer")]
pub struct PyNsga2Optimizer {
    n_individuals: usize,
    n_iterations: usize,
    crossover_rate: f64,
    mutation_rate: f64,
    mutation_scale: f64,
    archive_size: usize,
    random_state: Option<u64>,
    best_front: Vec<Vec<f64>>,
    best_objectives: Vec<Vec<f64>>,
    #[pyo3(get)]
    population_: Vec<Vec<f64>>,
}

#[pymethods]
impl PyNsga2Optimizer {
    #[new]
    #[pyo3(signature = (
        n_individuals = 40,
        n_iterations = 100,
        crossover_rate = 0.9,
        mutation_rate = 0.1,
        mutation_scale = 0.1,
        archive_size = 50,
        random_state = None,
    ))]
    fn new(
        n_individuals: usize,
        n_iterations: usize,
        crossover_rate: f64,
        mutation_rate: f64,
        mutation_scale: f64,
        archive_size: usize,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            n_individuals,
            n_iterations,
            crossover_rate,
            mutation_rate,
            mutation_scale,
            archive_size,
            random_state,
            best_front: Vec::new(),
            best_objectives: Vec::new(),
            population_: Vec::new(),
        }
    }

    #[pyo3(signature = (objective, lower, upper))]
    fn fit(
        &mut self,
        py: Python<'_>,
        objective: PyObject,
        lower: Vec<f64>,
        upper: Vec<f64>,
    ) -> PyResult<()> {
        let bounds = build_bounds(lower, upper)?;
        let objective_function = make_multi_objective(py, objective, &bounds.midpoint())?;
        let mut optimizer = Nsga2Optimizer::new(
            self.n_individuals,
            self.n_iterations,
            self.crossover_rate,
            self.mutation_rate,
            self.mutation_scale,
            self.archive_size,
            bounds,
        );
        optimizer.set_random_seed(self.random_state);
        optimizer
            .fit_with_objective(&objective_function)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        self.population_ = optimizer.population;
        self.best_front = optimizer
            .best_front
            .iter()
            .map(|point| point.variables.clone())
            .collect();
        self.best_objectives = optimizer
            .best_front
            .iter()
            .map(|point| point.objectives.clone())
            .collect();
        Ok(())
    }

    fn predict(&self) -> PyResult<Vec<Vec<f64>>> {
        if self.best_front.is_empty() {
            return Err(PyValueError::new_err("must call fit() before predict()"));
        }
        Ok(self.best_front.clone())
    }

    fn score(&self) -> PyResult<f64> {
        if self.best_objectives.is_empty() {
            return Err(PyValueError::new_err("must call fit() before score()"));
        }
        if self.best_objectives[0].len() >= 2 {
            let points = self
                .best_front
                .iter()
                .zip(self.best_objectives.iter())
                .map(|(variables, objectives)| crate::algorithms::advanced::ParetoPoint {
                    variables: variables.clone(),
                    objectives: objectives.clone(),
                })
                .collect::<Vec<_>>();
            Ok(crate::algorithms::advanced::hypervolume_2d(&points, [1.0, 1.0]))
        } else {
            Ok(0.0)
        }
    }

    fn get_params(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("n_individuals".to_string(), self.n_individuals as f64);
        params.insert("n_iterations".to_string(), self.n_iterations as f64);
        params.insert("crossover_rate".to_string(), self.crossover_rate);
        params.insert("mutation_rate".to_string(), self.mutation_rate);
        params.insert("mutation_scale".to_string(), self.mutation_scale);
        params.insert("archive_size".to_string(), self.archive_size as f64);
        params
    }
}

/// Multi-objective PSO.
#[pyclass(name = "MopsoOptimizer")]
pub struct PyMopsoOptimizer {
    n_particles: usize,
    n_iterations: usize,
    w: f64,
    c1: f64,
    c2: f64,
    mutation_scale: f64,
    archive_size: usize,
    random_state: Option<u64>,
    best_front: Vec<Vec<f64>>,
    best_objectives: Vec<Vec<f64>>,
    #[pyo3(get)]
    population_: Vec<Vec<f64>>,
}

#[pymethods]
impl PyMopsoOptimizer {
    #[new]
    #[pyo3(signature = (
        n_particles = 30,
        n_iterations = 100,
        w = 0.7,
        c1 = 1.5,
        c2 = 1.5,
        mutation_scale = 0.1,
        archive_size = 50,
        random_state = None,
    ))]
    fn new(
        n_particles: usize,
        n_iterations: usize,
        w: f64,
        c1: f64,
        c2: f64,
        mutation_scale: f64,
        archive_size: usize,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            n_particles,
            n_iterations,
            w,
            c1,
            c2,
            mutation_scale,
            archive_size,
            random_state,
            best_front: Vec::new(),
            best_objectives: Vec::new(),
            population_: Vec::new(),
        }
    }

    #[pyo3(signature = (objective, lower, upper))]
    fn fit(
        &mut self,
        py: Python<'_>,
        objective: PyObject,
        lower: Vec<f64>,
        upper: Vec<f64>,
    ) -> PyResult<()> {
        let bounds = build_bounds(lower, upper)?;
        let objective_function = make_multi_objective(py, objective, &bounds.midpoint())?;
        let mut optimizer = MopsoOptimizer::new(
            self.n_particles,
            self.n_iterations,
            self.w,
            self.c1,
            self.c2,
            self.mutation_scale,
            self.archive_size,
            bounds,
        );
        optimizer.set_random_seed(self.random_state);
        optimizer
            .fit_with_objective(&objective_function)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        self.population_ = optimizer.population;
        self.best_front = optimizer
            .archive
            .iter()
            .map(|point| point.variables.clone())
            .collect();
        self.best_objectives = optimizer
            .archive
            .iter()
            .map(|point| point.objectives.clone())
            .collect();
        Ok(())
    }

    fn predict(&self) -> PyResult<Vec<Vec<f64>>> {
        if self.best_front.is_empty() {
            return Err(PyValueError::new_err("must call fit() before predict()"));
        }
        Ok(self.best_front.clone())
    }

    fn score(&self) -> PyResult<f64> {
        if self.best_objectives.is_empty() {
            return Err(PyValueError::new_err("must call fit() before score()"));
        }
        if self.best_objectives[0].len() >= 2 {
            let points = self
                .best_front
                .iter()
                .zip(self.best_objectives.iter())
                .map(|(variables, objectives)| crate::algorithms::advanced::ParetoPoint {
                    variables: variables.clone(),
                    objectives: objectives.clone(),
                })
                .collect::<Vec<_>>();
            Ok(crate::algorithms::advanced::hypervolume_2d(&points, [1.0, 1.0]))
        } else {
            Ok(0.0)
        }
    }

    fn get_params(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("n_particles".to_string(), self.n_particles as f64);
        params.insert("n_iterations".to_string(), self.n_iterations as f64);
        params.insert("w".to_string(), self.w);
        params.insert("c1".to_string(), self.c1);
        params.insert("c2".to_string(), self.c2);
        params.insert("mutation_scale".to_string(), self.mutation_scale);
        params.insert("archive_size".to_string(), self.archive_size as f64);
        params
    }
}

#[pyfunction(name = "two_opt")]
pub fn two_opt_py(tour: Vec<usize>, distance_matrix: Vec<Vec<f64>>) -> PyResult<(Vec<usize>, f64)> {
    two_opt(&tour, &distance_matrix).map_err(PyValueError::new_err)
}
