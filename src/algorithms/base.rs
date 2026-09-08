/// Base trait for all optimization algorithms
pub trait Optimizer {
    type Solution;

    /// Fit the optimizer to the problem
    fn fit(&mut self, problem: &dyn Problem) -> Result<(), OptimizationError>;

    /// Get the best solution found
    fn predict(&self) -> Option<Self::Solution>;

    /// Get the best score/fitness
    fn score(&self) -> Option<f64>;

    /// Get algorithm-specific parameters
    fn get_params(&self) -> std::collections::HashMap<String, f64>;
}

use crate::algorithms::advanced::ParetoPoint;
use crate::core::MultiObjectiveProblem;
pub use crate::core::Problem;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Base trait for optimizers that search for a Pareto front rather than a
/// single best solution (NSGA-II, MOPSO). Mirrors `Optimizer`, but takes a
/// `MultiObjectiveProblem` and exposes the resulting archive/front instead
/// of a single `predict()`/`score()` pair, since neither concept is
/// meaningful for a set of mutually non-dominated solutions.
pub trait MultiObjectiveOptimizer {
    /// Fit the optimizer to a multi-objective problem.
    fn fit(&mut self, problem: &dyn MultiObjectiveProblem) -> Result<(), OptimizationError>;

    /// The non-dominated solutions found (the Pareto front/archive).
    fn pareto_front(&self) -> Vec<ParetoPoint>;
}

/// Create a RNG from an optional seed. `None` draws from entropy.
pub fn make_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    }
}

/// Evaluate every candidate in `population` against `problem`, using a
/// thread pool (rayon) when there is more than one candidate.
///
/// This is safe to substitute for a sequential `.iter().map(evaluate)` only
/// where each candidate's fitness is independent of every other candidate's
/// fitness within the same batch (e.g. initializing a population, or a
/// generational algorithm like CMA-ES that samples a whole generation before
/// selecting on it) — it must not be used where one candidate's evaluation
/// result is consumed to construct or accept another candidate later in the
/// same batch (e.g. DE's or ABC's steady-state per-individual loops), since
/// that would change the algorithm's actual behavior, not just its speed.
///
/// For a Python-supplied objective, each evaluation still acquires the GIL
/// internally (see `bindings::make_objective`), so parallel evaluation only
/// speeds things up when that Python callable itself releases the GIL while
/// it runs (e.g. it spends its time inside NumPy/SciPy/C-extension code) —
/// a pure-Python objective will simply serialize on the GIL again. It is a
/// real win for objectives implemented directly as Rust `Problem`s.
pub fn evaluate_population(problem: &dyn Problem, population: &[Vec<f64>]) -> Vec<f64> {
    use rayon::prelude::*;
    if population.len() < 2 {
        return population.iter().map(|candidate| problem.evaluate(candidate)).collect();
    }
    population.par_iter().map(|candidate| problem.evaluate(candidate)).collect()
}

/// Error types for optimization
#[derive(Debug, Clone)]
pub enum OptimizationError {
    InvalidInput(String),
    ConvergenceError(String),
    DimensionMismatch(String),
}

impl std::fmt::Display for OptimizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizationError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            OptimizationError::ConvergenceError(msg) => write!(f, "Convergence error: {}", msg),
            OptimizationError::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {}", msg),
        }
    }
}

impl std::error::Error for OptimizationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ContinuousProblem;

    fn sphere() -> ContinuousProblem {
        ContinuousProblem {
            name: "sphere".to_string(),
            dimensions: 2,
            objective_function: Box::new(|x: &[f64]| x.iter().map(|&xi| xi * xi).sum()),
        }
    }

    #[test]
    fn evaluate_population_preserves_order_and_matches_sequential_evaluate() {
        let problem = sphere();
        let population =
            vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 0.0], vec![0.0, 3.0], vec![-1.0, -1.0]];
        let parallel: Vec<f64> = evaluate_population(&problem, &population);
        let sequential: Vec<f64> =
            population.iter().map(|candidate| problem.evaluate(candidate)).collect();
        assert_eq!(parallel, sequential);
    }

    #[test]
    fn evaluate_population_handles_empty_and_singleton_batches() {
        let problem = sphere();
        assert_eq!(evaluate_population(&problem, &[]), Vec::<f64>::new());
        assert_eq!(evaluate_population(&problem, &[vec![3.0, 4.0]]), vec![25.0]);
    }
}
