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

pub use crate::core::Problem;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Create a RNG from an optional seed. `None` draws from entropy.
pub fn make_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    }
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
