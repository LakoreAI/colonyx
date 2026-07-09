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

// Use the Problem trait from core module
pub use crate::core::Problem;

/// Error types for optimization
#[derive(Debug)]
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
