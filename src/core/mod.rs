pub mod bounds;
pub mod problem;
pub mod solution;

// Re-export main types
pub use bounds::Bounds;
pub use problem::{ContinuousProblem, DiscreteProblem, Problem};
pub use solution::{Solution, SolutionSet};
