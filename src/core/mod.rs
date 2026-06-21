pub mod problem;
pub mod solution;
pub mod bounds;

// Re-export main types
pub use problem::{Problem, ContinuousProblem, DiscreteProblem};
pub use solution::{Solution, SolutionSet};
pub use bounds::{Bounds, BoundConstraint}; 