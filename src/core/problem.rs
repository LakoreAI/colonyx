/// Generic problem trait
pub trait Problem {
    fn evaluate(&self, solution: &[f64]) -> f64;
    fn dimensions(&self) -> usize;
    fn is_discrete(&self) -> bool;
    fn name(&self) -> &str;

    /// Distance matrix for discrete (graph) problems, used by construction
    /// heuristics such as ACO. Continuous problems return `None`.
    fn distance_matrix(&self) -> Option<&[Vec<f64>]> {
        None
    }
}

/// Continuous problem (PSO, ABC)
pub struct ContinuousProblem {
    pub name: String,
    pub dimensions: usize,
    pub objective_function: Box<dyn Fn(&[f64]) -> f64>,
}

impl ContinuousProblem {
    pub fn sphere(dimensions: usize, _bounds: Option<crate::core::Bounds>) -> Self {
        Self {
            name: "Sphere".to_string(),
            dimensions,
            objective_function: Box::new(|x: &[f64]| x.iter().map(|&xi| xi * xi).sum()),
        }
    }
}

impl Problem for ContinuousProblem {
    fn evaluate(&self, solution: &[f64]) -> f64 {
        (self.objective_function)(solution)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn is_discrete(&self) -> bool {
        false
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Discrete problem (ACO, TSP)
pub struct DiscreteProblem {
    pub name: String,
    pub distance_matrix: Vec<Vec<f64>>,
}

impl Problem for DiscreteProblem {
    fn evaluate(&self, solution: &[f64]) -> f64 {
        let mut total = 0.0;
        let n = self.distance_matrix.len();
        for i in 0..solution.len() {
            let from = solution[i] as usize;
            let to = solution[(i + 1) % solution.len()] as usize;
            if from >= n || to >= n {
                return f64::INFINITY;
            }
            total += self.distance_matrix[from][to];
        }
        total
    }

    fn dimensions(&self) -> usize {
        self.distance_matrix.len()
    }

    fn is_discrete(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn distance_matrix(&self) -> Option<&[Vec<f64>]> {
        Some(&self.distance_matrix)
    }
}

#[cfg(test)]
mod tests {
    use super::{ContinuousProblem, DiscreteProblem, Problem};

    #[test]
    fn continuous_problem_evaluates_sphere() {
        let problem = ContinuousProblem::sphere(3, None);
        assert_eq!(problem.evaluate(&[1.0, 2.0, 3.0]), 14.0);
        assert!(!problem.is_discrete());
        assert_eq!(problem.dimensions(), 3);
    }

    #[test]
    fn discrete_problem_evaluates_tour_length() {
        let problem = DiscreteProblem {
            name: "toy".to_string(),
            distance_matrix: vec![
                vec![0.0, 1.0, 2.0],
                vec![1.0, 0.0, 3.0],
                vec![2.0, 3.0, 0.0],
            ],
        };
        assert_eq!(problem.evaluate(&[0.0, 1.0, 2.0]), 6.0);
        assert!(problem.is_discrete());
        assert_eq!(problem.distance_matrix().unwrap().len(), 3);
    }
}
