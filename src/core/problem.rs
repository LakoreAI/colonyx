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
        // Calculate tour length
        let mut total = 0.0;
        for i in 0..solution.len() {
            let from = solution[i] as usize;
            let to = solution[(i + 1) % solution.len()] as usize;
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