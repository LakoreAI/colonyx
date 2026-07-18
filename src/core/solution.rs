/// A solution to an optimization problem
#[derive(Debug, Clone)]
pub struct Solution {
    pub variables: Vec<f64>,
    pub fitness: Option<f64>,
    pub is_feasible: bool,
    pub metadata: std::collections::HashMap<String, String>,
}

impl Solution {
    pub fn new(variables: Vec<f64>) -> Self {
        Self {
            variables,
            fitness: None,
            is_feasible: true,
            metadata: std::collections::HashMap::new(),
        }
    }

    pub fn with_fitness(variables: Vec<f64>, fitness: f64) -> Self {
        Self {
            variables,
            fitness: Some(fitness),
            is_feasible: true,
            metadata: std::collections::HashMap::new(),
        }
    }

    pub fn set_fitness(&mut self, fitness: f64) {
        self.fitness = Some(fitness);
    }

    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    pub fn dimensions(&self) -> usize {
        self.variables.len()
    }
}

/// A set of solutions (population)
#[derive(Debug, Clone)]
pub struct SolutionSet {
    pub solutions: Vec<Solution>,
    pub best_index: Option<usize>,
    pub generation: usize,
}

impl SolutionSet {
    pub fn new(solutions: Vec<Solution>) -> Self {
        Self {
            solutions,
            best_index: None,
            generation: 0,
        }
    }

    pub fn find_best(&mut self) -> Option<&Solution> {
        if self.solutions.is_empty() {
            return None;
        }

        let mut best_idx = 0;
        let mut best_fitness = f64::INFINITY;
        let mut found = false;

        for (i, solution) in self.solutions.iter().enumerate() {
            if let Some(fitness) = solution.fitness {
                if !found || fitness < best_fitness {
                    best_fitness = fitness;
                    best_idx = i;
                    found = true;
                }
            }
        }

        if !found {
            self.best_index = None;
            return None;
        }

        self.best_index = Some(best_idx);
        Some(&self.solutions[best_idx])
    }

    pub fn get_best(&self) -> Option<&Solution> {
        let idx = self.best_index?;
        if idx < self.solutions.len() {
            Some(&self.solutions[idx])
        } else {
            None
        }
    }

    pub fn size(&self) -> usize {
        self.solutions.len()
    }

    pub fn push(&mut self, solution: Solution) {
        self.solutions.push(solution);
    }

    pub fn next_generation(&mut self) {
        self.generation += 1;
        self.best_index = None; // Reset best index for new generation
    }
}
