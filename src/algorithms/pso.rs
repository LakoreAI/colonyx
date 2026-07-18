use crate::algorithms::base::{make_rng, OptimizationError, Optimizer};
use crate::core::{Bounds, Problem, Solution};
use rand::Rng;
use std::collections::HashMap;

/// Initial velocities are drawn from +/- this fraction of each dimension's range.
const INIT_VELOCITY_SCALE: f64 = 0.1;

/// Particle Swarm Optimization for continuous minimization problems.
///
/// Bounds define the search space (and its dimensionality); the objective is
/// supplied through the `Problem` passed to `fit`.
#[derive(Debug)]
pub struct ParticleSwarm {
    pub n_particles: usize,
    pub n_iterations: usize,
    pub w: f64,  // Inertia weight
    pub c1: f64, // Cognitive (personal-best) coefficient
    pub c2: f64, // Social (global-best) coefficient

    bounds: Bounds,
    random_seed: Option<u64>,
    best_solution: Option<Solution>,
}

impl ParticleSwarm {
    pub fn new(
        n_particles: usize,
        n_iterations: usize,
        w: f64,
        c1: f64,
        c2: f64,
        bounds: Bounds,
    ) -> Self {
        Self {
            n_particles,
            n_iterations,
            w,
            c1,
            c2,
            bounds,
            random_seed: None,
            best_solution: None,
        }
    }

    /// Set the RNG seed for reproducible runs. `None` draws from entropy.
    pub fn set_random_seed(&mut self, seed: Option<u64>) {
        self.random_seed = seed;
    }
}

impl Optimizer for ParticleSwarm {
    type Solution = Solution;

    fn fit(&mut self, problem: &dyn Problem) -> Result<(), OptimizationError> {
        let dim = self.bounds.lower.len();
        if dim == 0 {
            return Err(OptimizationError::InvalidInput(
                "Bounds must have at least one dimension".to_string(),
            ));
        }
        if problem.dimensions() != dim {
            return Err(OptimizationError::DimensionMismatch(format!(
                "Problem has {} dimensions but bounds have {}",
                problem.dimensions(),
                dim
            )));
        }

        let mut rng = make_rng(self.random_seed);

        let ranges = self.bounds.ranges();

        let mut positions = vec![vec![0.0; dim]; self.n_particles];
        let mut velocities = vec![vec![0.0; dim]; self.n_particles];
        let mut personal_best = vec![vec![0.0; dim]; self.n_particles];
        let mut personal_best_fitness = vec![f64::INFINITY; self.n_particles];

        let mut global_best = vec![0.0; dim];
        let mut global_best_fitness = f64::INFINITY;

        // Initialize the swarm uniformly at random inside the bounds.
        for i in 0..self.n_particles {
            for d in 0..dim {
                positions[i][d] = self.bounds.lower[d] + rng.gen::<f64>() * ranges[d];
                velocities[i][d] = (rng.gen::<f64>() * 2.0 - 1.0) * ranges[d] * INIT_VELOCITY_SCALE;
            }

            let fitness = problem.evaluate(&positions[i]);
            personal_best[i].copy_from_slice(&positions[i]);
            personal_best_fitness[i] = fitness;

            if fitness < global_best_fitness {
                global_best_fitness = fitness;
                global_best.copy_from_slice(&positions[i]);
            }
        }

        // Main PSO loop.
        for _ in 0..self.n_iterations {
            for i in 0..self.n_particles {
                for d in 0..dim {
                    let r1 = rng.gen::<f64>();
                    let r2 = rng.gen::<f64>();
                    velocities[i][d] = self.w * velocities[i][d]
                        + self.c1 * r1 * (personal_best[i][d] - positions[i][d])
                        + self.c2 * r2 * (global_best[d] - positions[i][d]);
                    positions[i][d] += velocities[i][d];
                }

                self.bounds.clamp(&mut positions[i]);

                let fitness = problem.evaluate(&positions[i]);
                if fitness < personal_best_fitness[i] {
                    personal_best_fitness[i] = fitness;
                    personal_best[i].copy_from_slice(&positions[i]);

                    if fitness < global_best_fitness {
                        global_best_fitness = fitness;
                        global_best.copy_from_slice(&positions[i]);
                    }
                }
            }
        }

        self.best_solution = Some(Solution::with_fitness(global_best, global_best_fitness));
        Ok(())
    }

    fn predict(&self) -> Option<Self::Solution> {
        self.best_solution.clone()
    }

    fn score(&self) -> Option<f64> {
        self.best_solution.as_ref().and_then(|s| s.fitness)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ContinuousProblem;

    /// Sphere function: minimized at the origin with value 0.
    fn sphere(dim: usize) -> ContinuousProblem {
        ContinuousProblem {
            name: "sphere".to_string(),
            dimensions: dim,
            objective_function: Box::new(|x: &[f64]| x.iter().map(|&xi| xi * xi).sum()),
        }
    }

    #[test]
    fn minimizes_sphere_near_origin() {
        let bounds = Bounds::uniform(3, -5.0, 5.0).unwrap();
        let mut pso = ParticleSwarm::new(30, 100, 0.7, 1.5, 1.5, bounds);
        pso.set_random_seed(Some(42));
        pso.fit(&sphere(3)).unwrap();

        let score = pso.score().unwrap();
        assert!(score < 1e-3, "expected near-zero minimum, got {score}");
    }

    #[test]
    fn solution_stays_within_bounds() {
        let bounds = Bounds::uniform(2, -2.0, 2.0).unwrap();
        let mut pso = ParticleSwarm::new(20, 50, 0.7, 1.5, 1.5, bounds.clone());
        pso.set_random_seed(Some(1));
        pso.fit(&sphere(2)).unwrap();

        let solution = pso.predict().unwrap();
        assert!(bounds.contains(&solution.variables));
    }

    #[test]
    fn reproducible_with_seed() {
        let run = || {
            let bounds = Bounds::uniform(2, -5.0, 5.0).unwrap();
            let mut pso = ParticleSwarm::new(15, 40, 0.7, 1.5, 1.5, bounds);
            pso.set_random_seed(Some(7));
            pso.fit(&sphere(2)).unwrap();
            pso.score().unwrap()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn rejects_dimension_mismatch() {
        let bounds = Bounds::uniform(3, -1.0, 1.0).unwrap();
        let mut pso = ParticleSwarm::new(10, 10, 0.7, 1.5, 1.5, bounds);
        // Problem reports 2 dims, bounds have 3.
        assert!(pso.fit(&sphere(2)).is_err());
    }
}
