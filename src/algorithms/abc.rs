use crate::algorithms::base::{make_rng, OptimizationError, Optimizer};
use crate::core::{Bounds, Problem, Solution};
use rand::rngs::StdRng;
use rand::Rng;
use std::collections::HashMap;

/// Artificial Bee Colony for continuous minimization problems.
///
/// The colony maintains `n_bees / 2` food sources. Each cycle runs an employed
/// phase (one candidate per source), an onlooker phase (candidates biased
/// toward better sources), and a scout phase (abandon a source that has not
/// improved for `limit` trials).
#[derive(Debug)]
pub struct BeeColony {
    pub n_bees: usize,
    pub n_iterations: usize,
    pub limit: usize,

    bounds: Bounds,
    random_seed: Option<u64>,
    best_solution: Option<Solution>,
}

impl BeeColony {
    pub fn new(n_bees: usize, n_iterations: usize, limit: usize, bounds: Bounds) -> Self {
        Self {
            n_bees,
            n_iterations,
            limit,
            bounds,
            random_seed: None,
            best_solution: None,
        }
    }

    /// Set the RNG seed for reproducible runs. `None` draws from entropy.
    pub fn set_random_seed(&mut self, seed: Option<u64>) {
        self.random_seed = seed;
    }

    /// Draw a random position uniformly inside the bounds.
    fn random_source(&self, ranges: &[f64], rng: &mut StdRng) -> Vec<f64> {
        (0..self.bounds.lower.len())
            .map(|d| self.bounds.lower[d] + rng.gen::<f64>() * ranges[d])
            .collect()
    }

    /// Generate a neighbour of source `i`, evaluate it, and greedily keep it if
    /// it improves on the current source (resetting that source's trial count).
    fn exploit_source(
        &self,
        sources: &mut [Vec<f64>],
        objective: &mut [f64],
        trials: &mut [usize],
        i: usize,
        problem: &dyn Problem,
        rng: &mut StdRng,
    ) {
        let sn = sources.len();
        let dim = self.bounds.lower.len();

        let mut candidate = sources[i].clone();
        if sn == 1 {
            // Single source: random walk instead of differential mutation.
            let j = rng.gen_range(0..dim);
            let step = (rng.gen::<f64>() * 2.0 - 1.0) * (self.bounds.upper[j] - self.bounds.lower[j]) * 0.1;
            candidate[j] = sources[i][j] + step;
        } else {
            // Partner source k != i.
            let mut k = rng.gen_range(0..sn);
            while k == i {
                k = rng.gen_range(0..sn);
            }
            let j = rng.gen_range(0..dim);
            let phi = rng.gen::<f64>() * 2.0 - 1.0;
            candidate[j] = sources[i][j] + phi * (sources[i][j] - sources[k][j]);
        }
        self.bounds.clamp(&mut candidate);

        let fitness = problem.evaluate(&candidate);
        if fitness < objective[i] {
            sources[i] = candidate;
            objective[i] = fitness;
            trials[i] = 0;
        } else {
            trials[i] += 1;
        }
    }
}

/// Map a raw objective value (lower is better) to a positive selection fitness
/// (higher is better) for roulette-wheel weighting.
fn selection_fitness(objective: f64) -> f64 {
    if objective >= 0.0 {
        1.0 / (1.0 + objective)
    } else {
        1.0 + objective.abs()
    }
}

impl Optimizer for BeeColony {
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

        let sn = (self.n_bees / 2).max(1);

        let mut rng = make_rng(self.random_seed);

        let ranges = self.bounds.ranges();

        // Initialize food sources.
        let mut sources: Vec<Vec<f64>> = (0..sn)
            .map(|_| self.random_source(&ranges, &mut rng))
            .collect();
        let mut objective: Vec<f64> = sources.iter().map(|s| problem.evaluate(s)).collect();
        let mut trials = vec![0usize; sn];

        let mut best_index = 0;
        for i in 1..sn {
            if objective[i] < objective[best_index] {
                best_index = i;
            }
        }
        let mut best_position = sources[best_index].clone();
        let mut best_objective = objective[best_index];

        for _ in 0..self.n_iterations {
            // Employed bee phase: one candidate per source.
            for i in 0..sn {
                self.exploit_source(
                    &mut sources,
                    &mut objective,
                    &mut trials,
                    i,
                    problem,
                    &mut rng,
                );
            }

            // Onlooker bee phase: SN candidates, sources chosen by roulette wheel.
            let weights: Vec<f64> = objective.iter().map(|&f| selection_fitness(f)).collect();
            let total: f64 = weights.iter().sum();
            for _ in 0..sn {
                let selected = roulette_select(&weights, total, &mut rng);
                self.exploit_source(
                    &mut sources,
                    &mut objective,
                    &mut trials,
                    selected,
                    problem,
                    &mut rng,
                );
            }

            // Scout bee phase: abandon the most-stagnant source past the limit.
            if let Some((i, &t)) = trials.iter().enumerate().max_by_key(|(_, &t)| t) {
                if t > self.limit {
                    sources[i] = self.random_source(&ranges, &mut rng);
                    objective[i] = problem.evaluate(&sources[i]);
                    trials[i] = 0;
                }
            }

            // Memorize the best source found so far.
            for i in 0..sn {
                if objective[i] < best_objective {
                    best_objective = objective[i];
                    best_position.copy_from_slice(&sources[i]);
                }
            }
        }

        self.best_solution = Some(Solution::with_fitness(best_position, best_objective));
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
        params.insert("n_bees".to_string(), self.n_bees as f64);
        params.insert("n_iterations".to_string(), self.n_iterations as f64);
        params.insert("limit".to_string(), self.limit as f64);
        params
    }
}

/// Select an index in proportion to `weights` (which sum to `total`).
fn roulette_select(weights: &[f64], total: f64, rng: &mut StdRng) -> usize {
    if !(total > 0.0) || !total.is_finite() {
        return rng.gen_range(0..weights.len());
    }
    let threshold = rng.gen::<f64>() * total;
    let mut acc = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        acc += w;
        if acc >= threshold {
            return i;
        }
    }
    weights.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ContinuousProblem;

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
        let mut abc = BeeColony::new(30, 150, 20, bounds);
        abc.set_random_seed(Some(42));
        abc.fit(&sphere(3)).unwrap();

        let score = abc.score().unwrap();
        assert!(score < 1e-2, "expected near-zero minimum, got {score}");
    }

    #[test]
    fn solution_stays_within_bounds() {
        let bounds = Bounds::uniform(2, -2.0, 2.0).unwrap();
        let mut abc = BeeColony::new(20, 50, 10, bounds.clone());
        abc.set_random_seed(Some(1));
        abc.fit(&sphere(2)).unwrap();

        let solution = abc.predict().unwrap();
        assert!(bounds.contains(&solution.variables));
    }

    #[test]
    fn reproducible_with_seed() {
        let run = || {
            let bounds = Bounds::uniform(2, -5.0, 5.0).unwrap();
            let mut abc = BeeColony::new(20, 60, 10, bounds);
            abc.set_random_seed(Some(7));
            abc.fit(&sphere(2)).unwrap();
            abc.score().unwrap()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn rejects_dimension_mismatch() {
        let bounds = Bounds::uniform(3, -1.0, 1.0).unwrap();
        let mut abc = BeeColony::new(10, 10, 5, bounds);
        assert!(abc.fit(&sphere(2)).is_err());
    }
}
