use crate::core::{Problem, Solution};
use crate::algorithms::base::{Optimizer, OptimizationError};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

/// Smallest distance used when computing the inverse-distance heuristic, so
/// zero-distance edges don't produce an infinite attractiveness.
const MIN_DISTANCE: f64 = 1e-10;

pub struct AntColony {
    pub n_ants: usize,
    pub n_iterations: usize,
    pub alpha: f64, // Pheromone importance
    pub beta: f64, // Heuristic importance
    pub rho: f64, // Evaporation rate
    pub q: f64, // Q value for pheromone update
    
    // Internal state using core types
    pheromone_matrix: Option<Vec<Vec<f64>>>,
    best_solution: Option<Solution>,

    random_seed: Option<u64>,
}

impl AntColony {
    pub fn new(
        n_ants: usize,
        n_iterations: usize,
        alpha: f64,
        beta: f64,
        rho: f64,
        q: f64,
    ) -> Self {
        Self {
            n_ants,
            n_iterations,
            alpha,
            beta,
            rho,
            q,
            pheromone_matrix: None,
            best_solution: None,
            random_seed: None,
        }
    }
    
    /// Set the RNG seed for reproducible runs. `None` draws from entropy.
    pub fn set_random_seed(&mut self, seed: Option<u64>) {
        self.random_seed = seed;
    }

    fn initialize_pheromone_matrix(&mut self, size: usize) {
        let initial_pheromone = 1.0 / (size as f64);
        self.pheromone_matrix = Some(vec![vec![initial_pheromone; size]; size]);
    }

    /// Construct one ant's tour by probabilistic edge selection.
    ///
    /// For discrete problems the ant walks the graph, choosing the next city
    /// with probability proportional to `pheromone^alpha * (1/distance)^beta`
    /// over the unvisited cities (roulette-wheel selection). Non-discrete or
    /// uninitialised problems fall back to the trivial identity tour.
    fn construct_solution(&self, problem: &dyn Problem, rng: &mut StdRng) -> Solution {
        let n = problem.dimensions();

        let (distances, pheromone) = match (problem.distance_matrix(), self.pheromone_matrix.as_ref()) {
            (Some(distances), Some(pheromone)) => (distances, pheromone),
            _ => {
                let variables = (0..n).map(|i| i as f64).collect();
                return Solution::new(variables);
            }
        };

        let mut visited = vec![false; n];
        let mut tour = Vec::with_capacity(n);

        let start = rng.gen_range(0..n);
        tour.push(start);
        visited[start] = true;
        let mut current = start;

        for _ in 1..n {
            let next = self.select_next_city(current, &visited, distances, pheromone, rng);
            tour.push(next);
            visited[next] = true;
            current = next;
        }

        Solution::new(tour.into_iter().map(|c| c as f64).collect())
    }

    /// Pick the next city from `current` among unvisited cities, weighting each
    /// candidate by `pheromone^alpha * (1/distance)^beta`.
    fn select_next_city(
        &self,
        current: usize,
        visited: &[bool],
        distances: &[Vec<f64>],
        pheromone: &[Vec<f64>],
        rng: &mut StdRng,
    ) -> usize {
        let n = visited.len();

        let mut candidates: Vec<(usize, f64)> = Vec::with_capacity(n);
        let mut total = 0.0;
        for city in 0..n {
            if visited[city] {
                continue;
            }
            let tau = pheromone[current][city].powf(self.alpha);
            let eta = (1.0 / distances[current][city].max(MIN_DISTANCE)).powf(self.beta);
            let weight = tau * eta;
            total += weight;
            candidates.push((city, weight));
        }

        // Degenerate weights (all zero / non-finite): pick a uniform unvisited city.
        if !(total > 0.0) || !total.is_finite() {
            let pick = rng.gen_range(0..candidates.len());
            return candidates[pick].0;
        }

        // Roulette-wheel selection.
        let threshold = rng.gen::<f64>() * total;
        let mut acc = 0.0;
        for (city, weight) in &candidates {
            acc += weight;
            if acc >= threshold {
                return *city;
            }
        }

        // Floating-point fall-through: return the last candidate.
        candidates.last().unwrap().0
    }

    /// Apply one Ant System pheromone update: evaporate every edge by `rho`,
    /// then deposit `q / tour_length` on the edges each ant traversed.
    ///
    /// Deposits are symmetric (`i->j` and `j->i`) since tours are undirected.
    fn update_pheromones(&mut self, solutions: &[Solution]) {
        let rho = self.rho;
        let q = self.q;

        let pheromone = match self.pheromone_matrix.as_mut() {
            Some(pheromone) => pheromone,
            None => return,
        };
        let n = pheromone.len();

        // Evaporation.
        for i in 0..n {
            for j in 0..n {
                pheromone[i][j] *= 1.0 - rho;
            }
        }

        // Deposit, proportional to tour quality (shorter tour => more pheromone).
        for solution in solutions {
            let length = match solution.fitness {
                Some(length) if length > 0.0 => length,
                _ => continue,
            };
            let deposit = q / length;

            let tour = &solution.variables;
            let m = tour.len();
            for idx in 0..m {
                let from = tour[idx] as usize;
                let to = tour[(idx + 1) % m] as usize;
                pheromone[from][to] += deposit;
                pheromone[to][from] += deposit;
            }
        }
    }
}

impl Optimizer for AntColony {
    type Solution = Solution;
    
    fn fit(&mut self, problem: &dyn Problem) -> Result<(), OptimizationError> {
        if problem.dimensions() == 0 {
            return Err(OptimizationError::InvalidInput(
                "Problem must have at least one dimension".to_string()
            ));
        }
        
        // Initialize pheromone matrix for discrete problems
        if problem.is_discrete() {
            self.initialize_pheromone_matrix(problem.dimensions());
        }
        
        let mut best_fitness = f64::INFINITY;

        let mut rng = match self.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        // Main ACO loop
        for iteration in 0..self.n_iterations {
            let mut solutions = Vec::new();

            // Generate solutions with ants
            for _ant in 0..self.n_ants {
                let mut solution = self.construct_solution(problem, &mut rng);
                
                // Evaluate solution
                let fitness = problem.evaluate(&solution.variables);
                solution.set_fitness(fitness);
                solution.add_metadata("iteration".to_string(), iteration.to_string());
                
                // Update best solution
                if fitness < best_fitness {
                    best_fitness = fitness;
                    self.best_solution = Some(solution.clone());
                }
                
                solutions.push(solution);
            }

            // Evaporate, then let this iteration's ants deposit pheromone.
            self.update_pheromones(&solutions);
        }

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
        params.insert("n_ants".to_string(), self.n_ants as f64);
        params.insert("n_iterations".to_string(), self.n_iterations as f64);
        params.insert("alpha".to_string(), self.alpha);
        params.insert("beta".to_string(), self.beta);
        params.insert("rho".to_string(), self.rho);
        params.insert("q".to_string(), self.q);
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::DiscreteProblem;

    /// A ring graph on `n` cities: adjacent cities cost 1, everything else 10.
    /// The unique optimal tour is the ring itself, of length `n`.
    fn ring_problem(n: usize) -> DiscreteProblem {
        let mut matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let gap = ((i as i64) - (j as i64)).abs();
                matrix[i][j] = if i == j {
                    0.0
                } else if gap == 1 || gap == (n as i64 - 1) {
                    1.0
                } else {
                    10.0
                };
            }
        }
        DiscreteProblem {
            name: "ring".to_string(),
            distance_matrix: matrix,
        }
    }

    #[test]
    fn constructs_valid_permutation() {
        let mut aco = AntColony::new(10, 20, 1.0, 2.0, 0.5, 1.0);
        aco.set_random_seed(Some(1));
        aco.fit(&ring_problem(6)).unwrap();

        let solution = aco.predict().unwrap();
        let mut tour: Vec<usize> = solution.variables.iter().map(|&x| x as usize).collect();
        tour.sort_unstable();
        assert_eq!(tour, (0..6).collect::<Vec<_>>());
    }

    #[test]
    fn evaporation_scales_pheromone_by_one_minus_rho() {
        let mut aco = AntColony::new(5, 1, 1.0, 2.0, 0.5, 1.0);
        aco.initialize_pheromone_matrix(4);
        let before = aco.pheromone_matrix.as_ref().unwrap()[0][1];

        // No solutions => pure evaporation, no deposit.
        aco.update_pheromones(&[]);
        let after = aco.pheromone_matrix.as_ref().unwrap()[0][1];

        assert!(after < before);
        assert!((after - before * (1.0 - 0.5)).abs() < 1e-12);
    }

    #[test]
    fn deposit_adds_symmetric_pheromone_on_used_edges() {
        let mut aco = AntColony::new(1, 1, 1.0, 2.0, 0.0, 2.0); // rho=0 => no evaporation
        aco.initialize_pheromone_matrix(3);
        let base = aco.pheromone_matrix.as_ref().unwrap()[0][1];

        // Tour 0->1->2->0 with length 4 => deposit q/L = 2/4 = 0.5 per edge.
        let solution = Solution::with_fitness(vec![0.0, 1.0, 2.0], 4.0);
        aco.update_pheromones(std::slice::from_ref(&solution));

        let pheromone = aco.pheromone_matrix.as_ref().unwrap();
        assert!((pheromone[0][1] - (base + 0.5)).abs() < 1e-12);
        assert!((pheromone[1][0] - (base + 0.5)).abs() < 1e-12); // symmetric
        assert!((pheromone[1][2] - (base + 0.5)).abs() < 1e-12);
    }

    #[test]
    fn learns_optimal_ring_from_pheromone_only() {
        // beta = 0 disables the distance heuristic, so reaching the optimum
        // proves the pheromone deposit/evaporation loop is doing the learning.
        let n = 8;
        let mut aco = AntColony::new(20, 100, 1.0, 0.0, 0.5, 1.0);
        aco.set_random_seed(Some(42));
        aco.fit(&ring_problem(n)).unwrap();

        assert_eq!(aco.score().unwrap(), n as f64);
    }

    #[test]
    fn rejects_empty_problem() {
        let mut aco = AntColony::new(5, 10, 1.0, 2.0, 0.5, 1.0);
        let empty = DiscreteProblem {
            name: "empty".to_string(),
            distance_matrix: vec![],
        };
        assert!(aco.fit(&empty).is_err());
    }
}