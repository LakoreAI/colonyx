use crate::algorithms::base::{OptimizationError, Optimizer};
use crate::core::{Bounds, Problem, Solution};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::f64::consts::PI;

const MIN_STEP_DENOMINATOR: f64 = 1e-12;
const TWO_OPT_EPSILON: f64 = 1e-12;

fn clamp_value(value: f64, lower: f64, upper: f64) -> f64 {
    value.max(lower).min(upper)
}

fn distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&left, &right)| {
            let delta = left - right;
            delta * delta
        })
        .sum::<f64>()
        .sqrt()
}

fn random_position(bounds: &Bounds, ranges: &[f64], rng: &mut StdRng) -> Vec<f64> {
    (0..bounds.lower.len())
        .map(|index| bounds.lower[index] + rng.gen::<f64>() * ranges[index])
        .collect()
}

fn standard_normal(rng: &mut StdRng) -> f64 {
    let u1 = rng.gen::<f64>().clamp(MIN_STEP_DENOMINATOR, 1.0);
    let u2 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

fn tour_length(matrix: &[Vec<f64>], tour: &[usize]) -> f64 {
    let mut total = 0.0;
    for index in 0..tour.len() {
        let start = tour[index];
        let end = tour[(index + 1) % tour.len()];
        total += matrix[start][end];
    }
    total
}

pub fn two_opt(tour: &[usize], distance_matrix: &[Vec<f64>]) -> Result<(Vec<usize>, f64), String> {
    if distance_matrix.is_empty() {
        return Err("distance_matrix must be non-empty".to_string());
    }

    let size = distance_matrix.len();
    for (index, row) in distance_matrix.iter().enumerate() {
        if row.len() != size {
            return Err(format!(
                "distance_matrix must be square: row {} has length {}, expected {}",
                index,
                row.len(),
                size
            ));
        }
    }

    let route = tour.to_vec();
    if route.len() < 4 {
        let length = tour_length(distance_matrix, &route);
        return Ok((route, length));
    }

    let mut best_route = route;
    let mut best_length = tour_length(distance_matrix, &best_route);
    let mut improved = true;

    while improved {
        improved = false;
        let n = best_route.len();
        for start in 1..n - 2 {
            for end in start + 2..=n {
                let mut candidate = best_route.clone();
                candidate[start..end].reverse();
                let candidate_length = tour_length(distance_matrix, &candidate);
                if candidate_length + TWO_OPT_EPSILON < best_length {
                    best_route = candidate;
                    best_length = candidate_length;
                    improved = true;
                    break;
                }
            }
            if improved {
                break;
            }
        }
    }

    Ok((best_route, best_length))
}

pub struct GreyWolfOptimizer {
    pub n_wolves: usize,
    pub n_iterations: usize,
    pub bounds: Bounds,
    pub random_seed: Option<u64>,
    pub best_solution: Option<Solution>,
    pub history: Vec<f64>,
    pub population: Vec<Vec<f64>>,
}

impl GreyWolfOptimizer {
    pub fn new(n_wolves: usize, n_iterations: usize, bounds: Bounds) -> Self {
        Self {
            n_wolves,
            n_iterations,
            bounds,
            random_seed: None,
            best_solution: None,
            history: Vec::new(),
            population: Vec::new(),
        }
    }

    pub fn set_random_seed(&mut self, seed: Option<u64>) {
        self.random_seed = seed;
    }
}

impl Optimizer for GreyWolfOptimizer {
    type Solution = Solution;

    fn fit(&mut self, problem: &dyn Problem) -> Result<(), OptimizationError> {
        let dimension = self.bounds.lower.len();
        if dimension == 0 {
            return Err(OptimizationError::InvalidInput(
                "Problem must have at least one dimension".to_string(),
            ));
        }
        if self.n_wolves < 3 {
            return Err(OptimizationError::InvalidInput(
                "n_wolves must be at least 3 for grey wolf optimization".to_string(),
            ));
        }

        let mut rng = match self.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        if problem.dimensions() != dimension {
            return Err(OptimizationError::DimensionMismatch(format!(
                "Problem has {} dimensions but bounds have {}",
                problem.dimensions(),
                dimension
            )));
        }
        let ranges = self.bounds.ranges();
        let mut wolves: Vec<Vec<f64>> = (0..self.n_wolves)
            .map(|_| random_position(&self.bounds, &ranges, &mut rng))
            .collect();
        let mut scores: Vec<f64> = wolves.iter().map(|wolf| problem.evaluate(wolf)).collect();

        let best_index = scores
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(index, _)| index)
            .unwrap();
        let mut best_position = wolves[best_index].clone();
        let mut best_score = scores[best_index];

        for iteration in 0..self.n_iterations {
            let mut ranking: Vec<usize> = (0..self.n_wolves).collect();
            ranking.sort_by(|left, right| scores[*left].partial_cmp(&scores[*right]).unwrap());
            let alpha = wolves[ranking[0]].clone();
            let beta = wolves[ranking[1]].clone();
            let delta = wolves[ranking[2]].clone();
            let a_value = 2.0
                * (1.0 - iteration as f64 / (self.n_iterations.saturating_sub(1).max(1) as f64));

            for wolf_index in 0..self.n_wolves {
                let mut candidate = wolves[wolf_index].clone();
                for dimension_index in 0..dimension {
                    let r1 = rng.gen::<f64>();
                    let r2 = rng.gen::<f64>();
                    let a1 = 2.0 * a_value * r1 - a_value;
                    let c1 = 2.0 * r2;
                    let d_alpha = (c1 * alpha[dimension_index] - candidate[dimension_index]).abs();
                    let x1 = alpha[dimension_index] - a1 * d_alpha;

                    let r1 = rng.gen::<f64>();
                    let r2 = rng.gen::<f64>();
                    let a2 = 2.0 * a_value * r1 - a_value;
                    let c2 = 2.0 * r2;
                    let d_beta = (c2 * beta[dimension_index] - candidate[dimension_index]).abs();
                    let x2 = beta[dimension_index] - a2 * d_beta;

                    let r1 = rng.gen::<f64>();
                    let r2 = rng.gen::<f64>();
                    let a3 = 2.0 * a_value * r1 - a_value;
                    let c3 = 2.0 * r2;
                    let d_delta = (c3 * delta[dimension_index] - candidate[dimension_index]).abs();
                    let x3 = delta[dimension_index] - a3 * d_delta;

                    candidate[dimension_index] = clamp_value(
                        (x1 + x2 + x3) / 3.0,
                        self.bounds.lower[dimension_index],
                        self.bounds.upper[dimension_index],
                    );
                }

                let score = problem.evaluate(&candidate);
                wolves[wolf_index] = candidate.clone();
                scores[wolf_index] = score;
                if score < best_score {
                    best_score = score;
                    best_position = candidate;
                }
            }

            self.history.push(best_score);
        }

        self.population = wolves;
        self.best_solution = Some(Solution::with_fitness(best_position, best_score));
        Ok(())
    }

    fn predict(&self) -> Option<Self::Solution> {
        self.best_solution.clone()
    }

    fn score(&self) -> Option<f64> {
        self.best_solution
            .as_ref()
            .and_then(|solution| solution.fitness)
    }

    fn get_params(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("n_wolves".to_string(), self.n_wolves as f64);
        params.insert("n_iterations".to_string(), self.n_iterations as f64);
        params
    }
}

pub struct FireflyOptimizer {
    pub n_fireflies: usize,
    pub n_iterations: usize,
    pub beta0: f64,
    pub gamma: f64,
    pub alpha: f64,
    pub bounds: Bounds,
    pub random_seed: Option<u64>,
    pub best_solution: Option<Solution>,
    pub history: Vec<f64>,
    pub population: Vec<Vec<f64>>,
}

impl FireflyOptimizer {
    pub fn new(
        n_fireflies: usize,
        n_iterations: usize,
        beta0: f64,
        gamma: f64,
        alpha: f64,
        bounds: Bounds,
    ) -> Self {
        Self {
            n_fireflies,
            n_iterations,
            beta0,
            gamma,
            alpha,
            bounds,
            random_seed: None,
            best_solution: None,
            history: Vec::new(),
            population: Vec::new(),
        }
    }

    pub fn set_random_seed(&mut self, seed: Option<u64>) {
        self.random_seed = seed;
    }
}

impl Optimizer for FireflyOptimizer {
    type Solution = Solution;

    fn fit(&mut self, problem: &dyn Problem) -> Result<(), OptimizationError> {
        let dimension = self.bounds.lower.len();
        if dimension == 0 {
            return Err(OptimizationError::InvalidInput(
                "Problem must have at least one dimension".to_string(),
            ));
        }
        if problem.dimensions() != dimension {
            return Err(OptimizationError::DimensionMismatch(format!(
                "Problem has {} dimensions but bounds have {}",
                problem.dimensions(),
                dimension
            )));
        }
        if self.n_fireflies < 2 {
            return Err(OptimizationError::InvalidInput(
                "n_fireflies must be at least 2 for firefly optimization".to_string(),
            ));
        }

        let mut rng = match self.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        let ranges = self.bounds.ranges();
        let mut fireflies: Vec<Vec<f64>> = (0..self.n_fireflies)
            .map(|_| random_position(&self.bounds, &ranges, &mut rng))
            .collect();
        let mut scores: Vec<f64> = fireflies
            .iter()
            .map(|firefly| problem.evaluate(firefly))
            .collect();

        let best_index = scores
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(index, _)| index)
            .unwrap();
        let mut best_position = fireflies[best_index].clone();
        let mut best_score = scores[best_index];

        for _ in 0..self.n_iterations {
            for first_index in 0..self.n_fireflies {
                for second_index in 0..self.n_fireflies {
                    if scores[second_index] >= scores[first_index] {
                        continue;
                    }

                    let dist = distance(&fireflies[first_index], &fireflies[second_index]);
                    let beta = self.beta0 * (-self.gamma * dist * dist).exp();
                    let mut candidate = fireflies[first_index].clone();

                    for dimension_index in 0..dimension {
                        let random_step =
                            self.alpha * (rng.gen::<f64>() - 0.5) * ranges[dimension_index];
                        candidate[dimension_index] += beta
                            * (fireflies[second_index][dimension_index]
                                - candidate[dimension_index])
                            + random_step;
                        candidate[dimension_index] = clamp_value(
                            candidate[dimension_index],
                            self.bounds.lower[dimension_index],
                            self.bounds.upper[dimension_index],
                        );
                    }

                    let candidate_score = problem.evaluate(&candidate);
                    if candidate_score < scores[first_index] {
                        fireflies[first_index] = candidate.clone();
                        scores[first_index] = candidate_score;
                        if candidate_score < best_score {
                            best_score = candidate_score;
                            best_position = candidate;
                        }
                    }
                }
            }

            self.history.push(best_score);
        }

        self.population = fireflies;
        self.best_solution = Some(Solution::with_fitness(best_position, best_score));
        Ok(())
    }

    fn predict(&self) -> Option<Self::Solution> {
        self.best_solution.clone()
    }

    fn score(&self) -> Option<f64> {
        self.best_solution
            .as_ref()
            .and_then(|solution| solution.fitness)
    }

    fn get_params(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("n_fireflies".to_string(), self.n_fireflies as f64);
        params.insert("n_iterations".to_string(), self.n_iterations as f64);
        params.insert("beta0".to_string(), self.beta0);
        params.insert("gamma".to_string(), self.gamma);
        params.insert("alpha".to_string(), self.alpha);
        params
    }
}

pub struct SimulatedAnnealing {
    pub initial_temperature: f64,
    pub cooling_rate: f64,
    pub step_scale: f64,
    pub n_iterations: usize,
    pub bounds: Bounds,
    pub random_seed: Option<u64>,
    pub best_solution: Option<Solution>,
    pub history: Vec<f64>,
    pub population: Vec<Vec<f64>>,
}

impl SimulatedAnnealing {
    pub fn new(
        initial_temperature: f64,
        cooling_rate: f64,
        step_scale: f64,
        n_iterations: usize,
        bounds: Bounds,
    ) -> Self {
        Self {
            initial_temperature,
            cooling_rate,
            step_scale,
            n_iterations,
            bounds,
            random_seed: None,
            best_solution: None,
            history: Vec::new(),
            population: Vec::new(),
        }
    }

    pub fn set_random_seed(&mut self, seed: Option<u64>) {
        self.random_seed = seed;
    }
}

impl Optimizer for SimulatedAnnealing {
    type Solution = Solution;

    fn fit(&mut self, problem: &dyn Problem) -> Result<(), OptimizationError> {
        let dimension = self.bounds.lower.len();
        if dimension == 0 {
            return Err(OptimizationError::InvalidInput(
                "Problem must have at least one dimension".to_string(),
            ));
        }
        if problem.dimensions() != dimension {
            return Err(OptimizationError::DimensionMismatch(format!(
                "Problem has {} dimensions but bounds have {}",
                problem.dimensions(),
                dimension
            )));
        }
        let mut rng = match self.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        let ranges = self.bounds.ranges();

        let mut current = random_position(&self.bounds, &ranges, &mut rng);
        let mut current_score = problem.evaluate(&current);
        let mut best_position = current.clone();
        let mut best_score = current_score;
        let mut temperature = self.initial_temperature;

        for _ in 0..self.n_iterations {
            let mut candidate = current.clone();
            for dimension_index in 0..dimension {
                let step =
                    (rng.gen::<f64>() * 2.0 - 1.0) * self.step_scale * ranges[dimension_index];
                candidate[dimension_index] = clamp_value(
                    candidate[dimension_index] + step,
                    self.bounds.lower[dimension_index],
                    self.bounds.upper[dimension_index],
                );
            }

            let candidate_score = problem.evaluate(&candidate);
            let delta = candidate_score - current_score;
            if delta <= 0.0
                || rng.gen::<f64>() < (-delta / temperature.max(MIN_STEP_DENOMINATOR)).exp()
            {
                current = candidate;
                current_score = candidate_score;
            }

            if current_score < best_score {
                best_score = current_score;
                best_position = current.clone();
            }

            self.history.push(best_score);
            temperature *= self.cooling_rate;
        }

        self.population = vec![current];
        self.best_solution = Some(Solution::with_fitness(best_position, best_score));
        Ok(())
    }

    fn predict(&self) -> Option<Self::Solution> {
        self.best_solution.clone()
    }

    fn score(&self) -> Option<f64> {
        self.best_solution
            .as_ref()
            .and_then(|solution| solution.fitness)
    }

    fn get_params(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("initial_temperature".to_string(), self.initial_temperature);
        params.insert("cooling_rate".to_string(), self.cooling_rate);
        params.insert("step_scale".to_string(), self.step_scale);
        params.insert("n_iterations".to_string(), self.n_iterations as f64);
        params
    }
}

pub struct CuckooSearch {
    pub n_nests: usize,
    pub n_iterations: usize,
    pub pa: f64,
    pub alpha: f64,
    pub levy_scale: f64,
    pub bounds: Bounds,
    pub random_seed: Option<u64>,
    pub best_solution: Option<Solution>,
    pub history: Vec<f64>,
    pub population: Vec<Vec<f64>>,
}

impl CuckooSearch {
    pub fn new(
        n_nests: usize,
        n_iterations: usize,
        pa: f64,
        alpha: f64,
        levy_scale: f64,
        bounds: Bounds,
    ) -> Self {
        Self {
            n_nests,
            n_iterations,
            pa,
            alpha,
            levy_scale,
            bounds,
            random_seed: None,
            best_solution: None,
            history: Vec::new(),
            population: Vec::new(),
        }
    }

    pub fn set_random_seed(&mut self, seed: Option<u64>) {
        self.random_seed = seed;
    }
}

impl Optimizer for CuckooSearch {
    type Solution = Solution;

    fn fit(&mut self, problem: &dyn Problem) -> Result<(), OptimizationError> {
        let dimension = self.bounds.lower.len();
        if dimension == 0 {
            return Err(OptimizationError::InvalidInput(
                "Problem must have at least one dimension".to_string(),
            ));
        }
        if problem.dimensions() != dimension {
            return Err(OptimizationError::DimensionMismatch(format!(
                "Problem has {} dimensions but bounds have {}",
                problem.dimensions(),
                dimension
            )));
        }
        if self.n_nests < 2 {
            return Err(OptimizationError::InvalidInput(
                "n_nests must be at least 2 for cuckoo search".to_string(),
            ));
        }

        let mut rng = match self.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        let ranges = self.bounds.ranges();
        let mut nests: Vec<Vec<f64>> = (0..self.n_nests)
            .map(|_| random_position(&self.bounds, &ranges, &mut rng))
            .collect();
        let mut scores: Vec<f64> = nests.iter().map(|nest| problem.evaluate(nest)).collect();

        let best_index = scores
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(index, _)| index)
            .unwrap();
        let mut best_position = nests[best_index].clone();
        let mut best_score = scores[best_index];

        for _ in 0..self.n_iterations {
            for nest_index in 0..self.n_nests {
                let mut candidate = nests[nest_index].clone();
                let u = rng.gen::<f64>() - 0.5;
                let v = rng.gen::<f64>() - 0.5;
                let denominator = v.abs().max(MIN_STEP_DENOMINATOR).powf(1.0 / 1.5);
                let step = self.levy_scale * (u / denominator);
                for dimension_index in 0..dimension {
                    candidate[dimension_index] = clamp_value(
                        candidate[dimension_index] + self.alpha * step * ranges[dimension_index],
                        self.bounds.lower[dimension_index],
                        self.bounds.upper[dimension_index],
                    );
                }

                let candidate_score = problem.evaluate(&candidate);
                if candidate_score < scores[nest_index] {
                    nests[nest_index] = candidate;
                    scores[nest_index] = candidate_score;
                }
            }

            let abandon_count = (self.pa * self.n_nests as f64).ceil() as usize;
            let abandon_count = abandon_count.max(1).min(self.n_nests);
            let mut worst_indices: Vec<usize> = (0..self.n_nests).collect();
            worst_indices
                .sort_by(|left, right| scores[*right].partial_cmp(&scores[*left]).unwrap());
            for nest_index in worst_indices.into_iter().take(abandon_count) {
                nests[nest_index] = random_position(&self.bounds, &ranges, &mut rng);
                scores[nest_index] = problem.evaluate(&nests[nest_index]);
            }

            let iteration_best = scores
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(index, _)| index)
                .unwrap();
            if scores[iteration_best] < best_score {
                best_score = scores[iteration_best];
                best_position = nests[iteration_best].clone();
            }

            self.history.push(best_score);
        }

        self.population = nests;
        self.best_solution = Some(Solution::with_fitness(best_position, best_score));
        Ok(())
    }

    fn predict(&self) -> Option<Self::Solution> {
        self.best_solution.clone()
    }

    fn score(&self) -> Option<f64> {
        self.best_solution
            .as_ref()
            .and_then(|solution| solution.fitness)
    }

    fn get_params(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("n_nests".to_string(), self.n_nests as f64);
        params.insert("n_iterations".to_string(), self.n_iterations as f64);
        params.insert("pa".to_string(), self.pa);
        params.insert("alpha".to_string(), self.alpha);
        params.insert("levy_scale".to_string(), self.levy_scale);
        params
    }
}

pub struct BatAlgorithm {
    pub n_bats: usize,
    pub n_iterations: usize,
    pub fmin: f64,
    pub fmax: f64,
    pub alpha: f64,
    pub gamma: f64,
    pub loudness: f64,
    pub pulse_rate: f64,
    pub bounds: Bounds,
    pub random_seed: Option<u64>,
    pub best_solution: Option<Solution>,
    pub history: Vec<f64>,
    pub population: Vec<Vec<f64>>,
}

impl BatAlgorithm {
    pub fn new(
        n_bats: usize,
        n_iterations: usize,
        fmin: f64,
        fmax: f64,
        alpha: f64,
        gamma: f64,
        loudness: f64,
        pulse_rate: f64,
        bounds: Bounds,
    ) -> Self {
        Self {
            n_bats,
            n_iterations,
            fmin,
            fmax,
            alpha,
            gamma,
            loudness,
            pulse_rate,
            bounds,
            random_seed: None,
            best_solution: None,
            history: Vec::new(),
            population: Vec::new(),
        }
    }

    pub fn set_random_seed(&mut self, seed: Option<u64>) {
        self.random_seed = seed;
    }
}

impl Optimizer for BatAlgorithm {
    type Solution = Solution;

    fn fit(&mut self, problem: &dyn Problem) -> Result<(), OptimizationError> {
        let dimension = self.bounds.lower.len();
        if dimension == 0 {
            return Err(OptimizationError::InvalidInput(
                "Problem must have at least one dimension".to_string(),
            ));
        }
        if problem.dimensions() != dimension {
            return Err(OptimizationError::DimensionMismatch(format!(
                "Problem has {} dimensions but bounds have {}",
                problem.dimensions(),
                dimension
            )));
        }
        if self.n_bats < 2 {
            return Err(OptimizationError::InvalidInput(
                "n_bats must be at least 2 for bat algorithm".to_string(),
            ));
        }

        let mut rng = match self.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        let ranges = self.bounds.ranges();
        let mut positions: Vec<Vec<f64>> = (0..self.n_bats)
            .map(|_| random_position(&self.bounds, &ranges, &mut rng))
            .collect();
        let mut velocities = vec![vec![0.0; dimension]; self.n_bats];
        let mut frequencies = vec![0.0; self.n_bats];
        let mut scores: Vec<f64> = positions
            .iter()
            .map(|position| problem.evaluate(position))
            .collect();
        let mut loudness = vec![self.loudness; self.n_bats];
        let mut pulse = vec![self.pulse_rate; self.n_bats];

        let best_index = scores
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(index, _)| index)
            .unwrap();
        let mut best_position = positions[best_index].clone();
        let mut best_score = scores[best_index];

        for iteration in 0..self.n_iterations {
            for bat_index in 0..self.n_bats {
                frequencies[bat_index] = self.fmin + (self.fmax - self.fmin) * rng.gen::<f64>();
                for dimension_index in 0..dimension {
                    velocities[bat_index][dimension_index] +=
                        (positions[bat_index][dimension_index] - best_position[dimension_index])
                            * frequencies[bat_index];
                    positions[bat_index][dimension_index] = clamp_value(
                        positions[bat_index][dimension_index]
                            + velocities[bat_index][dimension_index],
                        self.bounds.lower[dimension_index],
                        self.bounds.upper[dimension_index],
                    );
                }

                if rng.gen::<f64>() > pulse[bat_index] {
                    for dimension_index in 0..dimension {
                        positions[bat_index][dimension_index] = clamp_value(
                            best_position[dimension_index] + 0.001 * standard_normal(&mut rng),
                            self.bounds.lower[dimension_index],
                            self.bounds.upper[dimension_index],
                        );
                    }
                }

                let candidate_score = problem.evaluate(&positions[bat_index]);
                if candidate_score <= scores[bat_index] && rng.gen::<f64>() < loudness[bat_index] {
                    scores[bat_index] = candidate_score;
                    loudness[bat_index] *= self.alpha;
                    pulse[bat_index] *= 1.0 - (-self.gamma * (iteration as f64 + 1.0)).exp();
                    if candidate_score < best_score {
                        best_score = candidate_score;
                        best_position = positions[bat_index].clone();
                    }
                }
            }

            self.history.push(best_score);
        }

        self.population = positions;
        self.best_solution = Some(Solution::with_fitness(best_position, best_score));
        Ok(())
    }

    fn predict(&self) -> Option<Self::Solution> {
        self.best_solution.clone()
    }

    fn score(&self) -> Option<f64> {
        self.best_solution
            .as_ref()
            .and_then(|solution| solution.fitness)
    }

    fn get_params(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("n_bats".to_string(), self.n_bats as f64);
        params.insert("n_iterations".to_string(), self.n_iterations as f64);
        params.insert("fmin".to_string(), self.fmin);
        params.insert("fmax".to_string(), self.fmax);
        params.insert("alpha".to_string(), self.alpha);
        params.insert("gamma".to_string(), self.gamma);
        params.insert("loudness".to_string(), self.loudness);
        params.insert("pulse_rate".to_string(), self.pulse_rate);
        params
    }
}

pub struct GlowwormOptimizer {
    pub n_worms: usize,
    pub n_iterations: usize,
    pub luciferin_decay: f64,
    pub luciferin_enhancement: f64,
    pub step_size: f64,
    pub neighborhood_radius: f64,
    pub bounds: Bounds,
    pub random_seed: Option<u64>,
    pub best_solution: Option<Solution>,
    pub history: Vec<f64>,
    pub population: Vec<Vec<f64>>,
}

impl GlowwormOptimizer {
    pub fn new(
        n_worms: usize,
        n_iterations: usize,
        luciferin_decay: f64,
        luciferin_enhancement: f64,
        step_size: f64,
        neighborhood_radius: f64,
        bounds: Bounds,
    ) -> Self {
        Self {
            n_worms,
            n_iterations,
            luciferin_decay,
            luciferin_enhancement,
            step_size,
            neighborhood_radius,
            bounds,
            random_seed: None,
            best_solution: None,
            history: Vec::new(),
            population: Vec::new(),
        }
    }

    pub fn set_random_seed(&mut self, seed: Option<u64>) {
        self.random_seed = seed;
    }
}

impl Optimizer for GlowwormOptimizer {
    type Solution = Solution;

    fn fit(&mut self, problem: &dyn Problem) -> Result<(), OptimizationError> {
        let dimension = self.bounds.lower.len();
        if dimension == 0 {
            return Err(OptimizationError::InvalidInput(
                "Problem must have at least one dimension".to_string(),
            ));
        }
        if problem.dimensions() != dimension {
            return Err(OptimizationError::DimensionMismatch(format!(
                "Problem has {} dimensions but bounds have {}",
                problem.dimensions(),
                dimension
            )));
        }
        if self.n_worms < 2 {
            return Err(OptimizationError::InvalidInput(
                "n_worms must be at least 2 for glowworm optimization".to_string(),
            ));
        }

        let mut rng = match self.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        let ranges = self.bounds.ranges();
        let mut worms: Vec<Vec<f64>> = (0..self.n_worms)
            .map(|_| random_position(&self.bounds, &ranges, &mut rng))
            .collect();
        let mut luciferin = vec![1.0; self.n_worms];
        let mut scores: Vec<f64> = worms.iter().map(|worm| problem.evaluate(worm)).collect();

        let best_index = scores
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(index, _)| index)
            .unwrap();
        let mut best_position = worms[best_index].clone();
        let mut best_score = scores[best_index];

        for _ in 0..self.n_iterations {
            luciferin = (0..self.n_worms)
                .map(|index| {
                    (1.0 - self.luciferin_decay) * luciferin[index]
                        + self.luciferin_enhancement
                            / (scores[index] + 1.0).max(MIN_STEP_DENOMINATOR)
                })
                .collect();

            for worm_index in 0..self.n_worms {
                let neighbors: Vec<usize> = (0..self.n_worms)
                    .filter(|&neighbor_index| {
                        neighbor_index != worm_index
                            && luciferin[neighbor_index] > luciferin[worm_index]
                            && distance(&worms[worm_index], &worms[neighbor_index])
                                <= self.neighborhood_radius
                    })
                    .collect();

                if !neighbors.is_empty() {
                    let target_index = neighbors
                        .iter()
                        .copied()
                        .min_by(|left, right| scores[*left].partial_cmp(&scores[*right]).unwrap())
                        .unwrap();
                    let mut candidate = worms[worm_index].clone();
                    for dimension_index in 0..dimension {
                        let direction =
                            worms[target_index][dimension_index] - candidate[dimension_index];
                        candidate[dimension_index] = clamp_value(
                            candidate[dimension_index] + self.step_size * direction,
                            self.bounds.lower[dimension_index],
                            self.bounds.upper[dimension_index],
                        );
                    }

                    let candidate_score = problem.evaluate(&candidate);
                    if candidate_score < scores[worm_index] {
                        worms[worm_index] = candidate;
                        scores[worm_index] = candidate_score;
                    }
                }

                if scores[worm_index] < best_score {
                    best_score = scores[worm_index];
                    best_position = worms[worm_index].clone();
                }
            }

            self.history.push(best_score);
        }

        self.population = worms;
        self.best_solution = Some(Solution::with_fitness(best_position, best_score));
        Ok(())
    }

    fn predict(&self) -> Option<Self::Solution> {
        self.best_solution.clone()
    }

    fn score(&self) -> Option<f64> {
        self.best_solution
            .as_ref()
            .and_then(|solution| solution.fitness)
    }

    fn get_params(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("n_worms".to_string(), self.n_worms as f64);
        params.insert("n_iterations".to_string(), self.n_iterations as f64);
        params.insert("luciferin_decay".to_string(), self.luciferin_decay);
        params.insert(
            "luciferin_enhancement".to_string(),
            self.luciferin_enhancement,
        );
        params.insert("step_size".to_string(), self.step_size);
        params.insert("neighborhood_radius".to_string(), self.neighborhood_radius);
        params
    }
}

pub struct BacterialForagingOptimizer {
    pub n_bacteria: usize,
    pub n_iterations: usize,
    pub n_chemotactic_steps: usize,
    pub n_reproduction_steps: usize,
    pub elimination_probability: f64,
    pub step_scale: f64,
    pub bounds: Bounds,
    pub random_seed: Option<u64>,
    pub best_solution: Option<Solution>,
    pub history: Vec<f64>,
    pub population: Vec<Vec<f64>>,
}

impl BacterialForagingOptimizer {
    pub fn new(
        n_bacteria: usize,
        n_iterations: usize,
        n_chemotactic_steps: usize,
        n_reproduction_steps: usize,
        elimination_probability: f64,
        step_scale: f64,
        bounds: Bounds,
    ) -> Self {
        Self {
            n_bacteria,
            n_iterations,
            n_chemotactic_steps,
            n_reproduction_steps,
            elimination_probability,
            step_scale,
            bounds,
            random_seed: None,
            best_solution: None,
            history: Vec::new(),
            population: Vec::new(),
        }
    }

    pub fn set_random_seed(&mut self, seed: Option<u64>) {
        self.random_seed = seed;
    }
}

impl Optimizer for BacterialForagingOptimizer {
    type Solution = Solution;

    fn fit(&mut self, problem: &dyn Problem) -> Result<(), OptimizationError> {
        let dimension = self.bounds.lower.len();
        if dimension == 0 {
            return Err(OptimizationError::InvalidInput(
                "Problem must have at least one dimension".to_string(),
            ));
        }
        if problem.dimensions() != dimension {
            return Err(OptimizationError::DimensionMismatch(format!(
                "Problem has {} dimensions but bounds have {}",
                problem.dimensions(),
                dimension
            )));
        }
        if self.n_bacteria < 2 {
            return Err(OptimizationError::InvalidInput(
                "n_bacteria must be at least 2 for bacterial foraging".to_string(),
            ));
        }

        let mut rng = match self.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        let ranges = self.bounds.ranges();
        let mut bacteria: Vec<Vec<f64>> = (0..self.n_bacteria)
            .map(|_| random_position(&self.bounds, &ranges, &mut rng))
            .collect();
        let mut scores: Vec<f64> = bacteria
            .iter()
            .map(|bacterium| problem.evaluate(bacterium))
            .collect();

        let best_index = scores
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(index, _)| index)
            .unwrap();
        let mut best_position = bacteria[best_index].clone();
        let mut best_score = scores[best_index];

        for _reproduction_round in 0..self.n_reproduction_steps {
            for _ in 0..self.n_chemotactic_steps {
                for bacterium_index in 0..self.n_bacteria {
                    let current = bacteria[bacterium_index].clone();
                    let mut candidate = current.clone();
                    for dimension_index in 0..dimension {
                        let step = (rng.gen::<f64>() * 2.0 - 1.0)
                            * self.step_scale
                            * ranges[dimension_index];
                        candidate[dimension_index] = clamp_value(
                            candidate[dimension_index] + step,
                            self.bounds.lower[dimension_index],
                            self.bounds.upper[dimension_index],
                        );
                    }

                    let candidate_score = problem.evaluate(&candidate);
                    if candidate_score < scores[bacterium_index]
                        || candidate_score <= scores[bacterium_index] + 1e-12
                    {
                        bacteria[bacterium_index] = candidate;
                        scores[bacterium_index] = candidate_score;
                    }

                    if scores[bacterium_index] < best_score {
                        best_score = scores[bacterium_index];
                        best_position = bacteria[bacterium_index].clone();
                    }
                }
            }

            let mut order: Vec<usize> = (0..self.n_bacteria).collect();
            order.sort_by(|left, right| scores[*left].partial_cmp(&scores[*right]).unwrap());
            let survivors = (self.n_bacteria / 2).max(1);
            let surviving_indices = &order[..survivors];
            let mut replicated_bacteria = Vec::with_capacity(self.n_bacteria);
            let mut replicated_scores = Vec::with_capacity(self.n_bacteria);
            while replicated_bacteria.len() < self.n_bacteria {
                for &index in surviving_indices {
                    replicated_bacteria.push(bacteria[index].clone());
                    replicated_scores.push(scores[index]);
                    if replicated_bacteria.len() >= self.n_bacteria {
                        break;
                    }
                }
            }
            bacteria = replicated_bacteria;
            scores = replicated_scores;

            for bacterium_index in 0..self.n_bacteria {
                if rng.gen::<f64>() < self.elimination_probability {
                    bacteria[bacterium_index] = random_position(&self.bounds, &ranges, &mut rng);
                    scores[bacterium_index] = problem.evaluate(&bacteria[bacterium_index]);
                    if scores[bacterium_index] < best_score {
                        best_score = scores[bacterium_index];
                        best_position = bacteria[bacterium_index].clone();
                    }
                }
            }

            self.history.push(best_score);
        }

        self.population = bacteria;
        self.best_solution = Some(Solution::with_fitness(best_position, best_score));
        Ok(())
    }

    fn predict(&self) -> Option<Self::Solution> {
        self.best_solution.clone()
    }

    fn score(&self) -> Option<f64> {
        self.best_solution
            .as_ref()
            .and_then(|solution| solution.fitness)
    }

    fn get_params(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("n_bacteria".to_string(), self.n_bacteria as f64);
        params.insert("n_iterations".to_string(), self.n_iterations as f64);
        params.insert(
            "n_chemotactic_steps".to_string(),
            self.n_chemotactic_steps as f64,
        );
        params.insert(
            "n_reproduction_steps".to_string(),
            self.n_reproduction_steps as f64,
        );
        params.insert(
            "elimination_probability".to_string(),
            self.elimination_probability,
        );
        params.insert("step_scale".to_string(), self.step_scale);
        params
    }
}

pub struct DifferentialEvolution {
    pub n_individuals: usize,
    pub n_iterations: usize,
    pub f: f64,
    pub cr: f64,
    pub bounds: Bounds,
    pub random_seed: Option<u64>,
    pub best_solution: Option<Solution>,
    pub history: Vec<f64>,
    pub population: Vec<Vec<f64>>,
}

impl DifferentialEvolution {
    pub fn new(n_individuals: usize, n_iterations: usize, f: f64, cr: f64, bounds: Bounds) -> Self {
        Self {
            n_individuals,
            n_iterations,
            f,
            cr,
            bounds,
            random_seed: None,
            best_solution: None,
            history: Vec::new(),
            population: Vec::new(),
        }
    }

    pub fn set_random_seed(&mut self, seed: Option<u64>) {
        self.random_seed = seed;
    }
}

impl Optimizer for DifferentialEvolution {
    type Solution = Solution;

    fn fit(&mut self, problem: &dyn Problem) -> Result<(), OptimizationError> {
        let dimension = self.bounds.lower.len();
        if dimension == 0 {
            return Err(OptimizationError::InvalidInput(
                "Problem must have at least one dimension".to_string(),
            ));
        }
        if problem.dimensions() != dimension {
            return Err(OptimizationError::DimensionMismatch(format!(
                "Problem has {} dimensions but bounds have {}",
                problem.dimensions(),
                dimension
            )));
        }
        if self.n_individuals < 4 {
            return Err(OptimizationError::InvalidInput(
                "n_individuals must be at least 4 for differential evolution".to_string(),
            ));
        }

        let mut rng = match self.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        let ranges = self.bounds.ranges();
        let mut population: Vec<Vec<f64>> = (0..self.n_individuals)
            .map(|_| random_position(&self.bounds, &ranges, &mut rng))
            .collect();
        let mut scores: Vec<f64> = population
            .iter()
            .map(|individual| problem.evaluate(individual))
            .collect();

        let best_index = scores
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(index, _)| index)
            .unwrap();
        let mut best_position = population[best_index].clone();
        let mut best_score = scores[best_index];

        for _ in 0..self.n_iterations {
            for target_index in 0..self.n_individuals {
                let pool: Vec<usize> = (0..self.n_individuals)
                    .filter(|&index| index != target_index)
                    .collect();
                let mut selected = Vec::new();
                while selected.len() < 3 {
                    let candidate = pool[rng.gen_range(0..pool.len())];
                    if !selected.contains(&candidate) {
                        selected.push(candidate);
                    }
                }
                let a_index = selected[0];
                let b_index = selected[1];
                let c_index = selected[2];

                let base = &population[a_index];
                let donor = &population[b_index];
                let difference = &population[c_index];
                let mutant: Vec<f64> = (0..dimension)
                    .map(|dimension_index| {
                        base[dimension_index]
                            + self.f * (donor[dimension_index] - difference[dimension_index])
                    })
                    .collect();

                let mut trial = population[target_index].clone();
                let crossover_index = rng.gen_range(0..dimension);
                for dimension_index in 0..dimension {
                    if rng.gen::<f64>() < self.cr || dimension_index == crossover_index {
                        trial[dimension_index] = mutant[dimension_index];
                    }
                    trial[dimension_index] = clamp_value(
                        trial[dimension_index],
                        self.bounds.lower[dimension_index],
                        self.bounds.upper[dimension_index],
                    );
                }

                let score = problem.evaluate(&trial);
                if score <= scores[target_index] {
                    population[target_index] = trial.clone();
                    scores[target_index] = score;
                    if score < best_score {
                        best_score = score;
                        best_position = trial;
                    }
                }
            }

            self.history.push(best_score);
        }

        self.population = population;
        self.best_solution = Some(Solution::with_fitness(best_position, best_score));
        Ok(())
    }

    fn predict(&self) -> Option<Self::Solution> {
        self.best_solution.clone()
    }

    fn score(&self) -> Option<f64> {
        self.best_solution
            .as_ref()
            .and_then(|solution| solution.fitness)
    }

    fn get_params(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("n_individuals".to_string(), self.n_individuals as f64);
        params.insert("n_iterations".to_string(), self.n_iterations as f64);
        params.insert("f".to_string(), self.f);
        params.insert("cr".to_string(), self.cr);
        params
    }
}
