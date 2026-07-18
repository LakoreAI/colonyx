use crate::algorithms::base::{make_rng, OptimizationError, Optimizer};
use crate::algorithms::continuous::two_opt;
use crate::core::{Bounds, Problem, Solution};
use rand::rngs::StdRng;
use rand::{seq::SliceRandom, Rng};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ParetoPoint {
    pub variables: Vec<f64>,
    pub objectives: Vec<f64>,
}

pub fn dominates(left: &[f64], right: &[f64]) -> bool {
    let mut strictly_better = false;
    for (a, b) in left.iter().zip(right.iter()) {
        if a > b {
            return false;
        }
        if a < b {
            strictly_better = true;
        }
    }
    strictly_better
}

pub fn non_dominated_sort(points: &[ParetoPoint]) -> Vec<Vec<usize>> {
    let mut domination_counts = vec![0usize; points.len()];
    let mut dominated_sets: Vec<Vec<usize>> = vec![Vec::new(); points.len()];
    let mut fronts: Vec<Vec<usize>> = Vec::new();

    for i in 0..points.len() {
        for j in 0..points.len() {
            if i == j {
                continue;
            }
            if dominates(&points[i].objectives, &points[j].objectives) {
                dominated_sets[i].push(j);
            } else if dominates(&points[j].objectives, &points[i].objectives) {
                domination_counts[i] += 1;
            }
        }
        if domination_counts[i] == 0 {
            if fronts.is_empty() {
                fronts.push(Vec::new());
            }
            fronts[0].push(i);
        }
    }

    let mut current = 0;
    while current < fronts.len() {
        let mut next_front = Vec::new();
        for &i in &fronts[current] {
            for &j in &dominated_sets[i] {
                domination_counts[j] -= 1;
                if domination_counts[j] == 0 {
                    next_front.push(j);
                }
            }
        }
        if !next_front.is_empty() {
            fronts.push(next_front);
        }
        current += 1;
    }

    fronts
}

pub fn crowding_distance(points: &[ParetoPoint], front: &[usize]) -> Vec<(usize, f64)> {
    if front.is_empty() {
        return Vec::new();
    }
    if front.len() == 1 {
        return vec![(front[0], f64::INFINITY)];
    }

    let objective_count = points[front[0]].objectives.len();
    let mut distances = vec![0.0; front.len()];

    for objective_index in 0..objective_count {
        let mut sorted = front.to_vec();
        sorted.sort_by(|&left, &right| {
            points[left].objectives[objective_index]
                .partial_cmp(&points[right].objectives[objective_index])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let min_value = points[*sorted.first().unwrap()].objectives[objective_index];
        let max_value = points[*sorted.last().unwrap()].objectives[objective_index];
        let span = (max_value - min_value).max(1e-12);
        distances[0] = f64::INFINITY;
        distances[front.len() - 1] = f64::INFINITY;
        for index in 1..sorted.len() - 1 {
            let previous = points[sorted[index - 1]].objectives[objective_index];
            let next = points[sorted[index + 1]].objectives[objective_index];
            let contribution = (next - previous) / span;
            if let Some(position) = front.iter().position(|candidate| *candidate == sorted[index]) {
                if distances[position].is_finite() {
                    distances[position] += contribution.abs();
                }
            }
        }
    }

    front
        .iter()
        .enumerate()
        .map(|(index, &candidate)| (candidate, distances[index]))
        .collect()
}

pub fn hypervolume_2d(points: &[ParetoPoint], reference: [f64; 2]) -> f64 {
    let mut front: Vec<[f64; 2]> = points
        .iter()
        .filter_map(|point| {
            if point.objectives.len() >= 2 {
                Some([point.objectives[0], point.objectives[1]])
            } else {
                None
            }
        })
        .collect();
    front.sort_by(|left, right| left[0].partial_cmp(&right[0]).unwrap_or(std::cmp::Ordering::Equal));

    let mut volume = 0.0;
    let mut best_y = reference[1];
    for point in front {
        if point[1] < best_y {
            volume += (reference[0] - point[0]).max(0.0) * (best_y - point[1]).max(0.0);
            best_y = point[1];
        }
    }
    volume
}

fn random_permutation(rng: &mut StdRng, size: usize) -> Vec<usize> {
    let mut values: Vec<usize> = (0..size).collect();
    values.shuffle(rng);
    values
}

fn tournament_pick(rng: &mut StdRng, population: &[Solution]) -> usize {
    let left = rng.gen_range(0..population.len());
    let right = rng.gen_range(0..population.len());
    let left_score = population[left].fitness.unwrap_or(f64::INFINITY);
    let right_score = population[right].fitness.unwrap_or(f64::INFINITY);
    if left_score <= right_score {
        left
    } else {
        right
    }
}

fn order_crossover(parent_a: &[usize], parent_b: &[usize], rng: &mut StdRng) -> Vec<usize> {
    let size = parent_a.len();
    let start = rng.gen_range(0..size);
    let end = rng.gen_range(start..size);
    let mut child = vec![usize::MAX; size];

    for index in start..=end {
        child[index] = parent_a[index];
    }

    if !child.contains(&usize::MAX) {
        return child;
    }

    let mut insert_index = (end + 1) % size;
    for value in parent_b.iter().copied().cycle() {
        if !child.contains(&value) {
            child[insert_index] = value;
            insert_index = (insert_index + 1) % size;
            if !child.contains(&usize::MAX) {
                break;
            }
        }
    }

    child
}

fn swap_mutation(tour: &mut [usize], rng: &mut StdRng, mutation_rate: f64) {
    if tour.len() < 2 || rng.gen::<f64>() > mutation_rate {
        return;
    }
    let left = rng.gen_range(0..tour.len());
    let right = rng.gen_range(0..tour.len());
    tour.swap(left, right);
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

#[derive(Debug)]
pub struct PermutationGeneticOptimizer {
    pub n_individuals: usize,
    pub n_iterations: usize,
    pub mutation_rate: f64,
    pub use_two_opt: bool,
    random_seed: Option<u64>,
    best_solution: Option<Solution>,
    pub history: Vec<f64>,
    pub population: Vec<Vec<f64>>,
}

impl PermutationGeneticOptimizer {
    pub fn new(
        n_individuals: usize,
        n_iterations: usize,
        mutation_rate: f64,
        use_two_opt: bool,
    ) -> Self {
        Self {
            n_individuals,
            n_iterations,
            mutation_rate,
            use_two_opt,
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

impl Optimizer for PermutationGeneticOptimizer {
    type Solution = Solution;

    fn fit(&mut self, problem: &dyn Problem) -> Result<(), OptimizationError> {
        self.history.clear();
        self.population.clear();
        let distances = problem.distance_matrix().ok_or_else(|| {
            OptimizationError::InvalidInput("GA requires a discrete distance matrix".to_string())
        })?;
        let n = distances.len();
        if n < 3 {
            return Err(OptimizationError::InvalidInput(
                "distance matrix must have at least 3 cities".to_string(),
            ));
        }

        let mut rng = make_rng(self.random_seed);

        let mut population: Vec<Solution> = (0..self.n_individuals)
            .map(|_| Solution::new(random_permutation(&mut rng, n).into_iter().map(|city| city as f64).collect()))
            .collect();

        for solution in &mut population {
            let mut fitness = problem.evaluate(&solution.variables);
            if self.use_two_opt && n > 3 {
                if let Ok((tour, length)) = two_opt(
                    &solution.variables.iter().map(|value| *value as usize).collect::<Vec<_>>(),
                    distances,
                ) {
                    solution.variables = tour.into_iter().map(|city| city as f64).collect();
                    fitness = length;
                }
            }
            solution.fitness = Some(fitness);
        }

        for _ in 0..self.n_iterations {
            population.sort_by(|left, right| {
                left.fitness
                    .unwrap_or(f64::INFINITY)
                    .partial_cmp(&right.fitness.unwrap_or(f64::INFINITY))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let best = population[0].clone();
            self.history.push(best.fitness.unwrap_or(f64::INFINITY));
            self.best_solution = Some(best.clone());

            let mut next_generation = vec![best];
            while next_generation.len() < self.n_individuals {
                let parent_a = tournament_pick(&mut rng, &population);
                let parent_b = tournament_pick(&mut rng, &population);
                let mut child_tour = order_crossover(
                    &population[parent_a]
                        .variables
                        .iter()
                        .map(|value| *value as usize)
                        .collect::<Vec<_>>(),
                    &population[parent_b]
                        .variables
                        .iter()
                        .map(|value| *value as usize)
                        .collect::<Vec<_>>(),
                    &mut rng,
                );
                swap_mutation(&mut child_tour, &mut rng, self.mutation_rate);
                let mut child = Solution::new(child_tour.into_iter().map(|city| city as f64).collect());
                let fitness = problem.evaluate(&child.variables);
                child.set_fitness(fitness);
                next_generation.push(child);
            }
            population = next_generation;
        }

        population.sort_by(|left, right| {
            left.fitness
                .unwrap_or(f64::INFINITY)
                .partial_cmp(&right.fitness.unwrap_or(f64::INFINITY))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.population = population
            .iter()
            .map(|solution| solution.variables.clone())
            .collect();
        self.best_solution = population.first().cloned();
        Ok(())
    }

    fn predict(&self) -> Option<Self::Solution> {
        self.best_solution.clone()
    }

    fn score(&self) -> Option<f64> {
        self.best_solution.as_ref().and_then(|solution| solution.fitness)
    }

    fn get_params(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("n_individuals".to_string(), self.n_individuals as f64);
        params.insert("n_iterations".to_string(), self.n_iterations as f64);
        params.insert("mutation_rate".to_string(), self.mutation_rate);
        params
    }
}

#[derive(Debug)]
pub struct BinaryParticleSwarm {
    pub n_particles: usize,
    pub n_iterations: usize,
    pub w: f64,
    pub c1: f64,
    pub c2: f64,
    random_seed: Option<u64>,
    pub best_solution: Option<Solution>,
    pub history: Vec<f64>,
    pub population: Vec<Vec<f64>>,
}

impl BinaryParticleSwarm {
    pub fn new(n_particles: usize, n_iterations: usize, w: f64, c1: f64, c2: f64) -> Self {
        Self {
            n_particles,
            n_iterations,
            w,
            c1,
            c2,
            random_seed: None,
            best_solution: None,
            history: Vec::new(),
            population: Vec::new(),
        }
    }

    pub fn set_random_seed(&mut self, seed: Option<u64>) {
        self.random_seed = seed;
    }

    pub fn fit_with_objective(
        &mut self,
        objective: &dyn Fn(&[f64]) -> f64,
        dimensions: usize,
    ) -> Result<(), OptimizationError> {
        self.history.clear();
        self.population.clear();
        if dimensions == 0 {
            return Err(OptimizationError::InvalidInput(
                "binary PSO requires at least one dimension".to_string(),
            ));
        }

        let mut rng = make_rng(self.random_seed);

        let mut positions: Vec<Vec<u8>> = (0..self.n_particles)
            .map(|_| {
                (0..dimensions)
                    .map(|_| if rng.gen::<f64>() < 0.5 { 1 } else { 0 })
                    .collect()
            })
            .collect();
        let mut velocities = vec![vec![0.0; dimensions]; self.n_particles];
        let mut personal_best = positions.clone();
        let mut personal_best_score = vec![f64::INFINITY; self.n_particles];
        let mut global_best = vec![0u8; dimensions];
        let mut global_best_score = f64::INFINITY;

        for index in 0..self.n_particles {
            let candidate: Vec<f64> = positions[index].iter().map(|value| *value as f64).collect();
            let score = objective(&candidate);
            personal_best_score[index] = score;
            if score < global_best_score {
                global_best_score = score;
                global_best = positions[index].clone();
            }
        }

        for _ in 0..self.n_iterations {
            for particle_index in 0..self.n_particles {
                for dimension in 0..dimensions {
                    let r1 = rng.gen::<f64>();
                    let r2 = rng.gen::<f64>();
                    velocities[particle_index][dimension] = self.w * velocities[particle_index][dimension]
                        + self.c1 * r1 * ((personal_best[particle_index][dimension] as f64) - (positions[particle_index][dimension] as f64))
                        + self.c2 * r2 * ((global_best[dimension] as f64) - (positions[particle_index][dimension] as f64));
                    let probability = sigmoid(velocities[particle_index][dimension]);
                    positions[particle_index][dimension] = if rng.gen::<f64>() < probability {
                        1
                    } else {
                        0
                    };
                }

                let candidate: Vec<f64> = positions[particle_index].iter().map(|value| *value as f64).collect();
                let score = objective(&candidate);
                if score < personal_best_score[particle_index] {
                    personal_best_score[particle_index] = score;
                    personal_best[particle_index] = positions[particle_index].clone();
                    if score < global_best_score {
                        global_best_score = score;
                        global_best = positions[particle_index].clone();
                    }
                }
            }
            self.history.push(global_best_score);
        }

        self.best_solution = Some(Solution::with_fitness(
            global_best.into_iter().map(|value| value as f64).collect(),
            global_best_score,
        ));
        self.population = positions
            .into_iter()
            .map(|position| position.into_iter().map(|value| value as f64).collect())
            .collect();
        Ok(())
    }
}

fn initialize_continuous_population(
    rng: &mut StdRng,
    bounds: &Bounds,
    size: usize,
) -> Vec<Vec<f64>> {
    let mut population = Vec::with_capacity(size);
    for _ in 0..size {
        let mut candidate = Vec::with_capacity(bounds.lower.len());
        for index in 0..bounds.lower.len() {
            let low = bounds.lower[index];
            let high = bounds.upper[index];
            candidate.push(low + rng.gen::<f64>() * (high - low));
        }
        population.push(candidate);
    }
    population
}

fn clamp_candidate(candidate: &mut [f64], bounds: &Bounds) {
    bounds.clamp(candidate);
}

fn evaluate_multi_objective(
    objective: &dyn Fn(&[f64]) -> Vec<f64>,
    population: &[Vec<f64>],
) -> Vec<ParetoPoint> {
    population
        .iter()
        .map(|candidate| ParetoPoint {
            variables: candidate.clone(),
            objectives: objective(candidate),
        })
        .collect()
}

fn archive_from_population(points: Vec<ParetoPoint>, archive_size: usize) -> Vec<ParetoPoint> {
    if points.is_empty() {
        return points;
    }
    let fronts = non_dominated_sort(&points);
    let mut archive = Vec::new();
    for front in fronts {
        if archive.len() + front.len() <= archive_size {
            archive.extend(front.into_iter().map(|index| points[index].clone()));
        } else {
            let distances = crowding_distance(&points, &front);
            let mut ranked = distances;
            ranked.sort_by(|left, right| {
                right
                    .1
                    .partial_cmp(&left.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for (index, _) in ranked {
                if archive.len() >= archive_size {
                    break;
                }
                archive.push(points[index].clone());
            }
            break;
        }
    }
    archive
}

fn select_archive_leader(rng: &mut StdRng, archive: &[ParetoPoint]) -> Vec<f64> {
    if archive.len() == 1 {
        return archive[0].variables.clone();
    }
    let fronts = non_dominated_sort(archive);
    let first_front = fronts.first().cloned().unwrap_or_default();
    let candidates = if first_front.is_empty() {
        archive.iter().map(|point| point.variables.clone()).collect::<Vec<_>>()
    } else {
        first_front
            .iter()
            .map(|index| archive[*index].variables.clone())
            .collect::<Vec<_>>()
    };
    candidates
        .choose(rng)
        .cloned()
        .unwrap_or_else(|| archive[0].variables.clone())
}

#[derive(Debug)]
pub struct Nsga2Optimizer {
    pub n_individuals: usize,
    pub n_iterations: usize,
    pub crossover_rate: f64,
    pub mutation_rate: f64,
    pub mutation_scale: f64,
    pub archive_size: usize,
    bounds: Bounds,
    random_seed: Option<u64>,
    pub population: Vec<Vec<f64>>,
    pub best_front: Vec<ParetoPoint>,
}

impl Nsga2Optimizer {
    pub fn new(
        n_individuals: usize,
        n_iterations: usize,
        crossover_rate: f64,
        mutation_rate: f64,
        mutation_scale: f64,
        archive_size: usize,
        bounds: Bounds,
    ) -> Self {
        Self {
            n_individuals,
            n_iterations,
            crossover_rate,
            mutation_rate,
            mutation_scale,
            archive_size,
            bounds,
            random_seed: None,
            population: Vec::new(),
            best_front: Vec::new(),
        }
    }

    pub fn set_random_seed(&mut self, seed: Option<u64>) {
        self.random_seed = seed;
    }

    pub fn fit_with_objective(
        &mut self,
        objective: &dyn Fn(&[f64]) -> Vec<f64>,
    ) -> Result<(), OptimizationError> {
        self.population.clear();
        let dim = self.bounds.lower.len();
        if dim == 0 {
            return Err(OptimizationError::InvalidInput("bounds must not be empty".to_string()));
        }
        let mut rng = make_rng(self.random_seed);

        let mut variables = initialize_continuous_population(&mut rng, &self.bounds, self.n_individuals);
        let mut evaluated = evaluate_multi_objective(objective, &variables);

        for _ in 0..self.n_iterations {
            let fronts = non_dominated_sort(&evaluated);
            let mut ranks = vec![usize::MAX; evaluated.len()];
            let mut crowding = vec![0.0; evaluated.len()];
            for (rank, front) in fronts.iter().enumerate() {
                for &index in front {
                    ranks[index] = rank;
                }
                for (index, distance) in crowding_distance(&evaluated, front) {
                    crowding[index] = distance;
                }
            }

            let tournament = |rng: &mut StdRng, ranks: &[usize], crowding: &[f64]| -> usize {
                let left = rng.gen_range(0..ranks.len());
                let right = rng.gen_range(0..ranks.len());
                let left_better = ranks[left] < ranks[right]
                    || (ranks[left] == ranks[right] && crowding[left] > crowding[right]);
                if left_better { left } else { right }
            };

            let mut offspring = Vec::with_capacity(self.n_individuals);
            while offspring.len() < self.n_individuals {
                let parent_a = &variables[tournament(&mut rng, &ranks, &crowding)];
                let parent_b = &variables[tournament(&mut rng, &ranks, &crowding)];
                let mut child = parent_a.clone();
                if rng.gen::<f64>() < self.crossover_rate {
                    let blend = rng.gen::<f64>();
                    for index in 0..dim {
                        child[index] = blend * parent_a[index] + (1.0 - blend) * parent_b[index];
                    }
                }
                for value in child.iter_mut() {
                    if rng.gen::<f64>() < self.mutation_rate {
                        let noise = (rng.gen::<f64>() * 2.0 - 1.0) * self.mutation_scale;
                        *value += noise;
                    }
                }
                clamp_candidate(&mut child, &self.bounds);
                offspring.push(child);
            }

            let offspring_evaluated = evaluate_multi_objective(objective, &offspring);
            let mut combined = evaluated.clone();
            combined.extend(offspring_evaluated);
            let combined_fronts = non_dominated_sort(&combined);

            let mut next_population = Vec::with_capacity(self.n_individuals);
            let mut next_evaluated = Vec::with_capacity(self.n_individuals);
            for front in combined_fronts {
                if next_population.len() + front.len() <= self.n_individuals {
                    for index in front {
                        next_population.push(combined[index].variables.clone());
                        next_evaluated.push(combined[index].clone());
                    }
                } else {
                    let distances = crowding_distance(&combined, &front);
                    let mut ranked = distances;
                    ranked.sort_by(|left, right| {
                        right
                            .1
                            .partial_cmp(&left.1)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    for (index, _) in ranked {
                        if next_population.len() >= self.n_individuals {
                            break;
                        }
                        next_population.push(combined[index].variables.clone());
                        next_evaluated.push(combined[index].clone());
                    }
                    break;
                }
            }
            variables = next_population;
            evaluated = next_evaluated;
        }

        self.population = variables;
        self.best_front = archive_from_population(evaluated.clone(), self.archive_size);
        Ok(())
    }
}

#[derive(Debug)]
pub struct MopsoOptimizer {
    pub n_particles: usize,
    pub n_iterations: usize,
    pub w: f64,
    pub c1: f64,
    pub c2: f64,
    pub mutation_scale: f64,
    pub archive_size: usize,
    bounds: Bounds,
    random_seed: Option<u64>,
    pub population: Vec<Vec<f64>>,
    pub archive: Vec<ParetoPoint>,
}

impl MopsoOptimizer {
    pub fn new(
        n_particles: usize,
        n_iterations: usize,
        w: f64,
        c1: f64,
        c2: f64,
        mutation_scale: f64,
        archive_size: usize,
        bounds: Bounds,
    ) -> Self {
        Self {
            n_particles,
            n_iterations,
            w,
            c1,
            c2,
            mutation_scale,
            archive_size,
            bounds,
            random_seed: None,
            population: Vec::new(),
            archive: Vec::new(),
        }
    }

    pub fn set_random_seed(&mut self, seed: Option<u64>) {
        self.random_seed = seed;
    }

    pub fn fit_with_objective(
        &mut self,
        objective: &dyn Fn(&[f64]) -> Vec<f64>,
    ) -> Result<(), OptimizationError> {
        self.population.clear();
        let dim = self.bounds.lower.len();
        if dim == 0 {
            return Err(OptimizationError::InvalidInput("bounds must not be empty".to_string()));
        }
        let mut rng = make_rng(self.random_seed);

        let mut positions = initialize_continuous_population(&mut rng, &self.bounds, self.n_particles);
        let mut velocities = vec![vec![0.0; dim]; self.n_particles];
        let mut personal_best = positions.clone();
        let mut personal_best_eval = evaluate_multi_objective(objective, &personal_best);
        self.archive = archive_from_population(personal_best_eval.clone(), self.archive_size);

        for _ in 0..self.n_iterations {
            for i in 0..self.n_particles {
                let leader = select_archive_leader(&mut rng, &self.archive);
                for d in 0..dim {
                    let r1 = rng.gen::<f64>();
                    let r2 = rng.gen::<f64>();
                    velocities[i][d] = self.w * velocities[i][d]
                        + self.c1 * r1 * (personal_best[i][d] - positions[i][d])
                        + self.c2 * r2 * (leader[d] - positions[i][d]);
                    positions[i][d] += velocities[i][d];
                }
                clamp_candidate(&mut positions[i], &self.bounds);

                let current = objective(&positions[i]);
                if dominates(&current, &personal_best_eval[i].objectives)
                {
                    personal_best[i] = positions[i].clone();
                    personal_best_eval[i] = ParetoPoint {
                        variables: positions[i].clone(),
                        objectives: current,
                    };
                }
            }

            let mut combined = self.archive.clone();
            combined.extend(personal_best_eval.clone());
            self.archive = archive_from_population(combined, self.archive_size);
        }

        self.population = positions;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::DiscreteProblem;

    fn make_ring4() -> DiscreteProblem {
        // 4-city ring: 0-1-2-3-0, each edge weight 1.0
        DiscreteProblem {
            name: "ring4".to_string(),
            distance_matrix: vec![
                vec![0.0, 1.0, 10.0, 1.0],
                vec![1.0, 0.0, 1.0, 10.0],
                vec![10.0, 1.0, 0.0, 1.0],
                vec![1.0, 10.0, 1.0, 0.0],
            ],
        }
    }

    // --- PermutationGeneticOptimizer ---

    #[test]
    fn pga_finds_optimal_tour() {
        let mut ga = PermutationGeneticOptimizer::new(30, 50, 0.1, false);
        ga.set_random_seed(Some(42));
        ga.fit(&make_ring4()).unwrap();
        // optimal tour length = 4 (0-1-2-3-0 or reverse)
        assert!(ga.score().unwrap() < 4.1);
    }

    #[test]
    fn pga_reproducible_with_seed() {
        let run = || {
            let mut ga = PermutationGeneticOptimizer::new(20, 30, 0.1, false);
            ga.set_random_seed(Some(7));
            ga.fit(&make_ring4()).unwrap();
            ga.score().unwrap()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn pga_rejects_small_matrix() {
        // matrix with 2 cities (below minimum of 3)
        let small = DiscreteProblem {
            name: "tiny".to_string(),
            distance_matrix: vec![vec![0.0, 1.0], vec![1.0, 0.0]],
        };
        let mut ga = PermutationGeneticOptimizer::new(10, 10, 0.1, false);
        assert!(ga.fit(&small).is_err());
    }

    #[test]
    fn pga_rejects_non_discrete() {
        let bounds = Bounds::uniform(2, -1.0, 1.0).unwrap();
        let mut ga = PermutationGeneticOptimizer::new(10, 10, 0.1, false);
        // feed a continuous problem -> no distance matrix -> error
        let contiguous = crate::core::ContinuousProblem::sphere(2, Some(bounds));
        assert!(ga.fit(&contiguous).is_err());
    }

    // --- BinaryParticleSwarm ---

    #[test]
    fn bpso_minimizes_onemax() {
        let mut bpso = BinaryParticleSwarm::new(20, 50, 0.7, 1.5, 1.5);
        bpso.set_random_seed(Some(42));
        // onemax: maximize sum -> minimize (dim - sum)
        bpso.fit_with_objective(&|x: &[f64]| {
            let sum: f64 = x.iter().sum();
            x.len() as f64 - sum
        }, 5).unwrap();
        let vars = bpso.best_solution.unwrap().variables;
        // all elements should be close to 1 (maximized sum)
        let sum: f64 = vars.iter().sum();
        assert!(sum > 4.0);
    }

    #[test]
    fn bpso_reproducible_with_seed() {
        let run = || {
            let mut bpso = BinaryParticleSwarm::new(15, 30, 0.7, 1.5, 1.5);
            bpso.set_random_seed(Some(7));
            bpso.fit_with_objective(&|x: &[f64]| x.iter().sum(), 3).unwrap();
            bpso.best_solution.unwrap().fitness.unwrap()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn bpso_rejects_zero_dimensions() {
        let mut bpso = BinaryParticleSwarm::new(10, 10, 0.7, 1.5, 1.5);
        assert!(bpso.fit_with_objective(&|_: &[f64]| 0.0, 0).is_err());
    }

    // --- Nsga2Optimizer ---

    #[test]
    fn nsga2_returns_pareto_front() {
        // minimize f1 = x, f2 = (x-1)^2
        let bounds = Bounds::uniform(1, -2.0, 3.0).unwrap();
        let mut nsga2 = Nsga2Optimizer::new(30, 30, 0.8, 0.2, 0.5, 30, bounds);
        nsga2.set_random_seed(Some(42));
        nsga2.fit_with_objective(&|x: &[f64]| vec![x[0], (x[0] - 1.0).powi(2)]).unwrap();
        assert!(!nsga2.best_front.is_empty());
        assert_eq!(nsga2.best_front[0].objectives.len(), 2);
    }

    #[test]
    fn nsga2_archive_respects_size() {
        let bounds = Bounds::uniform(1, -1.0, 1.0).unwrap();
        let mut nsga2 = Nsga2Optimizer::new(20, 10, 0.8, 0.2, 0.5, 5, bounds);
        nsga2.set_random_seed(Some(1));
        nsga2.fit_with_objective(&|x: &[f64]| vec![x[0], x[0] * x[0]]).unwrap();
        assert!(nsga2.best_front.len() <= 5);
    }

    // --- MopsoOptimizer ---

    #[test]
    fn mopso_returns_archive() {
        let bounds = Bounds::uniform(1, -2.0, 3.0).unwrap();
        let mut mopso = MopsoOptimizer::new(20, 30, 0.7, 1.5, 1.5, 0.1, 20, bounds);
        mopso.set_random_seed(Some(42));
        mopso.fit_with_objective(&|x: &[f64]| vec![x[0], (x[0] - 1.0).powi(2)]).unwrap();
        assert!(!mopso.archive.is_empty());
        assert_eq!(mopso.archive[0].objectives.len(), 2);
    }

    #[test]
    fn mopso_archive_respects_size() {
        let bounds = Bounds::uniform(1, -1.0, 1.0).unwrap();
        let mut mopso = MopsoOptimizer::new(15, 10, 0.7, 1.5, 1.5, 0.1, 5, bounds);
        mopso.set_random_seed(Some(1));
        mopso.fit_with_objective(&|x: &[f64]| vec![x[0], x[0] * x[0]]).unwrap();
        assert!(mopso.archive.len() <= 5);
    }

    // --- helper tests ---

    #[test]
    fn dominates_returns_true_when_strictly_better() {
        assert!(dominates(&[1.0, 2.0], &[3.0, 4.0]));
        assert!(dominates(&[1.0, 2.0], &[1.0, 4.0]));
    }

    #[test]
    fn dominates_returns_false_when_equal_or_worse() {
        assert!(!dominates(&[1.0, 2.0], &[1.0, 2.0]));
        assert!(!dominates(&[3.0, 2.0], &[1.0, 2.0]));
    }

    #[test]
    fn non_dominated_sort_produces_fronts() {
        let points = vec![
            ParetoPoint { variables: vec![0.0], objectives: vec![1.0, 5.0] },
            ParetoPoint { variables: vec![1.0], objectives: vec![2.0, 3.0] },
            ParetoPoint { variables: vec![2.0], objectives: vec![3.0, 1.0] },
            ParetoPoint { variables: vec![3.0], objectives: vec![4.0, 4.0] },
        ];
        let fronts = non_dominated_sort(&points);
        // Point 3 (4,4) is dominated by both 1 and 2.
        // Points 0 (1,5), 1 (2,3), 2 (3,1) are all non-dominated (incomparable).
        assert_eq!(fronts.len(), 2);
        // first front: indices of non-dominated points -> {0, 1, 2}
        for &index in &fronts[0] {
            assert!(index == 0 || index == 1 || index == 2);
        }
    }

    #[test]
    fn hypervolume_2d_computes_non_negative() {
        let points = vec![
            ParetoPoint { variables: vec![0.0], objectives: vec![1.0, 2.0] },
            ParetoPoint { variables: vec![1.0], objectives: vec![2.0, 1.0] },
        ];
        let vol = hypervolume_2d(&points, [3.0, 3.0]);
        assert!(vol > 0.0);
    }

    #[test]
    fn crowding_distance_assigns_infinity_to_ends() {
        let points = vec![
            ParetoPoint { variables: vec![0.0], objectives: vec![0.0, 3.0] },
            ParetoPoint { variables: vec![1.0], objectives: vec![1.0, 2.0] },
            ParetoPoint { variables: vec![2.0], objectives: vec![2.0, 1.0] },
            ParetoPoint { variables: vec![3.0], objectives: vec![3.0, 0.0] },
        ];
        let front = vec![0, 1, 2, 3];
        let distances = crowding_distance(&points, &front);
        assert_eq!(distances[0].1, f64::INFINITY);
        assert_eq!(distances[3].1, f64::INFINITY);
    }
}
