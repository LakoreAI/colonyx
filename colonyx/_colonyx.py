"""Pure-Python fallback for the compiled ``colonyx._colonyx`` extension."""

from __future__ import annotations

import math
import random
from numbers import Real
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence


def _ensure_square_matrix(distance_matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    matrix = [list(map(float, row)) for row in distance_matrix]
    if not matrix:
        raise ValueError("distance_matrix must be non-empty")
    size = len(matrix)
    for index, row in enumerate(matrix):
        if len(row) != size:
            raise ValueError(
                f"distance_matrix must be square: row {index} has length {len(row)}, expected {size}"
            )
    return matrix


def _ensure_bounds(lower: Sequence[float], upper: Sequence[float]) -> tuple[list[float], list[float]]:
    lower_values = [float(value) for value in lower]
    upper_values = [float(value) for value in upper]
    if len(lower_values) != len(upper_values):
        raise ValueError("Lower and upper bounds must have the same length")
    for index, (low, high) in enumerate(zip(lower_values, upper_values)):
        if low > high:
            raise ValueError(f"Lower bound {low} > upper bound {high} at index {index}")
    if not lower_values:
        raise ValueError("Bounds must have at least one dimension")
    return lower_values, upper_values


def _objective_value(objective: Callable[[List[float]], float], point: Sequence[float]) -> float:
    value = objective(list(point))
    if isinstance(value, Real):
        return float(value)
    raise TypeError("objective function must return a numeric value")


@dataclass
class AntColony:
    n_ants: int = 50
    n_iterations: int = 100
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.5
    q: float = 1.0
    random_state: Optional[int] = None
    use_two_opt: bool = True
    best_tour: Optional[List[int]] = None
    best_length: Optional[float] = None
    pheromone_matrix: Optional[List[List[float]]] = field(default=None, repr=False)
    history_: List[float] = field(default_factory=list, repr=False)
    population_: List[List[int]] = field(default_factory=list, repr=False)

    def fit(self, distance_matrix: Sequence[Sequence[float]]) -> None:
        matrix = _ensure_square_matrix(distance_matrix)
        size = len(matrix)
        rng = random.Random(self.random_state)
        initial_pheromone = 1.0 / size
        self.pheromone_matrix = [[initial_pheromone for _ in range(size)] for _ in range(size)]
        best_tour: Optional[List[int]] = None
        best_length = math.inf

        for _ in range(self.n_iterations):
            solutions: list[tuple[list[int], float]] = []
            for _ in range(self.n_ants):
                tour = self._construct_solution(matrix, rng)
                if self.use_two_opt and len(tour) > 3:
                    tour, _ = two_opt(tour, matrix)
                length = self._tour_length(matrix, tour)
                solutions.append((tour, length))
                if length < best_length:
                    best_length = length
                    best_tour = tour[:]
            self._update_pheromones(solutions)
            self.history_.append(best_length)
            self.population_ = [tour[:] for tour, _ in solutions]

        self.best_tour = best_tour
        self.best_length = best_length

    def _construct_solution(self, matrix: list[list[float]], rng: random.Random) -> list[int]:
        size = len(matrix)
        if self.pheromone_matrix is None:
            return list(range(size))

        visited = [False] * size
        tour = [rng.randrange(size)]
        visited[tour[0]] = True

        while len(tour) < size:
            current = tour[-1]
            candidates: list[tuple[int, float]] = []
            total = 0.0
            for city in range(size):
                if visited[city]:
                    continue
                tau = self.pheromone_matrix[current][city] ** self.alpha
                distance = max(matrix[current][city], 1e-10)
                eta = (1.0 / distance) ** self.beta
                weight = tau * eta
                candidates.append((city, weight))
                total += weight
            if not candidates:
                break
            if total <= 0.0 or not math.isfinite(total):
                next_city = rng.choice([city for city, _ in candidates])
            else:
                threshold = rng.random() * total
                accumulator = 0.0
                next_city = candidates[-1][0]
                for city, weight in candidates:
                    accumulator += weight
                    if accumulator >= threshold:
                        next_city = city
                        break
            tour.append(next_city)
            visited[next_city] = True

        return tour

    def _tour_length(self, matrix: list[list[float]], tour: Sequence[int]) -> float:
        total = 0.0
        size = len(tour)
        for index in range(size):
            start = tour[index]
            end = tour[(index + 1) % size]
            total += matrix[start][end]
        return total

    def _update_pheromones(self, solutions: list[tuple[list[int], float]]) -> None:
        if self.pheromone_matrix is None:
            return
        size = len(self.pheromone_matrix)
        for row in range(size):
            for column in range(size):
                self.pheromone_matrix[row][column] *= 1.0 - self.rho
        for tour, length in solutions:
            if length <= 0.0 or not math.isfinite(length):
                continue
            deposit = self.q / length
            for index in range(len(tour)):
                start = tour[index]
                end = tour[(index + 1) % len(tour)]
                self.pheromone_matrix[start][end] += deposit
                self.pheromone_matrix[end][start] += deposit

    def predict(self) -> List[int]:
        if self.best_tour is None:
            raise ValueError("must call fit() before predict()")
        return self.best_tour[:]

    def score(self) -> float:
        if self.best_length is None:
            raise ValueError("must call fit() before score()")
        return float(self.best_length)

    def get_params(self) -> dict[str, float]:
        return {
            "n_ants": float(self.n_ants),
            "n_iterations": float(self.n_iterations),
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "rho": float(self.rho),
            "q": float(self.q),
        }


def _tour_length_from_matrix(matrix: Sequence[Sequence[float]], tour: Sequence[int]) -> float:
    total = 0.0
    size = len(tour)
    for index in range(size):
        start = tour[index]
        end = tour[(index + 1) % size]
        total += float(matrix[start][end])
    return float(total)


def two_opt(tour: Sequence[int], distance_matrix: Sequence[Sequence[float]]) -> tuple[list[int], float]:
    """Improve a TSP tour with 2-opt local search."""
    matrix = _ensure_square_matrix(distance_matrix)
    route = list(tour)
    if len(route) < 4:
        return route, _tour_length_from_matrix(matrix, route)

    best_route = route[:]
    best_length = _tour_length_from_matrix(matrix, best_route)
    improved = True

    while improved:
        improved = False
        for start in range(1, len(best_route) - 2):
            for end in range(start + 2, len(best_route) + 1):
                candidate = best_route[:]
                candidate[start:end] = reversed(best_route[start:end])
                candidate_length = _tour_length_from_matrix(matrix, candidate)
                if candidate_length + 1e-12 < best_length:
                    best_route = candidate
                    best_length = candidate_length
                    improved = True
                    break
            if improved:
                break

    return best_route, float(best_length)


@dataclass
class ParticleSwarm:
    n_particles: int = 30
    n_iterations: int = 100
    w: float = 0.9
    c1: float = 2.0
    c2: float = 2.0
    random_state: Optional[int] = None
    best_position: Optional[List[float]] = None
    best_score: Optional[float] = None
    history_: List[float] = field(default_factory=list, repr=False)
    population_: List[List[float]] = field(default_factory=list, repr=False)

    def fit(
        self,
        objective: Callable[[List[float]], float],
        lower: Sequence[float],
        upper: Sequence[float],
    ) -> None:
        lower_values, upper_values = _ensure_bounds(lower, upper)
        dimension = len(lower_values)
        rng = random.Random(self.random_state)
        ranges = [high - low for low, high in zip(lower_values, upper_values)]

        positions = [[0.0] * dimension for _ in range(self.n_particles)]
        velocities = [[0.0] * dimension for _ in range(self.n_particles)]
        personal_best = [[0.0] * dimension for _ in range(self.n_particles)]
        personal_best_scores = [math.inf] * self.n_particles
        global_best = [0.0] * dimension
        global_best_score = math.inf

        probe = [(low + high) / 2.0 for low, high in zip(lower_values, upper_values)]
        _objective_value(objective, probe)

        for particle_index in range(self.n_particles):
            for dimension_index in range(dimension):
                positions[particle_index][dimension_index] = (
                    lower_values[dimension_index] + rng.random() * ranges[dimension_index]
                )
                velocities[particle_index][dimension_index] = (
                    (rng.random() * 2.0 - 1.0) * ranges[dimension_index] * 0.1
                )
            score = _objective_value(objective, positions[particle_index])
            personal_best[particle_index] = positions[particle_index][:]
            personal_best_scores[particle_index] = score
            if score < global_best_score:
                global_best_score = score
                global_best = positions[particle_index][:]

        for _ in range(self.n_iterations):
            for particle_index in range(self.n_particles):
                for dimension_index in range(dimension):
                    r1 = rng.random()
                    r2 = rng.random()
                    velocities[particle_index][dimension_index] = (
                        self.w * velocities[particle_index][dimension_index]
                        + self.c1
                        * r1
                        * (personal_best[particle_index][dimension_index] - positions[particle_index][dimension_index])
                        + self.c2
                        * r2
                        * (global_best[dimension_index] - positions[particle_index][dimension_index])
                    )
                    positions[particle_index][dimension_index] += velocities[particle_index][dimension_index]
                    positions[particle_index][dimension_index] = min(
                        max(positions[particle_index][dimension_index], lower_values[dimension_index]),
                        upper_values[dimension_index],
                    )

                score = _objective_value(objective, positions[particle_index])
                if score < personal_best_scores[particle_index]:
                    personal_best_scores[particle_index] = score
                    personal_best[particle_index] = positions[particle_index][:]
                    if score < global_best_score:
                        global_best_score = score
                        global_best = positions[particle_index][:]
            self.history_.append(global_best_score)

        self.best_position = global_best
        self.best_score = global_best_score
        self.population_ = [position[:] for position in positions]

    def predict(self) -> List[float]:
        if self.best_position is None:
            raise ValueError("must call fit() before predict()")
        return self.best_position[:]

    def score(self) -> float:
        if self.best_score is None:
            raise ValueError("must call fit() before score()")
        return float(self.best_score)

    def get_params(self) -> dict[str, float]:
        return {
            "n_particles": float(self.n_particles),
            "n_iterations": float(self.n_iterations),
            "w": float(self.w),
            "c1": float(self.c1),
            "c2": float(self.c2),
        }


@dataclass
class GreyWolfOptimizer:
    n_wolves: int = 30
    n_iterations: int = 100
    random_state: Optional[int] = None
    best_position: Optional[List[float]] = None
    best_score: Optional[float] = None
    history_: List[float] = field(default_factory=list, repr=False)
    population_: List[List[float]] = field(default_factory=list, repr=False)

    def fit(
        self,
        objective: Callable[[List[float]], float],
        lower: Sequence[float],
        upper: Sequence[float],
    ) -> None:
        lower_values, upper_values = _ensure_bounds(lower, upper)
        dimension = len(lower_values)
        if self.n_wolves < 3:
            raise ValueError("n_wolves must be at least 3 for grey wolf optimization")

        rng = random.Random(self.random_state)
        ranges = [high - low for low, high in zip(lower_values, upper_values)]
        wolves = [
            [lower_values[index] + rng.random() * ranges[index] for index in range(dimension)]
            for _ in range(self.n_wolves)
        ]
        scores = [_objective_value(objective, wolf) for wolf in wolves]

        best_index = min(range(self.n_wolves), key=lambda index: scores[index])
        best_position = wolves[best_index][:]
        best_score = scores[best_index]

        for iteration in range(self.n_iterations):
            ranking = sorted(range(self.n_wolves), key=lambda index: scores[index])
            alpha = wolves[ranking[0]][:]
            beta = wolves[ranking[1]][:]
            delta = wolves[ranking[2]][:]
            a_value = 2.0 * (1.0 - iteration / max(self.n_iterations - 1, 1))

            for wolf_index in range(self.n_wolves):
                candidate = wolves[wolf_index][:]
                for dimension_index in range(dimension):
                    r1 = rng.random()
                    r2 = rng.random()
                    a1 = 2.0 * a_value * r1 - a_value
                    c1 = 2.0 * r2
                    d_alpha = abs(c1 * alpha[dimension_index] - candidate[dimension_index])
                    x1 = alpha[dimension_index] - a1 * d_alpha

                    r1 = rng.random()
                    r2 = rng.random()
                    a2 = 2.0 * a_value * r1 - a_value
                    c2 = 2.0 * r2
                    d_beta = abs(c2 * beta[dimension_index] - candidate[dimension_index])
                    x2 = beta[dimension_index] - a2 * d_beta

                    r1 = rng.random()
                    r2 = rng.random()
                    a3 = 2.0 * a_value * r1 - a_value
                    c3 = 2.0 * r2
                    d_delta = abs(c3 * delta[dimension_index] - candidate[dimension_index])
                    x3 = delta[dimension_index] - a3 * d_delta

                    candidate[dimension_index] = (x1 + x2 + x3) / 3.0
                    candidate[dimension_index] = min(
                        max(candidate[dimension_index], lower_values[dimension_index]),
                        upper_values[dimension_index],
                    )

                score = _objective_value(objective, candidate)
                wolves[wolf_index] = candidate
                scores[wolf_index] = score
                if score < best_score:
                    best_score = score
                    best_position = candidate[:]

            self.history_.append(best_score)

        self.best_position = best_position
        self.best_score = best_score
        self.population_ = [wolf[:] for wolf in wolves]

    def predict(self) -> List[float]:
        if self.best_position is None:
            raise ValueError("must call fit() before predict()")
        return self.best_position[:]

    def score(self) -> float:
        if self.best_score is None:
            raise ValueError("must call fit() before score()")
        return float(self.best_score)

    def get_params(self) -> dict[str, float]:
        return {
            "n_wolves": float(self.n_wolves),
            "n_iterations": float(self.n_iterations),
        }


@dataclass
class FireflyOptimizer:
    n_fireflies: int = 30
    n_iterations: int = 100
    beta0: float = 1.0
    gamma: float = 1.0
    alpha: float = 0.2
    random_state: Optional[int] = None
    best_position: Optional[List[float]] = None
    best_score: Optional[float] = None
    history_: List[float] = field(default_factory=list, repr=False)
    population_: List[List[float]] = field(default_factory=list, repr=False)

    def fit(
        self,
        objective: Callable[[List[float]], float],
        lower: Sequence[float],
        upper: Sequence[float],
    ) -> None:
        lower_values, upper_values = _ensure_bounds(lower, upper)
        dimension = len(lower_values)
        if self.n_fireflies < 2:
            raise ValueError("n_fireflies must be at least 2 for firefly optimization")

        rng = random.Random(self.random_state)
        ranges = [high - low for low, high in zip(lower_values, upper_values)]
        fireflies = [
            [lower_values[index] + rng.random() * ranges[index] for index in range(dimension)]
            for _ in range(self.n_fireflies)
        ]
        scores = [_objective_value(objective, firefly) for firefly in fireflies]

        best_index = min(range(self.n_fireflies), key=lambda index: scores[index])
        best_position = fireflies[best_index][:]
        best_score = scores[best_index]

        for _ in range(self.n_iterations):
            for first_index in range(self.n_fireflies):
                for second_index in range(self.n_fireflies):
                    if scores[second_index] >= scores[first_index]:
                        continue
                    distance = math.sqrt(
                        sum(
                            (fireflies[first_index][dimension_index] - fireflies[second_index][dimension_index]) ** 2
                            for dimension_index in range(dimension)
                        )
                    )
                    beta = self.beta0 * math.exp(-self.gamma * distance * distance)
                    candidate = fireflies[first_index][:]
                    for dimension_index in range(dimension):
                        random_step = self.alpha * (rng.random() - 0.5) * ranges[dimension_index]
                        candidate[dimension_index] += beta * (
                            fireflies[second_index][dimension_index] - candidate[dimension_index]
                        ) + random_step
                        candidate[dimension_index] = min(
                            max(candidate[dimension_index], lower_values[dimension_index]),
                            upper_values[dimension_index],
                        )
                    candidate_score = _objective_value(objective, candidate)
                    if candidate_score < scores[first_index]:
                        fireflies[first_index] = candidate
                        scores[first_index] = candidate_score
                        if candidate_score < best_score:
                            best_score = candidate_score
                            best_position = candidate[:]

            self.history_.append(best_score)

        self.best_position = best_position
        self.best_score = best_score
        self.population_ = [firefly[:] for firefly in fireflies]

    def predict(self) -> List[float]:
        if self.best_position is None:
            raise ValueError("must call fit() before predict()")
        return self.best_position[:]

    def score(self) -> float:
        if self.best_score is None:
            raise ValueError("must call fit() before score()")
        return float(self.best_score)

    def get_params(self) -> dict[str, float]:
        return {
            "n_fireflies": float(self.n_fireflies),
            "n_iterations": float(self.n_iterations),
            "beta0": float(self.beta0),
            "gamma": float(self.gamma),
            "alpha": float(self.alpha),
        }


@dataclass
class SimulatedAnnealing:
    initial_temperature: float = 10.0
    cooling_rate: float = 0.95
    step_scale: float = 0.1
    n_iterations: int = 100
    random_state: Optional[int] = None
    best_position: Optional[List[float]] = None
    best_score: Optional[float] = None
    history_: List[float] = field(default_factory=list, repr=False)
    population_: List[List[float]] = field(default_factory=list, repr=False)

    def fit(
        self,
        objective: Callable[[List[float]], float],
        lower: Sequence[float],
        upper: Sequence[float],
    ) -> None:
        lower_values, upper_values = _ensure_bounds(lower, upper)
        dimension = len(lower_values)
        rng = random.Random(self.random_state)
        ranges = [high - low for low, high in zip(lower_values, upper_values)]

        current = [lower_values[index] + rng.random() * ranges[index] for index in range(dimension)]
        current_score = _objective_value(objective, current)
        best_position = current[:]
        best_score = current_score
        temperature = float(self.initial_temperature)

        for _ in range(self.n_iterations):
            candidate = current[:]
            for dimension_index in range(dimension):
                step = (rng.random() * 2.0 - 1.0) * self.step_scale * ranges[dimension_index]
                candidate[dimension_index] += step
                candidate[dimension_index] = min(
                    max(candidate[dimension_index], lower_values[dimension_index]),
                    upper_values[dimension_index],
                )

            candidate_score = _objective_value(objective, candidate)
            delta = candidate_score - current_score
            if delta <= 0.0 or rng.random() < math.exp(-delta / max(temperature, 1e-12)):
                current = candidate
                current_score = candidate_score

            if current_score < best_score:
                best_score = current_score
                best_position = current[:]

            self.history_.append(best_score)
            temperature *= self.cooling_rate

        self.best_position = best_position
        self.best_score = best_score
        self.population_ = [current[:]]

    def predict(self) -> List[float]:
        if self.best_position is None:
            raise ValueError("must call fit() before predict()")
        return self.best_position[:]

    def score(self) -> float:
        if self.best_score is None:
            raise ValueError("must call fit() before score()")
        return float(self.best_score)

    def get_params(self) -> dict[str, float]:
        return {
            "initial_temperature": float(self.initial_temperature),
            "cooling_rate": float(self.cooling_rate),
            "step_scale": float(self.step_scale),
            "n_iterations": float(self.n_iterations),
        }


@dataclass
class CuckooSearch:
    n_nests: int = 25
    n_iterations: int = 100
    pa: float = 0.25
    alpha: float = 0.01
    levy_scale: float = 1.0
    random_state: Optional[int] = None
    best_position: Optional[List[float]] = None
    best_score: Optional[float] = None
    history_: List[float] = field(default_factory=list, repr=False)
    population_: List[List[float]] = field(default_factory=list, repr=False)

    def fit(
        self,
        objective: Callable[[List[float]], float],
        lower: Sequence[float],
        upper: Sequence[float],
    ) -> None:
        lower_values, upper_values = _ensure_bounds(lower, upper)
        dimension = len(lower_values)
        if self.n_nests < 2:
            raise ValueError("n_nests must be at least 2 for cuckoo search")

        rng = random.Random(self.random_state)
        ranges = [high - low for low, high in zip(lower_values, upper_values)]
        nests = [
            [lower_values[index] + rng.random() * ranges[index] for index in range(dimension)]
            for _ in range(self.n_nests)
        ]
        scores = [_objective_value(objective, nest) for nest in nests]

        best_index = min(range(self.n_nests), key=lambda index: scores[index])
        best_position = nests[best_index][:]
        best_score = scores[best_index]

        def levy_step() -> float:
            u = rng.random() - 0.5
            v = rng.random() - 0.5
            denominator = max(abs(v), 1e-12) ** (1.0 / 1.5)
            return self.levy_scale * (u / denominator)

        for _ in range(self.n_iterations):
            for nest_index in range(self.n_nests):
                candidate = nests[nest_index][:]
                step = levy_step()
                for dimension_index in range(dimension):
                    candidate[dimension_index] += self.alpha * step * ranges[dimension_index]
                    candidate[dimension_index] = min(
                        max(candidate[dimension_index], lower_values[dimension_index]),
                        upper_values[dimension_index],
                    )
                candidate_score = _objective_value(objective, candidate)
                if candidate_score < scores[nest_index]:
                    nests[nest_index] = candidate
                    scores[nest_index] = candidate_score

            abandon_count = max(1, int(self.pa * self.n_nests))
            worst_indices = sorted(range(self.n_nests), key=lambda index: scores[index], reverse=True)[:abandon_count]
            for nest_index in worst_indices:
                nests[nest_index] = [
                    lower_values[dimension_index] + rng.random() * ranges[dimension_index]
                    for dimension_index in range(dimension)
                ]
                scores[nest_index] = _objective_value(objective, nests[nest_index])

            best_index = min(range(self.n_nests), key=lambda index: scores[index])
            if scores[best_index] < best_score:
                best_score = scores[best_index]
                best_position = nests[best_index][:]

            self.history_.append(best_score)

        self.best_position = best_position
        self.best_score = best_score
        self.population_ = [nest[:] for nest in nests]

    def predict(self) -> List[float]:
        if self.best_position is None:
            raise ValueError("must call fit() before predict()")
        return self.best_position[:]

    def score(self) -> float:
        if self.best_score is None:
            raise ValueError("must call fit() before score()")
        return float(self.best_score)

    def get_params(self) -> dict[str, float]:
        return {
            "n_nests": float(self.n_nests),
            "n_iterations": float(self.n_iterations),
            "pa": float(self.pa),
            "alpha": float(self.alpha),
            "levy_scale": float(self.levy_scale),
        }


@dataclass
class BatAlgorithm:
    n_bats: int = 30
    n_iterations: int = 100
    fmin: float = 0.0
    fmax: float = 2.0
    alpha: float = 0.9
    gamma: float = 0.9
    loudness: float = 1.0
    pulse_rate: float = 0.5
    random_state: Optional[int] = None
    best_position: Optional[List[float]] = None
    best_score: Optional[float] = None
    history_: List[float] = field(default_factory=list, repr=False)
    population_: List[List[float]] = field(default_factory=list, repr=False)

    def fit(
        self,
        objective: Callable[[List[float]], float],
        lower: Sequence[float],
        upper: Sequence[float],
    ) -> None:
        lower_values, upper_values = _ensure_bounds(lower, upper)
        dimension = len(lower_values)
        if self.n_bats < 2:
            raise ValueError("n_bats must be at least 2 for bat algorithm")

        rng = random.Random(self.random_state)
        ranges = [high - low for low, high in zip(lower_values, upper_values)]
        positions = [
            [lower_values[index] + rng.random() * ranges[index] for index in range(dimension)]
            for _ in range(self.n_bats)
        ]
        velocities = [[0.0] * dimension for _ in range(self.n_bats)]
        frequencies = [0.0] * self.n_bats
        scores = [_objective_value(objective, position) for position in positions]
        loudness = [float(self.loudness)] * self.n_bats
        pulse = [float(self.pulse_rate)] * self.n_bats

        best_index = min(range(self.n_bats), key=lambda index: scores[index])
        best_position = positions[best_index][:]
        best_score = scores[best_index]

        for iteration in range(self.n_iterations):
            for bat_index in range(self.n_bats):
                frequencies[bat_index] = self.fmin + (self.fmax - self.fmin) * rng.random()
                for dimension_index in range(dimension):
                    velocities[bat_index][dimension_index] += (
                        positions[bat_index][dimension_index] - best_position[dimension_index]
                    ) * frequencies[bat_index]
                    positions[bat_index][dimension_index] += velocities[bat_index][dimension_index]
                    positions[bat_index][dimension_index] = min(
                        max(positions[bat_index][dimension_index], lower_values[dimension_index]),
                        upper_values[dimension_index],
                    )

                if rng.random() > pulse[bat_index]:
                    for dimension_index in range(dimension):
                        positions[bat_index][dimension_index] = best_position[dimension_index] + 0.001 * rng.gauss(0.0, 1.0)
                        positions[bat_index][dimension_index] = min(
                            max(positions[bat_index][dimension_index], lower_values[dimension_index]),
                            upper_values[dimension_index],
                        )

                candidate_score = _objective_value(objective, positions[bat_index])
                if candidate_score <= scores[bat_index] and rng.random() < loudness[bat_index]:
                    scores[bat_index] = candidate_score
                    loudness[bat_index] *= self.alpha
                    pulse[bat_index] = pulse[bat_index] * (1.0 - math.exp(-self.gamma * (iteration + 1)))
                    if candidate_score < best_score:
                        best_score = candidate_score
                        best_position = positions[bat_index][:]

            self.history_.append(best_score)

        self.best_position = best_position
        self.best_score = best_score
        self.population_ = [position[:] for position in positions]

    def predict(self) -> List[float]:
        if self.best_position is None:
            raise ValueError("must call fit() before predict()")
        return self.best_position[:]

    def score(self) -> float:
        if self.best_score is None:
            raise ValueError("must call fit() before score()")
        return float(self.best_score)

    def get_params(self) -> dict[str, float]:
        return {
            "n_bats": float(self.n_bats),
            "n_iterations": float(self.n_iterations),
            "fmin": float(self.fmin),
            "fmax": float(self.fmax),
            "alpha": float(self.alpha),
            "gamma": float(self.gamma),
            "loudness": float(self.loudness),
            "pulse_rate": float(self.pulse_rate),
        }


@dataclass
class GlowwormOptimizer:
    n_worms: int = 30
    n_iterations: int = 100
    luciferin_decay: float = 0.4
    luciferin_enhancement: float = 0.6
    step_size: float = 0.1
    neighborhood_radius: float = 1.0
    random_state: Optional[int] = None
    best_position: Optional[List[float]] = None
    best_score: Optional[float] = None
    history_: List[float] = field(default_factory=list, repr=False)
    population_: List[List[float]] = field(default_factory=list, repr=False)

    def fit(
        self,
        objective: Callable[[List[float]], float],
        lower: Sequence[float],
        upper: Sequence[float],
    ) -> None:
        lower_values, upper_values = _ensure_bounds(lower, upper)
        dimension = len(lower_values)
        if self.n_worms < 2:
            raise ValueError("n_worms must be at least 2 for glowworm optimization")

        rng = random.Random(self.random_state)
        ranges = [high - low for low, high in zip(lower_values, upper_values)]
        worms = [
            [lower_values[index] + rng.random() * ranges[index] for index in range(dimension)]
            for _ in range(self.n_worms)
        ]
        luciferin = [1.0] * self.n_worms
        scores = [_objective_value(objective, worm) for worm in worms]

        best_index = min(range(self.n_worms), key=lambda index: scores[index])
        best_position = worms[best_index][:]
        best_score = scores[best_index]

        for _ in range(self.n_iterations):
            luciferin = [
                (1.0 - self.luciferin_decay) * luciferin[index] + self.luciferin_enhancement / max(scores[index] + 1.0, 1e-12)
                for index in range(self.n_worms)
            ]

            for worm_index in range(self.n_worms):
                neighbors = [
                    neighbor_index
                    for neighbor_index in range(self.n_worms)
                    if neighbor_index != worm_index
                    and luciferin[neighbor_index] > luciferin[worm_index]
                    and math.sqrt(
                        sum(
                            (worms[worm_index][dimension_index] - worms[neighbor_index][dimension_index]) ** 2
                            for dimension_index in range(dimension)
                        )
                    ) <= self.neighborhood_radius
                ]

                if neighbors:
                    target_index = min(neighbors, key=lambda index: scores[index])
                    candidate = worms[worm_index][:]
                    for dimension_index in range(dimension):
                        direction = worms[target_index][dimension_index] - candidate[dimension_index]
                        candidate[dimension_index] += self.step_size * direction
                        candidate[dimension_index] = min(
                            max(candidate[dimension_index], lower_values[dimension_index]),
                            upper_values[dimension_index],
                        )
                    candidate_score = _objective_value(objective, candidate)
                    if candidate_score < scores[worm_index]:
                        worms[worm_index] = candidate
                        scores[worm_index] = candidate_score

                if scores[worm_index] < best_score:
                    best_score = scores[worm_index]
                    best_position = worms[worm_index][:]

            self.history_.append(best_score)

        self.best_position = best_position
        self.best_score = best_score
        self.population_ = [worm[:] for worm in worms]

    def predict(self) -> List[float]:
        if self.best_position is None:
            raise ValueError("must call fit() before predict()")
        return self.best_position[:]

    def score(self) -> float:
        if self.best_score is None:
            raise ValueError("must call fit() before score()")
        return float(self.best_score)

    def get_params(self) -> dict[str, float]:
        return {
            "n_worms": float(self.n_worms),
            "n_iterations": float(self.n_iterations),
            "luciferin_decay": float(self.luciferin_decay),
            "luciferin_enhancement": float(self.luciferin_enhancement),
            "step_size": float(self.step_size),
            "neighborhood_radius": float(self.neighborhood_radius),
        }


@dataclass
class BacterialForagingOptimizer:
    n_bacteria: int = 30
    n_iterations: int = 100
    n_chemotactic_steps: int = 10
    n_reproduction_steps: int = 4
    elimination_probability: float = 0.25
    step_scale: float = 0.1
    random_state: Optional[int] = None
    best_position: Optional[List[float]] = None
    best_score: Optional[float] = None
    history_: List[float] = field(default_factory=list, repr=False)
    population_: List[List[float]] = field(default_factory=list, repr=False)

    def fit(
        self,
        objective: Callable[[List[float]], float],
        lower: Sequence[float],
        upper: Sequence[float],
    ) -> None:
        lower_values, upper_values = _ensure_bounds(lower, upper)
        dimension = len(lower_values)
        if self.n_bacteria < 2:
            raise ValueError("n_bacteria must be at least 2 for bacterial foraging")

        rng = random.Random(self.random_state)
        ranges = [high - low for low, high in zip(lower_values, upper_values)]
        bacteria = [
            [lower_values[index] + rng.random() * ranges[index] for index in range(dimension)]
            for _ in range(self.n_bacteria)
        ]
        scores = [_objective_value(objective, bacterium) for bacterium in bacteria]

        best_index = min(range(self.n_bacteria), key=lambda index: scores[index])
        best_position = bacteria[best_index][:]
        best_score = scores[best_index]

        for reproduction_round in range(self.n_reproduction_steps):
            for _ in range(self.n_chemotactic_steps):
                for bacterium_index in range(self.n_bacteria):
                    current = bacteria[bacterium_index][:]
                    candidate = current[:]
                    for dimension_index in range(dimension):
                        step = (rng.random() * 2.0 - 1.0) * self.step_scale * ranges[dimension_index]
                        candidate[dimension_index] += step
                        candidate[dimension_index] = min(
                            max(candidate[dimension_index], lower_values[dimension_index]),
                            upper_values[dimension_index],
                        )

                    candidate_score = _objective_value(objective, candidate)
                    if candidate_score < scores[bacterium_index]:
                        bacteria[bacterium_index] = candidate
                        scores[bacterium_index] = candidate_score
                    else:
                        current_score = scores[bacterium_index]
                        if candidate_score <= current_score + 1e-12:
                            bacteria[bacterium_index] = candidate
                            scores[bacterium_index] = candidate_score

                    if scores[bacterium_index] < best_score:
                        best_score = scores[bacterium_index]
                        best_position = bacteria[bacterium_index][:]

            order = sorted(range(self.n_bacteria), key=lambda index: scores[index])
            survivors = max(1, self.n_bacteria // 2)
            surviving_indices = order[:survivors]
            survivors_bacteria = [bacteria[index][:] for index in surviving_indices]
            survivors_scores = [scores[index] for index in surviving_indices]
            replicated_bacteria: list[list[float]] = []
            replicated_scores: list[float] = []
            while len(replicated_bacteria) < self.n_bacteria:
                for bacterium, score in zip(survivors_bacteria, survivors_scores):
                    replicated_bacteria.append(bacterium[:])
                    replicated_scores.append(score)
                    if len(replicated_bacteria) >= self.n_bacteria:
                        break
            bacteria = replicated_bacteria[: self.n_bacteria]
            scores = replicated_scores[: self.n_bacteria]

            for bacterium_index in range(self.n_bacteria):
                if rng.random() < self.elimination_probability:
                    bacteria[bacterium_index] = [
                        lower_values[dimension_index] + rng.random() * ranges[dimension_index]
                        for dimension_index in range(dimension)
                    ]
                    scores[bacterium_index] = _objective_value(objective, bacteria[bacterium_index])
                    if scores[bacterium_index] < best_score:
                        best_score = scores[bacterium_index]
                        best_position = bacteria[bacterium_index][:]

            self.history_.append(best_score)

        self.best_position = best_position
        self.best_score = best_score
        self.population_ = [bacterium[:] for bacterium in bacteria]

    def predict(self) -> List[float]:
        if self.best_position is None:
            raise ValueError("must call fit() before predict()")
        return self.best_position[:]

    def score(self) -> float:
        if self.best_score is None:
            raise ValueError("must call fit() before score()")
        return float(self.best_score)

    def get_params(self) -> dict[str, float]:
        return {
            "n_bacteria": float(self.n_bacteria),
            "n_iterations": float(self.n_iterations),
            "n_chemotactic_steps": float(self.n_chemotactic_steps),
            "n_reproduction_steps": float(self.n_reproduction_steps),
            "elimination_probability": float(self.elimination_probability),
            "step_scale": float(self.step_scale),
        }


@dataclass
class DifferentialEvolution:
    n_individuals: int = 40
    n_iterations: int = 100
    f: float = 0.8
    cr: float = 0.9
    random_state: Optional[int] = None
    best_position: Optional[List[float]] = None
    best_score: Optional[float] = None
    history_: List[float] = field(default_factory=list, repr=False)
    population_: List[List[float]] = field(default_factory=list, repr=False)

    def fit(
        self,
        objective: Callable[[List[float]], float],
        lower: Sequence[float],
        upper: Sequence[float],
    ) -> None:
        lower_values, upper_values = _ensure_bounds(lower, upper)
        dimension = len(lower_values)
        if self.n_individuals < 4:
            raise ValueError("n_individuals must be at least 4 for differential evolution")

        rng = random.Random(self.random_state)
        ranges = [high - low for low, high in zip(lower_values, upper_values)]

        population = [
            [lower_values[index] + rng.random() * ranges[index] for index in range(dimension)]
            for _ in range(self.n_individuals)
        ]
        scores = [_objective_value(objective, individual) for individual in population]

        best_index = min(range(self.n_individuals), key=lambda index: scores[index])
        best_position = population[best_index][:]
        best_score = scores[best_index]

        for _ in range(self.n_iterations):
            for target_index in range(self.n_individuals):
                pool = [index for index in range(self.n_individuals) if index != target_index]
                a_index, b_index, c_index = rng.sample(pool, 3)

                base = population[a_index]
                donor = population[b_index]
                difference = population[c_index]

                mutant = [
                    base[dimension_index]
                    + self.f * (donor[dimension_index] - difference[dimension_index])
                    for dimension_index in range(dimension)
                ]

                target = population[target_index]
                trial = target[:]
                crossover_index = rng.randrange(dimension)
                for dimension_index in range(dimension):
                    if rng.random() < self.cr or dimension_index == crossover_index:
                        trial[dimension_index] = mutant[dimension_index]
                    trial[dimension_index] = min(
                        max(trial[dimension_index], lower_values[dimension_index]),
                        upper_values[dimension_index],
                    )

                score = _objective_value(objective, trial)
                if score <= scores[target_index]:
                    population[target_index] = trial
                    scores[target_index] = score
                    if score < best_score:
                        best_score = score
                        best_position = trial[:]

            self.history_.append(best_score)

        self.best_position = best_position
        self.best_score = best_score
        self.population_ = [individual[:] for individual in population]

    def predict(self) -> List[float]:
        if self.best_position is None:
            raise ValueError("must call fit() before predict()")
        return self.best_position[:]

    def score(self) -> float:
        if self.best_score is None:
            raise ValueError("must call fit() before score()")
        return float(self.best_score)

    def get_params(self) -> dict[str, float]:
        return {
            "n_individuals": float(self.n_individuals),
            "n_iterations": float(self.n_iterations),
            "f": float(self.f),
            "cr": float(self.cr),
        }


@dataclass
class BeeColony:
    n_bees: int = 50
    n_iterations: int = 100
    limit: int = 10
    random_state: Optional[int] = None
    best_position: Optional[List[float]] = None
    best_score: Optional[float] = None
    history_: List[float] = field(default_factory=list, repr=False)
    population_: List[List[float]] = field(default_factory=list, repr=False)

    def fit(
        self,
        objective: Callable[[List[float]], float],
        lower: Sequence[float],
        upper: Sequence[float],
    ) -> None:
        lower_values, upper_values = _ensure_bounds(lower, upper)
        dimension = len(lower_values)
        rng = random.Random(self.random_state)
        ranges = [high - low for low, high in zip(lower_values, upper_values)]
        source_count = max(self.n_bees // 2, 1)

        def random_source() -> list[float]:
            return [lower_values[index] + rng.random() * ranges[index] for index in range(dimension)]

        sources = [random_source() for _ in range(source_count)]
        objective_scores = [_objective_value(objective, source) for source in sources]
        trials = [0] * source_count

        best_index = min(range(source_count), key=lambda index: objective_scores[index])
        best_position = sources[best_index][:]
        best_score = objective_scores[best_index]

        for _ in range(self.n_iterations):
            for source_index in range(source_count):
                self._exploit_source(
                    sources,
                    objective_scores,
                    trials,
                    source_index,
                    objective,
                    rng,
                    lower_values,
                    upper_values,
                    ranges,
                )

            weights = [self._selection_fitness(score) for score in objective_scores]
            total = sum(weights)
            for _ in range(source_count):
                selected = self._roulette_select(weights, total, rng)
                self._exploit_source(
                    sources,
                    objective_scores,
                    trials,
                    selected,
                    objective,
                    rng,
                    lower_values,
                    upper_values,
                    ranges,
                )

            if trials:
                worst_index = max(range(source_count), key=lambda index: trials[index])
                if trials[worst_index] > self.limit:
                    sources[worst_index] = random_source()
                    objective_scores[worst_index] = _objective_value(objective, sources[worst_index])
                    trials[worst_index] = 0

            current_best_index = min(range(source_count), key=lambda index: objective_scores[index])
            if objective_scores[current_best_index] < best_score:
                best_score = objective_scores[current_best_index]
                best_position = sources[current_best_index][:]
            self.history_.append(best_score)

        self.best_position = best_position
        self.best_score = best_score
        self.population_ = [source[:] for source in sources]

    def _exploit_source(
        self,
        sources: list[list[float]],
        objective_scores: list[float],
        trials: list[int],
        source_index: int,
        objective: Callable[[List[float]], float],
        rng: random.Random,
        lower_values: list[float],
        upper_values: list[float],
        ranges: list[float],
    ) -> None:
        source_count = len(sources)
        dimension = len(sources[source_index])
        partner = rng.randrange(source_count)
        while partner == source_index and source_count > 1:
            partner = rng.randrange(source_count)

        coordinate = rng.randrange(dimension)
        phi = rng.random() * 2.0 - 1.0

        candidate = sources[source_index][:]
        candidate[coordinate] = sources[source_index][coordinate] + phi * (
            sources[source_index][coordinate] - sources[partner][coordinate]
        )
        for index in range(dimension):
            candidate[index] = min(max(candidate[index], lower_values[index]), upper_values[index])

        score = _objective_value(objective, candidate)
        if score < objective_scores[source_index]:
            sources[source_index] = candidate
            objective_scores[source_index] = score
            trials[source_index] = 0
        else:
            trials[source_index] += 1

    @staticmethod
    def _selection_fitness(objective: float) -> float:
        if objective >= 0.0:
            return 1.0 / (1.0 + objective)
        return 1.0 + abs(objective)

    @staticmethod
    def _roulette_select(weights: Sequence[float], total: float, rng: random.Random) -> int:
        if total <= 0.0 or not math.isfinite(total):
            return rng.randrange(len(weights))
        threshold = rng.random() * total
        accumulator = 0.0
        for index, weight in enumerate(weights):
            accumulator += weight
            if accumulator >= threshold:
                return index
        return len(weights) - 1

    def predict(self) -> List[float]:
        if self.best_position is None:
            raise ValueError("must call fit() before predict()")
        return self.best_position[:]

    def score(self) -> float:
        if self.best_score is None:
            raise ValueError("must call fit() before score()")
        return float(self.best_score)

    def get_params(self) -> dict[str, float]:
        return {
            "n_bees": float(self.n_bees),
            "n_iterations": float(self.n_iterations),
            "limit": float(self.limit),
        }
