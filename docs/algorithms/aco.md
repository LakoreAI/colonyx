# Ant Colony Optimization

## Definition

Ant Colony Optimization (ACO) is a discrete, construction-based metaheuristic
for combinatorial problems such as the Traveling Salesman Problem. A colony
of ants each builds a tour city-by-city, biased by a shared pheromone
matrix (learned experience) and a heuristic (inverse edge distance). After
every ant finishes, pheromone evaporates everywhere and is redeposited on
the edges good tours used, so the colony's search converges toward
consistently short tours over successive iterations. `AntColony` also
supports `basic`, `acs`, `elitist`, and `mmas` pheromone-update variants —
see [ACO Variants](aco-variants.md).

## Pseudocode

```text
initialize pheromone[i][j] = 1 / n_cities for all edges
best_tour = None

for iteration in 1..n_iterations:
    solutions = []
    for ant in 1..n_ants:
        tour = [random start city]
        while unvisited cities remain:
            if variant == "acs" and random() < q0:
                next_city = argmax over unvisited of pheromone[cur][c]^alpha * (1/distance[cur][c])^beta
            else:
                # roulette-wheel selection weighted by the same quantity
                next_city = roulette_select(unvisited, weight = pheromone^alpha * (1/distance)^beta)
            tour.append(next_city)
        if use_two_opt: tour = two_opt(tour, distance_matrix)
        length = evaluate(tour)
        if length < best_length: best_tour, best_length = tour, length
        solutions.append((tour, length))

    # pheromone update, once per iteration
    pheromone *= (1 - rho)                      # evaporation, every edge
    if variant != "acs":
        for (tour, length) in solutions:
            deposit q / length on every edge of tour       # all ants deposit
    if variant in {"elitist", "acs", "mmas"}:
        deposit elitist_weight * q / best_length on every edge of best_tour
    if variant == "mmas":
        clamp every pheromone[i][j] to [tau_min, tau_max]
```

## Mathematical Formulation

An ant at city \(i\) chooses the next unvisited city \(j\) by roulette-wheel
selection weighted by pheromone and inverse distance:

$$
P_{ij} = \frac{\tau_{ij}^{\alpha}\,\eta_{ij}^{\beta}}{\displaystyle\sum_{l \in \text{allowed}} \tau_{il}^{\alpha}\,\eta_{il}^{\beta}}, \qquad \eta_{ij} = \frac{1}{d_{ij}}
$$

For the `acs` variant, with probability `q0` the ant instead exploits
greedily rather than sampling from \(P_{ij}\):

$$
j = \operatorname*{arg\,max}_{l \in \text{allowed}} \tau_{il}^{\alpha}\,\eta_{il}^{\beta}
$$

Every edge evaporates once per iteration:

$$
\tau_{ij} \leftarrow (1-\rho)\,\tau_{ij}
$$

Then deposit happens. For `basic`/`elitist`/`mmas`, every ant \(k\) deposits
on the edges of its tour (`acs` skips this all-ants deposit entirely):

$$
\tau_{ij} \mathrel{+}= \frac{q}{L_k} \quad \text{for each edge } (i,j) \text{ on ant } k\text{'s tour}
$$

For `elitist`, `acs`, and `mmas`, the best-so-far tour additionally deposits
with an extra weight (this is `acs`'s *only* deposit):

$$
\tau_{ij} \mathrel{+}= \frac{\text{elitist_weight} \cdot q}{L^{*}} \quad \text{for each edge on the best-so-far tour}
$$

`mmas` finally clamps every \(\tau_{ij}\) to \([\tau_{\min}, \tau_{\max}]\).

## Use when

- You have a square distance matrix.
- You want a tour/permutation solution.

## API

- Rust class: `colonyx._colonyx.AntColony`
- Python mode: `AutoColony(mode="aco")`
- Helper: `two_opt` for local tour improvement

## Parameters

- `n_ants`
- `n_iterations`
- `alpha`, `beta`
- `rho`, `q`
- `use_two_opt`

## Notes

- The Rust implementation handles pheromone updates and construction.
- `use_two_opt=True` enables local tour refinement.

## Example

```python
from colonyx import AutoColony

distance_matrix = [
    [0.0, 1.0, 9.0, 9.0],
    [1.0, 0.0, 1.0, 9.0],
    [9.0, 1.0, 0.0, 1.0],
    [9.0, 9.0, 1.0, 0.0],
]

optimizer = AutoColony(mode="aco", n_iterations=100, random_state=7)
optimizer.fit(distance_matrix)
print(optimizer.predict(), optimizer.score())
```
