# ACO Variants

`AntColony` supports several discrete-search variants through the `variant`
argument.

## Definition

All four variants share the same ACO loop described in [ACO](aco.md) — ants
construct tours, pheromone evaporates, then deposits happen. What differs
between variants is *how* a city is chosen during construction and *who*
deposits pheromone afterward.

## Pseudocode (deltas from the base ACO loop)

```text
select_next_city(current, unvisited):
    weight(city) = pheromone[current][city]^alpha * (1 / distance[current][city])^beta
    if variant == acs and random() < q0:
        return the unvisited city with the highest weight   # greedy exploitation
    else:
        return roulette_wheel_select(unvisited, weighted by weight(city))

update_pheromones(this_iteration's tours):
    evaporate every edge by (1 - rho)
    if variant != acs:
        every ant deposits q / tour_length on the edges of its own tour
    if variant in {elitist, acs, mmas}:
        the best-so-far ant additionally deposits elitist_weight * q / tour_length
        # (this is ACS's *only* deposit source, since it skipped the step above)
    if variant == mmas:
        clamp every pheromone value to [tau_min, tau_max]
```

## Mathematical Formulation

This page only states the *deltas* from [ACO](aco.md)'s base equations —
see there for the full \(P_{ij}\) roulette-wheel and evaporation formulas.

The `acs` variant replaces roulette sampling with greedy exploitation,
gated by `q0`:

$$
j =
\begin{cases}
\operatorname*{arg\,max}_{l \in \text{allowed}} \tau_{il}^{\alpha}\eta_{il}^{\beta} & \text{with probability } q_0 \\
\text{sample from } P_{i \cdot} & \text{otherwise}
\end{cases}
$$

Deposit sources differ per variant. Let \(D_{\text{all}}\) be "every ant
deposits \(q/L_k\) on its own tour" and \(D_{\text{elite}}\) be "the
best-so-far tour additionally deposits \(\text{elitist_weight} \cdot q / L^{*}\)":

$$
\text{deposit}(\text{variant}) =
\begin{cases}
D_{\text{all}} & \text{basic} \\
D_{\text{all}} + D_{\text{elite}} & \text{elitist, mmas} \\
D_{\text{elite}} \text{ only} & \text{acs}
\end{cases}
$$

`mmas` additionally clamps every pheromone value after deposit:

$$
\tau_{ij} \leftarrow \min\bigl(\max(\tau_{ij}, \tau_{\min}), \tau_{\max}\bigr)
$$

## Variants

- `basic` — standard Ant System: every ant deposits pheromone each iteration; roulette-wheel city choice only.
- `acs` — Ant Colony System: with probability `q0`, the next city is chosen greedily (highest weight) instead of by roulette wheel; only the best-so-far ant deposits pheromone (weighted by `elitist_weight`), not every ant.
- `elitist` — same construction and per-ant deposit as `basic`, plus an extra reinforcing deposit from the best-so-far ant.
- `mmas` — Max-Min Ant System: same deposits as `elitist`, plus every pheromone value is clamped to `[tau_min, tau_max]` after each update, preventing premature convergence from unbounded pheromone growth.

## API

- Rust class: `colonyx._colonyx.AntColony`
- Python mode: `AutoColony(mode="aco")` does not expose `variant` — use the
  `AntColony` class directly for non-`basic` variants (see below).

## Example

```python
from colonyx import AntColony

distance_matrix = [
    [0.0, 1.0, 9.0, 9.0],
    [1.0, 0.0, 1.0, 9.0],
    [9.0, 1.0, 0.0, 1.0],
    [9.0, 9.0, 1.0, 0.0],
]

optimizer = AntColony(n_ants=20, n_iterations=100, variant="elitist", random_state=42)
optimizer.fit(distance_matrix)
print(optimizer.predict(), optimizer.score())
```
