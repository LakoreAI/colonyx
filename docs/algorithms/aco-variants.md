---
title: Ant Colony Optimization Variants
description: AntColony variant argument in colonyx — basic Ant System, Ant Colony System (ACS), Elitist AS, and MAX-MIN Ant System (MMAS) for tuning exploration vs. exploitation in combinatorial search.
---

# ACO Variants (ACS, Elitist AS, MMAS)

`AntColony` supports several discrete-search variants through the `variant`
argument.

!!! abstract "TL;DR"
    Four pheromone-update strategies on top of the same [ACO](aco.md) loop, all selected via `AntColony(variant=...)`: `basic` (standard Ant System), `acs` (Ant Colony System — greedy exploitation + only-the-best deposits), `elitist` (extra reinforcement from the best-so-far tour), and `mmas` (MAX-MIN Ant System — elitist plus pheromone clamping to prevent premature convergence). `AutoColony(mode="aco")` only exposes `basic`; use the `AntColony` class directly for the others.

## What do these variants change, and why does it matter?

Plain Ant System — the original ACO algorithm — has a well-known weakness: because every ant deposits pheromone on its own tour every iteration, pheromone levels can grow essentially without bound on frequently-used edges, and the colony can converge prematurely onto a mediocre tour before it's explored enough of the search space. The three refinements documented here (ACS, Elitist AS, MMAS) each address that failure mode differently, and picking between them is really a question of how you want to trade off exploration against exploitation. `Elitist` is the smallest change: it keeps everything about basic Ant System and simply adds one more deposit from the best tour found so far, reinforcing good solutions faster without changing anything else. `ACS` goes further, replacing roulette-wheel city selection with a greedy shortcut most of the time (controlled by `q0`) and restricting pheromone deposits to only the best-so-far ant — a more exploitative search that converges faster but risks getting stuck if `q0` is too aggressive. `MMAS` takes the opposite concern seriously: it clamps every pheromone value into a fixed range (`[tau_min, tau_max]`) after each update, which directly prevents the runaway pheromone growth that causes premature convergence in the other variants, at the cost of one extra hyperparameter pair to tune.

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

## When to use it (and when not to)

- `basic` — a solid default and the right choice when you have no reason to favor exploration or exploitation specifically; it's also what `AutoColony(mode="aco")` uses under the hood.
- `elitist` — reach for this when basic Ant System converges too slowly and you want faster reinforcement of good tours without adding new hyperparameters to tune (`elitist_weight` has a sensible default).
- `acs` — reach for this when you want faster, more exploitative convergence and are willing to tune `q0`; it's a common choice for larger instances where basic Ant System takes too long to focus.
- `mmas` — reach for this when you're seeing premature convergence (the colony settling on a mediocre tour early and the score history flatlining) with the other variants; the pheromone-clamping directly targets that failure mode.
- Compared to [Permutation GA](permutation-ga.md): both solve the same distance-matrix input; ACO's pheromone trails give it a persistent, self-reinforcing memory across the whole run that a GA's population doesn't have in the same form, which can help on structured instances where good sub-tours recur.

## API

- Rust class: `colonyx._colonyx.AntColony`
- Python mode: `AutoColony(mode="aco")` does not expose `variant` — use the
  `AntColony` class directly for non-`basic` variants (see below).

## Parameters (variant-specific)

These extend the base parameters documented on the [ACO](aco.md) page (`n_ants`, `alpha`, `beta`, `rho`, `q`, `use_two_opt`).

| Parameter | Default | Meaning | Tuning notes |
|---|---|---|---|
| `variant` | `"basic"` | One of `"basic"`, `"acs"`, `"elitist"`, `"mmas"`. | Selects the city-choice and pheromone-deposit rule described above. |
| `q0` | `0.9` | ACS-only: probability of greedy (highest-weight) city selection instead of roulette-wheel sampling. | Higher values push ACS toward pure exploitation; lower values make it behave closer to basic/elitist. |
| `elitist_weight` | `2.0` | Multiplier on the best-so-far ant's extra pheromone deposit (used by `elitist`, `acs`, and `mmas`). | Raise it to reinforce the current best tour more aggressively. |
| `tau_min` | `1e-4` | MMAS-only: lower clamp on any pheromone value. | Keep above 0 to preserve some non-zero probability of exploring any edge. |
| `tau_max` | `10.0` | MMAS-only: upper clamp on any pheromone value. | Lower it to limit how dominant any single edge can become. |

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

## Further reading

- Dorigo, M., & Gambardella, L. M. (1997). *Ant Colony System: A Cooperative Learning Approach to the Traveling Salesman Problem.* IEEE Transactions on Evolutionary Computation, 1(1), 53–66. — Ant Colony System (`acs`).
- Stützle, T., & Hoos, H. H. (2000). *MAX–MIN Ant System.* Future Generation Computer Systems, 16(8), 889–914. — MAX-MIN Ant System (`mmas`).
- Dorigo, M., Maniezzo, V., & Colorni, A. (1996). *Ant System: Optimization by a Colony of Cooperating Agents.* IEEE Transactions on Systems, Man, and Cybernetics, 26(1), 29–41. — the original Ant System (`basic`) and Elitist AS.
- [ACO](aco.md) — the shared base loop these variants modify.
- [Advanced Algorithms](advanced.md) · [Algorithms Overview](../algorithms.md)
