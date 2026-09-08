---
title: NSGA-II Multi-Objective Optimizer
description: Nsga2Optimizer in colonyx — a compact NSGA-II implementation with non-dominated sorting and crowding distance for multi-objective optimization problems.
---

# Non-dominated Sorting Genetic Algorithm II (NSGA-II)

`Nsga2Optimizer` is a compact multi-objective optimizer with Pareto ranking.

!!! abstract "TL;DR"
    A genetic algorithm for problems with multiple, conflicting objectives — instead of one best answer, it returns a Pareto front: a set of solutions where improving one objective requires worsening another. Ranks candidates by non-dominated "fronts" and, within a front, by crowding distance, so the search both converges toward the front and spreads out along it. Compare against [MOPSO](mopso.md), which solves the same problem class with particle-swarm mechanics instead of a GA.

## What is multi-objective optimization, and what does NSGA-II do about it?

Every algorithm documented elsewhere in colonyx optimizes a single scalar objective — there's always one number to minimize, and therefore one clear "best" solution. Many real problems don't fit that: minimizing cost and minimizing weight are both desirable in an engineering design, but they typically trade off against each other, so there's no single point that's best on both simultaneously. The right output for a genuinely multi-objective problem isn't one solution — it's the **Pareto front**: the set of solutions where no other candidate is at least as good on every objective and strictly better on one. NSGA-II (Deb et al., 2002) finds an approximation to that front with a genetic algorithm modified in two ways: candidates are ranked by **non-dominated sorting** into "fronts" (front 0 is dominated by nothing in the population, front 1 is dominated only by front 0, and so on — see the math below), and within a front they're additionally ranked by **crowding distance**, a measure of how isolated a point is from its front-mates. Selecting parents that are both low-rank (near the true front) and high-crowding-distance (in a sparsely covered region of it) pushes the population to converge toward the Pareto front while still spreading out along its whole length, rather than collapsing onto one small cluster of it.

<figure markdown>
![Four-step NSGA-II loop: rank the population into non-dominated fronts and compute crowding distance, tournament-select parents by rank then crowding to build offspring via crossover and mutation, combine parents and offspring and refill the next generation front-by-front while cutting the sparsest members of a partially-kept front, then repeat](../assets/diagrams/nsga2.svg)
<figcaption>Front 0 (amber) is never guaranteed the same members twice — it's recomputed from scratch every generation as better candidates displace the old front.</figcaption>
</figure>

## Definition

NSGA-II (Non-dominated Sorting Genetic Algorithm II) evolves a population
toward the Pareto front of a vector-valued objective — the set of solutions
where no objective can be improved without worsening another. Each
generation is ranked into non-dominated "fronts" (front 0 dominates nothing,
front 1 is dominated only by front 0, and so on) and individuals are
additionally ranked by crowding distance within their front, so the search
both converges toward the front and spreads out along it.

## Pseudocode

```text
population = n_individuals random candidates within bounds
evaluated = objective(candidate) for each candidate   # vector of objectives

for iteration in 1..n_iterations:
    fronts = non_dominated_sort(evaluated)
    for each candidate: rank = its front index, crowding = crowding_distance within that front

    offspring = []
    while len(offspring) < n_individuals:
        parent_a = binary_tournament(population, preferring lower rank, then higher crowding)
        parent_b = binary_tournament(population, same rule)
        child = blend_crossover(parent_a, parent_b) with probability crossover_rate
        mutate each gene of child with probability mutation_rate, scale mutation_scale
        clamp child to bounds
        offspring.append(child)

    combined = population + offspring, evaluated together
    combined_fronts = non_dominated_sort(combined)
    next_population = fill from combined_fronts in order (front 0, then front 1, ...),
                       breaking ties within a partially-included front by crowding distance,
                       until next_population reaches n_individuals
    population = next_population

best_front = archive_from_population(final evaluated population, archive_size)
return best_front   # the Pareto front, size-capped by archive_size
```

## Mathematical Formulation

**Dominance.** \(x\) dominates \(y\) (written \(x \prec y\)) iff \(x\) is no
worse than \(y\) on every objective and strictly better on at least one:

$$
x \prec y \iff \bigl(\forall k:\, f_k(x) \le f_k(y)\bigr) \land \bigl(\exists k:\, f_k(x) < f_k(y)\bigr)
$$

**Non-dominated sorting** partitions the population into fronts
\(\mathcal{F}_0, \mathcal{F}_1, \dots\): \(\mathcal{F}_0\) is dominated by
nothing in the population; \(\mathcal{F}_r\) is dominated only by points in
\(\mathcal{F}_0, \dots, \mathcal{F}_{r-1}\).

**Crowding distance** measures how isolated a point is within its own
front, summed over objectives \(k\) (boundary points get \(\infty\) so they
are always kept):

$$
d_i = \sum_k \frac{f_k(x_{i+1}) - f_k(x_{i-1})}{f_k^{\max} - f_k^{\min}}
$$

where \(x_{i-1}, x_{i+1}\) are \(x_i\)'s neighbors in the front when sorted
by objective \(k\).

**Binary tournament selection** prefers the lower front rank, then higher
crowding distance:

$$
\text{better}(x, y) =
\begin{cases}
x & \text{rank}(x) < \text{rank}(y) \\
y & \text{rank}(y) < \text{rank}(x) \\
\arg\max(d_x, d_y) & \text{rank}(x) = \text{rank}(y)
\end{cases}
$$

Offspring are formed by blend crossover (probability `crossover_rate`) and
per-gene Gaussian-like mutation (probability `mutation_rate`, magnitude
`mutation_scale`); the next generation is filled front-by-front from the
combined parent+offspring pool, breaking ties in a partially-included front
by crowding distance.

## When to use it (and when not to)

- You need a small, dependency-free multi-objective baseline you can drop into a Python objective that returns a vector of scores.
- You want a Pareto front rather than a single point, because your objectives genuinely conflict.
- Compared to [MOPSO](mopso.md): both return a Pareto set from the same kind of input (a vector-valued objective plus box bounds), so the choice comes down to search dynamics — NSGA-II's crossover/mutation tends to explore more disruptively and its crowding-distance mechanism is specifically designed for even front coverage; MOPSO's particle-swarm leader-following tends to converge faster but relies on the archive's non-dominated front for diversity. For a small number of objectives (2–3), both are reasonable starting points — try both and compare front quality.
- Skip it for single-objective problems — reach for one of the [core `AutoColony` modes](../algorithms.md) instead, since NSGA-II's non-dominated sorting degenerates to plain fitness ranking when there's only one objective and buys you nothing.

## API

- Rust class: `colonyx._colonyx.Nsga2Optimizer`

## Parameters

| Parameter | Default | Meaning | Tuning notes |
|---|---|---|---|
| `n_individuals` | `40` | Population size. | Larger populations give denser front coverage at the cost of more evaluations per generation. |
| `n_iterations` | `100` | Number of generations. | Increase until the front stops visibly improving between runs. |
| `crossover_rate` | `0.9` | Probability that two parents produce a blended child. | Standard GA practice keeps this high (0.7–0.95); lower values slow convergence. |
| `mutation_rate` | `0.1` | Per-gene mutation probability. | Higher values add diversity but can disrupt convergence near the front. |
| `mutation_scale` | `0.1` | Magnitude of each mutation step. | Scale relative to your bounds' range; too large and mutation acts like re-randomization. |
| `archive_size` | `50` | Maximum number of solutions kept in the returned Pareto front. | Raise it for finer front resolution; it only truncates the *output*, not the working population. |
| `random_state` | `None` | Seed for reproducible runs. | Set an integer for deterministic results. |

## Example

```python
from colonyx import Nsga2Optimizer

def objectives(x):
    return [sum(value * value for value in x), sum((value - 1.0) ** 2 for value in x)]

optimizer = Nsga2Optimizer(n_individuals=30, n_iterations=50, random_state=42)
optimizer.fit(objectives, lower=[0.0, 0.0], upper=[1.0, 1.0])
print(optimizer.predict())
```

!!! warning
    `score()` reports the 2-objective hypervolume of the Pareto archive, so it
    requires `objectives(x)` to return at least 2 values. It raises
    `ValueError` if `fit()` was called with a single-objective function —
    `predict()` (the Pareto front itself) still works either way.

## Further reading

- Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). *A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II.* IEEE Transactions on Evolutionary Computation, 6(2), 182–197.
- [MOPSO](mopso.md) — the particle-swarm counterpart for the same multi-objective problem class.
- [Advanced Algorithms](advanced.md) · [Algorithms Overview](../algorithms.md)
