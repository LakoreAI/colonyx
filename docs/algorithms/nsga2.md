# NSGA-II

`Nsga2Optimizer` is a compact multi-objective optimizer with Pareto ranking.

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

## Use when

- You need a small multi-objective baseline.
- You want a Pareto archive from a Python objective that returns a vector.

## API

- Rust class: `colonyx._colonyx.Nsga2Optimizer`

## Parameters

- `n_individuals`
- `n_iterations`
- `crossover_rate`
- `mutation_rate`
- `mutation_scale`
- `archive_size`

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
