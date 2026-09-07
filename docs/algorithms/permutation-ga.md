# Permutation Genetic Optimizer

`PermutationGeneticOptimizer` is a permutation-based genetic algorithm for
TSP-style combinatorial search.

## Definition

A genetic algorithm whose individuals are permutations (tours) rather than
real-valued vectors. It evolves a population of tours over generations using
order crossover (which preserves relative city order from both parents,
unlike single-point crossover on a permutation) and swap mutation, with
elitism carrying the best tour into the next generation unchanged. The
initial population can optionally be locally refined with 2-opt.

## Pseudocode

```text
population = n_individuals random permutations of the n cities
for each individual:
    evaluate tour length
    if use_two_opt: locally refine the tour with 2-opt, re-evaluate

for iteration in 1..n_iterations:
    sort population by tour length
    best = population[0]                     # elitism
    record best.length in history

    next_generation = [best]
    while next_generation has fewer than n_individuals members:
        parent_a = tournament_pick(population)
        parent_b = tournament_pick(population)
        child = order_crossover(parent_a, parent_b)
        swap_mutation(child, probability=mutation_rate)
        evaluate child's tour length
        next_generation.append(child)
    population = next_generation

return the shortest tour found
```

!!! note "2-opt only refines the initial population"
    `use_two_opt` runs once, on the randomly generated starting tours — it
    isn't re-applied to crossover/mutation offspring each generation.

## Mathematical Formulation

Fitness is the closed-tour length over the distance matrix \(d\):

$$
f(\pi) = \sum_{i=0}^{n-1} d\bigl(\pi_i, \pi_{(i+1) \bmod n}\bigr)
$$

**Tournament selection** picks 2 individuals uniformly at random and keeps
the one with lower \(f\):

$$
\text{tournament}(P) = \arg\min\bigl(f(\pi_a), f(\pi_b)\bigr), \quad a, b \sim U\{0, \dots, |P|-1\}
$$

**Order crossover (OX)** builds a child from a random contiguous slice
\([s, e]\) of parent \(A\), then fills the remaining positions with the
cities of parent \(B\) in their cyclic order, skipping any city already
placed:

$$
\text{child}_i =
\begin{cases}
A_i & s \le i \le e \\[2pt]
\text{next unused city of } B \text{ (cyclic order from } e+1) & \text{otherwise}
\end{cases}
$$

**Swap mutation** applies with probability `mutation_rate`, swapping two
positions chosen uniformly at random — at most once per child, not
per-gene:

$$
\text{swap}(\pi, j, k), \quad j, k \sim U\{0, \dots, n-1\}, \quad \text{applied iff } \text{rand}() \le \text{mutation_rate}
$$

The best individual from generation \(t\) is copied unchanged into
generation \(t+1\) (elitism) before the rest of the population is filled by
tournament selection + OX + swap mutation.

## Use when

- You have a distance matrix and need a valid permutation solution.
- You want a simple GA with order crossover and swap mutation.

## API

- Rust class: `colonyx._colonyx.PermutationGeneticOptimizer`

## Parameters

- `n_individuals`
- `n_iterations`
- `mutation_rate`
- `use_two_opt`

## Example

```python
import numpy as np
from colonyx import PermutationGeneticOptimizer

distance_matrix = np.array([
    [0.0, 1.0, 9.0, 9.0],
    [1.0, 0.0, 1.0, 9.0],
    [9.0, 1.0, 0.0, 1.0],
    [9.0, 9.0, 1.0, 0.0],
])

optimizer = PermutationGeneticOptimizer(n_individuals=40, n_iterations=100, random_state=7)
optimizer.fit(distance_matrix)
print(optimizer.predict())
print(optimizer.score())
```
