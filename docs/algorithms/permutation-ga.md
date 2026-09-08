---
title: Permutation Genetic Algorithm
description: PermutationGeneticOptimizer in colonyx — a permutation-encoded genetic algorithm with order crossover, swap mutation, and optional 2-opt refinement for TSP-style combinatorial search.
---

# Permutation Genetic Optimizer (Permutation GA)

`PermutationGeneticOptimizer` is a permutation-based genetic algorithm for
TSP-style combinatorial search.

!!! abstract "TL;DR"
    A genetic algorithm whose chromosomes are whole permutations (tours) instead of real-valued vectors, evolved with order crossover, swap mutation, and elitism. Give it a square distance matrix; it returns the shortest tour it found. Use it as a GA-flavored alternative to [ACO](aco.md)/[ACO variants](aco-variants.md) on the same combinatorial search problems.

## What is a permutation genetic algorithm?

A standard genetic algorithm evolves vectors of independent genes, and standard crossover operators (single-point, uniform) assume genes can be swapped freely between parents without breaking validity. That assumption fails for permutation problems like the traveling salesman problem: if a child inherits city 3 from one parent and city 3 again from the other, the resulting "tour" visits city 3 twice and some other city not at all — it isn't a valid permutation anymore. `PermutationGeneticOptimizer` solves this by encoding each individual directly as a permutation of city indices and using crossover/mutation operators designed to preserve validity: order crossover (OX) builds a child that inherits a contiguous slice from one parent and fills the rest from the other parent's relative order, and swap mutation exchanges two positions rather than perturbing a single gene in isolation. The result is a general-purpose evolutionary search over the same TSP-style search space that [Ant Colony Optimization](aco.md) targets, but built on genetic-algorithm mechanics (selection, crossover, mutation, elitism) instead of ACO's pheromone-based collective learning.

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

## When to use it (and when not to)

- You have a distance matrix and need a valid permutation solution — the encoding guarantees every candidate is a legal tour, so there's no repair step to write.
- You want a simple, well-understood GA (order crossover + swap mutation + elitism) rather than a pheromone-based collective search.
- Compared to [`AntColony`](aco.md) (and its [variants](aco-variants.md)): both solve the same TSP-style problem from the same distance-matrix input, but ACO's pheromone trails give it an implicit, self-reinforcing memory of good edges across the whole run, while the GA's memory is entirely encoded in its current population — for small-to-medium instances the two are often competitive, and it's worth trying both with `use_two_opt=True` before committing.
- Skip it if your problem isn't naturally a permutation (use one of the continuous `AutoColony` modes like [PSO](pso.md) or [ABC](abc.md) instead) or if you need a Pareto front over multiple objectives (use [NSGA-II](nsga2.md)).

## API

- Rust class: `colonyx._colonyx.PermutationGeneticOptimizer`

## Parameters

| Parameter | Default | Meaning | Tuning notes |
|---|---|---|---|
| `n_individuals` | `40` | Population size (number of tours per generation). | Larger populations explore more of the permutation space per generation at the cost of more objective evaluations; scale up with the number of cities. |
| `n_iterations` | `100` | Number of generations to evolve. | Increase for larger instances or when the score history hasn't flattened yet. |
| `mutation_rate` | `0.1` | Probability that a child undergoes one swap mutation. | Higher values add diversity and help escape local optima, but too high turns the search into undirected shuffling. |
| `use_two_opt` | `True` | Whether the initial random population is locally refined with 2-opt before the GA loop starts. | Keep it on for a much stronger starting population; it only runs once, not every generation (see the note below). |
| `random_state` | `None` | Seed for reproducible runs. | Set an integer for deterministic results, e.g. in tests or benchmarks. |

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

## Further reading

- Davis, L. (1985). *Applying Adaptive Algorithms to Epigenetic Design Problems* — the order crossover (OX) operator used here.
- Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems.* University of Michigan Press.
- [ACO](aco.md) and [ACO variants](aco-variants.md) — the pheromone-based alternative for the same distance-matrix input.
- [Advanced Algorithms](advanced.md) · [Algorithms Overview](../algorithms.md)
