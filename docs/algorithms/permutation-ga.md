# Permutation Genetic Optimizer

`PermutationGeneticOptimizer` is a permutation-based genetic algorithm for
TSP-style combinatorial search.

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
