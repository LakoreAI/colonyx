# Ant Colony Optimization

`AntColony` is the discrete optimizer for TSP-style problems.

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

optimizer = AutoColony(mode="aco", n_iterations=100, random_state=7)
optimizer.fit(distance_matrix)
```
