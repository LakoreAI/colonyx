# NSGA-II

`Nsga2Optimizer` is a compact multi-objective optimizer with Pareto ranking.

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
