# MOPSO

`MopsoOptimizer` is a compact multi-objective particle swarm optimizer.

## Use when

- You want PSO-style movement with a Pareto archive.
- You can provide a vector-valued objective and box bounds.

## API

- Rust class: `colonyx._colonyx.MopsoOptimizer`

## Parameters

- `n_particles`
- `n_iterations`
- `w`
- `c1`
- `c2`
- `mutation_scale`
- `archive_size`

## Example

```python
from colonyx import MopsoOptimizer

def objectives(x):
    return [sum(value * value for value in x), sum((value - 1.0) ** 2 for value in x)]

optimizer = MopsoOptimizer(n_particles=30, n_iterations=50, random_state=42)
optimizer.fit(objectives, lower=[0.0, 0.0], upper=[1.0, 1.0])
print(optimizer.predict())
```
