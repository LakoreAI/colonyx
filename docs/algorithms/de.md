# Differential Evolution

`DifferentialEvolution` mutates, crosses over, and selects greedily.

## Use when

- You want a strong general-purpose continuous optimizer.
- You want a simple, well-known population heuristic.

## API

- Rust class: `colonyx._colonyx.DifferentialEvolution`
- Python mode: `AutoColony(mode="de")`

## Parameters

- `n_individuals`
- `n_iterations`
- `f`
- `cr`

## Example

```python
from colonyx import AutoColony

optimizer = AutoColony(mode="de", n_iterations=100, random_state=7)
optimizer.fit(lambda x: sum(v * v for v in x), bounds=[(-5, 5), (-5, 5)])
```
