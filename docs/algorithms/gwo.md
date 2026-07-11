# Grey Wolf Optimization

`GreyWolfOptimizer` tracks alpha, beta, and delta leaders.

## Use when

- You want a compact continuous optimizer.
- You want a simple update rule with leader guidance.

## API

- Rust class: `colonyx._colonyx.GreyWolfOptimizer`
- Python mode: `AutoColony(mode="gwo")`

## Parameters

- `n_wolves`
- `n_iterations`

## Example

```python
from colonyx import AutoColony

optimizer = AutoColony(mode="gwo", n_iterations=100, random_state=7)
optimizer.fit(lambda x: sum(v * v for v in x), bounds=[(-5, 5), (-5, 5)])
```
