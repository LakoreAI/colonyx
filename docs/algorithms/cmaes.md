# CMA-ES

`CmaEsOptimizer` is a separable (diagonal-covariance) CMA-ES: each axis adapts
its own variance rather than a full covariance matrix, so it cannot model
axis rotation. The step size (`sigma`) follows real cumulative step-size
adaptation (CSA) via an evolution path, the same mechanism full CMA-ES uses.

## Use when

- You want a strong continuous optimizer with covariance adaptation.
- You can provide bounds and a reasonable iteration budget.

## API

- Rust class: `colonyx._colonyx.CmaEsOptimizer`
- Python mode: `AutoColony(mode="cmaes")`

## Parameters

- `n_individuals`
- `n_iterations`
- `sigma`

## Example

```python
from colonyx import AutoColony

optimizer = AutoColony(mode="cmaes", n_iterations=100, random_state=7)
optimizer.fit(lambda x: sum(v * v for v in x), bounds=[(-5, 5), (-5, 5)])
```
