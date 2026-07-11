# Firefly Algorithm

`FireflyOptimizer` moves candidates toward brighter neighbors.

## Use when

- You want multimodal search behavior.
- You are tuning `beta0`, `gamma`, and step noise.

## API

- Rust class: `colonyx._colonyx.FireflyOptimizer`
- Python mode: `AutoColony(mode="fa")`

## Parameters

- `n_fireflies`
- `n_iterations`
- `beta0`, `gamma`, `alpha`

## Example

```python
from colonyx import AutoColony

optimizer = AutoColony(mode="fa", n_iterations=100, random_state=7)
optimizer.fit(lambda x: sum(v * v for v in x), bounds=[(-5, 5), (-5, 5)])
```
