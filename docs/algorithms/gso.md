# Glowworm Swarm Optimization

`GlowwormOptimizer` uses luciferin values and neighborhood movement.

## Use when

- You want multi-agent search with local neighborhood discovery.
- You care about neighborhood radius and luciferin decay.

## API

- Rust class: `colonyx._colonyx.GlowwormOptimizer`
- Python mode: `AutoColony(mode="gso")`

## Parameters

- `n_worms`
- `n_iterations`
- `luciferin_decay`
- `luciferin_enhancement`
- `step_size`
- `neighborhood_radius`

## Example

```python
from colonyx import AutoColony

optimizer = AutoColony(mode="gso", n_iterations=100, random_state=7)
optimizer.fit(lambda x: sum(v * v for v in x), bounds=[(-5, 5), (-5, 5)])
```
