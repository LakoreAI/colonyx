# Bacterial Foraging Optimization

`BacterialForagingOptimizer` models chemotaxis, reproduction, and elimination.

## Use when

- You want a more involved continuous heuristic.
- You can spend time tuning reproduction and elimination settings.

## API

- Rust class: `colonyx._colonyx.BacterialForagingOptimizer`
- Python mode: `AutoColony(mode="bfo")`

## Parameters

- `n_bacteria`
- `n_iterations`
- `n_chemotactic_steps`
- `n_reproduction_steps`
- `elimination_probability`
- `step_scale`

## Example

```python
from colonyx import AutoColony

optimizer = AutoColony(mode="bfo", n_iterations=100, random_state=7)
optimizer.fit(lambda x: sum(v * v for v in x), bounds=[(-5, 5), (-5, 5)])
```
