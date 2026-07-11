# Artificial Bee Colony

`BeeColony` models employed, onlooker, and scout phases.

## Use when

- You want a population-based continuous optimizer.
- You are comfortable tuning a food-source limit.

## API

- Rust class: `colonyx._colonyx.BeeColony`
- Python mode: `AutoColony(mode="abc")`

## Parameters

- `n_bees`
- `n_iterations`
- `limit`

## Example

```python
from colonyx import AutoColony

optimizer = AutoColony(mode="abc", n_iterations=100, random_state=7)
optimizer.fit(lambda x: sum(v * v for v in x), bounds=[(-5, 5), (-5, 5)])
```

## Notes

- Strong fit for low- to medium-dimensional continuous landscapes.
