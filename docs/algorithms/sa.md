# Simulated Annealing

`SimulatedAnnealing` is a single-solution local search optimizer.

## Use when

- You want a lightweight baseline.
- You prefer simple step/noise tuning over population logic.

## API

- Rust class: `colonyx._colonyx.SimulatedAnnealing`
- Python mode: `AutoColony(mode="sa")`

## Parameters

- `initial_temperature`
- `cooling_rate`
- `step_scale`
- `n_iterations`

## Example

```python
from colonyx import AutoColony

optimizer = AutoColony(mode="sa", n_iterations=100, random_state=7)
optimizer.fit(lambda x: sum(v * v for v in x), bounds=[(-5, 5), (-5, 5)])
```
