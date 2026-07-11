# Bat Algorithm

`BatAlgorithm` uses frequency, loudness, and pulse-rate updates.

## Use when

- You want another population-based continuous heuristic.
- You are comfortable with more parameters than PSO.

## API

- Rust class: `colonyx._colonyx.BatAlgorithm`
- Python mode: `AutoColony(mode="ba")`

## Parameters

- `n_bats`
- `n_iterations`
- `fmin`, `fmax`
- `alpha`, `gamma`
- `loudness`, `pulse_rate`

## Example

```python
from colonyx import AutoColony

optimizer = AutoColony(mode="ba", n_iterations=100, random_state=7)
optimizer.fit(lambda x: sum(v * v for v in x), bounds=[(-5, 5), (-5, 5)])
```
