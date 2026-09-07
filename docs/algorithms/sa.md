# Simulated Annealing

## Definition

Simulated Annealing (SA) is a single-solution continuous optimizer, unlike
every other algorithm on this page which maintains a population. It takes
one random step at a time and always accepts an improving step; a
worsening step is still accepted with probability `exp(-delta / temperature)`,
so early on (high temperature) it can escape local minima, and as
`temperature` cools geometrically it settles into pure hill-descending.

## Pseudocode

```text
current = random position inside bounds
current_score = evaluate(current)
best = current, best_score = current_score
temperature = initial_temperature

for iteration in 1..n_iterations:
    candidate = current + random_step_scaled_by(step_scale)   # per dimension
    clamp candidate to bounds
    candidate_score = evaluate(candidate)
    delta = candidate_score - current_score
    if delta <= 0 or random(0, 1) < exp(-delta / temperature):
        current, current_score = candidate, candidate_score   # accept (maybe worse)
    if current_score < best_score:
        best, best_score = current, current_score
    temperature *= cooling_rate
```

## Mathematical Formulation

Let \(\Delta = f(x') - f(x)\) be the change in objective value for a
candidate step \(x'\). It's accepted (replacing the current solution) with
probability:

$$
P(\text{accept}) = \begin{cases} 1 & \Delta \le 0 \\[4pt] \exp\!\left(-\dfrac{\Delta}{T}\right) & \Delta > 0 \end{cases}
$$

The temperature cools geometrically each iteration:

$$
T \leftarrow T \cdot \text{cooling_rate}
$$

so \(P(\text{accept})\) for a fixed worsening \(\Delta\) shrinks over the
run, moving SA from broad exploration toward pure hill-descending.

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
print(optimizer.predict(), optimizer.score())
```
