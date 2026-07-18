# ACO Variants

`AntColony` supports several discrete-search variants through the `variant`
argument.

## Variants

- `basic` — standard ant system
- `acs` — ant colony system with greedy choice probability
- `elitist` — reinforces the best tour more strongly
- `mmas` — max-min style pheromone clipping

## API

- Rust class: `colonyx._colonyx.AntColony`
- Python mode: `AutoColony(mode="aco")`

## Example

```python
from colonyx import AntColony

optimizer = AntColony(mode="aco", variant="elitist", n_iterations=100, random_state=42)
```
