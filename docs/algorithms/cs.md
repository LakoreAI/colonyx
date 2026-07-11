# Cuckoo Search

`CuckooSearch` combines Lévy-flight steps with nest replacement.

## Use when

- You want exploration-heavy continuous search.
- You can tune abandonment probability.

## API

- Rust class: `colonyx._colonyx.CuckooSearch`
- Python mode: `AutoColony(mode="cs")`

## Parameters

- `n_nests`
- `n_iterations`
- `pa`
- `alpha`
- `levy_scale`

## Example

```python
from colonyx import AutoColony

optimizer = AutoColony(mode="cs", n_iterations=100, random_state=7)
optimizer.fit(lambda x: sum(v * v for v in x), bounds=[(-5, 5), (-5, 5)])
```
