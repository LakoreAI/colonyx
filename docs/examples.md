# Examples

## Continuous optimization

```python
from colonyx import AutoColony

def sphere(x):
    return sum(value * value for value in x)

optimizer = AutoColony(mode="pso", n_iterations=100, random_state=42)
optimizer.fit(sphere, bounds=[(-5, 5), (-5, 5), (-5, 5)])
print(optimizer.predict())
print(optimizer.score())
```

## Discrete optimization

```python
from colonyx import AutoColony

distance_matrix = [
    [0.0, 1.0, 9.0, 9.0],
    [1.0, 0.0, 1.0, 9.0],
    [9.0, 1.0, 0.0, 1.0],
    [9.0, 9.0, 1.0, 0.0],
]

optimizer = AutoColony(mode="aco", n_iterations=100, random_state=7)
optimizer.fit(distance_matrix)
print(optimizer.predict())
print(optimizer.score())
```

## Auto mode

```python
from colonyx import AutoColony

optimizer = AutoColony(mode="auto", n_iterations=100, random_state=7)
optimizer.fit(lambda x: sum(value * value for value in x), bounds=[(-5, 5), (-5, 5)])
```
