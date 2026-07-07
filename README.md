# colonyx

**colonyx** is a Python library for solving optimization problems using swarm intelligence algorithms like Ant Colony Optimization (ACO), Particle Swarm Optimization (PSO), and Artificial Bee Colony (ABC).

While the interface is Pythonic and easy to use, the core is written in Rust to deliver better performance for larger or more complex problems.

> This is an early version — contributions, suggestions, and feedback are all welcome.

## Features

- Ant Colony Optimization (ACO) — for discrete problems like TSP
- Particle Swarm Optimization (PSO) — for continuous function optimization
- Artificial Bee Colony (ABC) — inspired by bee foraging behavior
- Simple, clean Python API
- Fast backend powered by Rust

## Installation

```bash

pip install colonyx

```

*(Coming soon to PyPI — for now, install from source)*

## Example

All algorithms are used through the unified `AutoColony` interface, selected
via the `mode` parameter.

**Continuous optimization (PSO / ABC)** — minimize an objective function over a
box, given per-dimension `bounds`:

```python
from colonyx import AutoColony

def sphere(x):
    return sum(xi * xi for xi in x)  # minimum 0 at the origin

opt = AutoColony(mode="pso", n_iterations=150, random_state=42)
opt.fit(sphere, bounds=[(-5, 5), (-5, 5), (-5, 5)])

opt.predict()  # best position, ~ [0, 0, 0]
opt.score()    # objective value at that position, ~ 0

# Artificial Bee Colony works the same way:
AutoColony(mode="abc", n_iterations=200).fit(sphere, bounds=[(-5, 5)] * 3)
```

**Discrete optimization (ACO)** — find a short tour through a square distance
matrix (TSP):

```python
import numpy as np
from colonyx import AutoColony

distance_matrix = np.array([
    [0, 1, 9, 9, 1],
    [1, 0, 1, 9, 9],
    [9, 1, 0, 1, 9],
    [9, 9, 1, 0, 1],
    [1, 9, 9, 1, 0],
], dtype=float)

opt = AutoColony(mode="aco", n_iterations=100, random_state=42)
opt.fit(distance_matrix)

opt.predict()  # best tour, e.g. [0, 1, 2, 3, 4]
opt.score()    # tour length (lower is better)
```

Use `mode="auto"` to let colonyx pick ACO for a square matrix or PSO for an
objective function automatically.

## License

MIT
