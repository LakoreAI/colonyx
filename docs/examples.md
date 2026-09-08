---
title: "colonyx examples and gallery"
description: "Runnable colonyx examples: continuous and discrete optimization, benchmarking against known optima, and advanced algorithms like NSGA-II and permutation GA."
---

# Examples & Gallery

Every example on this page is a complete, runnable snippet against the real `colonyx` API — copy any block into a Python session with `colonyx` installed and it will run as-is. If you're new to `colonyx`, start with [Getting Started](getting-started.md) for the install steps and a slower walkthrough of the `fit`/`bounds` calling convention used throughout.

## Continuous, discrete, and auto mode

=== "Continuous"

    ```python
    from colonyx import AutoColony

    def sphere(x):
        return sum(value * value for value in x)

    optimizer = AutoColony(mode="pso", n_iterations=100, random_state=42)
    optimizer.fit(sphere, bounds=[(-5, 5), (-5, 5), (-5, 5)])
    print(optimizer.predict())
    print(optimizer.score())
    ```

    `mode="pso"` selects [Particle Swarm Optimization](algorithms/pso.md). `sphere` is a classic textbook test function whose global minimum is `0` at the origin, which makes it a good sanity check: after fitting, `optimizer.predict()` should return a point close to `[0, 0, 0]` and `optimizer.score()` a value close to `0`.

=== "Discrete"

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

    `mode="aco"` selects [Ant Colony Optimization](algorithms/aco.md), which expects `X` to be a square distance matrix rather than a callable. This particular matrix is constructed so the cheapest tour visits nodes in ring order — a good first check that ACO is finding the structure you'd expect by inspection.

=== "Auto"

    ```python
    from colonyx import AutoColony

    optimizer = AutoColony(mode="auto", n_iterations=100, random_state=7)
    optimizer.fit(lambda x: sum(value * value for value in x), bounds=[(-5, 5), (-5, 5)])
    ```

    `mode="auto"` defers the algorithm choice to `recommend_algorithm()`, which looks at whether `X` is callable versus a square matrix, and at problem dimensionality for continuous objectives, to pick a sensible default — see the [Algorithms overview](algorithms.md#how-to-choose) for the exact logic.

## Benchmarking against a known optimum

```python
from colonyx import AutoColony
from colonyx.benchmarks import benchmark_suite

problem = benchmark_suite()["rastrigin"]
optimizer = AutoColony(mode="de", n_iterations=200, random_state=7)
optimizer.fit(problem.objective, bounds=[problem.bounds[0]] * 5)

print(optimizer.score(), "vs known optimum", problem.minimum)
```

`benchmark_suite()` returns a dictionary of standard test functions used throughout the optimization literature — Rastrigin here is a deliberately deceptive one, with many regularly spaced local minima surrounding its single global minimum, which makes it a useful stress test for how well an algorithm avoids getting trapped. Comparing `optimizer.score()` directly against `problem.minimum` tells you the absolute optimization gap; for a normalized version of this comparison across many runs and multiple algorithms, see `AutoColony.performance_metrics()` and `optimization_gap()` in [AutoColony API](autocolony-api.md).

See [Benchmarking & Metrics](benchmarking.md) for comparing several
optimizers at once and running significance tests between them.

## Advanced algorithms

These are used directly as Rust-backed classes, not through `AutoColony`, because they don't fit the single-objective continuous-or-distance-matrix shape that `AutoColony(mode=...)` is built around. See the [Advanced Algorithms overview](algorithms/advanced.md) for when to reach for each one.

### Permutation GA

```python
from colonyx import PermutationGeneticOptimizer

distance_matrix = [
    [0.0, 1.0, 9.0, 9.0],
    [1.0, 0.0, 1.0, 9.0],
    [9.0, 1.0, 0.0, 1.0],
    [9.0, 9.0, 1.0, 0.0],
]

optimizer = PermutationGeneticOptimizer(n_individuals=40, n_iterations=100, random_state=7)
optimizer.fit(distance_matrix)
print(optimizer.predict(), optimizer.score())
```

[Permutation GA](algorithms/permutation-ga.md) solves the same kind of problem as ACO — a tour over a distance matrix — but through order-crossover genetic operators on a population of permutations instead of pheromone trails, which can be a useful alternative when you want to compare two structurally different search strategies on the same instance.

### NSGA-II

```python
from colonyx import Nsga2Optimizer

def objectives(x):
    return [sum(v * v for v in x), sum((v - 1.0) ** 2 for v in x)]

optimizer = Nsga2Optimizer(
    n_individuals=30,
    n_iterations=50,
    crossover_rate=0.9,
    mutation_rate=0.2,
    mutation_scale=0.1,
    archive_size=20,
    random_state=7,
)
optimizer.fit(objectives, lower=[0.0, 0.0], upper=[1.0, 1.0])
print(optimizer.predict())  # Pareto front: list of variable vectors
```

`objectives` here returns two competing scalar values instead of one — distance from the origin and distance from `1.0` in every dimension — so there's no single "best" point, only trade-offs between the two. [NSGA-II](algorithms/nsga2.md) handles this by returning `predict()` as a whole Pareto front of non-dominated solutions rather than one best vector, letting you pick a trade-off after the fact instead of collapsing the objectives into one weighted score up front.

### ACO variants

`AntColony` (the Rust-backed class) takes `variant` directly — there's no
`mode` argument on this class, unlike `AutoColony`:

```python
from colonyx import AntColony

optimizer = AntColony(n_ants=20, n_iterations=100, variant="mmas", random_state=7)
optimizer.fit(distance_matrix)
```

`variant="mmas"` selects Max-Min Ant System, one of four pheromone-update strategies `AntColony` supports directly (`basic`, `acs`, `elitist`, `mmas`) — see [ACO Variants](algorithms/aco-variants.md) for how each one changes convergence behavior and when to prefer it over the plain `basic` variant that `AutoColony(mode="aco")` uses by default.
