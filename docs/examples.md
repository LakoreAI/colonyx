# Examples & Gallery

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

=== "Auto"

    ```python
    from colonyx import AutoColony

    optimizer = AutoColony(mode="auto", n_iterations=100, random_state=7)
    optimizer.fit(lambda x: sum(value * value for value in x), bounds=[(-5, 5), (-5, 5)])
    ```

## Benchmarking against a known optimum

```python
from colonyx import AutoColony
from colonyx.benchmarks import benchmark_suite

problem = benchmark_suite()["rastrigin"]
optimizer = AutoColony(mode="de", n_iterations=200, random_state=7)
optimizer.fit(problem.objective, bounds=[problem.bounds[0]] * 5)

print(optimizer.score(), "vs known optimum", problem.minimum)
```

See [Benchmarking & Metrics](benchmarking.md) for comparing several
optimizers at once and running significance tests between them.

## Advanced algorithms

These are used directly as Rust-backed classes, not through `AutoColony`.

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

### ACO variants

`AntColony` (the Rust-backed class) takes `variant` directly — there's no
`mode` argument on this class, unlike `AutoColony`:

```python
from colonyx import AntColony

optimizer = AntColony(n_ants=20, n_iterations=100, variant="mmas", random_state=7)
optimizer.fit(distance_matrix)
```
