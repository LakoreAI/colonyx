<p align="center">
  <img src="colonyx-logo.svg" alt="Colonyx Logo" width="200" height="200">
</p>

# colonyx

`colonyx` is a Python optimization library with a Rust core.

It exposes a unified `AutoColony` interface for:

- `aco` for discrete problems such as TSP
- `pso`, `abc`, `gwo`, `fa`, `sa`, `cs`, `ba`, `gso`, `bfo`, and `de` for continuous optimization

## Design

- Rust contains the search algorithms and core problem types.
- Python provides the public interface, validation, and sklearn compatibility.
- The compiled extension is exposed as `colonyx._colonyx`.
- Rust users can also depend on the crate directly; see `docs/rust.md`.

## Quick example

```python
from colonyx import AutoColony

def sphere(x):
    return sum(value * value for value in x)

optimizer = AutoColony(mode="pso", n_iterations=100, random_state=42)
optimizer.fit(sphere, bounds=[(-5, 5), (-5, 5), (-5, 5)])

print(optimizer.predict())
print(optimizer.score())
```

## Next steps

- See `docs/autocolony-api.md` for the main sklearn-style interface.
- See `docs/cli.md` for the command-line interface.
- Use `mode="auto"` to let the library choose a backend.
- See `docs/rust.md` for Rust crate usage.
- See `docs/algorithms.md` for algorithm-level notes.
- See `docs/release.md` for local build and publish steps.

## By problem type

### Discrete

- `ACO` for square distance matrices
- `two_opt` for local tour refinement

### Continuous

- `PSO` for general-purpose continuous optimization
- `ABC` for population-based search with food sources
- `GWO`, `FA`, `SA`, `CS`, `BA`, `GSO`, `BFO`, `DE`, and `CMA-ES` for alternative heuristics
- `BinaryParticleSwarm`, `PermutationGeneticOptimizer`, `Nsga2Optimizer`, and
  `MopsoOptimizer` for advanced discrete and multi-objective search
- `AntColony` variants `basic`, `acs`, `elitist`, and `mmas`

### Interface

- `AutoColony` for sklearn-style usage and algorithm selection
