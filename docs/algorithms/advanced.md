# Advanced Algorithms

These algorithms extend the core Rust-backed optimizer set.

## Definition

"Advanced" here means algorithms that solve a problem shape the 12
`AutoColony` modes don't cover directly: permutation/combinatorial search
with a genetic algorithm, binary (bit-vector) search with PSO dynamics, and
multi-objective search that returns a Pareto front/archive instead of a
single best point. They aren't `AutoColony` modes — you use each one as a
Rust-backed class directly — but they reuse the same core Rust types
(`Bounds`, `Solution`, `Problem`) as every other algorithm in colonyx.

## Pages

- `PermutationGeneticOptimizer` — `algorithms/permutation-ga.md`
- `BinaryParticleSwarm` — `algorithms/binary-pso.md`
- `Nsga2Optimizer` — `algorithms/nsga2.md`
- `MopsoOptimizer` — `algorithms/mopso.md`
- ACO variants — `algorithms/aco-variants.md`

## Summary

- `PermutationGeneticOptimizer` solves permutation/TSP-style problems.
- `BinaryParticleSwarm` solves bit-vector optimization problems.
- `Nsga2Optimizer` and `MopsoOptimizer` provide multi-objective search.
- `AntColony` supports `basic`, `acs`, `elitist`, and `mmas` variants.

## Rust API note

On the Rust side, `BinaryParticleSwarm` implements the same `Optimizer` trait
as the continuous/discrete algorithms, and `Nsga2Optimizer`/`MopsoOptimizer`
both implement a `MultiObjectiveOptimizer` trait (`fit()` against a
`MultiObjectiveProblem`, plus `pareto_front()`). This only matters if you're
writing Rust code that needs to hold several of these optimizers
polymorphically (e.g. `Vec<Box<dyn MultiObjectiveOptimizer>>`); from Python,
nothing here changes how you call these classes.
