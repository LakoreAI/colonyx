---
title: Advanced Optimization Algorithms
description: Permutation GA, Binary PSO, NSGA-II, MOPSO, and ACO variants in colonyx — specialized swarm and evolutionary algorithms for combinatorial, binary, and multi-objective search problems.
---

# Advanced Algorithms

The 12 modes behind `AutoColony` (see [Algorithms Overview](../algorithms.md)) cover the common case: a single-objective search over either a continuous box (`bounds=[(low, high), ...]`) or a square distance matrix. Real problems don't always fit that mold — sometimes a solution has to be a *permutation* rather than a vector, sometimes the decision variables are *binary* rather than continuous, and sometimes you're optimizing several conflicting objectives at once instead of one score. colonyx ships five more algorithms, all implemented in the same Rust core as the rest of the library, for exactly those cases. Because they don't share `AutoColony`'s unified parameter surface, you instantiate each one as its own class imported directly from `colonyx`.

!!! abstract "TL;DR"
    Five specialized optimizers for problem shapes the core `AutoColony` modes don't cover: permutations (`PermutationGeneticOptimizer`), binary vectors (`BinaryParticleSwarm`), and multi-objective search returning a Pareto front (`Nsga2Optimizer`, `MopsoOptimizer`). A fifth page documents the `variant` argument on `AntColony` itself (`basic`, `acs`, `elitist`, `mmas`). All are used as direct classes, not through `AutoColony(mode=...)`.

## The family at a glance

| Algorithm | Class | Problem shape | Returns |
|---|---|---|---|
| [Permutation GA](permutation-ga.md) | `PermutationGeneticOptimizer` | Combinatorial / TSP-style (a distance matrix) | One permutation (tour) |
| [Binary PSO](binary-pso.md) | `BinaryParticleSwarm` | Binary decision vector (feature masks, subset selection) | One bit vector |
| [NSGA-II](nsga2.md) | `Nsga2Optimizer` | Multi-objective, continuous box bounds | A Pareto front (many solutions) |
| [MOPSO](mopso.md) | `MopsoOptimizer` | Multi-objective, continuous box bounds | A Pareto archive (many solutions) |
| [ACO variants](aco-variants.md) | `AntColony(variant=...)` | Combinatorial / TSP-style, with tunable exploitation vs. exploration | One tour, under a different search strategy |

Two ways to read that table: rows 1 and 5 both solve TSP-style combinatorial problems, so if you already have a distance matrix, the question is genetic algorithm (`PermutationGeneticOptimizer`) vs. ant-colony variant (`AntColony(variant=...)`) — see each page's "When to use" section for the trade-off. Rows 3 and 4 both solve the same kind of problem — multiple objectives over a continuous box — via two different metaheuristic families (genetic algorithm vs. particle swarm); they're the closest head-to-head comparison in this set.

## What unifies these five

Despite the different encodings, every algorithm here reuses the same Rust building blocks as the rest of colonyx: the `Bounds` type for box-constrained search spaces, the `Solution`/`Problem` abstractions for representing candidates and their fitness, and — for the two multi-objective optimizers — a shared `archive_from_population` routine that performs Pareto-dominance filtering and crowding-distance truncation. On the Rust side, `BinaryParticleSwarm` implements the same `Optimizer` trait as the continuous/discrete single-objective algorithms, while `Nsga2Optimizer` and `MopsoOptimizer` both implement a `MultiObjectiveOptimizer` trait (`fit()` against a `MultiObjectiveProblem`, plus `pareto_front()`). This only matters if you're writing Rust code that needs to hold several optimizers polymorphically (e.g. `Vec<Box<dyn MultiObjectiveOptimizer>>`) — from Python, nothing here changes how you call these classes; each one follows the same `optimizer.fit(...)` / `optimizer.predict()` / `optimizer.score()` shape as every other algorithm in colonyx.

## Further reading

- [Algorithms Overview](../algorithms.md) — the 12 unified `AutoColony` modes.
- [AutoColony API](../autocolony-api.md) — the unified interface these five algorithms sit alongside.
- Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems.* University of Michigan Press — the genetic-algorithm foundations behind both permutation GA and NSGA-II.
