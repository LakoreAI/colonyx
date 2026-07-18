# Algorithms

## Discrete

- `ACO` builds tours over a distance matrix.
- `two_opt` refines a tour with local edge reversal.

## Continuous

- `PSO` maintains a swarm with velocity updates.
- `ABC` explores food sources with employed, onlooker, and scout phases.
- `GWO` updates wolves using alpha, beta, and delta leaders.
- `FA` moves fireflies toward brighter neighbors.
- `SA` performs probabilistic single-solution search.
- `CS` uses Lévy-flight steps and nest abandonment.
- `BA` combines frequency, loudness, and pulse rate.
- `GSO` models luciferin-driven neighborhood movement.
- `BFO` uses chemotaxis, reproduction, and elimination.
- `DE` applies mutation, crossover, and greedy selection.
- `CMA-ES` adapts a diagonal covariance model over generations.
- Advanced pages:
  - `PermutationGeneticOptimizer` — `algorithms/permutation-ga.md`
  - `BinaryParticleSwarm` — `algorithms/binary-pso.md`
  - `Nsga2Optimizer` — `algorithms/nsga2.md`
  - `MopsoOptimizer` — `algorithms/mopso.md`
  - `AntColony` variants — `algorithms/aco-variants.md`

## Implementation notes

- The Rust core owns all objective evaluation loops.
- Python only passes callables, bounds, and distance matrices into Rust.
- `AutoColony` chooses the backend and keeps sklearn-style metadata.
- Advanced algorithms reuse the same Rust core types (`Bounds`, `Solution`,
  `Problem`) rather than adding a separate execution path.
