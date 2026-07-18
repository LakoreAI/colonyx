# API

## `AutoColony`

Main entry point for optimization.

Key arguments:

- `mode`
- `n_iterations`
- `random_state`
- algorithm-specific parameters such as `n_particles`, `n_bees`, or `n_individuals`

Methods:

- `fit(X, y=None, bounds=None)`
- `predict()`
- `score()`
- `get_params()`
- `set_params()`

## Rust-backed exports

Available from `colonyx._colonyx`:

- `AntColony`
- `ParticleSwarm`
- `BeeColony`
- `GreyWolfOptimizer`
- `FireflyOptimizer`
- `SimulatedAnnealing`
- `CuckooSearch`
- `BatAlgorithm`
- `GlowwormOptimizer`
- `BacterialForagingOptimizer`
- `DifferentialEvolution`
- `CmaEsOptimizer`
- `PermutationGeneticOptimizer`
- `BinaryParticleSwarm`
- `Nsga2Optimizer`
- `MopsoOptimizer`
- `two_opt`

## Return values

- Continuous optimizers return the best position and best score.
- ACO returns the best tour and tour length.
- Multi-objective optimizers return a Pareto archive via `predict()`.
