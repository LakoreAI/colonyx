# AutoColony API Reference

`AutoColony` is the public sklearn-style interface for colonyx.

## Constructor

```python
AutoColony(
    mode="auto",
    n_iterations=100,
    random_state=None,
    ...
)
```

### Common arguments

- `mode`: backend selector; supported values include `aco`, `pso`, `abc`, `gwo`, `fa`, `sa`, `cs`, `ba`, `gso`, `bfo`, `de`, and `auto`
- `n_iterations`: iteration budget for the selected optimizer
- `random_state`: reproducibility seed

### Parameter table

| Parameter | Applies to | Meaning |
| --- | --- | --- |
| `n_ants`, `alpha`, `beta`, `rho`, `q`, `use_two_opt` | `aco` | Ant colony search behavior |
| `n_particles`, `w`, `c1`, `c2` | `pso` | Particle movement and attraction |
| `n_bees`, `limit` | `abc` | Food source population and abandonment |
| `n_wolves` | `gwo` | Wolf population size |
| `n_fireflies`, `beta0`, `gamma`, `fa_alpha` | `fa` | Firefly attraction and randomness |
| `initial_temperature`, `cooling_rate`, `step_scale` | `sa` | Annealing schedule and move scale |
| `n_nests`, `pa`, `levy_scale` | `cs` | Nest count and Lévy-flight exploration |
| `n_bats`, `fmin`, `fmax`, `bat_alpha`, `bat_gamma`, `loudness`, `pulse_rate` | `ba` | Bat motion and acceptance settings |
| `n_worms`, `luciferin_decay`, `luciferin_enhancement`, `gso_step_size`, `neighborhood_radius` | `gso` | Glowworm luciferin dynamics |
| `n_bacteria`, `n_chemotactic_steps`, `n_reproduction_steps`, `elimination_probability`, `bfo_step_scale` | `bfo` | Bacterial foraging schedule |
| `n_individuals`, `f`, `cr` | `de` | Mutation factor and crossover rate |
| `n_individuals`, `sigma` | `cmaes` | Diagonal covariance adaptation and step scale |

### Algorithm-specific arguments

- `n_ants`, `alpha`, `beta`, `rho`, `q`
- `n_particles`, `w`, `c1`, `c2`
- `n_bees`, `limit`
- `n_wolves`
- `n_fireflies`, `beta0`, `gamma`, `fa_alpha`
- `initial_temperature`, `cooling_rate`, `step_scale`
- `n_nests`, `pa`, `levy_scale`
- `n_bats`, `fmin`, `fmax`, `bat_alpha`, `bat_gamma`, `loudness`, `pulse_rate`
- `n_worms`, `luciferin_decay`, `luciferin_enhancement`, `gso_step_size`, `neighborhood_radius`
- `n_bacteria`, `n_chemotactic_steps`, `n_reproduction_steps`, `elimination_probability`, `bfo_step_scale`
- `n_individuals`, `f`, `cr`
- `n_individuals`, `sigma`

## Fit contract

```python
fit(X, y=None, bounds=None)
```

### Input forms

- `X` as a callable objective function for continuous modes
- `X` as a square distance matrix for ACO
- `X` and `y` as tabular data for sklearn compatibility

### Bounds

Continuous modes require `bounds=[(low, high), ...]`.

## Predict and score

- `predict()` returns the best position or tour found
- `score()` returns the best objective value or tour length
- sklearn compatibility mode keeps the estimator usable in pipelines

## Behavior

- `mode="auto"` selects ACO for discrete square matrices.
- `mode="auto"` selects PSO for continuous objectives by default.
- The Rust extension performs the actual optimization work.
- `AntColony` exposes `variant="basic" | "acs" | "elitist" | "mmas"` for ACO variants.

## Related objects

- `colonyx._colonyx.AntColony`
- `colonyx._colonyx.ParticleSwarm`
- `colonyx._colonyx.BeeColony`
- `colonyx._colonyx.GreyWolfOptimizer`
- `colonyx._colonyx.FireflyOptimizer`
- `colonyx._colonyx.SimulatedAnnealing`
- `colonyx._colonyx.CuckooSearch`
- `colonyx._colonyx.BatAlgorithm`
- `colonyx._colonyx.GlowwormOptimizer`
- `colonyx._colonyx.BacterialForagingOptimizer`
- `colonyx._colonyx.DifferentialEvolution`
- `colonyx._colonyx.CmaEsOptimizer`
- `colonyx._colonyx.BinaryParticleSwarm`
- `colonyx._colonyx.PermutationGeneticOptimizer`
- `colonyx._colonyx.Nsga2Optimizer`
- `colonyx._colonyx.MopsoOptimizer`
- `colonyx._colonyx.two_opt`
