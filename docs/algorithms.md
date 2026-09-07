# Algorithms

Every algorithm below is reachable through `AutoColony(mode=...)`, or
directly as a Rust-backed class from `colonyx._colonyx` /  `colonyx`.

## Discrete

| Algorithm | Mode | Use case |
| --- | --- | --- |
| [ACO](algorithms/aco.md) | `aco` | TSP-style tours over a square distance matrix |
| `two_opt` | — | Local edge-reversal refinement, used internally by ACO (`use_two_opt=True`) |

## Continuous

| Algorithm | Mode | Mechanism |
| --- | --- | --- |
| [PSO](algorithms/pso.md) | `pso` | Swarm with velocity updates toward personal/global bests |
| [ABC](algorithms/abc.md) | `abc` | Employed, onlooker, and scout bee phases over food sources |
| [GWO](algorithms/gwo.md) | `gwo` | Wolves converge toward alpha/beta/delta leaders |
| [FA](algorithms/fa.md) | `fa` | Fireflies move toward brighter neighbors |
| [SA](algorithms/sa.md) | `sa` | Single-solution probabilistic search with a cooling schedule |
| [CS](algorithms/cs.md) | `cs` | Lévy-flight steps and nest abandonment |
| [BA](algorithms/ba.md) | `ba` | Frequency, loudness, and pulse-rate tuning per bat |
| [GSO](algorithms/gso.md) | `gso` | Luciferin-driven dynamic neighborhoods |
| [BFO](algorithms/bfo.md) | `bfo` | Chemotaxis, reproduction, and elimination-dispersal |
| [DE](algorithms/de.md) | `de` | Mutation, crossover, and greedy selection |
| [CMA-ES](algorithms/cmaes.md) | `cmaes` | Diagonal covariance adaptation with real cumulative step-size control |

## Advanced & multi-objective

These aren't `AutoColony` modes — use them as Rust-backed classes directly.

| Algorithm | Page | Solves |
| --- | --- | --- |
| `PermutationGeneticOptimizer` | [Permutation GA](algorithms/permutation-ga.md) | Permutation/TSP-style problems via order crossover |
| `BinaryParticleSwarm` | [Binary PSO](algorithms/binary-pso.md) | Bit-vector optimization |
| `Nsga2Optimizer` | [NSGA-II](algorithms/nsga2.md) | Multi-objective search via non-dominated sorting |
| `MopsoOptimizer` | [MOPSO](algorithms/mopso.md) | Multi-objective PSO with a Pareto archive |
| `AntColony` variants | [ACO Variants](algorithms/aco-variants.md) | `basic`, `acs`, `elitist`, `mmas` pheromone-update strategies |

## Implementation notes

- The Rust core owns every objective evaluation loop; Python only passes
  callables, bounds, and distance matrices in.
- `AutoColony` chooses the backend (or accepts an explicit `mode`) and keeps
  sklearn-style metadata (`get_params`, `set_params`, `score_history_`, ...).
- Advanced algorithms reuse the same Rust core types (`Bounds`, `Solution`,
  `Problem`) rather than a separate execution path; `Nsga2Optimizer` and
  `MopsoOptimizer` additionally implement a shared `MultiObjectiveOptimizer`
  trait — see [Rust Usage](rust.md).
- Population-based continuous algorithms parallelize independent fitness
  evaluations with Rayon where doing so doesn't change the algorithm's
  result — see [Benchmarking & Metrics](benchmarking.md) and [Rust Usage](rust.md).
