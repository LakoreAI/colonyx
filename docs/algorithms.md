---
title: "colonyx algorithms — full comparison"
description: "Compare all 16 colonyx optimization algorithms — PSO, ACO, ABC, GWO, DE, CMA-ES and more — by problem type, mechanism, key parameters, and when to use each."
---

# Algorithms

`colonyx` implements 16 swarm intelligence and metaheuristic optimization algorithms, all reachable through the unified `AutoColony(mode=...)` interface, or directly as Rust-backed classes from `colonyx` / `colonyx._colonyx` for the advanced and multi-objective algorithms that don't have an `AutoColony` mode. Every algorithm here belongs to the same broad family — population-based, gradient-free, nature-inspired search — but they differ in what kind of search space they were designed for (discrete versus continuous versus bit-vector versus multi-objective) and in the specific mechanism they use to balance exploring new regions of the space against exploiting the best regions found so far. This page is a map: use the tables below to find the right algorithm for your problem, then follow the link to its dedicated page for the full mechanism, parameter reference, and worked example.

## How to choose

If you already know your problem is a **discrete/combinatorial** one — a tour, a route, an ordering — reach for [ACO](algorithms/aco.md) or [Permutation GA](algorithms/permutation-ga.md). If it's a **continuous** black-box function over a bounded box, start with [PSO](algorithms/pso.md) for low dimensionality or [ABC](algorithms/abc.md) once dimensionality grows, which is exactly what `AutoColony(mode="auto")` does internally via `recommend_algorithm()`. If your continuous surface is known to be highly multi-modal (many local optima that trap simple hill-climbing), [Simulated Annealing](algorithms/sa.md), [Differential Evolution](algorithms/de.md), and [CMA-ES](algorithms/cmaes.md) tend to be more robust than a plain swarm. If you're optimizing **two or more competing objectives** at once rather than a single scalar, neither of the above applies — you want [NSGA-II](algorithms/nsga2.md) or [MOPSO](algorithms/mopso.md), which return a Pareto front instead of a single best point. And if your decision variables are inherently **binary** (feature selection, on/off switches, knapsack-style problems), [Binary PSO](algorithms/binary-pso.md) operates directly on bit vectors instead of forcing a continuous relaxation.

## Discrete

| Algorithm | Mode | Use case |
| --- | --- | --- |
| [ACO](algorithms/aco.md) | `aco` | TSP-style tours over a square distance matrix |
| `two_opt` | — | Local edge-reversal refinement, used internally by ACO (`use_two_opt=True`) |

## Continuous

Every row below is reachable via `AutoColony(mode=...)` with a callable objective and `bounds=[(low, high), ...]`. The "Key parameters" column lists the unified `AutoColony` attribute names and their defaults — see each algorithm's own page for what they control and how to tune them, and [AutoColony API](autocolony-api.md) for the complete parameter reference across all modes.

| Algorithm | Mode | Intuition | Key parameters (defaults) |
| --- | --- | --- | --- |
| [PSO](algorithms/pso.md) | `pso` | A swarm of particles fly through the space, each pulled toward its own best-ever position and the swarm's best-ever position | `n_particles=30`, `w=0.9`, `c1=2.0`, `c2=2.0` |
| [ABC](algorithms/abc.md) | `abc` | Employed, onlooker, and scout bees take turns exploiting known good food sources and abandoning exhausted ones | `n_bees=50`, `limit=10` |
| [GWO](algorithms/gwo.md) | `gwo` | The pack encircles prey by converging toward its three best-ranked wolves (alpha, beta, delta) each iteration | `n_wolves=30` |
| [FA](algorithms/fa.md) | `fa` | Dimmer fireflies move toward brighter ones, with attraction fading over distance | `n_fireflies=30`, `beta0=1.0`, `gamma=1.0`, `fa_alpha=0.2` |
| [SA](algorithms/sa.md) | `sa` | A single solution takes random steps, accepting worse moves with a probability that shrinks as a "temperature" cools | `initial_temperature=10.0`, `cooling_rate=0.95`, `step_scale=0.1` |
| [CS](algorithms/cs.md) | `cs` | Nests are replaced by Lévy-flight-generated candidates, with a fraction of the worst nests abandoned each round | `n_nests=25`, `pa=0.25`, `cs_alpha=0.01`, `levy_scale=1.0` |
| [BA](algorithms/ba.md) | `ba` | Bats vary emitted-pulse frequency and loudness as they home in on prey, echolocation-style | `n_bats=30`, `fmin=0.0`, `fmax=2.0`, `bat_alpha=0.9`, `bat_gamma=0.9`, `loudness=1.0`, `pulse_rate=0.5` |
| [GSO](algorithms/gso.md) | `gso` | Glowworms carry a luciferin "brightness" score and move toward brighter neighbors within a self-adjusting neighborhood radius | `n_worms=30`, `luciferin_decay=0.4`, `luciferin_enhancement=0.6`, `gso_step_size=0.1`, `neighborhood_radius=1.0` |
| [BFO](algorithms/bfo.md) | `bfo` | Bacteria alternate chemotaxis (swim/tumble toward nutrients), reproduction of the fittest half, and random elimination-dispersal | `n_bacteria=30`, `n_chemotactic_steps=10`, `n_reproduction_steps=4`, `elimination_probability=0.25`, `bfo_step_scale=0.1` |
| [DE](algorithms/de.md) | `de` | New candidates are formed by adding a scaled difference between two population members to a third, then crossing over with the target | `n_individuals=40`, `f=0.8`, `cr=0.9` |
| [CMA-ES](algorithms/cmaes.md) | `cmaes` | Samples from a multivariate Gaussian whose covariance is adapted each generation to follow the shape of the fitness landscape | `n_individuals=40`, `cmaes_sigma=0.5` |

## Advanced & multi-objective

These aren't `AutoColony` modes — use them as Rust-backed classes directly, imported straight from `colonyx`.

| Algorithm | Page | Solves |
| --- | --- | --- |
| `PermutationGeneticOptimizer` | [Permutation GA](algorithms/permutation-ga.md) | Permutation/TSP-style problems via order crossover |
| `BinaryParticleSwarm` | [Binary PSO](algorithms/binary-pso.md) | Bit-vector optimization |
| `Nsga2Optimizer` | [NSGA-II](algorithms/nsga2.md) | Multi-objective search via non-dominated sorting |
| `MopsoOptimizer` | [MOPSO](algorithms/mopso.md) | Multi-objective PSO with a Pareto archive |
| `AntColony` variants | [ACO Variants](algorithms/aco-variants.md) | `basic`, `acs`, `elitist`, `mmas` pheromone-update strategies |

See [Advanced Algorithms overview](algorithms/advanced.md) for a fuller introduction to when you'd reach for this group instead of an `AutoColony` mode.

## Implementation notes

- The Rust core owns every objective evaluation loop; Python only passes
  callables, bounds, and distance matrices in — this is the main reason `colonyx` doesn't pay Python's per-call interpreter overhead across thousands of fitness evaluations. See [Rust Usage](rust.md) for how the crate is organized.
- `AutoColony` chooses the backend (or accepts an explicit `mode`) and keeps
  sklearn-style metadata (`get_params`, `set_params`, `score_history_`, ...) — see [AutoColony API](autocolony-api.md).
- Advanced algorithms reuse the same Rust core types (`Bounds`, `Solution`,
  `Problem`) rather than a separate execution path; `Nsga2Optimizer` and
  `MopsoOptimizer` additionally implement a shared `MultiObjectiveOptimizer`
  trait — see [Rust Usage](rust.md).
- Population-based continuous algorithms parallelize independent fitness
  evaluations with Rayon where doing so doesn't change the algorithm's
  result — see [Benchmarking & Metrics](benchmarking.md) and [Rust Usage](rust.md).

## Frequently asked questions

### What's the difference between all these algorithms if they're all "swarm intelligence"?

They share the same population-based, gradient-free search loop, but each one encodes a different heuristic for balancing exploration (searching new regions) against exploitation (refining known good regions), usually borrowed from a specific natural or physical process — bird flocking for PSO, pheromone trails for ACO, bee foraging for ABC, simulated cooling for SA, and so on. In practice, the choice matters because different heuristics converge at different rates and get stuck in local optima with different frequencies depending on how rugged and high-dimensional your objective surface is — which is why `colonyx` gives you many of them behind one interface rather than betting everything on a single default.

### I don't know anything about my objective function — which mode should I start with?

Use `mode="auto"`. It inspects whether `X` is a callable or a square matrix, and for continuous objectives it uses dimensionality to pick [PSO](algorithms/pso.md) (low-dimensional) or [ABC](algorithms/abc.md) (higher-dimensional) as a reasonable default via `recommend_algorithm()`. Once you have a working baseline, come back to this page's comparison table and try one or two alternatives suited to your problem's specific shape.

### Are these algorithms guaranteed to find the global optimum?

No — like all metaheuristics, none of these algorithms carry a formal convergence guarantee to the global optimum on an arbitrary black-box function, and that's an inherent trade-off for giving up the differentiability requirement gradient-based methods rely on. What you get instead is a good empirical track record across many problem classes, tunable exploration/exploitation behavior, and the ability to run multiple seeds and compare results with the tools in [Benchmarking & Metrics](benchmarking.md), including paired and Wilcoxon significance tests for deciding whether one configuration is really better than another.
