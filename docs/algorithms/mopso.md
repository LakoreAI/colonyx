---
title: MOPSO Multi-Objective Particle Swarm
description: MopsoOptimizer in colonyx — multi-objective particle swarm optimization using a Pareto archive and leader selection for problems with conflicting objectives.
---

# Multi-Objective Particle Swarm Optimization (MOPSO)

`MopsoOptimizer` is a compact multi-objective particle swarm optimizer.

!!! abstract "TL;DR"
    [PSO](pso.md) adapted for multiple conflicting objectives: instead of one global best, particles follow a randomly chosen "leader" from an archive of non-dominated (Pareto-optimal) solutions, and a personal best is only replaced when a new position strictly dominates it. Returns a Pareto archive rather than a single point. Compare against [NSGA-II](nsga2.md), the genetic-algorithm counterpart for the same problem class.

## What is MOPSO?

Standard PSO relies on a single scalar fitness value to decide two things: which of a particle's own visited positions was best (its personal best), and which position in the whole swarm was best overall (the global best that pulls every particle toward it). Neither concept survives contact with multiple objectives directly, because "best" isn't well-defined when two solutions each win on a different objective. MOPSO keeps PSO's core velocity/position update but replaces both fitness comparisons with Pareto dominance: a candidate replaces a particle's personal best only if it *dominates* it (no worse on every objective, strictly better on at least one — the same relation NSGA-II uses), and instead of one fixed global best, each particle is pulled toward a **leader** sampled from an archive of non-dominated solutions found so far. That archive is itself maintained across iterations, growing as new non-dominated points are found and pruned back to `archive_size` by crowding distance when it overflows. The result is PSO's fast, momentum-driven search behavior redirected toward populating a whole Pareto front instead of homing in on one point.

## Definition

Multi-Objective Particle Swarm Optimization adapts PSO to vector-valued
objectives by replacing the single "global best" with an archive of
non-dominated solutions found so far. Each particle is pulled toward its own
best-known position and toward a leader sampled from that archive (rather
than one fixed global best), and a particle's personal best is only replaced
when a new position *dominates* it — not merely scores better on one
objective.

## Pseudocode

```text
positions = n_particles random candidates within bounds
personal_best = positions, evaluated
archive = non_dominated(personal_best), capped to archive_size

for iteration in 1..n_iterations:
    for each particle i:
        leader = a candidate from the archive's non-dominated front (random choice among ties)
        for each dimension d:
            velocity[i][d] = w * velocity[i][d]
                + c1 * random() * (personal_best[i][d] - position[i][d])
                + c2 * random() * (leader[d] - position[i][d])
            position[i][d] += velocity[i][d]
        clamp position[i] to bounds
        evaluate position[i]
        if position[i] dominates personal_best[i]: personal_best[i] = position[i]

    archive = non_dominated(archive + personal_best), capped to archive_size

return archive   # the Pareto archive
```

!!! note "`mutation_scale` is currently unused"
    `MopsoOptimizer` accepts a `mutation_scale` parameter (matching
    `Nsga2Optimizer`'s constructor shape), but verified against the current
    Rust source, it isn't applied anywhere in `fit_with_objective` — there is
    no mutation step in the velocity/position update above. Setting it has
    no effect on the search.

## Mathematical Formulation

The velocity/position update is the standard PSO rule, but the attraction
targets are dominance-based rather than scalar-fitness-based:

$$
v_{id} \leftarrow w\,v_{id} + c_1 r_1 (p_{id} - x_{id}) + c_2 r_2 (\ell_d - x_{id}), \qquad x_{id} \leftarrow x_{id} + v_{id}
$$

where \(p_i\) is particle \(i\)'s personal best and \(\ell\) is a **leader**
drawn uniformly at random from the current non-dominated front of the
archive (not one fixed global best):

$$
\ell \sim U\bigl(\text{non_dominated}(\text{archive})\bigr)
$$

Personal best is replaced only under strict Pareto dominance, using the
same relation as NSGA-II:

$$
p_i \leftarrow x_i \quad \text{iff} \quad x_i \prec p_i
$$

After every iteration the archive is rebuilt from
\(\text{archive} \cup \{p_i\}_{i=1}^{n}\), keeping only non-dominated points
and truncating by crowding distance when it exceeds `archive_size` — the
same `archive_from_population` logic NSGA-II's front-filling uses. There is
no mutation term in this update (see the note above on `mutation_scale`).

## When to use it (and when not to)

- You want PSO-style movement — fast, momentum-driven convergence — applied to a multi-objective problem, and you can provide a vector-valued objective plus box bounds.
- Compared to [NSGA-II](nsga2.md): both solve the same problem class from the same input shape; MOPSO's swarm dynamics often converge faster toward the front, while NSGA-II's crossover/mutation and crowding-distance selection are specifically tuned for even front spread — for a small number of objectives (2–3), try both and compare the resulting archives.
- Skip it for single-objective problems (use a [core `AutoColony` mode](../algorithms.md) instead) — dominance-based comparisons add overhead that buys nothing when there's only one objective to compare on.

## API

- Rust class: `colonyx._colonyx.MopsoOptimizer`

## Parameters

| Parameter | Default | Meaning | Tuning notes |
|---|---|---|---|
| `n_particles` | `30` | Swarm size. | More particles give denser Pareto-archive coverage per iteration. |
| `n_iterations` | `100` | Number of update steps. | Increase until the archive stops visibly improving between runs. |
| `w` | `0.7` | Inertia weight. | Higher favors exploration, lower favors exploitation, same as continuous PSO. |
| `c1` | `1.5` | Cognitive coefficient — pull toward a particle's own (dominance-based) personal best. | Raise it to trust each particle's own history more. |
| `c2` | `1.5` | Social coefficient — pull toward the sampled archive leader. | Raise it for faster convergence toward the current front at the cost of diversity. |
| `mutation_scale` | `0.1` | Accepted by the constructor for parity with `Nsga2Optimizer`, but **currently has no effect** — see the note above; there is no mutation step in the velocity/position update. | Leave at the default; changing it does nothing today. |
| `archive_size` | `50` | Maximum number of solutions kept in the returned Pareto archive. | Raise it for finer front resolution. |
| `random_state` | `None` | Seed for reproducible runs. | Set an integer for deterministic results. |

## Example

```python
from colonyx import MopsoOptimizer

def objectives(x):
    return [sum(value * value for value in x), sum((value - 1.0) ** 2 for value in x)]

optimizer = MopsoOptimizer(n_particles=30, n_iterations=50, random_state=42)
optimizer.fit(objectives, lower=[0.0, 0.0], upper=[1.0, 1.0])
print(optimizer.predict())
```

!!! warning
    `score()` reports the 2-objective hypervolume of the Pareto archive, so it
    requires `objectives(x)` to return at least 2 values. It raises
    `ValueError` if `fit()` was called with a single-objective function —
    `predict()` (the Pareto archive itself) still works either way.

## Further reading

- Coello Coello, C. A., & Lechuga, M. S. (2002). *MOPSO: A Proposal for Multiple Objective Particle Swarm Optimization.* Proceedings of the 2002 Congress on Evolutionary Computation.
- Coello Coello, C. A., Pulido, G. T., & Lechuga, M. S. (2004). *Handling Multiple Objectives with Particle Swarm Optimization.* IEEE Transactions on Evolutionary Computation, 8(3), 256–279.
- [NSGA-II](nsga2.md) — the genetic-algorithm counterpart for the same multi-objective problem class.
- [Advanced Algorithms](advanced.md) · [Algorithms Overview](../algorithms.md)
