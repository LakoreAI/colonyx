# MOPSO

`MopsoOptimizer` is a compact multi-objective particle swarm optimizer.

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

## Use when

- You want PSO-style movement with a Pareto archive.
- You can provide a vector-valued objective and box bounds.

## API

- Rust class: `colonyx._colonyx.MopsoOptimizer`

## Parameters

- `n_particles`
- `n_iterations`
- `w`
- `c1`
- `c2`
- `mutation_scale`
- `archive_size`

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
