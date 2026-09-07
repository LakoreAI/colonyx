# Particle Swarm Optimization

## Definition

Particle Swarm Optimization (PSO) is a population-based continuous
optimizer: a swarm of particles, each with a position and velocity, moves
through the search space pulled toward two points — the particle's own
best-ever position and the swarm's best-ever position. This blend of
individual memory and social influence lets the swarm converge on good
regions of a smooth objective without gradient information.

## Pseudocode

```text
positions, velocities = random init inside bounds  # velocities scaled small
fitness = evaluate(positions)                       # batched, in parallel
personal_best, personal_best_fitness = positions, fitness
global_best, global_best_fitness = argmin(fitness)

for iteration in 1..n_iterations:
    for i in 1..n_particles:
        for each dimension d:
            r1, r2 = random(0, 1), random(0, 1)
            velocity[i][d] = w * velocity[i][d]
                           + c1 * r1 * (personal_best[i][d] - position[i][d])
                           + c2 * r2 * (global_best[d]      - position[i][d])
            position[i][d] += velocity[i][d]
        clamp position[i] to bounds
        fitness_i = evaluate(position[i])
        if fitness_i < personal_best_fitness[i]:
            personal_best[i], personal_best_fitness[i] = position[i], fitness_i
            if fitness_i < global_best_fitness:
                global_best, global_best_fitness = position[i], fitness_i
```

## Mathematical Formulation

For particle \(i\), dimension \(d\), at iteration \(t\), with independently
drawn \(r_1, r_2 \sim U(0,1)\) per dimension per particle per step:

$$
v_{i,d} \leftarrow w\,v_{i,d} + c_1 r_1 \,(p_{i,d} - x_{i,d}) + c_2 r_2\,(g_d - x_{i,d})
$$

$$
x_{i,d} \leftarrow x_{i,d} + v_{i,d}
$$

where \(p_i\) is particle \(i\)'s personal-best position and \(g\) is the
swarm's global-best position. \(x_i\) is then clamped back into
`bounds`, and \(p_i\)/\(g\) are updated whenever the clamped \(x_i\)
improves on them.

## Use when

- You have a callable objective function.
- You can provide box bounds for each dimension.

## API

- Rust class: `colonyx._colonyx.ParticleSwarm`
- Python mode: `AutoColony(mode="pso")`

## Parameters

- `n_particles`
- `n_iterations`
- `w`, `c1`, `c2`

## Example

```python
from colonyx import AutoColony

optimizer = AutoColony(mode="pso", n_iterations=100, random_state=7)
optimizer.fit(lambda x: sum(v * v for v in x), bounds=[(-5, 5), (-5, 5)])
print(optimizer.predict(), optimizer.score())
```

## Notes

- The Rust core manages particle positions, velocities, and bests.
- Good general-purpose baseline for smooth continuous problems.
