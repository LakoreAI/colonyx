# Particle Swarm Optimization

`ParticleSwarm` is the default continuous optimizer.

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
```

## Notes

- The Rust core manages particle positions, velocities, and bests.
- Good general-purpose baseline for smooth continuous problems.
