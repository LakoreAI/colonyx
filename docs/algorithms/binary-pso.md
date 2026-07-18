# Binary Particle Swarm

`BinaryParticleSwarm` is a bit-vector PSO variant for discrete search.

## Use when

- You want PSO-style dynamics on binary decisions.
- You can encode the problem as a binary vector objective.

## API

- Rust class: `colonyx._colonyx.BinaryParticleSwarm`

## Parameters

- `n_particles`
- `n_iterations`
- `w`
- `c1`
- `c2`

## Example

```python
from colonyx import BinaryParticleSwarm

def objective(bits):
    return sum(bits)

optimizer = BinaryParticleSwarm(n_particles=30, n_iterations=100, random_state=7)
optimizer.fit(objective, lower=[0.0] * 10, upper=[1.0] * 10)
print(optimizer.predict())
print(optimizer.score())
```
