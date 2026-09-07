# Binary Particle Swarm

`BinaryParticleSwarm` is a bit-vector PSO variant for discrete search.

## Definition

The standard PSO velocity update rule (inertia + attraction to a particle's
own best and to the swarm's global best), but applied to bit positions
rather than real-valued coordinates: a particle's velocity is squashed
through a sigmoid to get a probability, and each bit is set to `1` with that
probability rather than moved continuously. This gives PSO-style dynamics
(momentum, social/cognitive attraction) over binary decision problems (e.g.
subset selection, feature masks) instead of real-valued search spaces.

## Pseudocode

```text
positions = n_particles random bit vectors of length `dimensions`
velocities = zeros
personal_best = positions, evaluated
global_best = the best-scoring particle so far

for iteration in 1..n_iterations:
    for each particle i:
        for each bit d:
            velocity[i][d] = w * velocity[i][d]
                + c1 * random() * (personal_best[i][d] - position[i][d])
                + c2 * random() * (global_best[d] - position[i][d])
            probability = sigmoid(velocity[i][d])
            position[i][d] = 1 if random() < probability else 0
        evaluate position[i]
        if better than personal_best[i]: personal_best[i] = position[i]
        if better than global_best: global_best = position[i]
    record global_best score in history

return global_best
```

## Mathematical Formulation

The velocity update is identical in shape to continuous PSO, applied
per-bit:

$$
v_{ij} \leftarrow w\,v_{ij} + c_1 r_1 (p_{ij} - x_{ij}) + c_2 r_2 (g_j - x_{ij}), \quad r_1, r_2 \sim U(0,1)
$$

Position, however, is not `x + v` — each bit is resampled from a Bernoulli
distribution whose probability is the sigmoid of the velocity:

$$
S(v_{ij}) = \frac{1}{1 + e^{-v_{ij}}}, \qquad
x_{ij} =
\begin{cases}
1 & \text{if } \text{rand}() < S(v_{ij}) \\
0 & \text{otherwise}
\end{cases}
$$

A large positive velocity pushes \(S(v_{ij}) \to 1\) (bit likely `1`); a
large negative velocity pushes it toward `0` — the same directional pull as
continuous PSO, just expressed as a probability instead of a displacement.

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
