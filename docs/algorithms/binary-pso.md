---
title: Binary Particle Swarm Optimization
description: BinaryParticleSwarm in colonyx — a sigmoid-transformed PSO variant for bit-vector optimization problems like feature selection and subset selection.
---

# Binary Particle Swarm (Binary PSO)

`BinaryParticleSwarm` is a bit-vector PSO variant for discrete search.

!!! abstract "TL;DR"
    [PSO](pso.md)'s velocity update, applied per-bit and squashed through a sigmoid so each bit is resampled as 0 or 1 instead of moved continuously. Use it when your decision variables are binary — feature selection, knapsack-style subset problems — and you want PSO-style social/cognitive dynamics instead of a bitwise genetic algorithm.

## What is Binary PSO?

Continuous PSO moves particles through real-valued space by adding a velocity to a position each iteration — that update has no meaning for a bit vector, since "position + velocity" isn't a bit. Kennedy and Eberhart's binary PSO keeps the same velocity equation (inertia, pull toward a particle's own best position, pull toward the swarm's global best) but reinterprets velocity as a *probability signal* rather than a displacement: each per-bit velocity is passed through a sigmoid function to squash it into `(0, 1)`, and the corresponding bit is then resampled as `1` with that probability. A particle whose velocity for a given bit keeps growing positive will, over iterations, almost always set that bit to `1`; a strongly negative velocity drives it toward `0`. This gives you the same momentum and swarm-communication dynamics that make continuous PSO effective, applied to problems where the natural encoding is a bit vector rather than a point in space.

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

## When to use it (and when not to)

- You want PSO-style dynamics — momentum, social and cognitive attraction — on a problem whose decision variables are inherently binary (feature masks, item-selection/knapsack-style problems, on/off switches).
- You can express the problem as an objective function over a bit vector.
- Compared to continuous [PSO](pso.md): use `BinaryParticleSwarm` when the variables truly are binary; don't round continuous PSO's output to 0/1 as a substitute, since the search dynamics (sigmoid-probability resampling vs. continuous displacement) are genuinely different and Binary PSO is tuned for the discrete case.
- Compared to a bitwise genetic algorithm: Binary PSO tends to converge faster on smoother binary landscapes thanks to the social/cognitive pull, while a GA's crossover can explore more disruptively — try both if convergence stalls.

## API

- Rust class: `colonyx._colonyx.BinaryParticleSwarm`

## Parameters

| Parameter | Default | Meaning | Tuning notes |
|---|---|---|---|
| `n_particles` | `30` | Swarm size. | More particles cover more of the bit-vector space per iteration; scale with the number of dimensions. |
| `n_iterations` | `100` | Number of update steps. | Increase for higher-dimensional bit vectors or slow convergence. |
| `w` | `0.7` | Inertia weight — how much of the previous velocity carries forward. | Higher values favor exploration (bits flip more freely); lower values favor exploitation. |
| `c1` | `1.5` | Cognitive coefficient — pull toward a particle's own best-known bit vector. | Raise it if particles should trust their own history more than the swarm. |
| `c2` | `1.5` | Social coefficient — pull toward the swarm's global best. | Raise it for faster convergence toward the current best at the cost of diversity. |
| `random_state` | `None` | Seed for reproducible runs. | Set an integer for deterministic results. |

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

## Further reading

- Kennedy, J., & Eberhart, R. C. (1997). *A Discrete Binary Version of the Particle Swarm Algorithm.* IEEE International Conference on Systems, Man, and Cybernetics.
- [PSO](pso.md) — the continuous-space algorithm this variant reinterprets for binary decisions.
- [Advanced Algorithms](advanced.md) · [Algorithms Overview](../algorithms.md)
