# Firefly Algorithm

## Definition

The Firefly Algorithm (FA) is a continuous optimizer where each firefly's
brightness is its fitness. Every firefly is attracted to every *brighter*
neighbor, with attraction falling off with distance (`gamma` controls how
sharply); a firefly with no brighter neighbor just takes a small random
step. Because attraction is pairwise and distance-based rather than
funneled through a single global best, FA tends to explore multiple
promising regions at once — useful on multimodal landscapes.

## Pseudocode

```text
fireflies = random init inside bounds
scores = evaluate(fireflies)                          # batched, in parallel

for iteration in 1..n_iterations:
    for i in 1..n_fireflies:
        for j in 1..n_fireflies:
            if scores[j] >= scores[i]: continue        # j must be brighter
            dist = euclidean_distance(firefly[i], firefly[j])
            beta = beta0 * exp(-gamma * dist^2)         # attraction, decays with distance
            candidate = firefly[i]
                      + beta * (firefly[j] - firefly[i])
                      + fa_alpha * (random(0, 1) - 0.5) * range   # per dimension
            clamp candidate to bounds
            if evaluate(candidate) < scores[i]:
                firefly[i], scores[i] = candidate, evaluate(candidate)
    track best (firefly, score) seen so far
```

## Mathematical Formulation

Attractiveness decays with squared Euclidean distance \(r_{ij}\) between
fireflies \(i\) and \(j\):

$$
\beta(r_{ij}) = \beta_0\, e^{-\gamma r_{ij}^{2}}
$$

When firefly \(j\) is strictly brighter than firefly \(i\) (lower objective
value), \(i\) moves toward \(j\) plus a random-walk term scaled by that
dimension's bound range \(\text{range}_d = \text{upper}_d - \text{lower}_d\):

$$
x_{i,d} \leftarrow x_{i,d} + \beta(r_{ij})\,(x_{j,d} - x_{i,d}) + \text{fa_alpha}\,(u_d - 0.5)\,\text{range}_d, \qquad u_d \sim U(0,1)
$$

The moved candidate is clamped to `bounds` and kept only if it improves on
firefly \(i\)'s current score.

## Use when

- You want multimodal search behavior.
- You are tuning `beta0`, `gamma`, and step noise.

## API

- Rust class: `colonyx._colonyx.FireflyOptimizer`
- Python mode: `AutoColony(mode="fa")`

## Parameters

- `n_fireflies`
- `n_iterations`
- `beta0`, `gamma`, `fa_alpha`

!!! note
    `fa_alpha` (randomness step size) is named that way, not `alpha`, so it
    doesn't collide with other modes' own `alpha`-named parameters on
    `AutoColony`. The underlying Rust `FireflyOptimizer` constructor takes it
    as `alpha`.

## Example

```python
from colonyx import AutoColony

optimizer = AutoColony(mode="fa", n_iterations=100, random_state=7)
optimizer.fit(lambda x: sum(v * v for v in x), bounds=[(-5, 5), (-5, 5)])
print(optimizer.predict(), optimizer.score())
```
