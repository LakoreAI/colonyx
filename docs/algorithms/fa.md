---
title: Firefly Algorithm (FA) in Python
description: How colonyx implements the Firefly Algorithm (FA) for multimodal continuous optimization, with the attraction/distance math, parameters, and a runnable example.
---

# Firefly Algorithm (FA)

!!! abstract "TL;DR"
    The Firefly Algorithm is a continuous optimizer where every candidate solution is attracted to every *better* candidate, with attraction strength decaying by distance rather than funneled through a single global best — making it a strong choice for multimodal landscapes with several promising regions. In colonyx it's `AutoColony(mode="fa")`.

## What is the Firefly Algorithm?

The Firefly Algorithm is inspired by the flashing behavior real fireflies use to attract mates: each firefly's flash brightness signals its quality as a potential mate, brighter fireflies attract more attention, and that attraction fades the further away the observer is. FA treats each candidate solution as a firefly whose "brightness" is simply how good its objective value is, and simulates the same distance-decaying attraction to drive the search — every firefly is pulled toward every firefly that's brighter than it, with the pull weakening the farther apart they are.

This pairwise, all-to-all attraction is what structurally distinguishes FA from swarm optimizers like [PSO](pso.md) or [GWO](gwo.md), where every particle is ultimately pulled toward one shared best point (or a small handful of leaders). In FA, a firefly sitting near a good-but-not-global-best region still gets pulled toward other nearby brighter fireflies in that same region, even while a completely different cluster of fireflies is independently converging on a different promising region elsewhere in the search space. Because attraction decays with distance rather than being global, FA naturally tends to maintain multiple simultaneous search threads across a multimodal landscape instead of immediately collapsing everything onto whichever region currently looks best — and a firefly with no brighter neighbor nearby simply takes a small random step instead, keeping the search from stalling.

## How colonyx implements it

The core loop colonyx's Rust implementation (`colonyx._colonyx.FireflyOptimizer`) runs is:

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

Because every firefly is compared against every other firefly each iteration, FA's per-iteration cost scales quadratically with `n_fireflies` — keep that in mind before setting the swarm size very large.

### The distance-decaying attraction math

Attractiveness decays with squared Euclidean distance \(r_{ij}\) between fireflies \(i\) and \(j\):

$$
\beta(r_{ij}) = \beta_0\, e^{-\gamma r_{ij}^{2}}
$$

When firefly \(j\) is strictly brighter than firefly \(i\) (lower objective value), \(i\) moves toward \(j\) plus a random-walk term scaled by that dimension's bound range \(\text{range}_d = \text{upper}_d - \text{lower}_d\), where \(\alpha\) is `fa_alpha`, the randomization step-size factor:

$$
x_{i,d} \leftarrow x_{i,d} + \beta(r_{ij})\,(x_{j,d} - x_{i,d}) + \alpha\,(u_d - 0.5)\,\text{range}_d, \qquad u_d \sim U(0,1)
$$

The moved candidate is clamped to `bounds` and kept only if it improves on firefly \(i\)'s current score.

## When to use it (and when not to)

Reach for FA when your objective is multimodal — it has several local optima and you'd like the search to keep multiple promising regions alive simultaneously rather than converging early onto whichever region looks best first, which is the failure mode a single-global-best optimizer like [PSO](pso.md) is more prone to. FA's tunable attraction-decay (`gamma`) also gives you direct control over how "clustered" versus "global" the pairwise attraction behaves. Its main cost is the quadratic pairwise comparison per iteration, so it scales worse than PSO or [GWO](gwo.md) to very large swarms; for high-dimensional problems where you'd rather not pay that quadratic cost, consider [Artificial Bee Colony](abc.md) instead. FA does not handle discrete tour/permutation problems — see [ACO](aco.md) for those.

## Parameters

| Parameter | Default | Meaning | Tuning guidance |
|---|---|---|---|
| `n_fireflies` | `30` | Number of fireflies in the population. | More fireflies explore more of the space but cost quadratically more per iteration (all-pairs comparison); 20-40 is a reasonable range before that cost adds up. |
| `beta0` | `1.0` | Attractiveness at zero distance. | Higher values make nearby brighter fireflies pull harder; rarely needs much tuning away from the default. |
| `gamma` | `1.0` | Light absorption coefficient — controls how fast attraction decays with distance. | Lower values let attraction reach farther (more global pull, faster convergence, less multimodal spread); higher values make attraction very short-range (more local clustering, better multimodal coverage but slower convergence). |
| `fa_alpha` | `0.2` | Randomization step-size factor (unified attribute name; the underlying Rust `FireflyOptimizer` constructor takes it as `alpha`). | Higher values add more exploratory noise to every move; lower it as the run progresses (or just set it lower from the start) if you want tighter late-stage convergence. |
| `n_iterations` | `100` | Number of iterations to run. | Increase for harder or higher-dimensional objectives. |

!!! note
    `fa_alpha` is named that way, not `alpha`, so it doesn't collide with other modes' own `alpha`-named parameters on the shared `AutoColony` interface (ACO's pheromone-importance `alpha`, Cuckoo Search's step-size `cs_alpha`, and Bat Algorithm's loudness-decay `bat_alpha` follow the same disambiguation pattern).

## Example

```python
from colonyx import AutoColony

optimizer = AutoColony(mode="fa", n_iterations=100, random_state=7)
optimizer.fit(lambda x: sum(v * v for v in x), bounds=[(-5, 5), (-5, 5)])
print(optimizer.predict(), optimizer.score())
```

## Further reading

- Yang, X.-S. (2008). *Nature-Inspired Metaheuristic Algorithms.* Luniver Press — the original Firefly Algorithm.
- Yang, X.-S. (2009). *Firefly algorithms for multimodal optimization.* Stochastic Algorithms: Foundations and Applications (SAGA 2009) — the multimodal-optimization framing this page draws on.
- [Algorithms overview](../algorithms.md) — compare FA against every other algorithm colonyx ships.
- [AutoColony API reference](../autocolony-api.md) — the full unified interface.
