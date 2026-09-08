---
title: Glowworm Swarm Optimization (GSO)
description: Glowworm Swarm Optimization in colonyx — a multimodal continuous optimizer using luciferin values and local neighborhoods. How it works, parameters, and an example.
---

# Glowworm Swarm Optimization (GSO)

!!! abstract "TL;DR"
    Glowworm Swarm Optimization models each candidate as a glowworm carrying a luciferin (light) value proportional to its fitness, moving toward brighter neighbors within a local radius rather than a single global best. Its lack of a shared global attractor lets subgroups of the swarm converge on different local optima at once, making it a strong choice for multimodal problems where you want more than one good solution, not just the single best. Run it in colonyx with `AutoColony(mode="gso")`.

## What is Glowworm Swarm Optimization?

Real glowworms glow using a chemical called luciferin, and the brightness of that glow can vary from one individual to the next. GSO turns this into a search strategy: every candidate solution is a "glowworm," and its luciferin value is updated each iteration to track how good its current position is — a lower (better) objective value produces a larger luciferin increment. Each glowworm then looks around a local neighborhood, defined by a `neighborhood_radius`, for other glowworms that are glowing brighter than it is, and takes a step toward the brightest one it can see. Glowworms that see no brighter neighbor within range simply hold their position for that iteration, aside from the luciferin update. This local, radius-bounded attraction is the key structural difference from swarm algorithms like PSO or ABC that pull the whole population toward one shared best position: because a glowworm can only be influenced by what's actually near it, the population can naturally split into several subgroups, each converging on a different local optimum, instead of collapsing onto a single point.

## How colonyx implements it

The Rust implementation in `src/algorithms/continuous.rs` updates luciferin values for the whole population before computing any moves, then moves each glowworm toward the brightest neighbor it can see:

```text
worms[1..n_worms] <- random positions within bounds
luciferin[i] <- 1.0 for all worms
scores <- evaluate(worms)
best <- worm with the lowest score

for iteration in 1..n_iterations:
    for each worm i:
        luciferin[i] <- (1 - luciferin_decay) * luciferin[i]
                         + luciferin_enhancement / (scores[i] + 1)

    for each worm i:
        neighbors <- worms j != i where luciferin[j] > luciferin[i]
                     and distance(worm_i, worm_j) <= neighborhood_radius
        if neighbors is non-empty:
            target <- brightest-scoring neighbor in `neighbors`
            candidate <- worm_i + gso_step_size * (target - worm_i)
            if score(candidate) < scores[i]:
                worm_i <- candidate; scores[i] <- score(candidate)
        update best if scores[i] improves on it
```

$$
\ell_i \leftarrow (1-\text{luciferin\_decay})\,\ell_i + \frac{\text{luciferin\_enhancement}}{s_i + 1}
$$

where \(s_i\) is worm \(i\)'s objective value. The neighbor set and target follow directly from luciferin and distance:

$$
N_i = \{\, j \neq i : \ell_j > \ell_i,\ \lVert x_i - x_j \rVert \le \text{neighborhood\_radius} \,\}
$$

$$
j^* = \operatorname*{arg\,min}_{j \in N_i} s_j
$$

\(j^*\) is the single best-scoring neighbor in range in this implementation, rather than a luciferin-proportional probabilistic choice as used in some GSO variants in the literature. The move toward it is accepted greedily, only if it actually improves the glowworm's score:

$$
x_i \leftarrow \operatorname{clamp}\!\big(x_i + \text{gso\_step\_size}\,(x_{j^*} - x_i)\big) \ \text{ if it improves } s_i
$$

## When to use it (and when not to)

Reach for GSO specifically when your objective is multimodal and you care about finding several distinct good solutions rather than a single global optimum — its neighborhood-bounded attraction is what makes subgroup formation possible, something single-global-best algorithms like [PSO](pso.md) or the [Artificial Bee Colony](abc.md) cannot do without extra machinery. If you only need one best answer and speed matters more than diversity of solutions, PSO or [Differential Evolution](de.md) will typically converge faster with fewer parameters to tune. GSO's behavior is quite sensitive to `neighborhood_radius`: too small and glowworms rarely see a brighter neighbor at all, effectively freezing the search; too large and it behaves closer to a single-swarm algorithm, losing the multimodal advantage that is GSO's main reason to exist.

## Parameters

| Parameter | Default | Meaning | Tuning guidance |
|---|---|---|---|
| `n_worms` | `30` | Population size — number of glowworms maintained per generation. | Increase for higher-dimensional or more multimodal objectives; more worms means more chances to discover separate local optima. |
| `luciferin_decay` | `0.4` | Fraction of each worm's luciferin value that decays away each iteration. | Higher values make luciferin track recent fitness more tightly; lower values give it more memory of past performance. |
| `luciferin_enhancement` | `0.6` | Scale applied to the fitness-based luciferin increment. | Raise it to make brightness differences between worms more pronounced, sharpening the pull toward better neighbors. |
| `gso_step_size` | `0.1` | Fraction of the distance to the brightest neighbor covered per move (passed to the Rust constructor as `step_size`). | Larger values move faster toward a bright neighbor but can overshoot; smaller values converge more cautiously. |
| `neighborhood_radius` | `1.0` | Maximum distance at which one glowworm can see and be attracted to another. | The most impactful parameter to tune — scale it relative to your bounds' range; too small isolates worms, too large collapses GSO's multimodal behavior. |
| `n_iterations` | `100` | Number of generations to run. | Increase for harder or higher-dimensional problems. |

## Example

```python
from colonyx import AutoColony

def sphere(x):
    return sum(xi * xi for xi in x)

optimizer = AutoColony(mode="gso", n_iterations=100, neighborhood_radius=1.0, random_state=7)
optimizer.fit(sphere, bounds=[(-5, 5), (-5, 5)])

optimizer.predict()  # best position found, ~ [0, 0]
optimizer.score()    # objective value at that position, ~ 0
```

## Further reading

- Krishnanand, K.N. and Ghose, D. (2005/2006). *Glowworm Swarm Optimization: A New Method for Optimizing Multi-modal Functions*.
- [Algorithms overview](../algorithms.md) — compare GSO against every other algorithm colonyx ships.
- [AutoColony API reference](../autocolony-api.md) — the unified `mode=` interface used above.
