---
title: Differential Evolution (DE)
description: Differential Evolution in colonyx — a strong general-purpose continuous optimizer using mutation, crossover, and greedy steady-state selection. Explained with an example.
---

# Differential Evolution (DE)

!!! abstract "TL;DR"
    Differential Evolution builds a trial vector for every individual from the scaled difference of two other individuals, mixes it into the target via crossover, and keeps whichever is better — immediately, in a steady-state loop rather than a synchronized generational swap. It's one of the strongest, simplest general-purpose continuous optimizers available, and a solid default before reaching for something more specialized. Run it in colonyx with `AutoColony(mode="de")`.

## What is Differential Evolution?

Differential Evolution optimizes a population of candidate vectors by exploiting the differences between them directly. For each individual in the population — the "target" — the algorithm picks three other distinct individuals at random, computes the vector difference between two of them, scales that difference, and adds it to the third to form a "mutant" vector. This mutant is then mixed into a copy of the target through crossover, swapping in the mutant's values at randomly chosen dimensions to produce a "trial" vector. If the trial scores at least as well as the target it was built from, it replaces the target outright. The elegance of DE is that the scale and direction of its mutation step come entirely from the current spread of the population itself — as the population converges, differences between individuals shrink automatically, so DE naturally transitions from broad exploration early in a run to fine local refinement later, without any explicit cooling schedule or step-size parameter to hand-tune.

## How colonyx implements it

The Rust implementation in `src/algorithms/continuous.rs` runs DE as a steady-state loop: a trial replaces its target the moment it's found to be at least as good, rather than waiting for a full generation to complete before any replacements happen.

```text
population[1..n_individuals] <- random positions within bounds
scores <- evaluate(population)
best <- individual with the lowest score

for iteration in 1..n_iterations:
    for each target individual i:
        a, b, c <- three distinct individuals != i, chosen at random
        mutant <- population[a] + f * (population[b] - population[c])

        trial <- copy of population[i]
        pick one random dimension to force-copy from `mutant` (ensures
        the trial differs from the target in at least one dimension)
        for each dimension d:
            if random() < cr or d is the forced dimension:
                trial_d <- mutant_d
            clamp trial_d to bounds

        if score(trial) <= scores[i]:
            population[i] <- trial          # replaces the target immediately
            scores[i] <- score(trial)
            update best if it improves on it
```

The classic DE/rand/1 mutation, built from three distinct individuals other than the target:

$$
v = x_{r_1} + f \cdot (x_{r_2} - x_{r_3})
$$

Binomial crossover, with one dimension \(j_{\text{rand}}\) always forced from the mutant so the trial can never equal the target exactly:

$$
u_j =
\begin{cases}
v_j & \text{if } \operatorname{rand}_j < cr \text{ or } j = j_{\text{rand}} \\
x_{ij} & \text{otherwise}
\end{cases}
$$

Selection is greedy and immediate:

$$
x_i \leftarrow u \quad \text{if } f(u) \le f(x_i)
$$

!!! warning "Steady-state, not generational"
    Because replacement happens immediately rather than at the end of a synchronized generation, this loop cannot be parallelized across individuals without changing its results: a later target's `a`/`b`/`c` draw can land on an individual that was already mutated earlier in the same pass over the population. This matches the standard steady-state DE variant, but it's worth knowing if you're comparing convergence curves against a generational DE implementation elsewhere.

## When to use it (and when not to)

Differential Evolution is a strong first choice for general continuous optimization: it has only three parameters, no problem-specific tuning of a cooling schedule or neighborhood structure, and is competitive with more specialized algorithms across a wide range of objective shapes. Start here before reaching for [Cuckoo Search](cs.md) or [Bacterial Foraging Optimization](bfo.md), both of which need more parameter tuning for comparable results on well-behaved problems. If your objective is severely ill-conditioned — narrow, curved valleys where the right search direction differs sharply from axis-aligned steps — [CMA-ES](cmaes.md) will typically outperform DE, because it explicitly adapts its search distribution to that shape, something DE's difference-vector mutation does not do directly.

## Parameters

| Parameter | Default | Meaning | Tuning guidance |
|---|---|---|---|
| `n_individuals` | `40` | Population size — number of candidate vectors maintained throughout the run. | Increase for higher-dimensional problems; a common rule of thumb is 5-10× the number of dimensions. |
| `f` | `0.8` | Differential weight — the scale applied to the difference vector in mutation. | Typical effective range is `0.4`-`1.0`; higher values increase step size and exploration, lower values refine more locally. |
| `cr` | `0.9` | Crossover probability — the chance each dimension is taken from the mutant rather than the target. | Higher values (near `0.9`-`1.0`) work well for separable problems; lower values (`0.1`-`0.3`) can help on problems where dimensions interact strongly. |
| `n_iterations` | `100` | Number of passes over the population. | Increase for harder or higher-dimensional problems; DE often converges in fewer iterations than heavier heuristics like BFO. |

## Example

```python
from colonyx import AutoColony

def sphere(x):
    return sum(xi * xi for xi in x)

optimizer = AutoColony(mode="de", n_iterations=100, f=0.8, cr=0.9, random_state=7)
optimizer.fit(sphere, bounds=[(-5, 5), (-5, 5)])

optimizer.predict()  # best position found, ~ [0, 0]
optimizer.score()    # objective value at that position, ~ 0
```

## Further reading

- Storn, R. and Price, K. (1997). *Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces*. Journal of Global Optimization.
- [Algorithms overview](../algorithms.md) — compare Differential Evolution against every other algorithm colonyx ships.
- [AutoColony API reference](../autocolony-api.md) — the unified `mode=` interface used above.
