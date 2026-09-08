---
title: Bacterial Foraging Optimization (BFO)
description: Bacterial Foraging Optimization in colonyx — a three-loop continuous optimizer modeling E. coli chemotaxis, reproduction, and elimination-dispersal.
---

# Bacterial Foraging Optimization (BFO)

!!! abstract "TL;DR"
    Bacterial Foraging Optimization models how *E. coli* forage for nutrients: bacteria take small random "swim" steps (chemotaxis), the fitter half of the population reproduces to replace the rest (reproduction), and bacteria occasionally get wiped out and re-randomized elsewhere (elimination-dispersal). It nests all three loops inside one another, which gives it a lot of structure to tune but makes it a capable, if more involved, continuous heuristic. Run it in colonyx with `AutoColony(mode="bfo")`.

## What is Bacterial Foraging Optimization?

*E. coli* bacteria forage by alternating between two movement modes: a "tumble," which reorients them in a random direction, and a "swim," which carries them forward in whatever direction they're currently facing. Over many such steps, bacteria drift toward regions richer in nutrients. Bacterial Foraging Optimization borrows this as chemotaxis — the innermost of its three nested search loops — where each bacterium takes small random steps and keeps any step that improves its own score.

<figure markdown>
![BFO's three nested loops: chemotaxis takes a small step and keeps it if it improves, repeated n_chemotactic_steps times before reproduction sorts the population and replicates the fitter half, repeated n_reproduction_steps times before elimination-dispersal randomly resets bacteria with probability p_ed, then the whole nest repeats](../assets/diagrams/bfo.svg)
<figcaption>The three loops run at three different speeds — chemotaxis many times per reproduction round, reproduction several times per elimination-dispersal round — not as three equal, sequential steps.</figcaption>
</figure>

Layered on top of chemotaxis is reproduction: periodically, the fitter half of the population is duplicated to replace the weaker half, concentrating the search around currently promising bacteria the way real bacterial populations grow faster where nutrients are abundant. The outermost loop, elimination-dispersal, occasionally kills off a bacterium and drops it at a fresh random position, which is what lets the algorithm escape a region it has over-committed to and keeps the whole population from converging prematurely on one local optimum. This three-level nesting — chemotaxis inside reproduction inside elimination-dispersal — is what sets BFO apart from colonyx's other single-loop swarm algorithms.

## How colonyx implements it

The Rust implementation in `src/algorithms/continuous.rs` runs the three loops in the order above, with `n_iterations` controlling the outermost elimination-dispersal loop:

```text
bacteria[1..n_bacteria] <- random positions within bounds
scores <- evaluate(bacteria)
best <- bacterium with the lowest score

for elimination_round in 1..n_iterations:               # outermost
    for reproduction_round in 1..n_reproduction_steps:   # middle
        for chemotactic_step in 1..n_chemotactic_steps:  # innermost
            for each bacterium i:
                candidate <- bacterium_i + bfo_step_scale * random_direction * range
                if score(candidate) < scores[i]:
                    bacterium_i <- candidate; scores[i] <- score(candidate)
                update best if scores[i] improves on it

        # reproduction: the fitter half of the population replaces the rest
        sort bacteria by score
        survivors <- best n_bacteria/2 bacteria
        bacteria <- survivors, repeated to refill the population

    # elimination-dispersal
    for each bacterium i:
        if random() < elimination_probability:
            bacterium_i <- random position within bounds; re-evaluate
            update best if it improves on it
```

Each chemotactic step perturbs every dimension independently with uniform noise — not a normalized tumble-direction vector, as some canonical BFO descriptions use — and is accepted greedily if it improves the bacterium's own score, where \(c\) is `bfo_step_scale`, the chemotactic step size:

$$
x_{ij} \leftarrow \operatorname{clamp}\!\big(x_{ij} + (2r-1)\cdot c \cdot \text{range}_j\big), \qquad r \sim U(0,1)
$$

At the end of each reproduction round, the population is truncated to its fitter half and replicated back to full size:

$$
n_{\text{survivors}} = \max\!\left(1, \left\lfloor \frac{n_{\text{bacteria}}}{2} \right\rfloor\right)
$$

At the end of each elimination-dispersal round, every bacterium is independently reset to a fresh random position with probability \(p_{ed}\) (`elimination_probability`):

$$
P(\text{eliminate } i) = p_{ed}
$$

!!! note "A real correctness detail worth knowing"
    The three-loop nesting (elimination-dispersal > reproduction > chemotaxis) matters: an earlier version of this implementation had `n_reproduction_steps` and `n_iterations` mixed up, and `n_reproduction_steps` was briefly a dead parameter. Both are now correctly wired into their own loop level, matching the canonical BFO structure — worth knowing if you're comparing colonyx's BFO against another library's and the iteration counts don't line up the way you expect.

## When to use it (and when not to)

BFO is worth reaching for when you want a more thoroughly-structured continuous heuristic than a single-loop algorithm like [PSO](pso.md) or [ABC](abc.md) and are willing to spend time tuning its reproduction and elimination-dispersal settings on top of the usual population size and step scale — the nested-loop structure gives it a built-in mechanism for both local refinement (chemotaxis) and periodic diversification (elimination-dispersal) without needing an external restart strategy. If you'd rather not tune five separate parameters, [Glowworm Swarm Optimization](gso.md) or plain PSO get you a comparable diversification/refinement trade-off with fewer knobs. BFO's total number of objective evaluations grows as `n_iterations * n_reproduction_steps * n_chemotactic_steps * n_bacteria`, which can add up fast — budget for that multiplicative cost before scaling any of the three loop-count parameters up.

## Parameters

| Parameter | Default | Meaning | Tuning guidance |
|---|---|---|---|
| `n_bacteria` | `30` | Population size — number of bacteria maintained throughout the run. | Increase for higher-dimensional problems; remember every extra bacterium multiplies the total evaluation cost. |
| `n_chemotactic_steps` | `10` | Number of local random-walk steps per reproduction round (innermost loop). | More steps let each bacterium refine its position further before reproduction culls the population. |
| `n_reproduction_steps` | `4` | Number of reproduction rounds per elimination-dispersal round (middle loop). | More rounds concentrate the search around currently-fit bacteria more aggressively before the next dispersal event. |
| `elimination_probability` | `0.25` | Probability that any given bacterium is reset to a random position at the end of an elimination-dispersal round. | Raise it to escape local optima more readily at the cost of losing progress; lower it to let good regions persist longer. |
| `bfo_step_scale` | `0.1` | Scale of each chemotactic random step relative to the bounds' range (passed to the Rust constructor as `step_scale`). | Larger values explore faster but overshoot narrow optima; shrink it for fine-grained local search. |
| `n_iterations` | `100` | Number of elimination-dispersal rounds (outermost loop). | The most expensive parameter to raise, since it multiplies against both inner loop counts — increase cautiously. |

## Example

```python
from colonyx import AutoColony

def sphere(x):
    return sum(xi * xi for xi in x)

optimizer = AutoColony(mode="bfo", n_iterations=50, n_chemotactic_steps=8, random_state=7)
optimizer.fit(sphere, bounds=[(-5, 5), (-5, 5)])

optimizer.predict()  # best position found, ~ [0, 0]
optimizer.score()    # objective value at that position, ~ 0
```

## Further reading

- Passino, K.M. (2002). *Biomimicry of Bacterial Foraging for Distributed Optimization and Control*. IEEE Control Systems Magazine.
- [Algorithms overview](../algorithms.md) — compare BFO against every other algorithm colonyx ships.
- [AutoColony API reference](../autocolony-api.md) — the unified `mode=` interface used above.
