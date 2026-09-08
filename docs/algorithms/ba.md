---
title: Bat Algorithm (BA)
description: Bat Algorithm in colonyx — a continuous optimizer inspired by microbat echolocation, with frequency, loudness, and pulse-rate updates. How it works, parameters, and an example.
---

# Bat Algorithm (BA)

!!! abstract "TL;DR"
    The Bat Algorithm models microbat echolocation: each bat flies toward the current best solution using a randomly tuned frequency, occasionally switches to a small local random walk, and only accepts a move if it improves its own score and passes a loudness-weighted acceptance check. It is a solid general-purpose continuous optimizer with more knobs than PSO, useful when you want fine control over the exploration-to-exploitation balance over a run. Run it in colonyx with `AutoColony(mode="ba")`.

## What is the Bat Algorithm?

Microbats navigate and hunt using echolocation: they emit pulses of sound and interpret the returning echoes to build a picture of their surroundings and locate prey. The Bat Algorithm borrows this behavior as a metaphor for continuous optimization. Each bat represents a candidate solution with a position and velocity, plus three echolocation-inspired properties — an emission frequency, a loudness, and a pulse rate. As the swarm searches, bats fly toward the best-known solution with a velocity update scaled by a randomly drawn frequency between `fmin` and `fmax`. With some probability tied to the pulse rate, a bat instead takes a small random walk around the best solution, mimicking a bat homing in tightly once it has detected nearby prey. A move is only accepted, and that bat's loudness and pulse rate updated, if it both improves the bat's own score and passes a loudness-weighted acceptance check — which is what lets loudness decay and pulse rate rise together as the swarm converges, the same way a real bat quiets down and pulses faster as it closes in on a target.

## How colonyx implements it

The Rust implementation in `src/algorithms/continuous.rs` runs one velocity/position update per bat per iteration, with the local-walk fallback and loudness-weighted acceptance check applied on top:

```text
bats[1..n_bats] <- random positions within bounds
velocities <- 0
loudness[i] <- loudness (initial, per bat)
pulse[i] <- pulse_rate (initial, per bat)
scores <- evaluate(bats)
best <- bat with the lowest score

for iteration in 1..n_iterations:
    for each bat i:
        frequency_i <- fmin + (fmax - fmin) * random()
        velocity_i <- velocity_i + (position_i - best_position) * frequency_i
        position_i <- clamp(position_i + velocity_i)

        if random() > pulse[i]:
            position_i <- clamp(best_position + small_random_walk)

        candidate_score <- score(position_i)
        if candidate_score <= scores[i] and random() < loudness[i]:
            scores[i] <- candidate_score
            loudness[i] <- loudness[i] * bat_alpha
            pulse[i] <- pulse_rate * (1 - exp(-bat_gamma * iteration))  # recomputed fresh, not compounded
            update best if candidate_score improves on it
```

$$
f_i = f_{\min} + (f_{\max}-f_{\min})\,r, \qquad r \sim U(0,1)
$$

$$
v_i \leftarrow v_i + (x_i - x_{\text{best}})\,f_i, \qquad
x_i \leftarrow \operatorname{clamp}(x_i + v_i)
$$

With probability \(1 - r_i\) (that is, when `random() > pulse_rate`), a local random walk replaces the velocity-based move instead:

$$
x_i \leftarrow \operatorname{clamp}\!\big(x_{\text{best}} + 0.001 \cdot \mathcal{N}(0,1)\big)
$$

A candidate is accepted only if it improves \(x_i\)'s own score *and* passes a loudness-weighted coin flip; acceptance is what triggers both updates below:

$$
A_i \leftarrow \text{bat\_alpha} \cdot A_i, \qquad
r_i(t) = r_i(0)\left(1 - e^{-\text{bat\_gamma}\, t}\right)
$$

!!! note "Pulse-rate schedule detail"
    The pulse-rate schedule above is recomputed from each bat's *initial* pulse rate every time a move is accepted, rather than compounded onto its previous value. This matches Yang's original formulation of the algorithm rather than a naive iterative decay.

## When to use it (and when not to)

The Bat Algorithm is a good fit when you want another population-based continuous heuristic beyond [PSO](pso.md) and are comfortable tuning a few more parameters in exchange for finer control over the exploration/exploitation trade-off — the loudness and pulse-rate schedules give you an explicit dial on how quickly the swarm shifts from broad search to local refinement, which plain PSO does not expose directly. If you would rather not tune that many parameters, start with [PSO](pso.md) or [Differential Evolution](de.md) first; both are simpler and competitive on well-behaved objectives. For objectives with many separated local optima, the abandonment mechanism in [Cuckoo Search](cs.md) tends to explore more aggressively than BA's local-walk fallback.

## Parameters

| Parameter | Default | Meaning | Tuning guidance |
|---|---|---|---|
| `n_bats` | `30` | Population size — number of bats maintained per generation. | Increase for higher-dimensional problems; each bat costs one objective evaluation per iteration. |
| `fmin` | `0.0` | Minimum emission frequency used to scale the velocity update. | Leave at `0` unless you have a reason to keep every bat moving at some minimum speed. |
| `fmax` | `2.0` | Maximum emission frequency. | Raise it to allow larger velocity jumps toward the best-known solution; lower it for gentler convergence. |
| `bat_alpha` | `0.9` | Loudness decay factor (passed to the Rust `BatAlgorithm` constructor as `alpha`). | Values closer to `1.0` keep loudness — and therefore acceptance of worse-scoring moves — high for longer, favoring exploration. |
| `bat_gamma` | `0.9` | Pulse-rate growth rate (passed to the Rust constructor as `gamma`). | Higher values ramp pulse rate up faster, shifting the swarm toward local random-walk refinement sooner. |
| `loudness` | `1.0` | Initial loudness for every bat. | Start high (near `1.0`) so early iterations accept more exploratory moves; it decays automatically via `bat_alpha`. |
| `pulse_rate` | `0.5` | Initial pulse rate for every bat. | Start moderate; it grows automatically via `bat_gamma` as the run progresses. |
| `n_iterations` | `100` | Number of generations to run. | Increase for harder problems; both loudness and pulse rate need enough iterations to visibly converge. |

## Example

```python
from colonyx import AutoColony

def sphere(x):
    return sum(xi * xi for xi in x)

optimizer = AutoColony(mode="ba", n_iterations=100, random_state=7)
optimizer.fit(sphere, bounds=[(-5, 5), (-5, 5)])

optimizer.predict()  # best position found, ~ [0, 0]
optimizer.score()    # objective value at that position, ~ 0
```

## Further reading

- Yang, X.-S. (2010). *A New Metaheuristic Bat-Inspired Algorithm*. Nature Inspired Cooperative Strategies for Optimization (NICSO).
- [Algorithms overview](../algorithms.md) — compare the Bat Algorithm against every other algorithm colonyx ships.
- [AutoColony API reference](../autocolony-api.md) — the unified `mode=` interface used above.
