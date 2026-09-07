# Bat Algorithm

`BatAlgorithm` uses frequency, loudness, and pulse-rate updates.

## Definition

The Bat Algorithm (BA) is a population-based continuous optimizer inspired
by the echolocation behavior of microbats. Each bat has a position,
velocity, emission frequency, loudness, and pulse rate. Bats fly toward the
current best solution with a velocity update driven by a randomly drawn
frequency; with some probability (controlled by the pulse rate) a bat
instead takes a small random walk around the best solution. A move is only
accepted — updating that bat's loudness and pulse rate — if it improves the
bat's own score and passes a loudness-weighted acceptance check, which lets
loudness decay and pulse rate rise as the swarm converges.

## Pseudocode

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

## Mathematical Formulation

$$
f_i = f_{\min} + (f_{\max}-f_{\min})\,r, \qquad r \sim U(0,1)
$$

$$
v_i \leftarrow v_i + (x_i - x_{\text{best}})\,f_i, \qquad
x_i \leftarrow \operatorname{clamp}(x_i + v_i)
$$

With probability \(1 - r_i\) (i.e. when \(\text{random}() > r_i\)), a local
random walk replaces the move instead:

$$
x_i \leftarrow \operatorname{clamp}\!\big(x_{\text{best}} + 0.001 \cdot \mathcal{N}(0,1)\big)
$$

A candidate is accepted only if it improves \(x_i\)'s own score *and* passes
a loudness-weighted coin flip (\(\text{random}() < A_i\)); acceptance is what
triggers both updates below:

$$
A_i \leftarrow \text{bat_alpha} \cdot A_i, \qquad
r_i(t) = r_i(0)\left(1 - e^{-\text{bat_gamma}\, t}\right)
$$

## Use when

- You want another population-based continuous heuristic.
- You are comfortable with more parameters than PSO.

## API

- Rust class: `colonyx._colonyx.BatAlgorithm`
- Python mode: `AutoColony(mode="ba")`

## Parameters

- `n_bats`
- `n_iterations`
- `fmin`, `fmax`
- `bat_alpha`, `bat_gamma` (the Rust `BatAlgorithm` constructor takes these as `alpha`/`gamma`)
- `loudness`, `pulse_rate`

## Example

```python
from colonyx import AutoColony

optimizer = AutoColony(mode="ba", n_iterations=100, random_state=7)
optimizer.fit(lambda x: sum(v * v for v in x), bounds=[(-5, 5), (-5, 5)])
```

## Notes

- The pulse-rate schedule `r_i(t) = r_i(0) * (1 - exp(-bat_gamma * t))` is
  recomputed from each bat's *initial* pulse rate every acceptance, rather
  than compounded onto its previous value — this matches Yang's original
  formulation.
