# Glowworm Swarm Optimization

`GlowwormOptimizer` uses luciferin values and neighborhood movement.

## Definition

Glowworm Swarm Optimization (GSO) is a population-based continuous
optimizer that models each candidate as a "glowworm" carrying a luciferin
value — brighter glow means a better-scoring position. Each iteration, every
glowworm looks at other glowworms within its `neighborhood_radius` that
glow brighter than it does, and takes a step toward the brightest one found.
Glowworms with no brighter neighbor in range stay put for that iteration
(other than the luciferin update), which is what lets the swarm split
across multiple local optima instead of collapsing onto one.

## Pseudocode

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

## Mathematical Formulation

$$
\ell_i \leftarrow (1-\text{luciferin_decay})\,\ell_i + \frac{\text{luciferin_enhancement}}{s_i + 1}
$$

where \(s_i\) is worm \(i\)'s objective value — a lower (better) \(s_i\)
yields a larger luciferin increment. The neighbor set and target are:

$$
N_i = \{\, j \neq i : \ell_j > \ell_i,\ \lVert x_i - x_j \rVert \le \text{neighborhood_radius} \,\}, \qquad
j^* = \operatorname*{arg\,min}_{j \in N_i} s_j
$$

\(j^*\) is the single best-scoring neighbor in range — not a
luciferin-proportional probabilistic choice as in some GSO variants. The
move is accepted greedily:

$$
x_i \leftarrow \operatorname{clamp}\!\big(x_i + \text{gso_step_size}\,(x_{j^*} - x_i)\big) \ \text{ if it improves } s_i
$$

## Use when

- You want multi-agent search with local neighborhood discovery.
- You care about neighborhood radius and luciferin decay.

## API

- Rust class: `colonyx._colonyx.GlowwormOptimizer`
- Python mode: `AutoColony(mode="gso")`

## Parameters

- `n_worms`
- `n_iterations`
- `luciferin_decay`
- `luciferin_enhancement`
- `gso_step_size` (the Rust `GlowwormOptimizer` constructor takes this as `step_size`)
- `neighborhood_radius`

## Example

```python
from colonyx import AutoColony

optimizer = AutoColony(mode="gso", n_iterations=100, random_state=7)
optimizer.fit(lambda x: sum(v * v for v in x), bounds=[(-5, 5), (-5, 5)])
```
