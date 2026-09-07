# Differential Evolution

`DifferentialEvolution` mutates, crosses over, and selects greedily.

## Definition

Differential Evolution (DE) is a population-based continuous optimizer that
generates a trial vector for each individual by adding a scaled difference
between two other individuals to a third, then mixing that mutant into the
target via crossover. If the trial scores at least as well as its target,
it replaces the target immediately — DE here is a steady-state loop, so a
trial can be selected against a target that was itself already replaced
earlier in the same generation, not a synchronized generation-wide swap.

## Pseudocode

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

## Mathematical Formulation

Mutant vector (DE/rand/1), from three distinct individuals \(\neq i\):

$$
v = x_{r_1} + f \cdot (x_{r_2} - x_{r_3})
$$

Binomial crossover, with one dimension \(j_{\text{rand}}\) always forced
from the mutant so the trial can never equal the target:

$$
u_j =
\begin{cases}
v_j & \text{if } \operatorname{rand}_j < cr \text{ or } j = j_{\text{rand}} \\
x_{ij} & \text{otherwise}
\end{cases}
$$

Greedy, immediate (steady-state) replacement:

$$
x_i \leftarrow u \quad \text{if } f(u) \le f(x_i)
$$

## Use when

- You want a strong general-purpose continuous optimizer.
- You want a simple, well-known population heuristic.

## API

- Rust class: `colonyx._colonyx.DifferentialEvolution`
- Python mode: `AutoColony(mode="de")`

## Parameters

- `n_individuals`
- `n_iterations`
- `f`
- `cr`

## Example

```python
from colonyx import AutoColony

optimizer = AutoColony(mode="de", n_iterations=100, random_state=7)
optimizer.fit(lambda x: sum(v * v for v in x), bounds=[(-5, 5), (-5, 5)])
```

## Notes

- Because replacement is immediate (steady-state), this loop cannot be
  parallelized across individuals without changing its results — a later
  target's `a`/`b`/`c` draw can land on an individual mutated earlier in the
  same generation.
