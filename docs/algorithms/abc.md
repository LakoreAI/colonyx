# Artificial Bee Colony

## Definition

Artificial Bee Colony (ABC) is a population-based continuous optimizer that
models three bee roles working over a fixed set of food sources (candidate
solutions). Employed bees each refine their own source; onlooker bees are
recruited toward the better sources (roulette-wheel, weighted by fitness);
and a scout bee abandons whichever source has gone the longest without
improving, replacing it with a fresh random one — this keeps the colony from
stagnating on a food source that's stopped paying off.

## Pseudocode

```text
n_sources = max(n_bees // 2, 1)
sources = random init inside bounds
objective = evaluate(sources)                       # batched, in parallel
trials = [0] * n_sources                             # stagnation counters

for iteration in 1..n_iterations:
    # employed bee phase: one candidate per source
    for i in 1..n_sources:
        candidate = perturb source[i] toward a random partner source[k]
        if evaluate(candidate) < objective[i]:
            source[i], objective[i], trials[i] = candidate, evaluate(candidate), 0
        else:
            trials[i] += 1

    # onlooker bee phase: n_sources candidates, sources chosen by roulette
    # wheel weighted by selection_fitness(objective) (higher fitness = better)
    for _ in 1..n_sources:
        i = roulette_select(sources, weights = selection_fitness(objective))
        (same perturb-and-greedily-replace step as above, on source i)

    # scout bee phase: abandon the single most-stagnant source past `limit`
    i = argmax(trials)
    if trials[i] > limit:
        source[i] = random position; objective[i] = evaluate(source[i]); trials[i] = 0

    track best (source, objective) seen so far
```

## Mathematical Formulation

A candidate neighbor of source \(i\) perturbs a single, randomly chosen
dimension \(j\) using a randomly chosen partner source \(k \neq i\):

$$
v_{i,j} = x_{i,j} + \phi\,(x_{i,j} - x_{k,j}), \qquad \phi \sim U(-1, 1)
$$

(all other dimensions of \(v_i\) are copied unchanged from \(x_i\)). With
only one source (\(n_{\text{sources}}=1\)), there's no partner to difference
against, so it falls back to a small random walk on one dimension instead.
The candidate is kept only if it strictly improves the source's objective
value (greedy selection).

Onlooker bees pick a source by roulette wheel, weighted by a fitness
transform of the raw (minimized) objective value \(f\):

$$
\text{fit}(f) = \begin{cases} \dfrac{1}{1+f} & f \ge 0 \\[4pt] 1 + |f| & f < 0 \end{cases}, \qquad
P_i = \frac{\text{fit}(f_i)}{\displaystyle\sum_k \text{fit}(f_k)}
$$

## Use when

- You want a population-based continuous optimizer.
- You are comfortable tuning a food-source limit.

## API

- Rust class: `colonyx._colonyx.BeeColony`
- Python mode: `AutoColony(mode="abc")`

## Parameters

- `n_bees`
- `n_iterations`
- `limit`

## Example

```python
from colonyx import AutoColony

optimizer = AutoColony(mode="abc", n_iterations=100, random_state=7)
optimizer.fit(lambda x: sum(v * v for v in x), bounds=[(-5, 5), (-5, 5)])
print(optimizer.predict(), optimizer.score())
```

## Notes

- Strong fit for low- to medium-dimensional continuous landscapes.
