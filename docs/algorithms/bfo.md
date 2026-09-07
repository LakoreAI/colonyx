# Bacterial Foraging Optimization

`BacterialForagingOptimizer` models chemotaxis, reproduction, and elimination.

## Definition

Bacterial Foraging Optimization (BFO) is a population-based continuous
optimizer modeled on how bacteria like *E. coli* forage: bacteria take small
random "swim" steps (chemotaxis), the fittest half of the population
reproduces to replace the rest (reproduction), and occasionally a bacterium
is wiped out and re-randomized elsewhere (elimination-dispersal). Unlike
most algorithms in colonyx, BFO nests three loops rather than one: chemotaxis
runs inside reproduction, which runs inside elimination-dispersal.

## Pseudocode

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

## Mathematical Formulation

Each chemotactic step perturbs every dimension independently with uniform
noise (not a normalized tumble-direction vector as in some canonical BFO
descriptions):

$$
x_{ij} \leftarrow \operatorname{clamp}\!\big(x_{ij} + (2r-1)\cdot \text{bfo_step_scale}\cdot \text{range}_j\big), \qquad r \sim U(0,1)
$$

accepted greedily if it improves \(x_i\)'s score. At the end of each
reproduction round, the population is truncated to its fitter half and
replicated back to full size:

$$
n_{\text{survivors}} = \max\!\left(1, \left\lfloor \frac{n_{\text{bacteria}}}{2} \right\rfloor\right)
$$

At the end of each elimination-dispersal round, every bacterium is
independently reset to a fresh random position with probability:

$$
P(\text{eliminate } i) = \text{elimination_probability}
$$

## Use when

- You want a more involved continuous heuristic.
- You can spend time tuning reproduction and elimination settings.

## API

- Rust class: `colonyx._colonyx.BacterialForagingOptimizer`
- Python mode: `AutoColony(mode="bfo")`

## Parameters

- `n_bacteria`
- `n_iterations`
- `n_chemotactic_steps`
- `n_reproduction_steps`
- `elimination_probability`
- `bfo_step_scale` (the Rust `BacterialForagingOptimizer` constructor takes this as `step_scale`)

## Example

```python
from colonyx import AutoColony

optimizer = AutoColony(mode="bfo", n_iterations=100, random_state=7)
optimizer.fit(lambda x: sum(v * v for v in x), bounds=[(-5, 5), (-5, 5)])
```

## Notes

- The three-loop nesting (elimination-dispersal > reproduction > chemotaxis)
  is a real correctness detail: an earlier version of this implementation
  had `n_reproduction_steps` and `n_iterations` mixed up, and
  `n_reproduction_steps` was briefly a dead parameter. Both are now wired
  into their own loop level, matching canonical BFO.
