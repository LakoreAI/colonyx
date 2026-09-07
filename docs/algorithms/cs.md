# Cuckoo Search

`CuckooSearch` combines Lévy-flight steps with nest replacement.

## Definition

Cuckoo Search (CS) is a population-based continuous optimizer inspired by
the brood-parasitism of cuckoo birds. Each "nest" holds a candidate
solution; every iteration, each nest is perturbed by a Lévy-flight step —
a heavy-tailed random step that mixes many small moves with occasional
large jumps — and the perturbed candidate replaces the nest if it improves
on it. A fraction of the worst nests are then abandoned and re-randomized
each iteration, which is what CS calls "host discovery" of the parasitic egg.

## Pseudocode

```text
nests[1..n_nests] <- random positions within bounds
scores <- evaluate(nests)
best <- nest with the lowest score

for iteration in 1..n_iterations:
    for each nest i:
        for each dimension d:
            u, v <- independent draws  # u, v ~ centered noise, one pair per dimension
            step_d <- levy_scale * u / |v|^(1/1.5)
            candidate_d <- clamp(nest[i]_d + cs_alpha * step_d * range_d)
        if score(candidate) < scores[i]:
            nest[i] <- candidate
            scores[i] <- score(candidate)

    abandon_count <- max(1, ceil(pa * n_nests))
    for each of the `abandon_count` worst-scoring nests:
        replace with a random position within bounds; re-evaluate

    update best from the current nests
```

## Mathematical Formulation

Each nest's step is drawn from two independent uniform variates per
dimension (this is a simplified heavy-tailed step, not literature Lévy
flight via Mantegna's algorithm and normal draws):

$$
u, v \sim U(-0.5, 0.5), \qquad
\text{step}_j = \text{levy_scale} \cdot \frac{u_j}{|v_j|^{1/1.5}}
$$

$$
x_{ij} \leftarrow \operatorname{clamp}\!\big(x_{ij} + \text{cs_alpha} \cdot \text{step}_j \cdot \text{range}_j\big)
$$

Abandonment replaces the worst-scoring nests, in count:

$$
n_{\text{abandon}} = \max\!\big(1,\ \lceil pa \cdot n_{\text{nests}} \rceil\big)
$$

## Use when

- You want exploration-heavy continuous search.
- You can tune abandonment probability.

## API

- Rust class: `colonyx._colonyx.CuckooSearch`
- Python mode: `AutoColony(mode="cs")`

## Parameters

- `n_nests`
- `n_iterations`
- `pa`
- `cs_alpha` (Lévy-flight step scale; the Rust `CuckooSearch` constructor takes this as `alpha`)
- `levy_scale`

## Example

```python
from colonyx import AutoColony

optimizer = AutoColony(mode="cs", n_iterations=100, random_state=7)
optimizer.fit(lambda x: sum(v * v for v in x), bounds=[(-5, 5), (-5, 5)])
```
