# Grey Wolf Optimization

## Definition

Grey Wolf Optimizer (GWO) is a compact continuous optimizer inspired by grey
wolf pack hunting hierarchy. Each iteration, the three fittest wolves
(alpha, beta, delta) are treated as the pack's best estimates of the prey's
location; every other wolf moves to the average of three positions
computed by circling each leader, with the circling radius shrinking
linearly over the run (`a` decays from 2 to 0) so the pack transitions from
wide exploration to tight exploitation.

## Pseudocode

```text
wolves = random init inside bounds
scores = evaluate(wolves)                            # batched, in parallel

for iteration in 1..n_iterations:
    alpha, beta, delta = the 3 wolves with lowest score
    a = 2 * (1 - iteration / (n_iterations - 1))      # decays 2 -> 0

    for each wolf:
        for each dimension d, and for each leader in {alpha, beta, delta}:
            r1, r2 = random(0, 1), random(0, 1)
            A = 2*a*r1 - a
            C = 2*r2
            x_leader = leader[d] - A * abs(C * leader[d] - wolf[d])
        wolf[d] = clamp(mean(x_alpha, x_beta, x_delta))
    scores = evaluate(wolves)
    track best (wolf, score) seen so far
```

## Mathematical Formulation

The exploration coefficient decays linearly over the run, from 2 to 0:

$$
a_t = 2\left(1 - \frac{t}{T-1}\right), \qquad t = 0, \dots, T-1
$$

For each leader \(k \in \{\alpha, \beta, \delta\}\), with independently
drawn \(r_1, r_2 \sim U(0,1)\) per dimension per leader:

$$
A_k = 2 a_t r_1 - a_t, \qquad C_k = 2 r_2
$$

$$
D_k = \left| C_k X_k - X \right|, \qquad X_k' = X_k - A_k D_k
$$

The wolf's new position is the (clamped) average of the three leader-guided
estimates:

$$
X \leftarrow \operatorname{clamp}\!\left(\frac{X_\alpha' + X_\beta' + X_\delta'}{3}\right)
$$

## Use when

- You want a compact continuous optimizer.
- You want a simple update rule with leader guidance.

## API

- Rust class: `colonyx._colonyx.GreyWolfOptimizer`
- Python mode: `AutoColony(mode="gwo")`

## Parameters

- `n_wolves`
- `n_iterations`

## Example

```python
from colonyx import AutoColony

optimizer = AutoColony(mode="gwo", n_iterations=100, random_state=7)
optimizer.fit(lambda x: sum(v * v for v in x), bounds=[(-5, 5), (-5, 5)])
print(optimizer.predict(), optimizer.score())
```
