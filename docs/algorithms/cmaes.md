# CMA-ES

`CmaEsOptimizer` is a separable (diagonal-covariance) CMA-ES: each axis adapts
its own variance rather than a full covariance matrix, so it cannot model
axis rotation. The step size (`sigma`) follows real cumulative step-size
adaptation (CSA) via an evolution path, the same mechanism full CMA-ES uses.

## Definition

Covariance Matrix Adaptation Evolution Strategy (CMA-ES) is a continuous
optimizer that maintains a search distribution — a mean position plus a
per-axis variance and a global step size `sigma` — and adapts that
distribution generation by generation. Each generation, a whole population
is sampled from the current distribution and evaluated as one independent
batch (no candidate's fitness depends on another's within the generation);
the better half is used to recompute the mean, and an evolution path tracks
how that mean has been moving to adapt `sigma` up or down. This
implementation is *separable* CMA-ES: it adapts a diagonal covariance
(independent per-axis variances) rather than a full covariance matrix, so
it cannot model correlations/rotation between dimensions — but `sigma`
itself follows the real cumulative step-size adaptation (CSA) mechanism
full CMA-ES uses, not a fixed decay schedule.

## Pseudocode

```text
mean <- midpoint of the bounds
covariance[d] <- 1.0 for each dimension d   # diagonal only
sigma <- cmaes_sigma
mu <- n_individuals / 2                     # number of individuals used to recombine
weights <- log-decreasing weights over the best `mu` ranks, summing to 1
p_sigma <- 0                                # evolution path

for generation in 1..n_iterations:
    for each of n_individuals candidates:
        candidate[d] <- clamp(mean[d] + sigma * sqrt(covariance[d]) * standard_normal())
    scores <- evaluate(candidates)          # independent batch, safe to parallelize
    update best from `scores`

    rank candidates by score
    new_mean <- weighted average of the best `mu` candidates (by `weights`)

    step <- (new_mean - mean) / (sigma * sqrt(covariance))   # per-axis, using the OLD covariance
    p_sigma <- (1 - c_sigma) * p_sigma + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * step
    sigma <- sigma * exp((c_sigma / d_sigma) * (|p_sigma| / chi_n - 1))

    new_covariance[d] <- weighted variance of the best `mu` candidates around `new_mean`
    covariance[d] <- 0.8 * covariance[d] + 0.2 * new_covariance[d]   # blended, not replaced outright
    mean <- new_mean
```

`c_sigma`, `d_sigma`, `mu_eff`, and `chi_n` are the standard CSA constants
from Hansen's CMA-ES tutorial, derived once from `mu` and the problem's
dimensionality before the generation loop starts.

## Mathematical Formulation

Sampling and log-decreasing recombination weights (\(n\) = dimensions):

$$
x_k \sim \mathcal{N}\big(m,\ \sigma^2 \operatorname{diag}(C)\big), \qquad k = 1, \dots, \lambda
$$

$$
w_i = \frac{\ln(\mu+0.5) - \ln(i)}{\sum_{j=1}^{\mu}\big(\ln(\mu+0.5)-\ln(j)\big)}, \qquad i = 1, \dots, \mu
$$

$$
m \leftarrow \sum_{i=1}^{\mu} w_i\, x_{i:\lambda}
$$

CSA constants, derived once from \(\mu\) and \(n\):

$$
\mu_{\text{eff}} = \frac{1}{\sum_i w_i^2}, \qquad
c_\sigma = \frac{\mu_{\text{eff}}+2}{n+\mu_{\text{eff}}+5}, \qquad
d_\sigma = 1 + c_\sigma + 2\max\!\left(0,\ \sqrt{\tfrac{\mu_{\text{eff}}-1}{n+1}}-1\right)
$$

$$
\chi_n \approx \sqrt{n}\left(1 - \frac{1}{4n} + \frac{1}{21n^2}\right)
$$

Evolution path and step-size update, using the *old* \(\sigma\) and \(C\):

$$
p_\sigma \leftarrow (1-c_\sigma)\,p_\sigma + \sqrt{c_\sigma(2-c_\sigma)\mu_{\text{eff}}}\ \frac{m - m_{\text{old}}}{\sigma\sqrt{\operatorname{diag}(C)}}
$$

$$
\sigma \leftarrow \sigma \exp\!\left(\frac{c_\sigma}{d_\sigma}\left(\frac{\lVert p_\sigma\rVert}{\chi_n}-1\right)\right)
$$

Diagonal covariance, blended with the previous value rather than replaced
outright (an EMA-style update, not the canonical rank-\(\mu\) learning-rate
formula):

$$
C_{jj} \leftarrow 0.8\,C_{jj} + 0.2\cdot\frac{\sum_{i=1}^{\mu} w_i\,(x_{i:\lambda,j}-m_j)^2}{\text{range}_j^2}
$$

## Use when

- You want a strong continuous optimizer with covariance adaptation.
- You can provide bounds and a reasonable iteration budget.

## API

- Rust class: `colonyx._colonyx.CmaEsOptimizer`
- Python mode: `AutoColony(mode="cmaes")`

## Parameters

- `n_individuals`
- `n_iterations`
- `cmaes_sigma` via `AutoColony(mode="cmaes")`; the same value is `sigma` when
  constructing `colonyx._colonyx.CmaEsOptimizer` directly (it's renamed on
  `AutoColony` so it doesn't collide with other modes' own `sigma`-shaped
  parameters)

## Example

```python
from colonyx import AutoColony

optimizer = AutoColony(mode="cmaes", n_iterations=100, cmaes_sigma=0.5, random_state=7)
optimizer.fit(lambda x: sum(v * v for v in x), bounds=[(-5, 5), (-5, 5)])
```

Using the Rust class directly, the same parameter is `sigma`:

```python
from colonyx._colonyx import CmaEsOptimizer

optimizer = CmaEsOptimizer(n_individuals=40, n_iterations=100, sigma=0.5, random_state=7)
```

!!! tip "Parallel fitness evaluation"
    Each generation's `n_individuals` candidates are evaluated in parallel
    (via Rayon) before ranking, since canonical CMA-ES never lets one
    candidate's fitness influence another's within the same generation. This
    is a real speedup when the objective is CPU-heavy — a plain Python
    objective still serializes on the GIL, but one that spends its time
    inside NumPy/SciPy/C-extension code (which releases the GIL) benefits.
