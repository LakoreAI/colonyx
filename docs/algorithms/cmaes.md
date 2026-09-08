---
title: CMA-ES
description: CMA-ES in colonyx — a separable Covariance Matrix Adaptation Evolution Strategy with real cumulative step-size adaptation, for ill-conditioned continuous optimization.
---

# Covariance Matrix Adaptation Evolution Strategy (CMA-ES)

!!! abstract "TL;DR"
    CMA-ES maintains a search distribution — a mean position, a per-axis variance, and a global step size — and reshapes that distribution generation by generation toward the local optimum's contour. colonyx's implementation is *separable* CMA-ES: it adapts a diagonal covariance (independent per-axis variances) rather than a full covariance matrix, so it can't model rotated correlations between dimensions, but its step size follows the real cumulative step-size adaptation (CSA) mechanism, not a fixed decay schedule. It's the strongest choice here for ill-conditioned or badly-scaled continuous objectives. Run it in colonyx with `AutoColony(mode="cmaes")`.

## What is CMA-ES?

Most population-based optimizers move a fixed-shape cloud of candidates around the search space. CMA-ES instead treats the population as samples from a multivariate normal distribution and actively reshapes that distribution as it learns about the objective. Each generation, a full population is drawn from the current distribution — a mean plus a step size `sigma` and a covariance describing how spread out the distribution is along each axis — and evaluated as one independent batch, meaning no candidate's fitness within a generation depends on another's. The better-performing half of that batch is used to recompute the mean, pulling the distribution toward promising territory, while an "evolution path" tracks the recent history of how that mean has moved in order to adapt `sigma` up when progress is fast and consistent, or down when the search is oscillating. This adaptive reshaping is what lets CMA-ES handle objectives that are stretched, skewed, or poorly scaled along different axes far better than algorithms that take axis-aligned or fixed-radius steps. colonyx's implementation is specifically *separable* CMA-ES: it adapts a diagonal covariance matrix, meaning each axis gets its own independently-learned variance, rather than the full covariance matrix that models correlations and rotation between axes. That's a deliberate trade-off — it makes the algorithm cheaper and simpler while still capturing the most common real-world failure mode of badly-scaled objectives (one dimension needing much larger steps than another) even though it can't fully align with a rotated elliptical valley the way full CMA-ES can. The step-size mechanism, however, is not simplified: `sigma` follows real cumulative step-size adaptation via an evolution path, the same core mechanism full CMA-ES uses.

## How colonyx implements it

The Rust implementation in `src/algorithms/continuous.rs` derives the CSA constants once from the population's recombination weight `mu` and the problem's dimensionality, then runs one sample-rank-update cycle per generation:

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

`c_sigma`, `d_sigma`, `mu_eff`, and `chi_n` are the standard CSA constants from Hansen's CMA-ES tutorial, derived once from `mu` and the problem's dimensionality before the generation loop starts.

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

Diagonal covariance, blended with the previous value rather than replaced outright — an EMA-style update rather than the canonical rank-\(\mu\) learning-rate formula:

$$
C_{jj} \leftarrow 0.8\,C_{jj} + 0.2\cdot\frac{\sum_{i=1}^{\mu} w_i\,(x_{i:\lambda,j}-m_j)^2}{\text{range}_j^2}
$$

!!! tip "Parallel fitness evaluation"
    Each generation's `n_individuals` candidates are evaluated in parallel (via Rayon) before ranking, since canonical CMA-ES never lets one candidate's fitness influence another's within the same generation. This is a real speedup when the objective is CPU-heavy — a plain Python objective still serializes on the GIL, but one that spends its time inside NumPy/SciPy/C-extension code (which releases the GIL) benefits.

## When to use it (and when not to)

Reach for CMA-ES when your objective is ill-conditioned, badly scaled, or has a curved, elongated optimum — the kind of landscape where [PSO](pso.md) or [Differential Evolution](de.md) tend to zigzag inefficiently because they take steps that aren't shaped to match the local geometry. Its per-axis variance adaptation lets it stretch its search distribution along whichever dimensions actually need larger steps, converging faster and more reliably than axis-agnostic algorithms on this kind of problem. It's overkill for simple, well-scaled, roughly convex objectives, where DE or PSO will get you there with less machinery and fewer objective evaluations per generation to reach a comparable result. Because colonyx's variant is separable rather than full CMA-ES, it still can't fully exploit a rotated elliptical valley (correlations between dimensions) — if you know your problem has that specific structure and need the absolute best convergence rate, a full-covariance CMA-ES implementation outside colonyx would do better, at higher computational cost per generation.

## Parameters

| Parameter | Default | Meaning | Tuning guidance |
|---|---|---|---|
| `n_individuals` | `40` | Population size per generation (\(\lambda\)); recombination uses the best half (\(\mu = \lambda/2\)). | Larger populations improve robustness on hard, noisy, or highly multimodal objectives, at the cost of more evaluations per generation. |
| `cmaes_sigma` | `0.5` | Initial global step size (passed to the Rust `CmaEsOptimizer` constructor as `sigma`, renamed on `AutoColony` so it doesn't collide with other modes' own `sigma`-shaped parameters). | Set it relative to how large a fraction of your bounds' range you expect the initial useful step to be; it self-adapts from there via CSA. |
| `n_iterations` | `100` | Number of generations to run. | CMA-ES often needs fewer iterations than DE or PSO to converge on ill-conditioned problems, since its adaptive step size gets the search direction right faster. |

## Example

```python
from colonyx import AutoColony

def sphere(x):
    return sum(xi * xi for xi in x)

optimizer = AutoColony(mode="cmaes", n_iterations=100, cmaes_sigma=0.5, random_state=7)
optimizer.fit(sphere, bounds=[(-5, 5), (-5, 5)])

optimizer.predict()  # best position found, ~ [0, 0]
optimizer.score()    # objective value at that position, ~ 0
```

Using the Rust class directly, the same parameter is named `sigma` instead of `cmaes_sigma`:

```python
from colonyx._colonyx import CmaEsOptimizer

optimizer = CmaEsOptimizer(n_individuals=40, n_iterations=100, sigma=0.5, random_state=7)
```

## Further reading

- Hansen, N. and Ostermeier, A. (2001). *Completely Derandomized Self-Adaptation in Evolution Strategies*. Evolutionary Computation.
- [Algorithms overview](../algorithms.md) — compare CMA-ES against every other algorithm colonyx ships.
- [AutoColony API reference](../autocolony-api.md) — the unified `mode=` interface used above.
