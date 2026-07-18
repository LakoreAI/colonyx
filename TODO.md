# ColonyX — Actionable TODO

Actionable findings from a deep codebase analysis (July 2026). Items grouped into
incremental steps so each step leaves the project in a strictly better state.

---

## Step 1 — Fix Critical Bugs

- [ ] **`cmaes_sigma` missing from `get_params()`/`set_params()`** (`auto.py:1117–1165`).
  `GridSearchCV` for `mode="cmaes"` silently fails because `get_params()` omits
  `cmaes_sigma` and `set_params()` rejects it with `ValueError`.
- [ ] **CuckooSearch `alpha` wired to `levy_scale` instead of its own parameter**
  (`auto.py:718`).  The backend constructor receives `alpha=levy_scale`, so the
  user cannot control the CS `alpha` independently.
- [ ] **BFO always accepts every candidate** (`continuous.rs:~1117`).  The condition
  `candidate_score <= scores[i] + 1e-12` is always true when the first clause
  fails, eliminating selection pressure.
- [ ] **`DiscreteProblem::evaluate()` panics on out-of-range tour indices**
  (`problem.rs:61–62`).  A `f64` → `usize` cast with no bounds check can OOB
  the `distance_matrix`.
- [ ] **MOPSO personal-best update overly permissive** (`advanced.rs:751–758`).
  Non-dominated solutions always replace `personal_best`, causing fluctuation.
- [ ] **Python `AutoColony` `score()` in compatibility mode returns `-best_score`**
  (`auto.py:940`).  Hard-codes sign convention; breaks if `best_score` is negative.
- [ ] **`check_graph_adjacency` raises semantically wrong `BoundsError` for sparse
  input** (`utils.py:113`).

## Step 2 — Close Critical Test Gaps

- [ ] **`cmaes` mode**: zero tests at any layer.  Add `tests/test_cmaes.py` with
  direct binding + `AutoColony` facade tests.
- [ ] **`bfo` mode**: no functional test (only appears in param-grid assertions).
- [ ] **`BinaryParticleSwarm`**: registered but untested.
- [ ] **Rust `continuous.rs`**: add `#[cfg(test)]` for GWO, FA, SA, CS, BA, GSO,
  BFO, DE, CMA-ES (9 algorithms, zero Rust unit tests).
- [ ] **Rust `advanced.rs`**: add `#[cfg(test)]` for PermutationGA, NSGA-II, MOPSO.
- [ ] **CLI**: zero test coverage for all 4 subcommands (`optimize`, `benchmark`,
  `report`, `list`).
- [ ] **Error-case tests**: predict-before-fit, bounds violations, reproducibility,
  bad-objective rejection — only covered for PSO/ABC/ACO; missing for GWO, FA,
  SA, CS, BA, GSO, BFO, DE, CMA-ES.

## Step 3 — Fix Rust Algorithm Correctness

- [ ] **CMA-ES is not real CMA-ES** (`continuous.rs`).
  - Diagonal-only covariance (cannot model axis rotation).
  - Hardcoded `sigma *= 0.99` decay instead of cumulative step-size adaptation.
  Either implement proper CSA + full covariance, or document as "separable CMA-ES
  with fixed schedule" and rename to avoid misleading users.
- [ ] **Cuckoo Search Levy flight uses same step for all dimensions**
  (`continuous.rs:604–605`).  Draw independent `u`/`v` per dimension.
- [ ] **Bat Algorithm pulse-rate update formula is non-standard**
  (`continuous.rs:801`).  The compounding `pulse[i] *= 1 - exp(-gamma * (t+1))`
  does not match the canonical BA.
- [ ] **MOPSO personal-best replacement logic** (`advanced.rs:751–758`): tighten
  to only replace when `current` dominates `personal_best`.
- [ ] **ABC single food-source degeneracy** (`abc.rs:63`): when `sn == 1` the
  mutation produces the same point; no progress made.
- [ ] **ABC `selection_fitness` for negative objectives** (`abc.rs:87–93`):
  `1.0 + abs(objective)` can give enormous weight to highly negative values.
  Use a bounded transformation or shift by the minimum.
- [ ] **ACO pheromone update for ACS variant** (`aco.rs:213`): standard ACS uses
  local + global-best-only updates; the current code applies full global update
  (all ants + elitist) to all variants.

## Step 4 — Refactor Rust Code Quality

- [ ] **Extract `make_rng(seed: Option<u64>) -> StdRng` helper** — the same
  `StdRng::seed_from_u64` / `StdRng::from_entropy` pattern is duplicated 12+ times.
- [ ] **Consolidate validation boilerplate in `continuous.rs`** — ~25 identical
  lines at the start of every `fit()` (dimension check, range computation, init).
  Move into a helper `validate_and_init()`.
- [ ] **Prevent `partial_cmp().unwrap()` panic on NaN fitness** — replace
  `.unwrap()` with `.unwrap_or(std::cmp::Ordering::Equal)` or use `f64::total_cmp`.
- [ ] **Make `history`/`population` fields `pub(crate)` or private with accessors**
  — currently public in all `continuous.rs` structs, breaking encapsulation.
- [ ] **Add `#[derive(Debug)]` to all algorithm structs** — missing on BeeColony,
  GreyWolfOptimizer, FireflyOptimizer, SimulatedAnnealing, CuckooSearch,
  BatAlgorithm, GlowwormOptimizer, BacterialForagingOptimizer,
  DifferentialEvolution, CmaEsOptimizer.
- [ ] **Add `Clone` to `OptimizationError`** (`base.rs`).
- [ ] **Handle multi-`fit()` calls** — ensure `history`/`population` vectors are
  cleared on each call.

## Step 5 — Fix Python `auto.py` Architecture

- [ ] **Replace 8× parameter enumeration with a registry dict** (`auto.py`).
  `__init__`, `get_params`, `set_params`, `_filter_params`, `_create_algorithm`,
  `parameter_mapping`, `suggest_parameters`, and `default_param_grids` each
  repeat the same ~50 params.  A single `_ALGORITHM_PARAMS` dict per algorithm
  would be the source of truth.
- [ ] **`_detect_problem_type` returns `"sklearn"` — an internal mode leaked into
  the public API** (`auto.py:203`).  `recommend_algorithm()` can return
  `mode: "sklearn"`, which is not in `_valid_modes` and breaks `_create_algorithm`.
- [ ] **`parameter_mapping()` is dead code** (`auto.py:258–355`) — never called
  internally.  Remove or wire it in.
- [ ] **`resolve_parameter_conflicts` stores ignored params but never exposes them**
  (`auto.py:622–632`).  Either surface them or remove.
- [ ] **`diversity_score()` has unreachable dead branch** (`auto.py:964–966`).
- [ ] **`optimization_metrics` returns `"best_score"` that can be negative MSE**
  (`auto.py:987–994`) — sign semantics differ between optimization and
  compatibility mode.

## Step 6 — Fix Sklearn Version Compatibility

- [ ] **Bump minimum `scikit-learn` to 1.4** (or add version-gated code paths).
  `__sklearn_tags__` requires ≥1.6; `validate_data` returning `(X, y)` requires
  ≥1.4.  Current `scikit-learn>=1.0` is wrong.
- [ ] **Eager native import** (`__init__.py:19`): `from . import _colonyx` loads
  the Rust module at import time.  A missing `.so` blocks all package imports
  (even Python-only benchmarks).  Switch to lazy import (pattern already used in
  `_create_algorithm`).

## Step 7 — Fix Multi-Objective Trait Design

- [ ] **`BinaryPSO`, `NSGA-II`, `MOPSO` do not implement `Optimizer`** — they
  have `fit_with_objective()` instead of `fit(&mut self, &dyn Problem)`, so they
  cannot be used through the uniform trait.  Either:
  - Extend the `Optimizer` trait to support multi-objective (breaking change), or
  - Keep the separate method but add a uniform dispatch layer in bindings.
- [ ] **Add `predict()`/`score()` to MOPSO** — currently has neither.

## Step 8 — Fix CLI & Metrics

- [ ] **`paired_significance_test` returns bogus `pvalue=1.0` when scipy is absent**
  (`metrics.py:338`).  Raise `ImportError` with a clear message instead.
- [ ] **`profile_callable` reads non-existent attributes, always returns NaN**
  (`metrics.py:62–65`).  Document that `__profile_*` attrs must be pre-set, or
  set them inside `profile_callable` itself.
- [ ] **CLI: no error handling for per-mode failures in batch benchmarks**
  (`cli.py:92–169`).  A single failing mode crashes the entire benchmark.
- [ ] **CLI: massive duplication between `_run_benchmark` and `_run_report`**
  (`cli.py:92–169`).  Extract shared logic.
- [ ] **CLI: hardcoded mode lists duplicated** (`cli.py:42,95,117`).  Share a
  constant.

## Step 9 — Additional Refinements (Low Priority)

- [ ] **Dead code: `BoundConstraint` enum** (`bounds.rs:82–92`) — never referenced.
- [ ] **`n_iterations` field ignored by BFO** (`continuous.rs`).  The outer loop
  uses `n_reproduction_steps`; `n_iterations` is stored but never read.
- [ ] **ACO `get_params` omits `use_two_opt`, `variant`, `q0`, `elitist_weight`,
  `tau_min`, `tau_max`** (`aco.rs`).
- [ ] **`cmaes` omitted from `default_param_grids()`** (`auto.py:1033–1047`).
- [ ] **`SolutionSet::find_best()` returns first element when all fitnesses are
  `None`** (`solution.rs:59–77`).  Should return `None`.
- [ ] **`SolutionSet::get_best()` can return stale index** — no bounds check
  after `next_generation()` resets it.
- [ ] **Rust `Bounds::clamp()` silently skips extra dimensions** (`bounds.rs:56`).
- [ ] **Python `check_optimization_problem` misclassifies any square 2D array as
  "discrete"** (`utils.py:135–141`).  Add an explicit heuristic or require the
  user to specify problem type.
- [ ] **`benchmark_suite()` returns fixed-arity (1D) problem definitions only**
  (`benchmarks.py:75–118`).  `_expand_bounds` is private in `cli.py`.
- [ ] **CLI CSV output missing trailing newline** (`cli.py:158`).
- [ ] **`robustness_score()` division-by-near-zero edge case** (`auto.py:978–985`).
- [ ] **Schwefel benchmark `optimum` precision truncated** (`benchmarks.py:116`).
- [ ] **`paired_significance_test` only supports parametric t-test** — add
  Wilcoxon signed-rank for non-normal benchmark scores.
