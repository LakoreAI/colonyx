# ColonyX — Actionable TODO

Actionable findings from a deep codebase analysis (July 2026). Items grouped into
incremental steps so each step leaves the project in a strictly better state.

---

## Step 1 — Fix Critical Bugs

> **Verified 2026-08-08**: everything below except the `score()` sign item was
> already fixed (see `CHANGELOG.md` 0.2.0) — this section had drifted out of
> date. Re-checked directly against current source before touching anything
> else in this file.

- [x] **`cmaes_sigma` missing from `get_params()`/`set_params()`** — present in
  both today (`auto.py`); `GridSearchCV` for `mode="cmaes"` works.
- [x] **CuckooSearch `alpha` wired to `levy_scale`** — `auto.py` correctly maps
  `cs_alpha` → `alpha` and `levy_scale` → `levy_scale` as independent params.
- [x] **BFO always accepts every candidate** — acceptance condition is
  `candidate_score < scores[i]`, real selection pressure.
- [x] **`DiscreteProblem::evaluate()` panics on out-of-range tour indices** —
  bounds-checked, returns `f64::INFINITY` instead of panicking.
- [x] **MOPSO personal-best update overly permissive** — tightened to only
  replace when `current` dominates `personal_best`.
- [x] **`check_graph_adjacency` raises semantically wrong error for sparse
  input** — raises `OptimizationError`, not `BoundsError`.
- [ ] **Python `AutoColony` `score()` in compatibility mode returns `-best_score`**
  (`auto.py`).  Still hard-codes sign convention: in sklearn-compatibility
  mode `best_score` is just the target value of the best row (not an error
  metric), so negating it to fake a "higher is better" score is semantically
  odd regardless of sign. Left as-is pending a design decision on what
  `score()` should mean in that fallback path — not a straightforward bugfix.

### Newly found and fixed (2026-08-08)

- [x] **BFO `n_reproduction_steps` was still a dead field.** The 0.2.0 fix
  wired `n_iterations` into the chemotaxis loop but never added a middle
  reproduction loop, so `n_reproduction_steps` had zero effect on behavior.
  `fit()` now nests elimination-dispersal (`n_iterations`) > reproduction
  (`n_reproduction_steps`) > chemotaxis (`n_chemotactic_steps`), matching
  canonical BFO.
- [x] **`scikit-learn>=1.0` / `requires-python>=3.8` were both wrong** —
  `base.py`'s `__sklearn_tags__`/`InputTags` usage requires scikit-learn ≥1.6,
  which itself requires Python ≥3.9. Reproduced the import failure on a clean
  Python 3.8 environment; bumped both floors (see `pyproject.toml`,
  `.python-version`, `mypy.ini`, `ruff.toml`, `tox.ini`).
- [x] **`@dataclass(..., slots=True)` requires Python ≥3.10** (`utils.py`,
  `base.py`, `benchmarks.py`) — colonyx never actually imported on its own
  declared minimum Python (3.8, then 3.9). Dropped `slots=True`; it was a
  pure micro-optimization with no behavioral dependency.

## Step 2 — Close Critical Test Gaps

> **Verified 2026-08-08**: fully resolved except CLI. `tests/test_cmaes.py`,
> `tests/test_bfo.py`, `tests/test_cs_ba_gso.py`, `tests/test_gwo.py`,
> `tests/test_fa.py`, `tests/test_sa.py`, `tests/test_de.py` all exist with
> full error-case coverage (predict-before-fit, bad-objective, mismatched
> bounds, reproducibility). `continuous.rs` and `advanced.rs` both have
> `#[cfg(test)]` modules covering every algorithm listed below, including
> `BinaryParticleSwarm` (`bpso_*` tests), PermutationGA (`pga_*`), NSGA-II
> (`nsga2_*`), and MOPSO (`mopso_*`). 134 Python + 74 Rust tests pass.

- [x] **`cmaes` mode**: `tests/test_cmaes.py`, direct binding + `AutoColony`.
- [x] **`bfo` mode**: `tests/test_bfo.py`, direct binding + `AutoColony`.
- [x] **`BinaryParticleSwarm`**: `advanced.rs::tests::bpso_*` (3 tests).
- [x] **Rust `continuous.rs`**: `#[cfg(test)]` present for all 9 algorithms.
- [x] **Rust `advanced.rs`**: `#[cfg(test)]` present for PermutationGA,
  NSGA-II, and MOPSO.
- [ ] **CLI**: still zero test coverage for all 4 subcommands (`optimize`,
  `benchmark`, `report`, `list`) — no `tests/test_cli.py` exists.
- [x] **Error-case tests**: predict-before-fit, bounds violations,
  reproducibility, bad-objective rejection now covered for every mode.

## Step 3 — Fix Rust Algorithm Correctness

> **Verified 2026-08-08**: Cuckoo Search, MOPSO, and ABC single-source items
> below were already fixed. CMA-ES, Bat Algorithm, and ACO/ACS were fixed in
> this pass (see `CHANGELOG.md` Unreleased). ABC `selection_fitness` turned
> out to already match the canonical Karaboga & Basturk (2007) ABC formula —
> not a bug, so it's noted rather than "fixed".

- [x] **CMA-ES is not real CMA-ES** (`continuous.rs`). Step size (`sigma`) now
  follows genuine cumulative step-size adaptation (evolution path, `c_sigma`/
  `d_sigma`/`chi_n`) instead of a fixed `sigma *= 0.99` decay. Covariance is
  still diagonal-only — documented as separable CMA-ES in code comments and
  `docs/algorithms/cmaes.md` rather than silently claiming full CMA-ES.
- [x] **Cuckoo Search Levy flight uses same step for all dimensions** — `u`/`v`
  are already drawn independently inside the per-dimension loop.
- [x] **Bat Algorithm pulse-rate update formula is non-standard** — now
  computes `r_i(t) = r_i(0) * (1 - exp(-gamma*t))` from each bat's initial
  rate every time (via the new `bat_pulse_rate()` helper) instead of
  compounding `*=` onto the previous value.
- [x] **MOPSO personal-best replacement logic** — already tightened to only
  replace when `current` dominates `personal_best`.
- [x] **ABC single food-source degeneracy** — already fixed with a
  random-walk fallback when `sn == 1`.
- [~] **ABC `selection_fitness` for negative objectives** (`abc.rs`): not a
  bug. `fit = 1/(1+f)` for `f>=0`, `fit = 1+|f|` for `f<0` is the standard
  ABC fitness transform from the original paper; the "unbounded weight"
  behavior for very negative objectives is a known, accepted property of the
  canonical formula itself, not a colonyx-specific defect.
- [x] **ACO pheromone update for ACS variant** (`aco.rs`): ACS now deposits
  global pheromone only via the best-so-far ant (skips the all-ants Ant
  System deposit that Basic/Elitist/MMAS still use). True ACS *local*
  pheromone updates during tour construction are still not modeled — would
  need a new `xi` parameter and pyo3/Python surface changes, left for a
  separate pass given the added API surface.

## Step 4 — Refactor Rust Code Quality

> **Verified 2026-08-08**: everything below except the `history`/`population`
> encapsulation item was already done (see `CHANGELOG.md` 0.2.0).

- [x] **Extract `make_rng(seed: Option<u64>) -> StdRng` helper** — present in
  `base.rs`.
- [x] **Consolidate validation boilerplate in `continuous.rs`** —
  `validate_and_init()` exists and is used by all 9 continuous `fit()`s.
- [x] **Prevent `partial_cmp().unwrap()` panic on NaN fitness** — no bare
  `partial_cmp(...).unwrap()` remains anywhere in `src/`; all use
  `unwrap_or(Ordering::Equal)`.
- [x] **Add `#[derive(Debug)]` to all algorithm structs** — present on all of
  them today.
- [x] **Add `Clone` to `OptimizationError`** — `#[derive(Debug, Clone)]` in
  `base.rs`.
- [x] **Handle multi-`fit()` calls** — every `continuous.rs` `fit()` starts
  with `self.history.clear(); self.population.clear();` (9 occurrences).
- [ ] **Make `history`/`population` fields `pub(crate)` or private with
  accessors** — still `pub` in every `continuous.rs` struct. Left as-is: it's
  a real encapsulation smell but not a bug, and touching it means updating
  every algorithm struct plus checking every `bindings.rs` access site for a
  purely cosmetic win.

## Step 5 — Fix Python `auto.py` Architecture

- [ ] **Replace 8× parameter enumeration with a registry dict** (`auto.py`).
  `__init__`, `get_params`, `set_params`, `_filter_params`, `_create_algorithm`,
  `parameter_mapping`, `suggest_parameters`, and `default_param_grids` each
  repeat the same ~50 params.  A single `_ALGORITHM_PARAMS` dict per algorithm
  would be the source of truth. Left undone: this is the single biggest
  remaining item in this file, and it touches the sklearn-compatibility-
  critical `get_params`/`set_params` contract for every one of the 12 modes —
  worth doing as its own focused, carefully-tested pass rather than folded
  into a broader cleanup.
- [x] **`_detect_problem_type` dead code removed** — it duplicated
  `recommend_algorithm()`'s classification but was never actually called from
  `fit()` (which calls `recommend_algorithm()` directly); deleted rather than
  fixed. `recommend_algorithm()` returning `mode: "sklearn"` is intentional,
  internal-only plumbing: `fit()` explicitly special-cases
  `algorithm_mode == "sklearn"` *before* it would ever reach
  `_create_algorithm`, so nothing was actually broken — the original "breaks
  `_create_algorithm`" claim didn't hold up under a direct read of `fit()`.
- [~] **`parameter_mapping()` is dead code** — not true: it's called directly
  in `tests/test_sklearn_compat.py` as public introspection API. "Never
  called internally" was accurate but irrelevant; kept as-is.
- [~] **`resolve_parameter_conflicts` never exposes ignored params** — not
  true: it sets `self.parameter_conflicts_`, which is asserted on directly in
  `tests/test_sklearn_compat.py::test_parameter_conflicts_are_recorded_for_explicit_modes`.
- [x] **`diversity_score()` has unreachable dead branch** — both arms of the
  nested `if` returned `0.0`; collapsed to one `return 0.0`.
- [x] **`score()` returns a raw, arbitrary sign-flip in sklearn-compatibility
  mode** (this is the Step 1 `score()` item, now fixed — see there). Also
  fixes this Step 5 item's underlying complaint about sign semantics
  differing between optimization and compatibility mode.

## Step 6 — Fix Sklearn Version Compatibility

- [x] **Bump minimum `scikit-learn`** — bumped to `>=1.6` (covers the
  `__sklearn_tags__` requirement) and `requires-python` to `>=3.9` (sklearn
  1.6 itself needs Python ≥3.9). Reproduced the previous `ImportError` on a
  clean Python 3.8 + scikit-learn 1.3.2 environment before fixing; see Step 1
  "newly found" for the second, previously-hidden `slots=True` incompatibility
  this uncovered along the way.
- [ ] **Eager native import** (`__init__.py:19`): `from . import _colonyx` loads
  the Rust module at import time.  A missing `.so` blocks all package imports
  (even Python-only benchmarks).  Switch to lazy import (pattern already used in
  `_create_algorithm`). Left undone: `__init__.py` re-exports ~20 Rust classes
  directly (not just through `AutoColony`), so doing this properly means
  converting the whole module to PEP 562 lazy `__getattr__` attributes, not a
  one-line change — and the practical benefit is narrow (only helps
  pure-Python-only usage of `benchmarks`/`metrics`/`utils` when the compiled
  extension is missing, an already-broken install either way).

## Step 7 — Fix Multi-Objective Trait Design

- [ ] **`BinaryPSO`, `NSGA-II`, `MOPSO` do not implement `Optimizer`** — they
  have `fit_with_objective()` instead of `fit(&mut self, &dyn Problem)`, so they
  cannot be used through the uniform trait.  Either:
  - Extend the `Optimizer` trait to support multi-objective (breaking change), or
  - Keep the separate method but add a uniform dispatch layer in bindings.
  Left undone: real architectural question, not a bug — nothing in the
  codebase currently needs to hold these polymorphically as `dyn Optimizer`,
  and the fix requires either a breaking trait change or a new dispatch
  layer, both bigger than this pass's scope.
- [x] **Add `predict()`/`score()` to MOPSO** — already present, just at the
  pyo3 binding layer (`PyMopsoOptimizer::predict()`/`score()` in
  `bindings.rs`) rather than on the inner `MopsoOptimizer` Rust struct
  itself. `score()` returns the 2D hypervolume of the Pareto archive.
  Verified working end-to-end from Python.

## Step 8 — Fix CLI & Metrics

- [x] **`paired_significance_test` returns bogus `pvalue=1.0` when scipy is
  absent** — now raises `ImportError` with a clear message instead of
  fabricating a p-value that ignores the actual computed statistic.
- [x] **`profile_callable` reads non-existent attributes, always returns NaN**
  — it tried to read `__profile_*` attributes that nothing ever set on a
  plain callable (only `profile_optimization_run` set them, on a different
  kind of object). Rewritten as an honestly-generic profiler: elapsed
  time/peak memory are real, the optimization-specific fields are explicit
  `nan`/`0` with a docstring pointing to `profile_optimization_run` for real
  optimizer profiling.
- [x] **CLI: no error handling for per-mode failures in batch benchmarks** —
  `benchmark_optimizers()` now catches a factory/run failure per mode, warns
  (`RuntimeWarning`) and reports via an optional `on_error` callback, and
  excludes just that mode from the result instead of crashing the batch. CLI
  wires `on_error` to print `"[mode] failed and was skipped: ..."` to stderr.
- [x] **CLI: massive duplication between `_run_benchmark` and `_run_report`**
  — extracted shared `_run_benchmark_suite()` (setup + `benchmark_optimizers`
  call) and `_print_visualization()` helpers.
- [x] **CLI: hardcoded mode lists duplicated** — shared `_CONTINUOUS_MODES`/
  `_ALL_MODES` module-level constants.

## Step 9 — Additional Refinements (Low Priority)

> **Verified 2026-08-08**: `BoundConstraint`, `SolutionSet::find_best()`/
> `get_best()`, `Bounds::clamp()`, and `robustness_score()` were all already
> fixed/removed. CLI CSV newline fixed this pass. The rest are either
> inherent ambiguities (not clean bugs) or negligible/feature-scope items,
> noted below rather than changed.

- [x] **Dead code: `BoundConstraint` enum** — no longer exists anywhere in
  `src/`.
- [x] **`n_iterations`/`n_reproduction_steps` loop nesting in BFO** — a prior
  fix pointed the outer loop at `n_iterations` (was reading
  `n_reproduction_steps`), which left `n_reproduction_steps` itself dead;
  both are now wired into their own nested loop (see Step 1, "newly found").
- [x] **ACO `get_params` omits `use_two_opt`, `variant`, `q0`, `elitist_weight`,
  `tau_min`, `tau_max`** — all present in `aco.rs::get_params()` today.
- [x] **`cmaes` omitted from `default_param_grids()`** — present in
  `auto.py::default_param_grids()` and `default_param_distributions()` today.
- [x] **`SolutionSet::find_best()` returns first element when all fitnesses
  are `None`** — already returns `None` via a `found` flag.
- [x] **`SolutionSet::get_best()` can return stale index** — already
  bounds-checks `idx < self.solutions.len()` before returning.
- [x] **Rust `Bounds::clamp()` silently skips extra dimensions** — already
  `assert!`s (panics with a clear message) instead of silently skipping.
- [~] **Python `check_optimization_problem` misclassifies any square 2D array
  as "discrete"** — a genuine, inherent ambiguity (a square tabular
  covariate matrix is indistinguishable from a distance matrix by shape
  alone), not a clean bug to silently "fix". Would need an explicit
  `problem_type` override parameter threaded through `fit()`/
  `recommend_algorithm()`/etc. — a real feature, left for a dedicated pass.
- [~] **`benchmark_suite()` returns fixed-arity (1D) problem definitions
  only; `_expand_bounds` is private in `cli.py`** — `_expand_bounds` already
  handles arbitrary `--dimensions` correctly for every current benchmark
  (replicates/truncates/extends), so this wasn't blocking real usage; making
  it public API is a minor completeness nice-to-have, not a bug.
- [x] **CLI CSV output missing trailing newline** — files written via
  `--output` now always end with exactly one `\n`; stdout printing is
  unchanged (`print()` already added its own).
- [x] **`robustness_score()` division-by-near-zero edge case** — already
  guarded with `if abs(mean_value) < 1e-12: ...` before dividing.
- [~] **Schwefel benchmark `optimum` precision truncated** — checked
  numerically: `schwefel([420.9687])` vs `schwefel([420.968746])` differ by
  ~3e-8, both ~1.27e-5 away from the claimed `minimum=0.0` regardless (the
  418.9829 constant in the formula is itself a rounded literature value).
  Not worth changing.
- [ ] **`paired_significance_test` only supports parametric t-test** — add
  Wilcoxon signed-rank for non-normal benchmark scores. A real feature
  addition, not a bug; left for a dedicated pass.
