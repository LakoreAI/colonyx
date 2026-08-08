# Changelog

## Unreleased

### Fixed

- BFO `n_reproduction_steps` was still a dead field after the 0.2.0 fix (which wired `n_iterations` into the loop but left the middle reproduction loop absent). `fit()` now nests elimination-dispersal (`n_iterations`) > reproduction (`n_reproduction_steps`) > chemotaxis (`n_chemotactic_steps`), matching canonical BFO.
- Bat Algorithm pulse rate no longer compounds (`pulse *= ...` each acceptance); it now follows the canonical Yang formula `r_i(t) = r_i(0) * (1 - exp(-gamma*t))`, recomputed from each bat's initial rate.
- Ant Colony System (`variant="acs"`) no longer applies the Ant-System-style all-ants pheromone deposit; ACS now deposits only via the best-so-far ant, as in the literature (true ACS local updates during construction are still not modeled).
- CMA-ES step size (`sigma`) now follows real cumulative step-size adaptation (an evolution path with the standard `c_sigma`/`d_sigma`/`chi_n` terms) instead of a fixed `sigma *= 0.99` decay. Covariance is still diagonal-only (separable CMA-ES); docs and code comments now say so explicitly.
- `scikit-learn>=1.0` / `requires-python>=3.8` were incompatible with the actual code: `base.py` uses the `__sklearn_tags__`/`InputTags` API added in scikit-learn 1.6, which itself requires Python ≥3.9. Reproduced the resulting `ImportError` on a clean Python 3.8 install and bumped both constraints (`requires-python>=3.9`, `scikit-learn>=1.6`) so `pip install colonyx` resolves to a working environment.
- `@dataclass(..., slots=True)` in `utils.py`, `base.py`, and `benchmarks.py` requires Python ≥3.10 — `import colonyx` raised `TypeError: dataclass() got an unexpected keyword argument 'slots'` on 3.8 and 3.9, i.e. colonyx never actually imported on its own declared minimum Python. Dropped `slots=True` (pure micro-optimization, no behavioral dependency) to match the real 3.9+ floor.
- `AutoColony.score()` in sklearn-compatibility mode returned `-best_score`, an arbitrary sign flip of a raw target value with no real meaning. It now defaults to `-MSE` against the targets seen during `fit()` when `y` is omitted, matching the semantics already used when `y` is passed explicitly.
- `metrics.profile_callable()` tried to read `__profile_*` attributes off an arbitrary callable that nothing ever set, silently returning fabricated `nan`/`0` placeholders. Rewritten as an honestly-generic profiler (real elapsed time/peak memory; optimization-specific fields explicitly `nan`/`0` with a docstring pointing to `profile_optimization_run` for real optimizer profiling).
- `metrics.paired_significance_test()` returned a fabricated `pvalue=1.0` when scipy was unavailable, regardless of the actual computed statistic. Now raises `ImportError` instead of misrepresenting an untested statistic as "not significant".
- CLI (`benchmark`/`report`): a single failing mode used to crash the entire batch. `metrics.benchmark_optimizers()` now catches per-mode failures, warns (`RuntimeWarning`), reports via an optional `on_error` callback, and simply excludes that mode from the results; the CLI wires this to print `"[mode] failed and was skipped: ..."` to stderr.
- CLI `report --format csv --output <file>`: the file used to be missing its trailing newline (`buffer.getvalue().strip()` ate it). Files now always end with exactly one `\n`; stdout printing is unchanged.

### Removed

- `AutoColony._detect_problem_type()` — dead code duplicating `recommend_algorithm()`'s classification; `fit()` never called it.
- `AutoColony.diversity_score()`'s unreachable nested branch (both arms returned `0.0`).

### Added

- Rust unit tests covering the four algorithm fixes above (`bfo_n_reproduction_steps_is_not_ignored`, `bat_pulse_rate_matches_canonical_formula_and_does_not_compound`, `acs_variant_skips_uniform_all_ant_deposit` / `basic_variant_still_deposits_from_every_ant`, `cmaes_sigma_uses_step_size_adaptation_not_fixed_decay`).
- `tests/test_cli.py` — first-ever test coverage for the CLI's `optimize`/`benchmark`/`report` subcommands (7 tests), including regression tests for the per-mode-failure and CSV-newline fixes above.
- Python regression tests for the `score()`, `profile_callable`, and `paired_significance_test` fixes above.
- CLI: shared `_CONTINUOUS_MODES`/`_ALL_MODES` constants and a shared `_run_benchmark_suite()`/`_print_visualization()` helper, replacing duplicated mode lists and near-identical logic between `_run_benchmark` and `_run_report`.

### Verified (already fixed or not actually bugs)

- A full re-audit of `TODO.md` Steps 4–9 found most remaining items were already resolved (make_rng/validate_and_init/Debug/Clone/multi-fit-clearing in Rust; SolutionSet/Bounds bounds-checks; ACO get_params completeness; robustness_score's zero-division guard; MOPSO already has predict()/score() at the pyo3 layer) or were misdiagnosed (parameter_mapping()/resolve_parameter_conflicts() are used public API, not dead code). See `TODO.md` for the full, now-accurate breakdown of what's genuinely still open (the `auto.py` parameter-registry refactor, the multi-objective `Optimizer` trait design, lazy native import, and a couple of low-value/feature-scope items).

## 0.1.1

### Added

- Rust usage documentation with direct examples for continuous and discrete optimizers.
- Rust library target exposure so the optimization core can be used from Rust code.

### Changed

- The loader shim now recognizes the Rust library artifact produced by the Rust crate target.

## 0.1.0

### Added

- Rust-backed core algorithms for discrete and continuous optimization.
- Added a CMA-ES-style continuous optimizer.
- Python loader shim for the compiled `colonyx._colonyx` extension.
- `AutoColony` as the main sklearn-style interface.
- Library docs scaffold and release checklist.

### Notes

- This is the first Rust-first release line for the project.
- The Python layer is intentionally thin; the optimization loop lives in Rust.

## 0.2.0

### Fixed

- `AutoColony.get_params()`/`set_params()` now include `cmaes_sigma` — `GridSearchCV` for CMA-ES no longer fails.
- Cuckoo Search `alpha` parameter correctly wired (was aliased to `levy_scale`).
- BFO selection pressure restored (acceptance condition was always true).
- `DiscreteProblem::evaluate()` no longer panics on out-of-range tour indices.
- MOPSO personal-best update tightened (only replaces when `current` dominates).
- `check_graph_adjacency` raises correct `TypeError` for sparse input.
- CS Levy flight now draws independent steps per dimension.
- ABC single food-source (`sn==1`) degeneracy fixed with random-walk fallback.
- BFO `n_iterations` wired into loop count (was reading `n_reproduction_steps`).
- `partial_cmp().unwrap()` replaced with `unwrap_or(Ordering::Equal)` across all algorithms (NaN-safety).
- `SolutionSet::find_best()` returns `None` when all fitnesses are `None`.
- `SolutionSet::get_best()` bounds-checks index before access.
- `Bounds::clamp()` asserts on dimension mismatch instead of silently skipping.
- Multi-`fit()` calls now clear `history`/`population` vectors.

### Added

- 114 new tests (62 Python + 52 Rust) — zero-coverage algorithms CMA-ES and BFO now fully tested.
- Rust `#[cfg(test)]` modules for `continuous.rs` (36 tests) and `advanced.rs` (16 tests), matching the PSO/ABC/ACO pattern.
- Error-case tests (predict-before-fit, bounds violations, reproducibility, bad-objective rejection) for GWO, FA, SA, CS, BA, GSO, DE, BFO, CMA-ES.
- `make_rng(seed)` helper extracted in `base.rs`, eliminating 16 identical `match` blocks.
- `validate_and_init()` helper consolidating dimension/seed/ranges boilerplate in 9 `fit()` methods.
- `#[derive(Debug)]` on all 16 algorithm structs.
- `Clone` on `OptimizationError`.
- `AcoVariant`, `q0`, `elitist_weight`, `tau_min`, `tau_max` now exposed in `AntColony::get_params()`.

### Changed

- Internal RNG construction centralized via `make_rng()`.
- Validation boilerplate in `continuous.rs` reduced from ~135 lines to ~45.
- Python tests expanded from 72 to 134; Rust tests from 17 to 69 (203 total, all passing).
- Removed dead `BoundConstraint` enum.
