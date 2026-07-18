# Changelog

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
