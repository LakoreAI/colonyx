# Changelog

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

## Unreleased

### Added

- Rust-backed implementations for ACO, PSO, ABC, GWO, FA, SA, CS, BA, GSO, BFO, and DE.
- Thin Python loader shim for the compiled `colonyx._colonyx` extension.
- Library docs scaffold with `mkdocs`.
- Release checklist for PyPI and crates.io.
- Package metadata for PyPI/crates.io release readiness.

### Changed

- Python now acts as the public interface; optimization work happens in Rust.
- `AutoColony` continues to expose the same unified optimization entry point.
- `colonyx.__version__` now resolves from installed package metadata when available.

### Notes

- Local development still builds the extension from source.
- Publish steps still require tokens or trusted publishing setup.
