# Project Guidelines

## Architecture

`colonyx` is a Python package (`colonyx/`) backed by a Rust extension module (`_colonyx` built via PyO3/maturin). `colonyx/_colonyx.py` is a loader shim that discovers the compiled `.so`/`.dylib` from `sys.path` and `target/`.

- **Python API**: `colonyx/__init__.py` re-exports from the compiled Rust module + pure modules (`auto.py`, `base.py`, `benchmarks.py`, `datasets.py`, `metrics.py`, `utils.py`, `cli.py`).
- **Rust source**: `src/lib.rs` defines the `_colonyx` pymodule and registers `#[pyclass]` wrappers from `src/bindings.rs`. `src/core/` has shared types (`Bounds`, `ContinuousProblem`, `DiscreteProblem`, `Solution`). `src/algorithms/` implements each algorithm + the `Optimizer` trait in `base.rs`.
- **Adding an algorithm**: implement in `src/algorithms/`, add `#[pyclass]` in `src/bindings.rs`, register in `src/lib.rs`, re-export in `colonyx/__init__.py`, add mode mapping in `colonyx/auto.py`.
- **`AutoColony`** (in `auto.py`) is the main sklearn-compatible interface (`fit/predict/score`, `__sklearn_tags__`). Extends `BaseOptimizer` + `TransformerMixin`. Modes: `auto`, `aco`, `pso`, `abc`, `gwo`, `fa`, `sa`, `cs`, `ba`, `gso`, `bfo`, `de`, `cmaes`.

## Build & Dev

| Command | Purpose |
|---|---|
| `maturin develop` | **Required first step** — builds Rust extension in editable mode |
| `maturin build --release` | Produces a distributable wheel |
| `cargo check` | Rust compile-check |
| `cargo clippy --all-targets --all-features -- -D warnings` | Rust lint (CI demands no warnings) |
| `cargo fmt --all` | Rust formatting (max_width=100) |
| `RUSTFLAGS='-C link-args=-undefined -C link-args=dynamic_lookup' cargo build --release` | macOS native build for release validation |

## Test

- `uv run pytest` (preferred, uv-managed env) or `pytest` after `maturin develop`.
- Tests import the Rust module directly via `colonyx._colonyx`. Test files in `tests/test_*.py`, one per algorithm.
- Every test must use `random_state` for reproducibility. Assert optimal value, bounds respect, and error on predict-before-fit.
- Run `bandit -r colonyx -q` for security scanning (CI requirement).

## Lint & Typecheck

- `ruff` (lint+format, line-length=100, select E/F/I/B, ignore E501)
- `mypy` (python_version=3.8, ignore_missing_imports)
- Pre-commit hooks: ruff (lint + format) + mypy

## Key Constraints

- Python 3.8+, numpy>=1.21, scikit-learn>=1.0
- Rust edition 2021, pyo3 0.25.0, rand 0.8
- Version in `Cargo.toml` is source of truth (currently 0.1.1)
- `.env` contains publish tokens — do not commit

## CI / Release

- CI: Rust checks → `uv sync` → import smoke test → `bandit` → `pytest`. Docs: `mkdocs build --strict`.
- Release: Push tag → maturin-action builds + uploads wheels to PyPI.
