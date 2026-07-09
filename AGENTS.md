# Repository Guidelines

## Project Structure & Module Organization
`colonyx` is a Python package backed by Rust. The public Python API lives in `colonyx/` (`__init__.py`, `auto.py`), while the Rust extension and optimization logic live under `src/` (`lib.rs`, `bindings.rs`, `src/core/`, `src/algorithms/`). Tests are in `tests/`, and project notes live in `docs/`.

## Build, Test, and Development Commands
- `maturin develop` builds the Rust extension in editable mode for local development.
- `maturin build` produces a distributable wheel and checks the Rust/Python bridge.
- `pytest` runs the Python test suite in `tests/`.
- `uv run pytest` is preferred when using the checked-in `uv.lock` and managed dev environment.

## Coding Style & Naming Conventions
Use standard Python style: 4-space indentation, `snake_case` for functions and modules, and `PascalCase` for classes such as `AutoColony`. Keep public arguments and defaults explicit. In Rust, follow `rustfmt` defaults and keep module names lowercase with clear file names such as `pso.rs` or `solution.rs`.

## Testing Guidelines
The existing suite uses `pytest` with files named `tests/test_*.py`. Add tests alongside the relevant algorithm or API area, and prefer focused assertions for parameter validation, return types, and deterministic behavior via `random_state`. When adding Rust-side behavior, mirror it with Python-facing tests whenever possible.

## Commit & Pull Request Guidelines
Recent commits are short and imperative, often in the form of brief updates or merge commits. Keep commit messages concise and action-oriented, for example `Add ABC bounds validation`. Pull requests should include a summary of the change, the affected algorithm or API surface, and test results (`pytest`, `maturin develop`, or `maturin build`).

## Configuration Notes
This repository targets Python 3.8+ and depends on `numpy` and `scikit-learn`. Keep changes compatible with the compiled `_colonyx` module exposed from `colonyx/__init__.py`.
