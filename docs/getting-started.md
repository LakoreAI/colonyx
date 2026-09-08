---
title: "Getting started with colonyx"
description: "Install colonyx from PyPI or build it from source with maturin and Rust, then run your first continuous and discrete optimization with AutoColony."
---

# Getting Started

`colonyx` needs nothing more than `pip install colonyx` for normal use, because the PyPI package ships a prebuilt Rust extension — you only need a Rust toolchain if you're building from a source checkout. This page covers both paths, then walks through your first continuous and discrete optimization runs.

## Install from PyPI

```bash
pip install colonyx
```

That's all most users need — skip to [Continuous optimization](#continuous-optimization) below. This installs a wheel containing the compiled `colonyx._colonyx` extension module (built by [maturin](https://www.maturin.rs/) from the Rust crate under `src/`), so there is no Rust compilation step on your machine and no C/C++ toolchain requirement either. `colonyx` supports CPython and PyPy on Python 3.9 through 3.13, with `numpy>=1.21` and `scikit-learn>=1.6` as its only runtime dependencies.

## Install from source

Building from source requires a Rust toolchain and, ideally,
[`maturin`](https://www.maturin.rs/) to build the `colonyx._colonyx`
extension in editable mode:

```bash
python3 -m pip install -U pip maturin
maturin develop
```

`maturin develop` compiles the Rust crate defined in `Cargo.toml`/`src/` and installs the resulting extension module directly into your active Python environment as `colonyx._colonyx`, which `colonyx/__init__.py` then re-exports as the public classes (`AutoColony`, `ParticleSwarm`, `AntColony`, and so on). Use this path if you're contributing to the Rust core, need a debug build to reproduce a numerical issue, or want to track an unreleased change on `main` before it reaches PyPI.

!!! note "macOS extension-module linking"
    On macOS, `cargo build`/`cargo test` (not `maturin develop`) need an
    extra linker flag because pyo3's `extension-module` feature doesn't
    produce a fully-linked binary Cargo can run directly:

    ```bash
    RUSTFLAGS='-C link-args=-undefined -C link-args=dynamic_lookup' cargo build --release
    ```

    This only matters if you're running `cargo build`/`cargo test` directly against the Rust crate; `maturin develop` and `pip install colonyx` already handle this for you.

## Development import

From a checkout, Python can import the local build artifact through `colonyx._colonyx`.

```python
import colonyx
from colonyx import AutoColony
```

## Rust development

If you want to work on the Rust core directly, build the crate with `cargo`
or `maturin` and import the shared modules from `colonyx::core` and
`colonyx::algorithms`.

```bash
cargo build
```

See [Rust Usage](rust.md) for a full Rust usage example, including the shared `Bounds`/`Solution`/`Problem` types every algorithm module builds on.

## Continuous optimization

Pass a callable objective and per-dimension bounds:

```python
optimizer = AutoColony(mode="de", n_iterations=100, random_state=7)
optimizer.fit(lambda x: sum(v * v for v in x), bounds=[(-5, 5), (-5, 5)])
```

`bounds` is a sequence of `(low, high)` pairs, one per dimension of the search space — here, `[(-5, 5), (-5, 5)]` optimizes a two-dimensional sphere function inside a `[-5, 5] × [-5, 5]` box. `mode="de"` selects [Differential Evolution](algorithms/de.md); swap it for `"pso"`, `"abc"`, `"gwo"`, `"fa"`, `"sa"`, `"cs"`, `"ba"`, `"gso"`, `"bfo"`, or `"cmaes"` to reach the other continuous algorithms, all of which share this same `fit(objective, bounds=...)` calling convention through `AutoColony`. See [AutoColony API](autocolony-api.md) for the full parameter reference and [Algorithms](algorithms.md) for guidance on choosing between them.

## Discrete optimization

Pass a square distance matrix for ACO:

```python
optimizer = AutoColony(mode="aco", n_iterations=100, random_state=7)
optimizer.fit([[0.0, 1.0], [1.0, 0.0]])
```

Here `X` is a square distance matrix rather than a callable: entry `(i, j)` is the cost of traveling from node `i` to node `j`, and [Ant Colony Optimization](algorithms/aco.md) searches for a short tour that visits every node once. No `bounds` argument is needed or accepted in this mode. After `fit()`, `optimizer.predict()` returns the best tour found as a list of node indices and `optimizer.score()` returns its total length — lower is better. For a worked example against a benchmark suite with a known optimum, see [Examples & Gallery](examples.md).

## Frequently asked questions

### Do I need a Rust toolchain to use colonyx?

No, not for normal use. `pip install colonyx` installs a prebuilt wheel with the Rust extension already compiled inside it. You only need Rust and `maturin` if you're building from a source checkout, as described above.

### What's the difference between `fit(objective, bounds=...)` and `fit(distance_matrix)`?

`colonyx` infers which calling convention to use from what you pass as `X`. A callable `X` (a Python function or lambda) is treated as a continuous objective to minimize, and requires a `bounds` argument describing the search box. A square 2D array or nested list is treated as a distance matrix for [ACO](algorithms/aco.md), and `bounds` is neither needed nor accepted. `mode="auto"` uses this same distinction, plus dimensionality, to pick an algorithm for you — see `recommend_algorithm()` in [AutoColony API](autocolony-api.md).

### Can I reproduce the exact same result across runs?

Yes — pass `random_state=<int>` to `AutoColony`, as every example on this page does. All randomness in the underlying Rust algorithms is seeded from this value, so two runs with the same mode, parameters, and `random_state` produce identical results.

### What if `fit()` raises a `ValueError` about bounds or mode?

`AutoColony` validates its inputs eagerly: passing a callable objective without `bounds` for a continuous mode, passing a non-square matrix to `mode="aco"`, or passing an unknown `mode` string all raise a descriptive `ValueError` immediately rather than failing deep inside the Rust core. Read the message — it names exactly what was expected — and check the corresponding section of [AutoColony API](autocolony-api.md).
