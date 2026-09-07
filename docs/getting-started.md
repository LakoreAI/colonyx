# Getting Started

## Install from PyPI

```bash
pip install colonyx
```

That's all most users need — skip to [Continuous optimization](#continuous-optimization)
below.

## Install from source

Building from source requires a Rust toolchain and, ideally,
[`maturin`](https://www.maturin.rs/) to build the `colonyx._colonyx`
extension in editable mode:

```bash
python3 -m pip install -U pip maturin
maturin develop
```

!!! note "macOS extension-module linking"
    On macOS, `cargo build`/`cargo test` (not `maturin develop`) need an
    extra linker flag because pyo3's `extension-module` feature doesn't
    produce a fully-linked binary Cargo can run directly:

    ```bash
    RUSTFLAGS='-C link-args=-undefined -C link-args=dynamic_lookup' cargo build --release
    ```

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

See `docs/rust.md` for a full Rust usage example.

## Continuous optimization

Pass a callable objective and per-dimension bounds:

```python
optimizer = AutoColony(mode="de", n_iterations=100, random_state=7)
optimizer.fit(lambda x: sum(v * v for v in x), bounds=[(-5, 5), (-5, 5)])
```

## Discrete optimization

Pass a square distance matrix for ACO:

```python
optimizer = AutoColony(mode="aco", n_iterations=100, random_state=7)
optimizer.fit([[0.0, 1.0], [1.0, 0.0]])
```
