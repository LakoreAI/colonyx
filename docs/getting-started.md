# Getting Started

## Install from source

```bash
python3 -m pip install -U pip
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
