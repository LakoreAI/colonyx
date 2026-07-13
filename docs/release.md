# Release

## Local build

```bash
RUSTFLAGS='-C link-args=-undefined -C link-args=dynamic_lookup' cargo build --release
```

When `maturin` is available, prefer:

```bash
maturin build --release
```

## Local import check

```bash
python3 -c "import colonyx; print(colonyx.__version__)"
```

For scripted checks and publish helpers, see `scripts/README.md`.

## Offline package validation

When you do not have network access, you can still validate the crate manifest and packaging output:

```bash
cargo package --allow-dirty --no-verify --offline
```

## PyPI

Before publishing:

1. Bump the version in `Cargo.toml`.
2. Update the fallback version in `colonyx/__init__.py` if you keep one.
3. Build the wheel and sdist with `maturin`.
4. Upload with `twine` or `maturin publish`.

Dry-run first:

```bash
maturin build --release
twine check dist/*
```

## crates.io

Before publishing:

1. Confirm `Cargo.toml` metadata is complete.
2. Bump the crate version in `Cargo.toml`.
3. Run `cargo publish --dry-run`.
4. Publish with `cargo publish`.

## Rust usage

Rust consumers can depend on the crate directly and use the modules under
`colonyx::core` and `colonyx::algorithms`.

```toml
[dependencies]
colonyx = "0.1.1"
```

See `docs/rust.md` for examples.

If you only want to validate the manifest, use:

```bash
cargo publish --dry-run
```
