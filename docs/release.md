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

1. Bump the version in `pyproject.toml`.
2. Build the wheel and sdist with `maturin`.
3. Upload with `twine` or `maturin publish`.

Dry-run first:

```bash
maturin build --release
twine check dist/*
```

## crates.io

Before publishing:

1. Confirm `Cargo.toml` metadata is complete.
2. Bump the crate version.
3. Run `cargo publish --dry-run`.
4. Publish with `cargo publish`.

If you only want to validate the manifest, use:

```bash
cargo publish --dry-run
```
