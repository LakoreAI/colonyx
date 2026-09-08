---
title: Release Process
description: How to build, version, and publish colonyx to PyPI and crates.io, and how to check which version you have installed.
---

# Release

This page documents how colonyx itself is built, versioned, and published — useful if you're contributing to the project, packaging it for another environment, or just trying to figure out which version you have installed and whether it's current.

## Checking your installed version

```bash
python3 -c "import colonyx; print(colonyx.__version__)"
```

`colonyx.__version__` is read from the installed package's metadata via `importlib.metadata.version("colonyx")`, falling back to a hardcoded string (`0.3.0` as of this writing) only for the rare case of running from an editable source checkout without installed metadata. colonyx follows [semantic versioning](https://semver.org/): patch releases (`0.3.0` → `0.3.1`) are bug fixes with no API changes, minor releases (`0.3.x` → `0.4.0`) can add new modes, parameters, or metrics without breaking existing code, and — once the project reaches `1.0.0` — a major version bump would be reserved for breaking changes to `AutoColony`'s public interface. Before `1.0.0` (the package is currently `Development Status :: 3 - Alpha` per `pyproject.toml`), treat even minor versions as potentially carrying small interface adjustments, and pin an exact version in production dependencies rather than a loose range.

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
colonyx = "0.3"
```

See [Rust Usage](rust.md) for examples.

If you only want to validate the manifest, use:

```bash
cargo publish --dry-run
```

## Where to next

- [Rust Usage](rust.md) for why the optimization core is written in Rust and how to consume it as a crate.
- [Getting Started](getting-started.md) for installing the published package rather than building from source.
- [API Reference](api.md) for what's actually exported once you've installed a given version.
