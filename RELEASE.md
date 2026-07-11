# Release Checklist

This checklist avoids any secret handling. Use it when you are ready to publish.

For scripted local checks and publishing helpers, see `scripts/README.md`.

## Before release

- Bump `version` in `Cargo.toml`.
- Keep the Python package version in sync if you expose it in `colonyx/__init__.py`; the current code reads package metadata first.
- Update `README.md` and `docs/release.md` if the public API changed.
- Run `cargo fmt --all`.
- Run `cargo check`.
- Run `python3 -m py_compile colonyx/__init__.py colonyx/_colonyx.py colonyx/auto.py`.

## Build artifacts

- Build the native extension locally.
  - `RUSTFLAGS='-C link-args=-undefined -C link-args=dynamic_lookup' cargo build --release`
- Build a wheel and sdist with `maturin` when available.
  - `maturin build --release`
  - `maturin sdist`

## Validation

- Run `pytest` in the prepared environment.
- Verify imports from a clean shell:
  - `python3 -c "import colonyx; print(colonyx.__version__)"`
- Validate crate packaging offline when network access is unavailable:
  - `cargo package --allow-dirty --no-verify --offline`

## Publish

- PyPI:
  - `twine upload dist/*` or `maturin publish`
- crates.io:
  - `cargo publish --dry-run`
  - `cargo publish`

## Notes

- If you need tokens or trusted publishing setup, handle that separately.
- Keep the release tags annotated and versioned consistently across Python and Rust.
