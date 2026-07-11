#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if command -v maturin >/dev/null 2>&1; then
  maturin build --release
  maturin sdist
else
  RUSTFLAGS='-C link-args=-undefined -C link-args=dynamic_lookup' cargo build --release
fi
