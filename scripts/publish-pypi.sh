#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -z "${MATURIN_PYPI_TOKEN:-}" && -z "${TWINE_USERNAME:-}" ]]; then
  echo "Set MATURIN_PYPI_TOKEN for maturin publish or TWINE_USERNAME/TWINE_PASSWORD for twine upload." >&2
  exit 1
fi

if command -v maturin >/dev/null 2>&1; then
  maturin build --release
  maturin publish --non-interactive
elif command -v twine >/dev/null 2>&1; then
  twine upload dist/*
else
  echo "Install maturin or twine before publishing to PyPI." >&2
  exit 1
fi
