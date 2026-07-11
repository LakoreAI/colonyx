#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

cargo publish --dry-run

if [[ -n "${CARGO_REGISTRY_TOKEN:-}" ]]; then
  export CARGO_REGISTRY_TOKEN
elif [[ -n "${CRATES_IO_TOKEN:-}" ]]; then
  export CARGO_REGISTRY_TOKEN="$CRATES_IO_TOKEN"
fi

cargo publish
