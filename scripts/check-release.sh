#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

cargo fmt --all
cargo check --offline
python3 -m py_compile colonyx/__init__.py colonyx/_colonyx.py colonyx/auto.py colonyx/cli.py colonyx/metrics.py
python3 -c "import colonyx; print(colonyx.__version__)"
cargo package --allow-dirty --no-verify --offline
