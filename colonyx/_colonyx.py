"""Loader shim for the compiled ``colonyx._colonyx`` Rust extension."""

from __future__ import annotations

import sys
import sysconfig
from importlib.machinery import ExtensionFileLoader
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _candidate_paths() -> list[Path]:
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    package_dir = Path(__file__).resolve().parent
    package_name = package_dir.name
    project_root = package_dir.parent

    candidates: list[Path] = []
    for entry in map(Path, sys.path):
        if not entry:
            continue

        search_roots = [entry, entry / package_name]
        for root in search_roots:
            if not root.exists():
                continue
            for path in root.glob(f"_colonyx*{suffix}"):
                if path.resolve().parent == package_dir:
                    continue
                candidates.append(path)

    for pattern in ("libcolonyx*", "lib_colonyx*"):
        for path in project_root.glob(f"target/**/{pattern}"):
            if path.suffix in {".dylib", ".so"}:
                candidates.append(path)

    return sorted({path.resolve() for path in candidates})


def _load_native_module():
    candidates = _candidate_paths()
    if not candidates:
        raise ImportError(
            "The Rust extension colonyx._colonyx is not installed. "
            "Run `maturin develop` or install a built wheel first."
        )

    loader = ExtensionFileLoader("_colonyx", str(candidates[0]))
    spec = spec_from_file_location("_colonyx", candidates[0], loader=loader)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Rust extension from {candidates[0]}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_native = _load_native_module()

for name in dir(_native):
    if not name.startswith("_"):
        globals()[name] = getattr(_native, name)

__all__ = [name for name in dir(_native) if not name.startswith("_")]
__doc__ = _native.__doc__
