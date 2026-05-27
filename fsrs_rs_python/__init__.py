from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
_CRATE_DIR = _ROOT / "fsrs-rs-python"
_MANIFEST_PATH = _CRATE_DIR / "Cargo.toml"
_MODULE_NAME = f"{__name__}.fsrs_rs_python"


def _extension_candidates() -> list[Path]:
    candidates: list[Path] = []
    for profile in ("release", "debug"):
        profile_dir = _CRATE_DIR / "target" / profile
        if not profile_dir.exists():
            continue
        candidates.extend(sorted(profile_dir.glob("libfsrs_rs_python.*"), reverse=True))
    return candidates


def _load_extension() -> object:
    for candidate in _extension_candidates():
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[_MODULE_NAME] = module
        spec.loader.exec_module(module)
        return module
    raise ImportError("Unable to load fsrs-rs-python extension from local build artifacts")


def _build_extension() -> None:
    subprocess.run(
        ["cargo", "build", "--manifest-path", str(_MANIFEST_PATH)],
        cwd=_ROOT,
        check=True,
    )


try:
    fsrs_rs_python = _load_extension()
except ImportError:
    _build_extension()
    fsrs_rs_python = _load_extension()

__doc__ = fsrs_rs_python.__doc__
if hasattr(fsrs_rs_python, "__all__"):
    __all__ = fsrs_rs_python.__all__
else:
    __all__ = [name for name in dir(fsrs_rs_python) if not name.startswith("_")]

globals().update({name: getattr(fsrs_rs_python, name) for name in __all__})
