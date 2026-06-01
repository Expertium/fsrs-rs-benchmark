from __future__ import annotations

import importlib.machinery
import importlib.util
import itertools
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

_CRATE_DIR = Path(__file__).resolve().parent
_ROOT = _CRATE_DIR.parent
_MANIFEST_PATH = _CRATE_DIR / "Cargo.toml"
_MODULE_NAME = f"{__name__}.fsrs_rs_python"
_EXTENSION_SUFFIXES = tuple(importlib.machinery.EXTENSION_SUFFIXES)
_EXTENSION_BASENAMES = ("libfsrs_rs_python", "fsrs_rs_python")
_PYD_SUFFIX = next((s for s in _EXTENSION_SUFFIXES if s.endswith(".pyd")), None)


def _candidate_extensions() -> list[Path]:
    """Built extension files in the Cargo target dir (release before debug)."""
    candidates: list[Path] = []
    seen: set[Path] = set()
    for profile, subdir, basename in itertools.product(
        ("release", "debug"), ("", "deps"), _EXTENSION_BASENAMES
    ):
        directory = _CRATE_DIR / "target" / profile / subdir
        if not directory.exists():
            continue
        for candidate in sorted(directory.glob(f"{basename}.*")):
            if candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)

    # On Windows, Cargo produces a .dll; copy it to .pyd so Python can import it.
    if sys.platform == "win32" and _PYD_SUFFIX:
        for candidate in list(candidates):
            if candidate.suffix != ".dll":
                continue
            alias = candidate.with_name(candidate.stem + _PYD_SUFFIX)
            if not alias.exists():
                try:
                    shutil.copy2(candidate, alias)
                except PermissionError:
                    pass
            if alias not in seen:
                candidates.append(alias)
                seen.add(alias)
    return candidates


def _load_extension() -> ModuleType | None:
    """Try to import a pre-built extension from the Cargo target directory."""
    for candidate in _candidate_extensions():
        if not any(map(candidate.name.endswith, _EXTENSION_SUFFIXES)):
            continue
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[_MODULE_NAME] = module
        try:
            spec.loader.exec_module(module)
            return module
        except Exception:
            sys.modules.pop(_MODULE_NAME, None)
    return None


def _build_extension() -> None:
    try:
        subprocess.run(
            [
                "cargo", "build", "--release",
                "--manifest-path", str(_MANIFEST_PATH),
                "--features", "pyo3/extension-module",
            ],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ImportError(
            "Failed to build fsrs-rs-python: `cargo` not found on PATH. "
            "Please install Rust/Cargo first."
        ) from exc
    except subprocess.CalledProcessError as exc:
        details = "\n".join(filter(None, map(str.strip, filter(None, (exc.stdout, exc.stderr)))))
        raise ImportError(
            "Failed to build fsrs-rs-python. "
            "Please make sure Rust/Cargo is installed and available on PATH."
            + "\n" * bool(details) + details
        ) from exc


_ext = _load_extension()
if _ext is None:
    _build_extension()
    _ext = _load_extension()
    if _ext is None:
        raise ImportError("Built fsrs-rs-python, but importing it still failed.")

__doc__ = _ext.__doc__
__all__ = getattr(_ext, "__all__", [name for name in dir(_ext) if not name.startswith("_")])
globals().update({name: getattr(_ext, name) for name in __all__})
