from __future__ import annotations

import importlib.machinery
import importlib.util
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


def _load_extension() -> ModuleType | None:
    """Try to load a pre-built extension from the Cargo target directory."""
    candidates: list[Path] = []
    seen: set[Path] = set()
    for profile in ("release", "debug"):
        for subdir in ("", "deps"):
            d = _CRATE_DIR / "target" / profile / subdir
            if d.exists():
                for basename in _EXTENSION_BASENAMES:
                    for candidate in sorted(d.glob(f"{basename}.*")):
                        if candidate not in seen:
                            candidates.append(candidate)
                            seen.add(candidate)

    # On Windows, Cargo produces a .dll; copy it to .pyd so Python can import it.
    if sys.platform == "win32":
        pyd_suffix = next((s for s in _EXTENSION_SUFFIXES if s.endswith(".pyd")), None)
        if pyd_suffix:
            for candidate in list(candidates):
                if candidate.suffix == ".dll":
                    alias = candidate.with_name(candidate.stem + pyd_suffix)
                    if not alias.exists():
                        try:
                            shutil.copy2(candidate, alias)
                        except PermissionError:
                            pass
                    if alias not in seen:
                        candidates.append(alias)
                        seen.add(alias)

    for candidate in candidates:
        if not any(candidate.name.endswith(s) for s in _EXTENSION_SUFFIXES):
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
        details = "\n".join(
            part.strip() for part in (exc.stdout, exc.stderr) if part and part.strip()
        )
        raise ImportError(
            "Failed to build fsrs-rs-python. "
            "Please make sure Rust/Cargo is installed and available on PATH."
            + (f"\n{details}" if details else "")
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
