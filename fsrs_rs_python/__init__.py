from __future__ import annotations

import importlib.machinery
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType


_ROOT = Path(__file__).resolve().parent.parent
_CRATE_DIR = _ROOT / "fsrs-rs-python"
_MANIFEST_PATH = _CRATE_DIR / "Cargo.toml"
_MODULE_NAME = f"{__name__}.fsrs_rs_python"
_VALID_EXTENSION_SUFFIXES = tuple(importlib.machinery.EXTENSION_SUFFIXES)
_EXTENSION_BASENAMES = ("libfsrs_rs_python", "fsrs_rs_python")


def _matches_extension_suffix(candidate: Path) -> bool:
    return any(candidate.name.endswith(suffix) for suffix in _VALID_EXTENSION_SUFFIXES)


def _prepare_windows_extension_aliases() -> None:
    if sys.platform != "win32":
        return

    pyd_suffix = next(
        (
            suffix
            for suffix in _VALID_EXTENSION_SUFFIXES
            if suffix.endswith(".pyd")
        ),
        _VALID_EXTENSION_SUFFIXES[0],
    )

    for profile in ("release", "debug"):
        profile_dir = _CRATE_DIR / "target" / profile
        if not profile_dir.exists():
            continue
        for basename in _EXTENSION_BASENAMES:
            for candidate in sorted(profile_dir.glob(f"{basename}.dll")):
                alias = candidate.with_name(f"{basename}{pyd_suffix}")
                if (
                    alias.exists()
                    and alias.stat().st_mtime_ns >= candidate.stat().st_mtime_ns
                ):
                    continue
                shutil.copy2(candidate, alias)


def _extension_candidates() -> list[Path]:
    _prepare_windows_extension_aliases()
    candidates: list[Path] = []
    for profile in ("release", "debug"):
        profile_dir = _CRATE_DIR / "target" / profile
        if not profile_dir.exists():
            continue
        for basename in _EXTENSION_BASENAMES:
            candidates.extend(
                sorted(
                    candidate
                    for candidate in profile_dir.glob(f"{basename}.*")
                    if _matches_extension_suffix(candidate)
                )
            )
    return candidates


def _load_extension() -> ModuleType:
    errors: list[str] = []
    for candidate in _extension_candidates():
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, candidate)
        if spec is None or spec.loader is None:
            errors.append(f"{candidate}: missing import spec or loader")
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[_MODULE_NAME] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(_MODULE_NAME, None)
            errors.append(f"{candidate}: {exc}")
            continue
        return module
    details = "\n".join(errors)
    raise ImportError(
        "Unable to load fsrs-rs-python extension from local build artifacts"
        + (f"\n{details}" if details else "")
    )


def _build_extension() -> None:
    try:
        subprocess.run(
            ["cargo", "build", "--release", "--manifest-path", str(_MANIFEST_PATH)],
            cwd=_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ImportError(
            "Failed to build the vendored fsrs-rs-python extension because `cargo` "
            "was not found on PATH. Please install Rust/Cargo first."
        ) from exc
    except subprocess.CalledProcessError as exc:
        details = "\n".join(
            part.strip() for part in (exc.stdout, exc.stderr) if part and part.strip()
        )
        raise ImportError(
            "Failed to build the vendored fsrs-rs-python extension. "
            "Please make sure Rust/Cargo is installed and available on PATH."
            + (f"\n{details}" if details else "")
        ) from exc


try:
    fsrs_rs_python = _load_extension()
except ImportError:
    _build_extension()
    try:
        fsrs_rs_python = _load_extension()
    except ImportError as exc:
        raise ImportError(
            "Built the vendored fsrs-rs-python extension, but importing it still failed."
        ) from exc

__doc__ = fsrs_rs_python.__doc__
if hasattr(fsrs_rs_python, "__all__"):
    __all__ = fsrs_rs_python.__all__
else:
    __all__ = [name for name in dir(fsrs_rs_python) if not name.startswith("_")]

globals().update({name: getattr(fsrs_rs_python, name) for name in __all__})
