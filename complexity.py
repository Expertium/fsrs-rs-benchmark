"""
complexity.py — repo-wide code complexity metrics

Walks all Python and Rust source files in the repository (excluding build
artifacts in `target/` directories and Python caches in `__pycache__/`) and
prints three aggregate metrics:

  * AST node count  — Python: stdlib `ast`; Rust: tree-sitter CST
  * Cyclomatic complexity — Python: `radon`; Rust: `lizard`
  * Lines of code (LOC) — raw line count for each file

Usage:
    uv run complexity.py          # from repo root
    python complexity.py          # if dependencies are already installed
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import NamedTuple

try:
    from radon.complexity import cc_visit
except ImportError:
    sys.exit("radon is required: pip install radon")

try:
    import lizard
except ImportError:
    sys.exit("lizard is required: pip install lizard")

try:
    import tree_sitter_rust as _tsr
    from tree_sitter import Language, Parser as TSParser

    _RUST_LANG = Language(_tsr.language())
    _rust_parser = TSParser(_RUST_LANG)
except ImportError:
    sys.exit("tree-sitter-rust is required: pip install tree-sitter tree-sitter-rust")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent

# `profiling/` holds profiling-only tooling (constraint 10) that is never imported
# by the timed compute_parameters() path, so it is excluded from the complexity score.
_SKIP_DIRS = {"target", "__pycache__", ".git", "profiling"}
# Read-only tooling — the results aggregator, the history plotter, and this
# complexity scorer itself — is not part of the optimization's mutation surface,
# so it is excluded from the score (per CLAUDE.md). benchmark.py is excluded too:
# it is the correctness-reference harness, only ever run for the bit-for-bit check,
# never on the optimization's timed path. Still COUNTED: compute_parameters.py and
# its preprocessing (config/data_loader/utils/features/*, fsrs_rs_python/__init__),
# where the "no moving Rust into untimed Python" rule applies.
_SKIP_FILES = {"benchmark.py", "complexity.py", "evaluate.py", "plot_history.py"}


def _collect_files(root: Path, suffix: str) -> list[Path]:
    files = []
    for path in sorted(root.rglob(f"*{suffix}")):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if path.name in _SKIP_FILES:
            continue
        files.append(path)
    return files


def _count_rust_ast_nodes(root) -> int:
    """Iterative node count to avoid recursion-depth issues on large files."""
    total = 0
    stack = [root]
    while stack:
        node = stack.pop()
        total += 1
        stack.extend(node.children)
    return total


def _loc(text: str) -> int:
    return text.count("\n") + (1 if text and not text.endswith("\n") else 0)


# ---------------------------------------------------------------------------
# Per-file metrics
# ---------------------------------------------------------------------------

class FileMetrics(NamedTuple):
    path: Path
    loc: int
    ast_nodes: int
    cyclomatic: int


def _total_cyclomatic(blocks) -> int:
    """Cyclomatic complexity over every function, recursing into nested
    closures/lambdas and class methods.

    Summing only the top-level blocks returned by ``cc_visit`` lets code hide
    branches inside nested closures (radon does not roll a closure's branches
    up into its enclosing function), which makes the score game-able — exactly
    what it is meant to prevent. Recursing closes that loophole so a branch
    costs the same whether it sits in a function body or is buried in a lambda.
    """
    total = 0
    for block in blocks:
        methods = getattr(block, "methods", None)
        if methods is not None:  # a class: count its methods, not its rolled-up total
            total += _total_cyclomatic(methods)
        else:  # a function: its own branches plus any nested closures/lambdas
            total += block.complexity
            total += _total_cyclomatic(getattr(block, "closures", []) or [])
    return total


def _python_metrics(path: Path) -> FileMetrics:
    source = path.read_text(encoding="utf-8")

    tree = ast.parse(source)
    ast_nodes = sum(1 for _ in ast.walk(tree))

    cyclomatic = _total_cyclomatic(cc_visit(source))

    return FileMetrics(path=path, loc=_loc(source), ast_nodes=ast_nodes, cyclomatic=cyclomatic)


def _rust_metrics(path: Path) -> FileMetrics:
    source_bytes = path.read_bytes()
    source_text = source_bytes.decode("utf-8", errors="replace")

    ts_tree = _rust_parser.parse(source_bytes)
    ast_nodes = _count_rust_ast_nodes(ts_tree.root_node)

    lz = lizard.analyze_file.analyze_source_code(path.name, source_text)
    cyclomatic = sum(fn.cyclomatic_complexity for fn in lz.function_list)

    return FileMetrics(path=path, loc=_loc(source_text), ast_nodes=ast_nodes, cyclomatic=cyclomatic)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    python_files = _collect_files(_REPO_ROOT, ".py")
    rust_files = _collect_files(_REPO_ROOT, ".rs")

    all_metrics: list[FileMetrics] = []

    print(f"{'File':<60}  {'LOC':>6}  {'AST nodes':>10}  {'Cyclomatic':>10}")
    print("-" * 92)

    print("\n[Python]")
    py_metrics = [_python_metrics(p) for p in python_files]
    for m in py_metrics:
        rel = m.path.relative_to(_REPO_ROOT)
        print(f"  {str(rel):<58}  {m.loc:>6}  {m.ast_nodes:>10}  {m.cyclomatic:>10}")
    all_metrics.extend(py_metrics)

    print("\n[Rust]")
    rs_metrics = [_rust_metrics(p) for p in rust_files]
    for m in rs_metrics:
        rel = m.path.relative_to(_REPO_ROOT)
        print(f"  {str(rel):<58}  {m.loc:>6}  {m.ast_nodes:>10}  {m.cyclomatic:>10}")
    all_metrics.extend(rs_metrics)

    total_loc = sum(m.loc for m in all_metrics)
    total_ast = sum(m.ast_nodes for m in all_metrics)
    total_cyclo = sum(m.cyclomatic for m in all_metrics)

    py_loc = sum(m.loc for m in py_metrics)
    py_ast = sum(m.ast_nodes for m in py_metrics)
    py_cyclo = sum(m.cyclomatic for m in py_metrics)

    rs_loc = sum(m.loc for m in rs_metrics)
    rs_ast = sum(m.ast_nodes for m in rs_metrics)
    rs_cyclo = sum(m.cyclomatic for m in rs_metrics)

    print()
    print("-" * 92)
    print(f"\n{'Subtotals':<60}  {'LOC':>6}  {'AST nodes':>10}  {'Cyclomatic':>10}")
    print(f"  {'Python':<58}  {py_loc:>6}  {py_ast:>10}  {py_cyclo:>10}")
    print(f"  {'Rust':<58}  {rs_loc:>6}  {rs_ast:>10}  {rs_cyclo:>10}")
    print()
    print(f"  {'TOTAL':<58}  {total_loc:>6}  {total_ast:>10}  {total_cyclo:>10}")
    print(f"  Score={int(total_loc*0.3 + total_ast*0.05 + total_cyclo*4.25)}")


if __name__ == "__main__":
    main()
