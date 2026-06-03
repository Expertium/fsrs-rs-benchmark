"""
log_history.py — append one iteration to the campaign history (read-only tooling).

The durable per-iteration record is `result/history.jsonl` (machine-readable, with a
private `comment`) plus the human-readable table in `result/history.md`. Writing both by
hand every iteration is error-prone — the spots that bite are the derived numbers
(`complexity_threshold = complexity_ratio ** 2.5`, the cumulative product of accepted
`speed_ratio`s) and the markdown row formatting. This script does exactly those:

    # append an iteration (JSON object on stdin; long `comment` survives fine this way)
    python log_history.py add <<'JSON'
    {"iteration": 19, "time_before": 145.0, "time_after": 70.0, "speed_ratio": 2.0,
     "mean_speedup": 2.1, "complexity_after": 7250, "checks_passed": true,
     "status": "accepted", "summary": "…(<=15 words)…", "comment": "…private notes…"}
    JSON

    # integrity check: md cumulative == product of accepted jsonl speed_ratios, 1 row per iter
    python log_history.py verify

Pass `--benchmark` to operate on the Phase-2 benchmark() history instead
(`result/history_benchmark.jsonl` + `.md`) — same format, separate files:

    python log_history.py add --benchmark <<'JSON'
    { … }
    JSON
    python log_history.py verify --benchmark

`add` auto-fills the fields you can derive, so you only pass what you measured:
  * `timestamp`           — now() if absent
  * `complexity_before`   — the last ACCEPTED entry's `complexity_after` if absent
  * `complexity_ratio`    — complexity_after / complexity_before
  * `complexity_threshold`— complexity_ratio ** 2.5  (constraint 13's bar)
It then APPENDS to history.jsonl, APPENDS one row to the history.md table (existing rows are
never rewritten), and recomputes only the cumulative line. It refuses to add a duplicate
iteration number.

This is bookkeeping tooling, not part of the optimized compute path, so it is excluded from
complexity.py (like plot_history.py) — it must never count against the constraint-13 budget.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent

# Phase 1 (the default): optimize compute_parameters(). Phase 2 (--benchmark): optimize
# benchmark(). Same per-entry format and same derived-field logic; only the target files and the
# md header preamble differ. main() swaps these module globals when --benchmark is passed.
_JSONL = _REPO / "result" / "history.jsonl"
_MD = _REPO / "result" / "history.md"

_TABLE_HEADER = (
    "| iter | time_before (ms) | time_after (ms) | speed_ratio | cplx_before | cplx_after |"
    " cplx_ratio | cplx^2.5 | checks | status | summary |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|\n"
)

_MD_HEADER = (
    "# FSRS-rs speed autoresearch — iteration history\n\n"
    "Accept metric: **median per-user speed_ratio ≥ 1.05** (constraint 12) AND **speed_ratio ≥"
    " complexity_ratio^2.5** (constraint 13). speed_ratio is the *median of per-user ratios*,"
    " measured back-to-back vs the then-current champion. Times are machine/session-specific"
    " (informational).\n\n" + _TABLE_HEADER
)

# Phase-2 header: optimizing benchmark() (benchmark.py's O(N^2) per-prefix path). Only BIT-FOR-BIT
# changes are viable there — its mean log loss sits ~5e-5 under the band ceiling, so any inexact
# change busts it. time_before/after = the per-user MIN-of-3 Rust benchmark() time SUMMED over the
# 5 TimeSeriesSplit folds (profiling/measure_benchmark.py), median across users.
_MD_HEADER_BENCH = (
    "# FSRS-rs speed autoresearch — benchmark() iteration history (Phase 2)\n\n"
    "Accept metric: **median per-user speed_ratio ≥ 1.05** (constraint 12) AND **speed_ratio ≥"
    " complexity_ratio^2.5** (constraint 13). Phase 2 optimizes benchmark() (benchmark.py's O(N^2)"
    " per-prefix anchor path); only **BIT-FOR-BIT** changes are viable (its mean log loss is ~5e-5"
    " under the band ceiling). speed_ratio is the *median of per-user ratios* of the"
    " summed-over-folds Rust benchmark() time, measured back-to-back vs the then-current champion."
    " Times are machine/session-specific (informational).\n\n" + _TABLE_HEADER
)

# The fields a markdown row never derives — `add` requires these on stdin.
_REQUIRED = ("iteration", "time_before", "time_after", "speed_ratio",
             "complexity_after", "checks_passed", "status", "summary")


def _load_jsonl() -> list[dict]:
    if not _JSONL.exists():
        return []
    return [json.loads(line) for line in _JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]


def _cumulative(entries: list[dict]) -> float:
    """Running product of the accepted iterations' median speed_ratio (the headline metric)."""
    prod = 1.0
    for e in entries:
        if e.get("status") == "accepted":
            prod *= e["speed_ratio"]
    return prod


def _md_row(e: dict) -> str:
    chk = "✓" if e.get("checks_passed") else "✗"
    return (
        f"| {e['iteration']} | {e['time_before']:.0f} | {e['time_after']:.0f} | "
        f"{e['speed_ratio']:.3f} | {e['complexity_before']} | {e['complexity_after']} | "
        f"{e['complexity_ratio']:.3f} | {e['complexity_threshold']:.3f} | {chk} | "
        f"{e['status']} | {e['summary']} |"
    )


def _cumulative_line(entries: list[dict]) -> str:
    return (
        f"**Cumulative speed_ratio (product of accepted): ×{_cumulative(entries):.3f}** — "
        "upward-biased (winner's curse); anchor periodically vs iter-0 baseline."
    )


def cmd_add() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        print("[add] expected a JSON object on stdin", file=sys.stderr)
        return 2
    e = json.loads(raw)

    missing = [k for k in _REQUIRED if k not in e]
    if missing:
        print(f"[add] missing required field(s): {', '.join(missing)}", file=sys.stderr)
        return 2

    entries = _load_jsonl()
    if any(x["iteration"] == e["iteration"] for x in entries):
        print(f"[add] iteration {e['iteration']} already logged — refusing to duplicate", file=sys.stderr)
        return 1

    # Derived fields (only filled if not explicitly provided).
    e.setdefault("timestamp", datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
    if "complexity_before" not in e:
        accepted = [x for x in entries if x.get("status") == "accepted"]
        if not accepted:
            print("[add] no prior accepted entry to infer complexity_before — pass it explicitly", file=sys.stderr)
            return 2
        e["complexity_before"] = accepted[-1]["complexity_after"]
    cb, ca = e["complexity_before"], e["complexity_after"]
    ratio = ca / cb
    e.setdefault("complexity_ratio", round(ratio, 4))
    e.setdefault("complexity_threshold", round(ratio ** 2.5, 4))
    e.setdefault("mean_speedup", e["speed_ratio"])
    e.setdefault("comment", "")

    # Canonical key order (matches the existing log).
    order = ["iteration", "timestamp", "time_before", "time_after", "speed_ratio", "mean_speedup",
             "complexity_before", "complexity_after", "checks_passed", "status", "summary",
             "comment", "complexity_ratio", "complexity_threshold"]
    ordered = {k: e[k] for k in order if k in e}
    ordered.update({k: v for k, v in e.items() if k not in ordered})

    # Append to the jsonl.
    with _JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ordered) + "\n")
    all_entries = entries + [ordered]

    # Append the row to the md table (existing rows untouched) and refresh the cumulative line.
    _rewrite_md(all_entries, append_only_row=_md_row(ordered))

    sr = ordered["speed_ratio"]
    thr = ordered["complexity_threshold"]
    gate = "PASS" if (sr >= 1.05 and sr >= thr) else "below a bar"
    print(f"[add] iter {ordered['iteration']} ({ordered['status']}): speed_ratio {sr:.4f}, "
          f"cplx_ratio {ordered['complexity_ratio']:.4f} (thr {thr:.4f}) [{gate}]")
    print(f"[add] cumulative (product of accepted): ×{_cumulative(all_entries):.3f}")
    return 0


def _rewrite_md(entries: list[dict], append_only_row: str | None = None) -> None:
    """Refresh history.md. If `append_only_row` is given and the file already has a table, only the
    new row + cumulative line change (historical rows are preserved byte-for-byte). Otherwise the
    whole table is rebuilt from the jsonl."""
    if append_only_row is not None and _MD.exists():
        lines = _MD.read_text(encoding="utf-8").splitlines()
        cum_idx = next((i for i, l in enumerate(lines) if l.startswith("**Cumulative")), None)
        if cum_idx is not None:
            insert_at = cum_idx
            while insert_at > 0 and not lines[insert_at - 1].startswith("|"):
                insert_at -= 1
            lines.insert(insert_at, append_only_row)
            cum_idx = next(i for i, l in enumerate(lines) if l.startswith("**Cumulative"))
            lines[cum_idx] = _cumulative_line(entries)
            _MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return
    # Full rebuild (no existing table to append to).
    rows = "\n".join(_md_row(e) for e in entries)
    _MD.write_text(_MD_HEADER + rows + "\n\n" + _cumulative_line(entries) + "\n", encoding="utf-8")


def cmd_verify() -> int:
    entries = _load_jsonl()
    ok = True
    iters = [e["iteration"] for e in entries]
    if len(iters) != len(set(iters)):
        print("[verify] FAIL: duplicate iteration numbers in history.jsonl", file=sys.stderr)
        ok = False
    if _MD.exists():
        md = _MD.read_text(encoding="utf-8")
        n_rows = sum(1 for l in md.splitlines()
                     if l.startswith("|") and l[1:].lstrip()[:1].isdigit())
        if n_rows != len(entries):
            print(f"[verify] FAIL: history.md has {n_rows} data rows, history.jsonl has {len(entries)}", file=sys.stderr)
            ok = False
        want = _cumulative_line(entries)
        if want not in md:
            have = next((l for l in md.splitlines() if l.startswith("**Cumulative")), "<none>")
            print(f"[verify] FAIL: cumulative line mismatch\n  md:   {have}\n  want: {want}", file=sys.stderr)
            ok = False
    print(f"[verify] {len(entries)} iterations, cumulative ×{_cumulative(entries):.3f} — {'OK' if ok else 'PROBLEMS'}")
    return 0 if ok else 1


def main() -> int:
    argv = [a for a in sys.argv[1:] if a != "--benchmark"]
    benchmark = "--benchmark" in sys.argv[1:]
    if not argv or argv[0] not in {"add", "verify"}:
        print(__doc__)
        return 2
    if benchmark:
        global _JSONL, _MD, _MD_HEADER
        _JSONL = _REPO / "result" / "history_benchmark.jsonl"
        _MD = _REPO / "result" / "history_benchmark.md"
        _MD_HEADER = _MD_HEADER_BENCH
    return cmd_add() if argv[0] == "add" else cmd_verify()


if __name__ == "__main__":
    raise SystemExit(main())
