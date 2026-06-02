"""Append a history entry and regenerate result/history.md (profiling-only tooling).

The durable record is result/history.jsonl (one JSON object per line). This script
appends one entry and rebuilds the human-readable result/history.md table from it.
The private `comment` field (CLAUDE.md history item 12) is stored in the JSONL but
NEVER shown in the .md.

Add an entry (all numeric fields optional; ratios/threshold are derived):
  python profiling/log_history.py add \
      --iteration 1 --time-before 3067 --time-after 2900 \
      --speed-ratio 1.058 --mean-speedup 1.061 \
      --complexity-before 4771 --complexity-after 4780 \
      --checks-passed true --status accepted \
      --summary "hoist loop-invariant weight slices out of seq loop" \
      --comment "burn accumulates grads in reverse-topo order; bit-for-bit held"

Just rebuild the .md from the .jsonl (e.g. after hand-editing):
  python profiling/log_history.py rebuild
"""

import argparse
import datetime as _dt
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_JSONL = _REPO / "result" / "history.jsonl"
_MD = _REPO / "result" / "history.md"


def _rebuild_md() -> None:
    rows = []
    if _JSONL.exists():
        rows = [json.loads(l) for l in _JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows.sort(key=lambda r: r["iteration"])

    lines = []
    lines.append("# FSRS-rs speed autoresearch — iteration history")
    lines.append("")
    lines.append("Accept metric: **median per-user speed_ratio ≥ 1.05** (constraint 12) AND "
                 "**speed_ratio ≥ complexity_ratio^2.5** (constraint 13). "
                 "speed_ratio is the *median of per-user ratios*, measured back-to-back vs the "
                 "then-current champion. Times are machine/session-specific (informational).")
    lines.append("")
    # Per CLAUDE.md history items 3-11, the displayed speed metric is the MEDIAN speed_ratio
    # only. mean_speedup is informational (constraint 12) — kept in the .jsonl, NOT shown here.
    hdr = ("| iter | time_before (ms) | time_after (ms) | speed_ratio | "
           "cplx_before | cplx_after | cplx_ratio | cplx^2.5 | checks | status | summary |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    lines.append(hdr)
    lines.append(sep)
    cum = 1.0
    for r in rows:
        sr = r.get("speed_ratio")
        if r.get("status") == "accepted" and sr is not None:
            cum *= float(sr)
        def fmt(x, nd=3):
            return "" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))
        lines.append(
            f"| {r.get('iteration','')} "
            f"| {fmt(r.get('time_before'),0)} "
            f"| {fmt(r.get('time_after'),0)} "
            f"| {fmt(r.get('speed_ratio'))} "
            f"| {fmt(r.get('complexity_before'),0)} "
            f"| {fmt(r.get('complexity_after'),0)} "
            f"| {fmt(r.get('complexity_ratio'))} "
            f"| {fmt(r.get('complexity_threshold'))} "
            f"| {'✓' if r.get('checks_passed') else '✗'} "
            f"| {r.get('status','')} "
            f"| {r.get('summary','')} |"
        )
    lines.append("")
    lines.append(f"**Cumulative speed_ratio (product of accepted): ×{cum:.3f}** "
                 f"— upward-biased (winner's curse); anchor periodically vs iter-0 baseline.")
    lines.append("")
    _MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_bool(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "y", "t")


def cmd_add(a: argparse.Namespace) -> None:
    entry = {
        "iteration": a.iteration,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "time_before": a.time_before,
        "time_after": a.time_after,
        "speed_ratio": a.speed_ratio,
        "mean_speedup": a.mean_speedup,
        "complexity_before": a.complexity_before,
        "complexity_after": a.complexity_after,
        "checks_passed": _parse_bool(a.checks_passed),
        "status": a.status,
        "summary": a.summary,
        "comment": a.comment,
    }
    if a.complexity_before and a.complexity_after:
        cr = a.complexity_after / a.complexity_before
        entry["complexity_ratio"] = round(cr, 4)
        entry["complexity_threshold"] = round(cr ** 2.5, 4)
    _JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(_JSONL, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _rebuild_md()
    print(f"appended iter {a.iteration} ({a.status}); rebuilt {_MD.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pa = sub.add_parser("add")
    pa.add_argument("--iteration", type=int, required=True)
    pa.add_argument("--time-before", type=float, default=None)
    pa.add_argument("--time-after", type=float, default=None)
    pa.add_argument("--speed-ratio", type=float, default=None)
    pa.add_argument("--mean-speedup", type=float, default=None)
    pa.add_argument("--complexity-before", type=int, default=None)
    pa.add_argument("--complexity-after", type=int, default=None)
    pa.add_argument("--checks-passed", default="false")
    pa.add_argument("--status", required=True, choices=["accepted", "rejected"])
    pa.add_argument("--summary", required=True)
    pa.add_argument("--comment", default="")
    sub.add_parser("rebuild")
    a = ap.parse_args()
    if a.cmd == "add":
        cmd_add(a)
    else:
        _rebuild_md()
        print(f"rebuilt {_MD.name}")


if __name__ == "__main__":
    main()
