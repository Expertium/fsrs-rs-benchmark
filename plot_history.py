#!/usr/bin/env python3
"""
Visualize the FSRS-rs speed autoresearch iteration history.

Reads ``result/history.jsonl`` and draws median per-user time vs iteration:

* green dots + green step line — champions (the best-so-far frontier), each
  annotated with its (truncated) summary at an angle;
* grey dots                   — rejected variants.

Expected per-row fields: ``iteration``, ``status`` ("accepted"/"rejected"),
``summary``, and ``time_after`` (the candidate's median per-user time, ms).

    python plot_history.py                 # -> result/history_plot.png
    python plot_history.py --no-summaries
    python plot_history.py --summary-len 60 --rotation 35
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: render straight to a file, no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

REPO = Path(__file__).resolve().parent
HISTORY = REPO / "result" / "history.jsonl"
DEFAULT_OUT = REPO / "result" / "history_plot.png"

# The best-so-far frontier is the set of accepted records (the iter-0 baseline
# is "accepted" too). "champion" is not a status value.
CHAMPION = {"accepted"}
GREEN = "#2ca02c"
GREEN_DARK = "#176117"
GREY = "0.70"


def load(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"no history at {path}")
    rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    rows = [r for r in rows if r.get("time_after") is not None]
    rows.sort(key=lambda r: r["iteration"])
    return rows


def truncate(s: str, n: int) -> str:
    s = " ".join((s or "").split())
    if len(s) <= n:
        return s
    return s[:n].rsplit(" ", 1)[0] + "…"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-summaries", action="store_true",
                    help="hide the champion summary labels")
    ap.add_argument("--summary-len", type=int, default=44,
                    help="max chars of summary shown per champion (default 44)")
    ap.add_argument("--rotation", type=float, default=40.0,
                    help="summary label angle, degrees (default 40)")
    args = ap.parse_args()

    rows = load(HISTORY)
    champs = [r for r in rows if r.get("status") in CHAMPION]
    rejects = [r for r in rows if r.get("status") == "rejected"]
    if not champs:
        raise SystemExit("no champion records to plot")

    xs = [r["iteration"] for r in rows]
    ys = [r["time_after"] for r in rows]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    yr = (ymax - ymin) or 1.0

    fig, ax = plt.subplots(figsize=(14, 7))

    if rejects:
        ax.scatter([r["iteration"] for r in rejects],
                   [r["time_after"] for r in rejects],
                   s=28, c=GREY, zorder=2, label="rejected")

    # best-so-far frontier: step through champions, then a flat tail to the edge
    cx = [r["iteration"] for r in champs]
    cy = [r["time_after"] for r in champs]
    ax.step(cx + [xmax], cy + [cy[-1]], where="post", color=GREEN, lw=2.3, zorder=3)
    ax.scatter(cx, cy, s=62, color=GREEN, edgecolors="white", linewidths=0.8,
               zorder=4, label="champion")

    if not args.no_summaries:
        for r in champs:
            ax.annotate(truncate(r.get("summary", ""), args.summary_len),
                        (r["iteration"], r["time_after"]),
                        textcoords="offset points", xytext=(7, 7),
                        rotation=args.rotation, rotation_mode="anchor",
                        ha="left", va="bottom", fontsize=7, color=GREEN_DARK)

    ax.set_xlabel("Iteration", fontsize=15)
    ax.set_ylabel("Median per-user time (ms)", fontsize=15)
    ax.set_title(f"FSRS-rs speed autoresearch — {len(champs)-1} accepted speedups, {len(rejects)} rejected variants",
                 fontsize=18)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.9)

    # extra top headroom for the angled labels (generic, scale-agnostic)
    pad_top = (0.50 if not args.no_summaries else 0.10) * yr
    ax.set_ylim(ymin - 0.06 * yr, ymax + pad_top)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.set_xlim(xmin - 0.6, xmax + max(2.0, 0.08 * (xmax - xmin)))

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}  ({len(champs)} champions, {len(rejects)} rejected, "
          f"latest iter {xmax})")


if __name__ == "__main__":
    main()
