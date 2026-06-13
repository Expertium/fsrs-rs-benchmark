#!/usr/bin/env python3
"""
Hyperparameter tuner for the FSRS-7 Rust fork — the n_epoch x batch_size Pareto grid.

Ported from the CUDA autoresearch tuner (src/autoresearch/hp_tune.py). The CUDA
version drove ``docker compose run ... run.sh`` and read diagnostics.json; this
drives the fork's OWN harnesses as subprocesses, injecting the (n_epoch,
batch_size) operating point through the ``FSRS_N_EPOCHS`` / ``FSRS_BATCH_SIZE`` env
vars. Those are read inside Rust ``compute_parameters()`` (training.rs), so a cell
needs NO rebuild — every cell reuses the one champion .pyd.

BOTH axes come from ONE benchmark.py run per cell (Andrew 2026-06-12):
  * LOSS  = benchmark.py's cross-validated mean by-user log loss. Train==test loss
    cannot see overfitting, so a loss-only search would always push to
    max-epoch/min-batch; the speed/loss trade-off only exists on a held-out metric.
  * SPEED = sum over users of their summed-over-folds Rust benchmark() training
    seconds (benchmark.py's per-user ``time_ms`` — the binding's monotonic
    Rust-region clock, Python excluded; 1 rep, ample for >=2x cell differences).

The GOLD STANDARD is (n_epoch=8, batch_size=256) — the shipped Rust defaults. A
grid cell re-anchors the gold only if it PARETO-DOMINATES it: no worse on either
axis (log loss within LL_TOL, time within SPEED_TOL) and strictly better on >=1.

    python hp_tune.py --epoch-batch-grid --max-user-id 100   # the 20-cell Pareto grid
    python hp_tune.py --time-noise 3     --max-user-id 100    # calibrate SPEED_TOL
    python hp_tune.py --plot                                  # redraw result/hp_grid_plot.png

If the grid MOVES the gold, the fine training HPs (LR / Adam betas / L2 / recency)
have to be re-tuned at the new operating point — they are conditional on it. That
coordinate-descent pass is a deliberate follow-up (it edits training.rs consts and
rebuilds per trial), and per the user's rule it only runs IF the gold changes; see
``fine_hp_followup()``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
RESULT = REPO / "result"
CP_RESULT = RESULT / "compute_parameters-FSRS-rs-short-secs-recency.jsonl"
BM_RESULT = RESULT / "FSRS-rs-short-secs-recency.jsonl"
GRID_JSON = RESULT / "hp_grid.json"
GRID_PLOT = RESULT / "hp_grid_plot.png"

# ── the operating-point grid ─────────────────────────────────────────────────
# Gold = the shipped Rust defaults (training.rs TrainingConfig: num_epochs=9 —
# the era-iter-1 champion: no per-epoch validation, last-epoch params —
# batch_size=256). The grid sweeps these cells and re-anchors to any that
# Pareto-dominates the gold. Epoch set per Andrew (2026-06-12): {6, 9, 15, 20, 30};
# the gold (9,256) is one of the 20 cells, so the sweep = 1 gold + 19 candidates.
GOLD_EPOCH, GOLD_BATCH = 9, 256
GRID_EPOCHS = [6, 9, 15, 20, 30]
GRID_BATCHES = [128, 256, 512, 1024]
# The committed LR (training.rs default; the pre-194 value — the 3k ablation measured the
# iter-194 LR/L2/B2 triple, tuned for the batch reshuffle we dropped, -0.00025 WORSE at 3k).
GOLD_LR = 0.0188


def _scaled_lr(batch: int) -> float:
    """Adam sqrt LR-batch scaling, anchored at the committed (GOLD_LR, GOLD_BATCH): gives each
    grid cell a roughly-right LR — a cheap stand-in for re-tuning LR per cell, so batch != 256
    cells aren't handicapped by an LR tuned at 256 (ported from the CUDA tuner; the winner's LR
    is then fine-tuned precisely by the follow-up pass). Injected per cell via FSRS_LR."""
    return round(GOLD_LR * math.sqrt(batch / GOLD_BATCH), 4)

# LL_TOL: log-loss tie band (user 2026-06-05: raise to 5e-5 for the Rust port — the
# training is NOT bit-exact across runs, but the cross-val by-user mean is far more
# stable than that, so 5e-5 is a safe "same loss" band).
LL_TOL = 5e-5
# SPEED_TOL: fractional time band — within +/-this counts as "same speed". Default
# is conservative; refine it from `--time-noise` (it prints a suggested value).
SPEED_TOL = 0.03

# Harness invocation (the documented FSRS-rs config: short-term reviews, seconds
# intervals, recency weighting). --processes matches the campaign default.
COMMON_ARGS = ["--algo", "FSRS-rs", "--short", "--secs", "--recency"]


def _run_harness(script: str, n_users: int, epoch: int, batch: int, processes: int,
                 result_file: Path) -> list[dict]:
    """Run one harness (compute_parameters.py / benchmark.py) at the given operating
    point on the first ``n_users`` users, then return its per-user rows. The result
    file is REMOVED first — both harnesses skip already-processed users, so a stale
    file would make the run a silent no-op. The operating point rides in via env."""
    if result_file.exists():
        result_file.unlink()
    env = {**os.environ, "FSRS_N_EPOCHS": str(epoch), "FSRS_BATCH_SIZE": str(batch),
           "FSRS_LR": str(_scaled_lr(batch))}
    cmd = [sys.executable, script, *COMMON_ARGS,
           "--processes", str(processes), "--max-user-id", str(n_users)]
    proc = subprocess.run(cmd, cwd=str(REPO), env=env,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{script} exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout[-1500:]}\n"
            f"--- stderr ---\n{proc.stderr[-1500:]}"
        )
    if not result_file.exists():
        raise RuntimeError(f"{script} produced no {result_file.name}")
    return [json.loads(line) for line in result_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_cell(epoch: int, batch: int, n_users: int, processes: int) -> dict:
    """Measure one (epoch, batch) cell with ONE benchmark.py run on the same ``n_users`` users
    (per Andrew 2026-06-12 — both axes from benchmark()):
      * LOSS  = cross-val mean by-user log loss
      * SPEED = sum over users of their summed-over-folds Rust benchmark() training seconds
        (the binding's monotonic Rust-region time, recorded per user in benchmark.py's
        ``time_ms``; 1 rep — cell-to-cell differences are >=2x vs the ~1-2% timing noise)
    Returns a cell dict; on harness failure returns it with by_user/seconds = None
    so one bad cell can't kill the grid."""
    t0 = time.time()
    try:
        bm_rows = _run_harness("benchmark.py", n_users, epoch, batch, processes, BM_RESULT)
        by_user = statistics.mean(r["metrics"]["LogLoss"] for r in bm_rows)
        seconds = sum(r["time_ms"] for r in bm_rows) / 1000.0
        items = sum(r["size"] for r in bm_rows)
    except Exception as e:  # noqa: BLE001 — record the failure, keep sweeping
        print(f"[grid] epoch {epoch:>2} batch {batch:>4}: FAILED ({e})", flush=True)
        return {"epoch": epoch, "batch": batch, "lr": _scaled_lr(batch), "by_user": None,
                "seconds": None, "items": None, "throughput": None, "error": str(e)}
    throughput = items / seconds if seconds else 0.0
    print(f"[grid] epoch {epoch:>2} batch {batch:>4} (LR {_scaled_lr(batch):g}): "
          f"by_user={by_user:.6f}  train={seconds:.1f}s  throughput={throughput:,.0f} reviews/s  "
          f"({time.time() - t0:.0f}s wall)", flush=True)
    return {"epoch": epoch, "batch": batch, "lr": _scaled_lr(batch), "by_user": by_user,
            "seconds": seconds, "items": items, "throughput": throughput}


# ── Pareto logic ─────────────────────────────────────────────────────────────
def dominates(cand: dict, gold_ll: float, gold_s: float) -> bool:
    """True iff ``cand`` Pareto-DOMINATES the gold: no worse on either axis (log
    loss within LL_TOL, time within SPEED_TOL) and strictly better on >= 1 axis.
    This is the user's "beats the gold on at least one axis without losing the
    other" rule."""
    if cand["by_user"] is None:
        return False
    not_worse = (cand["by_user"] <= gold_ll + LL_TOL) and (cand["seconds"] <= gold_s * (1 + SPEED_TOL))
    strictly_better = (cand["by_user"] < gold_ll - LL_TOL) or (cand["seconds"] < gold_s * (1 - SPEED_TOL))
    return not_worse and strictly_better


def pareto_front(cells: list[dict]) -> list[dict]:
    """The non-dominated cells: lower loss is better, higher throughput (= lower
    seconds) is better. A cell is on the front if no other cell is >= on speed AND
    <= on loss with a strict win somewhere. Used for the plot's green frontier."""
    ok = [c for c in cells if c.get("by_user") is not None]
    front = []
    for c in ok:
        dominated = any(
            o is not c
            and o["by_user"] <= c["by_user"]
            and o["seconds"] <= c["seconds"]
            and (o["by_user"] < c["by_user"] or o["seconds"] < c["seconds"])
            for o in ok
        )
        if not dominated:
            front.append(c)
    return sorted(front, key=lambda c: c["by_user"])


# ── the grid ─────────────────────────────────────────────────────────────────
def epoch_batch_grid(n_users: int, processes: int) -> None:
    """Measure the gold standard (8, 256) first, then the 19 other cells, judging
    each against the gold. Picks the operating point = lowest log loss among cells
    not slower than gold (tiebreak fastest) — so if nothing dominates, gold stays.
    Writes result/hp_grid.json and the plot. Does NOT edit any config or commit:
    re-anchoring the gold is a deliberate step the user takes after review."""
    print(f"[grid] gold = (epoch {GOLD_EPOCH}, batch {GOLD_BATCH}); {n_users} users/cell, "
          f"--processes {processes}.  LL_TOL={LL_TOL:g} SPEED_TOL={SPEED_TOL:g}", flush=True)
    print(f"[grid] measuring gold first ...", flush=True)
    gold = run_cell(GOLD_EPOCH, GOLD_BATCH, n_users, processes)
    if gold["by_user"] is None:
        sys.exit("[grid] ABORT: gold cell failed — no reference, no decision.")
    g_ll, g_s = gold["by_user"], gold["seconds"]
    print(f"[grid] GOLD: by_user={g_ll:.6f}  train={g_s:.1f}s  "
          f"(the 19 candidates are judged against this)\n", flush=True)

    # Persist after EVERY cell (the 3k sweep runs for days — a crash must not lose
    # finished cells; --plot and a restart can read the partial file) AND refresh the
    # Speed-vs-LogLoss plot so the frontier fills in live as cells land. The plot is
    # best-effort: a render failure must never kill a multi-day sweep.
    def checkpoint(cells: list[dict]) -> None:
        GRID_JSON.write_text(json.dumps({
            "gold": {"epoch": GOLD_EPOCH, "batch": GOLD_BATCH, "by_user": g_ll, "seconds": g_s},
            "cells": cells, "partial": True,
            "n_users": n_users, "ll_tol": LL_TOL, "speed_tol": SPEED_TOL,
        }, indent=2), encoding="utf-8")
        try:
            plot_grid()
        except Exception as e:  # noqa: BLE001
            print(f"[grid] plot refresh skipped: {e}", flush=True)

    cells = [gold]
    checkpoint(cells)
    for epoch in GRID_EPOCHS:
        for batch in GRID_BATCHES:
            if (epoch, batch) == (GOLD_EPOCH, GOLD_BATCH):
                continue
            c = run_cell(epoch, batch, n_users, processes)
            cells.append(c)
            checkpoint(cells)
            if c["by_user"] is not None:
                dl = c["by_user"] - g_ll
                ds = 100 * (c["seconds"] - g_s) / g_s
                if dominates(c, g_ll, g_s):
                    tag = "DOMINATES gold"
                elif c["by_user"] > g_ll + LL_TOL and c["seconds"] > g_s * (1 + SPEED_TOL):
                    tag = "worse on both"
                else:
                    tag = "mixed (trade-off)"
                print(f"        vs gold: d_loss={dl:+.6f}  d_time={ds:+.1f}%  -> {tag}", flush=True)

    ok = [c for c in cells if c["by_user"] is not None]
    dominators = [c for c in ok if dominates(c, g_ll, g_s)]
    not_slower = [c for c in ok if c["seconds"] <= g_s * (1 + SPEED_TOL)]
    winner = min(not_slower, key=lambda c: (c["by_user"], c["seconds"]))
    improved = (winner["epoch"], winner["batch"]) != (GOLD_EPOCH, GOLD_BATCH) and dominates(winner, g_ll, g_s)

    print(f"\n[grid] gold ({GOLD_EPOCH},{GOLD_BATCH}): by_user={g_ll:.6f}  train={g_s:.1f}s", flush=True)
    if dominators:
        print(f"[grid] {len(dominators)} cell(s) Pareto-dominate gold:", flush=True)
        for c in sorted(dominators, key=lambda c: (c["by_user"], c["seconds"])):
            print(f"        epoch {c['epoch']:>2} batch {c['batch']:>4}: by_user={c['by_user']:.6f} "
                  f"({c['by_user'] - g_ll:+.6f})  train={c['seconds']:.1f}s "
                  f"({100 * (c['seconds'] - g_s) / g_s:+.1f}%)", flush=True)
    else:
        print("[grid] no cell Pareto-dominates gold — operating point stays at "
              f"({GOLD_EPOCH},{GOLD_BATCH}).", flush=True)
    print(f"[grid] WINNER: epoch={winner['epoch']} batch={winner['batch']}  "
          f"{'[Pareto win over gold]' if improved else '[= gold standard]'}", flush=True)

    GRID_JSON.write_text(json.dumps({
        "gold": {"epoch": GOLD_EPOCH, "batch": GOLD_BATCH, "by_user": g_ll, "seconds": g_s},
        "winner": winner, "improved_over_gold": improved,
        "dominators": dominators, "cells": cells,
        "n_users": n_users, "ll_tol": LL_TOL, "speed_tol": SPEED_TOL,
    }, indent=2), encoding="utf-8")
    print(f"\n[grid] wrote {GRID_JSON.relative_to(REPO)}", flush=True)
    plot_grid()
    if improved:
        print("[grid] NEXT: the gold MOVED -> re-tune LR/betas/L2/recency at the new "
              "operating point (fine_hp_followup), then commit the re-anchor.", flush=True)
    else:
        print("[grid] NEXT: gold unchanged -> LR/betas/L2/recency need NO re-tuning.", flush=True)


# ── time-noise: calibrate SPEED_TOL ──────────────────────────────────────────
def time_noise(n: int, n_users: int, processes: int) -> None:
    """Measure the noise floor on the SPEED axis: run the gold cell (8, 256)
    n+1 times — 1 warm-up (discarded) + n measured — and report the train-seconds
    spread. log loss is (near-)deterministic, so only time carries noise; this
    spread is what SPEED_TOL must cover so time jitter can't fake a speed win."""
    print(f"[noise] gold cell (epoch {GOLD_EPOCH}, batch {GOLD_BATCH}); "
          f"{n_users} users; 1 warm-up + {n} measured runs.", flush=True)
    secs: list[float] = []
    for i in range(n + 1):
        c = run_cell(GOLD_EPOCH, GOLD_BATCH, n_users, processes)
        label = "warm-up (discarded)" if i == 0 else f"run {i}/{n}"
        print(f"[noise] {label:<20} train={c['seconds']:.2f}s  by_user={c['by_user']:.7f}", flush=True)
        if i > 0:
            secs.append(c["seconds"])
    mean = statistics.mean(secs)
    sd = statistics.pstdev(secs)
    lo, hi = min(secs), max(secs)
    rng_frac = (hi - lo) / mean if mean else 0.0
    suggested = max(0.02, round(1.5 * rng_frac, 3))
    print(f"\n[noise] train_seconds over {n} runs: mean={mean:.2f}s std={sd:.3f}s "
          f"min={lo:.2f}s max={hi:.2f}s", flush=True)
    print(f"[noise] CV={100 * sd / mean:.2f}%  range/mean={100 * rng_frac:.2f}%", flush=True)
    print(f"[noise] current SPEED_TOL={SPEED_TOL} ({100 * SPEED_TOL:.0f}%); suggested "
          f">= {suggested} (~1.5x observed range/mean, floor 2%).", flush=True)
    (RESULT / "hp_time_noise.json").write_text(json.dumps({
        "epoch": GOLD_EPOCH, "batch": GOLD_BATCH, "n_users": n_users, "n": n,
        "train_seconds": secs, "mean": mean, "std": sd, "min": lo, "max": hi,
        "range_frac": rng_frac, "current_speed_tol": SPEED_TOL, "suggested_speed_tol": suggested,
    }, indent=2), encoding="utf-8")


# ── the plot (Speed vs Log loss, with the Pareto frontier) ───────────────────
def plot_grid() -> None:
    """Speed (throughput, items/s; up = faster) vs Log loss (right = worse) for
    every grid cell. The Pareto frontier is drawn in green and connected, each
    frontier cell labelled (n_epoch, batch_size); dominated cells are grey; the
    gold standard is ringed. Reads result/hp_grid.json -> result/hp_grid_plot.png."""
    if not GRID_JSON.exists():
        sys.exit(f"[plot] {GRID_JSON} not found — run --epoch-batch-grid first.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(GRID_JSON.read_text(encoding="utf-8"))
    cells = [c for c in data["cells"] if c.get("by_user") is not None]
    gold = data["gold"]
    front = pareto_front(cells)
    front_keys = {(c["epoch"], c["batch"]) for c in front}

    fig, ax = plt.subplots(figsize=(7, 6))
    # dominated cells (grey)
    for c in cells:
        if (c["epoch"], c["batch"]) not in front_keys:
            ax.scatter(c["by_user"], c["throughput"], c="grey", s=40, zorder=2)
    # Pareto frontier (green line + points + labels)
    fx = [c["by_user"] for c in front]
    fy = [c["throughput"] for c in front]
    ax.plot(fx, fy, "-", color="#2ca02c", lw=2.5, zorder=3)
    ax.scatter(fx, fy, c="#2ca02c", s=55, zorder=4)
    # Labels: point inward near the right/top edges so they never clip the border.
    xr = (max(fx) - min(fx)) or 1.0
    yr = (max(fy) - min(fy)) or 1.0
    for c in front:
        right = (c["by_user"] - min(fx)) / xr > 0.62
        top = (c["throughput"] - min(fy)) / yr > 0.88
        dx, ha = (-7, "right") if right else (7, "left")
        dy, va = (-9, "top") if top else (7, "bottom")
        ax.annotate(f"({c['epoch']}, {c['batch']})",
                    (c["by_user"], c["throughput"]),
                    textcoords="offset points", xytext=(dx, dy), ha=ha, va=va,
                    color="#2ca02c", fontsize=9, fontweight="bold")
    # ring the gold standard
    g = next((c for c in cells if c["epoch"] == gold["epoch"] and c["batch"] == gold["batch"]), None)
    if g is not None:
        ax.scatter(g["by_user"], g["throughput"], facecolors="none",
                   edgecolors="black", s=160, lw=1.8, zorder=5,
                   label=f"gold ({gold['epoch']}, {gold['batch']})")
        ax.legend(loc="upper left", fontsize=9)

    ax.set_xlabel("Log loss  (cross-val by-user; lower = better)", fontsize=12)
    ax.set_ylabel("Speed  (items / s; higher = faster)", fontsize=12)
    ax.set_title("n_epoch x batch_size Pareto grid", fontsize=13)
    ax.grid(True, alpha=0.25)
    ax.margins(0.10)  # data headroom so inward-pointing labels stay clear of the spines
    fig.tight_layout()
    fig.savefig(GRID_PLOT, dpi=130)
    print(f"[plot] wrote {GRID_PLOT.relative_to(REPO)}", flush=True)


def fine_hp_followup() -> None:
    """CONDITIONAL second phase — only if the grid moves the gold. Coordinate
    descent over the fine training HPs (LR / Adam betas / L2 strength / recency
    C0+EXP) at the NEW operating point, accepting a step iff it lowers cross-val
    by-user log loss by >= the threshold. Unlike the grid (env-driven, no rebuild),
    these HPs are Rust consts in training.rs, so each trial edits the const literal
    and rebuilds (cargo --release, ~35 s). Not built until the grid actually moves
    the gold — per the user's rule, an unchanged gold needs no fine re-tune."""
    raise NotImplementedError(
        "fine_hp_followup is the conditional re-tune; build it only if "
        "--epoch-batch-grid reports the gold MOVED."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epoch-batch-grid", action="store_true",
                    help="run the 20-cell n_epoch x batch_size Pareto grid")
    ap.add_argument("--time-noise", type=int, nargs="?", const=3, default=None, metavar="N",
                    help="measure the train-seconds noise floor at the gold cell (default N=3, "
                         "+1 warm-up) and suggest a SPEED_TOL")
    ap.add_argument("--plot", action="store_true",
                    help="redraw result/hp_grid_plot.png from result/hp_grid.json and exit")
    ap.add_argument("--max-user-id", type=int, default=100,
                    help="users per cell (controls sweep duration; default 100). Each cell runs "
                         "benchmark.py on these users, so keep it modest for the 20-cell sweep.")
    ap.add_argument("--processes", type=int, default=10, help="harness worker processes (default 10)")
    args = ap.parse_args()

    if args.plot:
        plot_grid()
        return
    if args.time_noise is not None:
        time_noise(args.time_noise, args.max_user_id, args.processes)
        return
    if args.epoch_batch_grid:
        epoch_batch_grid(args.max_user_id, args.processes)
        return
    ap.error("nothing to do: pass --epoch-batch-grid, --time-noise, or --plot")


if __name__ == "__main__":
    main()
