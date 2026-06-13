#!/usr/bin/env python3
"""Stage 3: local 3k refinement of ONE chosen cell's fine HPs (run AFTER the grid + gold pick).

The per-cell grid (tuner_engine.py --grid) tunes each cell's 6 fine HPs on a 500-user cached
subset (Stage 1) and confirms loss/speed on full 3k (Stage 2). Those HPs are subset-tuned, not
3k-tuned, because the 3k item cache (~230 GB) does not fit 64 GB RAM. Once Andrew picks the gold
operating point, this squeezes that one cell's HPs a bit more on the FULL 3k.

Adaptive coordinate descent on 3k is impossible (can't cache -> each step would rebuild all 3k
items). So this is a PATTERN SEARCH: each round evaluates the current center plus every active
HP's up/down candidate TOGETHER in a SINGLE loop-inversion pass over 3k (te.confirm_on_full builds
each user once, evals all candidate configs against it, discards — RAM-safe). It then takes the
best improving move per HP, re-centers, and repeats. The reported HPs are always a config that was
DIRECTLY measured on 3k (the global argmin over everything evaluated), so the result can only beat
or tie the starting (subset-tuned) HPs — never regress.

This does NOT touch tuner_engine.py (kept byte-frozen while the grid runs); it imports its helpers.

    python stage3_refine.py                 # refine hp_grid.json's auto-winner on 3k
    python stage3_refine.py --cell 9 256     # refine a specific (epoch, batch) cell
    python stage3_refine.py --users 3000 --rounds 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import pyarrow.parquet as pq  # noqa: E402
import tuner_engine as te  # noqa: E402  (frozen during the grid; reused here)
import hp_tune as ht  # noqa: E402

STAGE3_JSON = te.RESULT / "hp_grid_stage3.json"
IMPROVE_EPS = 2e-6  # 3k by_user is deterministic within a run-context; floor at the 6dp grain


def _load_cell(epoch: int | None, batch: int | None) -> dict:
    """Pick the cell to refine from result/hp_grid.json: the explicit (epoch, batch) if given,
    else the grid's computed winner. Returns the cell dict (carrying its 6 tuned HPs)."""
    data = json.loads(ht.GRID_JSON.read_text(encoding="utf-8"))
    cells = data["cells"]
    if epoch is not None and batch is not None:
        for c in cells:
            if c["epoch"] == epoch and c["batch"] == batch:
                return c
        sys.exit(f"[stage3] cell ({epoch},{batch}) not found in {ht.GRID_JSON.name}")
    w = data.get("winner")
    if w is None:
        sys.exit("[stage3] no winner in hp_grid.json and no --cell given")
    return w


def refine(epoch: int, batch: int, start_hps: dict, n_users: int, P: int, rounds: int) -> None:
    all_ids = sorted(u.as_py() for u in pq.ParquetDataset(te.config.data_path / "revlogs").partitioning.dictionaries[0])
    ids = all_ids[:n_users]
    op = {"epoch": epoch, "batch": batch}
    HP_KEYS = ("lr", "beta1", "beta2", "c0", "exp", "l2")
    center = {k: float(start_hps[k]) for k in HP_KEYS}
    active = {n: True for n, *_ in te.HP_SPECS}

    print(f"[stage3] refining cell (epoch {epoch}, batch {batch}) on {n_users} users, {P} workers.\n"
          f"[stage3] start HPs: " + " ".join(f"{k}={center[k]:g}" for k in HP_KEYS), flush=True)

    measured: dict[tuple, dict] = {}  # frozenset(hps.items()) -> {hps, by_user, seconds}

    def eval_configs(cfgs: list[dict]) -> list[dict]:
        """Run a list of HP configs over all 3k users in ONE loop-inversion pass; cache results."""
        configs = [(op, c) for c in cfgs]
        res, _items = te.confirm_on_full(configs, ids, P)
        out = []
        for c, r in zip(cfgs, res):
            rec = {"hps": dict(c), "by_user": r["by_user"], "seconds": r["seconds"]}
            measured[frozenset(c.items())] = rec
            out.append(rec)
        return out

    t0 = time.time()
    base = eval_configs([dict(center)])[0]
    best = base
    print(f"[stage3] baseline 3k by_user={base['by_user']:.6f}  train={base['seconds']:.1f}s", flush=True)

    for r in range(rounds):
        cfgs = [dict(center)]  # measure the current center again this round (cheap; in the same pass)
        tags = [("__center__", None)]
        for name, kind, step, lo, hi in te.HP_SPECS:
            if not active[name]:
                continue
            for v in te._candidates(kind, step, lo, hi, center[name]):
                t = dict(center)
                t[name] = v
                cfgs.append(t)
                tags.append((name, v))
        if len(cfgs) == 1:
            print(f"[stage3] round {r}: all HPs frozen — converged.", flush=True)
            break
        recs = eval_configs(cfgs)
        center_ll = recs[0]["by_user"]
        # best candidate per HP this round
        best_by_hp: dict[str, tuple] = {}
        for (name, v), rec in zip(tags, recs):
            if name == "__center__":
                continue
            ll = rec["by_user"]
            print(f"      r{r} {name}: {center[name]:g}->{v:g}  by_user={ll:.6f} "
                  f"(d={ll-center_ll:+.6f})", flush=True)
            if name not in best_by_hp or ll < best_by_hp[name][0]:
                best_by_hp[name] = (ll, v)
        improved = False
        for name, (ll, v) in best_by_hp.items():
            if ll < center_ll - IMPROVE_EPS:
                center[name] = v
                improved = True
            else:
                active[name] = False  # this HP is locally optimal at this granularity
        # track the global best directly-measured config
        for rec in recs:
            if rec["by_user"] < best["by_user"] - IMPROVE_EPS:
                best = rec
        print(f"      round {r}: center by_user={center_ll:.6f}; "
              f"{'moved' if improved else 'no improving move — stop'}", flush=True)
        if not improved:
            break

    # If the final combined center wasn't directly measured this session, measure it now so the
    # returned HPs are always a real 3k measurement.
    ck = frozenset({k: center[k] for k in HP_KEYS}.items())
    if ck not in measured:
        rec = eval_configs([dict(center)])[0]
        if rec["by_user"] < best["by_user"] - IMPROVE_EPS:
            best = rec

    elapsed = (time.time() - t0) / 60.0
    delta = best["by_user"] - base["by_user"]
    print(f"\n[stage3] DONE in {elapsed:.0f} min. baseline {base['by_user']:.6f} -> "
          f"best {best['by_user']:.6f} ({delta:+.6f})", flush=True)
    print(f"[stage3] refined HPs: " + " ".join(f"{k}={best['hps'][k]:g}" for k in HP_KEYS), flush=True)
    STAGE3_JSON.write_text(json.dumps({
        "epoch": epoch, "batch": batch, "n_users": n_users,
        "start_hps": start_hps,
        "baseline": base, "best": best, "improvement": -delta,
        "all_measured": list(measured.values()),
    }, indent=2), encoding="utf-8")
    print(f"[stage3] wrote {STAGE3_JSON.relative_to(REPO)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", type=int, nargs=2, metavar=("EPOCH", "BATCH"),
                    help="cell to refine (default: hp_grid.json's winner)")
    ap.add_argument("--users", type=int, default=3000, help="users for the 3k confirm/refine (default 3000)")
    ap.add_argument("--processes", type=int, default=15, help="worker processes (default 15)")
    ap.add_argument("--rounds", type=int, default=3, help="max pattern-search rounds (default 3)")
    args = ap.parse_args()
    epoch, batch = (args.cell if args.cell else (None, None))
    cell = _load_cell(epoch, batch)
    start = {k: cell[k] for k in ("lr", "beta1", "beta2", "c0", "exp", "l2")}
    refine(cell["epoch"], cell["batch"], start, args.users, args.processes, args.rounds)


if __name__ == "__main__":
    main()
