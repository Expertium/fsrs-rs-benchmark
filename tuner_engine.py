#!/usr/bin/env python3
"""In-memory evaluation engine for the per-cell HP tuner.

The aborted grid re-ran benchmark.py (a fresh subprocess) for every cell, and ~99% of each
cell's wall was Python that does NOT depend on (epoch, batch) or the fine HPs: loading 3k users'
parquet, building the expanding-window FSRSItems (O(N^2) Python->Rust marshaling), and the
forgetting-curve scoring. The fine-HP coordinate descent needs HUNDREDS of trials, so re-paying
that per trial is hopeless.

This builds each user's per-fold (train items, card_ids, test_set) ONCE and holds it in RAM, then
each trial is just: set the HP env vars, run compute_parameters() per fold (the Rust region we
time), predict the test folds, and sklearn-log_loss them. The fold splits and items do NOT depend
on the HPs, so they are built once and reused across every trial of every cell.

`--probe N` builds N users single-process, then measures one trial's wall + the cache RAM, and
extrapolates both to 3000 users / 15 workers — used to size the full sweep before launching it.
Profiling/tuning tool: never writes a result/ champion record (constraint 10).
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
RESULT = REPO / "result"
sys.path.insert(0, str(REPO))

# benchmark.py builds the global Config from sys.argv at import; mirror the grid's invocation.
_SAVED_ARGV = sys.argv
sys.argv = ["benchmark.py", "--algo", "FSRS-rs", "--short", "--secs", "--recency",
            "--processes", "1", "--max-user-id", "100000"]
import benchmark as bm  # noqa: E402
sys.argv = _SAVED_ARGV
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402
from fsrs_rs_python import FSRS  # noqa: E402
from data_loader import UserDataLoader  # noqa: E402

config = bm.config
HP_ENV_KEYS = ("FSRS_N_EPOCHS", "FSRS_BATCH_SIZE", "FSRS_LR",
               "FSRS_BETA1", "FSRS_BETA2", "FSRS_RECENCY_C0", "FSRS_RECENCY_EXP")


def build_user_folds(user_id: int, loader: UserDataLoader):
    """Replicate benchmark.process()'s fold construction ONCE: return a list of
    (items, card_ids, test_set) per time-series split, plus the item count. None if too small.

    Mirrors benchmark.py:244-269 exactly so the trained params / scored loss match benchmark.py."""
    try:
        dataset = loader.load_user_data(user_id)
    except Exception:  # noqa: BLE001 — inadequate data, etc.
        return None
    folds = []
    n_items = 0
    for split_i, (tr_idx, te_idx) in enumerate(
        TimeSeriesSplit(n_splits=config.n_splits).split(dataset)
    ):
        if config.train_equals_test:
            train_set = dataset.copy()
            test_set = dataset[
                dataset["review_th"] >= dataset.iloc[te_idx]["review_th"].min()
            ].copy()
        else:
            train_set = dataset.iloc[tr_idx]
            test_set = dataset.iloc[te_idx]
            if config.equalize_test_with_non_secs:
                train_set = dataset[dataset[f"{split_i}_train"]]
                test_set = dataset[dataset[f"{split_i}_test"]]
        if config.no_test_same_day:
            test_set = test_set[test_set["elapsed_days"] > 0].copy()
        if config.no_train_same_day:
            train_set = train_set[train_set["elapsed_days"] > 0].copy()
        if train_set.empty or test_set.empty:
            continue
        items, card_ids = bm.convert_to_items(train_set)
        n_items += len(items)
        # Pre-build the test rows' history FSRSItems ONCE (weight-independent) so per-trial
        # scoring skips the per-row marshaling and only reruns memory_state_batch + the curve.
        from fsrs_rs_python import FSRSItem
        test_hist = [FSRSItem(reviews=bm.build_reviews(row)) for _, row in test_set.iterrows()]
        folds.append((items, card_ids, test_set, test_hist))
        if config.train_equals_test:
            break
    if not folds:
        return None
    return {"folds": folds, "n_items": n_items}


def set_hp_env(op: dict, hps: dict) -> None:
    """Set the operating-point + fine-HP env vars the Rust training reads live per call."""
    for k in HP_ENV_KEYS:
        os.environ.pop(k, None)
    os.environ["FSRS_N_EPOCHS"] = str(op["epoch"])
    os.environ["FSRS_BATCH_SIZE"] = str(op["batch"])
    os.environ["FSRS_LR"] = repr(float(hps["lr"]))
    os.environ["FSRS_BETA1"] = repr(float(hps["beta1"]))
    os.environ["FSRS_BETA2"] = repr(float(hps["beta2"]))
    os.environ["FSRS_RECENCY_C0"] = repr(float(hps["c0"]))
    os.environ["FSRS_RECENCY_EXP"] = repr(float(hps["exp"]))


def eval_user(cache: dict, backend: FSRS) -> tuple[float, float]:
    """Train + score one cached user under the CURRENT env. Returns (logloss, rust_seconds).
    Matches benchmark.py: per-fold compute_parameters() -> predict() -> concat p/y -> log_loss."""
    p_all: list[float] = []
    y_all: list[float] = []
    secs = 0.0
    for items, card_ids, test_set, test_hist in cache["folds"]:
        try:
            params, s = backend.compute_parameters(items, card_ids)
            secs += s
        except Exception:  # noqa: BLE001 — inadequate fold -> defaults (secs stays 0), as benchmark
            params = bm.default_parameters()
        pp, yy, _ = bm.predict(test_set, params, history_items=test_hist)
        p_all.extend(pp)
        y_all.extend(yy)
    ll = round(log_loss(y_true=y_all, y_pred=p_all, labels=[0, 1]), 6)
    return ll, secs


def probe(n_users: int) -> None:
    """Build N users' cache single-process; measure one trial's wall + cache RAM; extrapolate."""
    import statistics
    try:
        import psutil  # type: ignore
        proc = psutil.Process()
        rss0 = proc.memory_info().rss
    except Exception:  # noqa: BLE001
        proc = None
        rss0 = 0

    loader = UserDataLoader(config)
    backend = FSRS(parameters=[])
    ids = bm.config  # placeholder
    import pyarrow.parquet as pq
    all_ids = sorted(u.as_py() for u in pq.ParquetDataset(config.data_path / "revlogs").partitioning.dictionaries[0])
    ids = all_ids[:n_users]

    t0 = time.perf_counter()
    cache = {}
    for u in ids:
        c = build_user_folds(u, loader)
        if c is not None:
            cache[u] = c
    build_s = time.perf_counter() - t0
    rss_cache = (proc.memory_info().rss - rss0) if proc else 0
    n_ok = len(cache)
    n_items = sum(c["n_items"] for c in cache.values())

    gold_op = {"epoch": 9, "batch": 256}
    gold_hps = {"lr": 0.0188, "beta1": 0.55, "beta2": 0.9913, "c0": 0.0667, "exp": 11.25}
    set_hp_env(gold_op, gold_hps)
    t0 = time.perf_counter()
    lls, total_secs = [], 0.0
    for u in cache:
        ll, s = eval_user(cache[u], backend)
        lls.append(ll); total_secs += s
    trial_s = time.perf_counter() - t0
    for k in HP_ENV_KEYS:
        os.environ.pop(k, None)

    by_user = statistics.mean(lls)
    print(f"\n=== PROBE: {n_ok}/{n_users} users built (single-process) ===")
    print(f"  cache build : {build_s:8.1f}s   ({1000*build_s/n_ok:.1f} ms/user)   one-time per worker")
    if proc:
        print(f"  cache RAM   : {rss_cache/2**30:8.2f} GB  ({rss_cache/n_ok/2**20:.2f} MB/user)  {n_items:,} items")
    print(f"  ONE TRIAL   : {trial_s:8.1f}s   ({1000*trial_s/n_ok:.1f} ms/user)   rust={total_secs:.1f}s")
    print(f"  by_user(gold): {by_user:.6f}")

    # Extrapolate to 3000 users across 15 workers (disjoint slices -> ~1x total RAM, parallel).
    P, U = 15, 3000
    per_worker = U / P
    trial_wall_3k = (trial_s / n_ok) * per_worker          # each worker does U/P users serially
    build_wall_3k = (build_s / n_ok) * per_worker
    ram_3k = (rss_cache / n_ok) * U if proc else 0          # total across all workers (disjoint)
    print(f"\n=== EXTRAPOLATION to {U} users / {P} workers ===")
    print(f"  per-trial wall : ~{trial_wall_3k/60:.1f} min   (one HP config, all {U} users)")
    print(f"  one-time build : ~{build_wall_3k/60:.1f} min   (worker startup, paid once)")
    if proc:
        print(f"  total cache RAM: ~{ram_3k/2**30:.1f} GB  (across all {P} workers; 64 GB box)")
    for ntrials in (300, 500):
        days = (build_wall_3k + ntrials * trial_wall_3k) / 86400
        print(f"  {ntrials} trials -> ~{days:.1f} days")


# ── persistent worker pool: cache a disjoint user slice per worker, eval configs on demand ─────
import multiprocessing as mp  # noqa: E402
import statistics  # noqa: E402


def _pool_worker(slice_ids, in_q, out_q):
    """Cache this worker's user slice ONCE, then loop: receive (op, hps) -> eval all my users ->
    send back {user: (logloss, rust_secs)}. The FSRSItem caches stay in the worker; only float
    results cross the queue."""
    loader = UserDataLoader(config)
    backend = FSRS(parameters=[])
    cache = {}
    for u in slice_ids:
        c = build_user_folds(u, loader)
        if c is not None:
            cache[u] = c
    out_q.put(("ready", len(cache), sum(c["n_items"] for c in cache.values())))
    while True:
        msg = in_q.get()
        if msg is None:
            return
        op, hps = msg
        try:
            set_hp_env(op, hps)
            out_q.put({u: eval_user(cache[u], backend) for u in cache})
        except Exception:  # noqa: BLE001 — never hang the pool: report the error so main aborts cleanly
            import traceback
            out_q.put(("__ERROR__", traceback.format_exc()))


class WorkerPool:
    """P persistent workers, each holding a disjoint slice of `ids` cached in RAM. evaluate(op,hps)
    broadcasts one HP config and returns (by_user mean, total rust secs, n_users). Used for the
    adaptive coordinate descent, where every step needs the aggregate loss over all subset users."""

    def __init__(self, ids, P):
        ctx = mp.get_context("spawn")
        slices = [ids[i::P] for i in range(P)]
        self.in_qs = [ctx.Queue() for _ in range(P)]
        self.out_qs = [ctx.Queue() for _ in range(P)]
        self.procs = []
        for i in range(P):
            p = ctx.Process(target=_pool_worker, args=(slices[i], self.in_qs[i], self.out_qs[i]),
                            daemon=True)
            p.start()
            self.procs.append(p)
        self.n_users = self.n_items = 0
        for q in self.out_qs:
            _, nu, ni = q.get()
            self.n_users += nu
            self.n_items += ni

    def evaluate(self, op: dict, hps: dict):
        for q in self.in_qs:
            q.put((op, hps))
        merged = {}
        for q in self.out_qs:
            r = q.get()
            if isinstance(r, tuple) and r and r[0] == "__ERROR__":
                raise RuntimeError(f"worker eval failed:\n{r[1]}")
            merged.update(r)
        lls = [ll for ll, _ in merged.values()]
        secs = sum(s for _, s in merged.values())
        return statistics.mean(lls), secs, len(lls)

    def close(self):
        for q in self.in_qs:
            q.put(None)
        for p in self.procs:
            p.join(timeout=10)


# ── per-cell coordinate descent over the 5 fine HPs (ported from the CUDA tuner) ───────────────
# (name, kind, step, lo, hi). "mul" perturbs v by *step / /step; "beta" perturbs on the (1-v)
# scale (so betas near 1 move sensibly). Bounds keep candidates sane.
HP_SPECS = [
    ("lr",    "mul",  1.5, 1e-3, 0.3),
    ("beta1", "beta", 1.5, 0.4,  0.98),
    ("beta2", "beta", 1.5, 0.4,  0.9995),
    ("c0",    "mul",  1.5, 0.005, 0.5),
    ("exp",   "mul",  1.5, 1.0,  20.0),
]


def _candidates(kind, step, lo, hi, v):
    raw = [v * step, v / step] if kind == "mul" else [1 - (1 - v) * step, 1 - (1 - v) / step]
    out = []
    for c in raw:
        c = round(float(c), 4)
        if lo <= c <= hi and abs(c - v) > 1e-9:
            out.append(c)
    return out


def coordinate_descent(pool: WorkerPool, op: dict, start_hps: dict, improve_eps: float,
                       rounds=4, max_evals=40):
    """Greedy coordinate descent minimizing by_user loss. Each HP tries up/down; keep the best
    improving move (must beat the incumbent by > improve_eps, the noise-robust threshold); freeze
    HPs that don't help; re-probe improving ones next round. The Rust training is NOT bit-exact
    run-to-run, so by_user carries ~noise; improve_eps must clear it. Returns
    (best_hps, best_loss, best_secs, n_evals, trials)."""
    best = dict(start_hps)
    bl, bs, _ = pool.evaluate(op, best)
    evals = 1
    trials = [{"hps": dict(best), "loss": bl, "secs": bs}]
    active = {n: True for n, *_ in HP_SPECS}
    print(f"      baseline by_user={bl:.6f}  (lr={best['lr']:g} b1={best['beta1']:g} "
          f"b2={best['beta2']:g} c0={best['c0']:g} exp={best['exp']:g})", flush=True)
    for r in range(rounds):
        improved = False
        for name, kind, step, lo, hi in HP_SPECS:
            if not active[name] or evals >= max_evals:
                continue
            results = []
            for cand in _candidates(kind, step, lo, hi, best[name]):
                if evals >= max_evals:
                    break
                trial = dict(best)
                trial[name] = cand
                ll, secs, _ = pool.evaluate(op, trial)
                evals += 1
                trials.append({"hps": dict(trial), "loss": ll, "secs": secs})
                tag = "  <-- best" if ll < bl - improve_eps else ""
                print(f"      r{r} {name}: {best[name]:g}->{cand:g}  by_user={ll:.6f} "
                      f"(d={ll-bl:+.6f}){tag}", flush=True)
                results.append((ll, cand, secs))
            if not results:
                active[name] = False
                continue
            results.sort(key=lambda x: x[0])
            cll, ccand, csecs = results[0]
            if cll < bl - improve_eps:
                bl, bs, best[name] = cll, csecs, ccand
                improved = True
            else:
                active[name] = False
        if not improved or evals >= max_evals:
            break
    return best, bl, bs, evals, trials


def validate(n_users: int) -> None:
    """Foundational check: the engine's per-user LogLoss must EQUAL benchmark.py's on the same
    users at the same HPs. Runs the engine at gold over the first N users, then runs benchmark.py
    as a subprocess with the SAME gold env over the same N users, and diffs per-user LogLoss."""
    import subprocess
    import pyarrow.parquet as pq
    gold_op = {"epoch": 9, "batch": 256}
    gold_hps = {"lr": 0.0188, "beta1": 0.55, "beta2": 0.9913, "c0": 0.0667, "exp": 11.25}
    all_ids = sorted(u.as_py() for u in pq.ParquetDataset(config.data_path / "revlogs").partitioning.dictionaries[0])
    ids = all_ids[:n_users]

    loader = UserDataLoader(config)
    backend = FSRS(parameters=[])
    set_hp_env(gold_op, gold_hps)
    eng = {}
    for u in ids:
        c = build_user_folds(u, loader)
        if c is not None:
            eng[u] = eval_user(c, backend)[0]
    for k in HP_ENV_KEYS:
        os.environ.pop(k, None)

    # benchmark.py with the SAME gold env over the same first-N users.
    bm_result = REPO / "result" / "FSRS-rs-short-secs-recency.jsonl"
    if bm_result.exists():
        bm_result.unlink()
    env = {**os.environ, "FSRS_N_EPOCHS": "9", "FSRS_BATCH_SIZE": "256",
           "FSRS_LR": repr(0.0188), "FSRS_BETA1": repr(0.55), "FSRS_BETA2": repr(0.9913),
           "FSRS_RECENCY_C0": repr(0.0667), "FSRS_RECENCY_EXP": repr(11.25)}
    cmd = [sys.executable, "benchmark.py", "--algo", "FSRS-rs", "--short", "--secs", "--recency",
           "--processes", "10", "--max-user-id", str(n_users)]
    print(f"[validate] running benchmark.py on {n_users} users at gold env ...", flush=True)
    subprocess.run(cmd, cwd=str(REPO), env=env, check=True, capture_output=True, text=True)
    bm = {int(json.loads(l)["user"]): round(json.loads(l)["metrics"]["LogLoss"], 6)
          for l in bm_result.read_text(encoding="utf-8").splitlines() if l.strip()}

    import statistics
    common = sorted(set(eng) & set(bm))
    mism = [(u, eng[u], bm[u]) for u in common if abs(eng[u] - bm[u]) > 1e-6]
    print(f"[validate] {len(common)} users compared; {len(mism)} per-user mismatches (>1e-6)")
    for u, e, b in mism[:10]:
        print(f"    user {u}: engine={e:.6f}  benchmark={b:.6f}  d={e-b:+.6f}")
    if common:
        print(f"[validate] by_user  engine={statistics.mean(eng[u] for u in common):.6f}  "
              f"benchmark={statistics.mean(bm[u] for u in common):.6f}")
    print(f"[validate] {'MATCH — engine replicates benchmark.py' if not mism else 'MISMATCH — fix before tuning'}")


# ── stage 2: confirm the per-cell winning HPs on the FULL user set (loop-inversion, RAM-safe) ──
def _confirm_worker(slice_ids, configs, out_q):
    """Stream this worker's users: build each user ONCE, eval ALL `configs` (the 20 cell winners)
    against it, discard. RAM = one user at a time -> safe at full 3k. Returns per-config
    accumulated (per-user logloss list, summed rust secs) + the config-independent item count."""
    loader = UserDataLoader(config)
    backend = FSRS(parameters=[])
    acc = [{"lls": [], "secs": 0.0} for _ in configs]
    n_items = 0
    for u in slice_ids:
        try:
            c = build_user_folds(u, loader)
            if c is None:
                continue
            # Eval all configs for this user into a scratch buffer; commit only if ALL succeed, so
            # every config's by_user is over the SAME user set (comparability), and one bad user
            # is skipped uniformly rather than hanging the run.
            scratch = []
            for op, hps in configs:
                set_hp_env(op, hps)
                scratch.append(eval_user(c, backend))
        except Exception:  # noqa: BLE001
            import traceback
            print(f"[confirm] user {u} skipped:\n{traceback.format_exc()}", flush=True)
            continue
        n_items += c["n_items"]
        for ci, (ll, secs) in enumerate(scratch):
            acc[ci]["lls"].append(ll)
            acc[ci]["secs"] += secs
    out_q.put((acc, n_items))


def confirm_on_full(configs, ids, P):
    """Run the `configs` over `ids` via P streaming workers; return per-config
    (by_user, seconds, n_users) + total item count."""
    ctx = mp.get_context("spawn")
    slices = [ids[i::P] for i in range(P)]
    out_qs = [ctx.Queue() for _ in range(P)]
    procs = [ctx.Process(target=_confirm_worker, args=(slices[i], configs, out_qs[i]), daemon=True)
             for i in range(P)]
    for p in procs:
        p.start()
    agg = [{"lls": [], "secs": 0.0} for _ in configs]
    total_items = 0
    for q in out_qs:
        acc, ni = q.get()
        total_items += ni
        for ci in range(len(configs)):
            agg[ci]["lls"].extend(acc[ci]["lls"])
            agg[ci]["secs"] += acc[ci]["secs"]
    for p in procs:
        p.join(timeout=10)
    out = []
    for a in agg:
        out.append({"by_user": statistics.mean(a["lls"]), "seconds": a["secs"], "n_users": len(a["lls"])})
    return out, total_items


# ── the two-stage per-cell grid ────────────────────────────────────────────────────────────────
def run_two_stage_grid(tune_users: int, confirm_users: int, P: int, max_cells: int = 0) -> None:
    """STAGE 1: per-cell coordinate descent over (lr, b1, b2, c0, exp) on `tune_users` cached users.
    STAGE 2: re-measure each cell's winning HPs on `confirm_users` (loop-inversion). Writes the
    per-cell best HPs + 3k loss/speed to result/hp_grid.json and renders the plot via hp_tune."""
    import pyarrow.parquet as pq
    import hp_tune as ht

    all_ids = sorted(u.as_py() for u in pq.ParquetDataset(config.data_path / "revlogs").partitioning.dictionaries[0])
    tune_ids = all_ids[:tune_users]
    confirm_ids = all_ids[:confirm_users]
    grid = [(ht.GOLD_EPOCH, ht.GOLD_BATCH)] + [
        (e, b) for e in ht.GRID_EPOCHS for b in ht.GRID_BATCHES if (e, b) != (ht.GOLD_EPOCH, ht.GOLD_BATCH)
    ]
    if max_cells:
        grid = grid[:max_cells]  # smoke-test limiter
    stage1_path = RESULT / "hp_grid_stage1.json"

    print(f"[grid] STAGE 1: per-cell coordinate descent on {tune_users} users, {P} workers. "
          f"5 HPs: lr/beta1/beta2/c0/exp.", flush=True)
    t0 = time.time()
    pool = WorkerPool(tune_ids, P)
    print(f"[grid] pool ready: {pool.n_users} users cached ({pool.n_items:,} items) "
          f"in {time.time()-t0:.0f}s", flush=True)

    # Measure the by_user run-to-run noise (Rust training isn't bit-exact): eval gold 3x, take the
    # spread, and set the coordinate-descent improvement threshold to a safe multiple of it so the
    # search keeps only moves that clearly beat noise. Floor 2e-6 (the 6dp record granularity).
    gold_op = {"epoch": ht.GOLD_EPOCH, "batch": ht.GOLD_BATCH}
    gold_hps = {"lr": ht._scaled_lr(ht.GOLD_BATCH), "beta1": 0.55, "beta2": 0.9913, "c0": 0.0667, "exp": 11.25}
    noise_lls = [pool.evaluate(gold_op, gold_hps)[0] for _ in range(3)]
    noise = max(noise_lls) - min(noise_lls)
    improve_eps = max(2e-6, 3.0 * noise)
    print(f"[grid] by_user noise over 3 gold evals: {noise:.2e} "
          f"(lls={[round(x,6) for x in noise_lls]}) -> improve_eps={improve_eps:.2e}", flush=True)

    cells = []
    for epoch, batch in grid:
        op = {"epoch": epoch, "batch": batch}
        start = {"lr": ht._scaled_lr(batch), "beta1": 0.55, "beta2": 0.9913, "c0": 0.0667, "exp": 11.25}
        print(f"\n[grid] cell (epoch {epoch}, batch {batch}) — start lr={start['lr']:g}", flush=True)
        best, bl, bs, nev, trials = coordinate_descent(pool, op, start, improve_eps)
        print(f"      BEST by_user={bl:.6f} after {nev} evals: lr={best['lr']:g} b1={best['beta1']:g} "
              f"b2={best['beta2']:g} c0={best['c0']:g} exp={best['exp']:g}", flush=True)
        cells.append({"epoch": epoch, "batch": batch, **best,
                      "subset_by_user": bl, "subset_secs": bs, "n_tune_evals": nev, "trials": trials})
        stage1_path.write_text(json.dumps(
            {"tune_users": pool.n_users, "cells": cells, "partial": True}, indent=2), encoding="utf-8")
    pool.close()
    print(f"\n[grid] STAGE 1 done in {(time.time()-t0)/60:.0f} min. Best HPs per cell saved to "
          f"{stage1_path.name}.", flush=True)

    print(f"\n[grid] STAGE 2: confirming {len(cells)} cells' HPs on {confirm_users} users "
          f"(loop-inversion) ...", flush=True)
    t1 = time.time()
    configs = [({"epoch": c["epoch"], "batch": c["batch"]},
                {k: c[k] for k in ("lr", "beta1", "beta2", "c0", "exp")}) for c in cells]
    results, total_items = confirm_on_full(configs, confirm_ids, P)
    for c, r in zip(cells, results):
        c["by_user"] = r["by_user"]
        c["seconds"] = r["seconds"]
        c["items"] = total_items
        c["throughput"] = total_items / r["seconds"] if r["seconds"] else 0.0
    print(f"[grid] STAGE 2 done in {(time.time()-t1)/60:.0f} min.", flush=True)

    gold = next(c for c in cells if c["epoch"] == ht.GOLD_EPOCH and c["batch"] == ht.GOLD_BATCH)
    g_ll, g_s = gold["by_user"], gold["seconds"]
    dominators = [c for c in cells if ht.dominates(c, g_ll, g_s)]
    not_slower = [c for c in cells if c["seconds"] <= g_s * (1 + ht.SPEED_TOL)]
    winner = min(not_slower, key=lambda c: (c["by_user"], c["seconds"]))
    improved = (winner["epoch"], winner["batch"]) != (ht.GOLD_EPOCH, ht.GOLD_BATCH) and ht.dominates(winner, g_ll, g_s)

    ht.GRID_JSON.write_text(json.dumps({
        "gold": {"epoch": ht.GOLD_EPOCH, "batch": ht.GOLD_BATCH, "by_user": g_ll, "seconds": g_s},
        "winner": winner, "improved_over_gold": improved, "dominators": dominators,
        "cells": cells, "n_users": confirm_users, "tune_users": tune_users,
        "ll_tol": ht.LL_TOL, "speed_tol": ht.SPEED_TOL, "partial": False,
        "note": "per-cell 5-HP tune (lr/beta1/beta2/c0/exp); HPs tuned on tune_users, "
                "loss+speed confirmed on n_users",
    }, indent=2), encoding="utf-8")
    print(f"\n[grid] gold ({ht.GOLD_EPOCH},{ht.GOLD_BATCH}): by_user={g_ll:.6f} train={g_s:.1f}s", flush=True)
    for c in sorted(cells, key=lambda c: (c["by_user"], c["seconds"])):
        tag = " DOMINATES" if ht.dominates(c, g_ll, g_s) else ""
        print(f"      ({c['epoch']:>2},{c['batch']:>4}) by_user={c['by_user']:.6f} "
              f"train={c['seconds']:.1f}s lr={c['lr']:g} b1={c['beta1']:g} b2={c['beta2']:g} "
              f"c0={c['c0']:g} exp={c['exp']:g}{tag}", flush=True)
    print(f"\n[grid] WINNER: epoch={winner['epoch']} batch={winner['batch']} "
          f"{'[Pareto win]' if improved else '[= gold]'}", flush=True)
    try:
        ht.plot_grid()
    except Exception as e:  # noqa: BLE001
        print(f"[grid] plot skipped: {e}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", type=int, metavar="N", help="build N users, measure trial + RAM, extrapolate")
    ap.add_argument("--validate", type=int, metavar="N", help="check engine LogLoss == benchmark.py on N users")
    ap.add_argument("--grid", action="store_true", help="run the two-stage per-cell 5-HP grid")
    ap.add_argument("--tune-users", type=int, default=500, help="stage-1 cached subset size (default 500)")
    ap.add_argument("--confirm-users", type=int, default=3000, help="stage-2 confirm size (default 3000)")
    ap.add_argument("--processes", type=int, default=15, help="worker processes (default 15)")
    ap.add_argument("--max-cells", type=int, default=0, help="limit grid to first N cells (smoke test; 0=all)")
    args = ap.parse_args()
    if args.probe:
        probe(args.probe)
    elif args.validate:
        validate(args.validate)
    elif args.grid:
        run_two_stage_grid(args.tune_users, args.confirm_users, args.processes, args.max_cells)
    else:
        ap.error("pass --probe N, --validate N, or --grid")
