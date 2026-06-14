#!/usr/bin/env python3
r"""Gated default-parameter meta-optimizer for the FSRS-7 Rust fork.

In-process port of the CUDA repo's `central_diff_init_w.py --gated-recency` (no docker; the docker
benchmark per eval is replaced by the compact-raw bindings). Tunes the 34 GLOBAL DEFAULT parameters
(the universal SGD init + L2 anchor) with central-difference Adam, decoupling:

  * PROXY  (drives the descent): 0-epoch `--default` logloss_by_user — apply the candidate defaults
    DIRECTLY to predict each user's 5-fold test folds (NO per-user training). Cheap.
  * GATE   (selection metric, every --recency-every steps): the real 5-fold cross-val logloss_by_user
    with the candidate as per-user SGD init AND L2 anchor, at the shipped (9,512)-gold recipe.

Selection = the recency-best checkpoint; step-0 = the unmodified shipped DEFAULT_PARAMETERS is always
in the comparison set, so "no improvement => keep the champion" is detectable. Crash-resumable atomic
checkpoint after every step; twin-axis plot (dense proxy + sparse gate).

Both proxy and gate reuse the SAME compact-raw fold cache (benchmark.build_user_raw_folds): the proxy
predicts the test folds with the candidate directly; the gate trains each fold first. Bindings used
(all verified bit-for-bit, see profiling/verify_raw.py + verify_gate.py): compute_parameters_raw with
init_w (train), memory_states_raw (test memory states), predict(precomputed_states=...) (curve).

⚠ The CUDA repo found post-8-epoch loss is ~INSENSITIVE to the SGD starting point (the gate's
objective), so the gate may come back ~flat (keep champion) — still a useful confirmation.

Profiling/research tool: never writes a result/ champion record (constraint 10). Run e.g.:
    python default_param_tune.py --steps 30 --recency-every 5 --max-user-id 10000 --processes 14
    python default_param_tune.py --probe 60        # size the run before launching
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
RESULT = REPO / "result"
OUTDIR = RESULT / "default_param_tune"
sys.path.insert(0, str(REPO))

# benchmark.py builds its global Config from sys.argv at import; mirror the gold-tuning invocation
# (--short --secs --recency, 5-fold). Save/restore argv so our own args don't leak into it.
_SAVED_ARGV = sys.argv
sys.argv = ["benchmark.py", "--algo", "FSRS-rs", "--short", "--secs", "--recency",
            "--processes", "1", "--max-user-id", "100000"]
import benchmark as bm  # noqa: E402
sys.argv = _SAVED_ARGV
from sklearn.metrics import log_loss  # noqa: E402
from fsrs_rs_python import FSRS, DEFAULT_PARAMETERS  # noqa: E402
from data_loader import UserDataLoader  # noqa: E402

config = bm.config

# ── Meta-optimizer config (identical to the CUDA originals) ─────────────────────────────────────
LR, BETA1, BETA2, EPS, H = 5e-3, 0.9, 0.999, 1e-8, 5e-3

# ── 34-param box bounds — mirror fsrs-rs/src/model.rs clip_parameters (511-554). ────────────────
BOUNDS = [
    (0.0001, 50.0), (0.0001, 100.0), (0.0001, 100.0), (0.0001, 100.0),  # 0-3 initial stabilities
    (1.0, 10.0), (0.001, 4.0), (0.1, 4.0),                              # 4 init_d0, 5 init_d1, 6 next_d_mult
    (0.0, 4.0), (0.0, 1.2), (0.3, 3.0), (0.01, 1.5), (0.1, 1.0),        # 7-14 long-trace stability
    (0.0, 3.5), (0.0, 1.0), (1.0, 7.0),
    (0.0, 4.0), (0.0, 2.0), (0.5, 6.0), (0.001, 1.5), (0.001, 1.0),     # 15-22 short-trace stability
    (0.0, 5.0), (0.0, 1.0), (1.0, 7.0),
    (0.01, 0.25), (0.01, 0.95), (0.2, 0.85), (0.5, 0.99),               # 23 decay1, 24 decay2, 25 base1, 26 base2
    (0.01, 1.0), (0.1, 1.0), (0.0, 0.9), (0.1, 1.1),                    # 27-30 curve weights/powers
    (0.0, 1.0), (0.0, 0.6), (0.0, 0.6),                                 # 31 d_weight, 32 d_decay, 33 s_decay1
]
assert len(BOUNDS) == 34


def clip_params(params, bounds=BOUNDS):
    return np.array([min(max(v, lo), hi) for v, (lo, hi) in zip(params, bounds)], dtype=float)


def check_constraints(p):
    """Ordering enforced by model.rs clip_parameters (558-561): initial stabilities ordered by
    rating (w0<=w1<=w2<=w3) and base2>=base1 (w26>=w25)."""
    return p[0] <= p[1] <= p[2] <= p[3] and p[26] >= p[25]


def repair_constraints(p):
    p = p.copy()
    p[0:4] = np.maximum.accumulate(p[0:4])     # w0<=w1<=w2<=w3
    p[26] = max(p[26], p[25])                   # base2 >= base1
    return p


# ── Per-user evaluation on the compact-raw fold cache ───────────────────────────────────────────
_BACKEND = None  # shared FSRS(parameters=[]) handle for compute_parameters_raw (init via init_w arg)


def _backend():
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = FSRS(parameters=[])
    return _BACKEND


def _user_proxy(uc, params):
    """FAST 0-epoch PROXY (descent driver): the SIMD windowed loss of `params` over the user's whole
    collection (no training). ~50x cheaper than the burn predict; minimax-approximate + recency-
    weighted, which is fine for driving the descent — the gate is the real selector."""
    full = uc["full_train"]
    return _backend().windowed_loss_raw(full[0], full[1], full[2], full[3], list(params))


def _user_gate(uc, params):
    """FAITHFUL 5-fold cross-val GATE (selection metric): train each fold from `params` as init +
    L2 anchor, predict the test fold with the frozen burn path, accumulate, sklearn log_loss."""
    p_all, y_all = [], []
    for fold in uc["folds"]:
        tr = fold["train"]
        weights, _ = _backend().compute_parameters_raw(tr[0], tr[1], tr[2], tr[3], None, params)
        scorer = FSRS(parameters=weights)
        stab, diff, sfast = scorer.memory_states_raw(
            uc["fd"], uc["fr"], uc["foff"], fold["tci"], fold["thl"])
        pp, yy, _ = bm.predict(fold["slim"], weights, precomputed_states=(stab, diff, sfast))
        p_all.extend(pp)
        y_all.extend(yy)
    return log_loss(y_true=y_all, y_pred=p_all, labels=[0, 1])


def build_user_cache(user_id, loader):
    """Compact-raw fold cache for one user (None if too small). Numpy stays compact (PyO3 reads it
    directly); slim test frames are built once. Mirrors how the (9,512) gold was measured."""
    try:
        dataset = loader.load_user_data(user_id)
    except Exception:  # noqa: BLE001
        return None
    if len(dataset) < 64:
        return None
    raw = bm.build_user_raw_folds(dataset, config)
    if raw is None:
        return None
    import pandas as pd
    full = raw["full"]
    folds = []
    for f in raw["folds"]:
        folds.append({
            "train": f["train_raw"],
            "tci": f["test_card_idx"],
            "thl": f["test_hist_len"],
            "slim": pd.DataFrame({"delta_t": f["test_dt"], "y": f["test_y"]}),
        })
    # full_train = whole-collection raw (deltas, ratings, review_ths, card_offsets) for the fast
    # windowed PROXY; fd/fr/foff alias it for memory_states_raw (test-history reconstruction).
    return {"full_train": full, "fd": full[0], "fr": full[1], "foff": full[3], "folds": folds}


# ── Persistent worker pool: each worker caches a disjoint user slice, evaluates param vectors ────
def _pool_worker(slice_ids, in_q, out_q):
    loader = UserDataLoader(config)
    cache = {}
    for u in slice_ids:
        c = build_user_cache(u, loader)
        if c is not None:
            cache[u] = c
    try:
        import psutil
        rss = psutil.Process().memory_info().rss
    except Exception:  # noqa: BLE001
        rss = 0
    out_q.put(("ready", len(cache), rss))
    while True:
        msg = in_q.get()
        if msg is None:
            return
        mode, vecs = msg
        try:
            fn = _user_gate if mode == "gate" else _user_proxy
            # Per candidate vector: sum per-user loss over this worker's slice + count.
            out = []
            for vec in vecs:
                s = 0.0
                for u in cache:
                    s += fn(cache[u], vec)
                out.append((s, len(cache)))
            out_q.put(out)
        except Exception:  # noqa: BLE001
            import traceback
            out_q.put(("__ERROR__", traceback.format_exc()))


class WorkerPool:
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
        self.n_users = 0
        self.total_rss = 0
        for q in self.out_qs:
            tag, nu, rss = q.get()
            assert tag == "ready"
            self.n_users += nu
            self.total_rss += rss

    def _run(self, mode, vecs):
        for q in self.in_qs:
            q.put((mode, vecs))
        # accumulate (sum_ll, n) per vector across workers
        totals = [[0.0, 0] for _ in vecs]
        for q in self.out_qs:
            r = q.get()
            if isinstance(r, tuple) and r and r[0] == "__ERROR__":
                raise RuntimeError(f"worker eval failed:\n{r[1]}")
            for i, (s, n) in enumerate(r):
                totals[i][0] += s
                totals[i][1] += n
        return [s / n if n else float("nan") for s, n in totals]

    def proxy(self, vecs):
        """0-epoch by_user loss for each candidate vector (list)."""
        return self._run("proxy", vecs)

    def gate(self, vec):
        """8/9-epoch recency cross-val by_user loss for ONE candidate vector."""
        return self._run("gate", [vec])[0]

    def close(self):
        for q in self.in_qs:
            q.put(None)
        for p in self.procs:
            p.join(timeout=10)


# ── Adam + central-difference gradients (faithful to the CUDA original) ──────────────────────────
class AdamCentralDiff:
    def __init__(self, params, lr=LR, beta1=BETA1, beta2=BETA2, eps=EPS, h=H):
        self.n = len(params)
        self.params = np.array(params, dtype=float)
        self.lr, self.beta1, self.beta2, self.eps, self.h = lr, beta1, beta2, eps, h
        self.m = np.zeros(self.n)
        self.v = np.zeros(self.n)
        self.t = 0
        self.counteval = 0

    def gradient(self, eval_batch):
        """Central-difference gradient at self.params; both perturbed points clipped to bounds, the
        denominator uses the realized step (correct at saturated bounds). 2N perturbations -> ONE
        batch call (the cache is loaded once per worker)."""
        cands, denoms = [], []
        for i in range(self.n):
            pp, pm = self.params.copy(), self.params.copy()
            pp[i] = min(max(self.params[i] + self.h, BOUNDS[i][0]), BOUNDS[i][1])
            pm[i] = min(max(self.params[i] - self.h, BOUNDS[i][0]), BOUNDS[i][1])
            cands.append(pp)
            cands.append(pm)
            denoms.append(pp[i] - pm[i])
        losses = eval_batch(cands)
        self.counteval += len(cands)
        grad = np.zeros(self.n)
        for i in range(self.n):
            d = denoms[i]
            grad[i] = (losses[2 * i] - losses[2 * i + 1]) / d if d != 0.0 else 0.0
        return grad

    def step(self, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        new = self.params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        new = clip_params(new)
        if not check_constraints(new):
            new = clip_params(repair_constraints(new))
        self.params = new
        return self.params


# ── Checkpoint IO (atomic) + twin-axis plot ─────────────────────────────────────────────────────
def _atomic_write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _save_plot(history, recency_history, path):
    if not history:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"   (plot skipped: {e})")
        return
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot([e["step"] for e in history], [e["loss"] for e in history],
             marker="o", color="tab:blue", label="default (0-epoch proxy)")
    ax1.set_xlabel("meta-step")
    ax1.set_ylabel("logloss_by_user — default (proxy)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True)
    ax2 = ax1.twinx()
    if recency_history:
        ax2.plot([e["step"] for e in recency_history], [e["loss"] for e in recency_history],
                 marker="s", color="tab:red", label="recency (cross-val gate, real metric)")
    ax2.set_ylabel("logloss_by_user — recency (gate)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.ticklabel_format(axis="y", style="plain", useOffset=False)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    ax1.set_title("Gated default-param meta-opt: descend proxy, select on recency gate")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ── Gated meta-opt loop ─────────────────────────────────────────────────────────────────────────
def run_gated(pool, start_params, max_steps, recency_every, ckpt, plot):
    opt = AdamCentralDiff(start_params)
    history, recency_history, completed = [], [], 0
    if ckpt.is_file():
        try:
            cp = json.loads(ckpt.read_text())
            opt.params = np.array(cp["params"]); opt.m = np.array(cp["m"]); opt.v = np.array(cp["v"])
            opt.t = int(cp["t"]); opt.counteval = int(cp["counteval"])
            history, recency_history, completed = cp["history"], cp["recency_history"], int(cp["completed_steps"])
            print(f"Resumed from {ckpt.name}: step {completed}")
        except Exception as e:  # noqa: BLE001
            print(f"Could not load checkpoint ({e}); starting fresh")

    def save():
        _atomic_write_json(ckpt, {
            "params": opt.params.tolist(), "m": opt.m.tolist(), "v": opt.v.tolist(),
            "t": int(opt.t), "counteval": int(opt.counteval), "history": history,
            "recency_history": recency_history, "completed_steps": completed,
            "max_steps": max_steps, "recency_every": recency_every,
        })
        _save_plot(history, recency_history, plot)

    if completed == 0 and not recency_history:
        l0 = pool.proxy([start_params])[0]
        r0 = pool.gate(start_params)
        opt.counteval += 2
        history.append({"step": 0, "params": list(start_params), "loss": l0})
        recency_history.append({"step": 0, "params": list(start_params), "loss": r0})
        print(f"  step-0 baseline (shipped DEFAULT_PARAMETERS): proxy {l0:.6f}  gate {r0:.8f}")
        save()

    for step in range(completed + 1, max_steps + 1):
        t0 = time.perf_counter()
        grad = opt.gradient(pool.proxy)
        new = opt.step(grad)
        loss = pool.proxy([new])[0]
        opt.counteval += 1
        history.append({"step": step, "params": new.tolist(), "loss": loss,
                        "grad_norm": float(np.linalg.norm(grad))})
        line = (f"[gated] step {step}/{max_steps}: proxy={loss:.6f}  "
                f"|grad|={np.linalg.norm(grad):.4f}  ({time.perf_counter()-t0:.0f}s)")
        if step % recency_every == 0:
            r = pool.gate(new)
            opt.counteval += 1
            recency_history.append({"step": step, "params": new.tolist(), "loss": r})
            best = min(recency_history, key=lambda e: e["loss"])
            line += f"  GATE={r:.8f} (best {best['loss']:.8f} @ step {best['step']})"
        completed = step
        save()
        print(line, flush=True)

    best = min(recency_history, key=lambda e: e["loss"])
    _atomic_write_json(OUTDIR / "summary_gated.json", {
        "best_gate_loss": best["loss"], "best_gate_step": best["step"], "best_params": best["params"],
        "step0_gate_loss": recency_history[0]["loss"],
        "recency_history": [{"step": e["step"], "loss": e["loss"]} for e in recency_history],
    })
    print(f"\n[gated] done. best GATE {best['loss']:.8f} @ step {best['step']} "
          f"(step 0 = shipped default = {recency_history[0]['loss']:.8f})")
    print(f"[gated] {'IMPROVED' if best['step'] != 0 else 'NO IMPROVEMENT -> keep champion'}")
    print(f"[gated] best_params = {[round(v, 5) for v in best['params']]}")
    return best


# ── user id discovery ───────────────────────────────────────────────────────────────────────────
def all_user_ids(limit):
    import pyarrow.parquet as pq
    ids = sorted(u.as_py() for u in
                 pq.ParquetDataset(config.data_path / "revlogs").partitioning.dictionaries[0])
    return ids[:limit]


def probe(n_users, P):
    """Build N users single-process, time one proxy batch (2N+1) and one gate; extrapolate.
    RAM is reported as MARGINAL cache (RSS delta from after-imports), since the ~1.1 GB torch/pandas
    base does NOT scale with users — it scales with WORKERS (each process pays it once)."""
    loader = UserDataLoader(config)
    ids = all_user_ids(n_users)
    default = list(DEFAULT_PARAMETERS)
    try:
        import psutil
        proc = psutil.Process()
        base_rss = proc.memory_info().rss  # after imports, before any user cache
    except Exception:  # noqa: BLE001
        proc = None
        base_rss = 0
    t0 = time.perf_counter()
    cache = {u: c for u in ids if (c := build_user_cache(u, loader)) is not None}
    build_s = time.perf_counter() - t0
    n_ok = len(cache)
    rss = (proc.memory_info().rss - base_rss) if proc else 0  # marginal cache only
    # one proxy eval (single vector) over all cached users
    t0 = time.perf_counter()
    proxy_mean = statistics.mean(_user_proxy(cache[u], default) for u in cache)
    proxy_s = time.perf_counter() - t0
    # one gate eval
    t0 = time.perf_counter()
    gate_mean = statistics.mean(_user_gate(cache[u], default) for u in cache)
    gate_s = time.perf_counter() - t0
    print(f"  step-0 proxy={proxy_mean:.6f}  gate={gate_mean:.6f}")
    base_gb = base_rss / 2**30
    print(f"\n=== PROBE: {n_ok}/{n_users} users (single process) ===")
    print(f"  cache build : {build_s:7.1f}s  ({1000*build_s/n_ok:.1f} ms/user)")
    if proc:
        print(f"  base RSS    : {base_gb:7.2f} GB  (per process; torch/pandas imports)")
        print(f"  cache (marg): {rss/2**30:7.2f} GB  ({rss/n_ok/2**20:.2f} MB/user)")
    print(f"  1 proxy eval: {proxy_s:7.2f}s  ({1000*proxy_s/n_ok:.1f} ms/user)")
    print(f"  1 gate  eval: {gate_s:7.2f}s  ({1000*gate_s/n_ok:.1f} ms/user)")
    # extrapolate to U users / P workers; a step = (2N+1) proxy + (1/recency_every) gate
    for U in (3000, 10000):
        pf = U / n_ok / P
        proxy_step = (2 * 34 + 1) * proxy_s * pf
        gate_step = gate_s * pf / 5  # one gate per 5 steps, amortized
        step_min = (proxy_step + gate_step) / 60
        ram = (rss / n_ok * U + base_gb * 2**30 * P) / 2**30  # cache scales w/ users, base w/ workers
        print(f"  -> {U} users / {P} workers: ~{step_min:.1f} min/step, "
              f"~{step_min*30/60:.1f} h for 30 steps, RAM ~{ram:.0f} GB "
              f"(~{rss/n_ok*U/2**30:.0f} cache + ~{base_gb*P:.0f} base)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--recency-every", type=int, default=5)
    ap.add_argument("--max-user-id", type=int, default=10000, help="number of users (largest cache)")
    ap.add_argument("--processes", type=int, default=14)
    ap.add_argument("--probe", type=int, default=0, help="probe N users (size the run) and exit")
    args = ap.parse_args(argv)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    if args.probe:
        probe(args.probe, args.processes)
        return

    start = clip_params(np.array(list(DEFAULT_PARAMETERS), dtype=float))
    for i, (p, (lo, hi)) in enumerate(zip(start, BOUNDS)):
        assert lo <= p <= hi, f"default param {i}={p} outside ({lo},{hi})"
    ids = all_user_ids(args.max_user_id)
    print(f"Gated default-param meta-opt: {len(ids)} users, {args.processes} workers, "
          f"{args.steps} steps, gate every {args.recency_every}.")
    print(f"Adam lr={LR} b1={BETA1} b2={BETA2} h={H}; {2*34+1} evals/step.")
    pool = WorkerPool(ids, args.processes)
    print(f"Pool ready: {pool.n_users} users cached, ~{pool.total_rss/2**30:.1f} GB total RAM.")
    try:
        run_gated(pool, start, args.steps, args.recency_every,
                  OUTDIR / "gated_results.json", OUTDIR / "loss_gated.png")
    finally:
        pool.close()


if __name__ == "__main__":
    main()
