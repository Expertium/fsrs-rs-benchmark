"""Phase-2 measurement harness: time the Rust benchmark() region per user (profiling-only).

Counterpart to profiling/measure.py, but for benchmark() instead of compute_parameters().
benchmark.py trains one weight-set per TimeSeriesSplit fold (n_splits=5), so a user's
benchmark() Rust cost = the SUM of its per-fold benchmark() calls. This harness reproduces
benchmark.py's exact fold train sets (with --short --secs --recency: train_equals_test off,
no day filters, no equalize), calls the timed binding FSRS.benchmark_timed() (returns
(weights, elapsed_secs); the PyO3 item conversion above the Rust timer is excluded, mirroring
compute_parameters' timing), and records the per-user min-of-3 total time plus the per-fold
trained weights flattened together (the bit-for-bit key: identical weights => identical
benchmark.py predictions / log loss).

NOT timed, NOT counted by complexity.py (lives in profiling/), never writes a champion file.
The .pyd-swap recipe is unchanged: build with `python profiling/measure.py build`, snapshot the
installed .pyd, swap to compare two binaries back-to-back in one session.

Subcommands
-----------
  run LABEL [--max-user-id N] [--processes P]
        Time every user (min-of-3) and snapshot per-user records to
        profiling/measure_bench/LABEL.jsonl. Parent + workers pinned to HIGH priority and
        disjoint 2-CPU blocks off logical CPUs 0-3 (same as compute_parameters.py).

  compare CHAMP CAND
        Read two snapshots and report:
          * speed_ratio = median over users of (t_champ / t_cand)   <- THE accept metric (>=1.05)
          * mean speedup = mean of the same ratios                  (informational)
          * weight drift: raw bit-for-bit diff AND 4-decimal-rounded diff (benchmark.py rounds
            weights to 4 places, so a 0 rounded-diff => identical benchmark.py output).
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path as _Path

# Repo root first on sys.path so config/data_loader/features resolve under spawn.
_REPO = _Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from typing import List, Optional

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
from tqdm.auto import tqdm  # type: ignore

from config import create_parser, Config
from data_loader import UserDataLoader
from fsrs_rs_python import FSRS, FSRSItem, FSRSReview, DEFAULT_PARAMETERS  # type: ignore[import-untyped]

_SNAP_DIR = _REPO / "profiling" / "measure_bench"
TIMING_REPEATS = 3
_PIN_RESERVED_CPUS = 4
_PIN_CPUS_PER_WORKER = 2

# Same big-first dispatch order as compute_parameters.py (warm-up heuristic; cuts wall-clock,
# does not change the recorded per-user numbers). Any user not listed sorts last by id.
USERS_BY_SIZE_DESC = [
    34, 17, 33, 40, 15, 6, 41, 14, 12, 21,
    25, 2, 38, 24, 8, 39, 32, 35, 18, 10,
    47, 20, 43, 28, 31, 37, 1, 30, 44, 11,
    7, 45, 5, 50, 29, 13, 4, 48, 23, 16,
    19, 27, 49, 3, 46, 9, 22, 36, 26, 42,
]


# ---------------------------------------------------------------------------
# Review-history parsing + fold construction (faithful to benchmark.py)
# ---------------------------------------------------------------------------
def _parse_scalar(value: object, type_name: str) -> str:
    if pd.isna(value):
        raise ValueError(f"Expected a {type_name} history value, got missing data")
    result = str(value).strip()
    if not result:
        raise ValueError(f"Expected a {type_name} history value, got empty text")
    return result


def parse_interval(value: object) -> float:
    return max(0.0, float(_parse_scalar(value, "numeric review")))


def parse_rating(value: object) -> int:
    return int(float(_parse_scalar(value, "rating")))


def parse_history(history: object, parser) -> list:
    if pd.isna(history):
        return []
    fields = (field.strip() for field in str(history).split(","))
    return [parser(field) for field in fields if field]


def build_reviews(row: pd.Series, *, include_current: bool = False) -> List[FSRSReview]:
    t_history = parse_history(row["t_history"], parse_interval)
    r_history = parse_history(row["r_history"], parse_rating)
    if include_current:
        t_history.append(parse_interval(row["delta_t"]))
        r_history.append(parse_rating(row["rating"]))
    if len(t_history) != len(r_history):
        raise ValueError("Review history lengths do not match")
    return [FSRSReview(delta_t=t, rating=r) for t, r in zip(t_history, r_history)]


def convert_to_items(df: pd.DataFrame) -> List[FSRSItem]:
    """benchmark.py's convert_to_items (plain path, no card_ids): one FSRSItem per review row,
    globally ordered by review_th."""
    pairs = []  # (review_th, FSRSItem)
    for _, group in df.sort_values(by=["card_id", "review_th"]).groupby("card_id"):
        for _, row in group.iterrows():
            item = FSRSItem(reviews=build_reviews(row, include_current=True))
            pairs.append((row["review_th"], item))
    pairs.sort(key=lambda pair: pair[0])
    return [item for _, item in pairs]


def build_folds(dataset: pd.DataFrame, config: Config) -> List[List[FSRSItem]]:
    """Reproduce benchmark.py's per-fold TRAIN sets (the only thing benchmark() trains on).
    With --short --secs --recency: train_equals_test off, no day filters, no equalize."""
    folds: List[List[FSRSItem]] = []
    for _, (train_index, test_index) in enumerate(
        TimeSeriesSplit(n_splits=config.n_splits).split(dataset)
    ):
        train_set = dataset.iloc[train_index]
        test_set = dataset.iloc[test_index]
        if train_set.empty or test_set.empty:
            continue
        folds.append(convert_to_items(train_set))
    return folds


# ---------------------------------------------------------------------------
# Worker pinning (mirrors compute_parameters.py) + stderr silencing
# ---------------------------------------------------------------------------
def _set_high_priority() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00000080)
    except Exception:
        pass


def _pin_to_cpus(worker_index: int) -> None:
    if sys.platform != "win32":
        return
    lo = _PIN_RESERVED_CPUS + worker_index * _PIN_CPUS_PER_WORKER
    hi = lo + _PIN_CPUS_PER_WORKER
    if hi > (os.cpu_count() or 0):
        return
    mask = 0
    for cpu in range(lo, hi):
        mask |= 1 << cpu
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        k32.SetProcessAffinityMask(k32.GetCurrentProcess(), mask)
    except Exception:
        pass


def _init_worker(counter, lock) -> None:
    _set_high_priority()
    with lock:
        worker_index = counter.value
        counter.value = worker_index + 1
    _pin_to_cpus(worker_index)
    # Silence the Rust `PROFILE total_train_region:` eprintln (fired once per benchmark() call,
    # i.e. ~5 folds * 3 reps per user) so it doesn't flood the console. Redirect this worker's
    # fd 2 (shared by Rust eprintln and Python stderr) to devnull; errors propagate via futures.
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Per-user timing
# ---------------------------------------------------------------------------
_CONFIG: Optional[Config] = None


def _worker_config(data_path: str, max_user_id: int) -> Config:
    global _CONFIG
    if _CONFIG is None:
        parser = create_parser()
        argv = ["--algo", "FSRS-rs", "--short", "--secs", "--recency",
                "--data", data_path, "--max-user-id", str(max_user_id), "--processes", "1"]
        args = parser.parse_args(argv)
        _CONFIG = Config(args)
        _CONFIG.partitions = "none"
    return _CONFIG


def process(user_id: int, data_path: str, max_user_id: int) -> Optional[dict]:
    """Time the sum of per-fold benchmark() Rust calls for one user, min-of-3."""
    try:
        config = _worker_config(data_path, max_user_id)
        dataset = UserDataLoader(config).load_user_data(user_id)
        folds = build_folds(dataset, config)
        if not folds:
            return None
        backend = FSRS(parameters=[])
        default_w = list(DEFAULT_PARAMETERS)
        times_ms: List[float] = []
        flat_params: List[float] = []
        for rep in range(TIMING_REPEATS):
            total_secs = 0.0
            for items in folds:
                try:
                    p, secs = backend.benchmark_timed(items)
                except Exception:
                    p, secs = default_w, 0.0  # inadequate-data fold: benchmark.py uses defaults
                total_secs += secs
                if rep == 0:
                    flat_params.extend(float(w) for w in p)
            times_ms.append(total_secs * 1000.0)
        return {
            "user": int(user_id),
            "time_ms": round(min(times_ms), 3),
            "time_ms_runs": [round(t, 3) for t in times_ms],
            "parameters": flat_params,
            "n_folds": len(folds),
            "size": sum(len(f) for f in folds),
        }
    except Exception as exc:  # surface as a string; never crash the pool
        return {"user": int(user_id), "error": repr(exc)}


# ---------------------------------------------------------------------------
# run / compare
# ---------------------------------------------------------------------------
def cmd_run(label: str, max_user_id: int, processes: int, data_path: str) -> int:
    _SNAP_DIR.mkdir(parents=True, exist_ok=True)
    # Discover user ids from the revlogs partition dir (same source benchmark.py uses).
    revlogs = _Path(data_path) / "revlogs"
    user_ids = []
    for entry in revlogs.glob("user_id=*"):
        try:
            uid = int(entry.name.split("=")[1])
        except (IndexError, ValueError):
            continue
        if max_user_id is not None and uid > max_user_id:
            continue
        user_ids.append(uid)
    rank = {uid: i for i, uid in enumerate(USERS_BY_SIZE_DESC)}
    user_ids.sort(key=lambda uid: (rank.get(uid, len(rank)), uid))

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    counter = manager.Value("i", 0)
    lock = manager.Lock()
    records = []
    with ProcessPoolExecutor(
        max_workers=processes, initializer=_init_worker, initargs=(counter, lock)
    ) as ex:
        futs = [ex.submit(process, uid, data_path, max_user_id) for uid in user_ids]
        for fut in (pbar := tqdm(as_completed(futs), total=len(futs), smoothing=0.03)):
            rec = fut.result()
            if rec is None:
                continue
            if "error" in rec:
                tqdm.write(f"user {rec['user']}: {rec['error']}")
                continue
            records.append(rec)
            pbar.set_description(f"user {rec['user']} {rec['time_ms']:.1f}ms")

    records.sort(key=lambda r: r["user"])
    snap = _SNAP_DIR / f"{label}.jsonl"
    with open(snap, "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    med = statistics.median(r["time_ms"] for r in records) if records else float("nan")
    total_rev = sum(r["size"] for r in records)
    print(f"[run] snapshot -> {snap}  users={len(records)} "
          f"median_time={med:.1f}ms total_train_rows={total_rev:,}", flush=True)
    return 0


def _load(label: str) -> dict:
    p = _SNAP_DIR / f"{label}.jsonl"
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["user"]: r for r in rows}


def cmd_compare(champ: str, cand: str) -> int:
    a, b = _load(champ), _load(cand)
    users = sorted(set(a) & set(b))
    if not users:
        print("[compare] no overlapping users!", flush=True)
        return 1
    ratios = []
    n_raw_diff = 0
    n_round_diff = 0
    max_abs = 0.0
    for u in users:
        tc, td = a[u]["time_ms"], b[u]["time_ms"]
        if td > 0:
            ratios.append(tc / td)
        pc, pd_ = a[u].get("parameters", []), b[u].get("parameters", [])
        if pc != pd_:
            n_raw_diff += 1
        if len(pc) == len(pd_):
            rc = [round(x, 4) for x in pc]
            rd = [round(x, 4) for x in pd_]
            if rc != rd:
                n_round_diff += 1
            max_abs = max(max_abs, max((abs(x - y) for x, y in zip(pc, pd_)), default=0.0))
        else:
            n_round_diff += 1
            max_abs = float("inf")

    speed_ratio = statistics.median(ratios)
    mean_speedup = statistics.mean(ratios)
    champ_med = statistics.median(a[u]["time_ms"] for u in users)
    cand_med = statistics.median(b[u]["time_ms"] for u in users)
    speed_ok = speed_ratio >= 1.05
    print(f"  users compared      : {len(users)}", flush=True)
    print(f"  median time champ   : {champ_med:.2f} ms", flush=True)
    print(f"  median time cand    : {cand_med:.2f} ms", flush=True)
    print(f"  SPEED_RATIO (median): {speed_ratio:.4f}   <- accept if >= 1.05   [{'OK' if speed_ok else 'NO'}]", flush=True)
    print(f"  mean speedup        : {mean_speedup:.4f}   (informational)", flush=True)
    print(f"  users w/ raw  weight diff : {n_raw_diff}/{len(users)}   ({'BIT-FOR-BIT' if n_raw_diff == 0 else 'reordered'})", flush=True)
    print(f"  users w/ 4dp  weight diff : {n_round_diff}/{len(users)}   (0 => identical benchmark.py output)", flush=True)
    print(f"  max |weight diff|         : {max_abs:g}   (diagnostic)", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("label")
    pr.add_argument("--max-user-id", type=int, default=50)
    pr.add_argument("--processes", type=int, default=10)
    pr.add_argument("--data", default="../anki-revlogs-10k")
    pc = sub.add_parser("compare")
    pc.add_argument("champ")
    pc.add_argument("cand")
    args = ap.parse_args()
    if args.cmd == "run":
        return cmd_run(args.label, args.max_user_id, args.processes, args.data)
    if args.cmd == "compare":
        return cmd_compare(args.champ, args.cand)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
