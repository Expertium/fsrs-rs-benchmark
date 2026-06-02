"""Time Rust compute_parameters() per user and score it with Rust evaluate().

Counterpart to benchmark.py, but:
  * trains with compute_parameters() (the function the speed campaign optimizes),
  * scores with Rust evaluate() instead of the Python forgetting curve,
  * uses train set == test set (no time-series split), so log loss is expected
    to be *lower* than benchmark.py's cross-validated number,
  * times each user's compute_parameters() 3x and records the MIN (ms). Timing
    noise is one-sided (interference only ever slows a run), so the fastest of
    the three is the cleanest estimate of the true compute floor.

The timed region is the Rust compute_parameters() call only (monotonic clock,
measured inside the extension); the Python here stays untimed.

To reduce measurement noise, launch with high priority pinned off CPUs 0-3:

    start /high /affinity 0xFFFFFFF0 python compute_parameters.py --algo FSRS-rs --short --secs --recency --processes 10 --max-user-id 50

On top of that, each pool worker pins itself to its own disjoint 2-CPU block and
re-applies HIGH priority (workers inherit the launch's affinity but not its
priority class), so cross-worker contention is constant rather than random.
"""

import json
import os
import statistics
import sys
from pathlib import Path as _Path

# Ensure the repo root is first on sys.path so local packages resolve even when
# this is invoked from another directory.
_REPO_ROOT = str(_Path(__file__).resolve().parent)
if not sys.path or sys.path[0] != _REPO_ROOT:
    sys.path.insert(0, _REPO_ROOT)

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pyarrow.parquet as pq  # type: ignore
import torch
from tqdm.auto import tqdm  # type: ignore

from config import create_parser, Config
from data_loader import UserDataLoader
from utils import catch_exceptions, sort_jsonl
from fsrs_rs_python import FSRS, FSRSItem, FSRSReview  # type: ignore[import-untyped]

# Number of times compute_parameters() is timed per user; the MIN is recorded.
TIMING_REPEATS = 3
# Worker CPU pinning: leave logical CPUs 0-3 for the OS, give each worker its own
# disjoint block of CPUs (<= 2, per the "<=2 threads per user" constraint).
_PIN_RESERVED_CPUS = 4
_PIN_CPUS_PER_WORKER = 2

# Canonical 50-user processing order, biggest collection first -> smallest last
# (sizes = item counts from result/compute_parameters-FSRS-rs-short-secs-recency.jsonl,
# the --short --secs --recency config this campaign runs). Dispatching the largest
# users first makes them double as a CPU warm-up: by the time the small,
# noise-sensitive users (which dominate the median metric, constraint 12) run, the
# machine is at thermal/cache equilibrium instead of cold. This is a pure
# measurement-noise heuristic -- it changes only scheduling order, not the recorded
# params/log loss (sort_jsonl re-sorts the output by user at the end anyway).
USERS_BY_SIZE_DESC = [
    34, 17, 33, 40, 15, 6, 41, 14, 12, 21,
    25, 2, 38, 24, 8, 39, 32, 35, 18, 10,
    47, 20, 43, 28, 31, 37, 1, 30, 44, 11,
    7, 45, 5, 50, 29, 13, 4, 48, 23, 16,
    19, 27, 49, 3, 46, 9, 22, 36, 26, 42,
]
assert len(USERS_BY_SIZE_DESC) == len(set(USERS_BY_SIZE_DESC)), \
    "duplicate user id in USERS_BY_SIZE_DESC"


def _set_high_priority() -> None:
    """Raise this worker to HIGH priority. Workers inherit the launch's CPU
    affinity but NOT its priority class, so we re-apply it here. HIGH needs no
    admin rights (unlike REALTIME). No-op off Windows; never raises."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # Type the handle explicitly: an untyped GetCurrentProcess return is
        # marshalled as a 32-bit int and SetPriorityClass then silently fails.
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.SetPriorityClass.restype = wintypes.BOOL
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00000080)
    except Exception:
        pass


def _pin_to_cpus(worker_index: int) -> None:
    """Pin this worker to its own disjoint block of CPUs so contention between
    the parallel workers is constant rather than random (a major source of
    run-to-run timing noise). Worker i gets CPUs [4 + 2i, 4 + 2i + 1]. No-op if
    there aren't enough CPUs or off Windows."""
    if sys.platform != "win32":
        return
    lo = _PIN_RESERVED_CPUS + worker_index * _PIN_CPUS_PER_WORKER
    hi = lo + _PIN_CPUS_PER_WORKER
    if hi > (os.cpu_count() or 0):
        return  # not enough CPUs to pin disjointly; keep the inherited affinity
    mask = 0
    for cpu in range(lo, hi):
        mask |= 1 << cpu
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = wintypes.BOOL
        kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), mask)
    except Exception:
        pass


def _init_worker(counter, lock) -> None:
    """Pool worker initializer: HIGH priority + a unique disjoint CPU block."""
    _set_high_priority()
    with lock:
        worker_index = counter.value
        counter.value = worker_index + 1
    _pin_to_cpus(worker_index)


# ---------------------------------------------------------------------------
# Review-history parsing (same shape as benchmark.py)
# ---------------------------------------------------------------------------


def _parse_scalar(value: object, type_name: str) -> str:
    if pd.isna(value):
        raise ValueError(f"Expected a {type_name} history value, got missing data")
    result = str(value).strip()
    if not result:
        raise ValueError(f"Expected a {type_name} history value, got empty text")
    return result


def parse_interval(value: object) -> float:
    """Parse a single delta_t field, clamped to be non-negative."""
    return max(0.0, float(_parse_scalar(value, "numeric review")))


def parse_rating(value: object) -> int:
    return int(float(_parse_scalar(value, "rating")))


def parse_history(history: object, parser) -> list:
    """Split a comma-separated history string and parse each non-empty field."""
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


def convert_to_items(df: pd.DataFrame) -> tuple[List[FSRSItem], List[int]]:
    """Convert a DataFrame to FSRSItems for fsrs-rs, ordered globally by review_th.

    Also returns each item's originating card id (parallel list, same order) so the optimizer can
    group a card's expanding-window prefix-items into one mini-batch (the O(N) window path). The
    items list and ordering are unchanged vs. before, so log loss / timing are unaffected by this.
    """
    pairs = []  # (review_th, card_id, FSRSItem)
    for card_id, group in df.sort_values(by=["card_id", "review_th"]).groupby("card_id"):
        for _, row in group.iterrows():
            item = FSRSItem(reviews=build_reviews(row, include_current=True))
            pairs.append((row["review_th"], int(card_id), item))
    # Sort by review_th only (the key); ties keep insertion order and never compare the items.
    pairs.sort(key=lambda pair: pair[0])
    items = [item for _, _, item in pairs]
    card_ids = [cid for _, cid, _ in pairs]
    return items, card_ids


parser = create_parser()
args, _ = parser.parse_known_args()
if args.algo == parser.get_default("algo"):
    args.algo = "FSRS-rs"
elif args.algo != "FSRS-rs":
    raise ValueError("compute_parameters only supports --algo FSRS-rs")
config = Config(args)
config.partitions = "none"

torch.manual_seed(config.seed)

# Distinct output name so results never collide with benchmark.py's.
OUTPUT_NAME = f"compute_parameters-{config.get_evaluation_file_name()}"


@catch_exceptions
def process(user_id: int, device_id: Optional[int] = None) -> tuple[dict, Optional[dict]]:
    """Train with compute_parameters() and score with evaluate(); train set == test set."""
    del device_id
    dataset = UserDataLoader(config).load_user_data(user_id)
    items, card_ids = convert_to_items(dataset)

    backend = FSRS(parameters=[])
    # compute_parameters() is deterministic (seeded), so params are identical
    # across repeats; only the timing varies. Record the min (cleanest) time.
    times_ms = []
    params: List[float] = []
    for _ in range(TIMING_REPEATS):
        try:
            params, secs = backend.compute_parameters(items, card_ids)
        except TypeError:
            # Champion/baseline builds that predate the card_ids arg take items only. The
            # TypeError is raised by PyO3 arg parsing BEFORE the Rust timer starts, so the
            # fallback call's `secs` is still a clean compute_parameters() timing.
            params, secs = backend.compute_parameters(items)
        times_ms.append(secs * 1000.0)
    params = [round(w, 4) for w in params]

    predictor = FSRS(parameters=params)
    log_loss = predictor.evaluate(items)  # train set == test set

    stats = {
        "user": int(user_id),
        "time_ms": round(min(times_ms)),
        "time_ms_runs": [round(t) for t in times_ms],
        "parameters": params,
        "metrics": {"LogLoss": round(log_loss, 6)},
        "size": len(items),
    }
    return stats, None


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    dataset = pq.ParquetDataset(config.data_path / "revlogs")
    Path("result").mkdir(parents=True, exist_ok=True)
    result_file = Path(f"result/{OUTPUT_NAME}.jsonl")
    if result_file.exists():
        processed_user = {row["user"] for row in sort_jsonl(result_file)}
    else:
        processed_user = set()

    unprocessed_users = []
    for user_id in dataset.partitioning.dictionaries[0]:
        user_id_value = user_id.as_py()
        if config.max_user_id is not None and user_id_value > config.max_user_id:
            continue
        if user_id_value in processed_user:
            continue
        unprocessed_users.append(user_id_value)
    # Dispatch biggest collections first so they warm up the CPU (see
    # USERS_BY_SIZE_DESC). ProcessPoolExecutor pulls submitted tasks FIFO, so
    # submission order == start order. Any user not in the hardcoded ranking
    # (e.g. --max-user-id > 50) sorts last, by ascending id, so behaviour stays
    # deterministic for non-canonical runs.
    _size_rank = {uid: i for i, uid in enumerate(USERS_BY_SIZE_DESC)}
    unprocessed_users.sort(key=lambda uid: (_size_rank.get(uid, len(_size_rank)), uid))

    # Hand each worker a unique index (via a manager-backed counter) so it can
    # claim its own disjoint CPU block in the initializer.
    manager = mp.Manager()
    worker_counter = manager.Value("i", 0)
    counter_lock = manager.Lock()

    with ProcessPoolExecutor(
        max_workers=config.num_processes,
        initializer=_init_worker,
        initargs=(worker_counter, counter_lock),
    ) as executor:
        futures = [executor.submit(process, user_id, None) for user_id in unprocessed_users]
        for future in (
            pbar := tqdm(as_completed(futures), total=len(futures), smoothing=0.03)
        ):
            try:
                result, error = future.result()
                if error:
                    tqdm.write(str(error))
                else:
                    stats, _ = result
                    with open(result_file, "a", encoding="utf-8", newline="\n") as f:
                        f.write(json.dumps(stats, ensure_ascii=False) + "\n")
                    pbar.set_description(f"Processed {stats['user']}")
            except Exception as e:
                tqdm.write(str(e))

    data = sort_jsonl(result_file)
    n_users = len(data)
    if n_users:
        total_items = sum(d["size"] for d in data)
        mean_ll = sum(d["metrics"]["LogLoss"] for d in data) / n_users
        # Median per-user time is the metric of record (constraint 12); mean log
        # loss is the conventional correctness aggregate that matches the references.
        median_time = statistics.median(d["time_ms"] for d in data)
        print(
            f"users={n_users} total_items={total_items} "
            f"LogLoss(mean)={mean_ll:.4f} time(median)={median_time:.1f}ms"
        )
    # Number of users AND dataset size must stay fixed for the canonical run
    # (constraint 5), so review preprocessing can't silently change.
    if config.max_user_id == 50:
        assert n_users == 50, f"expected 50 users, got {n_users}"
        assert total_items == 1_897_936, f"expected 1,897,936 items, got {total_items:,}"
