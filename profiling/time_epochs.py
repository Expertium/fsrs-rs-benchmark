"""Clean min-of-3 timing of the champion at chosen epoch counts (DIAGNOSTIC, profiling-only).

Companion to epoch_sweep.py: the sweep finds the epoch count that buys accuracy back to 0.3098;
this measures the clean median per-user time at 8 epochs (shipped) vs that count, so the README's
epoch-buyback speedup row uses a real measured time ratio rather than the sweep's rough single-run
times. Min-of-3 per user (timing noise is one-sided), workers self-pin via compute_parameters' pool
initializer. 50 users.

Run:  python profiling/time_epochs.py --algo FSRS-rs --short --secs --recency --max-user-id 50 --processes 10
"""

import statistics
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import compute_parameters as cp

EPOCH_COUNTS = [8, 20]
REPS = 3


def _time_user(user_id: int, num_epochs: int):
    ds = cp.UserDataLoader(cp.config).load_user_data(user_id)
    items, cids = cp.convert_to_items(ds)
    backend = cp.FSRS(parameters=[])
    times_ms = []
    for _ in range(REPS):
        _p, secs = backend.compute_parameters(items, cids, num_epochs)
        times_ms.append(secs * 1000.0)
    return int(user_id), min(times_ms)  # min-of-3 (cleanest floor)


def _median_at(num_epochs: int, users: list[int]) -> float:
    manager = mp.Manager()
    counter = manager.Value("i", 0)
    lock = manager.Lock()
    per_user = []
    with ProcessPoolExecutor(
        max_workers=cp.config.num_processes,
        initializer=cp._init_worker,
        initargs=(counter, lock),
    ) as ex:
        for f in as_completed([ex.submit(_time_user, u, ne) for ne in [num_epochs] for u in users]):
            _u, t = f.result()
            per_user.append(t)
    return statistics.median(per_user)


def main() -> None:
    users = list(cp.USERS_BY_SIZE_DESC)
    meds = {}
    for ne in EPOCH_COUNTS:
        meds[ne] = _median_at(ne, users)
        print(f"epochs={ne:4d}  median(min-of-3) per-user time = {meds[ne]:.2f} ms")
    base = meds[EPOCH_COUNTS[0]]
    print("\n=== time ratios vs epochs={} ===".format(EPOCH_COUNTS[0]))
    for ne in EPOCH_COUNTS:
        print(f"epochs={ne:4d}  t/t8 = {meds[ne] / base:.3f}x")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
