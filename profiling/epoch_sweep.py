"""Epoch-buyback sweep (DIAGNOSTIC, profiling-only — NOT part of the speed loop, not logged to history).

Reverting iter23 (the cruder windowed minimax) left compute_parameters' mean log loss at ~0.3110 on
the 50-user set, above the iter0 8-epoch baseline 0.3098. This sweeps `num_epochs` (log-spaced 8..200)
to find how many epochs of the (fast) champion buy the accuracy back down to 0.3098, and what the
relative speedup is there. It answers Andrew's "at what price (epochs) do we buy back the lost loss?"

Mean log loss is deterministic (seeded), so the loss curve uses a single parallel run per epoch count.
Bumping num_epochs also stretches the cosine-annealing LR schedule over the larger iteration count
(the standard "train longer" behaviour — same as the 800-epoch precedent), so each epoch count is a
genuine longer training, not just more steps at a collapsed LR.

The sweep is ascending and EARLY-STOPS: |loss - 0.3098| falls as epochs rise, hits a minimum at the
crossing, then grows once loss dips below 0.3098 — so we stop one step after loss first crosses below
the target. The shipped champion stays at 8 epochs (constraint 4); this only *characterises* the trade.

Per-epoch median time here is ROUGH (single run, workers under load) — use it only for the linear
trend. Final speedup numbers come from measure.py's clean min-of-3 at the two chosen epoch counts.

Run (CPU-pinned, like the timing harness):
    python profiling/epoch_sweep.py --algo FSRS-rs --short --secs --recency --max-user-id 50 --processes 10
"""

import json
import statistics
import sys
from pathlib import Path

# Repo root on sys.path so `import compute_parameters` resolves from profiling/.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Importing compute_parameters runs its module-level arg-parse against OUR argv (parse_known_args),
# building `config` for --short --secs --recency --max-user-id 50, and gives us its data/worker helpers.
import compute_parameters as cp

TARGET = 0.3098          # iter0 8-epoch baseline mean LogLoss (band centre) we want to buy back to
N_STEPS = 50             # log-spaced steps from MIN_EPOCHS..MAX_EPOCHS
MIN_EPOCHS, MAX_EPOCHS = 8, 200


def _train_eval(user_id: int, num_epochs: int):
    """Train ONE user at `num_epochs` and score it with the frozen evaluate() (train==test).
    Returns (user_id, log_loss, n_items, secs). Single run (loss is deterministic)."""
    dataset = cp.UserDataLoader(cp.config).load_user_data(user_id)
    items, card_ids = cp.convert_to_items(dataset)
    backend = cp.FSRS(parameters=[])
    params, secs = backend.compute_parameters(items, card_ids, num_epochs)
    params = [round(w, 6) for w in params]
    predictor = cp.FSRS(parameters=params)
    return int(user_id), float(predictor.evaluate(items)), len(items), float(secs)


def _eval_at(num_epochs: int, users: list[int]) -> tuple[float, float]:
    """Mean cp LogLoss (unweighted, the band metric) + rough median per-user time (ms) at num_epochs."""
    manager = mp.Manager()
    counter = manager.Value("i", 0)
    lock = manager.Lock()
    losses, times_ms = [], []
    with ProcessPoolExecutor(
        max_workers=cp.config.num_processes,
        initializer=cp._init_worker,
        initargs=(counter, lock),
    ) as ex:
        futs = [ex.submit(_train_eval, u, num_epochs) for u in users]
        for f in as_completed(futs):
            _u, ll, _sz, secs = f.result()
            losses.append(ll)
            times_ms.append(secs * 1000.0)
    return statistics.mean(losses), statistics.median(times_ms)


def main() -> None:
    users = list(cp.USERS_BY_SIZE_DESC)  # the canonical 50 users
    # log-spaced epoch counts, integer, deduped, ascending
    ratio = MAX_EPOCHS / MIN_EPOCHS
    raw = [MIN_EPOCHS * ratio ** (i / (N_STEPS - 1)) for i in range(N_STEPS)]
    epochs = sorted({int(round(x)) for x in raw})
    print(f"sweep epochs ({len(epochs)} of {N_STEPS} after integer-dedup): {epochs}\n")

    results: list[tuple[int, float, float]] = []  # (epochs, mean_loss, median_ms)
    best = None  # (abs_diff, epochs, loss)
    crossed = False
    for ne in epochs:
        ml, t_ms = _eval_at(ne, users)
        diff = abs(ml - TARGET)
        results.append((ne, ml, t_ms))
        flag = ""
        if best is None or diff < best[0]:
            best = (diff, ne, ml)
            flag = "  <- closest so far"
        print(f"epochs={ne:4d}  mean_logloss={ml:.6f}  |diff to {TARGET}|={diff:.6f}  ~time(med)={t_ms:.0f}ms{flag}")
        if ml < TARGET:
            if crossed:  # one confirming step past the crossing -> stop
                print(f"  -> loss below {TARGET} for a 2nd step; early-stop (minimum bracketed)")
                break
            crossed = True

    # report
    print("\n=== EPOCH-BUYBACK SWEEP RESULT ===")
    for ne, ml, t_ms in results:
        print(f"epochs={ne:4d}\tloss={ml:.6f}\t~time(med)={t_ms:.0f}ms")
    base_ms = results[0][2]
    print(f"\nClosest to {TARGET}: epochs={best[1]}  loss={best[2]:.6f}  |diff|={best[0]:.6f}")
    # rough relative slowdown vs 8 epochs (clean numbers come from measure.py)
    for ne, ml, t_ms in results:
        if ne == best[1]:
            print(f"  rough time(epochs={ne}) / time(epochs={MIN_EPOCHS}) = {t_ms / base_ms:.2f}x slower "
                  f"(=> measure cleanly for the README speedup)")

    out = Path("profiling/epoch_sweep")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "sweep.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for ne, ml, t_ms in results:
            f.write(json.dumps({"epochs": ne, "mean_logloss": round(ml, 6),
                                "rough_median_ms": round(t_ms, 1)}) + "\n")
    print(f"\nsaved -> {out / 'sweep.jsonl'}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
