"""Profiling-only: characterize what compute_parameters() actually chews on, and
find which structural quantity predicts its (already-measured) runtime.

For each of the 50 canonical users we reproduce the exact preprocessing
(UserDataLoader -> the same df compute_parameters converts to FSRSItems) and, per
row/item, take seq_len = (#history entries) + 1 (the current review) -- i.e. the
length of the FSRSItem the Rust trainer receives. We never run/clobber the timed
harness: runtimes are read from the existing result .jsonl (constraint 10).

Work proxies per user:
  n_items      = #FSRSItems  (== jsonl "size"; asserted)
  total_work   = sum of seq_len            (the uncapped O(N^2) expanding window)
  capped_work  = sum of min(seq_len, 64)   (what training sees: Rust caps at max_seq_len)
Then regress measured time_ms against each to see the dominant cost driver.

Run:  python profiling/profile_structure.py        (from the repo root)
Out:  prints a report; caches per-user rows to profiling/structure.json
"""
import json
import os
import sys
import time
from pathlib import Path

REPO = r"C:\Users\Andrew\fsrs-rs-speed-autoresearch"
os.chdir(REPO)
if sys.path[0] != REPO:
    sys.path.insert(0, REPO)

import numpy as np
from config import create_parser, Config
from data_loader import UserDataLoader

MAXLEN = 64  # = config.max_seq_len; the Rust trainer truncates sequences to this
RESULT = Path(REPO) / "result" / "compute_parameters-FSRS-rs-short-secs-recency.jsonl"


def load_times():
    times, sizes = {}, {}
    for line in open(RESULT, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            times[r["user"]] = r["time_ms"]
            sizes[r["user"]] = r["size"]
    return times, sizes


def field_count(s) -> int:
    """#comma-separated history entries in a t_history cell (0 if empty/NaN)."""
    if not isinstance(s, str) or not s:
        return 0
    return s.count(",") + 1


def linfit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    A = np.vstack([x, np.ones_like(x)]).T
    (m, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    yp = m * x + b
    r2 = 1 - ((y - yp) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return m, b, r2


def main():
    parser = create_parser()
    args = parser.parse_args(
        ["--algo", "FSRS-rs", "--short", "--secs", "--recency", "--max-user-id", "50"]
    )
    config = Config(args)
    config.partitions = "none"
    loader = UserDataLoader(config)
    times, sizes = load_times()

    rows = []
    t0 = time.monotonic()
    for u in range(1, 51):
        df = loader.load_user_data(u)
        col = "t_history" if "t_history" in df.columns else "t_history_secs"
        seq = df[col].map(field_count).to_numpy() + 1  # +1 for the current review
        capped = np.minimum(seq, MAXLEN)
        rows.append({
            "user": u,
            "n_items": int(seq.size),
            "jsonl_size": int(sizes[u]),
            "n_cards": int(df["card_id"].nunique()),
            "total_work": int(seq.sum()),
            "capped_work": int(capped.sum()),
            "max_seq": int(seq.max()),
            "mean_seq": round(float(seq.mean()), 2),
            "pct_items_over_cap": round(100 * float((seq > MAXLEN).mean()), 1),
            "time_ms": int(times[u]),
        })
        print(f"  user {u:>2}: items={seq.size:>6} cards={df['card_id'].nunique():>5} "
              f"mean_seq={seq.mean():>5.1f} max={seq.max():>4} t={times[u]:>6}ms")
    print(f"loaded 50 users in {time.monotonic() - t0:.0f}s")

    bad = [(r["user"], r["n_items"], r["jsonl_size"]) for r in rows if r["n_items"] != r["jsonl_size"]]
    print(f"\nVALIDATION n_items==jsonl size: {'OK' if not bad else f'MISMATCH {bad[:5]}'}")

    tot_items = sum(r["n_items"] for r in rows)
    tot_work = sum(r["total_work"] for r in rows)
    tot_capped = sum(r["capped_work"] for r in rows)
    print(f"\ntotals: items={tot_items:,}  total_work={tot_work:,}  capped_work={tot_capped:,}")
    print(f"  64-cap removes {100*(1-tot_capped/tot_work):.1f}% of raw expanding-window work")
    print(f"  mean items/card across users = {np.mean([r['n_items']/r['n_cards'] for r in rows]):.2f}")
    print(f"  mean seq_len (work/item) = {tot_capped/tot_items:.2f} timesteps")

    print("\n=== which structural quantity predicts time_ms? (single-predictor linear fit) ===")
    for key in ["n_items", "n_cards", "total_work", "capped_work"]:
        m, b, r2 = linfit([r[key] for r in rows], [r["time_ms"] for r in rows])
        print(f"  time ~ {key:<12}: R2={r2:.4f}  slope={m:.4g}  intercept={b:.4g}")

    tp = np.array([r["time_ms"] / r["capped_work"] for r in rows])
    print(f"\nthroughput (ms per capped-work-unit): mean={tp.mean():.3e}  "
          f"CV={100*tp.std()/tp.mean():.1f}%  (low CV => time is ~linear in capped_work)")

    out = Path(REPO) / "profiling" / "structure.json"
    out.write_text(json.dumps(rows, indent=0), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
