"""Prove the O(N) convert_to_items is bit-for-bit vs the current O(N^2) one (profiling-only).

OLD (current): per expanded prefix-row, re-parse the whole comma-history string and rebuild
FSRSReview objects -> O(N^2) per card. NEW: parse each card's full review sequence ONCE (from its
longest row), then each prefix item is a SLICE of that shared FSRSReview list -> O(N). Identical
items (same reviews, same global review_th order) => identical training => bit-for-bit.

Checks, per user: (1) item COUNT equal; (2) every item's repr() equal in order (value-identical
reviews); (3) card_ids list equal. Plus trained-params equality on the small users (end-to-end).
"""
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.argv = ["benchmark.py", "--algo", "FSRS-rs", "--short", "--secs", "--recency",
            "--processes", "1", "--max-user-id", "100000"]
import benchmark as bm  # noqa: E402
from fsrs_rs_python import FSRSItem, FSRS  # noqa: E402
from data_loader import UserDataLoader  # noqa: E402

config = bm.config


def _prefix_len(t) -> int:
    if pd.isna(t):
        return 0
    s = str(t).strip()
    if not s:
        return 0
    return s.count(",") + 1  # clean comma-join -> field count = commas + 1


def convert_to_items_v2(df):
    """O(N) twin of bm.convert_to_items."""
    pairs = []  # (review_th, card_id, FSRSItem)
    sdf = df.sort_values(by=["card_id", "review_th"])
    for card_id, group in sdf.groupby("card_id"):
        # group rows are in chronological (review_th) order; histories are cumulative prefixes.
        # The LAST row's full review list covers every earlier row's prefix -> parse once.
        full_reviews = bm.build_reviews(group.iloc[-1], include_current=True)
        n_full = len(full_reviews)
        for _, row in group.iterrows():
            L = _prefix_len(row["t_history"]) + 1  # = len(build_reviews(row, include_current=True))
            assert L <= n_full, f"prefix {L} > full {n_full} (card {card_id})"
            item = FSRSItem(reviews=full_reviews[:L])
            pairs.append((row["review_th"], int(card_id), item))
    pairs.sort(key=lambda p: p[0])
    items = [it for _, _, it in pairs]
    card_ids = [c for _, c, it in pairs]
    return items, card_ids


USERS = [42, 26, 36, 12, 21, 33, 34, 17]  # small -> big
loader = UserDataLoader(config)
backend = FSRS(parameters=[])

print(f"{'user':>5} | {'n_old':>7} {'n_new':>7} | repr_mismatch cid_mismatch | t_old(s) t_new(s) speedup")
all_ok = True
small = {42, 26, 36}
for u in USERS:
    ds = loader.load_user_data(u)
    t0 = time.perf_counter(); items_o, cids_o = bm.convert_to_items(ds); t_old = time.perf_counter() - t0
    t0 = time.perf_counter(); items_n, cids_n = convert_to_items_v2(ds); t_new = time.perf_counter() - t0
    repr_mm = "N/A"
    if len(items_o) == len(items_n):
        repr_mm = sum(1 for a, b in zip(items_o, items_n) if repr(a) != repr(b))
    cid_mm = sum(1 for a, b in zip(cids_o, cids_n) if a != b) if len(cids_o) == len(cids_n) else "LEN"
    ok = (len(items_o) == len(items_n)) and repr_mm == 0 and cid_mm == 0
    all_ok = all_ok and ok
    sp = t_old / t_new if t_new else 0
    print(f"{u:>5} | {len(items_o):>7} {len(items_n):>7} | {str(repr_mm):>13} {str(cid_mm):>12} | "
          f"{t_old:8.2f} {t_new:8.2f} {sp:6.1f}x   {'OK' if ok else 'FAIL'}")
    if u in small:
        p_o = backend.compute_parameters(items_o, cids_o)[0]
        p_n = backend.compute_parameters(items_n, cids_n)[0]
        pdiff = sum(1 for a, b in zip(p_o, p_n) if a != b)
        print(f"        params diff (expect 0): {pdiff}/{len(p_o)}")
        all_ok = all_ok and pdiff == 0

print(f"\n{'ALL BIT-FOR-BIT OK' if all_ok else 'MISMATCH — DO NOT SHIP'}")
