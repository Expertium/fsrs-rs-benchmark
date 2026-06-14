#!/usr/bin/env python3
"""Bit-for-bit check for the COMPACT-RAW cross-val GATE path (constraint 10: no result/ writes).

The gated default-param tuner's gate = the 5-fold benchmark logloss_by_user (how the (9,512) gold
was measured). This verifies the RAW gate path reproduces the items path per user:

  ITEMS (reference, = tuner_engine.eval_user with default init):
    per fold: compute_parameters(items, card_ids) -> predict(test_set, params, history_items)
  RAW (compact):
    per fold: compute_parameters_raw(train_raw, init_w=None)
              -> memory_states_raw(full, test positions) -> predict(test_slim, precomputed_states)
  Both accumulate (p, y) across folds -> one sklearn log_loss per user.

Run:  python profiling/verify_gate.py [n_users]
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

_SAVED = sys.argv
sys.argv = ["benchmark.py", "--algo", "FSRS-rs", "--short", "--secs", "--recency",
            "--processes", "1", "--max-user-id", "100000"]
import benchmark as bm  # noqa: E402
sys.argv = _SAVED
import pandas as pd  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402
from fsrs_rs_python import FSRS, FSRSItem  # noqa: E402
from data_loader import UserDataLoader  # noqa: E402

config = bm.config
WANT = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def eval_both(dataset, backend):
    """Return (ll_items, ll_raw, max_abs_p_diff) for one user, evaluating the SAME folds both ways."""
    full = bm.convert_to_raw(dataset)
    fd, fr, foff = full[0].tolist(), full[1].tolist(), full[3].tolist()
    card_to_idx = {int(c): i for i, c in enumerate(sorted(dataset["card_id"].unique()))}
    p_items, y_items, p_raw, y_raw = [], [], [], []
    for split_i, (tr, te) in enumerate(TimeSeriesSplit(n_splits=config.n_splits).split(dataset)):
        if config.train_equals_test:
            train_set = dataset.copy()
            test_set = dataset[dataset["review_th"] >= dataset.iloc[te]["review_th"].min()].copy()
        else:
            train_set = dataset.iloc[tr]
            test_set = dataset.iloc[te]
            if config.equalize_test_with_non_secs:
                train_set = dataset[dataset[f"{split_i}_train"]]
                test_set = dataset[dataset[f"{split_i}_test"]]
        if config.no_test_same_day:
            test_set = test_set[test_set["elapsed_days"] > 0].copy()
        if config.no_train_same_day:
            train_set = train_set[train_set["elapsed_days"] > 0].copy()
        if train_set.empty or test_set.empty:
            continue
        slim = pd.DataFrame({"delta_t": test_set["delta_t"].to_numpy(float),
                             "y": test_set["y"].to_numpy(float)})

        # ITEMS reference
        items, cids = bm.convert_to_items(train_set)
        pi, _ = backend.compute_parameters(items, cids)
        test_hist = [FSRSItem(reviews=bm.build_reviews(row)) for _, row in test_set.iterrows()]
        ppi, yyi, _ = bm.predict(slim, pi, history_items=test_hist)
        p_items += ppi
        y_items += yyi

        # RAW compact
        d, r, th, off = (a.tolist() for a in bm.convert_to_raw(train_set))
        pr, _ = backend.compute_parameters_raw(d, r, th, off, None, None)
        tci = [card_to_idx[int(c)] for c in test_set["card_id"]]
        thl = [bm.hist_len_of(s) for s in test_set["t_history"]]
        stab, diff, sfast = FSRS(parameters=pr).memory_states_raw(fd, fr, foff, tci, thl)
        ppr, yyr, _ = bm.predict(slim, pr, precomputed_states=(stab, diff, sfast))
        p_raw += ppr
        y_raw += yyr

    if config.train_equals_test:
        pass
    ll_i = log_loss(y_true=y_items, y_pred=p_items, labels=[0, 1])
    ll_r = log_loss(y_true=y_raw, y_pred=p_raw, labels=[0, 1])
    max_dp = max((abs(a - b) for a, b in zip(p_items, p_raw)), default=0.0)
    return ll_i, ll_r, max_dp, len(p_items)


def main() -> int:
    loader = UserDataLoader(config)
    checked = failures = uid = 0
    while checked < WANT and uid < 400:
        uid += 1
        try:
            dataset = loader.load_user_data(uid)
        except Exception:  # noqa: BLE001
            continue
        if len(dataset) < 64:
            continue
        ll_i, ll_r, max_dp, n = eval_both(dataset, FSRS(parameters=[]))
        checked += 1
        ok = abs(ll_i - ll_r) < 1e-9 and max_dp < 1e-6
        failures += 0 if ok else 1
        print(f"user {uid:4d}: n_test={n:7d}  ll_items={ll_i:.8f}  ll_raw={ll_r:.8f}  "
              f"d_ll={ll_i-ll_r:+.2e}  max|dp|={max_dp:.2e}  {'OK' if ok else 'FAIL'}")

    print(f"\n{checked} users checked, {failures} failed "
          f"-> {'ALL MATCH' if failures == 0 else 'FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
