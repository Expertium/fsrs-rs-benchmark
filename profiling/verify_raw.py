#!/usr/bin/env python3
"""Bit-for-bit checks for the COMPACT-RAW bindings (constraint 10: never writes a result/ record).

  1. compute_parameters_raw(flat arrays)   == compute_parameters(convert_to_items(df))   [params]
  2. evaluate_raw(flat arrays)              == evaluate(convert_to_items(df))             [logloss]
  3. compute_parameters_raw(init_w=DEFAULT) == compute_parameters_raw(init_w=None)        [params]
     (a non-empty custom init must be bit-for-bit with None when it IS the default)
  4. compute_parameters_raw(init_w=perturbed) != the default-init result                  [sanity:
     init_w actually threads through to the SGD start + L2 anchor]

Run:  python profiling/verify_raw.py [n_users]
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
from fsrs_rs_python import FSRS, DEFAULT_PARAMETERS  # noqa: E402
from data_loader import UserDataLoader  # noqa: E402

config = bm.config
WANT = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def train_set_for(dataset):
    if getattr(config, "train_equals_test", False):
        return dataset.copy()
    return dataset


def main() -> int:
    loader = UserDataLoader(config)
    default = list(DEFAULT_PARAMETERS)
    perturbed = default.copy()
    perturbed[4] *= 1.1  # nudge an initial-difficulty weight so the SGD start truly differs
    checked = 0
    failures = 0
    uid = 0
    while checked < WANT and uid < 400:
        uid += 1
        try:
            dataset = loader.load_user_data(uid)
        except Exception:  # noqa: BLE001
            continue
        df = train_set_for(dataset)
        items, card_ids = bm.convert_to_items(df)
        if len(items) < 64:
            continue
        raw = bm.convert_to_raw(df)
        d, r, th, off = (a.tolist() for a in raw)

        train = FSRS(parameters=[])
        pa, _ = train.compute_parameters(items, card_ids)
        pb, _ = train.compute_parameters_raw(d, r, th, off)
        pdef, _ = train.compute_parameters_raw(d, r, th, off, None, default)
        pper, _ = train.compute_parameters_raw(d, r, th, off, None, perturbed)

        scorer = FSRS(parameters=default)
        ll_items = scorer.evaluate(items)
        ll_raw = scorer.evaluate_raw(d, r, th, off)

        cp_ok = len(pa) == len(pb) and all(a == b for a, b in zip(pa, pb))
        eval_ok = ll_items == ll_raw
        initdef_ok = len(pb) == len(pdef) and all(a == b for a, b in zip(pb, pdef))
        initpert_ok = any(a != b for a, b in zip(pb, pper))  # perturbed init MUST move the result

        checked += 1
        ok = cp_ok and eval_ok and initdef_ok and initpert_ok
        failures += 0 if ok else 1
        print(f"user {uid:4d}: items={len(items):7d}  "
              f"cp_raw={'OK' if cp_ok else 'FAIL'}  "
              f"eval_raw={'OK' if eval_ok else 'FAIL'}  "
              f"init=DEF=={'OK' if initdef_ok else 'FAIL'}  "
              f"init!=perturb={'OK' if initpert_ok else 'FAIL'}")
        if not cp_ok:
            print(f"    cp max|d|={max(abs(a-b) for a, b in zip(pa, pb)):.3e}")
        if not eval_ok:
            print(f"    eval items={ll_items!r} raw={ll_raw!r} d={ll_items-ll_raw:.3e}")
        if not initdef_ok:
            print(f"    init=DEF max|d|={max(abs(a-b) for a, b in zip(pb, pdef)):.3e}")

    print(f"\n{checked} users checked, {failures} failed "
          f"-> {'ALL GOOD' if failures == 0 else 'FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
