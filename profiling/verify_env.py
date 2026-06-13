"""Verify the new FSRS_BETA1/2 + FSRS_RECENCY_C0/EXP env overrides (profiling-only).

Two things at once, in ONE process:
  (1) DEFAULT CORRECTNESS: training with no env, and with the env set to the shipped defaults
      (0.55 / 0.9913 / 0.0667 / 11.25), must produce BYTE-IDENTICAL params -> the override code
      defaults exactly to the old consts (so production stays bit-for-bit).
  (2) LIVE-ENV MECHANISM: changing os.environ BETWEEN compute_parameters() calls in the SAME
      process must change the trained params -> Rust std::env::var reads the live env per call.
      This is the mechanism the persistent in-memory tuner will rely on (no subprocess per trial).
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.argv = ["compute_parameters.py", "--algo", "FSRS-rs", "--short", "--secs", "--recency",
            "--processes", "1", "--max-user-id", "100000"]

import compute_parameters as cp  # noqa: E402  (builds global config from sys.argv)
from fsrs_rs_python import FSRS  # noqa: E402
from data_loader import UserDataLoader  # noqa: E402

USERS = [42, 26, 36]
backend = FSRS(parameters=[])  # mirror compute_parameters.process()'s backend

def _set(env: dict):
    for k in ("FSRS_BETA1", "FSRS_BETA2", "FSRS_RECENCY_C0", "FSRS_RECENCY_EXP"):
        os.environ.pop(k, None)
    for k, v in env.items():
        os.environ[k] = str(v)

def train(items, card_ids):
    return backend.compute_parameters(items, card_ids)[0]

loader = UserDataLoader(cp.config)
data = {}
for u in USERS:
    ds = loader.load_user_data(u)
    data[u] = cp.convert_to_items(ds)

def diff(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)

print("user | noenv-vs-defaultenv (expect 0) | noenv-vs-beta1=0.70 (expect >0) | noenv-vs-C0=0.20 (expect >0)")
for u in USERS:
    items, cids = data[u]
    _set({})
    base = train(items, cids)
    _set({"FSRS_BETA1": 0.55, "FSRS_BETA2": 0.9913, "FSRS_RECENCY_C0": 0.0667, "FSRS_RECENCY_EXP": 11.25})
    deflt = train(items, cids)
    _set({"FSRS_BETA1": 0.70})
    b70 = train(items, cids)
    _set({"FSRS_RECENCY_C0": 0.20})
    c20 = train(items, cids)
    _set({})
    print(f"{u:>4} | {diff(base, deflt):>30} | {diff(base, b70):>31} | {diff(base, c20):>28}")
print("\nbase params[:6] :", [round(x, 5) for x in base[:6]])
