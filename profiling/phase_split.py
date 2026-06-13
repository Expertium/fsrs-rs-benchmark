"""One-off diagnostic: where does benchmark.py's per-user WALL time go?

Replicates the hp-tune grid's exact config (FSRS-rs --short --secs --recency) and
times the three per-user phases separately on a small user sample, SINGLE-process so
it barely perturbs the running 15-proc grid:
  * LOAD  = load_user_data (read parquet + create_features expanding-window build)
  * TRAIN = Rust compute_parameters() (the binding-timed region we actually optimize)
  * SCORE = predict() forgetting curve + evaluate() (depends on trained params)
Profiling-only (constraint 10): never writes result/.  Usage: python profiling/phase_split.py [N_USERS]
"""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
# Build the global Config exactly like the grid's benchmark.py invocation.
sys.argv = ["benchmark.py", "--algo", "FSRS-rs", "--short", "--secs", "--recency",
            "--processes", "1", "--max-user-id", "100000"]

import pyarrow.parquet as pq  # noqa: E402
from data_loader import UserDataLoader  # noqa: E402
import benchmark as bm  # noqa: E402
config = bm.config  # benchmark.py builds the global Config from sys.argv at import

N = 30

# Pick the first N users that exist (smallest ids).
dataset = pq.ParquetDataset(config.data_path / "revlogs")
ids = sorted(u.as_py() for u in dataset.partitioning.dictionaries[0])[:N]

t_load = t_rust = t_itembuild = t_score = 0.0
n_ok = 0
loader = UserDataLoader(config)
for uid in ids:
    try:
        t0 = time.perf_counter()
        ds = loader.load_user_data(uid)
        t1 = time.perf_counter()
        # Re-implement process()'s fold loop but time train vs score separately.
        from sklearn.model_selection import TimeSeriesSplit
        w_list, testsets = [], []
        rust_secs = 0.0    # binding's monotonic Rust-region clock (the grid's "train" axis)
        wall_secs = 0.0    # full train() call incl. Python FSRSItem marshaling
        for split_i, (tr_idx, te_idx) in enumerate(TimeSeriesSplit(n_splits=config.n_splits).split(ds)):
            train_set = ds.iloc[tr_idx]
            test_set = ds.iloc[te_idx]
            if config.no_test_same_day:
                test_set = test_set[test_set["elapsed_days"] > 0].copy()
            if train_set.empty or test_set.empty:
                continue
            ta = time.perf_counter()
            params, secs = bm.train(train_set)
            tb = time.perf_counter()
            wall_secs += (tb - ta)
            rust_secs += secs
            testsets.append(test_set)
            w_list.append(params)
        t2 = time.perf_counter()
        p, y, preds = [], [], []
        for w, ts in zip(w_list, testsets):
            pp, yy, tsp = bm.predict(ts, w)
            p.extend(pp); y.extend(yy); preds.append(tsp)
        import pandas as pd
        save_tmp = pd.concat(preds)
        if "tensor" in save_tmp:
            del save_tmp["tensor"]
        bm.evaluate(y, p, save_tmp, config.get_evaluation_file_name(), uid, config, w_list)
        t3 = time.perf_counter()

        t_load += (t1 - t0)
        t_rust += rust_secs
        t_itembuild += (wall_secs - rust_secs)
        t_score += (t3 - t2)
        n_ok += 1
    except Exception as e:  # noqa: BLE001
        continue

tot = t_load + t_rust + t_itembuild + t_score
py = t_load + t_itembuild + t_score
print(f"\n{n_ok} users timed (single-process):")
print(f"  LOAD  (parquet + create_features) : {t_load:8.2f}s  {100*t_load/tot:5.1f}%   {1000*t_load/n_ok:7.1f} ms/user")
print(f"  ITEMBUILD (Python FSRSItem marshal): {t_itembuild:8.2f}s  {100*t_itembuild/tot:5.1f}%   {1000*t_itembuild/n_ok:7.1f} ms/user")
print(f"  RUST  (compute_parameters region) : {t_rust:8.2f}s  {100*t_rust/tot:5.1f}%   {1000*t_rust/n_ok:7.1f} ms/user")
print(f"  SCORE (predict + evaluate)        : {t_score:8.2f}s  {100*t_score/tot:5.1f}%   {1000*t_score/n_ok:7.1f} ms/user")
print(f"  TOTAL                             : {tot:8.2f}s          {1000*tot/n_ok:7.1f} ms/user")
print(f"  --> PYTHON (load+itembuild+score) : {py:8.2f}s  {100*py/tot:5.1f}%   = the non-Rust wall")
