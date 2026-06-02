"""Profiling-only driver: load ONE user and call compute_parameters() in a loop
so a sampling profiler (py-spy) has time to collect native Rust frames. Writes
nothing to result/ and is never imported by the timed path (constraint 10).

Usage:  python profiling/_profile_user.py [user_id=17] [reps=3]
Profile: py-spy record --native --rate 200 --format raw -o profiling/stacks.txt \
             -- python profiling/_profile_user.py 17 3
"""
import os
import sys
import time

REPO = r"C:\Users\Andrew\fsrs-rs-speed-autoresearch"

# Read our args BEFORE we overwrite sys.argv for compute_parameters' module-level parser.
_a = sys.argv[1:]
USER = int(_a[0]) if _a else 17
REPS = int(_a[1]) if len(_a) > 1 else 3

os.chdir(REPO)
if sys.path[0] != REPO:
    sys.path.insert(0, REPO)
# Make compute_parameters build the canonical config (--short --secs --recency) so the
# preprocessing matches the real harness, then reuse its loader/converter/backend.
sys.argv = ["compute_parameters.py", "--algo", "FSRS-rs", "--short", "--secs",
            "--recency", "--max-user-id", "50"]
import compute_parameters as cp  # noqa: E402

dataset = cp.UserDataLoader(cp.config).load_user_data(USER)
items, card_ids = cp.convert_to_items(dataset)
print(f"user {USER}: {len(items)} items; calling compute_parameters() x{REPS}", flush=True)

backend = cp.FSRS(parameters=[])
for i in range(REPS):
    t = time.monotonic()
    # Pass card_ids so we profile the windowed (O(N)) champion path, not the plain fallback.
    params, secs = backend.compute_parameters(items, card_ids)
    print(f"  rep {i}: rust={secs*1000:.0f}ms wall={(time.monotonic()-t)*1000:.0f}ms", flush=True)
