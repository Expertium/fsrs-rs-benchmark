import json
import pathlib
import numpy as np

# The dev model defines the user set; the others are compared over the same users.
DEV_MODEL = "compute_parameters-FSRS-rs-short-secs-recency"
MODELS = [
    DEV_MODEL,
    "FSRS-rs-short-secs-recency (baseline)",
    "FSRS-7-short-secs-recency",
]


def load(model):
    path = pathlib.Path(f"./result/{model}.jsonl")
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


common_users = {r["user"] for r in load(DEV_MODEL)}

for model in MODELS:
    rows = [r for r in load(model) if not common_users or r["user"] in common_users]
    if not rows:
        continue
    log_loss = sum(r["metrics"]["LogLoss"] for r in rows) / len(rows)
    print(f"Model: {model}")
    print(f"  users: {len(rows)}   reviews: {sum(r['size'] for r in rows)}   LogLoss (mean): {log_loss:.4f}")
    times = [r["time_ms"] for r in rows if "time_ms" in r]
    if times:
        print(f"  time (median): {np.median(times):.1f} ms")
    print()
