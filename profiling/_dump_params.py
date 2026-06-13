"""Train USERS with no env override; dump full params per user to argv[1] (profiling-only)."""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.argv2 = sys.argv[1]
sys.argv = ["compute_parameters.py", "--algo", "FSRS-rs", "--short", "--secs", "--recency",
            "--processes", "1", "--max-user-id", "100000"]
import compute_parameters as cp  # noqa: E402
from fsrs_rs_python import FSRS  # noqa: E402
from data_loader import UserDataLoader  # noqa: E402

for k in ("FSRS_BETA1", "FSRS_BETA2", "FSRS_RECENCY_C0", "FSRS_RECENCY_EXP",
          "FSRS_N_EPOCHS", "FSRS_BATCH_SIZE", "FSRS_LR"):
    os.environ.pop(k, None)

backend = FSRS(parameters=[])
loader = UserDataLoader(cp.config)
out = {}
for u in [42, 26, 36]:
    items, cids = cp.convert_to_items(loader.load_user_data(u))
    out[str(u)] = backend.compute_parameters(items, cids)[0]
Path(sys.argv2).write_text(json.dumps(out), encoding="utf-8")
print(f"wrote {sys.argv2}")
