"""Sequential 3k ablations to localize the Rust-vs-CUDA log-loss gap (2026-06-12).

Anchor: champion (9ep no-valid, iter-165, 194-HPs, t_max fix) measured 0.32097 on 3k.
Each ablation = one benchmark.py 3k run at 15 processes (~1.6h):
  A1  champion .pyd + FSRS_NO_OUTLIER=1   -> the fsrs-rs internal outlier filter's effect
                                             (CUDA does not filter for --secs)
  A2  a2_8ep_sel.pyd                      -> 8 epochs + best-epoch selection at 3k
                                             (the era-iter-1 change, validated only at 50u)
  A3  a3_oldhp.pyd                        -> pre-194 HPs (LR 0.0188 / L2 0.3333 / B2 0.9913)

Per-user results snapshot to profiling/ablation/<name>.jsonl; running summary to
profiling/ablation/summary.json after each run (crash-safe). Restores the champion
.pyd at the end. Run detached; profiling-only (constraint 10).
"""
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ABL = REPO / "profiling" / "ablation"
PYD = REPO / "fsrs_rs_python" / "target" / "release" / "fsrs_rs_python.cp312-win_amd64.pyd"
PYD_DEPS = REPO / "fsrs_rs_python" / "target" / "release" / "deps" / "fsrs_rs_python.cp312-win_amd64.pyd"
BM_RESULT = REPO / "result" / "FSRS-rs-short-secs-recency.jsonl"
SUMMARY = ABL / "summary.json"

RUNS = [
    {"name": "a1_no_outlier", "pyd": "champ.pyd", "env": {"FSRS_NO_OUTLIER": "1"}},
    {"name": "a2_8ep_sel", "pyd": "a2_8ep_sel.pyd", "env": {}},
    {"name": "a3_oldhp", "pyd": "a3_oldhp.pyd", "env": {}},
]


def install(pyd_name: str) -> None:
    src = ABL / pyd_name
    shutil.copyfile(src, PYD)
    shutil.copyfile(src, PYD_DEPS)


def bench3k(extra_env: dict) -> list[dict]:
    if BM_RESULT.exists():
        BM_RESULT.unlink()
    env = {**os.environ, **extra_env}
    cmd = [sys.executable, "benchmark.py", "--algo", "FSRS-rs", "--short", "--secs",
           "--recency", "--processes", "15", "--max-user-id", "3000"]
    proc = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"benchmark exited {proc.returncode}\n{proc.stderr[-1500:]}")
    return [json.loads(l) for l in BM_RESULT.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    summary = {"anchor_champion_3k": 0.32097, "runs": {}}
    try:
        for run in RUNS:
            t0 = time.time()
            print(f"[abl] {run['name']}: installing {run['pyd']}, env={run['env']}", flush=True)
            install(run["pyd"])
            rows = bench3k(run["env"])
            mean_ll = statistics.mean(r["metrics"]["LogLoss"] for r in rows)
            secs = sum(r.get("time_ms", 0.0) for r in rows) / 1000.0
            shutil.copyfile(BM_RESULT, ABL / f"{run['name']}.jsonl")
            summary["runs"][run["name"]] = {
                "mean_logloss": round(mean_ll, 6), "n_users": len(rows),
                "train_seconds": round(secs, 1), "wall_s": round(time.time() - t0),
                "delta_vs_champion": round(mean_ll - 0.32097, 6),
            }
            SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"[abl] {run['name']}: mean LL={mean_ll:.6f} "
                  f"(delta {mean_ll - 0.32097:+.6f})  [{time.time() - t0:.0f}s]", flush=True)
    finally:
        install("champ.pyd")
        print("[abl] champion .pyd restored", flush=True)
    print("[abl] DONE", flush=True)


if __name__ == "__main__":
    main()
