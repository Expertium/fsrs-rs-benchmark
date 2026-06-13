"""Live plot watcher for the detached hp-tune grid (profiling-only).

The grid process checkpoints result/hp_grid.json after every cell. This polls that
file's mtime and re-renders result/hp_grid_plot.png (via hp_tune.plot_grid) whenever
it changes — so the Speed-vs-LogLoss frontier fills in live while the sweep runs,
WITHOUT touching the grid process (useful when the grid is an already-running detached
process built from older code that didn't plot per-cell). Exits once the grid marks
hp_grid.json non-partial (sweep finished) or after a long idle. Safe to kill anytime;
never writes a champion / result record (constraint 10).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import hp_tune  # noqa: E402

GRID_JSON = hp_tune.GRID_JSON
POLL_SECS = 60
IDLE_LIMIT = 240  # ~4h of no change (>> the ~1.5h/cell cadence) -> assume done/dead

last_mtime = None
idle = 0
print(f"[plot-watcher] watching {GRID_JSON.name} every {POLL_SECS}s", flush=True)
while True:
    try:
        mtime = GRID_JSON.stat().st_mtime
    except FileNotFoundError:
        mtime = None
    if mtime is not None and mtime != last_mtime:
        last_mtime = mtime
        idle = 0
        try:
            data = json.loads(GRID_JSON.read_text(encoding="utf-8"))
            hp_tune.plot_grid()
            if not data.get("partial"):
                print("[plot-watcher] grid finished — final plot rendered, exiting.", flush=True)
                break
        except Exception as e:  # noqa: BLE001 — partial mid-write read, etc.; retry next poll
            print(f"[plot-watcher] skip ({e})", flush=True)
            last_mtime = None  # force a re-render once the write settles
    else:
        idle += 1
        if idle >= IDLE_LIMIT:
            print("[plot-watcher] no change for ~4h, exiting.", flush=True)
            break
    time.sleep(POLL_SECS)
