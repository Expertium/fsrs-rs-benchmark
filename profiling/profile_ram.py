"""Profiling-only: measure ONE worker's peak working-set (RAM) while it runs
compute_parameters() on a single user. Dependency-free (ctypes + Win32
GetProcessMemoryInfo); writes nothing to result/ and is never imported by the
timed path (constraint 10). Profiling-only per constraint 10.

Peak working set (PeakWorkingSetSize) is a monotonic high-water mark since
process start, so we snapshot it AFTER data load (P0 — captures the Python
interpreter + parquet-load transients) and again AFTER training (P1). The
*marginal* training footprint above the load baseline is P1 - P0, which is the
number the drop(train_dataset)/drop(dataloader_valid) change should shrink.

Usage:  python profiling/profile_ram.py [user_id=34] [reps=3]
        (user 34 = largest collection = worst-case worker RAM)
"""
import ctypes
import gc
import os
import sys
import time
from ctypes import wintypes

REPO = r"C:\Users\Andrew\fsrs-rs-speed-autoresearch"

_a = sys.argv[1:]
USER = int(_a[0]) if _a else 34
REPS = int(_a[1]) if len(_a) > 1 else 3


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


_psapi = ctypes.WinDLL("psapi", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_GetProcessMemoryInfo = _psapi.GetProcessMemoryInfo
_GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
_GetProcessMemoryInfo.restype = wintypes.BOOL


def _mem() -> "tuple[float, float]":
    """Return (current_working_set_MB, peak_working_set_MB) for this process."""
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    h = _kernel32.GetCurrentProcess()
    if not _GetProcessMemoryInfo(h, ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    mb = 1024.0 * 1024.0
    return counters.WorkingSetSize / mb, counters.PeakWorkingSetSize / mb


os.chdir(REPO)
if sys.path[0] != REPO:
    sys.path.insert(0, REPO)
sys.argv = ["compute_parameters.py", "--algo", "FSRS-rs", "--short", "--secs",
            "--recency", "--max-user-id", "50"]
import compute_parameters as cp  # noqa: E402

dataset = cp.UserDataLoader(cp.config).load_user_data(USER)
items = cp.convert_to_items(dataset)
backend = cp.FSRS(parameters=[])

gc.collect()
ws0, p0 = _mem()
print(f"user {USER}: {len(items)} items", flush=True)
print(f"  post-load:   working_set={ws0:8.1f} MB   peak={p0:8.1f} MB", flush=True)

for i in range(REPS):
    t = time.monotonic()
    params, secs = backend.compute_parameters(items)
    ws, pk = _mem()
    print(f"  rep {i}: rust={secs*1000:5.0f}ms  working_set={ws:8.1f} MB  peak={pk:8.1f} MB", flush=True)

ws1, p1 = _mem()
print(f"  post-train:  working_set={ws1:8.1f} MB   peak={p1:8.1f} MB", flush=True)
print(f"  >>> training marginal peak above load baseline (P1-P0) = {p1 - p0:8.1f} MB", flush=True)
print(f"  >>> absolute peak working set (P1)                     = {p1:8.1f} MB", flush=True)
