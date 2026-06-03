# FSRS-rs speed autoresearch — benchmark() iteration history (Phase 2)

Accept metric: **median per-user speed_ratio ≥ 1.05** (constraint 12) AND **speed_ratio ≥ complexity_ratio^2.5** (constraint 13). Phase 2 optimizes benchmark() (benchmark.py's O(N^2) per-prefix anchor path); only **BIT-FOR-BIT** changes are viable (its mean log loss is ~5e-5 under the band ceiling). speed_ratio is the *median of per-user ratios* of the summed-over-folds Rust benchmark() time, measured back-to-back vs the then-current champion. Times are machine/session-specific (informational).

| iter | time_before (ms) | time_after (ms) | speed_ratio | cplx_before | cplx_after | cplx_ratio | cplx^2.5 | checks | status | summary |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 278 | 278 | 1.000 | 7278 | 7278 | 1.000 | 1.000 | ✓ | accepted | baseline: benchmark() O(N^2) per-prefix path, inherits Phase-1 shared kernels |
| 1 | 278 | 227 | 1.160 | 7278 | 7296 | 1.002 | 1.006 | ✓ | accepted | skip dead BCE loss + per-group padding skip in O(N^2) kernels; bit-for-bit |

**Cumulative speed_ratio (product of accepted): ×1.160** — upward-biased (winner's curse); anchor periodically vs iter-0 baseline.
