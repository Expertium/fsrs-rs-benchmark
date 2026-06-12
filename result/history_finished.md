# FSRS-rs speed autoresearch — finished-model era history

Accept metric: **median per-user speed_ratio ≥ 1.05** (constraint 12) AND **speed_ratio ≥ complexity_ratio^2.5** (constraint 13), with the cross-val mean log loss (benchmark.py, 50 users) within ±0.0015 of the era baseline. speed_ratio is the *median of per-user ratios* of the summed-over-folds Rust benchmark() time, measured back-to-back vs the then-current champion. Times are machine/session-specific (informational).

| iter | time_before (ms) | time_after (ms) | speed_ratio | cplx_before | cplx_after | cplx_ratio | cplx^2.5 | checks | status | summary |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 218 | 218 | 1.000 | 7985 | 7985 | 1.000 | 1.000 | ✓ | accepted | Baseline: finished FSRS-7 port; 8 epochs with best-epoch selection |
| 1 | 218 | 161 | 1.343 | 7985 | 7930 | 0.993 | 0.983 | ✓ | accepted | Drop per-epoch validation + best-epoch selection; train 9 epochs, ship last-epoch params |
| 2 | 161 | 161 | 1.000 | 7930 | 7853 | 0.990 | 0.976 | ✓ | accepted | Drop the fsrs-rs outlier filter; train on all items (CUDA parity) |
| 3 | 161 | 161 | 1.000 | 7853 | 7774 | 0.990 | 0.975 | ✓ | accepted | Revert LR/L2/beta2 to pre-194 values 0.0188 0.3333 0.9913 |

**Cumulative speed_ratio (product of accepted): ×1.343** — upward-biased (winner's curse); anchor periodically vs iter-0 baseline.
