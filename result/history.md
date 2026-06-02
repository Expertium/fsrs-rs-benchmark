# FSRS-rs speed autoresearch — iteration history

Accept metric: **median per-user speed_ratio ≥ 1.05** (constraint 12) AND **speed_ratio ≥ complexity_ratio^2.5** (constraint 13). speed_ratio is the *median of per-user ratios*, measured back-to-back vs the then-current champion. Times are machine/session-specific (informational).

| iter | time_before (ms) | time_after (ms) | speed_ratio | cplx_before | cplx_after | cplx_ratio | cplx^2.5 | checks | status | summary |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 3067 | 3067 | 1.000 | 4771 | 4771 | 1.000 | 1.000 | ✓ | accepted | baseline: dual-trace FSRS-7 Rust port (iter-66 champion) |
| 1 | 2916 | 2884 | 1.035 | 4771 | 4703 | 0.986 | 0.965 | ✗ | rejected | hoist loop-invariant weight slices; remove single-version dispatch |
| 2 | 3012 | 1368 | 2.166 | 4771 | 5620 | 1.178 | 1.506 | ✓ | accepted | hand-written analytic BCE gradient replaces burn autodiff forward+backward in training |
| 3 | 1370 | 1274 | 1.077 | 5620 | 5605 | 0.997 | 0.993 | ✓ | accepted | parallelize analytic gradient across 2 threads; remove dead autodiff grad helpers |
| 4 | 1292 | 1295 | 0.990 | 5605 | 5605 | 1.000 | 1.000 | ✓ | rejected | validation pass via threaded manual forward instead of burn autodiff forward |
| 5 | 1276 | 1064 | 1.198 | 5605 | 5600 | 0.999 | 0.998 | ✓ | accepted | replace backward powf with division by cached forward values (b^(e-1)=b^e/b) |
| 6 | 1065 | 963 | 1.107 | 5600 | 5601 | 1.000 | 1.000 | ✓ | accepted | score validation each epoch with analytic forward over pre-extracted batches |
| 7 | 962 | 1025 | 0.944 | 5601 | 5635 | 1.006 | 1.015 | ✓ | rejected | thread the analytic validation forward across 2 threads (THREAD_MIN split) |
| 8 | 968 | 855 | 1.121 | 5602 | 5621 | 1.003 | 1.008 | ✓ | accepted | rewrite 12 forward powf as exp(e*ln), cache the ln, reuse in backward |
| 9 | 856 | 793 | 1.078 | 5621 | 5628 | 1.001 | 1.003 | ✓ | accepted | pre-extract training batches once; replicate dataloader shuffle to drive epoch order |
| 10 | 792 | 738 | 1.074 | 5628 | 5651 | 1.004 | 1.010 | ✓ | accepted | hoist loop-invariant weight constants (ln w27/w28, aa, exp 3w5) out of per-timestep loop |
| 11 | 746 | 716 | 1.039 | 5651 | 5633 | 0.997 | 0.992 | ✓ | rejected | drop dead train/valid tensor datasets + remove dummy autodiff backward (bit-for-bit, -complexity) |
| 12 | 738 | 580 | 1.292 | 5651 | 5515 | 0.976 | 0.941 | ✓ | accepted | build train/valid host batches directly (skip burn-tensor floor); remove dummy backward + dead dataloader code |
| 13 | 577 | 538 | 1.075 | 5515 | 5522 | 1.001 | 1.003 | ✓ | accepted | share ln(last_s/last_sf/last_d) across curve + both stability traces; 3 fewer ln per timestep (bit-for-bit) |
| 14 | 534 | 414 | 1.305 | 5522 | 5884 | 1.066 | 1.172 | ✓ | accepted | vectorize the validation forward with wide::f32x8 (8 cards/lane, exp8/ln8); bit-for-bit epoch selection |
| 15 | 410 | 168 | 2.500 | 5884 | 6458 | 1.098 | 1.262 | ✓ | accepted | vectorize analytic gradient forward+backward with wide::f32x8 (8 cards/lane); pad batches to multiple of 8 |
| 16 | 170 | 268 | 0.623 | 6458 | 6458 | 1.000 | 1.000 | ✓ | rejected | thread SIMD gradient across 2 cores (split card-groups) - REJECTED, vector units SMT-saturated |
| 17 | 172 | 145 | 1.156 | 6458 | 6460 | 1.000 | 1.001 | ✓ | accepted | Minimax (Remez) exp8 deg 6->4 and ln8 deg 4->2: 4 fewer FMAs |
| 18 | 145 | 46 | 3.118 | 6460 | 7207 | 1.116 | 1.315 | ✓ | accepted | O(N) expanding window: one per-card pass scores every timestep, replacing O(N^2) per-prefix forward |
| 19 | 47 | 40 | 1.145 | 7207 | 7204 | 1.000 | 0.999 | ✓ | accepted | Build host batches once (train==test): reuse for grad+validation, drop redundant clone+build |

**Cumulative speed_ratio (product of accepted): ×75.031** — upward-biased (winner's curse); anchor periodically vs iter-0 baseline.

