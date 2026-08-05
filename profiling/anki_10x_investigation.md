# Why Anki sees ~10× and not ~30× — decomposition of the shipped fsrs-rs speedups

**2026-08-05.** Anki [PR #4956](https://github.com/ankitects/anki/pull/4956) bumped fsrs-rs
5.2.0 → 6.6.1, which carries the speed work from
[#411](https://github.com/open-spaced-repetition/fsrs-rs/pull/411) (analytic gradient + host Adam +
`card_ids` window) and [#419](https://github.com/open-spaced-repetition/fsrs-rs/pull/419) (scalar
windowed wins). Anki passes `card_ids` and bumps epochs 5 → 8 (accuracy compensation for the
window). Observed user-facing speedup ≈ **10×**; the expectation from the FSRS-7 fork campaign was
~30× without SIMD. This note decomposes the gap **by measurement**.

## Method

Same harness as the campaign (`profiling/measure.py`, 50 srs-benchmark users, min-of-3 per user,
`speed_ratio` = median of per-user ratios). The Phase-3 pybind (`C:/Users/Andrew/fsrs-rs-v6-pybind`,
path-dep on `C:/Users/Andrew/fsrs-rs-upstream`) was built in two variants:
* **v5.2.0 legacy** (`/tmp/up_v52.pyd`) — Anki's "before" (burn autodiff, no window).
* **v6.6.0 env-toggled** (`/tmp/up_v66.pyd`) — `FSRS_UPSTREAM_NOCARDS` (drop card_ids),
  `FSRS_UPSTREAM_EPOCHS` (TrainingConfig num_epochs), and `FSRS_NO_EPOCH_VALID` (a **local
  measurement patch** in the upstream clone: skip the per-epoch validation pass + best-epoch
  selection, ship the last epoch; stock behavior when unset).

Data is day-truncated (FSRS-6 `delta_t: u32`), same-day-only items dropped (upstream input
convention) — identical across all cells. Known asymmetry: v5.2.0's `max_seq_len` default is 64 vs
v6.6.0's 256, so the new version trains on up-to-4× longer sequences; A→B therefore *understates*
the pure algorithmic gain.

## Results (median ms/user over 50 users; per-user median-of-ratios)

| cell | config | median ms | mean LogLoss |
|---|---|---|---|
| A | v5.2.0 stock (5 ep, burn) | 711.5 | 0.376568 |
| B | v6.6.0, no card_ids, 5 ep | 61.0 | 0.377219 |
| C | v6.6.0, card_ids, 5 ep | 23.0 | 0.376736 |
| D | **v6.6.0, card_ids, 8 ep = what Anki ships** | 38.5 | 0.376164 |
| E | D + per-epoch validation removed | 29.0 | 0.376319 |

| step | factor | meaning |
|---|---|---|
| A→B | **×10.97** | analytic gradient + host Adam + #419 scalar wins (no window) |
| B→C | **×2.39** | the `card_ids` O(N) window (fork's FSRS-7 equivalent was ×3.12) |
| C→D | ×0.71 | the 5→8 epoch bump (deliberate accuracy spend) |
| D→E | **×1.36** | dropping per-epoch validation + best-epoch (fork measured ×1.37 on FSRS-7) |
| **A→D** | **×21.99** | **shipped `compute_parameters` speedup on this data** |
| A→E | ×28.99 | ≈ the "should be ~30×" — reachable by dropping validation |

## Why users still see ~10×

1. **End-to-end dilution (the big one).** Anki's optimize flow runs, besides training: two full
   `evaluate()` passes (current + optimized params, to keep the better one), item building, revlog
   SQL, and optionally the 5-split health check. The plain `evaluate()` is still the burn per-item
   scorer — **not** accelerated by #411/#419. Measured (2× evaluate) / (new training time): 0.24×
   (5k items) → **1.61×** (54k items — evaluation now costs more than training!). Implied
   end-to-end `(A + 2·eval) / (D + 2·eval)`: **×7.6–×17.7** across the size spread — i.e. "~10×".
2. **Collection-size dependence.** Per-user A→D ranges ×7.4 (3.9k items) to ×57 (60k); the fixed
   per-user floor compresses small collections toward ×7–9, and typical real Anki collections are
   *smaller* than any of these 50 users. The maintainer's own thread numbers ("4–5× no cards,
   ~9× with") are consistent with small-collection measurements.
3. **×1.36 is still on the table**: per-epoch validation + best-epoch selection survives in v6.6.0
   (training.rs ~799–822). The PR thread already floated removing it (+1 epoch to compensate);
   measured here: ×1.36 for +0.00016 log loss at 8 ep.
4. **NOT the main story: "FSRS-6 has less leeway".** Mostly false — the analytic core transfers
   (×11), the window is somewhat smaller on day-granular data (×2.4 vs ×3.12), and the ceiling
   (×29) matches the ~30× expectation almost exactly.

## Actionable upstream follow-ups

1. **Drop per-epoch validation / best-epoch, +1 epoch** (their own deferred idea): ×1.36 on
   training for ~zero accuracy cost.
2. **Window the plain `evaluate()`**: #421 already gave the *time-series-split* evaluation
   `card_ids`; the plain `evaluate()` that Anki calls twice per optimize still walks every prefix
   item O(N²)-style. Windowing it removes the dominant *non-training* cost for big collections.
3. Together these roughly double the end-to-end optimize speed for large collections.

Artifacts: `profiling/measure/up_{A_v52,B_v66_nocards_5ep,C_v66_cards_5ep,D_v66_cards_8ep,E_v66_cards_8ep_noval}.jsonl`.
The `FSRS_NO_EPOCH_VALID` patch lives uncommitted in the upstream clone (see git stash/diff there);
the pybind variants: `fsrs-rs-v6-pybind/src/lib.rs` (v6.6.0) + the v5.2.0 variant described above.
