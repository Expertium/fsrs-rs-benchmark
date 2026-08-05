# Why Anki reads ~10× and not ~30× — decomposition of the shipped fsrs-rs speedups

**2026-08-05, CORRECTED same day.** Anki [PR #4956](https://github.com/ankitects/anki/pull/4956)
bumped fsrs-rs 5.2.0 → **6.6.1**, which carries
[#411](https://github.com/open-spaced-repetition/fsrs-rs/pull/411) (analytic gradient + host Adam +
`card_ids` window), [#419](https://github.com/open-spaced-repetition/fsrs-rs/pull/419) (scalar
windowed wins) **and [#424](https://github.com/open-spaced-repetition/fsrs-rs/pull/424) (skip
per-epoch validation + best-epoch selection)** — 6.6.1 = tag v6.6.0 + #424 exactly (merged
2026-06-09, published the same day via "bump (#425)"; note the crates.io 6.6.1 has no git tag,
which briefly misled this investigation into treating the v6.6.0 tag as what Anki ships). Anki
passes `card_ids` and sets 8 epochs. Observed user-facing speedup ≈ **10×** vs the ~30×-without-
SIMD expectation.

## Method

Same harness as the campaign (`profiling/measure.py`, 50 srs-benchmark users, min-of-3,
`speed_ratio` = median of per-user ratios). Phase-3 pybind (`C:/Users/Andrew/fsrs-rs-v6-pybind`,
path-dep on `C:/Users/Andrew/fsrs-rs-upstream`), two builds: **v5.2.0 legacy** (`/tmp/up_v52.pyd`,
Anki's "before") and **v6.6.0 env-toggled** (`/tmp/up_v66.pyd`): `FSRS_UPSTREAM_NOCARDS`,
`FSRS_UPSTREAM_EPOCHS`, `FSRS_NO_EPOCH_VALID` (local patch reproducing #424 on the v6.6.0 tag).
Data day-truncated (FSRS-6 `delta_t: u32`), same-day-only items dropped, identical across cells.
Known asymmetry: v5.2.0's default `max_seq_len` is 64 vs v6.6.0's 256 (the new version trains
up-to-4× longer sequences), so A→B *understates* the algorithmic gain.

## Results (median ms/user over 50 users; per-user median-of-ratios)

| cell | config | median ms | mean LogLoss |
|---|---|---|---|
| A | v5.2.0 stock (5 ep, burn) | 711.5 | 0.376568 |
| B | v6.6.0, no card_ids, 5 ep | 61.0 | 0.377219 |
| C | v6.6.0, card_ids, 5 ep | 23.0 | 0.376736 |
| D | v6.6.0, card_ids, 8 ep, validation on | 38.5 | 0.376164 |
| E | **D + no validation (#424) = what Anki ships as 6.6.1** | 29.0 | 0.376319 |

| step | factor | meaning |
|---|---|---|
| A→B | ×10.97 | analytic gradient + host Adam + #419 scalar wins (no window) |
| B→C | ×2.39 | the `card_ids` O(N) window (fork's FSRS-7 equivalent was ×3.12) |
| C→D | ×0.71 | the 5→8 epoch bump (deliberate accuracy spend) |
| D→E | ×1.36 | #424: no per-epoch validation (fork measured ×1.37 on FSRS-7) |
| **A→E** | **×28.99** | **shipped `compute_parameters` (Anki's exact config) — the full ~30×** |

So the *library* delivered the expected speedup. Epoch-compensation checks: 5ep+valid → 6ep noval
= ×1.068 faster AND −0.00028 log loss; 8ep+valid → 9ep noval = ×1.222, +0.00016.

**Reconciliation with upstream's own bench numbers** (their Criterion bench, one smallish
collection, with `card_ids`): #411 ×9.3 → #419 ×1.62 (27.756→17.103 ms) → #424 ×1.43
(15.374→10.738 ms) = **≈ ×21.5 at the 5-epoch default**, ≈ ×13–14 at Anki's 8 epochs. Our 50-user
measurement of the same config is ×29 median because the multiplier grows with collection size
(per-user ×5.3–×58); both scales tell the same story.

## Why users still see ~10×

1. **End-to-end dilution (now the whole story for large collections).** Anki's optimize also
   runs TWO full `evaluate()` passes (current + optimized params, to keep the better one), item
   building, revlog SQL, optionally the 5-split health check. Plain `evaluate()` is still the
   per-item O(sum-of-prefix-lengths) scorer — untouched by #411/#419/#424. Measured
   (2×evaluate)/(shipped training): 0.40× (5k items) → **2.11×** (54k — evaluation costs twice
   the training!). Implied end-to-end `(A+2ev)/(E+2ev)`: **×11.2–×19.5**, before counting the
   un-accelerated Anki-side costs — i.e. "~10×".
2. **Collection-size compression.** Per-user A→E spans ×5.3 (3.9k items) → ×58 (54k); the fixed
   per-user floor dominates small collections, and typical real Anki collections are smaller
   than any of these 50 users.
3. **NOT "FSRS-6 has less leeway"** — the analytic core transferred at full strength (×11) and
   the total matches the expectation.

## Actionable follow-up (the one real lever left)

**Window the plain `evaluate()`** — accept `card_ids` (same convention as
`ComputeParametersInput`), run each card's memory-state trajectory once, read every prefix item's
retrievability along the way (identical predictions, O(total reviews) model work). #421 did this
only for the time-series-split evaluation's *training* side. For big collections this alone is a
~2–3× end-to-end optimize win (it removes the dominant non-training cost). Implemented as the
`window-evaluate` branch in the upstream clone → PR.

Artifacts: `profiling/measure/up_{A_v52,B_v66_nocards_5ep,C_v66_cards_5ep,D_v66_cards_8ep,E_v66_cards_8ep_noval,F_v66_cards_6ep_noval,G_v66_cards_9ep_noval}.jsonl`.
