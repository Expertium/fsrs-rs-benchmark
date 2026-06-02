# FSRS-rs speed autoresearch

You are the autoresearcher. Your job is to make FSRS-7 parameter optimization **faster**, so real Anki users spend less time staring at the optimizer's progress bar and more time doing reviews. Making the number go down only counts if it actually helps users:

- A speedup that only works on AVX-512 CPUs and Void Linux doesn't count.
- A speedup that makes FSRS less accurate doesn't count (that's what the correctness bars are for).
- A speedup that only works when processing multiple users in parallel doesn't count.

You pick the directions, run the experiments, and report. Balancing broad exploration against squeezing one idea dry is your call. Budget: ~150–300 iterations over ~2 weeks — plenty of room for bold structural bets, so don't fear "wasting" iterations or declare convergence early. The user is a Python/PyTorch person, not a Rust dev: ask only for high-level feedback, never Rust technicalities.

## Host machine

- Windows 10 Pro 22H2 (build 19045), 64 GB RAM, Ryzen 9 5950x

## Running it

```
start /high /affinity 0xFFFFFFF0 python compute_parameters.py --algo FSRS-rs --short --secs --recency --processes 10 --max-user-id 50
```

- **`compute_parameters.py`** is the timing harness: it trains each user with FSRS-rs `compute_parameters()` (the function you optimize) and scores log loss with Rust `evaluate()`, using train set == test set. Per-user time (ms) is written to `result/compute_parameters-FSRS-rs-short-secs-recency.jsonl`.
- **`benchmark.py`** is the reference harness (`benchmark()` + 5-fold TimeSeriesSplit + Python forgetting curve). Use it together with **`evaluate.py`** for the bit-for-bit correctness check (constraint 3a).
- `/high /affinity 0xFFFFFFF0` plus the per-worker CPU pinning built into `compute_parameters.py` keeps the workers off logical CPUs 0–3, at HIGH priority, each on its own pair of cores — so cross-worker contention is constant rather than random. (Intended to cut measurement noise; empirically it mainly lowers/steadies the compute *level*, not the run-to-run variance — see *What the guards buy*.)
- **Big users first:** `compute_parameters.py` dispatches users largest-collection-first → smallest-last (hardcoded `USERS_BY_SIZE_DESC`). Measured benefit (3×10 experiment, 2026-06-01) is **scheduling, not noise**: dispatching the long jobs first balances the 10 workers so each run finishes ~**10% sooner** (less tail idle); run-to-run noise is unchanged. It changes only the scheduling order, not the recorded numbers (the output `.jsonl` is re-sorted by user).
- Already applied on this machine to pin the clock speed and remove turbo/thermal noise:
  ```
  powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 99
  powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMIN 99
  powercfg /setactive SCHEME_CURRENT
  ```

## Code layout

```
compute_parameters.py   # ★ timing harness you optimize: Rust compute_parameters() + Rust evaluate(), train==test, min-of-3
benchmark.py            # reference harness (5-fold TimeSeriesSplit + Python curve) for the 3a bit-for-bit check
evaluate.py             # Python forgetting-curve scorer used by benchmark.py (read-only; skipped by complexity.py)
config.py               # CLI args + Config (seed=42, max_seq_len, --data path, --processes)
data_loader.py          # per-user revlog loading from ../anki-revlogs-10k/revlogs/user_id=*
utils.py                # sort_jsonl(), catch_exceptions(), shared helpers
complexity.py           # constraint-13 score over .py/.rs (skips target/ __pycache__/ profiling/ + benchmark/evaluate/plot_history/complexity)
plot_history.py         # plots the speed history (read-only tooling; skipped by complexity.py)
features/               # review-history preprocessing = the O(N²) expanding window (N reviews -> N-1 items)
  base.py               #   builds t_history/r_history; caps each card at 2× max_seq_len reviews
  fsrs_engineer.py      #   FSRS (t_history, rating) feature engineering
  create_features.py    #   feature-builder dispatch
fsrs-rs/src/            # the Rust FSRS crate (path dep of the binding) = the main mutation surface (3a/6)
  model.rs              #   ★ forgetting curve + per-timestep recurrence: forward(), update_state()
  training.rs           #   ★ Rust compute_parameters(), train loop, Adam, epochs/batching, Dual35 penalty gradient
  inference.rs          #   evaluate() = the scorer; do NOT modify (constraint 11 anchor)
  lib.rs, error.rs      #   crate exports, DEFAULT_PARAMETERS, error types
fsrs_rs_python/         # PyO3 binding -> built (maturin) into the installed fsrs_rs_python extension
  src/lib.rs            #   Python entry points; times the Rust compute_parameters() with a monotonic clock
result/                 # per-user .jsonl; compute_parameters-*.jsonl = champion record, FSRS-rs-*.jsonl = benchmark output
profiling/              # profiling-only tools, skipped by complexity.py (see "Where the time goes")
evaluation/  raw/       # benchmark.py detailed-eval / --raw prediction dumps
```
Build artifacts (`*/target/`, `__pycache__/`) and the external dataset (`../anki-revlogs-10k/`) are omitted.

## Measurement protocol (how every timing number is produced)

- Time each of the 50 users 3 times; the official per-user time is the **MIN** of the 3. Timing noise is one-sided (interference only ever slows a run), so the fastest of the three is the cleanest estimate of the true compute floor, and it also rejects the cold first run. (Empirically, min ~halves per-user run-to-run noise vs. the median, for free.)
- Always time under `--processes 10` (parallel across users).
- **Timed region = `compute_parameters()` (Rust) only**: exclude data loading / I/O and the `evaluate()` call; use a monotonic clock. Python stays untimed.
- Time the champion and the candidate **back-to-back in the same session** (re-baseline every comparison). Never reuse a baseline from an earlier session — the machine drifts (thermal/turbo/background load) over a 150–300 iteration campaign.
- **Noise floor (measured: 3×10 identical-build runs, 2026-06-01).** Run-to-run noise is ~**1%**, *not* the ~0.35–0.5% earlier *single* back-to-back pairs implied (those were lucky-tight draws). Across 10 identical runs the median-over-50 time has CV ≈ **0.95–1.1%**; the accept metric itself — the median per-user speedup between two runs (constraint 12) — lands within **~0.7%** of 1.0 typically and up to **~2%** in the worst of 45 pairs. So the 5% bar clears *typical* noise ~7× but **worst-case paired noise only ~2.5×** — discount sub-~2% median "speedups" as likely noise. (The median does **not** average out per-user noise — it's one order statistic, not a mean — so its CV ≈ the ~1.2% per-user CV; the median is used for robustness to the big collections, not variance reduction.)
- **What the guards buy (same experiment).** Priority+affinity and big-first did **not** measurably change the noise floor — run-to-run CV was ~1.0% / 1.1% / 0.95% for no-guards / +prio+affinity / +big-first (indistinguishable at n=10), plausibly because min-of-3 already discards contention-driven slow runs. Their real payoff is elsewhere: CPU-pinning lowered the median compute *level* ~**2.6%** (less core-migration/cache thrash) and big-first cut *wall-clock* ~**10%**. Keep them for a lower/faster/steadier floor — just don't expect them to tighten run-to-run variance.

## Constraints

1. No CUDA, no GPU.
2. At most **2 threads per user**. (Multiple users are already processed in parallel across processes; don't use more than 2 threads on a *single* user's data.)
3. Two correctness bars, by change type:
   - **(a) Math-unchanged changes** (constraint 6 — fusing, caching, reordering, etc.): if the change does **not** restructure the autodiff graph (forward-only fusions, dead-code removal, pure caching that doesn't move taped ops), per-user results should reproduce the champion **bit-for-bit**. Verify with **both** harnesses (diff the result `.jsonl`; sanity-check aggregates with `evaluate.py`) — each catches what the other can't:
     - **benchmark.py** records per-user **params + log loss** → check both. Reference: 50 users, **1,581,505 reviews**, log loss **0.31205** (iter-66 dual-trace champion; was 0.3152 pre-port).
     - **compute_parameters.py** records per-user **params + log loss** (and time) → diff the params and log loss; its recorded time naturally varies run-to-run (see the Noise floor note), so ignore time for the bit-for-bit check. Reference: 50 users, **1,897,936 items**, log loss **0.3098** (iter-66 dual-trace champion; was 0.3085 pre-port). (Its params differ from benchmark.py's — different `max_seq_len` and train/test split — so treat each harness as its own within-harness bit-for-bit reference.)

     **Reassociation caveat (ruling 2026-06-01):** changes that restructure the *backward* graph (hoisting/fusing/caching **taped** ops — e.g. pre-slicing loop-invariant weights) reorder floating-point gradient accumulation under burn's `Autodiff<NdArray<f32>>`, so they will **not** be bit-for-bit even though the math is identical — Adam then wanders over the 8 epochs to slightly different but equally-good params. These are still valid; judge them by the **(b) average-log-loss band** below, not bit-for-bit. (A *forward* identical to the champion is a good sanity check that the math really is unchanged.)
   - **(b) Precision-trading (float32/float16) OR graph-reordering changes**: log loss may drift, but the **AVERAGE log loss** — the single aggregate `evaluate.py` reports (benchmark.py's mean over reviews ≈ **0.31205**; compute_parameters.py's mean LogLoss ≈ **0.3098**), **NOT** the max per-user change — must stay **within ±0.0010** of the champion's. Individual users' params and per-user log loss may move more; only the *aggregate average* is gated. A small regression or improvement inside that band is fine; a larger move means something broke (a real bug usually moves the average far more than reassociation does — iter-1 hoist moved it ~1e-6).
4. Don't change batch size, number of epochs, or any other SGD hyperparameter. No early stopping. Don't change `max_seq_len`.
5. Number of users must stay at **50**, and the dataset size must stay fixed — **benchmark.py: 1,581,505 reviews; compute_parameters.py: 1,897,936 items**. Assert this so you can't silently break review preprocessing.
6. Don't change FSRS-7's math or logic. Fusing operations, caching, float32/float16, etc. are all fair game **as long as the underlying math is identical**.
7. No CPU- or OS-specific tricks. Optimizations must work on any CPU/OS you can reasonably expect in 2026, **including smartphones**.
8. If Rust isn't low-level enough, custom SIMD / LLVM vectorization / other clever comp-sci is welcome — as long as it doesn't violate constraint 7 (stay portable; don't depend on AVX-512 etc.).
9. Any non-deterministic operation must be seeded; choose the seed once and never change it.
10. Feel free to build or import your own profiling tools and do profiling-only runs, but never modify the value written to the final .jsonl, and keep Python untimed. Also, profiling-only runs shouldn't be added to history.
11. **Mutation surface:** you may modify `compute_parameters()` and any func/struct it uses, plus FSRS-7 itself (as long as the math stays the same, per 3a). You may **not** modify `evaluate()` — the Rust scorer in `fsrs-rs/src/inference.rs`, invoked as `evaluate(items, |_| true)` by the binding (`fsrs_rs_python/src/lib.rs`) and as `predictor.evaluate(items)` in `compute_parameters.py`. It's the anti-cheating anchor.
    - Don't shift timed Rust work into untimed Python/preprocessing to beat the timer — the work has to actually disappear, not move off-clock. (For the same reason `complexity.py` counts both Python and Rust, so you can't move Rust into Python and call it a "simplification." `evaluate.py` and a few other diagnostics-only files are excluded from the complexity score.)
    - Optimizations must speed up **one user in isolation**. Anything that only pays off by batching 50 users (cross-user caching/amortization) is an artifact — a real Anki user optimizes one collection at a time.
12. **Acceptance — speed:** for each of the 50 users the per-user speedup is `s_u = t_champion_u / t_candidate_u` (each time = the min of its 3 runs). The single speed metric is **`speed_ratio` = the median of the 50 `s_u`** (the typical user's speedup). **Accept only if `speed_ratio ≥ 1.05`** (the median user is ≥5% faster). The median — not the mean of times — is used so a few slow/large collections can't carry the result and the ~1.5% warm-up drift can't fake a 5% median. Report the mean speedup too, but only as an informational "is one user dominating?" tell. (A paired Wilcoxon test was dropped: with per-user noise this low, the ~1-1.5% drift between two *identical* runs already drives p ≈ 0, so it fired on drift alone.)
13. **Acceptance — complexity:** the speedup must out-run added complexity — `speed_ratio ≥ complexity_ratio^2.5`, using the same median `speed_ratio` from constraint 12, with `complexity_ratio = c_candidate / c_champion` (>1 = more complex; see `complexity.py`). Examples: +1% complexity needs ≥ +2.52% speed; +5% needs ≥ +12.97%; +50% needs ≥ 2.76×. If complexity is flat or drops, only the 5% floor (constraint 12) binds. (Self-consistent under compounding: N changes each sitting on the limit keep cumulative speed vs. complexity on the same curve.)
Complexity score weights were chosen so that LOC contributes ~25% to the final score, AST node count and cyclomatic complexity contribute ~37.5% each. However, these proportions may have drifted after many iterations.

## Champion & compounding

The champion is the current fastest accepted version. Every candidate is measured against the **current** champion (not the original), so accepted speedups compound multiplicatively; on accept, the candidate becomes the new champion. If an idea just barely misses (e.g. +4.5% median when +5% is needed), it's your call whether to retry with a tweak or move on.

**Final validation:** before declaring victory / committing to GitHub, re-validate the final champion on **1000 users** (not just 50) to confirm the speedup generalizes across the review-count distribution. CPU-specific optimizations may be added as a bonus at that point, but they don't officially count for the loop.

## History bookkeeping

Keep a `.jsonl` log and a human-readable `.md`. Each entry records:
1. Iteration (0 = baseline, 1 = first change)
2. Timestamp
3. Median time (50 users) before the change
4. Median time (50 users) after
5. **Median speed ratio** (see constraint 12 above, it is not the ratio of medians, but the **median of ratios**)
6. Complexity score before
7. Complexity score after (see `complexity.py`)
8. **Complexity ratio**
9. True/False for "did all checks (log loss within bands, same number of reviews, same parameters (for changes that don't trade precision), etc.) pass?"
10. "accepted"/"rejected" status. "rejected" could mean either "improvement in speed wasn't sufficient" or "checks didn't pass"
11. A summary of the change **written before timing it** (≤15 words; one number = one word)
12. (`.jsonl` only) A private comment to your future self (e.g. notes to survive a compaction), but it must not be displayed in the .md file

**Progress plots (`plot_history.py`): two stacked views vs iteration.**
1. **Cumulative `speed_ratio`** (the headline) — the running product of the *accepted* iterations' median `speed_ratio` (item 5; rejects ×1). Read as "the median user is now X× faster than the iter-0 baseline." This *is* the accept metric (constraint 12) compounded, and it's **drift-immune**: each `speed_ratio` is a *within-session paired* ratio, so cross-session machine drift (~1%) cancels.
2. **Median per-user time (ms)** — intuitive, but **machine-specific and per-session, so it is NOT the accept/reject metric** (it's the ratio *of medians*, not the median *of ratios* that gates accepts, and it jitters with cross-session drift). Informational only; items 3–4 are its log evidence.
- **Caveat — the product is upward-biased:** you accept only when a noisy `speed_ratio` clears 1.05, so accepted ratios are selected high (winner's curse) and the ~1–2% paired noise compounds. **Anchor it** — every ~10–20 iters, re-measure the current champion vs the iter-0 baseline back-to-back in one session and plot that as a point: a direct, unbiased cumulative speedup whose gap to the product line is the accumulated bias. The final **1000-user** re-validation (see *Champion & compounding*) is the official total.

**Compaction:** run `/compact` every 5 iterations, unconditionally (regardless of how many were accepted vs rejected) — over a 150–300-iteration campaign this keeps the context window fresh so per-iteration reasoning doesn't degrade. The history `.jsonl`/`.md` above is the durable record across compactions, so nothing important is lost.

## Notes

- **The O(N²) expanding window** (a card with N reviews becomes N−1 separate FSRSItems): you can try to make it O(N), but the original author says it makes log loss worse even when you normalize so both implementations take the same wall-clock time. Expect it to need many attempts, and it may not pan out at all.
- **`result/*(old)*.jsonl`** — files in `result/` with `(old)` in their names are kept on purpose as comparison baselines (e.g. the pre-dual-trace single-trace numbers). They are not stale junk; leave them in place.

## Where the time goes (profiling)

Profiling lives in `profiling/` — **excluded from `complexity.py`** (never counts against constraint 13) and never imported by the timed path; profiling-only per constraint 10 (doesn't touch `result/`).
- **`profiling/profile_structure.py`** (`python profiling/profile_structure.py`): reproduces the 50-user preprocessing and regresses their *already-recorded* times against structural work proxies; caches `profiling/structure.json`.
- **`profiling/_profile_user.py`** (`python profiling/_profile_user.py <user_id> <reps>`): loads one user and loops `compute_parameters()` so a sampler can attach.

The training loop and optimizer are fixed, so these hold across FSRS-7 variants (different hyperparams / a heavier forgetting curve only *strengthen* the per-timestep pathways below):
- **Cost model:** time ≈ linear in **total sequence-work = Σ over items of sequence-length** (R²≈0.99 over the 50 users) + a fixed per-user floor (~0.18 s) that dominates tiny collections. `n_items` (R²≈0.65) and `n_cards` (≈0.16) are weak predictors — summed sequence length is what counts. The expanding window is **bounded** (the Python features cap each card at 2×`max_seq_len`; mean seq-len ≈ 9), so it is *not* a runaway O(N²): **per-timestep cost dominates, not the algorithmic blowup.**
- **Hot path:** `fsrs-rs/src/model.rs::forward()` — a **sequential per-timestep recurrence** (loop over seq-len, batched over cards) run under **burn `Autodiff<NdArray<f32>>`** for N epochs (each epoch = a train forward+backward, plus a full validation forward for best-epoch selection). The **autodiff backward over the recurrence is the prime cost**, and it scales with taped ops per timestep — so cutting per-step work helps the backward ~2–3× as much as the forward. Backend is already **f32** (so "use f32" is not a lever; only f16 remains, and it's risky per 3b).
- **Native profiling caveat:** py-spy `--native` returns 0 samples on the stripped release `.pyd` (no `.pdb`), and the work runs entirely inside the native Rust call, so the non-native sampler sees nothing either. For function-level Rust hotspots, rebuild with symbols (`fsrs_rs_python/`: add `[profile.release] debug = true` to `Cargo.toml`, then `maturin develop --release`) and sample via `_profile_user.py`; otherwise just read the (small) Rust source.

Candidate pathways, best-first (all subject to the correctness bars):
1. **Hoist loop-invariant weight ops + kill per-step allocations** (safe, math-preserving / constraint 6). Each timestep re-slices weights (`model.w.get(n)` = clone+slice → a fresh tensor) and recomputes weight-only subexpressions; pre-extract the weights once per forward and hoist weight-only terms out of the seq loop. Shrinks the autodiff tape → faster fwd+bwd and less memory. **Grows with a more complex forgetting curve.** Verify bit-for-bit (3a).
2. **Analytic / dual-number gradient for the main loss** (bold, biggest potential, high effort). Replace burn Autodiff for the BCE loss with a hand-written gradient — a `Dual35`-style template already exists in `training.rs` for the penalty term. Removing the tape could be the single biggest win; aim for the 3b band (±0.0010).
3. **Trim per-iteration / per-epoch overhead** (safe, modest; helps the fixed floor → small collections, the long tail of real users). Each train/valid iteration copies the params tensor to a host `Vec`; each epoch runs a full validation forward. Hunt redundant host round-trips and needless validation-set reshuffles.
4. **O(N²)→O(N) expanding window** (risky; see the O(N²) note above). Ceiling ≈ 6–7× less work, but the author (not Andrew aka not me) says it hurts log loss and is hard — not a first move.
