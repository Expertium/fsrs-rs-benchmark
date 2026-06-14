use fsrs::ComputeParametersInput;

use std::sync::Mutex;

use pyo3::prelude::*;

/// Rebuild the expanding-window prefix-items + their synthetic card ids from flat per-card arrays
/// (the COMPACT-RAW layout — see `compute_parameters_raw`). Returns `(train_set, card_ids)` in the
/// exact global order `convert_to_items` produces: per card the surviving prefixes (review j with
/// `review_ths[s+j] != 0`) reconstructed as reviews[0..=j], then a stable sort by review_th. Shared
/// by `compute_parameters_raw` (training) and `evaluate_raw` (0-epoch proxy) so both see one layout.
fn reconstruct_raw_items(
    deltas: &[f32],
    ratings: &[u32],
    review_ths: &[i64],
    card_offsets: &[usize],
) -> (Vec<fsrs::FSRSItem>, Vec<i64>) {
    let n_cards = card_offsets.len().saturating_sub(1);
    // (review_th key, prefix item, synthetic card id) in INSERTION order = (card index asc,
    // within-card position asc) — identical to convert_to_items's pre-sort `pairs` order.
    let mut scored: Vec<(i64, fsrs::FSRSItem, i64)> = Vec::new();
    for c in 0..n_cards {
        let s = card_offsets[c];
        let e = card_offsets[c + 1];
        let k = e - s; // card length (number of reviews)
        // A prefix of length j+1 scores review j (1..=k-1) IFF that review survived scoring
        // (review_ths[s+j] != 0); the sentinel-0 reviews are history-only — no prefix-item.
        for j in 1..k {
            let th = review_ths[s + j];
            if th == 0 {
                continue;
            }
            let reviews: Vec<fsrs::FSRSReview> = (0..=j)
                .map(|i| fsrs::FSRSReview {
                    rating: ratings[s + i],
                    delta_t: deltas[s + i],
                })
                .collect();
            scored.push((th, fsrs::FSRSItem { reviews }, c as i64));
        }
    }
    // Stable sort by review_th (ties keep insertion order) == Python `pairs.sort(key=review_th)`.
    scored.sort_by_key(|x| x.0);
    scored.into_iter().map(|(_, item, cid)| (item, cid)).unzip()
}

#[pyclass(module = "fsrs_rs_python")]
#[derive(Debug)]
pub struct FSRS(Mutex<fsrs::FSRS>);

#[pymethods]
impl FSRS {
    #[new]
    pub fn new(parameters: Vec<f32>) -> Self {
        Self(fsrs::FSRS::new(&parameters).unwrap().into())
    }

    /// Returns `(optimized_parameters, elapsed_seconds)`. Only the Rust
    /// `compute_parameters()` call is timed, with a monotonic clock; the
    /// PyO3 input conversion above it is excluded and Python stays untimed.
    ///
    /// `card_ids` (optional, parallel to `train_set`) labels each prefix-item with its originating
    /// card so training can group a card's expanding-window prefixes into one mini-batch (the O(N)
    /// window path). It is plain metadata Python already has; the timed work is unchanged.
    #[pyo3(signature = (train_set, card_ids=None, num_epochs=None))]
    pub fn compute_parameters(
        &self,
        train_set: Vec<FSRSItem>,
        card_ids: Option<Vec<i64>>,
        num_epochs: Option<usize>,
    ) -> (Vec<f32>, f64) {
        let input = ComputeParametersInput {
            train_set: train_set.iter().map(|x| x.0.clone()).collect(),
            progress: None,
            enable_short_term: true,
            enable_sched_penalties: false,
            num_relearning_steps: None,
            card_ids,
            num_epochs,
            init_w: None,
        };
        let start = std::time::Instant::now();
        let params = fsrs::compute_parameters(input).unwrap_or_default();
        let elapsed = start.elapsed().as_secs_f64();
        (params, elapsed)
    }

    /// COMPACT-RAW twin of `compute_parameters`: takes each card's FULL review sequence as flat
    /// O(N) arrays instead of the O(N^2) expanding-window prefix `FSRSItem`s, and rebuilds the
    /// identical `(train_set, card_ids)` *inside* Rust before calling the SAME unchanged
    /// `fsrs::compute_parameters`. This removes the O(N^2) Python-side object materialization and the
    /// PyO3 marshaling of every prefix (the ~68% tuning-eval cost / the 236 GB@3k RAM wall) while
    /// staying bit-for-bit: it produces byte-identical input to the existing optimizer.
    ///
    /// Layout (CSR across cards, each card's reviews in chronological / review_th order):
    ///  * `deltas[s..e]`, `ratings[s..e]`, `review_ths[s..e]` are card c's full sequence, where
    ///    `s = card_offsets[c]`, `e = card_offsets[c+1]` (so `card_offsets` has `n_cards + 1` entries).
    ///  * Cards MUST be supplied in ascending real-card-id order (matching `convert_to_items`'s
    ///    `groupby(card_id)`): the synthetic card id used here is the card's index `c`, which is
    ///    monotonic in the real card id, so the `(full_len, card_id)` batch sort tie-break and the
    ///    review_th-tie insertion order are reproduced exactly.
    ///  * Each review index j (1..K-1) that has a surviving prefix-item carries that prefix's global
    ///    recency key in `review_ths[s + j]`. Reviews that the preprocessing keeps only as HISTORY
    ///    (the never-scored initial review j=0, and same-timestamp `delta_t==0` middle reviews that
    ///    base.py drops from scoring but keeps in later prefixes) carry the sentinel `0` and produce
    ///    NO prefix-item. (Real review_ths are `range(1, n+1)` >= 1, so 0 is unambiguous.) A scored
    ///    prefix of length L = j+1 reconstructs as reviews[0..=j] — a head of the full sequence.
    ///
    /// `init_w` (optional, gated default-param tuner): when a non-empty vector is supplied it
    /// becomes the per-user SGD start AND L2 anchor (instead of the compiled DEFAULT_PARAMETERS);
    /// `None`/empty keeps DEFAULT_PARAMETERS, so this is bit-for-bit with `compute_parameters`.
    ///
    /// Only the `fsrs::compute_parameters` call is timed (monotonic clock); the reconstruction below
    /// the timer is excluded, mirroring `compute_parameters`.
    #[pyo3(signature = (deltas, ratings, review_ths, card_offsets, num_epochs=None, init_w=None))]
    pub fn compute_parameters_raw(
        &self,
        deltas: Vec<f32>,
        ratings: Vec<u32>,
        review_ths: Vec<i64>,
        card_offsets: Vec<usize>,
        num_epochs: Option<usize>,
        init_w: Option<Vec<f32>>,
    ) -> (Vec<f32>, f64) {
        let (train_set, card_ids) =
            reconstruct_raw_items(&deltas, &ratings, &review_ths, &card_offsets);
        let input = ComputeParametersInput {
            train_set,
            progress: None,
            enable_short_term: true,
            enable_sched_penalties: false,
            num_relearning_steps: None,
            card_ids: Some(card_ids),
            num_epochs,
            init_w,
        };
        let start = std::time::Instant::now();
        let params = fsrs::compute_parameters(input).unwrap_or_default();
        let elapsed = start.elapsed().as_secs_f64();
        (params, elapsed)
    }

    /// COMPACT-RAW twin of `evaluate`: reconstructs the same prefix-items the windowed path scores
    /// (identical set to `convert_to_items`) from flat per-card arrays, then runs the FROZEN Rust
    /// `evaluate()` under the current parameters. The 0-epoch proxy of the gated default-param tuner.
    pub fn evaluate_raw(
        &self,
        deltas: Vec<f32>,
        ratings: Vec<u32>,
        review_ths: Vec<i64>,
        card_offsets: Vec<usize>,
    ) -> f32 {
        let (train_set, _card_ids) =
            reconstruct_raw_items(&deltas, &ratings, &review_ths, &card_offsets);
        self.0
            .lock()
            .unwrap()
            .evaluate(train_set, |_| true)
            .unwrap()
            .log_loss
    }

    /// Log loss of `items` under the current parameters, computed by Rust
    /// `evaluate()` (the same metric the optimizer uses; not timed).
    pub fn evaluate(&self, items: Vec<FSRSItem>) -> f32 {
        let items: Vec<fsrs::FSRSItem> = items.iter().map(|x| x.0.clone()).collect();
        self.0
            .lock()
            .unwrap()
            .evaluate(items, |_| true)
            .unwrap()
            .log_loss
    }

    pub fn benchmark(&self, train_set: Vec<FSRSItem>) -> Vec<f32> {
        fsrs::benchmark(ComputeParametersInput {
            train_set: train_set.iter().map(|x| x.0.clone()).collect(),
            progress: None,
            enable_short_term: true,
            enable_sched_penalties: false,
            num_relearning_steps: None,
            card_ids: None,
            num_epochs: None,
            init_w: None,
        })
    }

    /// Like `benchmark`, but also returns the elapsed seconds of the Rust
    /// `fsrs::benchmark()` call (monotonic clock; the PyO3 item conversion above the
    /// timer is excluded, mirroring `compute_parameters`'s timing). Profiling-only: the
    /// Phase-2 measurement harness (`profiling/measure_benchmark.py`) uses this to time
    /// benchmark()'s Rust region; benchmark.py itself still calls the untimed `benchmark`.
    pub fn benchmark_timed(&self, train_set: Vec<FSRSItem>) -> (Vec<f32>, f64) {
        let input = ComputeParametersInput {
            train_set: train_set.iter().map(|x| x.0.clone()).collect(),
            progress: None,
            enable_short_term: true,
            enable_sched_penalties: false,
            num_relearning_steps: None,
            card_ids: None,
            num_epochs: None,
            init_w: None,
        };
        let start = std::time::Instant::now();
        let params = fsrs::benchmark(input);
        let elapsed = start.elapsed().as_secs_f64();
        (params, elapsed)
    }

    #[pyo3(signature = (items, starting_states=None))]
    pub fn memory_state_batch(
        &self,
        items: Vec<FSRSItem>,
        starting_states: Option<Vec<Option<MemoryState>>>,
    ) -> Vec<MemoryState> {
        let items: Vec<fsrs::FSRSItem> = items.iter().map(|x| x.0.clone()).collect();
        let starting_states = starting_states
            .unwrap_or_else(|| (0..items.len()).map(|_| None).collect())
            .into_iter()
            .map(|state| state.map(|x| x.0))
            .collect();
        self.0
            .lock()
            .unwrap()
            .memory_state_batch(items, starting_states)
            .unwrap()
            .into_iter()
            .map(MemoryState)
            .collect()
    }
}

#[pyclass(module = "fsrs_rs_python")]
#[derive(Debug, Clone)]
pub struct MemoryState(fsrs::MemoryState);

#[pymethods]
impl MemoryState {
    #[new]
    #[pyo3(signature = (stability, difficulty, stability_fast=None))]
    pub fn new(stability: f32, difficulty: f32, stability_fast: Option<f32>) -> Self {
        // Default the fast trace to the CUDA fsrs7_init fraction (0.8 * stability)
        // when not supplied, so 2-arg construction stays backward-compatible.
        let stability_fast = stability_fast.unwrap_or(0.8 * stability);
        Self(fsrs::MemoryState {
            stability,
            difficulty,
            stability_fast,
        })
    }
    #[getter]
    pub fn stability(&self) -> f32 {
        self.0.stability
    }
    #[getter]
    pub fn difficulty(&self) -> f32 {
        self.0.difficulty
    }
    #[getter]
    pub fn stability_fast(&self) -> f32 {
        self.0.stability_fast
    }
    pub fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

#[pyclass(module = "fsrs_rs_python")]
#[derive(Debug, Clone)]
pub struct FSRSItem(fsrs::FSRSItem);

#[pymethods]
impl FSRSItem {
    #[new]
    pub fn new(reviews: Vec<FSRSReview>) -> Self {
        Self(fsrs::FSRSItem {
            reviews: reviews.iter().map(|x| x.0).collect(),
        })
    }
    pub fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

#[pyclass(module = "fsrs_rs_python")]
#[derive(Debug, Clone)]
pub struct FSRSReview(fsrs::FSRSReview);

#[pymethods]
impl FSRSReview {
    #[new]
    pub fn new(rating: u32, delta_t: f32) -> Self {
        Self(fsrs::FSRSReview { rating, delta_t })
    }
    pub fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn fsrs_rs_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FSRS>()?;
    m.add_class::<MemoryState>()?;
    m.add_class::<FSRSItem>()?;
    m.add_class::<FSRSReview>()?;
    m.add("DEFAULT_PARAMETERS", fsrs::DEFAULT_PARAMETERS.to_vec())?;
    Ok(())
}
