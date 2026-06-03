use fsrs::ComputeParametersInput;

use std::sync::Mutex;

use pyo3::prelude::*;

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
    #[pyo3(signature = (train_set, card_ids=None))]
    pub fn compute_parameters(
        &self,
        train_set: Vec<FSRSItem>,
        card_ids: Option<Vec<i64>>,
    ) -> (Vec<f32>, f64) {
        let input = ComputeParametersInput {
            train_set: train_set.iter().map(|x| x.0.clone()).collect(),
            progress: None,
            enable_short_term: true,
            enable_sched_penalties: false,
            num_relearning_steps: None,
            card_ids,
        };
        let start = std::time::Instant::now();
        let params = fsrs::compute_parameters(input).unwrap_or_default();
        let elapsed = start.elapsed().as_secs_f64();
        (params, elapsed)
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
