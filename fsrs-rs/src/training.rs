use crate::error::Result;
use crate::model::{clip_parameters, parameters_to_model, Model};
use crate::{DEFAULT_PARAMETERS, FSRSError};
use burn::LearningRate;
use burn::backend::Autodiff;
use burn::backend::ndarray::NdArray;
use burn::config::Config;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::{DataLoaderIterator, Progress};
use burn::data::dataset::Dataset;
use burn::lr_scheduler::LrScheduler;
use burn::module::{AutodiffModule, Param};
use burn::nn::loss::Reduction;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::Backend;
use burn::tensor::cast::ToElement;
use burn::tensor::{Float, Int, Shape, Tensor, TensorData};
use burn::tensor::backend::AutodiffBackend;
use burn::train::TrainingInterrupter;
use burn::train::renderer::{MetricState, MetricsRenderer, TrainingProgress};
use core::marker::PhantomData;
use itertools::Itertools;
use log::info;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

#[path = "training_v7.rs"]
mod training_v7;

type B = NdArray<f32>;

const L2_PENALTY_WEIGHT: f64 = training_v7::PENALTY_W_L2;
const PENALTY_GRAD_LEN: usize = training_v7::GRAD_LEN;
const MIN_RETRIEVABILITY: f32 = 0.0001;
const MAX_RETRIEVABILITY: f32 = 0.9999;

type SchedulePenaltyFn = fn(&[f32], usize, bool) -> (f64, [f64; PENALTY_GRAD_LEN]);
type L2PenaltyFn = fn(&[f32], &[f32], usize, usize, f64, &[f32]) -> (f64, Vec<f32>);

fn schedule_penalty_fn() -> SchedulePenaltyFn {
    training_v7::maybe_schedule_penalty_value_and_grad
}

fn l2_penalty_fn() -> L2PenaltyFn {
    training_v7::l2_penalty_value_and_grad
}

// ========== ModelConfig ==========

#[derive(Config, Debug, Default)]
pub struct ModelConfig {
    #[config(default = false)]
    pub freeze_initial_stability: bool,
    pub initial_stability: Option<[f32; 4]>,
    pub initial_forgetting_curve: Option<[f32; 8]>,
    #[config(default = false)]
    pub freeze_short_term_stability: bool,
    #[config(default = 1)]
    pub num_relearning_steps: usize,
}

impl ModelConfig {
    #[cfg(test)]
    pub fn init<B: Backend>(&self) -> Model<B> {
        Model::new(self.clone())
    }
}

// ========== CosineAnnealingLR ==========

#[derive(Clone, Debug)]
pub(crate) struct CosineAnnealingLR {
    t_max: f64,
    eta_min: f64,
    init_lr: LearningRate,
    step_count: f64,
    current_lr: LearningRate,
}

impl CosineAnnealingLR {
    pub const fn init(t_max: f64, init_lr: LearningRate) -> Self {
        Self {
            t_max,
            eta_min: 0.0,
            init_lr,
            step_count: -1.0,
            current_lr: init_lr,
        }
    }
}

impl LrScheduler for CosineAnnealingLR {
    type Record<B: Backend> = usize;

    fn step(&mut self) -> LearningRate {
        self.step_count += 1.0;
        use std::f64::consts::PI;
        fn cosine_annealing_lr(
            init_lr: LearningRate,
            lr: LearningRate,
            step_count: f64,
            t_max: f64,
            eta_min: f64,
        ) -> LearningRate {
            if step_count == 0.0 {
                init_lr
            } else if (step_count - 1.0 - t_max) % (2.0 * t_max) == 0.0 {
                (init_lr - eta_min) * (1.0 - f64::cos(PI / t_max)) / 2.0
            } else {
                ((1.0 + f64::cos(PI * step_count / t_max))
                    / (1.0 + f64::cos(PI * (step_count - 1.0) / t_max)))
                .mul_add(lr - eta_min, eta_min)
            }
        }
        self.current_lr = cosine_annealing_lr(
            self.init_lr,
            self.current_lr,
            self.step_count,
            self.t_max,
            self.eta_min,
        );
        self.current_lr
    }

    fn to_record<B: Backend>(&self) -> Self::Record<B> {
        self.step_count as usize
    }

    fn load_record<B: Backend>(mut self, record: Self::Record<B>) -> Self {
        self.step_count = record as LearningRate;
        self
    }
}

// ========== Parameter Clipper ==========

pub(crate) fn parameter_clipper<B: Backend>(
    parameters: Param<Tensor<B, 1>>,
    enable_short_term: bool,
) -> Param<Tensor<B, 1>> {
    let (id, val) = parameters.consume();
    let mut clipped = clip_parameters(&val.to_data().to_vec().unwrap());
    if !enable_short_term {
        // w[26] controls short-term mixing; forcing it to 0 disables the short-term path.
        clipped[26] = 0.0;
    }
    Param::initialized(
        id,
        Tensor::from_data(TensorData::new(clipped, val.shape()), &val.device()).require_grad(),
    )
}

// ========== FSRSItem and FSRSReview ==========

/// Stores a list of reviews for a card, in chronological order. Each FSRSItem corresponds
/// to a single review, but contains the previous reviews of the card as well, after the
/// first one.
/// When used during review, the last item should include the correct delta_t, but
/// the provided rating is ignored as all four ratings are returned by .next_states()
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
pub struct FSRSItem {
    pub reviews: Vec<FSRSReview>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
pub struct FSRSReview {
    /// 1-4
    pub rating: u32,
    /// The number of days that passed (can be fractional).
    /// # Warning
    /// `delta_t` for item first(initial) review must be 0
    pub delta_t: f32,
}

const LONG_TERM_DELTA_T_BUCKET_DAYS: f32 = 1.0;

pub(crate) fn bucket_long_term_delta_t(delta_t: f32) -> f32 {
    if !delta_t.is_finite() {
        return 1.0;
    }
    let clamped = delta_t.max(1.0);
    (clamped / LONG_TERM_DELTA_T_BUCKET_DAYS).floor() * LONG_TERM_DELTA_T_BUCKET_DAYS
}

impl FSRSItem {
    // The previous reviews done before the current one.
    pub(crate) fn history(&self) -> impl Iterator<Item = &FSRSReview> {
        self.reviews.iter().take(self.reviews.len() - 1)
    }

    pub(crate) fn current(&self) -> &FSRSReview {
        self.reviews.last().unwrap()
    }

    pub fn long_term_review_cnt(&self) -> usize {
        self.reviews
            .iter()
            .filter(|review| review.delta_t >= 1.0)
            .count()
    }

    pub(crate) fn first_long_term_review(&self) -> FSRSReview {
        *self
            .reviews
            .iter()
            .find(|review| review.delta_t >= 1.0)
            .expect("Invalid FSRS item: at least one review with delta_t >= 1.0 is required")
    }

    pub(crate) fn r_matrix_index(&self) -> (u32, u32, u32) {
        let delta_t = self.current().delta_t as f64;
        let delta_t_bin = (2.48 * 3.62f64.powf(delta_t.log(3.62).floor()) * 100.0).round() as u32;
        let length = self.long_term_review_cnt() as f64 + 1.0;
        let length_bin = (1.99 * 1.89f64.powf(length.log(1.89).floor())).round() as u32;
        let lapse = self
            .history()
            .filter(|review| review.rating == 1 && review.delta_t >= 1.0)
            .count();
        if lapse == 0 {
            return (delta_t_bin, length_bin, 0);
        }
        let lapse_bin = (1.65 * 1.73f64.powf((lapse as f64).log(1.73).floor())).round() as u32;
        (delta_t_bin, length_bin, lapse_bin)
    }
}

pub fn filter_outlier(
    dataset_for_initialization: Vec<FSRSItem>,
    mut trainset: Vec<FSRSItem>,
) -> (Vec<FSRSItem>, Vec<FSRSItem>) {
    let to_key = |delta_t: f32| bucket_long_term_delta_t(delta_t).to_bits();
    let from_key = |key: u32| f32::from_bits(key);
    let mut groups = HashMap::<u32, HashMap<u32, Vec<FSRSItem>>>::new();

    for item in dataset_for_initialization.into_iter() {
        let first_review = item.reviews.first().unwrap();
        let first_long_term_review = item.first_long_term_review();
        let rating_group = groups.entry(first_review.rating).or_default();
        let delta_t_group = rating_group
            .entry(to_key(first_long_term_review.delta_t))
            .or_default();
        delta_t_group.push(item);
    }

    let mut filtered_items = vec![];
    let mut removed_pairs: [HashSet<_>; 5] = Default::default();

    for (rating, delta_t_groups) in groups.into_iter().sorted_by_key(|&(k, _)| k) {
        let mut sub_groups = delta_t_groups.into_iter().collect::<Vec<_>>();

        sub_groups.sort_by(|(delta_t_a, subv_a), (delta_t_b, subv_b)| {
            subv_b
                .len()
                .cmp(&subv_a.len())
                .then(from_key(*delta_t_b).total_cmp(&from_key(*delta_t_a)))
        });

        let total = sub_groups.iter().map(|(_, vec)| vec.len()).sum::<usize>();
        let mut has_been_removed = 0;

        for (delta_t, sub_group) in sub_groups.iter().rev() {
            if has_been_removed + sub_group.len() >= 20.max(total / 20) {
                if sub_group.len() >= 6
                    && from_key(*delta_t) <= if rating != 4 { 100.0 } else { 365.0 }
                {
                    filtered_items.extend_from_slice(sub_group);
                } else {
                    removed_pairs[rating as usize].insert(*delta_t);
                }
            } else {
                has_been_removed += sub_group.len();
                removed_pairs[rating as usize].insert(*delta_t);
            }
        }
    }
    trainset.retain(|item| {
        if item.long_term_review_cnt() == 0 {
            true
        } else {
            !removed_pairs[item.reviews[0].rating as usize]
                .contains(&to_key(item.first_long_term_review().delta_t))
        }
    });
    (filtered_items, trainset)
}

// ========== Dataset Types ==========

#[derive(Debug, Clone)]
pub(crate) struct WeightedFSRSItem {
    pub weight: f32,
    pub item: FSRSItem,
}

#[derive(Clone)]
pub(crate) struct FSRSBatcher<B: Backend> {
    _backend: PhantomData<B>,
}

impl<B: Backend> FSRSBatcher<B> {
    pub const fn new() -> Self {
        Self {
            _backend: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct FSRSBatch<B: Backend> {
    pub t_historys: Tensor<B, 2, Float>,
    pub r_historys: Tensor<B, 2, Float>,
    pub delta_ts: Tensor<B, 1, Float>,
    pub labels: Tensor<B, 1, Int>,
    pub weights: Tensor<B, 1, Float>,
}

impl<B: Backend> Batcher<B, WeightedFSRSItem, FSRSBatch<B>> for FSRSBatcher<B> {
    fn batch(&self, weighted_items: Vec<WeightedFSRSItem>, device: &B::Device) -> FSRSBatch<B> {
        let pad_size = weighted_items
            .iter()
            .map(|x| x.item.reviews.len())
            .max()
            .expect("FSRSItem is empty")
            - 1;

        let (time_histories, rating_histories) = weighted_items
            .iter()
            .map(|weighted_item| {
                let (mut delta_t, mut rating): (Vec<_>, Vec<_>) = weighted_item
                    .item
                    .history()
                    .map(|r| (r.delta_t, r.rating))
                    .unzip();
                delta_t.resize(pad_size, 0.0);
                rating.resize(pad_size, 0);
                let delta_t = Tensor::<B, 2>::from_floats(
                    TensorData::new(
                        delta_t,
                        Shape {
                            dims: vec![1, pad_size],
                        },
                    ),
                    device,
                );
                let rating = Tensor::<B, 2>::from_data(
                    TensorData::new(
                        rating,
                        Shape {
                            dims: vec![1, pad_size],
                        },
                    ),
                    device,
                );
                (delta_t, rating)
            })
            .unzip();

        let (delta_ts, labels, weights) = weighted_items
            .iter()
            .map(|weighted_item| {
                let current = weighted_item.item.current();
                let delta_t: Tensor<B, 1> = Tensor::from_floats([current.delta_t], device);
                let label = match current.rating {
                    1 => 0,
                    _ => 1,
                };
                let label: Tensor<B, 1, Int> = Tensor::from_ints([label], device);
                let weight: Tensor<B, 1> = Tensor::from_floats([weighted_item.weight], device);
                (delta_t, label, weight)
            })
            .multiunzip();

        let t_historys = Tensor::cat(time_histories, 0).transpose().to_device(device);
        let r_historys = Tensor::cat(rating_histories, 0)
            .transpose()
            .to_device(device);
        let delta_ts = Tensor::cat(delta_ts, 0).to_device(device);
        let labels = Tensor::cat(labels, 0).to_device(device);
        let weights = Tensor::cat(weights, 0).to_device(device);

        FSRSBatch {
            t_historys,
            r_historys,
            delta_ts,
            labels,
            weights,
        }
    }
}

pub(crate) struct FSRSDataset {
    pub(crate) items: Vec<WeightedFSRSItem>,
}

impl Dataset<WeightedFSRSItem> for FSRSDataset {
    fn len(&self) -> usize {
        self.items.len()
    }

    fn get(&self, index: usize) -> Option<WeightedFSRSItem> {
        self.items.get(index).cloned()
    }
}

impl From<Vec<WeightedFSRSItem>> for FSRSDataset {
    fn from(items: Vec<WeightedFSRSItem>) -> Self {
        Self {
            items: sort_items_by_review_length(items),
        }
    }
}

// ========== Batch Shuffle ==========

#[derive(Clone)]
pub(crate) struct BatchTensorDataset<B: Backend> {
    dataset: Vec<FSRSBatch<B>>,
}

impl<B: Backend> BatchTensorDataset<B> {
    pub fn new(dataset: FSRSDataset, batch_size: usize) -> Self {
        let device = B::Device::default();
        let batcher = FSRSBatcher::<B>::new();
        let dataset = dataset
            .items
            .chunks(batch_size)
            .map(|items| batcher.batch(items.to_vec(), &device))
            .collect();
        Self { dataset }
    }
}

impl<B: Backend> BatchTensorDataset<B> {
    fn get(&self, index: usize) -> Option<FSRSBatch<B>> {
        self.dataset.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

pub(crate) struct ShuffleDataLoader<B: Backend> {
    dataset: BatchTensorDataset<B>,
    rng: Mutex<StdRng>,
}

impl<B: Backend> ShuffleDataLoader<B> {
    pub fn new(dataset: BatchTensorDataset<B>, seed: u64) -> Self {
        Self {
            dataset,
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
        }
    }
}

pub(crate) struct ShuffleDataLoaderIterator<B: Backend> {
    current_index: usize,
    indices: Vec<usize>,
    dataset: BatchTensorDataset<B>,
}

impl<B: Backend> ShuffleDataLoaderIterator<B> {
    pub(crate) fn new(dataset: BatchTensorDataset<B>, indices: Vec<usize>) -> Self {
        Self {
            current_index: 0,
            indices,
            dataset,
        }
    }
}

impl<B: Backend> Iterator for ShuffleDataLoaderIterator<B> {
    type Item = FSRSBatch<B>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(index) = self.indices.get(self.current_index) {
            self.current_index += 1;
            return self.dataset.get(*index);
        }
        None
    }
}

impl<B: Backend> DataLoaderIterator<FSRSBatch<B>> for ShuffleDataLoaderIterator<B> {
    fn progress(&self) -> Progress {
        Progress::new(self.current_index, self.dataset.len())
    }
}

impl<B: Backend> ShuffleDataLoader<B> {
    pub(crate) fn iter(&self) -> ShuffleDataLoaderIterator<B> {
        let mut indices: Vec<_> = (0..self.dataset.len()).collect();
        indices.shuffle(&mut *self.rng.lock().unwrap());
        ShuffleDataLoaderIterator::new(self.dataset.clone(), indices)
    }
}

// ========== Dataset Helper Functions ==========

pub(crate) fn sort_items_by_review_length(
    mut weighted_items: Vec<WeightedFSRSItem>,
) -> Vec<WeightedFSRSItem> {
    weighted_items.sort_by_cached_key(|weighted_item| weighted_item.item.reviews.len());
    weighted_items
}

pub(crate) fn constant_weighted_fsrs_items(items: Vec<FSRSItem>) -> Vec<WeightedFSRSItem> {
    items
        .into_iter()
        .map(|item| WeightedFSRSItem { weight: 1.0, item })
        .collect()
}

/// The input items should be sorted by the review timestamp.
pub(crate) fn recency_weighted_fsrs_items(items: Vec<FSRSItem>) -> Vec<WeightedFSRSItem> {
    let length = (items.len() as f32 - 1.0).max(1.0);
    items
        .into_iter()
        .enumerate()
        .map(|(idx, item)| WeightedFSRSItem {
            weight: 0.25 + 0.75 * (idx as f32 / length).powi(3),
            item,
        })
        .collect()
}

pub(crate) fn prepare_training_data(items: Vec<FSRSItem>) -> (Vec<FSRSItem>, Vec<FSRSItem>) {
    let (mut dataset_for_initialization, mut trainset) = items
        .clone()
        .into_iter()
        .partition(|item| item.long_term_review_cnt() == 1);
    if std::env::var("FSRS_NO_OUTLIER").is_err() {
        (dataset_for_initialization, trainset) = filter_outlier(dataset_for_initialization, items);
    }
    (dataset_for_initialization, trainset)
}

// ========== BCE Loss ==========

pub struct BCELoss<B: Backend> {
    backend: PhantomData<B>,
}

impl<B: Backend> BCELoss<B> {
    pub const fn new() -> Self {
        Self {
            backend: PhantomData,
        }
    }
    pub fn forward(
        &self,
        retrievability: Tensor<B, 1>,
        labels: Tensor<B, 1>,
        weights: Tensor<B, 1>,
        mean: Reduction,
    ) -> Tensor<B, 1> {
        let loss = (labels.clone() * retrievability.clone().log()
            + (-labels + 1) * (-retrievability + 1).log())
            * weights.clone();
        match mean {
            Reduction::Mean => loss.mean().neg(),
            Reduction::Sum => loss.sum().neg(),
            Reduction::Auto => (loss.sum() / weights.sum()).neg(),
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward_classification(
        &self,
        t_historys: Tensor<B, 2>,
        r_historys: Tensor<B, 2>,
        delta_ts: Tensor<B, 1>,
        labels: Tensor<B, 1, Int>,
        weights: Tensor<B, 1>,
        reduce: Reduction,
    ) -> Tensor<B, 1> {
        let state = self.forward(t_historys, r_historys, None);
        let retrievability = self
            .power_forgetting_curve(delta_ts, state.stability)
            .clamp(MIN_RETRIEVABILITY, MAX_RETRIEVABILITY);
        BCELoss::new().forward(retrievability, labels.float(), weights, reduce)
    }
}

impl<B: AutodiffBackend> Model<B> {
    fn add_manual_weight_gradient(
        &self,
        mut gradients: B::Gradients,
        manual_grad: &[f32],
    ) -> B::Gradients {
        let grad_tensor = self.w.grad(&gradients).unwrap();
        let device = grad_tensor.device();
        let grad_len = grad_tensor.dims()[0];
        let mut data = vec![0.0f32; grad_len];
        for (dst, src) in data.iter_mut().zip(manual_grad.iter()) {
            *dst = *src;
        }
        let manual_tensor = Tensor::from_floats(data.as_slice(), &device);
        let updated_grad = grad_tensor + manual_tensor;
        self.w.grad_remove(&mut gradients);
        self.w.grad_replace(&mut gradients, updated_grad);
        gradients
    }

    fn freeze_initial_stability(&self, mut grad: B::Gradients) -> B::Gradients {
        let grad_tensor = self.w.grad(&grad).unwrap();
        let device = grad_tensor.device();
        let updated_grad_tensor = grad_tensor.slice_assign([0..4], Tensor::zeros([4], &device));

        self.w.grad_remove(&mut grad);
        self.w.grad_replace(&mut grad, updated_grad_tensor);
        grad
    }

    fn freeze_short_term_stability(&self, mut grad: B::Gradients) -> B::Gradients {
        let grad_tensor = self.w.grad(&grad).unwrap();
        let device = grad_tensor.device();
        let updated_grad_tensor = if grad_tensor.dims()[0] >= 35 {
            grad_tensor.slice_assign([16..27], Tensor::zeros([11], &device))
        } else {
            grad_tensor.slice_assign([17..20], Tensor::zeros([3], &device))
        };

        self.w.grad_remove(&mut grad);
        self.w.grad_replace(&mut grad, updated_grad_tensor);
        grad
    }
}

// ========== Progress Types ==========

#[derive(Debug, Default, Clone)]
pub struct ProgressState {
    pub epoch: usize,
    pub epoch_total: usize,
    pub items_processed: usize,
    pub items_total: usize,
}

#[derive(Debug, Default)]
pub struct CombinedProgressState {
    pub want_abort: bool,
    pub splits: Vec<ProgressState>,
    finished: bool,
}

impl CombinedProgressState {
    pub fn new_shared() -> Arc<Mutex<Self>> {
        Default::default()
    }

    pub fn current(&self) -> usize {
        self.splits.iter().map(|s| s.current()).sum()
    }

    pub fn total(&self) -> usize {
        self.splits.iter().map(|s| s.total()).sum()
    }

    pub const fn finished(&self) -> bool {
        self.finished
    }
}

#[derive(Clone)]
pub struct ProgressCollector {
    pub state: Arc<Mutex<CombinedProgressState>>,
    pub interrupter: TrainingInterrupter,
    /// The index of the split we should update.
    pub index: usize,
}

impl ProgressCollector {
    pub fn new(state: Arc<Mutex<CombinedProgressState>>, index: usize) -> Self {
        Self {
            state,
            interrupter: Default::default(),
            index,
        }
    }
}

impl ProgressState {
    pub const fn current(&self) -> usize {
        self.epoch.saturating_sub(1) * self.items_total + self.items_processed
    }

    pub const fn total(&self) -> usize {
        self.epoch_total * self.items_total
    }
}

impl MetricsRenderer for ProgressCollector {
    fn update_train(&mut self, _state: MetricState) {}

    fn update_valid(&mut self, _state: MetricState) {}

    fn render_train(&mut self, item: TrainingProgress) {
        let mut info = self.state.lock().unwrap();
        let split = &mut info.splits[self.index];
        split.epoch = item.epoch;
        split.epoch_total = item.epoch_total;
        split.items_processed = item.progress.items_processed;
        split.items_total = item.progress.items_total;
        if info.want_abort {
            self.interrupter.stop();
        }
    }

    fn render_valid(&mut self, _item: TrainingProgress) {}
}

// ========== Training Config ==========

#[derive(Config)]
pub(crate) struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = false)]
    pub enable_sched_penalties: bool,
    #[config(default = 8)]
    pub num_epochs: usize,
    #[config(default = 1024)]
    pub batch_size: usize,
    #[config(default = 2023)]
    pub seed: u64,
    #[config(default = 2e-2)]
    pub learning_rate: f64,
    #[config(default = 1024)]
    pub max_seq_len: usize,
}

// ========== ComputeParametersInput ==========

/// Input parameters for computing FSRS parameters
#[derive(Clone, Debug)]
pub struct ComputeParametersInput {
    /// The training set containing review history
    pub train_set: Vec<FSRSItem>,
    /// Optional progress tracking
    pub progress: Option<Arc<Mutex<CombinedProgressState>>>,
    /// Whether to enable short-term memory parameters
    pub enable_short_term: bool,
    /// Whether to enable FSRS-7 schedule penalties (penalty 1 & 2)
    pub enable_sched_penalties: bool,
    /// Number of relearning steps
    pub num_relearning_steps: Option<usize>,
}

impl Default for ComputeParametersInput {
    fn default() -> Self {
        Self {
            train_set: Vec::new(),
            progress: None,
            enable_short_term: true,
            enable_sched_penalties: false,
            num_relearning_steps: None,
        }
    }
}

fn normalize_training_set(train_set: Vec<FSRSItem>) -> Vec<FSRSItem> {
    train_set
        .into_iter()
        .map(|mut item| {
            for review in &mut item.reviews {
                review.delta_t = review.delta_t.max(0.0);
            }
            item
        })
        .collect()
}

/// Computes optimized parameters for the FSRS model based on training data.
///
/// This function trains the model on the provided dataset and returns optimized parameters.
///
/// # Arguments
/// * `input` - Input parameters including the training dataset and configuration
///
/// # Returns
/// A `Result<Vec<f32>>` containing the optimized parameters
pub fn compute_parameters(
    ComputeParametersInput {
        train_set,
        progress,
        enable_short_term,
        enable_sched_penalties,
        num_relearning_steps,
        ..
    }: ComputeParametersInput,
) -> Result<Vec<f32>> {
    let finish_progress = || {
        if let Some(progress) = &progress {
            progress.lock().unwrap().finished = true;
        }
    };

    let train_set = normalize_training_set(train_set);
    let (_, train_set) = prepare_training_data(train_set);
    if train_set.len() < 8 {
        finish_progress();
        return Ok(DEFAULT_PARAMETERS.to_vec());
    }

    let initialized_parameters = DEFAULT_PARAMETERS.to_vec();
    if train_set.len() < 64 {
        finish_progress();
        return Ok(initialized_parameters);
    }
    let config = TrainingConfig::new(
        ModelConfig {
            freeze_initial_stability: !enable_short_term,
            initial_stability: None,
            initial_forgetting_curve: None,
            freeze_short_term_stability: !enable_short_term,
            num_relearning_steps: num_relearning_steps.unwrap_or(1),
        },
        AdamConfig::new()
            .with_beta_1(0.8)
            .with_beta_2(0.85)
            .with_epsilon(1e-8),
    )
    .with_enable_sched_penalties(enable_sched_penalties);
    let mut weighted_train_set = recency_weighted_fsrs_items(train_set);
    weighted_train_set.retain(|item| item.item.reviews.len() <= config.max_seq_len);

    if let Some(progress) = &progress {
        let progress_state = ProgressState {
            epoch_total: config.num_epochs,
            items_total: weighted_train_set.len(),
            epoch: 0,
            items_processed: 0,
        };
        progress.lock().unwrap().splits = vec![progress_state];
    }
    let model = train::<Autodiff<B>>(
        weighted_train_set.clone(),
        weighted_train_set,
        &initialized_parameters,
        &config,
        progress.clone().map(|p| ProgressCollector::new(p, 0)),
    );

    let optimized_parameters = model
        .inspect_err(|_e| {
            finish_progress();
        })?
        .w
        .val()
        .to_data()
        .to_vec()
        .unwrap();

    finish_progress();

    if optimized_parameters
        .iter()
        .any(|parameter: &f32| parameter.is_infinite())
    {
        return Err(FSRSError::InvalidInput);
    }

    Ok(optimized_parameters)
}

pub fn benchmark(
    ComputeParametersInput {
        train_set,
        enable_short_term,
        enable_sched_penalties,
        num_relearning_steps,
        ..
    }: ComputeParametersInput,
) -> Vec<f32> {
    let train_set = normalize_training_set(train_set);
    let (_, train_set) = prepare_training_data(train_set);
    let initialized_parameters = DEFAULT_PARAMETERS.to_vec();
    let mut config = TrainingConfig::new(
        ModelConfig {
            freeze_initial_stability: !enable_short_term,
            initial_stability: None,
            initial_forgetting_curve: None,
            freeze_short_term_stability: !enable_short_term,
            num_relearning_steps: num_relearning_steps.unwrap_or(1),
        },
        AdamConfig::new()
            .with_beta_1(0.8)
            .with_beta_2(0.85)
            .with_epsilon(1e-8),
    )
    .with_enable_sched_penalties(enable_sched_penalties);
    // save RAM and speed up training
    config.max_seq_len = 64;
    let mut weighted_train_set = recency_weighted_fsrs_items(train_set);
    weighted_train_set.retain(|item| item.item.reviews.len() <= config.max_seq_len);
    let model = train::<Autodiff<B>>(
        weighted_train_set.clone(),
        weighted_train_set,
        &initialized_parameters,
        &config,
        None,
    );
    let parameters: Vec<f32> = model.unwrap().w.val().to_data().to_vec::<f32>().unwrap();
    parameters
}

fn train<B: AutodiffBackend>(
    train_set: Vec<WeightedFSRSItem>,
    test_set: Vec<WeightedFSRSItem>,
    initial_parameters: &[f32],
    config: &TrainingConfig,
    progress: Option<ProgressCollector>,
) -> Result<Model<B>> {
    B::seed(config.seed);

    // Training data
    let total_size = train_set.len();
    let iterations = (total_size / config.batch_size + 1) * config.num_epochs;
    let batch_dataset =
        BatchTensorDataset::<B>::new(FSRSDataset::from(train_set), config.batch_size);
    let dataloader_train = ShuffleDataLoader::new(batch_dataset, config.seed);

    let batch_dataset = BatchTensorDataset::<B::InnerBackend>::new(
        FSRSDataset::from(test_set.clone()),
        config.batch_size,
    );
    let dataloader_valid = ShuffleDataLoader::new(batch_dataset, config.seed);

    let mut lr_scheduler = CosineAnnealingLR::init(iterations as f64, config.learning_rate);
    let interrupter = TrainingInterrupter::new();
    let mut renderer: Box<dyn MetricsRenderer> = match progress {
        Some(mut progress) => {
            progress.interrupter = interrupter.clone();
            Box::new(progress)
        }
        None => Box::new(NoProgress {}),
    };

    let mut model: Model<B> = parameters_to_model::<B>(initial_parameters, &B::Device::default());
    let schedule_penalty = schedule_penalty_fn();
    let l2_penalty = l2_penalty_fn();
    let init_w = model.w.val();
    let init_w_vec = init_w.to_data().to_vec::<f32>().unwrap();
    let mut optim = config.optimizer.init::<B, Model<B>>();

    let mut best_loss = f64::INFINITY;
    let mut best_model = model.clone();
    for epoch in 1..=config.num_epochs {
        let mut iterator = dataloader_train.iter();
        let mut iteration = 0;
        while let Some(item) = iterator.next() {
            iteration += 1;
            let real_batch_size = item.delta_ts.shape().dims[0];
            let lr = LrScheduler::step(&mut lr_scheduler);
            let progress = iterator.progress();
            let l2_weight = L2_PENALTY_WEIGHT;
            let w_vec = model.w.val().to_data().to_vec::<f32>().unwrap();
            let (_l2_penalty_value, mut manual_grad) = l2_penalty(
                &w_vec,
                &init_w_vec,
                real_batch_size,
                total_size,
                l2_weight,
                &training_v7::PARAMS_STDDEV,
            );
            let (_schedule_value, schedule_grad) =
                schedule_penalty(&w_vec, real_batch_size, config.enable_sched_penalties);
            let inv_total = 1.0 / total_size as f64;
            for i in 0..manual_grad.len().min(schedule_grad.len()) {
                manual_grad[i] += (schedule_grad[i] * inv_total) as f32;
            }
            let loss = model.forward_classification(
                item.t_historys,
                item.r_historys,
                item.delta_ts,
                item.labels,
                item.weights,
                Reduction::Sum,
            );
            let mut gradients = loss.backward();
            gradients = model.add_manual_weight_gradient(gradients, &manual_grad);
            if config.model.freeze_initial_stability {
                gradients = model.freeze_initial_stability(gradients);
            }
            if config.model.freeze_short_term_stability {
                gradients = model.freeze_short_term_stability(gradients);
            }
            let grads = GradientsParams::from_grads(gradients, &model);
            model = optim.step(lr, model, grads);
            model.w = parameter_clipper(
                model.w,
                !config.model.freeze_short_term_stability,
            );
            renderer.render_train(TrainingProgress {
                progress,
                epoch,
                epoch_total: config.num_epochs,
                iteration,
            });

            if interrupter.should_stop() {
                break;
            }
        }

        if interrupter.should_stop() {
            break;
        }

        let model_valid = model.valid();
        let mut loss_valid = 0.0;
        for batch in dataloader_valid.iter() {
            let real_batch_size = batch.delta_ts.shape().dims[0];
            let l2_weight = L2_PENALTY_WEIGHT;
            let w_vec = model_valid.w.val().to_data().to_vec::<f32>().unwrap();
            let (l2_penalty_value, _) = l2_penalty(
                &w_vec,
                &init_w_vec,
                real_batch_size,
                total_size,
                l2_weight,
                &training_v7::PARAMS_STDDEV,
            );
            let (schedule_value, _) =
                schedule_penalty(&w_vec, real_batch_size, config.enable_sched_penalties);
            let schedule_penalty = schedule_value / total_size as f64;
            let loss = model_valid.forward_classification(
                batch.t_historys,
                batch.r_historys,
                batch.delta_ts,
                batch.labels,
                batch.weights,
                Reduction::Sum,
            );
            let loss = loss.into_scalar().to_f64();
            loss_valid += loss + l2_penalty_value + schedule_penalty;

            if interrupter.should_stop() {
                break;
            }
        }
        loss_valid /= test_set.len() as f64;
        info!("epoch: {:?} loss: {:?}", epoch, loss_valid);
        if loss_valid < best_loss {
            best_loss = loss_valid;
            best_model = model.clone();
        }
    }

    info!("best_loss: {:?}", best_loss);

    if interrupter.should_stop() {
        return Err(FSRSError::Interrupted);
    }

    Ok(best_model)
}

struct NoProgress {}

impl MetricsRenderer for NoProgress {
    fn update_train(&mut self, _state: MetricState) {}

    fn update_valid(&mut self, _state: MetricState) {}

    fn render_train(&mut self, _item: TrainingProgress) {}

    fn render_valid(&mut self, _item: TrainingProgress) {}
}

#[cfg(test)]
mod tests {
    use std::fs::create_dir_all;
    use std::path::Path;
    use std::thread;
    use std::time::Duration;

    use super::*;
    use crate::convertor_tests::anki21_sample_file_converted_to_fsrs;
    use crate::convertor_tests::data_from_csv;
    use crate::model::FSRS;
    use crate::DEFAULT_PARAMETERS;
    use burn::backend::NdArray;
    use burn::tensor::Shape;
    use itertools::Itertools;
    use log::LevelFilter;

    #[test]
    fn test_normalize_training_set_clamps_negative_intervals() {
        let train_set = vec![FSRSItem {
            reviews: vec![
                crate::FSRSReview {
                    rating: 1,
                    delta_t: -0.2,
                },
                crate::FSRSReview {
                    rating: 3,
                    delta_t: 0.49,
                },
                crate::FSRSReview {
                    rating: 3,
                    delta_t: 0.51,
                },
            ],
        }];
        let normalized = normalize_training_set(train_set);
        let days: Vec<f32> = normalized[0].reviews.iter().map(|r| r.delta_t).collect();
        assert_eq!(days, vec![0.0, 0.49, 0.51]);
    }

    #[test]
    fn test_compute_parameters_small_dataset_fsrs7_defaults() {
        let parameters = compute_parameters(ComputeParametersInput {
            train_set: vec![],
            progress: None,
            enable_short_term: true,
            enable_sched_penalties: true,
            num_relearning_steps: None,
        })
        .unwrap();
        assert_eq!(parameters, DEFAULT_PARAMETERS.to_vec());
    }

    #[test]
    fn test_sched_penalties_default_to_disabled() {
        assert!(!ComputeParametersInput::default().enable_sched_penalties);
        let config =
            TrainingConfig::new(ModelConfig::default(), AdamConfig::new().with_epsilon(1e-8));
        assert!(!config.enable_sched_penalties);
    }

    #[test]
    fn test_compute_parameters_fsrs7_with_same_day_only_items_no_panic() {
        let train_set = vec![
            FSRSItem {
                reviews: vec![
                    crate::FSRSReview {
                        rating: 2,
                        delta_t: 0.0,
                    },
                    crate::FSRSReview {
                        rating: 3,
                        delta_t: 0.5,
                    },
                ],
            },
            FSRSItem {
                reviews: vec![
                    crate::FSRSReview {
                        rating: 1,
                        delta_t: 0.0,
                    },
                    crate::FSRSReview {
                        rating: 2,
                        delta_t: 0.25,
                    },
                ],
            },
        ];

        let parameters = compute_parameters(ComputeParametersInput {
            train_set,
            progress: None,
            enable_short_term: true,
            enable_sched_penalties: true,
            num_relearning_steps: None,
        });

        assert!(parameters.is_ok());
        assert_eq!(parameters.unwrap().len(), 35);
    }

    #[test]
    fn test_training() {
        if std::env::var("SKIP_TRAINING").is_ok() {
            println!("Skipping test in CI");
            return;
        }

        let artifact_dir = std::env::var("BURN_LOG");

        if let Ok(artifact_dir) = artifact_dir {
            let _ = create_dir_all(&artifact_dir);
            let log_file = Path::new(&artifact_dir).join("training.log");
            fern::Dispatch::new()
                .format(|out, message, record| {
                    out.finish(format_args!(
                        "[{}][{}] {}",
                        record.target(),
                        record.level(),
                        message
                    ))
                })
                .level(LevelFilter::Info)
                .chain(fern::log_file(log_file).unwrap())
                .apply()
                .unwrap();
        }
        for items in [anki21_sample_file_converted_to_fsrs(), data_from_csv()] {
            for enable_short_term in [true, false] {
                let progress = CombinedProgressState::new_shared();
                let progress2 = Some(progress.clone());
                thread::spawn(move || {
                    let mut finished = false;
                    while !finished {
                        thread::sleep(Duration::from_millis(500));
                        let guard = progress.lock().unwrap();
                        finished = guard.finished();
                        println!("progress: {}/{}", guard.current(), guard.total());
                    }
                });

                let parameters = compute_parameters(ComputeParametersInput {
                    train_set: items.clone(),
                    progress: progress2,
                    enable_short_term,
                    enable_sched_penalties: true,
                    num_relearning_steps: None,
                })
                .unwrap();
                dbg!(&parameters);
                assert_eq!(parameters.len(), 35);

                // evaluate
                let model = FSRS::new(&parameters).unwrap();
                let metrics = model.evaluate(items.clone(), |_| true).unwrap();
                dbg!(&metrics);
            }
        }
    }

    #[test]
    fn test_manual_l2_penalty_matches_autodiff_gradient() {
        type B = Autodiff<NdArray<f32>>;
        let config = ModelConfig::default();
        let model: Model<B> = config.init();
        let device = model.w.device();
        let w_vec = model.w.val().to_data().to_vec::<f32>().unwrap();
        let mut init_w_vec = w_vec.clone();
        for (i, init) in init_w_vec.iter_mut().enumerate() {
            *init -= 0.05 * ((i + 1) as f32) / (PENALTY_GRAD_LEN as f32);
        }

        let init_w = Tensor::from_floats(init_w_vec.as_slice(), &device);
        let params_stddev = Tensor::from_floats(training_v7::PARAMS_STDDEV, &device);
        let penalty = (model.w.val() - init_w)
            .powi_scalar(2)
            .div(params_stddev.powi_scalar(2))
            .sum()
            .mul_scalar(L2_PENALTY_WEIGHT * 512.0 / 1000.0);
        let expected_value = penalty.clone().into_scalar().to_f64();
        let gradients = penalty.backward();
        let expected_grad = model
            .w
            .grad(&gradients)
            .unwrap()
            .to_data()
            .to_vec::<f32>()
            .unwrap();

        let (actual_value, actual_grad) = training_v7::l2_penalty_value_and_grad(
            &w_vec,
            &init_w_vec,
            512,
            1000,
            L2_PENALTY_WEIGHT,
            &training_v7::PARAMS_STDDEV,
        );
        assert!(
            (actual_value - expected_value).abs() < 1e-6,
            "l2 value mismatch actual={} expected={}",
            actual_value,
            expected_value
        );
        for (expected, actual) in expected_grad.iter().zip(actual_grad.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_lr_scheduler() {
        let mut lr_scheduler = CosineAnnealingLR::init(5.0, 4e-2);
        let lrs = (1..=11)
            .map(|_| {
                LrScheduler::step(&mut lr_scheduler);
                lr_scheduler.current_lr
            })
            .step_by(1)
            .collect::<Vec<_>>();
        use crate::test_helpers::TestHelper;
        lrs.assert_approx_eq([
            0.04,
            0.03618033988749895,
            0.026180339887498946,
            0.013819660112501051,
            0.0038196601125010526,
            0.0,
            0.003819660112501051,
            0.013819660112501048,
            0.026180339887498943,
            0.03618033988749895,
            0.039999999999999994,
        ]);
    }

    #[test]
    fn test_parameter_clipper_works() {
        use burn::backend::ndarray::NdArrayDevice;
        static DEVICE: NdArrayDevice = NdArrayDevice::Cpu;
        use crate::test_helpers::Tensor;
        let tensor = Tensor::from_floats(
            [0.0, -1000.0, 1000.0, 0.0, 1000.0, -1000.0, 1.0, 0.25, -0.1],
            &DEVICE,
        );

        let param = parameter_clipper(Param::from_tensor(tensor), true);
        let values = &param.to_data().to_vec::<f32>().unwrap();

        assert_eq!(
            values,
            &[0.0001, 0.0001, 100.0, 0.0001, 10.0, 0.001, 1.0, 0.25, 0.0]
        );
    }

    #[test]
    fn test_fsrs7_clipper_monotonic_bounds() {
        let mut params = vec![1000.0; 35];
        params[27] = -1.0;
        params[28] = 10.0;
        params[29] = 0.1;
        params[30] = 2.0;
        let clipped = clip_parameters(&params);
        assert_eq!(clipped.len(), 35);
        assert!(clipped[1] >= clipped[0]);
        assert!(clipped[2] >= clipped[1]);
        assert!(clipped[3] >= clipped[2]);
        assert!(clipped[28] >= clipped[27]);
        assert!(clipped[30] >= clipped[29]);
    }

    #[test]
    fn test_fsrs7_clipper_respects_disable_short_term() {
        use burn::backend::ndarray::NdArrayDevice;
        use crate::test_helpers::Tensor;
        let params = Tensor::from_floats(DEFAULT_PARAMETERS, &NdArrayDevice::Cpu);
        let clipped_on = parameter_clipper(Param::from_tensor(params.clone()), true)
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        let clipped_off = parameter_clipper(Param::from_tensor(params), false)
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(clipped_on[26] > 0.0);
        assert_eq!(clipped_off[26], 0.0);
    }

    #[test]
    fn test_fsrs7_clipper_handles_nan_without_panic() {
        let mut params = DEFAULT_PARAMETERS.to_vec();
        for idx in [0, 1, 2, 3, 27, 28, 29, 30] {
            params[idx] = f32::NAN;
        }
        let clipped = clip_parameters(&params);
        assert_eq!(clipped.len(), 35);
        assert!(clipped.iter().all(|v| v.is_finite()));
        assert!(clipped[1] >= clipped[0]);
        assert!(clipped[2] >= clipped[1]);
        assert!(clipped[3] >= clipped[2]);
        assert!(clipped[28] >= clipped[27]);
        assert!(clipped[30] >= clipped[29]);
    }

    #[test]
    fn test_simple_dataloader() {
        let train_set = anki21_sample_file_converted_to_fsrs()
            .into_iter()
            .sorted_by_cached_key(|item| item.reviews.len())
            .collect();
        let (_pre_train_set, train_set) = prepare_training_data(train_set);
        let dataset = FSRSDataset::from(constant_weighted_fsrs_items(train_set));
        let batch_size = 512;
        let seed = 114514;
        type Backend = NdArray<f32>;

        let dataset = BatchTensorDataset::<Backend>::new(dataset, batch_size);
        let dataloader = ShuffleDataLoader::new(dataset, seed);
        let mut iterator = dataloader.iter();
        let batch = iterator.next().unwrap();
        assert_eq!(
            batch.t_historys.shape(),
            Shape {
                dims: vec![5, batch_size]
            }
        );
        let batch = iterator.next().unwrap();
        assert_eq!(
            batch.t_historys.shape(),
            Shape {
                dims: vec![3, batch_size]
            }
        );

        let lengths = iterator
            .map(|batch| batch.t_historys.shape().dims[0])
            .collect::<Vec<_>>();
        assert_eq!(
            lengths,
            [
                3, 5, 19, 7, 2, 4, 4, 3, 6, 13, 4, 4, 7, 4, 6, 48, 11, 8, 9, 1, 2, 5, 3, 5, 6, 3
            ]
        );

        let mut iterator = dataloader.iter();
        let batch = iterator.next().unwrap();
        assert_eq!(
            batch.t_historys.shape(),
            Shape {
                dims: vec![4, batch_size]
            }
        );
        let batch = iterator.next().unwrap();
        assert_eq!(
            batch.t_historys.shape(),
            Shape {
                dims: vec![2, batch_size]
            }
        );

        let lengths = iterator
            .map(|batch| batch.t_historys.shape().dims[0])
            .collect::<Vec<_>>();
        assert_eq!(
            lengths,
            [
                11, 4, 5, 3, 1, 3, 13, 5, 4, 6, 2, 6, 19, 6, 3, 7, 4, 3, 48, 9, 5, 8, 5, 4, 3, 7
            ]
        );
    }

    #[test]
    fn test_from_anki() {
        use burn::data::dataloader::Dataset;
        use burn::tensor::Tolerance;

        let dataset = FSRSDataset::from(constant_weighted_fsrs_items(
            anki21_sample_file_converted_to_fsrs(),
        ));
        assert_eq!(
            dataset.get(704).unwrap().item,
            FSRSItem {
                reviews: vec![
                    crate::FSRSReview {
                        rating: 4,
                        delta_t: 0.0
                    },
                    crate::FSRSReview {
                        rating: 3,
                        delta_t: 3.0
                    }
                ],
            }
        );

        let batcher = FSRSBatcher::<NdArray<f32>>::new();
        use burn::backend::ndarray::NdArrayDevice;
        static DEVICE: NdArrayDevice = NdArrayDevice::Cpu;
        use burn::data::dataloader::DataLoaderBuilder;
        let dataloader = DataLoaderBuilder::new(batcher)
            .batch_size(1)
            .shuffle(42)
            .num_workers(4)
            .build(dataset);
        dbg!(
            dataloader
                .iter()
                .next()
                .expect("loader is empty")
                .r_historys
        );
    }

    #[test]
    fn test_batcher() {
        use burn::backend::ndarray::NdArrayDevice;
        use burn::tensor::Tolerance;
        static DEVICE: NdArrayDevice = NdArrayDevice::Cpu;
        let batcher = FSRSBatcher::<NdArray<f32>>::new();
        let items = [
            FSRSItem {
                reviews: [(4, 0), (3, 5)]
                    .into_iter()
                    .map(|(rating, delta_t)| crate::FSRSReview {
                        rating,
                        delta_t: delta_t as f32,
                    })
                    .collect(),
            },
            FSRSItem {
                reviews: [(4, 0), (3, 5), (3, 11)]
                    .into_iter()
                    .map(|(rating, delta_t)| crate::FSRSReview {
                        rating,
                        delta_t: delta_t as f32,
                    })
                    .collect(),
            },
            FSRSItem {
                reviews: [(4, 0), (3, 2)]
                    .into_iter()
                    .map(|(rating, delta_t)| crate::FSRSReview {
                        rating,
                        delta_t: delta_t as f32,
                    })
                    .collect(),
            },
            FSRSItem {
                reviews: [(4, 0), (3, 2), (3, 6)]
                    .into_iter()
                    .map(|(rating, delta_t)| crate::FSRSReview {
                        rating,
                        delta_t: delta_t as f32,
                    })
                    .collect(),
            },
            FSRSItem {
                reviews: [(4, 0), (3, 2), (3, 6), (3, 16)]
                    .into_iter()
                    .map(|(rating, delta_t)| crate::FSRSReview {
                        rating,
                        delta_t: delta_t as f32,
                    })
                    .collect(),
            },
            FSRSItem {
                reviews: [(4, 0), (3, 2), (3, 6), (3, 16), (3, 39)]
                    .into_iter()
                    .map(|(rating, delta_t)| crate::FSRSReview {
                        rating,
                        delta_t: delta_t as f32,
                    })
                    .collect(),
            },
            FSRSItem {
                reviews: [(1, 0), (1, 1)]
                    .into_iter()
                    .map(|(rating, delta_t)| crate::FSRSReview {
                        rating,
                        delta_t: delta_t as f32,
                    })
                    .collect(),
            },
            FSRSItem {
                reviews: [(1, 0), (1, 1), (3, 1)]
                    .into_iter()
                    .map(|(rating, delta_t)| crate::FSRSReview {
                        rating,
                        delta_t: delta_t as f32,
                    })
                    .collect(),
            },
        ];
        let items = items
            .into_iter()
            .map(|item| WeightedFSRSItem { weight: 1.0, item })
            .collect();
        let batch = batcher.batch(items, &DEVICE);
        batch.t_historys.to_data().assert_approx_eq::<f32>(
            &TensorData::from([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 5.0, 0.0, 2.0, 2.0, 2.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 6.0, 6.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 16.0, 0.0, 0.0],
            ]),
            Tolerance::absolute(1e-5),
        );
        batch.r_historys.to_data().assert_approx_eq::<f32>(
            &TensorData::from([
                [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 1.0, 1.0],
                [0.0, 3.0, 0.0, 3.0, 3.0, 3.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
            ]),
            Tolerance::absolute(1e-5),
        );

        batch.delta_ts.to_data().assert_approx_eq::<f32>(
            &TensorData::from([5.0, 11.0, 2.0, 6.0, 16.0, 39.0, 1.0, 1.0]),
            Tolerance::absolute(1e-5),
        );
        batch.labels.to_data().assert_approx_eq::<f32>(
            &TensorData::from([1, 1, 1, 1, 1, 1, 0, 1]),
            Tolerance::absolute(1e-5),
        );
    }

    #[test]
    fn test_filter_outlier() {
        let dataset = anki21_sample_file_converted_to_fsrs();
        let (mut dataset_for_initialization, mut trainset): (Vec<FSRSItem>, Vec<FSRSItem>) =
            dataset
                .into_iter()
                .partition(|item| item.long_term_review_cnt() == 1);
        assert_eq!(dataset_for_initialization.len(), 3315);
        assert_eq!(trainset.len(), 10975);
        (dataset_for_initialization, trainset) =
            filter_outlier(dataset_for_initialization, trainset);
        assert_eq!(dataset_for_initialization.len(), 3265);
        assert_eq!(trainset.len(), 10900);
    }

    #[test]
    fn test_filter_outlier_keeps_same_day_only_items_without_panic() {
        let dataset_for_initialization = vec![FSRSItem {
            reviews: vec![
                crate::FSRSReview {
                    rating: 3,
                    delta_t: 0.0,
                },
                crate::FSRSReview {
                    rating: 3,
                    delta_t: 2.0,
                },
            ],
        }];
        let same_day_only = FSRSItem {
            reviews: vec![
                crate::FSRSReview {
                    rating: 2,
                    delta_t: 0.0,
                },
                crate::FSRSReview {
                    rating: 3,
                    delta_t: 0.5,
                },
            ],
        };
        let trainset = vec![same_day_only.clone()];
        let (_filtered, trainset) = filter_outlier(dataset_for_initialization, trainset);
        assert_eq!(trainset, vec![same_day_only]);
    }

    #[test]
    fn test_filter_outlier_buckets_fractional_long_term_deltas() {
        let make_item = |delta_t: f32| FSRSItem {
            reviews: vec![
                crate::FSRSReview {
                    rating: 3,
                    delta_t: 0.0,
                },
                crate::FSRSReview { rating: 3, delta_t },
            ],
        };
        let mut dataset_for_initialization = vec![];
        dataset_for_initialization.extend((0..12).map(|_| make_item(1.2)));
        dataset_for_initialization.extend((0..12).map(|_| make_item(1.8)));
        let trainset = dataset_for_initialization.clone();
        let (filtered, trainset) = filter_outlier(dataset_for_initialization, trainset);
        assert_eq!(filtered.len(), 24);
        assert_eq!(trainset.len(), 24);
    }
}
