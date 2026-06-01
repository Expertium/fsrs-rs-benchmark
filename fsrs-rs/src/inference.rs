use itertools::izip;
use serde::Serialize;
use std::collections::HashMap;

use crate::training::{
    FSRSBatch, FSRSBatcher, recency_weighted_fsrs_items,
};
use crate::error::Result;
use crate::model::Model;
use crate::model::{FSRS, MemoryStateTensors};
use crate::training::BCELoss;
use crate::{FSRSError, FSRSItem};
use burn::nn::loss::Reduction;
use burn::tensor::ElementConversion;
use burn::tensor::cast::ToElement;
use burn::tensor::{Shape, Tensor, TensorData};
use burn::{data::dataloader::batcher::Batcher, tensor::backend::Backend};

/// FSRS-7 default parameters (35 values).
pub static DEFAULT_PARAMETERS: [f32; 35] = [
    0.041, 2.4175, 4.1283, 11.9709, 5.6385, 0.4468, 3.262, 2.3054, 0.1688, 1.3325, 0.3524, 0.0049,
    0.7503, 0.0896, 0.6625, 1.3, 0.882, 0.3072, 3.5875, 0.303, 0.0107, 0.2279, 2.6413, 0.5594, 1.3,
    2.5, 1.0, 0.0723, 0.1634, 0.5, 0.9555, 0.2245, 0.6232, 0.1362, 0.3862,
];
/// This is a slice for efficiency, and should be 35 in length.
pub type Parameters = [f32];

fn infer<B: Backend>(
    model: &Model<B>,
    batch: FSRSBatch<B>,
) -> (MemoryStateTensors<B>, Tensor<B, 1>) {
    let state = model.forward(batch.t_historys, batch.r_historys, None);
    let retrievability = model.power_forgetting_curve(batch.delta_ts, state.stability.clone());
    (state, retrievability)
}

#[derive(Debug, PartialEq, Clone, Copy, Serialize)]
pub struct MemoryState {
    pub stability: f32,
    pub difficulty: f32,
}

impl<B: Backend> From<MemoryStateTensors<B>> for MemoryState {
    fn from(m: MemoryStateTensors<B>) -> Self {
        Self {
            stability: m.stability.into_scalar().elem(),
            difficulty: m.difficulty.into_scalar().elem(),
        }
    }
}

#[derive(Default)]
struct RMatrixValue {
    predicted: f32,
    actual: f32,
    count: f32,
    weight: f32,
}

impl<B: Backend> FSRS<B> {
    fn items_to_tensors(&self, items: &[FSRSItem]) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let pad_size = items
            .iter()
            .map(|x| x.reviews.len())
            .max()
            .expect("FSRSItem is empty");
        let device = self.device();
        let (time_histories, rating_histories) = items
            .iter()
            .map(|item| {
                let (mut delta_t, mut rating): (Vec<_>, Vec<_>) =
                    item.reviews.iter().map(|r| (r.delta_t, r.rating)).unzip();
                delta_t.resize(pad_size, 0.0);
                rating.resize(pad_size, 0);
                let delta_t = Tensor::<B, 2>::from_floats(
                    TensorData::new(
                        delta_t,
                        Shape {
                            dims: vec![1, pad_size],
                        },
                    ),
                    &device,
                );
                let rating = Tensor::<B, 2>::from_data(
                    TensorData::new(
                        rating,
                        Shape {
                            dims: vec![1, pad_size],
                        },
                    ),
                    &device,
                );
                (delta_t, rating)
            })
            .unzip();

        let t_historys = Tensor::cat(time_histories, 0)
            .transpose()
            .to_device(&device); // [seq_len, batch_size]
        let r_historys = Tensor::cat(rating_histories, 0)
            .transpose()
            .to_device(&device); // [seq_len, batch_size]
        (t_historys, r_historys)
    }

    pub fn memory_state_batch(
        &self,
        items: Vec<FSRSItem>,
        starting_states: Vec<Option<MemoryState>>,
    ) -> Result<Vec<MemoryState>> {
        if items.is_empty() {
            return Ok(vec![]);
        }
        let (time_histories, rating_histories) = self.items_to_tensors(&items);
        let (stabilities, difficulties) = starting_states
            .iter()
            .map(|starting_state| {
                if let Some(state) = starting_state {
                    (state.stability, state.difficulty)
                } else {
                    (0.0, 0.0)
                }
            })
            .collect::<(Vec<f32>, Vec<f32>)>();
        let device = self.device();
        let starting_states = MemoryStateTensors {
            stability: Tensor::from_data(
                TensorData::new(
                    stabilities.clone(),
                    Shape {
                        dims: vec![stabilities.len()],
                    },
                ),
                &device,
            ),
            difficulty: Tensor::from_data(
                TensorData::new(
                    difficulties.clone(),
                    Shape {
                        dims: vec![difficulties.len()],
                    },
                ),
                &device,
            ),
        };
        let state = self
            .model()
            .forward(time_histories, rating_histories, Some(starting_states));
        let stability = state.stability.to_data().to_vec::<f32>().unwrap();
        let difficulty = state.difficulty.to_data().to_vec::<f32>().unwrap();
        Ok(stability
            .into_iter()
            .zip(difficulty)
            .map(|(stability, difficulty)| MemoryState {
                stability,
                difficulty,
            })
            .collect())
    }

    /// Determine how well the model and parameters predict performance.
    pub fn evaluate<F>(&self, items: Vec<FSRSItem>, mut progress: F) -> Result<ModelEvaluation>
    where
        F: FnMut(ItemProgress) -> bool,
    {
        if items.is_empty() {
            return Err(FSRSError::NotEnoughData);
        }
        let weighted_items = recency_weighted_fsrs_items(items);
        let device = self.device();
        let batcher = FSRSBatcher::new();
        let mut all_retrievability = vec![];
        let mut all_labels = vec![];
        let mut all_weights = vec![];
        let mut progress_info = ItemProgress {
            current: 0,
            total: weighted_items.len(),
        };
        let model = self.model();
        let mut r_matrix: HashMap<(u32, u32, u32), RMatrixValue> = HashMap::new();

        for chunk in weighted_items.chunks(512) {
            let batch = batcher.batch(chunk.to_vec(), &device);
            let (_state, retrievability) = infer::<B>(model, batch.clone());
            let pred = retrievability.clone().to_data().to_vec::<f32>().unwrap();
            let true_val = batch.labels.clone().to_data().to_vec::<i64>().unwrap();
            all_retrievability.push(retrievability);
            all_labels.push(batch.labels);
            all_weights.push(batch.weights);
            izip!(chunk, pred, true_val).for_each(|(weighted_item, p, y)| {
                let bin = weighted_item.item.r_matrix_index();
                let value = r_matrix.entry(bin).or_default();
                value.predicted += p;
                value.actual += y as f32;
                value.count += 1.0;
                value.weight += weighted_item.weight;
            });
            progress_info.current += chunk.len();
            if !progress(progress_info) {
                return Err(FSRSError::Interrupted);
            }
        }
        let rmse = (r_matrix
            .values()
            .map(|v| {
                let pred = v.predicted / v.count;
                let real = v.actual / v.count;
                (pred - real).powi(2) * v.weight
            })
            .sum::<f32>()
            / r_matrix.values().map(|v| v.weight).sum::<f32>())
        .sqrt();
        let all_retrievability = Tensor::cat(all_retrievability, 0);
        let all_labels = Tensor::cat(all_labels, 0).float();
        let all_weights = Tensor::cat(all_weights, 0);
        let loss =
            BCELoss::new().forward(all_retrievability, all_labels, all_weights, Reduction::Auto);
        Ok(ModelEvaluation {
            log_loss: loss.into_scalar().to_f32(),
            rmse_bins: rmse,
        })
    }
}

#[derive(Debug, Copy, Clone)]
pub struct ModelEvaluation {
    pub log_loss: f32,
    pub rmse_bins: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct ItemProgress {
    pub current: usize,
    pub total: usize,
}
