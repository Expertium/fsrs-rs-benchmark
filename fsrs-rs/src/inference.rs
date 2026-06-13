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

/// FSRS-7 finished-model default parameters (34 values; fsrs-autoresearch champion init_w).
/// Full index -> role layout in the comment below.
// Finished FSRS-7 (34 params). Layout by where each param is USED:
//  - 0-3   initial stability s0 (again, hard, good, easy)        [initial state]
//  - 4-5   init_d0, init_d1; 6 next_d_mult                        [difficulty update]
//  - 7-14  long-trace stability (sinc_base, sinc_s_exp, sinc_r_mult, fail_mult, fail_s_exp,
//          fail_r_mult, hard_penalty, easy_bonus)                 [slow stability update]
//  - 15-22 short-trace stability (same 8 roles)                   [fast stability update]
//  - 24 decay2, 26 base2, 27 base_weight1, 28 base_weight2, 29 s_weight_power1,
//    30 s_weight_power2, 31 d_weight, 32 d_decay                  [forgetting curve ONLY]
//  - 23 decay1, 25 base1, 33 s_decay1                             [forgetting curve AND the
//    fast-trace stability update — they build r1 (fast_component_recall), shared by both]
// (fail_d_exp dropped from both stability blocks vs the 36-param draft; values =
// fsrs-autoresearch FSRS7_DEFAULT_35_VALUES, rounded to <=4 dp.)
// ALL-POSITIVE CONVENTION (also applied in the CUDA constants): the three signed modulation
// params are stored SHIFTED so their clip ranges start at 0, and offset back in the formulas —
// d_weight effective = w-0.5 (range [0,1.0]); d_decay & s_decay1 effective = w-0.3 (range
// [0,0.6]). The stored defaults for 31/32/33 below are already the shifted (all-positive) values.
pub static DEFAULT_PARAMETERS: [f32; 34] = [
    0.1104, 2.2395, 3.9221, 11.7841, 6.1686, 0.6457, 3.6807, 1.9795, 0.0, 1.3826, 0.7024, 0.5999,
    0.8146, 0.6398, 1.0, 1.3207, 0.6707, 3.8668, 0.4416, 0.0934, 1.8631, 0.6162, 1.0869, 0.1567,
    0.0801, 0.2421, 0.9464, 0.1433, 0.7145, 0.0, 0.5667, 0.3734, 0.5333, 0.3048,
];
/// This is a slice for efficiency, and should be 34 in length.
pub type Parameters = [f32];

fn infer<B: Backend>(
    model: &Model<B>,
    batch: FSRSBatch<B>,
) -> (MemoryStateTensors<B>, Tensor<B, 1>) {
    let state = model.forward(batch.t_historys, batch.r_historys, None);
    let retrievability = model.power_forgetting_curve(
        batch.delta_ts,
        state.stability.clone(),
        state.stability_fast.clone(),
        state.difficulty.clone(),
    );
    (state, retrievability)
}

#[derive(Debug, PartialEq, Clone, Copy, Serialize)]
pub struct MemoryState {
    pub stability: f32,
    pub difficulty: f32,
    pub stability_fast: f32,
}

impl<B: Backend> From<MemoryStateTensors<B>> for MemoryState {
    fn from(m: MemoryStateTensors<B>) -> Self {
        Self {
            stability: m.stability.into_scalar().elem(),
            difficulty: m.difficulty.into_scalar().elem(),
            stability_fast: m.stability_fast.into_scalar().elem(),
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
        let mut stabilities = Vec::with_capacity(starting_states.len());
        let mut difficulties = Vec::with_capacity(starting_states.len());
        let mut stabilities_fast = Vec::with_capacity(starting_states.len());
        for starting_state in &starting_states {
            if let Some(state) = starting_state {
                stabilities.push(state.stability);
                difficulties.push(state.difficulty);
                stabilities_fast.push(state.stability_fast);
            } else {
                stabilities.push(0.0);
                difficulties.push(0.0);
                stabilities_fast.push(0.0);
            }
        }
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
            stability_fast: Tensor::from_data(
                TensorData::new(
                    stabilities_fast.clone(),
                    Shape {
                        dims: vec![stabilities_fast.len()],
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
        let stability_fast = state.stability_fast.to_data().to_vec::<f32>().unwrap();
        Ok(izip!(stability, difficulty, stability_fast)
            .map(|(stability, difficulty, stability_fast)| MemoryState {
                stability,
                difficulty,
                stability_fast,
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
