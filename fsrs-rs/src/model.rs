use crate::DEFAULT_PARAMETERS;
use crate::error::{FSRSError, Result};
use crate::inference::Parameters;
// FSRS-7 stability/difficulty clamp bounds (relocated from the removed simulator).
pub(crate) const S_MIN: f32 = 0.0001;
pub(crate) const S_MAX: f32 = 36500.0;
pub(crate) const D_MIN: f32 = 1.0;
pub(crate) const D_MAX: f32 = 10.0;
use crate::training::ModelConfig;
use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::{
    constant,
    module::{Module, Param},
    tensor::{Shape, Tensor, TensorData, backend::Backend},
};

pub(crate) mod model_v7 {
use super::{Fsrs7Ops, Get, MemoryStateTensors, Model, VersionOps, tensor_max, tensor_min};
use burn::tensor::{Tensor, backend::Backend};

pub(super) const PARAM_LEN: usize = 35;

impl<B: Backend> VersionOps<B> for Fsrs7Ops {
    fn apply_freeze_short_term(initial_params: &mut [f32]) {
        // FSRS-7: disable short-term contribution by forcing transition weight to 0,
        // making coefficient == 1.0 for every delta_t.
        initial_params[26] = 0.0;
    }

    fn power_forgetting_curve(model: &Model<B>, t: Tensor<B, 1>, s: Tensor<B, 1>) -> Tensor<B, 1> {
        power_forgetting_curve(model, t, s)
    }

    fn update_state(
        model: &Model<B>,
        delta_t: Tensor<B, 1>,
        rating: Tensor<B, 1>,
        last_s: Tensor<B, 1>,
        last_d: Tensor<B, 1>,
    ) -> MemoryStateTensors<B> {
        let delta_t = delta_t.clamp_min(0.0);
        let retrievability = power_forgetting_curve(model, delta_t.clone(), last_s.clone());
        let new_s_long_term = stability_for_set(
            model,
            last_s.clone(),
            last_d.clone(),
            retrievability.clone(),
            rating.clone(),
            7,
        );
        let new_s_short_term = stability_for_set(
            model,
            last_s.clone(),
            last_d.clone(),
            retrievability,
            rating.clone(),
            16,
        );
        let coefficient = transition_function(model, delta_t);
        // If short-term is disabled, w[26]=0 => coefficient=1, so short_weight=0.
        // That cancels the short-term branch and keeps only long-term stability.
        let short_weight = coefficient.clone().neg().add_scalar(1.0);
        let new_s = coefficient * new_s_long_term + short_weight * new_s_short_term;
        let new_d = next_difficulty(model, last_d, rating);
        MemoryStateTensors {
            stability: new_s,
            difficulty: new_d,
        }
    }
}

pub(super) fn power_forgetting_curve<B: Backend>(
    model: &Model<B>,
    t: Tensor<B, 1>,
    s: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let t_over_s = t.clamp_min(0.0) / s.clone();

    let decay1 = -model.w.get(27);
    let decay2 = -model.w.get(28);
    let base1 = model.w.get(29);
    let base2 = model.w.get(30);

    let factor1 = base1.clone().powf(decay1.clone().powi_scalar(-1)) - 1.0;
    let factor2 = base2.clone().powf(decay2.clone().powi_scalar(-1)) - 1.0;

    let r1 = (t_over_s.clone() * factor1 + 1.0).powf(decay1);
    let r2 = (t_over_s * factor2 + 1.0).powf(decay2);

    let weight1 = model.w.get(31) * s.clone().powf(-model.w.get(33));
    let weight2 = model.w.get(32) * s.powf(model.w.get(34));

    (weight1.clone() * r1 + weight2.clone() * r2) / (weight1 + weight2)
}

pub(super) fn stability_for_set<B: Backend>(
    model: &Model<B>,
    last_s: Tensor<B, 1>,
    last_d: Tensor<B, 1>,
    r: Tensor<B, 1>,
    rating: Tensor<B, 1>,
    start: usize,
) -> Tensor<B, 1> {
    let batch_size = rating.dims()[0];
    let device = rating.device();
    let hard_penalty = Tensor::ones([batch_size], &device)
        .mask_where(rating.clone().equal_elem(2), model.w.get(start + 7));
    let easy_bonus = Tensor::ones([batch_size], &device)
        .mask_where(rating.clone().equal_elem(4), model.w.get(start + 8));

    let new_s_fail = model.w.get(start + 3)
        * last_d.clone().powf(-model.w.get(start + 4))
        * ((last_s.clone() + 1).powf(model.w.get(start + 5)) - 1)
        * ((-r.clone() + 1) * model.w.get(start + 6)).exp();
    let pls = tensor_min(last_s.clone(), new_s_fail);

    let sinc = model.w.get(start).add_scalar(-1.5).exp()
        * last_d.neg().add_scalar(11.0)
        * last_s.clone().powf(-model.w.get(start + 1))
        * (((-r + 1) * model.w.get(start + 2)).exp() - 1)
        * hard_penalty
        * easy_bonus
        + 1;
    let new_s_success = tensor_max(pls.clone(), last_s * sinc);
    let success = rating.greater_elem(1);
    pls.mask_where(success, new_s_success)
}

pub(super) fn transition_function<B: Backend>(
    model: &Model<B>,
    delta_t: Tensor<B, 1>,
) -> Tensor<B, 1> {
    (model.w.get(26) * (-model.w.get(25) * delta_t).exp())
        .neg()
        .add_scalar(1.0)
}

pub(super) fn mean_reversion<B: Backend>(
    init: Tensor<B, 1>,
    current: Tensor<B, 1>,
) -> Tensor<B, 1> {
    init.mul_scalar(0.01) + current.mul_scalar(0.99)
}

pub(super) fn next_difficulty<B: Backend>(
    model: &Model<B>,
    difficulty: Tensor<B, 1>,
    rating: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let delta_d = -model.w.get(6) * (rating - 3);
    let new_d = difficulty.clone() + model.linear_damping(delta_d, difficulty);
    let device = new_d.device();
    let init = model.init_difficulty(Tensor::from_floats([4.0], &device));
    mean_reversion(init, new_d).clamp(super::D_MIN, super::D_MAX)
}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelVersion {
    Fsrs7,
}

impl core::fmt::Display for ModelVersion {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Fsrs7 => write!(f, "FSRS7"),
        }
    }
}

constant!(ModelVersion);

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub w: Param<Tensor<B, 1>>,
    version: ModelVersion,
}

pub(crate) trait Get<B: Backend, const N: usize> {
    fn get(&self, n: usize) -> Tensor<B, N>;
}

impl<B: Backend, const N: usize> Get<B, N> for Tensor<B, N> {
    fn get(&self, n: usize) -> Self {
        self.clone().slice([n..(n + 1)])
    }
}

fn tensor_min<B: Backend>(a: Tensor<B, 1>, b: Tensor<B, 1>) -> Tensor<B, 1> {
    a.clone().mask_where(a.clone().greater(b.clone()), b)
}

fn tensor_max<B: Backend>(a: Tensor<B, 1>, b: Tensor<B, 1>) -> Tensor<B, 1> {
    a.clone().mask_where(a.clone().lower(b.clone()), b)
}

pub(super) trait VersionOps<B: Backend> {
    fn apply_freeze_short_term(initial_params: &mut [f32]);
    fn power_forgetting_curve(model: &Model<B>, t: Tensor<B, 1>, s: Tensor<B, 1>) -> Tensor<B, 1>;
    fn update_state(
        model: &Model<B>,
        delta_t: Tensor<B, 1>,
        rating: Tensor<B, 1>,
        last_s: Tensor<B, 1>,
        last_d: Tensor<B, 1>,
    ) -> MemoryStateTensors<B>;
}

pub(super) struct Fsrs7Ops;

type ApplyFreezeShortTermFn = fn(&mut [f32]);
type PowerForgettingCurveFn<B> = fn(&Model<B>, Tensor<B, 1>, Tensor<B, 1>) -> Tensor<B, 1>;
type UpdateStateFn<B> =
    fn(&Model<B>, Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) -> MemoryStateTensors<B>;

#[derive(Clone, Copy)]
struct VersionFns<B: Backend> {
    apply_freeze_short_term: ApplyFreezeShortTermFn,
    power_forgetting_curve: PowerForgettingCurveFn<B>,
    update_state: UpdateStateFn<B>,
}

impl<B: Backend> VersionFns<B> {
    fn from_version(version: ModelVersion) -> Self {
        match version {
            ModelVersion::Fsrs7 => Self {
                apply_freeze_short_term: <Fsrs7Ops as VersionOps<B>>::apply_freeze_short_term,
                power_forgetting_curve: <Fsrs7Ops as VersionOps<B>>::power_forgetting_curve,
                update_state: <Fsrs7Ops as VersionOps<B>>::update_state,
            },
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn new_with_device(config: ModelConfig, device: &B::Device) -> Self {
        let mut initial_params = DEFAULT_PARAMETERS.to_vec();
        let version = ModelVersion::Fsrs7;
        if let Some(initial_stability) = config.initial_stability {
            initial_params[0..4].copy_from_slice(&initial_stability);
        }
        if let Some(initial_forgetting_curve) = config.initial_forgetting_curve {
            initial_params[27..35].copy_from_slice(&initial_forgetting_curve);
        }
        if config.freeze_short_term_stability {
            let ops = VersionFns::<B>::from_version(version);
            (ops.apply_freeze_short_term)(&mut initial_params);
        }

        Self {
            w: Param::from_tensor(Tensor::from_floats(
                TensorData::new(
                    initial_params.clone(),
                    Shape {
                        dims: vec![initial_params.len()],
                    },
                ),
                device,
            )),
            version,
        }
    }

    pub(crate) fn version(&self) -> ModelVersion {
        self.version
    }

    pub fn power_forgetting_curve(&self, t: Tensor<B, 1>, s: Tensor<B, 1>) -> Tensor<B, 1> {
        let ops = VersionFns::<B>::from_version(self.version());
        (ops.power_forgetting_curve)(self, t, s)
    }

    pub(crate) fn init_stability(&self, rating: Tensor<B, 1>) -> Tensor<B, 1> {
        self.w.val().select(0, rating.int() - 1)
    }

    fn init_difficulty(&self, rating: Tensor<B, 1>) -> Tensor<B, 1> {
        self.w.get(4) - (self.w.get(5) * (rating - 1)).exp() + 1
    }

    fn linear_damping(&self, delta_d: Tensor<B, 1>, old_d: Tensor<B, 1>) -> Tensor<B, 1> {
        old_d.neg().add_scalar(10.0) * delta_d.div_scalar(9.0)
    }

    fn step_with_ops(
        &self,
        ops: &VersionFns<B>,
        delta_t: Tensor<B, 1>,
        rating: Tensor<B, 1>,
        state: MemoryStateTensors<B>,
        nth: usize,
    ) -> MemoryStateTensors<B> {
        let last_s = state.stability.clone().clamp(S_MIN, S_MAX);
        let last_d = state.difficulty.clone().clamp(D_MIN, D_MAX);
        let mut new_state = (ops.update_state)(
            self,
            delta_t.clone(),
            rating.clone(),
            last_s.clone(),
            last_d.clone(),
        );

        if nth == 0 {
            let is_initial = state.stability.clone().equal_elem(0.0);
            let init_s = self.init_stability(rating.clone().clamp(1, 4));
            let init_d = self
                .init_difficulty(rating.clone().clamp(1, 4))
                .clamp(D_MIN, D_MAX);
            new_state.stability = new_state.stability.mask_where(is_initial.clone(), init_s);
            new_state.difficulty = new_state.difficulty.mask_where(is_initial, init_d);
        }

        // mask padding zeros for rating
        new_state.stability = new_state
            .stability
            .mask_where(rating.clone().equal_elem(0), last_s)
            .clamp(S_MIN, S_MAX);
        new_state.difficulty = new_state
            .difficulty
            .mask_where(rating.equal_elem(0), last_d);

        new_state
    }

    /// If [starting_state] is provided, it will be used instead of the default initial stability/
    /// difficulty.
    pub(crate) fn forward(
        &self,
        delta_ts: Tensor<B, 2>,
        ratings: Tensor<B, 2>,
        starting_state: Option<MemoryStateTensors<B>>,
    ) -> MemoryStateTensors<B> {
        let [seq_len, batch_size] = delta_ts.dims();
        let mut state = if let Some(state) = starting_state {
            state
        } else {
            MemoryStateTensors::zeros(batch_size)
        };
        let ops = VersionFns::<B>::from_version(self.version());
        for i in 0..seq_len {
            let delta_t = delta_ts.get(i).squeeze(0);
            let rating = ratings.get(i).squeeze(0);
            state = self.step_with_ops(&ops, delta_t, rating, state, i);
        }
        state
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MemoryStateTensors<B: Backend> {
    pub stability: Tensor<B, 1>,
    pub difficulty: Tensor<B, 1>,
}

impl<B: Backend> MemoryStateTensors<B> {
    pub(crate) fn zeros(batch_size: usize) -> MemoryStateTensors<B> {
        let device = B::Device::default();
        MemoryStateTensors {
            stability: Tensor::zeros([batch_size], &device),
            difficulty: Tensor::zeros([batch_size], &device),
        }
    }
}

/// This is the main structure provided by this crate. It can be used
/// for both parameter training, and for reviews.
#[derive(Debug, Clone)]
pub struct FSRS<B: Backend = NdArray> {
    model: Model<B>,
}

impl Default for FSRS<NdArray> {
    fn default() -> Self {
        Self::new(&[]).expect("Default parameters should be valid")
    }
}

impl FSRS<NdArray> {
    /// - Parameters must be provided before running commands that need them.
    /// - Parameters may be an empty slice to use the default values instead.
    pub fn new(parameters: &Parameters) -> Result<Self> {
        Self::new_with_backend(parameters, &NdArrayDevice::Cpu)
    }
}

impl<B: Backend> FSRS<B> {
    pub fn new_with_backend<B2: Backend>(
        parameters: &Parameters,
        device: &B2::Device,
    ) -> Result<FSRS<B2>> {
        let parameters = check_and_fill_parameters(parameters)?;
        let model = parameters_to_model::<B2>(&parameters, device);

        Ok(FSRS { model })
    }

    pub(crate) fn model(&self) -> &Model<B> {
        &self.model
    }

    pub(crate) fn device(&self) -> B::Device {
        self.model().w.device()
    }
}

// Maximum initial stability for FSRS-7 parameter clipping
const INIT_S_MAX: f32 = 100.0;

fn clamp_safe(value: f32, low: f32, high: f32) -> f32 {
    let low = if low.is_finite() { low } else { 0.0 };
    let high = if high.is_finite() { high } else { low };
    let (low, high) = if low <= high {
        (low, high)
    } else {
        (high, low)
    };
    let value = if value.is_finite() { value } else { low };
    value.clamp(low, high)
}

fn clip_fsrs7_parameters(parameters: &mut [f32]) {
    const FSRS7_PARAM_LEN: usize = 35;
    if parameters.len() < FSRS7_PARAM_LEN {
        return;
    }

    parameters[0] = clamp_safe(parameters[0], S_MIN, INIT_S_MAX / 2.0);
    parameters[1] = clamp_safe(parameters[1], parameters[0], INIT_S_MAX);
    parameters[2] = clamp_safe(parameters[2], parameters[1], INIT_S_MAX);
    parameters[3] = clamp_safe(parameters[3], parameters[2], INIT_S_MAX);

    parameters[4] = clamp_safe(parameters[4], 1.0, 10.0);
    parameters[5] = clamp_safe(parameters[5], 0.001, 4.0);
    parameters[6] = clamp_safe(parameters[6], 0.1, 4.0);

    parameters[7] = clamp_safe(parameters[7], 0.0, 4.0);
    parameters[8] = clamp_safe(parameters[8], 0.0, 1.2);
    parameters[9] = clamp_safe(parameters[9], 0.3, 3.0);
    parameters[10] = clamp_safe(parameters[10], 0.01, 1.5);
    parameters[11] = clamp_safe(parameters[11], 0.001, 0.9);
    parameters[12] = clamp_safe(parameters[12], 0.1, 1.0);
    parameters[13] = clamp_safe(parameters[13], 0.0, 3.5);
    parameters[14] = clamp_safe(parameters[14], 0.0, 1.0);
    parameters[15] = clamp_safe(parameters[15], 1.0, 7.0);

    parameters[16] = clamp_safe(parameters[16], 0.0, 4.0);
    parameters[17] = clamp_safe(parameters[17], 0.0, 2.0);
    parameters[18] = clamp_safe(parameters[18], 0.5, 6.0);
    parameters[19] = clamp_safe(parameters[19], 0.001, 1.5);
    parameters[20] = clamp_safe(parameters[20], 0.001, 2.0);
    parameters[21] = clamp_safe(parameters[21], 0.001, 1.0);
    parameters[22] = clamp_safe(parameters[22], 0.0, 5.0);
    parameters[23] = clamp_safe(parameters[23], 0.0, 1.0);
    parameters[24] = clamp_safe(parameters[24], 1.0, 7.0);

    parameters[25] = clamp_safe(parameters[25], 2.5, 15.0);
    parameters[26] = clamp_safe(parameters[26], 0.0, 1.0);

    parameters[27] = clamp_safe(parameters[27], 0.01, 0.25);
    parameters[28] = clamp_safe(parameters[28], parameters[27], 0.95);
    parameters[29] = clamp_safe(parameters[29], 0.5, 0.85);
    parameters[30] = clamp_safe(parameters[30], parameters[29], 0.99);
    parameters[31] = clamp_safe(parameters[31], 0.01, 1.0);
    parameters[32] = clamp_safe(parameters[32], 0.1, 1.0);
    parameters[33] = clamp_safe(parameters[33], 0.0, 0.9);
    parameters[34] = clamp_safe(parameters[34], 0.1, 1.1);
}

pub(crate) fn clip_parameters(parameters: &Parameters) -> Vec<f32> {
    let mut parameters = parameters.to_vec();
    clip_fsrs7_parameters(&mut parameters);
    parameters
}

pub(crate) fn parameters_to_model<B: Backend>(
    parameters: &Parameters,
    device: &B::Device,
) -> Model<B> {
    let config = ModelConfig::default();
    let mut model = Model::new_with_device(config.clone(), device);
    let clipped = clip_parameters(parameters);
    model.w = Param::from_tensor(Tensor::from_floats(
        TensorData::new(
            clipped.clone(),
            Shape {
                dims: vec![clipped.len()],
            },
        ),
        device,
    ));
    model.version = ModelVersion::Fsrs7;
    model
}

pub(crate) fn check_and_fill_parameters(parameters: &Parameters) -> Result<Vec<f32>, FSRSError> {
    let parameters = if parameters.is_empty() {
        DEFAULT_PARAMETERS.to_vec()
    } else if parameters.len() == model_v7::PARAM_LEN {
        parameters.to_vec()
    } else {
        return Err(FSRSError::InvalidParameters);
    };
    if parameters.iter().any(|&w| !w.is_finite()) {
        return Err(FSRSError::InvalidParameters);
    }
    Ok(parameters)
}
