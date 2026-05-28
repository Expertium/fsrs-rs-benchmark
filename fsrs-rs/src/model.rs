use crate::DEFAULT_PARAMETERS;
use crate::error::{FSRSError, Result};
use crate::inference::{MemoryState, Parameters};
use crate::simulation::{D_MAX, D_MIN, S_MAX, S_MIN};
use crate::training::ModelConfig;
use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::{
    constant,
    module::{Module, Param},
    tensor::{Shape, Tensor, TensorData, backend::Backend},
};

#[path = "model_v7.rs"]
pub(crate) mod model_v7;

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
    fn next_interval(
        model: &Model<B>,
        stability: Tensor<B, 1>,
        desired_retention: Tensor<B, 1>,
    ) -> Tensor<B, 1>;
    fn update_state(
        model: &Model<B>,
        delta_t: Tensor<B, 1>,
        rating: Tensor<B, 1>,
        last_s: Tensor<B, 1>,
        last_d: Tensor<B, 1>,
    ) -> MemoryStateTensors<B>;
    fn memory_state_from_sm2_fsrs(
        model: &Model<B>,
        ease_factor: f32,
        interval: f32,
        sm2_retention: f32,
    ) -> Result<MemoryState>;
    fn interval_at_retrievability(
        model: &Model<B>,
        stability: f32,
        target_retrievability: f32,
    ) -> f32;
}

pub(super) struct Fsrs7Ops;

type ApplyFreezeShortTermFn = fn(&mut [f32]);
type PowerForgettingCurveFn<B> = fn(&Model<B>, Tensor<B, 1>, Tensor<B, 1>) -> Tensor<B, 1>;
type NextIntervalFn<B> = fn(&Model<B>, Tensor<B, 1>, Tensor<B, 1>) -> Tensor<B, 1>;
type UpdateStateFn<B> =
    fn(&Model<B>, Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) -> MemoryStateTensors<B>;
type MemoryStateFromSm2Fn<B> = fn(&Model<B>, f32, f32, f32) -> Result<MemoryState>;
type IntervalAtRetrievabilityFn<B> = fn(&Model<B>, f32, f32) -> f32;

#[derive(Clone, Copy)]
struct VersionFns<B: Backend> {
    apply_freeze_short_term: ApplyFreezeShortTermFn,
    power_forgetting_curve: PowerForgettingCurveFn<B>,
    next_interval: NextIntervalFn<B>,
    update_state: UpdateStateFn<B>,
    memory_state_from_sm2: MemoryStateFromSm2Fn<B>,
    interval_at_retrievability: IntervalAtRetrievabilityFn<B>,
}

impl<B: Backend> VersionFns<B> {
    fn from_version(version: ModelVersion) -> Self {
        match version {
            ModelVersion::Fsrs7 => Self {
                apply_freeze_short_term: <Fsrs7Ops as VersionOps<B>>::apply_freeze_short_term,
                power_forgetting_curve: <Fsrs7Ops as VersionOps<B>>::power_forgetting_curve,
                next_interval: <Fsrs7Ops as VersionOps<B>>::next_interval,
                update_state: <Fsrs7Ops as VersionOps<B>>::update_state,
                memory_state_from_sm2: <Fsrs7Ops as VersionOps<B>>::memory_state_from_sm2_fsrs,
                interval_at_retrievability: <Fsrs7Ops as VersionOps<B>>::interval_at_retrievability,
            },
        }
    }
}

impl<B: Backend> Model<B> {
    #[cfg(test)]
    pub fn new(config: ModelConfig) -> Self {
        Self::new_with_device(config, &B::Device::default())
    }

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

    pub(crate) fn memory_state_from_sm2(
        &self,
        ease_factor: f32,
        interval: f32,
        sm2_retention: f32,
    ) -> Result<MemoryState> {
        let ops = VersionFns::<B>::from_version(self.version());
        (ops.memory_state_from_sm2)(self, ease_factor, interval, sm2_retention)
    }

    pub fn power_forgetting_curve(&self, t: Tensor<B, 1>, s: Tensor<B, 1>) -> Tensor<B, 1> {
        let ops = VersionFns::<B>::from_version(self.version());
        (ops.power_forgetting_curve)(self, t, s)
    }

    pub fn next_interval(
        &self,
        stability: Tensor<B, 1>,
        desired_retention: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        let ops = VersionFns::<B>::from_version(self.version());
        (ops.next_interval)(self, stability, desired_retention)
    }

    pub(crate) fn interval_at_retrievability(
        &self,
        stability: f32,
        target_retrievability: f32,
    ) -> f32 {
        let ops = VersionFns::<B>::from_version(self.version());
        (ops.interval_at_retrievability)(self, stability, target_retrievability)
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

    pub(crate) fn step(
        &self,
        delta_t: Tensor<B, 1>,
        rating: Tensor<B, 1>,
        state: MemoryStateTensors<B>,
        nth: usize,
    ) -> MemoryStateTensors<B> {
        let ops = VersionFns::<B>::from_version(self.version());
        self.step_with_ops(&ops, delta_t, rating, state, nth)
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

    pub(crate) fn from_state(state: MemoryState) -> Self {
        let device = B::Device::default();
        Self {
            stability: Tensor::from_floats([state.stability], &device),
            difficulty: Tensor::from_floats([state.difficulty], &device),
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

    parameters[0] = clamp_safe(parameters[0], crate::simulation::S_MIN, INIT_S_MAX / 2.0);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{Model as TestModel, TestHelper};
    use burn::tensor::TensorData;

    #[test]
    fn test_w() {
        let model: TestModel = Model::new(ModelConfig::default());
        assert_eq!(
            model.w.val().to_data(),
            TensorData::new(DEFAULT_PARAMETERS.to_vec(), Shape { dims: vec![35] })
        )
    }

    #[test]
    fn test_fsrs() {
        FSRS::default()
            .model()
            .w
            .to_data()
            .to_vec::<f32>()
            .unwrap()
            .assert_approx_eq(DEFAULT_PARAMETERS);
        assert!(FSRS::new(&[]).is_ok());
        assert!(FSRS::new(&[1.]).is_err());
        assert!(FSRS::new(DEFAULT_PARAMETERS.as_slice()).is_ok());
    }

    #[test]
    fn test_model_version_selection() {
        let model_v7: TestModel = Model::new(ModelConfig::default());
        assert_eq!(model_v7.version(), ModelVersion::Fsrs7);
    }
}
