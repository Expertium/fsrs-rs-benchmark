#![allow(clippy::single_range_in_vec_init)]

mod analytic;
mod error;
mod inference;
mod model;
mod training;

pub use training::{FSRSItem, FSRSReview};
pub use error::{FSRSError, Result};
pub use inference::{DEFAULT_PARAMETERS, ItemProgress, MemoryState, ModelEvaluation};
pub use model::FSRS;
pub use training::{
    CombinedProgressState, ComputeParametersInput, benchmark, compute_parameters,
    windowed_loss_with_params,
};
