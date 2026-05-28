#![allow(clippy::single_range_in_vec_init)]

#[cfg(test)]
mod convertor_tests;
mod error;
mod inference;
mod model;
mod simulation;
#[cfg(test)]
mod test_helpers;
mod training;

pub use training::{FSRSItem, FSRSReview, filter_outlier};
pub use error::{FSRSError, Result};
pub use inference::{
    DEFAULT_PARAMETERS, ItemProgress, ItemState, MemoryState, ModelEvaluation, NextStates,
    current_retrievability, evaluate_with_time_series_splits,
};
pub use model::FSRS;
pub use simulation::{
    CMRRTargetFn, Card, PostSchedulingFn, ReviewPriorityFn, ReviewRatingCostFn, RevlogEntry,
    RevlogReviewKind, SimulationResult, SimulatorConfig, expected_workload,
    expected_workload_with_existing_cards, extract_simulator_config, optimal_retention, simulate,
};
pub use training::{
    CombinedProgressState, ComputeParametersInput, benchmark, compute_parameters,
};
