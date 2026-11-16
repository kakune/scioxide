//! Resource contracts for quasi batch.
#![allow(clippy::unwrap_used)]

use scioxide_random::{QmcPlan, QuasiRng};
#[path = "../support/quasi.rs"]
mod quasi_fixture;

#[test]
fn prepared_batch_generation_allocates_nothing() {
    let directions = quasi_fixture::joe_kuo_d6_32().unwrap();
    let sobol_f64_plan = QmcPlan::sobol_with_direction_numbers(32, directions).unwrap();
    let mut sobol_f64 = sobol_f64_plan.sequence::<f64>().unwrap();
    let mut sobol_f64_output = vec![0.0; 32 * 1_024];
    let batch =
        scioxide_random::SamplingPolicy::advance_on_error(scioxide_random::OutputPolicy::Atomic);
    let mut sobol_f64_workspace = batch
        .workspace(scioxide_random::SamplingRequest::scalars(
            sobol_f64_output.len(),
        ))
        .unwrap();
    sobol_f64
        .points_into(batch, &mut sobol_f64_output, &mut sobol_f64_workspace)
        .unwrap();

    let sobol_f32_plan = QmcPlan::sobol(8).unwrap();
    let mut sobol_f32 = sobol_f32_plan.sequence::<f32>().unwrap();
    let mut sobol_f32_output = vec![0.0; 8 * 1_024];
    let batch =
        scioxide_random::SamplingPolicy::advance_on_error(scioxide_random::OutputPolicy::Atomic);
    let mut sobol_f32_workspace = batch
        .workspace(scioxide_random::SamplingRequest::scalars(
            sobol_f32_output.len(),
        ))
        .unwrap();
    sobol_f32
        .points_into(batch, &mut sobol_f32_output, &mut sobol_f32_workspace)
        .unwrap();

    let halton_plan = QmcPlan::halton(8).unwrap();
    let mut halton = halton_plan.sequence::<f64>().unwrap();
    let mut halton_output = vec![0.0; 8 * 1_024];
    let batch =
        scioxide_random::SamplingPolicy::advance_on_error(scioxide_random::OutputPolicy::Atomic);
    let mut halton_workspace = batch
        .workspace(scioxide_random::SamplingRequest::scalars(
            halton_output.len(),
        ))
        .unwrap();
    halton
        .points_into(batch, &mut halton_output, &mut halton_workspace)
        .unwrap();
}
