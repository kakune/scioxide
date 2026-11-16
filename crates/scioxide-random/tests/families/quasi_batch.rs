//! Public contiguous quasi-random batch contracts.

#![allow(clippy::unwrap_used)]

#[path = "../support/quasi.rs"]
mod quasi_fixture;

use scioxide_random::{QmcPlan, QuasiRng, SamplingWorkspace};

#[test]
fn canonical_custom_dimension_32_matches_repeated_points() {
    let directions = quasi_fixture::joe_kuo_d6_32().unwrap();
    let batch_plan = QmcPlan::sobol_with_direction_numbers(32, directions).unwrap();
    let mut batch = batch_plan.sequence::<f64>().unwrap();
    let mut repeated = batch.clone();
    batch.skip_to(73).unwrap();
    repeated.skip_to(73).unwrap();

    let actual = batch.points(257).unwrap();
    let mut expected = vec![0.0; actual.len()];
    expected.chunks_exact_mut(32).for_each(|point| {
        repeated.next_point_into(point).unwrap();
    });
    assert_eq!(actual, expected);
    assert_eq!(batch.next_point().unwrap(), repeated.next_point().unwrap());
}

#[test]
fn allocating_and_caller_owned_batches_are_identical() {
    let allocating_generator_plan = QmcPlan::sobol(8).unwrap();
    let mut allocating_generator = allocating_generator_plan.sequence::<f32>().unwrap();
    let mut buffered_generator = allocating_generator.clone();
    allocating_generator.skip(19).unwrap();
    buffered_generator.skip(19).unwrap();

    let allocating = allocating_generator.points(65).unwrap();
    let mut buffered = vec![0.0; allocating.len()];
    let mut workspace = SamplingWorkspace::new();
    buffered_generator
        .points_into(
            scioxide_random::SamplingPolicy::advance_on_error(
                scioxide_random::OutputPolicy::Atomic,
            ),
            &mut buffered,
            &mut workspace,
        )
        .unwrap();
    assert_eq!(allocating, buffered);
    assert_eq!(
        allocating_generator.next_point().unwrap(),
        buffered_generator.next_point().unwrap()
    );
}

#[test]
fn halton_uses_the_default_contiguous_batch_contract() {
    let batch_plan = QmcPlan::halton(8).unwrap();
    let mut batch = batch_plan.sequence::<f64>().unwrap();
    let mut repeated = batch.clone();
    batch.skip_to(101).unwrap();
    repeated.skip_to(101).unwrap();

    let mut actual = vec![0.0; 8 * 129];
    let mut workspace = SamplingWorkspace::new();
    batch
        .points_into(
            scioxide_random::SamplingPolicy::advance_on_error(
                scioxide_random::OutputPolicy::Atomic,
            ),
            &mut actual,
            &mut workspace,
        )
        .unwrap();
    let mut expected = vec![0.0; actual.len()];
    expected.chunks_exact_mut(8).for_each(|point| {
        repeated.next_point_into(point).unwrap();
    });
    assert_eq!(actual, expected);
    assert_eq!(batch.next_point().unwrap(), repeated.next_point().unwrap());

    let malformed_plan = QmcPlan::halton(3).unwrap();
    let mut malformed = malformed_plan.sequence::<f64>().unwrap();
    let mut output = [-1.0; 5];
    let error = malformed
        .points_into(
            scioxide_random::SamplingPolicy::advance_on_error(
                scioxide_random::OutputPolicy::Atomic,
            ),
            &mut output,
            &mut workspace,
        )
        .unwrap_err();
    assert_eq!(
        error.kind(),
        scioxide_core::SciOxideErrorKind::DimensionMismatch
    );
    assert_eq!(output, [-1.0; 5]);
}
