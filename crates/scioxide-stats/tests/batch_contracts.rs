//! Shared-view admission and reservation recovery across numerical families.
#![allow(clippy::unwrap_used)]

use scioxide_core::{Complex, ExecutionContext, NumericScalar, StridedLayout};
use scioxide_stats::*;

#[test]
fn axes_and_strides_have_explicit_output_order() {
    let backing = [1.0_f64, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0];
    let view = BatchView::new(
        &backing,
        StridedLayout::new(&[2, 2], &[4, 2]).unwrap(),
        Axis::Index(1),
    )
    .unwrap();
    let rows = reduce_moments_with_options(view, StatisticsOptions::new()).unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].mean().unwrap(), Some(1.5));
    assert_eq!(rows[1].mean().unwrap(), Some(3.5));
    let columns = reduce_moments_with_options(
        view.with_axis(Axis::Index(0)).unwrap(),
        StatisticsOptions::new(),
    )
    .unwrap();
    assert_eq!(columns[0].mean().unwrap(), Some(2.0));
    assert_eq!(columns[1].mean().unwrap(), Some(3.0));
}

#[test]
fn missing_and_non_finite_policies_are_independent() {
    let values = [1.0_f64, f64::NAN, 3.0, f64::INFINITY];
    let validity = [true, false, true, true];
    let view = BatchView::new(
        &values,
        StridedLayout::new(&[2, 2], &[2, 1]).unwrap(),
        Axis::All,
    )
    .unwrap()
    .with_validity(&validity)
    .unwrap();
    assert!(reduce_moments_with_options(view, StatisticsOptions::new()).is_err());
    let options = StatisticsOptions::new()
        .with_missing_policy(MissingPolicy::Skip)
        .with_non_finite_policy(NonFinitePolicy::Skip);
    let skipped = reduce_moments_with_options(view, options).unwrap();
    assert_eq!(skipped[0].count(), 2);
    assert_eq!(skipped[0].mean().unwrap(), Some(2.0));
}

#[test]
fn paired_and_weighted_reductions_preserve_pairing() {
    let x_values = [1.0_f64, 2.0, 3.0];
    let y_values = [2.0_f64, 4.0, 6.0];
    let weights = [1.0_f64, 2.0, 1.0];
    let valid = [true, false, true];
    let layout = StridedLayout::new(&[1, 3], &[3, 1]).unwrap();
    let x = BatchView::new(&x_values, layout, Axis::All)
        .unwrap()
        .with_validity(&valid)
        .unwrap();
    let y = BatchView::new(&y_values, layout, Axis::All).unwrap();
    let options = StatisticsOptions::new().with_missing_policy(MissingPolicy::Skip);
    let covariance = reduce_covariance_with_options(x, y, options).unwrap();
    assert_eq!(covariance[0].count(), 2);
    assert_eq!(covariance[0].population_covariance().unwrap(), Some(2.0));
    let weighted = reduce_weighted_moments_with_options(
        x.with_weights(Weights::<_, Reliability>::new(&weights))
            .unwrap(),
        options,
    )
    .unwrap();
    assert_eq!(weighted[0].observation_count(), 2);
    assert_eq!(weighted[0].mean().unwrap(), Some(2.0));
}

#[test]
fn stable_partition_mode_matches_its_resolved_plan_repeatedly() {
    let values = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let view = BatchView::new(
        &values,
        StridedLayout::new(&[2, 3], &[3, 1]).unwrap(),
        Axis::Index(0),
    )
    .unwrap();
    let options = StatisticsOptions::new().with_merge_policy(MergePolicy::StablePartitions);
    let first = reduce_moments_with_options(view, options).unwrap();
    let second = reduce_moments_with_options(view, options).unwrap();
    assert_eq!(first, second);
    assert_eq!(first[0].mean().unwrap(), Some(2.5));
    assert_eq!(first[2].mean().unwrap(), Some(4.5));
}

#[cfg(feature = "std")]
#[test]
fn stable_partition_mode_is_independent_of_thread_completion_order() {
    let values: [f64; 2048] =
        core::array::from_fn(|index| f64::from(u32::try_from(index % 17).unwrap()) - 8.0);
    let shape = [values.len()];
    let view = BatchView::new(
        &values,
        StridedLayout::new(&shape, &[1]).unwrap(),
        Axis::All,
    )
    .unwrap();
    let workers = core::num::NonZeroUsize::MIN.saturating_add(3);
    let executor = scioxide_core::ThreadExecutor::new(workers);
    let context = ExecutionContext::new(&executor);
    let options = StatisticsOptions::new()
        .with_merge_policy(MergePolicy::StablePartitions)
        .with_execution_policy(scioxide_core::ExecutionPolicy::Parallel(workers));
    let request = ReductionWorkspace::workspace_requirements(view, options, context).unwrap();
    let mut workspace = ReductionWorkspace::workspace(&request).unwrap();
    let mut output = [OnlineMoments::new(MomentOrder::Second).snapshot(); 1];
    reduce_moments_into(view, options, &mut output, &mut workspace, context).unwrap();
    let expected = output;
    for _ in 0..8 {
        reduce_moments_into(view, options, &mut output, &mut workspace, context).unwrap();
        assert_eq!(output, expected);
    }
}

#[test]
fn invalid_masks_shapes_and_weights_are_typed() {
    let values = [1.0_f64, 2.0];
    let view = BatchView::new(
        &values,
        StridedLayout::new(&[1, 2], &[2, 1]).unwrap(),
        Axis::All,
    )
    .unwrap();
    assert!(view.with_validity(&[true]).is_err());
    let wrong = BatchView::new(
        &values,
        StridedLayout::new(&[2, 1], &[1, 1]).unwrap(),
        Axis::All,
    )
    .unwrap();
    assert!(reduce_covariance_with_options(view, wrong, StatisticsOptions::new()).is_err());
    let invalid = [1.0_f64, -1.0];
    assert!(
        reduce_weighted_moments_with_options(
            view.with_weights(Weights::<_, Reliability>::new(&invalid))
                .unwrap(),
            StatisticsOptions::new()
        )
        .is_err()
    );
}

#[test]
fn output_allocation_failure_precedes_observation_access() {
    let view = BatchView::<f64>::new(
        &[],
        StridedLayout::new(&[usize::MAX, 0], &[0, 1]).unwrap(),
        Axis::Index(1),
    )
    .unwrap();
    let error = reduce_moments_with_options(view, StatisticsOptions::new()).unwrap_err();
    assert!(matches!(
        error.view(),
        scioxide_core::SciOxideErrorView::AllocationFailure { .. }
    ));
}

#[test]
fn complex_strided_lanes_apply_observation_masks_before_values() {
    let values = [
        Complex::new(1.0_f64, 1.0),
        Complex::new(10.0, -10.0),
        Complex::new(f64::NAN, f64::NAN),
        Complex::new(20.0, -20.0),
        Complex::new(5.0, 5.0),
        Complex::new(f64::INFINITY, f64::NAN),
    ];
    let mask = [true, false, true, true, true, false];
    let view = BatchView::new(
        &values,
        StridedLayout::new(&[3, 2], &[2, 1]).unwrap(),
        Axis::Index(0),
    )
    .unwrap()
    .with_validity(&mask)
    .unwrap();
    let options = StatisticsOptions::new().with_missing_policy(MissingPolicy::Skip);
    let context = ExecutionContext::sequential();
    let request =
        StatisticsWorkspace::workspace_requirements(view, None, options, context).unwrap();
    let mut workspace = StatisticsWorkspace::workspace(&request).unwrap();
    let mut output = [StatisticValue::Location(Complex::new(99.0, 99.0)); 2];
    statistics_into(
        view,
        Statistic::mean(),
        options,
        &mut output,
        &mut workspace,
        context,
    )
    .unwrap();
    for (actual, expected) in output
        .into_iter()
        .zip([Complex::new(3.0, 3.0), Complex::new(15.0, -15.0)])
    {
        assert!(
            (actual.location().unwrap() - expected).abs() <= 64.0 * f64::EPSILON * expected.abs()
        );
    }
    statistics_into(
        view,
        Statistic::variance(0),
        options,
        &mut output,
        &mut workspace,
        context,
    )
    .unwrap();
    for (actual, expected) in output.into_iter().zip([8.0, 50.0]) {
        assert!((actual.dispersion().unwrap() - expected).abs() <= 64.0 * f64::EPSILON * expected);
    }
    let before = output;
    assert!(
        statistics_into(
            view,
            Statistic::mean(),
            StatisticsOptions::new(),
            &mut output,
            &mut workspace,
            context
        )
        .is_err()
    );
    assert_eq!(output, before);
}

#[test]
fn failed_wide_power_projection_does_not_poison_a_narrower_reuse() {
    let values = [1.0_f64, 3.0, f64::MAX, -f64::MAX];
    let wide = BatchView::new(
        &values,
        StridedLayout::new(&[2, 2], &[2, 1]).unwrap(),
        Axis::Index(1),
    )
    .unwrap();
    let options = StatisticsOptions::new();
    let context = ExecutionContext::sequential();
    let request = ReductionWorkspace::workspace_requirements(wide, options, context).unwrap();
    let mut workspace = ReductionWorkspace::workspace(&request).unwrap();
    let mut output = [Some(99.0); 2];
    assert!(
        reduce_moment_into(
            wide,
            2,
            Some(0.0),
            options,
            &mut output,
            &mut workspace,
            context
        )
        .is_err()
    );
    assert_eq!(output, [Some(99.0); 2]);
    let narrow = BatchView::new(
        &values[..2],
        StridedLayout::new(&[1, 2], &[2, 1]).unwrap(),
        Axis::Index(1),
    )
    .unwrap();
    reduce_moment_into(
        narrow,
        2,
        Some(0.0),
        options,
        &mut output[..1],
        &mut workspace,
        context,
    )
    .unwrap();
    assert!((output[0].unwrap() - 5.0).abs() <= f64::EPSILON);
    reduce_moment_into(
        narrow,
        2,
        None,
        options,
        &mut output[..1],
        &mut workspace,
        context,
    )
    .unwrap();
    assert!((output[0].unwrap() - 1.0).abs() <= f64::EPSILON);
}
