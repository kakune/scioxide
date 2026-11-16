#![allow(clippy::unwrap_used)]
#[cfg(feature = "std")]
use core::num::NonZeroUsize;

use scioxide_core::{ExecutionContext, StridedLayout};
#[cfg(feature = "std")]
use scioxide_core::{ExecutionPolicy, ThreadExecutor};
use scioxide_stats::*;

#[cfg(feature = "std")]
#[test]
fn strided_batches_match_executor_and_admission_is_transactional() {
    let x: Vec<f64> = (0..64).map(|i| (i % 11) as f64).collect();
    // Four interleaved sample lanes, also testing non-final sample axes.
    let view = BatchView::new(
        &x,
        StridedLayout::new(&[16, 4], &[4, 1]).unwrap(),
        Axis::Index(0),
    )
    .unwrap();
    let workers = NonZeroUsize::new(4).unwrap();
    let pool = ThreadExecutor::new(workers);
    let context = ExecutionContext::new(&pool);
    let mut options =
        StatisticsOptions::new().with_execution_policy(ExecutionPolicy::Parallel(workers));
    let plan = BinPlan::uniform(&[(0., 10., 5)]).unwrap();
    let mut w = BinnedWorkspace::default();
    w.try_reserve(
        &(BinnedWorkspace::workspace_requirements(
            &plan,
            view,
            BinnedStatistic::Count,
            options,
            context,
        )
        .unwrap()),
    )
    .unwrap();
    let mut parallel = [BinEstimate::default(); 20];
    let mut report = [BinnedDiagnostics::default(); 4];
    binned_statistics_into(
        &plan,
        view,
        None,
        BinnedStatistic::Count,
        options,
        &mut parallel,
        &mut report,
        &mut w,
        context,
    )
    .unwrap();
    options = options.with_execution_policy(ExecutionPolicy::Sequential);
    let mut sequential = parallel;
    binned_statistics_into(
        &plan,
        view,
        None,
        BinnedStatistic::Count,
        options,
        &mut sequential,
        &mut report,
        &mut w,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(parallel, sequential);
    let mut cdf_with_options = CdfWorkspace::default();
    options = options.with_execution_policy(ExecutionPolicy::Parallel(workers));
    cdf_with_options
        .try_reserve(&(CdfWorkspace::workspace_requirements(view, 2, options, context).unwrap()))
        .unwrap();
    let mut p = [0.; 8];
    cdf_into(
        view,
        &[3., 6.],
        CdfMethod::Empirical,
        options,
        &mut p,
        &mut cdf_with_options,
        context,
    )
    .unwrap();
    let expected = p;
    options = options.with_execution_policy(ExecutionPolicy::Sequential);
    cdf_into(
        view,
        &[3., 6.],
        CdfMethod::Empirical,
        options,
        &mut p,
        &mut cdf_with_options,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(p, expected);
    let kde = GaussianKde::build(view, 0, KdeOptions::default(), context).unwrap();
    let mut wk = KdeWorkspace::default();
    options = options.with_execution_policy(ExecutionPolicy::Parallel(workers));
    wk.try_reserve(&(KdeWorkspace::workspace_requirements(&kde, view, options, context).unwrap()))
        .unwrap();
    let mut p = [0.; 64];
    kde_into(
        &kde,
        view,
        scioxide_stats::KdeScale::Density,
        options,
        &mut p,
        &mut wk,
        context,
    )
    .unwrap();
    let expected = p;
    options = options.with_execution_policy(ExecutionPolicy::Sequential);
    kde_into(
        &kde,
        view,
        scioxide_stats::KdeScale::Density,
        options,
        &mut p,
        &mut wk,
        ExecutionContext::sequential(),
    )
    .unwrap();
    assert_eq!(p, expected);
    let mut mask = [true; 64];
    mask[63] = false;
    let masked = view.with_validity(&mask).unwrap();
    assert!(
        kde_into(
            &kde,
            masked,
            scioxide_stats::KdeScale::Density,
            options,
            &mut p,
            &mut wk,
            context
        )
        .is_err()
    );
    assert_eq!(p, expected);
    assert!(
        cdf_into(
            view,
            &[f64::NAN, 6.],
            CdfMethod::Empirical,
            options,
            &mut [0.; 8],
            &mut cdf_with_options,
            context
        )
        .is_err()
    );
}
#[test]
fn broadcast_empty_and_missing_inputs() {
    let x = [1.0];
    let weights = [1.0, 0., 2., 3.];
    let view = BatchView::new(
        &x,
        StridedLayout::new_broadcast(&[4], &[0]).unwrap(),
        Axis::Index(0),
    )
    .unwrap()
    .with_weights(Weights::<_, Mass>::new(&weights))
    .unwrap();
    let options = StatisticsOptions::new();
    let context = ExecutionContext::sequential();
    let mut w = CdfWorkspace::default();
    w.try_reserve(&(CdfWorkspace::workspace_requirements(view, 2, options, context).unwrap()))
        .unwrap();
    let mut out = [0.; 2];
    cdf_into(
        view,
        &[0., 1.],
        CdfMethod::Empirical,
        options,
        &mut out,
        &mut w,
        context,
    )
    .unwrap();
    assert_eq!(out, [0., 1.]);
    let plan = BinPlan::uniform(&[(0., 2., 2)]).unwrap();
    let empty =
        BatchView::new(&[], StridedLayout::new(&[0], &[1]).unwrap(), Axis::Index(0)).unwrap();
    let mut w = BinnedWorkspace::default();
    w.try_reserve(
        &(BinnedWorkspace::workspace_requirements(
            &plan,
            empty,
            BinnedStatistic::Count,
            options,
            context,
        )
        .unwrap()),
    )
    .unwrap();
    let mut bins = [BinEstimate::default(); 2];
    let mut report = [BinnedDiagnostics::default()];
    binned_statistics_into(
        &plan,
        empty,
        None,
        BinnedStatistic::Count,
        options,
        &mut bins,
        &mut report,
        &mut w,
        context,
    )
    .unwrap();
    assert_eq!(bins[0].count, 0);
    w.try_reserve(
        &(BinnedWorkspace::workspace_requirements(
            &plan,
            empty,
            BinnedStatistic::Density,
            options,
            context,
        )
        .unwrap()),
    )
    .unwrap();
    assert!(
        binned_statistics_into(
            &plan,
            empty,
            None,
            BinnedStatistic::Density,
            options,
            &mut bins,
            &mut report,
            &mut w,
            context
        )
        .is_err()
    );
}

#[test]
fn cdf_later_batch_failure_is_atomic_and_admission_precedes_ordering() {
    let values = [1.0_f64, 10., 2., f64::NAN, 3., 30.];
    let mask = [true, true, true, true, false, true];
    let view = BatchView::new(
        &values,
        StridedLayout::new(&[3, 2], &[2, 1]).unwrap(),
        Axis::Index(0),
    )
    .unwrap();
    let options = StatisticsOptions::new();
    let context = ExecutionContext::sequential();
    let mut workspace = CdfWorkspace::default();
    workspace
        .try_reserve(&(CdfWorkspace::workspace_requirements(view, 3, options, context).unwrap()))
        .unwrap();
    let queries = [20., 2., -1.];
    let mut output = [42.; 6];
    for method in [
        CdfMethod::Empirical,
        CdfMethod::Survival,
        CdfMethod::Estimated(QuantileRule::new(QuantileMethod::Linear)),
    ] {
        assert!(
            cdf_into(
                view,
                &queries,
                method,
                options,
                &mut output,
                &mut workspace,
                context
            )
            .is_err()
        );
        assert_eq!(output, [42.; 6]);
    }
    let input = view.with_validity(&mask).unwrap();
    let skip = options.with_missing_policy(MissingPolicy::Skip);
    cdf_into(
        input,
        &queries,
        CdfMethod::Empirical,
        skip,
        &mut output,
        &mut workspace,
        context,
    )
    .unwrap();
    for (actual, expected) in output.into_iter().zip([1., 2. / 3., 0., 0.5, 0., 0.]) {
        assert!((actual - expected).abs() <= f64::EPSILON);
    }
    let skip = options.with_non_finite_policy(NonFinitePolicy::Skip);
    cdf_into(
        view,
        &queries,
        CdfMethod::Survival,
        skip,
        &mut output,
        &mut workspace,
        context,
    )
    .unwrap();
    for (actual, expected) in output.into_iter().zip([0., 1. / 3., 1., 0.5, 1., 1.]) {
        assert!((actual - expected).abs() <= f64::EPSILON);
    }
    let expected = output;
    // Zero-weight observations are omitted before their non-finite values are read.
    let counts = [1, 1, 1, 1, 0, 1];
    let masses = [1., 1., 1., 1., 0., 1.];
    macro_rules! check_weights {
        ($input:expr) => {{
            let weights = $input;
            cdf_into(
                view.with_weights(weights).unwrap(),
                &queries,
                CdfMethod::Survival,
                options,
                &mut output,
                &mut workspace,
                context,
            )
            .unwrap();
            assert_eq!(output, expected);
        }};
    }
    check_weights!(Weights::<_, Frequency>::new(&counts));
    check_weights!(Weights::<_, Mass>::new(&masses));
    let none = [false; 6];
    let expected = output;
    assert!(
        cdf_into(
            view.with_validity(&none).unwrap(),
            &queries,
            CdfMethod::Empirical,
            options.with_missing_policy(MissingPolicy::Skip),
            &mut output,
            &mut workspace,
            context
        )
        .is_err()
    );
    assert_eq!(output, expected);
}

#[cfg(feature = "std")]
#[test]
fn ordered_mass_cdf_batches_match_parallel_execution() {
    let values: Vec<f64> = (0..64).map(|i| f64::from(i % 11)).collect();
    let weights: Vec<f64> = (0..64).map(|i| f64::from(i % 5)).collect();
    let mask: Vec<bool> = (0..64).map(|i| i % 7 != 0).collect();
    let view = BatchView::new(
        &values,
        StridedLayout::new(&[16, 4], &[4, 1]).unwrap(),
        Axis::Index(0),
    )
    .unwrap()
    .with_weights(Weights::<_, Mass>::new(&weights))
    .unwrap()
    .with_validity(&mask)
    .unwrap();
    let workers = NonZeroUsize::new(4).unwrap();
    let pool = ThreadExecutor::new(workers);
    let context = ExecutionContext::new(&pool);
    let parallel = StatisticsOptions::new()
        .with_missing_policy(MissingPolicy::Skip)
        .with_execution_policy(ExecutionPolicy::Parallel(workers));
    let mut workspace = CdfWorkspace::default();
    workspace
        .try_reserve(&(CdfWorkspace::workspace_requirements(view, 4, parallel, context).unwrap()))
        .unwrap();
    let queries = [6., 3., f64::NEG_INFINITY, 3.];
    for method in [
        CdfMethod::Empirical,
        CdfMethod::Survival,
        CdfMethod::Estimated(QuantileRule::inverse_ecdf()),
    ] {
        let expected = cdf_with_options(
            view,
            &queries,
            method,
            parallel.with_execution_policy(ExecutionPolicy::Sequential),
        )
        .unwrap();
        let mut output = [0.; 16];
        cdf_into(
            view,
            &queries,
            method,
            parallel,
            &mut output,
            &mut workspace,
            context,
        )
        .unwrap();
        assert_eq!(output.as_slice(), expected);
    }
}

#[test]
fn binned_admission_failures_are_atomic_and_skips_precede_values() {
    let x = [0.25_f64, 1.25, f64::NAN, 1.75];
    let values = [2., 4., f64::NAN, 8.];
    let mask = [true, true, false, true];
    let counts = [1, 1, 0, 1];
    let masses = [1., 1., 0., 1.];
    let view = BatchView::new(
        &x,
        StridedLayout::new(&[2, 2], &[2, 1]).unwrap(),
        Axis::Index(1),
    )
    .unwrap();
    let plan = BinPlan::uniform(&[(0., 2., 2)]).unwrap();
    let options = StatisticsOptions::new();
    let context = ExecutionContext::sequential();
    for statistic in [
        BinnedStatistic::Count,
        BinnedStatistic::Mean,
        BinnedStatistic::Median,
        BinnedStatistic::StandardDeviation,
    ] {
        let mut workspace = BinnedWorkspace::default();
        workspace
            .try_reserve(
                &(BinnedWorkspace::workspace_requirements(
                    &plan, view, statistic, options, context,
                )
                .unwrap()),
            )
            .unwrap();
        let sentinel = BinEstimate {
            count: 42,
            mass: Some(42.),
            value: Some(42.),
        };
        let mut output = [sentinel; 4];
        let mut diagnostics = [BinnedDiagnostics {
            skipped: 42,
            ..BinnedDiagnostics::default()
        }; 2];
        let expected = diagnostics;
        assert!(
            binned_statistics_into(
                &plan,
                view,
                Some(&values),
                statistic,
                options,
                &mut output,
                &mut diagnostics,
                &mut workspace,
                context
            )
            .is_err()
        );
        assert_eq!(output, [sentinel; 4]);
        assert_eq!(diagnostics, expected);
        macro_rules! check_input {
            ($input:expr) => {{
                let input = $input;
                binned_statistics_into(
                    &plan,
                    input,
                    Some(&values),
                    statistic,
                    options.with_missing_policy(MissingPolicy::Skip),
                    &mut output,
                    &mut diagnostics,
                    &mut workspace,
                    context,
                )
                .unwrap();
                assert_eq!(diagnostics[1].skipped, 1);
                assert_eq!(output[3].count, 1);
            }};
        }
        check_input!(view.with_validity(&mask).unwrap());
        check_input!(
            view.with_weights(Weights::<_, Frequency>::new(&counts))
                .unwrap()
        );
        check_input!(view.with_weights(Weights::<_, Mass>::new(&masses)).unwrap());
        binned_statistics_into(
            &plan,
            view,
            Some(&values),
            statistic,
            options.with_non_finite_policy(NonFinitePolicy::Skip),
            &mut output,
            &mut diagnostics,
            &mut workspace,
            context,
        )
        .unwrap();
        assert_eq!(diagnostics[1].skipped, 1);
    }
}

#[test]
fn binned_counts_round_once_and_detect_frequency_overflow() {
    let x = [0.25_f32, 1.25];
    let view = BatchView::new(&x, StridedLayout::new(&[2], &[1]).unwrap(), Axis::Index(0)).unwrap();
    let plan = BinPlan::uniform(&[(0., 2., 2)]).unwrap();
    let options = StatisticsOptions::new();
    let context = ExecutionContext::sequential();
    let counts = [(1u64 << 24) + 1, (1u64 << 24) + 3];
    let result = binned_statistics(
        &plan,
        view.with_weights(Weights::<_, Frequency>::new(&counts))
            .unwrap(),
        None,
        BinnedStatistic::Count,
        options,
    )
    .unwrap();
    assert_eq!(result.bins[0].count as u64, counts[0]);
    assert_eq!(result.bins[1].count as u64, counts[1]);
    assert_eq!(result.bins[0].value, Some((1u32 << 24) as f32));
    assert_eq!(result.bins[1].value, Some(((1u32 << 24) + 4) as f32));
    let plan = BinPlan::uniform(&[(0., 2., 1)]).unwrap();
    let mut workspace = BinnedWorkspace::default();
    workspace
        .try_reserve(
            &(BinnedWorkspace::workspace_requirements(
                &plan,
                view,
                BinnedStatistic::Count,
                options,
                context,
            )
            .unwrap()),
        )
        .unwrap();
    let mut output = [BinEstimate {
        count: 7,
        mass: Some(7.),
        value: Some(7.),
    }];
    let mut diagnostics = [BinnedDiagnostics::default()];
    let expected = output;
    assert!(
        binned_statistics_into(
            &plan,
            view.with_weights(Weights::<_, Frequency>::new(&[usize::MAX as u64, 1]))
                .unwrap(),
            None,
            BinnedStatistic::Count,
            options,
            &mut output,
            &mut diagnostics,
            &mut workspace,
            context
        )
        .is_err()
    );
    assert_eq!(output, expected);
    assert_eq!(diagnostics, [BinnedDiagnostics::default()]);
}

#[cfg(target_pointer_width = "64")]
#[test]
fn binned_integer_projections_avoid_binary64_double_rounding() {
    let x = [0.5_f32];
    let view = BatchView::new(&x, StridedLayout::new(&[1], &[1]).unwrap(), Axis::Index(0)).unwrap();
    let counts = [(1u64 << 53) + (1u64 << 29) + 1];
    let view = view
        .with_weights(Weights::<_, Frequency>::new(&counts))
        .unwrap();
    let plan = BinPlan::uniform(&[(0., 1., 1)]).unwrap();
    let expected = ((1u64 << 53) + (1u64 << 30)) as f32;
    for statistic in [
        BinnedStatistic::Count,
        BinnedStatistic::Mass,
        BinnedStatistic::CumulativeFrequency,
    ] {
        let result =
            binned_statistics(&plan, view, None, statistic, StatisticsOptions::new()).unwrap();
        assert_eq!(result.bins[0].value, Some(expected));
        assert_eq!(result.bins[0].mass, Some(expected));
        assert_eq!(result.diagnostics[0].admitted_mass, Some(expected));
    }
}

#[cfg(feature = "std")]
#[test]
fn binned_parallel_weighted_reductions_and_late_errors_match_sequential() {
    let x: Vec<f64> = (0..64).map(|i| f64::from(i % 11)).collect();
    let weights: Vec<f64> = (0..64).map(|i| f64::from(i % 5)).collect();
    let view = BatchView::new(
        &x,
        StridedLayout::new(&[16, 4], &[4, 1]).unwrap(),
        Axis::Index(0),
    )
    .unwrap()
    .with_weights(Weights::<_, Mass>::new(&weights))
    .unwrap();
    let plan = BinPlan::uniform(&[(0., 10., 5)]).unwrap();
    let workers = NonZeroUsize::new(4).unwrap();
    let pool = ThreadExecutor::new(workers);
    let context = ExecutionContext::new(&pool);
    let parallel =
        StatisticsOptions::new().with_execution_policy(ExecutionPolicy::Parallel(workers));
    for statistic in [
        BinnedStatistic::Count,
        BinnedStatistic::Mean,
        BinnedStatistic::StandardDeviation,
        BinnedStatistic::Median,
    ] {
        let expected =
            binned_statistics(&plan, view, Some(&x), statistic, StatisticsOptions::new()).unwrap();
        let mut workspace = BinnedWorkspace::default();
        workspace
            .try_reserve(
                &(BinnedWorkspace::workspace_requirements(
                    &plan, view, statistic, parallel, context,
                )
                .unwrap()),
            )
            .unwrap();
        let mut output = [BinEstimate::default(); 20];
        let mut diagnostics = [BinnedDiagnostics::default(); 4];
        binned_statistics_into(
            &plan,
            view,
            Some(&x),
            statistic,
            parallel,
            &mut output,
            &mut diagnostics,
            &mut workspace,
            context,
        )
        .unwrap();
        assert_eq!(output.as_slice(), expected.bins);
        assert_eq!(diagnostics.as_slice(), expected.diagnostics);
        let mut invalid = x.clone();
        invalid[63] = f64::NAN;
        let bad = BatchView::new(
            &invalid,
            StridedLayout::new(&[16, 4], &[4, 1]).unwrap(),
            Axis::Index(0),
        )
        .unwrap();
        assert!(
            binned_statistics_into(
                &plan,
                bad,
                Some(&x),
                statistic,
                parallel,
                &mut output,
                &mut diagnostics,
                &mut workspace,
                context
            )
            .is_err()
        );
        assert_eq!(output.as_slice(), expected.bins);
        assert_eq!(diagnostics.as_slice(), expected.diagnostics);
        // Fail during result projection after earlier bin state has been reduced.
        if statistic == BinnedStatistic::Mean {
            workspace
                .try_reserve(
                    &(BinnedWorkspace::workspace_requirements(
                        &plan,
                        view,
                        BinnedStatistic::Sum,
                        parallel,
                        context,
                    )
                    .unwrap()),
                )
                .unwrap();
            assert!(
                binned_statistics_into(
                    &plan,
                    view,
                    Some(&[f64::MAX; 64]),
                    BinnedStatistic::Sum,
                    parallel,
                    &mut output,
                    &mut diagnostics,
                    &mut workspace,
                    context
                )
                .is_err()
            );
            assert_eq!(output.as_slice(), expected.bins);
            assert_eq!(diagnostics.as_slice(), expected.diagnostics);
        }
    }
}
