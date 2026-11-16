//! Axis, mask, trim, order, and executor contracts for descriptive statistics.
#![allow(clippy::unwrap_used)]
#[path = "support/reductions.rs"]
mod reduction_test;
use scioxide_core::{ExecutionContext, NumericScalar, RealScalar};
use scioxide_linalg::MatrixView;
use scioxide_stats::*;

/// Run a block once per sampling weight model, binding the model type and its label.
macro_rules! each_model {
    ($model:ident, $mode:ident => $body:block) => {{
        {
            type $model = Frequency;
            let $mode = WeightSemantics::Frequency;
            let _ = $mode;
            $body
        }
        {
            type $model = Reliability;
            let $mode = WeightSemantics::Reliability;
            let _ = $mode;
            $body
        }
        {
            type $model = Analytic;
            let $mode = WeightSemantics::Analytic;
            let _ = $mode;
            $body
        }
    }};
}

fn scalar<S: RealScalar>(x: f64) -> S {
    S::try_from_f64_lossy(x).unwrap()
}
fn close<S: RealScalar>(actual: S, expected: f64) {
    assert!(
        (actual - scalar::<S>(expected)).abs()
            <= scalar::<S>(256.0)
                * <S as RealScalar>::EPSILON
                * scalar::<S>(expected.abs().max(1.0))
    );
}
fn exercise<S: RealScalar + NumericScalar<Real = S>>() {
    let values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0].map(scalar::<S>);
    let view_shape = [2, 3];
    let view_strides = [3, 1];
    let view = BatchView::new(
        &values,
        scioxide_core::StridedLayout::new(&view_shape, &view_strides).unwrap(),
        Axis::All,
    )
    .unwrap();
    let options_axis = Axis::All;
    let options = StatisticsOptions::new().with_moment_order(MomentOrder::Fourth);
    let limits = TrimLimits::new(Some(scalar::<S>(1.0)), Some(scalar::<S>(4.0))).unwrap();
    let s = reduction_test::trimmed(
        view.with_axis(options_axis).unwrap(),
        limits,
        options,
        ExecutionContext::sequential(),
    )
    .unwrap()[0];
    assert_eq!(s.count(), 4);
    close(s.mean().unwrap().unwrap(), 2.5);
    close(s.minimum().unwrap(), 1.0);
    close(s.maximum().unwrap(), 4.0);
    close(s.variance(1).unwrap().unwrap(), 5.0 / 3.0);
    close(
        s.standard_error(1).unwrap().unwrap(),
        (5.0_f64 / 12.0).sqrt(),
    );
    let s = reduction_test::trimmed(
        view.with_axis(options_axis).unwrap(),
        limits.with_inclusive(
            scioxide_stats::TrimInclusion::Exclusive,
            scioxide_stats::TrimInclusion::Exclusive,
        ),
        options,
        ExecutionContext::sequential(),
    )
    .unwrap()[0];
    assert_eq!(s.count(), 2);
    close(s.mean().unwrap().unwrap(), 2.5);
    let limits = TrimLimits::new(Some(scalar::<S>(9.0)), None).unwrap();
    assert_eq!(
        reduction_test::trimmed(
            view.with_axis(options_axis).unwrap(),
            limits,
            options,
            ExecutionContext::sequential()
        )
        .unwrap()[0]
            .count(),
        0
    );
    let fifth = reduction_test::power(
        view.with_axis(options_axis).unwrap(),
        5,
        None,
        options,
        ExecutionContext::sequential(),
    )
    .unwrap();
    close(fifth[0].unwrap(), 0.0);
    let sixth = reduction_test::power(
        view.with_axis(options_axis).unwrap(),
        6,
        None,
        options,
        ExecutionContext::sequential(),
    )
    .unwrap();
    close(sixth[0].unwrap(), 85.18229166666667);
    let raw = reduction_test::power(
        view.with_axis(options_axis).unwrap(),
        3,
        Some(scalar::<S>(0.0)),
        options,
        ExecutionContext::sequential(),
    )
    .unwrap();
    close(raw[0].unwrap(), 37.5);
    for (axis, expected) in [
        (Axis::Index(1), vec![2.0 / 3.0, 2.0 / 3.0]),
        (Axis::Index(0), vec![2.25, 2.25, 2.25]),
    ] {
        let result = reduction_test::power(
            view.with_axis(axis).unwrap(),
            2,
            None,
            StatisticsOptions::new(),
            ExecutionContext::sequential(),
        )
        .unwrap();
        for (a, e) in result.into_iter().zip(expected) {
            close(a.unwrap(), e);
        }
    }
    let weights = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0].map(scalar::<S>);
    let w = MatrixView::from_slice(2, 3, &weights).unwrap();
    each_model!(M, _mode => {
        let s = reduction_test::weighted_covariance::<_, M>(
            view.with_axis(options_axis).unwrap(),
            view.with_axis(options_axis).unwrap(),
            w,
            options,
            ExecutionContext::sequential(),
        )
        .unwrap()[0];
        close(s.correlation().unwrap().unwrap(), 1.0);
        let marginal = s.marginal_x();
        close(
            s.population_covariance().unwrap().unwrap(),
            marginal
                .weighted_population_variance()
                .unwrap()
                .unwrap()
                .try_to_f64_exact()
                .unwrap(),
        );
    });
}
#[test]
fn real_precision_and_axis_coverage() {
    exercise::<f32>();
    exercise::<f64>();
}

#[test]
fn policies_and_empty_axes_are_preserved() {
    let values = [1.0_f64, f64::NAN, 3.0, 999.0];
    let mask = [true, true, true, false];
    let view_shape = [2, 2];
    let view_strides = [2, 1];
    let view = BatchView::new(
        &values,
        scioxide_core::StridedLayout::new(&view_shape, &view_strides).unwrap(),
        Axis::All,
    )
    .unwrap()
    .with_validity(&mask)
    .unwrap();
    let options_axis = Axis::All;
    let options = StatisticsOptions::new()
        .with_missing_policy(MissingPolicy::Skip)
        .with_non_finite_policy(NonFinitePolicy::Skip)
        .with_moment_order(MomentOrder::Fourth);
    let s = reduction_test::trimmed(
        view.with_axis(options_axis).unwrap(),
        TrimLimits::new(None, None).unwrap(),
        options,
        ExecutionContext::sequential(),
    )
    .unwrap()[0];
    assert_eq!(s.count(), 2);
    assert_eq!(s.mean().unwrap(), Some(2.0));
    assert_eq!(
        reduction_test::power(
            view.with_axis(options_axis).unwrap(),
            6,
            None,
            options,
            ExecutionContext::sequential()
        )
        .unwrap(),
        vec![Some(1.0)]
    );
    assert!(
        reduction_test::power(
            view.with_axis(options_axis).unwrap(),
            2,
            Some(f64::INFINITY),
            options,
            ExecutionContext::sequential()
        )
        .is_err()
    );
    assert!(TrimLimits::new(Some(2.0), Some(1.0)).is_err());
    let empty_shape = [0, 3];
    let empty_strides = [3, 1];
    let empty = BatchView::new(
        &[] as &[f64],
        scioxide_core::StridedLayout::new(&empty_shape, &empty_strides).unwrap(),
        Axis::All,
    )
    .unwrap();
    assert_eq!(
        reduction_test::power(
            empty.with_axis(Axis::Index(0)).unwrap(),
            0,
            None,
            StatisticsOptions::new(),
            ExecutionContext::sequential()
        )
        .unwrap(),
        vec![None; 3]
    );
    assert!(
        reduction_test::power(
            empty.with_axis(Axis::Index(1)).unwrap(),
            0,
            None,
            StatisticsOptions::new(),
            ExecutionContext::sequential()
        )
        .unwrap()
        .is_empty()
    );
    let extremes = [-f64::MAX, f64::MAX];
    let view_shape = [1, 2];
    let view_strides = [2, 1];
    let view = BatchView::new(
        &extremes,
        scioxide_core::StridedLayout::new(&view_shape, &view_strides).unwrap(),
        Axis::All,
    )
    .unwrap();
    assert_eq!(
        reduction_test::power(
            view.with_axis(options_axis).unwrap(),
            1,
            None,
            options,
            ExecutionContext::sequential()
        )
        .unwrap(),
        vec![Some(0.0)]
    );
    assert_eq!(
        reduction_test::power(
            view.with_axis(options_axis).unwrap(),
            5,
            None,
            options,
            ExecutionContext::sequential()
        )
        .unwrap(),
        vec![Some(0.0)]
    );
}

#[test]
fn weighted_covariance_merge_and_failure_are_atomic() {
    each_model!(M, mode => {
        let mut a = OnlineWeightedCovariance::<f64, M>::new(MomentOrder::Fourth);
        let mut b = OnlineWeightedCovariance::<f64, M>::new(MomentOrder::Fourth);
        let mut whole = OnlineWeightedCovariance::<f64, M>::new(MomentOrder::Fourth);
        for (i, (x, y, w)) in [(1.0, 2.0, 1.0), (3.0, 9.0, 2.0), (7.0, 4.0, 3.0)]
            .into_iter()
            .enumerate()
        {
            whole.observe(x, y, w).unwrap();
            if i == 0 {
                a.observe(x, y, w).unwrap();
            } else {
                b.observe(x, y, w).unwrap();
            }
        }
        a.merge(&b).unwrap();
        let s = a.snapshot();
        let reference = whole.snapshot();
        close(
            s.correlation().unwrap().unwrap(),
            reference.correlation().unwrap().unwrap(),
        );
        close(
            s.covariance_of_weighted_means().unwrap().unwrap(),
            reference.covariance_of_weighted_means().unwrap().unwrap(),
        );
        let before = a;
        assert!(a.observe(1.0, f64::NAN, 1.0).is_err());
        assert_eq!(a, before);
        assert!(
            a.merge(&OnlineWeightedCovariance::<f64, M>::new(MomentOrder::Second))
                .is_err()
        );
        assert_eq!(a, before);
        if mode != WeightSemantics::Frequency {
            assert!(s.marginal_x().skewness(BiasCorrection::Corrected).is_err());
        }
    });
}

#[cfg(feature = "std")]
#[test]
fn all_extended_reductions_replay_with_a_fixed_executor_plan() {
    use core::num::NonZeroUsize;
    use scioxide_core::{ExecutionPolicy, ThreadExecutor};
    use scioxide_stats::*;
    let workers = NonZeroUsize::new(4).unwrap();
    let executor = ThreadExecutor::new(workers);
    let values = (0..4096)
        .map(|i| f64::from(i % 17) - 8.0)
        .collect::<Vec<_>>();
    let weights = vec![1.0_f64; values.len()];
    let view_shape = [64, 64];
    let view_strides = [64, 1];
    let view = BatchView::new(
        &values,
        scioxide_core::StridedLayout::new(&view_shape, &view_strides).unwrap(),
        Axis::All,
    )
    .unwrap();
    let weights = MatrixView::from_slice(64, 64, &weights).unwrap();
    let options_axis = Axis::All;
    let options = StatisticsOptions::new()
        .with_merge_policy(MergePolicy::StablePartitions)
        .with_execution_policy(ExecutionPolicy::Parallel(workers))
        .with_moment_order(MomentOrder::Fourth);
    let context = ExecutionContext::new(&executor);
    let expected =
        reduction_test::moments(view.with_axis(options_axis).unwrap(), options, context).unwrap();
    let power = reduction_test::power(
        view.with_axis(options_axis).unwrap(),
        8,
        None,
        options,
        context,
    )
    .unwrap();
    let cov = reduction_test::weighted_covariance::<_, Frequency>(
        view.with_axis(options_axis).unwrap(),
        view.with_axis(options_axis).unwrap(),
        weights,
        options,
        context,
    )
    .unwrap();
    for _ in 0..8 {
        assert_eq!(
            reduction_test::moments(view.with_axis(options_axis).unwrap(), options, context)
                .unwrap(),
            expected
        );
        assert_eq!(
            reduction_test::power(
                view.with_axis(options_axis).unwrap(),
                8,
                None,
                options,
                context
            )
            .unwrap(),
            power
        );
        assert_eq!(
            reduction_test::weighted_covariance::<_, Frequency>(
                view.with_axis(options_axis).unwrap(),
                view.with_axis(options_axis).unwrap(),
                weights,
                options,
                context
            )
            .unwrap(),
            cov
        );
    }
}
