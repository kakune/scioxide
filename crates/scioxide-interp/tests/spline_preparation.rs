//! Preparation storage, singular admission and fallible band construction.
#![allow(clippy::unwrap_used)]

use scioxide_core::{ExecutionContext, SciOxideErrorKind};
use scioxide_interp::{
    BSplineBasis, ExtrapolationPolicy, Interpolant, PenalizedSplinePlan, SplinePenalty,
    TensorSplineInterpolationPlan,
};

fn basis(n: usize, p: usize) -> BSplineBasis<f64> {
    let knots = core::iter::repeat_n(0., p + 1)
        .chain((1..n - p).map(|i| i as f64 / (n - p) as f64))
        .chain(core::iter::repeat_n(1., p + 1))
        .collect();
    BSplineBasis::new(knots, p).unwrap()
}

fn nodes(basis: &BSplineBasis<f64>) -> Vec<f64> {
    let p = basis.degree();
    (0..basis.control_count())
        .map(|i| basis.knots()[i + 1..=i + p].iter().sum::<f64>() / p as f64)
        .collect()
}

#[test]
fn repeated_knots_and_clamped_endpoints_use_the_shared_span_limits() {
    for multiplicity in [2, 3, 4] {
        let knots = core::iter::repeat_n(0., 4)
            .chain(core::iter::repeat_n(0.5, multiplicity))
            .chain(core::iter::repeat_n(1., 4))
            .collect();
        let basis = BSplineBasis::new(knots, 3).unwrap();
        let nodes = if multiplicity == 4 {
            vec![0., 0.1, 0.3, 0.49, 0.5, 0.7, 0.9, 1.]
        } else {
            nodes(&basis)
        };
        let values: Vec<_> = nodes.iter().map(|&x| x * x).collect();
        let model = TensorSplineInterpolationPlan::new([basis], [nodes.clone()])
            .unwrap()
            .fit(&values)
            .unwrap();
        for (&x, &expected) in nodes.iter().zip(&values) {
            assert!(
                (model.evaluate([x], ExtrapolationPolicy::Reject).unwrap() - expected).abs()
                    <= 64. * f64::EPSILON
            );
        }
    }
}

#[test]
fn degree_zero_and_descending_cubic_axes_fit_without_query_allocations() {
    let zero = BSplineBasis::new(vec![0., 1., 2., 3.], 0).unwrap();
    let plan = TensorSplineInterpolationPlan::new([zero], [vec![0.5, 1.5, 2.5]]).unwrap();
    assert_eq!(
        plan.fit(&[1., 2., 3.]).unwrap().coefficients(),
        &[1., 2., 3.]
    );

    let basis = basis(128, 3);
    let mut nodes = nodes(&basis);
    nodes.reverse();
    let values: Vec<_> = nodes.iter().map(|&x| [x * x, 1. - x]).collect();
    let plan = TensorSplineInterpolationPlan::new([basis], [nodes]).unwrap();
    let mut workspace = plan.workspace(2).unwrap();
    let mut out = vec![0.; 256];
    plan.fit_into(
        &values,
        out.as_mut_slice(),
        &mut workspace,
        ExecutionContext::sequential_native(),
    )
    .unwrap();
    let spline = plan.fit(&values).unwrap();
    for (&actual, &expected) in out.iter().zip(spline.coefficients()) {
        assert!((actual - expected).abs() < 32. * f64::EPSILON);
    }
}

#[test]
fn inactive_high_order_penalty_has_degree_sized_storage_and_zero_diagnostics() {
    let n = 128;
    let basis = basis(n, 3);
    let samples = nodes(&basis);
    let penalty = SplinePenalty::new(0., n - 1).unwrap();
    assert_eq!(
        PenalizedSplinePlan::preparation_bytes(&basis, n, penalty).unwrap(),
        PenalizedSplinePlan::preparation_bytes(&basis, n, SplinePenalty::new(0., 2).unwrap())
            .unwrap(),
    );
    let values: Vec<_> = samples.iter().map(|x| x * x).collect();
    let plan = PenalizedSplinePlan::new(basis, samples, None, penalty).unwrap();
    let fit = plan.fit(&values).unwrap();
    assert_eq!(fit.report().penalty(), 0.);
    assert!(fit.report().normal_residual() < 128. * f64::EPSILON);
    assert!(fit.report().data_fidelity() < 1e-25);
    let mut workspace = plan.workspace(1).unwrap();
    let mut out = vec![0.; n];
    let _ = plan
        .fit_into(
            &values,
            out.as_mut_slice(),
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
}

#[test]
fn preparation_queries_report_invalid_shapes_and_overflow() {
    let basis = basis(8, 3);
    assert_eq!(
        PenalizedSplinePlan::preparation_bytes(&basis, 0, SplinePenalty::new(0., 2).unwrap())
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        PenalizedSplinePlan::preparation_bytes(&basis, 4, SplinePenalty::new(1., 8).unwrap())
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::InvalidInput
    );
    assert_eq!(
        PenalizedSplinePlan::preparation_bytes(
            &basis,
            usize::MAX,
            SplinePenalty::new(1., 2).unwrap()
        )
        .unwrap_err()
        .kind(),
        SciOxideErrorKind::SizeOverflow
    );
    assert_eq!(
        TensorSplineInterpolationPlan::<f64, 0>::preparation_bytes(&[])
            .unwrap_err()
            .kind(),
        SciOxideErrorKind::InvalidInput
    );
}
