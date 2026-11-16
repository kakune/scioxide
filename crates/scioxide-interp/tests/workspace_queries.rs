//! Allocation and reuse regressions for method-specific interpolation workspaces.

#![allow(clippy::unwrap_used)]

mod support;

use support::assert_error;

use scioxide_core::SciOxideError;
use scioxide_interp::{
    BSpline, BSplineBasis, BulirschStoerInterpolator, ExtrapolationPolicy, Interpolant,
    NevilleInterpolator,
};
use scioxide_interp::{Differentiable, TwiceDifferentiable};

#[test]
fn warmed_workspace_queries_execute() {
    let x = [0.0_f64, 1.0, 2.0, 3.0];
    let polynomial_y = [1.0_f64, 2.0, 5.0, 10.0];
    let rational =
        BulirschStoerInterpolator::new(x.to_vec(), vec![1.0, 0.5, 1.0 / 3.0, 0.25]).unwrap();
    let spline = BSpline::new(
        [BSplineBasis::new(vec![0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0], 2).unwrap()],
        vec![0.0, 1.0, 1.0, 0.0],
    )
    .unwrap();
    let polynomial = NevilleInterpolator::new(x.to_vec(), polynomial_y.to_vec()).unwrap();
    let mut workspace = polynomial.workspace().unwrap();
    let mut spline_workspace = spline.workspace().unwrap();
    let mut rational_workspace = rational.workspace().unwrap();

    let result = (0..32)
        .try_fold(0.0, |total, _| {
            let (mut neville, mut estimate) = (0.0, 0.0);
            polynomial.evaluate_with_error_into(
                [1.5],
                ExtrapolationPolicy::Reject,
                &mut neville,
                &mut estimate,
                &mut workspace,
            )?;
            let mut rational_value = 0.0;
            rational.evaluate_into(
                [1.5],
                ExtrapolationPolicy::Reject,
                &mut rational_value,
                &mut rational_workspace,
            )?;
            let mut spline_value = 0.0;
            spline.evaluate_into(
                [1.5],
                ExtrapolationPolicy::Reject,
                &mut spline_value,
                &mut spline_workspace,
            )?;
            let mut spline_derivative = 0.0;
            spline.derivative_into(
                [1.5],
                [1],
                None,
                ExtrapolationPolicy::Reject,
                &mut spline_derivative,
                &mut spline_workspace,
            )?;
            Ok::<f64, SciOxideError>(
                total + neville + rational_value + spline_value + spline_derivative,
            )
        })
        .unwrap();

    assert!(result.is_finite());
}

#[test]
fn unreserved_and_failed_queries_leave_workspace_reusable_without_allocation() {
    use scioxide_core::Capacity;
    use scioxide_interp::Interpolant;
    // Nine nodes need more scratch than a singleton model's inline workspace.
    let nodes: Vec<f32> = (0_u8..9).map(f32::from).collect();
    let values: Vec<f32> = nodes.iter().map(|x| x * x).collect();
    let polynomial = NevilleInterpolator::new(nodes, values).unwrap();
    let singleton = NevilleInterpolator::new(vec![0.0_f32], vec![0.0]).unwrap();
    let mut workspace = singleton.workspace().unwrap();
    assert_eq!(
        polynomial.workspace_capacity(&workspace),
        Capacity::NeedsGrowth
    );
    let mut value = 7.0;
    let outcome = polynomial.evaluate_into(
        [0.5],
        ExtrapolationPolicy::Reject,
        &mut value,
        &mut workspace,
    );
    assert_error(
        &outcome,
        scioxide_core::SciOxideErrorKind::DimensionMismatch,
    );
    assert_eq!(value, 7.0);
    let mut workspace = polynomial.workspace().unwrap();
    let pole = BulirschStoerInterpolator::new(vec![0.0_f32, 1.0], vec![1.0, -1.0]).unwrap();
    let mut rational_workspace = pole.workspace().unwrap();
    {
        let mut rational_value = 7.0;
        assert!(
            pole.evaluate_into(
                [0.5],
                ExtrapolationPolicy::Reject,
                &mut rational_value,
                &mut rational_workspace
            )
            .is_err()
        );
        assert_eq!(rational_value, 7.0);
        polynomial
            .evaluate_into(
                [0.5],
                ExtrapolationPolicy::Reject,
                &mut value,
                &mut workspace,
            )
            .unwrap();
        assert!((value - 0.25).abs() < 2e-6);
    };
    assert_eq!(
        polynomial.workspace_capacity(&workspace),
        Capacity::Sufficient
    );
}

#[test]
fn prepared_bicubic_value_gradient_and_hessian_execute() {
    use scioxide_interp::{BicubicInterpolator, Interpolant};
    let x = vec![-1.0_f64, 0.0, 0.5, 2.0];
    let y = vec![-2.0, 0.0, 1.0];
    let z = y
        .iter()
        .flat_map(|&b| x.iter().map(move |&a| a * a + b * b + a * b))
        .collect();
    let interp = BicubicInterpolator::new(x, y, z).unwrap();
    let policy = ExtrapolationPolicy::Reject;
    let mut workspace = interp.workspace().unwrap();
    let (mut value, mut gradient, mut hessian) = (0.0, [0.0; 2], [[0.0; 2]; 2]);
    {
        interp
            .evaluate_into([0.25, 0.5], policy, &mut value, &mut workspace)
            .unwrap();
        interp
            .gradient_into([0.25, 0.5], policy, &mut gradient, &mut workspace)
            .unwrap();
        interp
            .hessian_into([0.25, 0.5], policy, &mut hessian, &mut workspace)
            .unwrap();
        // Owned scalar queries stage inline and allocate nothing either.
        assert_eq!(interp.evaluate([0.25, 0.5], policy).unwrap(), value);
    };
    assert!((value - 0.4375).abs() < 1e-12);
    assert!((gradient[0] - 1.0).abs() < 1e-12);
    assert!((hessian[0][1] - 1.0).abs() < 1e-12);
}

#[test]
fn clamped_affine_calculus_and_reports_execute_and_commit_atomically() {
    use scioxide_interp::{Interpolant, NewtonInterpolator};
    let polynomial = NewtonInterpolator::new(
        vec![0.0_f64, 1.0, 2.0],
        vec![vec![0.0, 1.0], vec![1.0, 3.0], vec![4.0, 5.0]],
    )
    .unwrap();
    let prepared = polynomial.prepare_antiderivative().unwrap();
    let mut output = [99.0; 2];
    let mut scratch = [0.0; 2];
    let mut workspace = polynomial.workspace().unwrap();
    {
        let report = polynomial
            .report([3.0], ExtrapolationPolicy::Clamp, &mut workspace)
            .unwrap();
        assert_eq!(report.effective(), [2.0]);
        for _ in 0..4 {
            assert!(
                prepared
                    .integral_into(
                        -1.0,
                        3.0,
                        ExtrapolationPolicy::Reject,
                        &mut output,
                        &mut scratch
                    )
                    .is_err()
            );
            assert_eq!(output, [99.0; 2]);
            prepared
                .integral_into(
                    -1.0,
                    3.0,
                    ExtrapolationPolicy::Clamp,
                    &mut output,
                    &mut scratch,
                )
                .unwrap();
            assert!((output[0] - 20.0 / 3.0).abs() < 1e-12);
            polynomial
                .gradient_into(
                    [3.0],
                    ExtrapolationPolicy::Clamp,
                    &mut output,
                    &mut workspace,
                )
                .unwrap();
            assert_eq!(output, [0.0; 2]);
            output = [99.0; 2];
        }
    };
}

#[test]
fn reserved_family_batches_execute() {
    use scioxide_core::{ExecutionContext, ExecutionPolicy, NdView, NdViewMut, StridedLayout};
    use scioxide_interp::*;
    fn check_protocol<I: Interpolant<f64, 1>>(model: &I) {
        let plan = BatchEvaluator::new(model, ExecutionPolicy::Sequential);
        let mut workspace = plan.workspace(BatchRequest::values(3)).unwrap();
        let mut data = vec![0.; 3 * model.value_dimension()];
        let q = NdView::new(
            &[0.25, 0.5, 0.75],
            StridedLayout::new(&[3, 1], &[1, 1]).unwrap(),
        )
        .unwrap();
        let shape = [3, model.value_dimension()];
        let strides = [model.value_dimension(), 1];
        let mut output =
            NdViewMut::new(&mut data, StridedLayout::new(&shape, &strides).unwrap()).unwrap();
        for _ in 0..8 {
            plan.evaluate_into(
                &q,
                ExtrapolationPolicy::Reject,
                &mut output,
                None,
                &mut workspace,
                ExecutionContext::sequential(),
            )
            .unwrap();
        }
    }
    let nodes = vec![0., 1., 2.];
    check_protocol(
        &NewtonInterpolator::new(
            nodes.clone(),
            vec![vec![1., 2., 3.], vec![2., 4., 6.], vec![3., 6., 9.]],
        )
        .unwrap(),
    );
    check_protocol(&BulirschStoerInterpolator::new(nodes.clone(), vec![1., 0.5, 1. / 3.]).unwrap());
    check_protocol(&NevilleInterpolator::new(nodes.clone(), vec![1., 2., 5.]).unwrap());
    check_protocol(&BSpline::linear(nodes, vec![1., 2., 3.]).unwrap());
}
