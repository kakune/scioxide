//! Focused numerical and contract tests for tensor, penalized, and rational splines.

#![allow(clippy::unwrap_used, clippy::panic)]

use scioxide_core::{
    Budget, BudgetKind, Complex, ExecutionContext, ProblemToken, SolveBudgets, SolveIdentity,
    SolveState,
};
#[cfg(feature = "std")]
use scioxide_core::{NdView, NdViewMut, StridedLayout};
use scioxide_interp::{
    BSpline, BSplineBasis, Differentiable, ExtrapolationPolicy, Integrable, Interpolant, Nurbs,
    PenalizedSplinePlan, SplinePenalty, TensorSplineInterpolationPlan,
};

const REJECT: ExtrapolationPolicy<f64> = ExtrapolationPolicy::Reject;

fn linear_basis() -> BSplineBasis<f64> {
    BSplineBasis::new(vec![0.0, 0.0, 1.0, 1.0], 1).unwrap()
}

#[test]
fn basis_partition_derivative_and_integral() {
    let basis = linear_basis();
    let mut workspace = basis.workspace().unwrap();
    let mut values = [0.0; 2];
    basis
        .evaluate_into(0.25, 0, None, REJECT, &mut values, &mut workspace)
        .unwrap();
    assert_eq!(values, [0.75, 0.25]);
    basis
        .evaluate_into(0.25, 1, None, REJECT, &mut values, &mut workspace)
        .unwrap();
    assert_eq!(values, [-1.0, 1.0]);
    basis
        .integrals_into(0.0, 1.0, REJECT, &mut values, &mut workspace)
        .unwrap();
    assert!((values[0] - 0.5).abs() < 1e-14);
    assert!((values[1] - 0.5).abs() < 1e-14);
}

#[test]
fn tensor_separability_derivatives_and_box_integral() {
    let basis = linear_basis();
    // f(x,y) = 1 + 3x + 2y at the four tensor controls.
    let spline: BSpline<f64, f64, 2> =
        BSpline::new([basis.clone(), basis], vec![1.0, 3.0, 4.0, 6.0]).unwrap();
    assert!((spline.evaluate([0.25, 0.5], REJECT).unwrap() - 2.75).abs() < 1e-14);
    let dx = spline
        .derivative([0.25, 0.5], [1, 0], None, REJECT)
        .unwrap();
    assert!((dx - 3.0).abs() < 1e-14);
    let dy = spline
        .derivative([0.25, 0.5], [0, 1], None, REJECT)
        .unwrap();
    assert!((dy - 2.0).abs() < 1e-14);
    let integral = spline.integral([0.0, 0.0], [1.0, 1.0], REJECT).unwrap();
    assert!((integral - 3.5).abs() < 1e-14);
}

#[test]
fn complex_tensor_components_are_native() {
    let basis = linear_basis();
    let spline = BSpline::new(
        [basis],
        vec![Complex::new(1.0, -1.0), Complex::new(3.0, 1.0)],
    )
    .unwrap();
    assert_eq!(
        spline.evaluate([0.5], REJECT).unwrap(),
        Complex::new(2.0, 0.0)
    );
}

#[test]
fn nurbs_common_weight_scale_and_unit_weight_equivalence() {
    let basis = linear_basis();
    let polynomial = BSpline::new([basis.clone()], vec![2.0, 6.0]).unwrap();
    let unit = Nurbs::new([basis.clone()], vec![2.0, 6.0], vec![1.0, 1.0]).unwrap();
    let scaled = Nurbs::new([basis], vec![2.0, 6.0], vec![1e200, 1e200]).unwrap();
    let expected = polynomial.evaluate([0.375], REJECT).unwrap();
    let actual = unit.evaluate([0.375], REJECT).unwrap();
    assert_eq!(actual, expected);
    let actual = scaled.evaluate([0.375], REJECT).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn rational_linear_derivative_matches_closed_form() {
    let model = Nurbs::new([linear_basis()], vec![0.0, 1.0], vec![1.0, 2.0]).unwrap();
    let value = model.evaluate([0.25], REJECT).unwrap();
    let [gradient] = model.gradient([0.25], REJECT).unwrap();
    assert!((value - 0.4).abs() < 1e-14);
    assert!((gradient - 1.28).abs() < 1e-14);
}

#[test]
fn penalized_fit_solves_normal_equations_and_supports_complex_channels() {
    let basis = BSplineBasis::new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], 2).unwrap();
    let samples = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    let plan = PenalizedSplinePlan::new(
        basis,
        samples.clone(),
        None,
        SplinePenalty::new(0.1, 2).unwrap(),
    )
    .unwrap();
    let values = samples
        .iter()
        .map(|&x| Complex::new(1.0 + 2.0 * x, -0.5 + x))
        .collect::<Vec<_>>();
    let fit = plan.fit(&values).unwrap();
    // One diagnostic per real component.
    assert_eq!(fit.diagnostics().len(), 2);
    for diagnostic in fit.diagnostics() {
        assert!(diagnostic.normal_residual() < 2e-14);
        assert!(diagnostic.data_fidelity() >= 0.0);
        assert!(diagnostic.penalty() >= 0.0);
    }
    assert!(plan.diagonal_ratio() > 0.0);
    let output = fit.spline().evaluate([0.4], REJECT).unwrap();
    assert!((output.re - 1.8).abs() < 1e-13);
    assert!((output.im + 0.1).abs() < 1e-13);
}

#[test]
fn smoothing_rejects_nonunique_and_invalid_weight_domains() {
    let basis = linear_basis();
    assert!(
        PenalizedSplinePlan::<f64>::new(
            basis.clone(),
            vec![0.0, 1.0],
            Some(vec![0.0, 0.0]),
            SplinePenalty::new(1.0, 1).unwrap(),
        )
        .is_err()
    );
    assert!(
        PenalizedSplinePlan::<f64>::new(
            basis,
            vec![0.0, 1.0],
            Some(vec![1.0, -1.0]),
            SplinePenalty::new(1.0, 1).unwrap(),
        )
        .is_err()
    );
}

#[test]
fn clustered_knots_publish_conditioning_warning_and_residual() {
    let basis =
        BSplineBasis::new(vec![0.0_f64, 0.0, 0.0, 0.49, 0.490001, 1.0, 1.0, 1.0], 2).unwrap();
    let samples = (0..33).map(|i| i as f64 / 32.0).collect::<Vec<_>>();
    let values = samples.iter().map(|x| 1.0 + x + x * x).collect::<Vec<_>>();
    let plan =
        PenalizedSplinePlan::<f64>::new(basis, samples, None, SplinePenalty::new(1e-4, 2).unwrap())
            .unwrap();
    assert!(plan.diagonal_ratio() > 0.0 && plan.diagonal_ratio() <= 1.0);
    let fit = plan.fit(&values).unwrap();
    assert!(fit.diagnostics()[0].normal_residual() < 5e-14);
    assert_eq!(
        fit.report().normal_residual(),
        fit.diagnostics()[0].normal_residual()
    );
}

#[test]
fn smoothing_scope_accounts_factor_columns_and_rhs() {
    let identity = SolveIdentity::new(
        ProblemToken::new(0x5350_4c49_4e45),
        scioxide_core::algorithm_id!("interp", "penalized-spline-test", 1),
        2,
        1,
    );
    let mut state = SolveState::new();
    let mut budgets = SolveBudgets::unbounded();
    let mut context = state
        .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
        .unwrap();
    let plan = PenalizedSplinePlan::<f64>::build(
        linear_basis(),
        vec![0.0, 0.5, 1.0],
        None,
        SplinePenalty::new(0.25, 1).unwrap(),
        &mut context,
    )
    .unwrap();
    // Factor columns and right-hand sides are counted once, in nested epochs.
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(context.counts().nested().iterations(), 2);
    let mut workspace = plan.workspace(1).unwrap();
    let mut coefficients = [0.0; 2];
    let _ = plan
        .fit_into(
            &[0.0, 0.5, 1.0],
            &mut coefficients,
            &mut workspace,
            &mut context,
        )
        .unwrap();
    assert_eq!(context.counts().own().iterations(), 0);
    assert_eq!(context.counts().total().unwrap().iterations(), 3);

    // Iteration allowances are scope-local: an ancestor limit records the nested
    // work without bounding it, while invalid values fail before any solve.
    let mut limited_state = SolveState::new();
    let mut limited_budget =
        SolveBudgets::unbounded().with_limit(BudgetKind::Iteration, Budget::try_from(1).unwrap());
    let mut limited = limited_state
        .begin_solve(
            ExecutionContext::sequential(),
            identity,
            &mut limited_budget,
        )
        .unwrap();
    let mut workspace = plan.workspace(2).unwrap();
    let mut pairs = [7.0; 4];
    assert_eq!(
        plan.fit_into(
            &[[0.0, 1.0], [f64::NAN, 2.0], [1.0, 3.0]],
            &mut pairs,
            &mut workspace,
            &mut limited
        )
        .unwrap_err()
        .kind(),
        scioxide_core::SciOxideErrorKind::NonFinite
    );
    assert_eq!(pairs, [7.0; 4]);
    assert_eq!(limited.counts().total().unwrap().iterations(), 0);
    let report = plan
        .fit_into(
            &[[0.0, 1.0], [0.5, 2.0], [1.0, 3.0]],
            &mut pairs,
            &mut workspace,
            &mut limited,
        )
        .unwrap();
    assert!(report.normal_residual() < 1e-14);
    assert_eq!(limited.counts().own().iterations(), 0);
    assert_eq!(limited.counts().nested().iterations(), 2);

    // Cancellation stops preparation and fits without publishing anything.
    let token = scioxide_core::CancellationToken::new();
    token.cancel();
    let cancelled = ExecutionContext::sequential().with_cancellation(&token);
    assert!(
        PenalizedSplinePlan::<f64>::build(
            linear_basis(),
            vec![0.0, 0.5, 1.0],
            None,
            SplinePenalty::new(0.25, 1).unwrap(),
            cancelled,
        )
        .is_err()
    );
    let mut coefficients = [5.0; 2];
    let mut single = plan.workspace(1).unwrap();
    assert!(
        plan.fit_into(&[0.0, 0.5, 1.0], &mut coefficients, &mut single, cancelled)
            .is_err()
    );
    assert_eq!(coefficients, [5.0; 2]);
    assert!(single.diagnostics().is_empty());
}

#[test]
fn prepared_smoothing_execution_reuses_workspace() {
    let basis = linear_basis();
    let plan = PenalizedSplinePlan::<f64>::new(
        basis,
        vec![0.0, 0.5, 1.0],
        None,
        SplinePenalty::new(0.25, 1).unwrap(),
    )
    .unwrap();
    let mut workspace = plan.workspace(2).unwrap();
    let values = [[0.0, 1.0], [0.5, 2.0], [1.0, 3.0]];
    let mut first = [0.0; 4];
    let mut second = [0.0; 4];
    let report = plan
        .fit_into(
            &values,
            &mut first,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    let again = plan
        .fit_into(
            &values,
            &mut second,
            &mut workspace,
            ExecutionContext::sequential_native(),
        )
        .unwrap();
    assert_eq!(again, report);
    assert_eq!(first, second);
    assert_eq!(workspace.diagnostics().len(), 2);
    let owned = plan.fit(&values).unwrap();
    assert_eq!(owned.spline().coefficients(), first);
    assert_eq!(*owned.report(), report);
}

#[test]
fn tensor_grid_plan_interpolates_and_normalizes_descending_axes() {
    let basis = linear_basis();
    let plan = TensorSplineInterpolationPlan::<f64, 2>::new(
        [basis.clone(), basis],
        [vec![1.0, 0.0], vec![0.0, 1.0]],
    )
    .unwrap();
    assert_eq!(plan.descending_axes(), [true, false]);
    // Caller order is x=[1,0], y=[0,1], so f=x+2y gives [1,3,0,2].
    let spline = plan.fit(&[1.0, 3.0, 0.0, 2.0]).unwrap();
    assert!((spline.evaluate([0.25, 0.75], REJECT).unwrap() - 1.75).abs() < 1e-14);
}

#[test]
fn quadratic_polynomial_reproduction_and_calculus() {
    let basis = BSplineBasis::new(vec![0.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0], 2).unwrap();
    // Quadratic Bezier controls for x^2.
    let spline = BSpline::new([basis], vec![0.0, 0.0, 1.0]).unwrap();
    assert!((spline.evaluate([0.3], REJECT).unwrap() - 0.09).abs() < 1e-14);
    assert!((spline.derivative([0.3], [1], None, REJECT).unwrap() - 0.6).abs() < 1e-14);
    assert!((spline.integral([0.0], [1.0], REJECT).unwrap() - 1.0 / 3.0).abs() < 1e-14);
}

#[test]
fn nurbs_quarter_circle_projective_fixture() {
    let basis = BSplineBasis::new(vec![0.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0], 2).unwrap();
    let middle = 0.5_f64.sqrt();
    let curve = Nurbs::new(
        [basis],
        vec![[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        vec![1.0, middle, 1.0],
    )
    .unwrap();
    let output = curve.evaluate([0.5], REJECT).unwrap();
    assert!((output[0] - middle).abs() < 2e-14);
    assert!((output[1] - middle).abs() < 2e-14);
    assert!((output[0] * output[0] + output[1] * output[1] - 1.0).abs() < 3e-14);
}

#[test]
fn f32_tensor_and_smoothing_paths() {
    let basis = BSplineBasis::new(vec![0.0_f32, 0.0, 1.0, 1.0], 1).unwrap();
    let spline = BSpline::new([basis.clone()], vec![1.0, 3.0]).unwrap();
    let derivative = spline
        .derivative([0.4], [1], None, ExtrapolationPolicy::Reject)
        .unwrap();
    assert!((derivative - 2.0).abs() < 2e-6);
    let plan = PenalizedSplinePlan::<f32>::new(
        basis,
        vec![0.0, 0.5, 1.0],
        None,
        SplinePenalty::new(0.2, 1).unwrap(),
    )
    .unwrap();
    let fit = plan.fit(&[0.0, 0.5, 1.0]).unwrap();
    assert!(fit.diagnostics()[0].normal_residual() < 2e-6);
}

#[test]
fn invalid_shapes_and_weights_leave_output_unchanged() {
    let spline = BSpline::new([linear_basis()], vec![1.0, 2.0]).unwrap();
    let mut workspace = spline.workspace().unwrap();
    let mut output = [7.0, 11.0];
    assert!(
        spline
            .evaluate_into([0.5], REJECT, &mut output, &mut workspace)
            .is_err()
    );
    assert_eq!(output, [7.0, 11.0]);
    assert!(Nurbs::new([linear_basis()], vec![1.0, 2.0], vec![1.0, 0.0]).is_err());

    let discontinuous_basis = BSplineBasis::new(vec![0.0, 0.0, 0.5, 0.5, 1.0, 1.0], 1).unwrap();
    let discontinuous = BSpline::new([discontinuous_basis], vec![0.0, 1.0, 3.0, 5.0]).unwrap();
    let mut discontinuous_workspace = discontinuous.workspace().unwrap();
    let mut derivative = 13.0;
    assert!(
        discontinuous
            .derivative_into(
                [0.5],
                [1],
                None,
                REJECT,
                &mut derivative,
                &mut discontinuous_workspace,
            )
            .is_err()
    );
    assert_eq!(derivative, 13.0);
}

#[test]
fn tensor_and_nurbs_use_the_shared_point_protocol() {
    let basis = linear_basis();
    let tensor = BSpline::new(
        [basis.clone()],
        vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)],
    )
    .unwrap();
    let mut tensor_workspace = tensor.workspace().unwrap();
    let mut value = Complex::new(7.0, 11.0);
    tensor
        .evaluate_into([0.5], REJECT, &mut value, &mut tensor_workspace)
        .unwrap();
    assert_eq!(value, Complex::new(2.0, 3.0));

    let nurbs = Nurbs::new([basis], vec![0.0, 1.0], vec![1.0, 2.0]).unwrap();
    let mut nurbs_workspace = nurbs.workspace().unwrap();
    let mut gradient = [0.0];
    nurbs
        .gradient_into([0.25], REJECT, &mut gradient, &mut nurbs_workspace)
        .unwrap();
    assert!((gradient[0] - 1.28).abs() < 1e-14);
}

#[cfg(feature = "std")]
#[test]
fn tensor_batch_executor_matches_sequential_bits() {
    use scioxide_interp::{BatchEvaluator, BatchRequest};
    let spline = BSpline::new([linear_basis()], vec![1.0, 3.0]).unwrap();
    let workers = core::num::NonZeroUsize::new(2).unwrap();
    let pool = scioxide_core::ThreadExecutor::new(workers);
    let plan = BatchEvaluator::new(&spline, scioxide_core::ExecutionPolicy::Parallel(workers));
    let query_storage = [0.1, 0.5, 0.9];
    let queries = NdView::new(
        &query_storage,
        StridedLayout::new(&[3, 1], &[1, 1]).unwrap(),
    )
    .unwrap();
    let mut results = [[0.0; 3]; 2];
    for (output, context) in results.iter_mut().zip([
        ExecutionContext::sequential_native(),
        ExecutionContext::new(&pool),
    ]) {
        let mut view =
            NdViewMut::new(output, StridedLayout::new(&[3, 1], &[1, 1]).unwrap()).unwrap();
        let mut workspace = plan.workspace(BatchRequest::values(3)).unwrap();
        plan.evaluate_into(&queries, REJECT, &mut view, None, &mut workspace, context)
            .unwrap();
    }
    assert_eq!(results[0], results[1]);
}

macro_rules! knot_constructors {
    ($module:ident, $s:ty) => {
        mod $module {
            use scioxide_core::{InvalidInputReason, SciOxideError, SciOxideErrorView};
            use scioxide_interp::{
                BSpline, BSplineBasis, CubicSpline, ExtrapolationPolicy, Interpolant,
            };
            type S = $s;

            /// The tolerance `2¹⁰ ε · max|y|` of the interpolation comparisons.
            fn tolerance(values: &[S]) -> S {
                1024.0 * S::EPSILON * values.iter().fold(0.0, |m: S, v| m.max(v.abs()))
            }

            fn reason(error: &SciOxideError) -> (&'static str, InvalidInputReason) {
                match error.view() {
                    SciOxideErrorView::InvalidInput {
                        argument, reason, ..
                    } => (argument.as_str(), reason),
                    _ => panic!("expected InvalidInput, got {error:?}"),
                }
            }

            #[test]
            fn clamped_knots_repeat_the_end_breakpoints() {
                let b: [S; 4] = [0.0, 0.5, 2.0, 3.0];
                let knots = |p| BSplineBasis::clamped(&b, p).unwrap().knots().to_vec();
                assert_eq!(knots(1), [0.0, 0.0, 0.5, 2.0, 3.0, 3.0]);
                assert_eq!(knots(3), [0.0, 0.0, 0.0, 0.0, 0.5, 2.0, 3.0, 3.0, 3.0, 3.0]);
                let quintic = [[0.0; 6].as_slice(), &[0.5, 2.0], &[3.0; 6]].concat();
                assert_eq!(knots(5), quintic);
                for p in [0, 1, 3, 5] {
                    let basis = BSplineBasis::clamped(&b, p).unwrap();
                    assert_eq!(basis.control_count(), 3 + p);
                    assert_eq!((basis.domain().lower(), basis.domain().upper()), (0.0, 3.0));
                }
                let error = |b: &[S]| reason(&BSplineBasis::clamped(b, 3).unwrap_err());
                assert_eq!(error(&[]), ("breakpoints", InvalidInputReason::Empty));
                assert_eq!(
                    error(&[1.0]),
                    ("breakpoints", InvalidInputReason::OutOfRange)
                );
                for bad in [[0.0, 1.0, 1.0], [0.0, 2.0, 1.0]] {
                    assert_eq!(
                        error(&bad),
                        ("breakpoints", InvalidInputReason::Inconsistent)
                    );
                }
                assert!(matches!(
                    BSplineBasis::<S>::clamped(&[0.0, S::NAN], 3)
                        .unwrap_err()
                        .view(),
                    SciOxideErrorView::NonFinite { .. }
                ));
            }

            #[test]
            fn not_a_knot_knots_omit_the_nodes_next_to_each_end() {
                let x: [S; 8] = [0.0, 1.0, 1.5, 2.0, 4.0, 5.0, 7.0, 8.0];
                let knots = |p| BSplineBasis::not_a_knot(&x, p).unwrap().knots().to_vec();
                assert_eq!(knots(1), [0.0, 0.0, 1.0, 1.5, 2.0, 4.0, 5.0, 7.0, 8.0, 8.0]);
                let cubic = [[0.0; 4].as_slice(), &[1.5, 2.0, 4.0, 5.0], &[8.0; 4]].concat();
                assert_eq!(knots(3), cubic);
                let quintic = [[0.0; 6].as_slice(), &[2.0, 4.0], &[8.0; 6]].concat();
                assert_eq!(knots(5), quintic);
                for p in [1, 3, 5] {
                    assert_eq!(BSplineBasis::not_a_knot(&x, p).unwrap().control_count(), 8);
                    let fewest = BSplineBasis::not_a_knot(&x[..p + 1], p).unwrap();
                    assert_eq!(fewest.control_count(), p + 1);
                    assert_eq!(
                        reason(&BSplineBasis::not_a_knot(&x[..p], p).unwrap_err()),
                        ("nodes", InvalidInputReason::OutOfRange)
                    );
                }
                let error = |x: &[S], p| reason(&BSplineBasis::not_a_knot(x, p).unwrap_err());
                // The degree is checked before the nodes.
                for p in [0, 2, 4] {
                    assert_eq!(error(&x, p), ("degree", InvalidInputReason::OutOfRange));
                    assert_eq!(error(&[], p), ("degree", InvalidInputReason::OutOfRange));
                }
                assert_eq!(error(&[], 3), ("nodes", InvalidInputReason::Empty));
                for bad in [[0.0, 1.0, 1.0, 2.0], [0.0, 2.0, 1.0, 3.0]] {
                    assert_eq!(error(&bad, 3), ("nodes", InvalidInputReason::Inconsistent));
                }
            }

            #[test]
            fn degree_one_interpolation_has_the_coefficients_of_the_linear_spline() {
                let x: Vec<S> = vec![0.0, 1.0, 1.5, 2.0, 4.0];
                let y: Vec<S> = vec![1.0, -2.0, 0.5, 3.0, 2.0];
                let tol = tolerance(&y);
                let fitted = BSpline::interpolate([x.clone()], &y, 1).unwrap();
                let linear = BSpline::linear(x, y).unwrap();
                assert_eq!(fitted.bases()[0].knots(), linear.bases()[0].knots());
                for (a, b) in fitted.coefficients().iter().zip(linear.coefficients()) {
                    assert!((a - b).abs() <= tol, "{a} vs {b}");
                }
            }

            #[test]
            fn cubic_interpolation_matches_the_not_a_knot_cubic_spline() {
                let x: Vec<S> = vec![0.0, 0.4, 1.0, 1.7, 2.5, 3.0, 4.2];
                let y: Vec<S> = x.iter().map(|&t| (2.0 * t).sin() + 0.25 * t * t).collect();
                let tol = tolerance(&y);
                let fitted = BSpline::interpolate([x.clone()], &y, 3).unwrap();
                let spline = CubicSpline::new(x.clone(), y).unwrap();
                let midpoints = x.windows(2).map(|w| 0.5 * (w[0] + w[1]));
                for t in x.iter().copied().chain(midpoints) {
                    let a = fitted.evaluate([t], ExtrapolationPolicy::Reject).unwrap();
                    let b = spline.evaluate([t], ExtrapolationPolicy::Reject).unwrap();
                    assert!((a - b).abs() <= tol, "{t}: {a} vs {b}");
                }
            }

            #[test]
            fn bicubic_interpolation_reproduces_a_bicubic_polynomial() {
                let f = |x: S, y: S| {
                    1.0 + x - 2.0 * y + x * x * y + 0.5 * x * x * x - y * y * y
                        + 0.25 * x * x * x * y * y * y
                };
                let x: Vec<S> = vec![0.0, 0.5, 1.25, 2.0, 2.5, 3.0];
                let y: Vec<S> = vec![-1.0, 0.0, 0.75, 1.5, 2.0];
                let values: Vec<S> = x
                    .iter()
                    .flat_map(|&a| y.iter().map(move |&b| f(a, b)))
                    .collect();
                let tol = tolerance(&values);
                let spline = BSpline::interpolate([x, y], &values, 3).unwrap();
                for point in [[0.1, -0.9], [1.0, 0.3], [2.2, 1.9], [3.0, 2.0], [1.6, 1.1]] {
                    let value = spline.evaluate(point, ExtrapolationPolicy::Reject).unwrap();
                    let expected = f(point[0], point[1]);
                    assert!(
                        (value - expected).abs() <= tol,
                        "{point:?}: {value} vs {expected}"
                    );
                }
                assert_eq!(
                    reason(&BSpline::interpolate([vec![0.0, 1.0, 2.0]], &[0.0; 3], 2).unwrap_err()),
                    ("degree", InvalidInputReason::OutOfRange)
                );
                assert!(matches!(
                    BSpline::interpolate([vec![0.0, 1.0], vec![0.0, 1.0]], &[0.0; 3], 1)
                        .unwrap_err()
                        .view(),
                    SciOxideErrorView::DimensionMismatch { .. }
                ));
            }
        }
    };
}

knot_constructors!(single, f32);
knot_constructors!(double, f64);
