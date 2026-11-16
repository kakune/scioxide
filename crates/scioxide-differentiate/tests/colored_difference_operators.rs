//! Sparse-owned coloring feeds core numerical recovery across storage forms.
#![allow(clippy::unwrap_used)]
use scioxide_core::{
    Complex, ExecutionContext, LinearOperator, NumericScalar, ProblemToken, RealScalar,
    SolveBudgets, SolveIdentity, SolveState,
};
use scioxide_differentiate::*;
use scioxide_linalg::Matrix;
use scioxide_sparse::{ExplicitZeroPolicy, derivative_coloring};
fn qualify<T: NumericScalar>() {
    let c = |v| T::Real::try_from_f64_lossy(v).unwrap();
    let value = |v| T::from_components(c(v), c(v * 0.25));
    let dense = Matrix::from_vec(
        3,
        3,
        vec![
            value(2.0),
            value(0.0),
            value(0.0),
            value(-1.0),
            value(3.0),
            value(0.0),
            value(0.0),
            value(0.0),
            value(4.0),
        ],
    )
    .unwrap();
    let sparse = dense.to_sparse_csr(ExplicitZeroPolicy::Drop).unwrap();
    let schedule = derivative_coloring(sparse.structural_pattern()).unwrap();
    assert_eq!(schedule.color_count(), 2);
    let plan = ColoredJacobianPlan::new(
        &schedule,
        AdaptiveDifference::default().with_execution(scioxide_core::ExecutionPolicy::Auto),
    )
    .unwrap();
    let view = dense.view();
    for operator in [
        &dense.view().try_finite().unwrap() as &(dyn LinearOperator<T> + Sync),
        &view.try_finite().unwrap(),
        &sparse,
    ] {
        let scratch_len = operator.apply_workspace_len().unwrap();
        assert!(scratch_len <= 3);
        let function = |x: &[T::Real], y: &mut [T]| {
            let x = [T::from_real(x[0]), T::from_real(x[1]), T::from_real(x[2])];
            operator.apply_into(&x, y, &mut [T::ZERO; 3][..scratch_len])
        };
        let result = plan
            .evaluate(function, &[c(0.5), c(1.0), c(2.0)], None)
            .unwrap();
        assert!(result.report().is_success());
        for (&actual, &expected) in result
            .values()
            .as_ref()
            .unwrap()
            .iter()
            .zip(sparse.values())
        {
            assert!((actual - expected).abs() < c(0.001));
        }
        let mut w = plan.workspace(()).unwrap();
        let mut output = vec![T::ZERO; plan.output_len()];
        let mut errors = vec![c(0.0); plan.output_len()];
        let mut state = SolveState::new();
        let mut budgets = SolveBudgets::unbounded();
        let identity = SolveIdentity::new(
            ProblemToken::new(470),
            scioxide_core::algorithm_id!("test", "operators", 1),
            0,
            0,
        );
        let mut ctx = state
            .begin_solve(ExecutionContext::sequential(), identity, &mut budgets)
            .unwrap();
        let report = plan
            .evaluate_into(
                scioxide_differentiate::Parallel(function),
                &[c(0.5), c(1.0), c(2.0)],
                None,
                &mut output,
                &mut errors,
                &mut w,
                &mut ctx,
            )
            .unwrap();
        assert!(report.is_success());
        assert_eq!(output, result.values().unwrap());
    }
}
#[test]
fn real32() {
    qualify::<f32>();
}
#[test]
fn real64() {
    qualify::<f64>();
}
#[test]
fn complex32() {
    qualify::<Complex<f32>>();
}
#[test]
fn complex64() {
    qualify::<Complex<f64>>();
}
#[test]
fn one_call_coloring_uses_sparse_owned_provider_in_both_precisions() {
    use scioxide_core::StructuralPatternView;
    use scioxide_sparse::derivative_coloring;
    let pattern = StructuralPatternView::new(2, 2, &[0, 1, 3], &[0, 0, 1]).unwrap();
    let result = (derivative_coloring)(pattern)
        .and_then(|schedule| {
            schedule.validate_pattern(pattern)?;
            scioxide_differentiate::ColoredJacobianPlan::new(
                &schedule,
                scioxide_differentiate::AdaptiveDifference::default(),
            )
            .and_then(|plan| {
                plan.evaluate(
                    |x: &[f64], y: &mut [_]| {
                        y[0] = x[0] * x[0];
                        y[1] = x[0] + x[1] * x[1];
                        Ok(())
                    },
                    &[1.0_f64, 2.0],
                    None,
                )
            })
        })
        .unwrap();
    let dense = scioxide_differentiate::AdaptiveDifference::default()
        .jacobian(
            |x: &[f64], y: &mut [_]| {
                y[0] = x[0] * x[0];
                y[1] = x[0] + x[1] * x[1];
                Ok(())
            },
            &[1.0_f64, 2.0],
            2,
            None,
        )
        .unwrap();
    assert!(result.report().is_success());
    for (&value, expected) in result.values().unwrap().iter().zip([2.0, 1.0, 4.0]) {
        assert!((value - expected).abs() <= f64::SQRT_EPSILON);
    }
    assert!(dense.report().is_success());
    for (&value, expected) in dense.values().unwrap().iter().zip([2.0, 0.0, 1.0, 4.0]) {
        assert!((value - expected).abs() <= f64::SQRT_EPSILON);
    }
    let result = (derivative_coloring)(pattern)
        .and_then(|schedule| {
            schedule.validate_pattern(pattern)?;
            scioxide_differentiate::ColoredJacobianPlan::new(
                &schedule,
                scioxide_differentiate::AdaptiveDifference::default(),
            )
            .and_then(|plan| {
                plan.evaluate(
                    |x: &[f32], y: &mut [_]| {
                        y[0] = x[0] * x[0];
                        y[1] = x[0] + x[1] * x[1];
                        Ok(())
                    },
                    &[1.0_f32, 2.0],
                    None,
                )
            })
        })
        .unwrap();
    assert!(result.report().is_success());
    for (&value, expected) in result.values().unwrap().iter().zip([2.0, 1.0, 4.0]) {
        assert!((value - expected).abs() <= f32::SQRT_EPSILON);
    }
}
