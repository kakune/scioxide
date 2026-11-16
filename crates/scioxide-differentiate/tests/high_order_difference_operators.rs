//! Borrowed dense, sparse and operator callbacks drive node-difference plans and complex steps.
#![allow(clippy::unwrap_used)]
use scioxide_core::{Complex, LinearOperator, NumericScalar, RealScalar};
use scioxide_differentiate::*;
use scioxide_linalg::Matrix;
use scioxide_sparse::ExplicitZeroPolicy;
fn qualify<S: RealScalar + NumericScalar<Real = S>>()
where
    Complex<S>: NumericScalar<Real = S>,
{
    let c = |v| S::try_from_f64_lossy(v).unwrap();
    let dense = Matrix::from_vec(2, 2, vec![c(2.0), c(0.0), c(-1.0), c(3.0)]).unwrap();
    let sparse = dense.to_sparse_csr(ExplicitZeroPolicy::Drop).unwrap();
    let view = dense.view();
    let operators: [&dyn LinearOperator<S>; 3] = [
        &dense.view().try_finite().unwrap(),
        &view.try_finite().unwrap(),
        &sparse,
    ];
    let nodes = [c(0.0), c(0.2), c(0.5), c(0.8), c(1.0)];
    let plan = PhysicalDifferencePlan::new(
        DerivativeOperation::Jacobian,
        &[c(0.5), c(0.5)],
        &[&nodes, &nodes],
        2,
        scioxide_differentiate::NodeDifferenceOptions::default(),
    )
    .unwrap();
    for operator in operators {
        let mut scratch = vec![S::ZERO; operator.apply_workspace_len().unwrap()];
        let result = plan
            .evaluate(
                |x: &[S], y: &mut [_]| operator.apply_into(x, y, &mut scratch),
                None,
            )
            .unwrap();
        for (actual, expected) in result.values().unwrap().iter().zip(dense.as_slice()) {
            assert!((*actual - *expected).abs() <= c(32.0) * S::EPSILON);
        }
    }
    let dense = Matrix::from_vec(
        2,
        2,
        dense
            .as_slice()
            .iter()
            .map(|&v| Complex::new(v, S::ZERO))
            .collect(),
    )
    .unwrap();
    let sparse = dense.to_sparse_csr(ExplicitZeroPolicy::Drop).unwrap();
    let view = dense.view();
    let operators: [&dyn LinearOperator<Complex<S>>; 3] = [
        &dense.view().try_finite().unwrap(),
        &view.try_finite().unwrap(),
        &sparse,
    ];
    let method = ComplexStep::default().with_step(S::EPSILON);
    for operator in operators {
        let mut scratch =
            vec![Complex::new(S::ZERO, S::ZERO); operator.apply_workspace_len().unwrap()];
        let result = method
            .jacobian(
                |x: &[Complex<S>], y: &mut [Complex<S>]| operator.apply_into(x, y, &mut scratch),
                &[c(0.5), c(0.5)],
                2,
            )
            .unwrap();
        for (actual, expected) in result.values().unwrap().iter().zip(dense.as_slice()) {
            assert!((*actual - expected.real_component()).abs() <= c(8.0) * S::EPSILON);
        }
    }
}
#[test]
fn dense_sparse_view_operators_f32() {
    qualify::<f32>();
}
#[test]
fn dense_sparse_view_operators_f64() {
    qualify::<f64>();
}
