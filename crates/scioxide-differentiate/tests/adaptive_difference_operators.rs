//! Storage-independent differentiation tests; no materialization in the callback.
#![allow(clippy::unwrap_used)]
use scioxide_core::{LinearOperator, NumericScalar, RealScalar};
use scioxide_differentiate::*;
use scioxide_linalg::Matrix;
use scioxide_sparse::ExplicitZeroPolicy;

fn qualify<S: RealScalar + NumericScalar<Real = S>>() {
    let c = |v| S::try_from_f64_lossy(v).unwrap();
    let dense = Matrix::from_vec(2, 2, vec![c(2.0), c(0.0), c(-1.0), c(3.0)]).unwrap();
    let sparse = dense.to_sparse_csr(ExplicitZeroPolicy::Drop).unwrap();
    let view = dense.view();
    let operators: [&dyn LinearOperator<S>; 3] = [
        &dense.view().try_finite().unwrap(),
        &view.try_finite().unwrap(),
        &sparse,
    ];
    for operator in operators {
        let mut scratch = vec![c(0.0); operator.apply_workspace_len().unwrap()];
        let result = AdaptiveDifference::default()
            .jacobian(
                |x: &[S], y: &mut [S]| operator.apply_into(x, y, &mut scratch),
                &[c(0.5), c(1.0)],
                2,
                None,
            )
            .unwrap();
        assert!(result.report().is_success(), "{:?}", result.report());
        for (&actual, &expected) in result.values().unwrap().iter().zip(dense.as_slice()) {
            assert!((actual - expected).abs() <= S::SQRT_EPSILON * c(8.0));
        }
        let result = AdaptiveDifference::default()
            .with_zero_direction(ZeroDirectionPolicy::MathematicalZero)
            .jvp(
                |x: &[S], y: &mut [S]| operator.apply_into(x, y, &mut scratch),
                &[c(0.5), c(1.0)],
                &[c(1.0), c(-2.0)],
                2,
                None,
            )
            .unwrap();
        assert!(result.report().is_success(), "{:?}", result.report());
        for (&actual, expected) in result.values().unwrap().iter().zip([c(2.0), c(-7.0)]) {
            assert!((actual - expected).abs() <= S::SQRT_EPSILON * c(8.0));
        }
    }
}

#[test]
fn dense_view_sparse_operator_f32() {
    qualify::<f32>();
}
#[test]
fn dense_view_sparse_operator_f64() {
    qualify::<f64>();
}
