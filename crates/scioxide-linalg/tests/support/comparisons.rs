//! Scalar comparisons with explicit, caller-selected error allowances.
use scioxide_core::NumericScalar;
pub fn within<S: NumericScalar>(actual: S, expected: S, allowance: S::Real) {
    within_labeled(actual, expected, allowance, "comparison");
}
pub fn within_labeled<S: NumericScalar>(actual: S, expected: S, allowance: S::Real, label: &str) {
    assert!(
        (actual - expected).abs() <= allowance,
        "{label}: {actual:?} != {expected:?}; allowance {allowance:?}"
    );
}

pub fn matrix_error<S: NumericScalar>(
    a: &scioxide_linalg::Matrix<S>,
    b: &scioxide_linalg::Matrix<S>,
) -> f64 {
    use scioxide_core::RealScalar;
    assert_eq!((a.rows(), a.cols()), (b.rows(), b.cols()));
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .fold(0.0_f64, |sum, (&a, &b)| sum.hypot((a - b).abs().into_f64()))
}
pub fn matrix_within<S: NumericScalar>(
    a: &scioxide_linalg::Matrix<S>,
    b: &scioxide_linalg::Matrix<S>,
    tolerance: f64,
) {
    let error = matrix_error(a, b);
    assert!(error <= tolerance, "error {error} > {tolerance}");
}
pub fn slices_within(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(x, y)| (x - y).abs() < 1e-9)
    );
}
