//! The shared relative tolerance assertion of the polynomial integration tests.
use scioxide_core::{AlgebraicScalar, NumericScalar, RealScalar};

/// Assert `|actual - expected| <= factor · ε · (1 + |expected|)` in the
/// scalar's own precision; complex values use their modulus.
#[track_caller]
pub(crate) fn assert_close<C: NumericScalar>(actual: C, expected: C, factor: impl Into<f64>) {
    let factor = C::Real::try_from_f64_lossy(factor.into()).unwrap();
    let tolerance = factor * C::Real::EPSILON * (C::Real::ONE + expected.abs());
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual:?} != {expected:?}, tolerance {tolerance:?}"
    );
}
