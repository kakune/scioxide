//! Assertions shared by interpolation unit and integration tests.
#![allow(dead_code, clippy::unwrap_used, clippy::panic)]

use scioxide_core::{RealScalar, SciOxideError, SciOxideErrorKind, Tolerance};

/// Convert fixture input to the target scalar, allowing IEEE rounding.
pub(crate) fn from_f64<S: RealScalar>(value: f64) -> S {
    S::try_from_f64_lossy(value).unwrap()
}

/// Assert finite values satisfy `|actual - expected| <= absolute + relative * |expected|`.
/// Tolerances are explicit; zero relative tolerance selects an absolute bound.
pub(crate) fn assert_close<S: RealScalar, E: RealScalar>(
    actual: S,
    expected: E,
    absolute: f64,
    relative: f64,
) {
    let actual = actual.try_to_f64_exact().unwrap();
    let expected = expected.try_to_f64_exact().unwrap();
    let tolerance = Tolerance::new(absolute, relative).unwrap();
    let bound = tolerance.threshold(expected.abs()).unwrap();
    assert!(
        actual.is_finite() && expected.is_finite() && (actual - expected).abs() <= bound,
        "actual={actual:.17e}, expected={expected:.17e}, absolute={absolute:e}, relative={relative:e}"
    );
}

/// Assert an error classification and return the error for reason/origin checks.
pub(crate) fn assert_error<T>(
    result: &Result<T, SciOxideError>,
    kind: SciOxideErrorKind,
) -> &SciOxideError {
    let Err(error) = result else {
        panic!("expected {kind:?}, got success")
    };
    assert_eq!(error.kind(), kind, "unexpected error: {error:?}");
    error
}
