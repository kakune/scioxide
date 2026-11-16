//! One-shot forms equal the configured forms they wrap, bit for bit, and
//! report their errors under their own parameter names.
#![allow(clippy::unwrap_used, clippy::panic)]
use scioxide_core::{
    Argument, InvalidInputReason, NumericScalar, RealScalar, SciOxideError, SciOxideErrorKind,
    SciOxideErrorView,
};

mod features;
mod fir;
mod iir;
mod resampling;
mod spectra;

/// Bit patterns of the real and imaginary parts, so `-0.0 != 0.0` and equal
/// NaN payloads compare equal.
fn bits<T: NumericScalar>(values: &[T]) -> Vec<(u64, u64)> {
    values
        .iter()
        .map(|x| {
            (
                x.real_component().into_f64().to_bits(),
                x.imaginary_component().into_f64().to_bits(),
            )
        })
        .collect()
}

/// Error of an expected failure.
#[track_caller]
fn error<T>(result: Result<T, SciOxideError>) -> SciOxideError {
    match result {
        Ok(_) => panic!("expected an error, got Ok"),
        Err(error) => error,
    }
}

#[track_caller]
fn error_kind<T>(result: Result<T, SciOxideError>) -> SciOxideErrorKind {
    error(result).kind()
}

/// Argument, reason and detail of an invalid-input error.
#[track_caller]
fn invalid<T>(result: Result<T, SciOxideError>) -> (Argument, InvalidInputReason, String) {
    match error(result).view() {
        SciOxideErrorView::InvalidInput {
            argument,
            reason,
            detail,
        } => (argument, reason, detail.into()),
        view => panic!("expected InvalidInput, got {view:?}"),
    }
}

/// Argument and reason of an invalid-input error.
#[track_caller]
fn argument_and_reason<T>(result: Result<T, SciOxideError>) -> (Argument, InvalidInputReason) {
    let (argument, reason, _) = invalid(result);
    (argument, reason)
}

/// `a` and `b` print alike, which for floats other than NaN means identical bits.
#[track_caller]
fn identical<T: core::fmt::Debug + ?Sized>(a: &T, b: &T) {
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
}
