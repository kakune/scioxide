//! Shared FFT-local dimension, input and arithmetic errors.
use scioxide_core::{Complex, NonFiniteOrigin, NumericalIssue, RealScalar, SciOxideError};
/// Validate that all values in a complex-valued slice are finite.
///
/// # Errors
///
/// - `scioxide_core::SciOxideErrorKind::NonFinite` — if any real/imag component is NaN or infinite.
pub(crate) fn check_complex_slice_finite<S: scioxide_core::RealScalar>(
    input: &[Complex<S>],
    origin: NonFiniteOrigin,
) -> Result<(), SciOxideError> {
    if input.iter().all(|x| x.re.is_finite() && x.im.is_finite()) {
        Ok(())
    } else {
        Err(SciOxideError::non_finite(
            origin,
            "transform values contain a non-finite component",
        ))
    }
}

pub(crate) fn nonfinite() -> SciOxideError {
    SciOxideError::non_finite(
        NonFiniteOrigin::Input,
        "transform parameters and samples must be finite",
    )
}

pub(crate) fn numerical(issue: NumericalIssue, detail: &'static str) -> SciOxideError {
    SciOxideError::numerical_instability(issue, detail)
}

/// Construct the error for arithmetic overflow from finite values.
pub(crate) fn arithmetic_overflow(detail: &'static str) -> SciOxideError {
    numerical(NumericalIssue::Overflow, detail)
}

/// Project an arithmetic result, reporting range failures as numerical errors.
/// Ordinary rounding, including signed underflow to zero, is permitted.
pub(crate) fn project<S: RealScalar>(value: f64) -> Result<S, SciOxideError> {
    if value.is_nan() {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Intermediate,
            "transform arithmetic produced NaN",
        ));
    }
    if value.is_infinite() {
        return Err(arithmetic_overflow("transform result exceeds scalar range"));
    }
    S::try_from_f64_lossy(value).map_err(|error| match error.view() {
        scioxide_core::SciOxideErrorView::ScalarConversion {
            reason: scioxide_core::ScalarConversionReason::OutOfRange,
            ..
        } => arithmetic_overflow("transform result exceeds scalar range"),
        _ => error,
    })
}

pub(crate) fn invalid(
    reason: scioxide_core::InvalidInputReason,
    argument: scioxide_core::Argument,
    detail: &'static str,
) -> SciOxideError {
    SciOxideError::invalid_input(argument, reason, detail)
}

pub(crate) fn length(actual: usize, expected: usize) -> Result<(), SciOxideError> {
    if actual == expected {
        Ok(())
    } else {
        Err(SciOxideError::dimension_mismatch(
            scioxide_core::Dimensions::one(expected),
            scioxide_core::Dimensions::one(actual),
        ))
    }
}

pub(crate) fn sufficient(actual: usize, required: usize) -> Result<(), SciOxideError> {
    if actual < required {
        length(actual, required)
    } else {
        Ok(())
    }
}

/// Accept a nonempty transform before any division or length-derived allocation.
pub(crate) fn nonzero(n: usize, argument: scioxide_core::Argument) -> Result<(), SciOxideError> {
    if n == 0 {
        Err(invalid(
            scioxide_core::InvalidInputReason::Empty,
            argument,
            "transform length must be >= 1",
        ))
    } else {
        Ok(())
    }
}

/// Numeric input check, enforced in release builds, during an existing input pass.
/// The comparisons reject NaN and infinities as well as excess magnitude.
#[inline(always)]
pub(crate) fn bounded<S: RealScalar>(value: Complex<S>) -> Complex<S> {
    assert!(
        (value.re.abs() <= S::ONE) & (value.im.abs() <= S::ONE),
        "bounded FFT input must have finite components with magnitude at most one"
    );
    value
}

/// Finite real-transform inputs produced an unrepresentable output.
pub(crate) fn real_transform_overflow() -> SciOxideError {
    arithmetic_overflow("finite real-transform operands produced a non-finite output")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_complex_slice_finite_non_finite() {
        let error = check_complex_slice_finite(
            &[Complex::new(1.0_f64, 0.0), Complex::new(f64::INFINITY, 0.0)],
            NonFiniteOrigin::Input,
        )
        .unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::NonFinite);
    }

    #[test]
    fn arithmetic_projection_preserves_range_and_rounding_contracts() {
        for error in [
            project::<f64>(f64::INFINITY).unwrap_err(),
            project::<f32>(f64::MAX).unwrap_err(),
        ] {
            assert!(matches!(
                error.view(),
                scioxide_core::SciOxideErrorView::NumericalInstability {
                    issue: NumericalIssue::Overflow,
                    ..
                }
            ));
        }
        assert!(matches!(
            project::<f64>(f64::NAN).unwrap_err().view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: NonFiniteOrigin::Intermediate,
                ..
            }
        ));
        assert_eq!(project::<f32>(f32::MAX as f64).unwrap(), f32::MAX);
        assert_eq!(
            project::<f32>(-1e-300).unwrap().to_bits(),
            (-0.0_f32).to_bits()
        );
        assert_eq!(
            project::<f64>(-0.0).unwrap().to_bits(),
            (-0.0_f64).to_bits()
        );
    }
}
