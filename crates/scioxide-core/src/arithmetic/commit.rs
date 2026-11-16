//! Final range checks and all-or-nothing copies of computed results.

use super::overflow;
use crate::{Dimensions, RealScalar, SciOxideError};

/// Returns `candidate` unchanged if it is finite, including subnormals and `-0`.
///
/// Use it on a result computed from operands that were already checked to be
/// finite: an infinite or NaN result is then an overflow, not an input error.
///
/// # Errors
///
/// [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
/// ([`Overflow`](crate::NumericalIssue::Overflow)) with `detail` if `candidate`
/// is not finite.
///
/// # Examples
///
/// ```
/// use scioxide_core::backend::checked_final_rounding;
///
/// let value = checked_final_rounding(-0.0_f64, "overflow")?;
/// assert_eq!(value.to_bits(), (-0.0_f64).to_bits());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn checked_final_rounding<S: RealScalar>(
    candidate: S,
    detail: &'static str,
) -> Result<S, SciOxideError> {
    if candidate.is_finite() {
        Ok(candidate)
    } else {
        Err(overflow(detail))
    }
}

/// Copies `candidate` into `output` after `validate` accepts it.
///
/// The lengths are checked first, then `validate` runs on `candidate`, and only
/// then is `output` written. On error or panic `output` is unchanged.
///
/// # Errors
///
/// - [`DimensionMismatch`](crate::SciOxideErrorKind::DimensionMismatch) if the
///   slices differ in length.
/// - Any error returned by `validate`.
///
/// # Examples
///
/// ```
/// use scioxide_core::backend::validate_then_commit;
///
/// let mut output = [0.0_f64; 2];
/// validate_then_commit(&[1.0, 2.0], &mut output, |_| Ok(()))?;
/// assert_eq!(output, [1.0, 2.0]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn validate_then_commit<T: Copy>(
    candidate: &[T],
    output: &mut [T],
    validate: impl FnOnce(&[T]) -> Result<(), SciOxideError>,
) -> Result<(), SciOxideError> {
    if output.len() != candidate.len() {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::one(candidate.len()),
            Dimensions::one(output.len()),
        ));
    }
    validate(candidate)?;
    output.copy_from_slice(candidate);
    Ok(())
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn failed_validation_does_not_commit() {
        let mut output = [9_u8, 9];
        let result = validate_then_commit(&[1_u8, 2], &mut output, |_| {
            Err(SciOxideError::size_overflow(crate::quantity!("candidate")))
        });
        assert!(matches!(
            (result).as_ref().unwrap_err().view(),
            crate::SciOxideErrorView::SizeOverflow { .. }
        ));
        assert_eq!(output, [9, 9]);
    }
}
