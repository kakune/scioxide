//! Extended-range arithmetic for decaying radial functions.

use scioxide_core::{NonFiniteOrigin, NumericalIssue, RealScalar, Scaled, SciOxideError};

/// Rounds an extended-range value, reporting overflow as a non-finite
/// intermediate.
pub(crate) fn project<S: RealScalar>(value: Scaled<S>) -> Result<S, SciOxideError> {
    value.round().map_err(|error| {
        if matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Overflow,
                ..
            }
        ) {
            SciOxideError::non_finite(
                NonFiniteOrigin::Intermediate,
                "radial function exceeds the finite scalar range",
            )
        } else {
            error
        }
    })
}

/// Computes `prefactor · exp(−r²)` (or `exp(−r²/2)`), carrying the rounding
/// residual of `r²` into the exponential. Near the smallest subnormal, rounding
/// the exponent first could change the result.
pub(crate) fn squared_decay<S: RealScalar>(
    prefactor: Scaled<S>,
    radius: Scaled<S>,
    radius_correction: Scaled<S>,
    half: bool,
) -> Result<S, SciOxideError> {
    let (square, residual) =
        scioxide_core::backend::two_product(radius.significand(), radius.significand());
    let shift = 2 * radius.exponent() - i64::from(half);
    corrected_decay(
        prefactor,
        Scaled::from_parts(square, shift)?,
        Scaled::from_parts(residual, shift)?
            .add(
                radius
                    .mul(radius_correction)?
                    .shifted(1 - i64::from(half))?,
            )?
            .add(radius_correction.square()?.shifted(-i64::from(half))?)?,
    )
}

/// Returns the product of two extended-range values and its exact rounding
/// residual. The inputs come from finite floats, so their exponent sum fits in
/// `i64`.
pub(crate) fn split_product<S: RealScalar>(
    a: Scaled<S>,
    b: Scaled<S>,
) -> Result<(Scaled<S>, Scaled<S>), SciOxideError> {
    let (product, residual) = scioxide_core::backend::two_product(a.significand(), b.significand());
    let exponent = a.exponent() + b.exponent();
    Ok((
        Scaled::from_parts(product, exponent)?,
        Scaled::from_parts(residual, exponent)?,
    ))
}

/// Rounds `prefactor · exp(−(exponent + correction))` after the multiplication.
/// The prefactor and the total exponent are nonnegative. Exponents beyond a
/// cutoff return zero without evaluating the exponential.
pub(crate) fn corrected_decay<S: RealScalar>(
    prefactor: Scaled<S>,
    exponent: Scaled<S>,
    correction: Scaled<S>,
) -> Result<S, SciOxideError> {
    if prefactor.is_zero() {
        return Ok(S::ZERO);
    }
    let minimum =
        Scaled::from_value(S::MIN_POSITIVE)?.exponent() - i64::from(S::MANTISSA_DIGITS - 1);
    // prefactor < 2^(k+1); an extra four binary orders leave room for
    // rounding the cutoff, well beyond the half-minimum-subnormal boundary.
    let cutoff = S::try_from_f64_lossy(
        (prefactor.exponent() - minimum + 4) as f64 * core::f64::consts::LN_2,
    )?;
    if exponent.cmp_value(Scaled::from_value(cutoff)?).is_gt() {
        return Ok(S::ZERO);
    }
    let tail = Scaled::from_log(S::ONE, -exponent.round()?)?;
    let correction = Scaled::from_log(S::ONE, -correction.round()?)?;
    project(prefactor.mul(tail)?.mul(correction)?)
}
