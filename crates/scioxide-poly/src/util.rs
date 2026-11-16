//! Internal utility helpers shared across all `scioxide-poly` modules.
//!
//! All items are `pub(crate)` — they are not part of the public API.

use alloc::vec::Vec;

use scioxide_core::{AlgebraicScalar, NumericScalar, RealScalar, ScaledValue, SciOxideError};

/// Rounding model `gamma_k = k u / (1 - k u)` for `k` rounding operations
/// (or coordinates), not an algorithm-specific error certificate.
///
/// A count that is not exactly representable in `R` exceeds `2^p`, so
/// `k u >= 2`; it is classified with every other `k u >= 1` count.
///
/// # Errors
///
/// Returns `NumericalInstability` (`IllConditioned`) when `k u >= 1`.
pub(crate) fn rounding_error<R: RealScalar>(count: usize) -> Result<R, SciOxideError> {
    R::try_from_usize_exact(count.max(1))
        .ok()
        .map(|count| count * R::EPSILON)
        .filter(|&product| product < R::ONE)
        .map(|product| product / (R::ONE - product))
        .ok_or_else(|| {
            SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::IllConditioned,
                "rounding model requires operation_count * epsilon < 1",
            )
        })
}

/// Collect fallible values into a fallibly reserved vector.
///
/// Allocation failure returns a structured error naming the
/// collected `quantity`; the first item error is returned unchanged.
pub(crate) fn try_collect<T, I>(
    values: I,
    quantity: scioxide_core::Quantity,
) -> Result<Vec<T>, SciOxideError>
where
    I: IntoIterator<Item = Result<T, SciOxideError>>,
{
    let iterator = values.into_iter();
    let (minimum, maximum) = iterator.size_hint();
    // Reserve fallibly up front from the size hint so allocation failure surfaces as a
    // `SciOxideError` instead of an abort; exact-size iterators then never reallocate
    // during the infallible `push` below.
    let capacity = maximum.unwrap_or(minimum);
    let mut collected = scioxide_core::backend::try_with_capacity(capacity, quantity)?;
    for value in iterator {
        collected.push(value?);
    }
    Ok(collected)
}

/// Grow `values` to at least `len` elements filled with `value`, keeping any
/// longer existing storage. Allocation failure keeps the existing elements.
pub(crate) fn try_grow<T: Clone>(
    values: &mut Vec<T>,
    len: usize,
    value: T,
    quantity: scioxide_core::Quantity,
) -> Result<(), SciOxideError> {
    if values.len() < len {
        scioxide_core::backend::try_reserve(values, len - values.len(), quantity)?;
        values.resize(len, value);
    }
    Ok(())
}

/// Reject borrowed or stored buffers shorter than their required length.
pub(crate) fn minimum_len(required: usize, actual: usize) -> Result<(), SciOxideError> {
    if actual < required {
        Err(SciOxideError::dimension_mismatch(required, actual))
    } else {
        Ok(())
    }
}

/// Reject an output whose length differs from its input.
pub(crate) fn exact_len(expected: usize, actual: usize) -> Result<(), SciOxideError> {
    if actual == expected {
        Ok(())
    } else {
        Err(SciOxideError::dimension_mismatch(expected, actual))
    }
}

/// Payload bytes of `len` values of `T`.
pub(crate) fn bytes_of<T>(len: usize) -> Result<usize, SciOxideError> {
    scioxide_core::backend::checked_mul(
        len,
        core::mem::size_of::<T>(),
        scioxide_core::quantity!("workspace_bytes"),
    )
}

/// Finiteness validators, one per error contract of the workspace
/// classification: caller input, callback results and returned output are
/// `NonFinite` with that origin; a value this crate's own arithmetic produced
/// from finite operands is numerical instability (`Breakdown` for NaN,
/// `Overflow` for infinity); a non-finite value merely observed inside an
/// algorithm is `NonFinite(Intermediate)`. Complex values are checked by
/// component.
fn finite_or<C: NumericScalar>(
    value: C,
    error: impl FnOnce() -> SciOxideError,
) -> Result<C, SciOxideError> {
    if value.has_finite_components() {
        Ok(value)
    } else {
        Err(error())
    }
}

pub(crate) fn finite_input<C: NumericScalar>(
    value: C,
    detail: &'static str,
) -> Result<C, SciOxideError> {
    finite_or(value, || {
        SciOxideError::non_finite(scioxide_core::NonFiniteOrigin::Input, detail)
    })
}

pub(crate) fn finite_callback<C: NumericScalar>(
    value: C,
    detail: &'static str,
) -> Result<C, SciOxideError> {
    finite_or(value, || {
        SciOxideError::non_finite(scioxide_core::NonFiniteOrigin::Callback, detail)
    })
}

pub(crate) fn finite_arithmetic<C: NumericScalar>(
    value: C,
    detail: &'static str,
) -> Result<C, SciOxideError> {
    finite_or(value, || {
        SciOxideError::numerical_instability(
            if value.real_component().is_nan() || value.imaginary_component().is_nan() {
                scioxide_core::NumericalIssue::Breakdown
            } else {
                scioxide_core::NumericalIssue::Overflow
            },
            detail,
        )
    })
}

pub(crate) fn finite_intermediate<C: NumericScalar>(
    value: C,
    detail: &'static str,
) -> Result<C, SciOxideError> {
    finite_or(value, || {
        SciOxideError::non_finite(scioxide_core::NonFiniteOrigin::Intermediate, detail)
    })
}

pub(crate) fn finite_output<C: NumericScalar>(
    value: C,
    detail: &'static str,
) -> Result<C, SciOxideError> {
    finite_or(value, || {
        SciOxideError::non_finite(scioxide_core::NonFiniteOrigin::Output, detail)
    })
}

pub(crate) fn scaled_complex<C: NumericScalar>(
    value: C,
) -> Result<ScaledValue<scioxide_core::Complex<C::Real>>, SciOxideError> {
    ScaledValue::from_value(scioxide_core::Complex::new(
        value.real_component(),
        value.imaginary_component(),
    ))
}

pub(crate) fn round_complex<C: NumericScalar>(
    value: ScaledValue<scioxide_core::Complex<C::Real>>,
) -> Result<C, SciOxideError> {
    let value = value.round()?;
    Ok(C::from_components(value.re, value.im))
}

pub(crate) fn coefficient_scale<C: NumericScalar>(coefficients: &[C]) -> C::Real {
    coefficients
        .iter()
        .fold(C::Real::ZERO, |scale, coefficient| {
            scale.max(coefficient.abs())
        })
}

/// Maximum binary exponent of nonzero coefficient components; zero vectors use zero.
pub(crate) fn coefficient_exponent<C: NumericScalar>(
    coefficients: &[C],
) -> Result<i64, SciOxideError> {
    validate_slice(coefficients)?;
    finite_coefficient_exponent(coefficients)
}

/// Scaling metadata for coefficients accepted by a finite-input validator.
pub(crate) fn finite_coefficient_exponent<C: NumericScalar>(
    coefficients: &[C],
) -> Result<i64, SciOxideError> {
    coefficients
        .iter()
        .try_fold(None, |largest: Option<i64>, &coefficient| {
            if coefficient == C::ZERO {
                return Ok(largest);
            }
            // Core's normalized exponent is floor(log2(|value|)).
            let exponent = scioxide_core::Scaled::from_value(
                coefficient
                    .real_component()
                    .abs()
                    .max(coefficient.imaginary_component().abs()),
            )?
            .exponent();
            Ok(Some(largest.map_or(exponent, |value| value.max(exponent))))
        })
        .map(|exponent| exponent.unwrap_or(0))
}

pub(crate) fn compare_complex<C: NumericScalar>(left: &C, right: &C) -> core::cmp::Ordering {
    left.real_component()
        .partial_cmp(&right.real_component())
        .unwrap_or(core::cmp::Ordering::Equal)
        .then_with(|| {
            left.imaginary_component()
                .partial_cmp(&right.imaginary_component())
                .unwrap_or(core::cmp::Ordering::Equal)
        })
}

pub(crate) fn invalid(
    argument: scioxide_core::Argument,
    reason: scioxide_core::InvalidInputReason,
    detail: &'static str,
) -> SciOxideError {
    SciOxideError::invalid_input(argument, reason, detail)
}

/// Operands of one operation must carry the same immutable policy. A mismatch is
/// `Inconsistent`, attributed to the caller parameter `argument` that supplied
/// the second policy, at every entry point.
pub(crate) fn same_policy<P: PartialEq>(
    policy: P,
    other: P,
    argument: scioxide_core::Argument,
) -> Result<(), SciOxideError> {
    if policy == other {
        Ok(())
    } else {
        Err(SciOxideError::invalid_input(
            argument,
            scioxide_core::InvalidInputReason::Inconsistent,
            "operands must carry the same immutable policy",
        ))
    }
}

pub(crate) fn empty_error(argument: scioxide_core::Argument) -> SciOxideError {
    SciOxideError::invalid_input(
        argument,
        scioxide_core::InvalidInputReason::Empty,
        "at least one coefficient is required",
    )
}

/// Validate monomial input once and borrow its prefix without exact trailing
/// zeros (at least one coefficient).
/// Empty input is distinct from the legitimate zero polynomial.
pub(crate) fn canonical_coefficients<C: NumericScalar>(
    values: &[C],
    argument: scioxide_core::Argument,
) -> Result<&[C], SciOxideError> {
    if values.is_empty() {
        return Err(empty_error(argument));
    }
    validate_slice(values)?;
    let len = values
        .iter()
        .rposition(|v| *v != C::ZERO)
        .map_or(1, |i| i + 1);
    Ok(&values[..len])
}

/// Structural rules shared by exact and approximate partial-fraction plans.
/// Domain-specific policy, product and coprimality checks remain with each plan.
pub(crate) fn validate_factors(
    denominator_degree: usize,
    denominator_monic: bool,
    mut factors: impl Iterator<Item = (usize, bool, usize)>,
) -> Result<(), SciOxideError> {
    if !denominator_monic {
        return Err(invalid(
            scioxide_core::argument!("denominator"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "denominator must be nonzero and monic",
        ));
    }
    let degree = factors.try_fold(0usize, |sum, (degree, monic, multiplicity)| {
        if degree == 0 || !monic || multiplicity == 0 {
            return Err(invalid(
                scioxide_core::argument!("factors"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "factors must be nonconstant, monic, and have positive multiplicities",
            ));
        }
        degree
            .checked_mul(multiplicity)
            .and_then(|n| sum.checked_add(n))
            .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("factor_degrees")))
    })?;
    if degree != denominator_degree {
        return Err(invalid(
            scioxide_core::argument!("factors"),
            scioxide_core::InvalidInputReason::Inconsistent,
            "factor degrees must sum to denominator degree",
        ));
    }
    Ok(())
}

/// [`finite_input`] over a caller slice; does not allocate, even on failure.
pub(crate) fn validate_slice<C: NumericScalar>(values: &[C]) -> Result<(), SciOxideError> {
    values
        .iter()
        .try_for_each(|&value| finite_input(value, "entries must have finite components").map(drop))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finiteness_validators_classify_by_contract() {
        use scioxide_core::{Complex, NonFiniteOrigin, NumericalIssue, SciOxideErrorView};
        let issue = |error: SciOxideError| match error.view() {
            SciOxideErrorView::NumericalInstability { issue, .. } => Some(issue),
            _ => None,
        };
        for value in [f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                issue(finite_arithmetic(value, "x").unwrap_err()),
                Some(NumericalIssue::Overflow)
            );
            let complex = Complex::new(1.0, value);
            assert_eq!(
                issue(finite_arithmetic(complex, "x").unwrap_err()),
                Some(NumericalIssue::Overflow)
            );
        }
        for value in [f64::NAN, -f64::NAN] {
            assert_eq!(
                issue(finite_arithmetic(value, "x").unwrap_err()),
                Some(NumericalIssue::Breakdown)
            );
            let complex = Complex::new(f64::INFINITY, value);
            assert_eq!(
                issue(finite_arithmetic(complex, "x").unwrap_err()),
                Some(NumericalIssue::Breakdown)
            );
        }
        assert_eq!(
            finite_arithmetic(Complex::new(1.0, -2.0), "x").unwrap(),
            Complex::new(1.0, -2.0)
        );
        for (error, expected) in [
            (
                finite_input(f64::NAN, "x").unwrap_err(),
                NonFiniteOrigin::Input,
            ),
            (
                finite_callback(f64::INFINITY, "x").unwrap_err(),
                NonFiniteOrigin::Callback,
            ),
            (
                finite_intermediate(f64::INFINITY, "x").unwrap_err(),
                NonFiniteOrigin::Intermediate,
            ),
            (
                finite_output(f64::NAN, "x").unwrap_err(),
                NonFiniteOrigin::Output,
            ),
            (
                validate_slice(&[1.0, f64::NAN]).unwrap_err(),
                NonFiniteOrigin::Input,
            ),
        ] {
            assert!(
                matches!(error.view(), SciOxideErrorView::NonFinite { origin, .. } if origin == expected)
            );
        }
    }

    #[test]
    fn coefficient_exponents_cover_zero_subnormals_and_range_boundaries() {
        for (value, exponent) in [
            (f64::from_bits(1), -1074),
            (f64::MIN_POSITIVE, -1022),
            (0.5, -1),
            (-8.0, 3),
            (f64::MAX, 1023),
        ] {
            assert_eq!(coefficient_exponent(&[value]).unwrap(), exponent);
        }
        assert_eq!(coefficient_exponent(&[f32::from_bits(1)]).unwrap(), -149);
        assert_eq!(coefficient_exponent(&[f32::MAX]).unwrap(), 127);
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                coefficient_exponent(&[value]).unwrap_err().view(),
                scioxide_core::SciOxideErrorView::NonFinite {
                    origin: scioxide_core::NonFiniteOrigin::Input,
                    ..
                }
            ));
        }
        assert_eq!(coefficient_exponent(&[0.0_f64, 0.125]).unwrap(), -3);
        assert_eq!(coefficient_exponent(&[0.0_f64, -0.0]).unwrap(), 0);
        assert_eq!(
            coefficient_exponent(&[scioxide_core::Complex::new(f64::MAX, f64::MAX)]).unwrap(),
            1023
        );
        assert!(coefficient_exponent(&[scioxide_core::Complex::new(f64::NAN, 1.0)]).is_err());
    }

    #[test]
    fn stage_specific_finite_checks_accept_finite_values() {
        assert!(finite_input(1.0_f64, "value").is_ok());
        assert!(finite_callback(1.0_f64, "value").is_ok());
        assert!(finite_arithmetic(1.0_f64, "value").is_ok());
        assert!(finite_output(1.0_f64, "value").is_ok());
    }

    #[test]
    fn stage_specific_finite_checks_preserve_provenance() {
        assert!(matches!(
            (finite_input(f64::NAN, "value")).unwrap_err().view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
        assert!(matches!(
            (finite_callback(f64::INFINITY, "value"))
                .unwrap_err()
                .view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Callback,
                ..
            }
        ));
        assert!(matches!(
            (finite_arithmetic(f64::INFINITY, "value"))
                .unwrap_err()
                .view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Overflow,
                ..
            }
        ));
        assert!(matches!(
            (finite_arithmetic(f64::NAN, "value")).unwrap_err().view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Breakdown,
                ..
            }
        ));
        assert!(matches!(
            (finite_output(f64::NAN, "value")).unwrap_err().view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Output,
                ..
            }
        ));
    }
}
