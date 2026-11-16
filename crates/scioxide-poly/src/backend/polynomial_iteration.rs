//! The Weierstrass correction used by simultaneous root iteration and by
//! root inclusion disks.
//!
//! To evaluate a polynomial at the iterates, use
//! [`checked_basis_evaluation`](crate::checked_basis_evaluation).

use scioxide_core::{Complex, RealScalar, Scaled, ScaledValue, SciOxideError};

/// Holds the result of [`weierstrass_correction`].
///
/// ```
/// use scioxide_core::{Complex, ScaledValue};
/// use scioxide_poly::backend::polynomial_iteration::*;
///
/// let point = Complex::new(1.0_f64, 0.0);
/// let value = ScaledValue::from_value(Complex::new(1.0, 0.0))?;
/// let outcome = weierstrass_correction(point, [point].into_iter(), value)?;
/// assert_eq!(outcome, WeierstrassCorrection::Coincident);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeierstrassCorrection<S: RealScalar> {
    /// The correction and the nonzero product of differences, both in extended
    /// exponent range.
    Defined {
        /// The Weierstrass correction `wᵢ`.
        correction: ScaledValue<Complex<S>>,
        /// The product `Πⱼ≠ᵢ (zᵢ − zⱼ)`.
        product: ScaledValue<Complex<S>>,
    },
    /// Another candidate equals `zᵢ`, so the product is zero.
    Coincident,
}

/// Computes the Weierstrass correction of the candidate `zᵢ` of a root of a
/// polynomial `p` of degree `n` with leading coefficient `aₙ`.
///
/// ```text
/// wᵢ = (p(zᵢ)/aₙ) / Πⱼ≠ᵢ (zᵢ − zⱼ)
/// ```
///
/// `value` is `p(zᵢ)/aₙ`, and `others` yields every candidate except `zᵢ`. The
/// update `zᵢ ← zᵢ − wᵢ` is one step of the Durand–Kerner (Weierstrass)
/// iteration. In exact arithmetic the disks with centers `zᵢ − wᵢ` and radii
/// `(n − 1)|wᵢ|` contain all roots of `p`; this function only computes `wᵢ` in
/// floating point.
///
/// The correction is undefined exactly when another candidate equals `zᵢ`
/// (componentwise `==`, so `+0` and `−0` are equal). The function then returns
/// [`WeierstrassCorrection::Coincident`] and leaves the response to the caller.
/// Candidates that are close but not equal get their ordinary correction.
///
/// The product is formed in the iterator's order. Pass the candidates in a
/// fixed order if the rounding must not depend on the input order. The product
/// is first computed in `S`. If an intermediate overflows, underflows or
/// becomes subnormal, the whole product is recomputed in extended exponent
/// range, so the result does not overflow or underflow. The cost is O(n) time,
/// and the iterator is cloned at most once.
///
/// # Errors
///
/// - `NonFinite` if `point` is not finite, or if a candidate reached before a
///   coincident one is not finite.
/// - `NumericalInstability` if the result exceeds the extended exponent range.
///
/// ```
/// use scioxide_core::{Complex, ScaledValue};
/// use scioxide_poly::backend::polynomial_iteration::*;
///
/// // p(z) = z² − 1 at the candidates 2 and −1: w = p(2) / (2 − (−1)) = 1.
/// let value = ScaledValue::from_value(Complex::new(3.0_f64, 0.0))?;
/// let others = [Complex::new(-1.0, 0.0)];
/// let outcome = weierstrass_correction(Complex::new(2.0, 0.0), others.into_iter(), value)?;
/// let WeierstrassCorrection::Defined { correction, .. } = outcome else {
///     unreachable!()
/// };
/// assert_eq!(correction.round()?, Complex::new(1.0, 0.0));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn weierstrass_correction<S: RealScalar, I>(
    point: Complex<S>,
    others: I,
    value: ScaledValue<Complex<S>>,
) -> Result<WeierstrassCorrection<S>, SciOxideError>
where
    I: Iterator<Item = Complex<S>> + Clone,
{
    crate::util::finite_input(point.re, "point must have finite components")?;
    crate::util::finite_input(point.im, "point must have finite components")?;
    if let Some(outcome) = native_weierstrass(point, others.clone(), value) {
        return Ok(outcome);
    }
    let point_scaled = ScaledValue::from_value(point)?;
    let mut product = ScaledValue::from_value(Complex::new(S::ONE, S::ZERO))?;
    for other in others {
        if other == point {
            return Ok(WeierstrassCorrection::Coincident);
        }
        product = product.mul(point_scaled.sub(ScaledValue::from_value(other)?)?)?;
    }
    if product.is_zero() {
        // Unreachable for finite distinct candidates; kept as a defensive rule.
        return Ok(WeierstrassCorrection::Coincident);
    }
    Ok(WeierstrassCorrection::Defined {
        correction: value.div(product)?,
        product,
    })
}

/// Computes the product and quotient in `S`, or returns `None` to retry in
/// extended range.
///
/// The caller has rejected non-finite points. A coincident pair is reported
/// here only when it is reached before any range loss; the extended retry
/// applies the same rule.
#[inline]
fn native_weierstrass<S: RealScalar, I: Iterator<Item = Complex<S>>>(
    point: Complex<S>,
    others: I,
    value: ScaledValue<Complex<S>>,
) -> Option<WeierstrassCorrection<S>> {
    let numerator = value.round().ok()?;
    let lost = |projected: S, original: Scaled<S>| {
        (projected.is_zero() && !original.is_zero())
            || (!projected.is_zero() && projected.abs() < S::MIN_POSITIVE)
    };
    if lost(numerator.re, value.real()) || lost(numerator.im, value.imaginary()) {
        return None;
    }
    let zero = Complex::new(S::ZERO, S::ZERO);
    let mut product = Complex::new(S::ONE, S::ZERO);
    for other in others {
        if other == point {
            return Some(WeierstrassCorrection::Coincident);
        }
        let difference = point - other;
        if !difference.re.is_finite() || !difference.im.is_finite() {
            return None;
        }
        let next = product * difference;
        if scioxide_core::backend::standard::product_needs_retry(product, difference, next)
            || next == zero
        {
            return None;
        }
        product = next;
    }
    let correction = numerator / product;
    if !correction.re.is_finite()
        || !correction.im.is_finite()
        || (correction.re != S::ZERO && correction.re.abs() < S::MIN_POSITIVE)
        || (correction.im != S::ZERO && correction.im.abs() < S::MIN_POSITIVE)
        || (correction == zero && numerator != zero)
        // A normal component cannot certify the other component: it may have
        // rounded to zero. Only an axis-aligned real divisor and a zero matching
        // numerator component prove that such a zero is structural.
        || (numerator != zero
            && ((correction.re == S::ZERO && (numerator.re != S::ZERO || product.im != S::ZERO))
                || (correction.im == S::ZERO
                    && (numerator.im != S::ZERO || product.im != S::ZERO))))
    {
        return None;
    }
    Some(WeierstrassCorrection::Defined {
        correction: ScaledValue::from_value(correction).ok()?,
        product: ScaledValue::from_value(product).ok()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defined(outcome: WeierstrassCorrection<f64>) -> Complex<f64> {
        let correction = match outcome {
            WeierstrassCorrection::Defined { correction, .. } => Some(correction),
            WeierstrassCorrection::Coincident => None,
        };
        correction.unwrap().round().unwrap()
    }

    #[test]
    fn extreme_products_retry_without_projecting_the_numerator() {
        for radius in [1e-200_f64, 1e200] {
            let point = Complex::new(radius, 0.0);
            let others = [Complex::new(-radius, 0.0), Complex::new(0.0, radius)];
            let r = ScaledValue::from_value(Complex::new(radius, 0.0)).unwrap();
            let value = r.mul(r).unwrap();
            assert!(native_weierstrass(point, others.into_iter(), value).is_none());
            let correction =
                defined(weierstrass_correction(point, others.into_iter(), value).unwrap());
            assert!((correction - Complex::new(0.25, 0.25)).norm() <= 32.0 * f64::EPSILON);
        }
    }

    #[test]
    fn subnormal_numerators_are_divided_before_projection() {
        let radius = 1e-100_f64;
        let point = Complex::new(radius, 0.0);
        let others = [Complex::new(0.0, 0.0), Complex::new(0.0, radius)];
        let tiny = ScaledValue::from_value(Complex::new(1e-160, 0.0)).unwrap();
        let value = tiny.mul(tiny).unwrap();
        assert!(native_weierstrass(point, others.into_iter(), value).is_none());
        let actual = defined(weierstrass_correction(point, others.into_iter(), value).unwrap());
        let expected = Complex::new(0.5e-120, 0.5e-120);
        assert!((actual - expected).norm() <= 32.0 * f64::EPSILON * expected.norm());
    }

    #[test]
    fn native_and_extended_paths_agree() {
        let point = Complex::new(1.0_f64, 0.5);
        let others = [Complex::new(-1.0, 0.0), Complex::new(0.0, 1.0)];
        let value = ScaledValue::from_value(Complex::new(0.5, -0.25)).unwrap();
        let native = defined(native_weierstrass(point, others.into_iter(), value).unwrap());
        let mut product = ScaledValue::from_value(Complex::new(1.0, 0.0)).unwrap();
        for other in others {
            product = product
                .mul(ScaledValue::from_value(point - other).unwrap())
                .unwrap();
        }
        let extended = value.div(product).unwrap().round().unwrap();
        assert!((native - extended).norm() <= 8.0 * f64::EPSILON * extended.norm());
    }
}
