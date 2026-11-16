//! Checked scalar arithmetic and unchecked predicates on finite values.

#[cfg(feature = "std")]
use super::overflow;
use super::{RangeAccumulator, checked_final_rounding, two_sum, validate_finite};
use crate::scalar::finite_parts;
use crate::{Dimensions, DomainReason, InvalidInputReason, RealScalar, Scaled, SciOxideError};

/// Returns whether one value is positive and the other negative.
///
/// A zero of either sign is never opposite to another value; subnormal values
/// count with their sign. Unlike testing `left * right < 0`, this cannot
/// underflow. The inputs are not checked and should be finite.
///
/// # Examples
///
/// ```
/// use scioxide_core::backend::opposite_signs;
///
/// assert!(opposite_signs(-f64::from_bits(1), 1.0));
/// assert!(!opposite_signs(-0.0_f64, 1.0));
/// ```
pub fn opposite_signs<S: RealScalar>(left: S, right: S) -> bool {
    let zero = S::ZERO;
    (left < zero && right > zero) || (left > zero && right < zero)
}

/// Computes the midpoint `(left + right) / 2`, correctly rounded.
///
/// The exact midpoint is rounded once, so the result never overflows and is
/// exact for subnormal inputs until that rounding. The order of the endpoints
/// does not matter. Equal endpoints of the same sign return that endpoint bit
/// for bit, and an exact zero midpoint is `+0`.
///
/// # Errors
///
/// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if either endpoint is NaN or
///   infinite.
///
/// # Examples
///
/// ```
/// use scioxide_core::checked_midpoint;
///
/// assert_eq!(checked_midpoint(1.0_f64, 3.0)?, 2.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline]
pub fn checked_midpoint<S: RealScalar>(left: S, right: S) -> Result<S, SciOxideError> {
    validate_finite(left, "left midpoint endpoint must be finite")?;
    validate_finite(right, "right midpoint endpoint must be finite")?;
    if left == right && left.is_sign_negative() == right.is_sign_negative() {
        return Ok(left);
    }
    // Normal power-of-two products are exact. Their sum supplies the sole
    // rounding, without intermediate overflow. Tiny endpoints retain exact
    // accumulation so halving cannot discard a subnormal bit.
    let exact_half = |x: S| x == S::ZERO || x.abs() >= S::TWO * S::MIN_POSITIVE;
    if exact_half(left) && exact_half(right) {
        return Ok(left * S::HALF + right * S::HALF);
    }
    let mut accumulator = RangeAccumulator::new();
    accumulator.add_product(left, S::HALF)?;
    accumulator.add_product(right, S::HALF)?;
    accumulator.finish()
}

/// Returns the representable value next to `value` in the direction of
/// `toward`.
///
/// This is the checked form of [`RealScalar::next_after`], with the same
/// handling of signed zeros.
///
/// # Errors
///
/// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if either input is NaN or
///   infinite.
///
/// # Examples
///
/// ```
/// use scioxide_core::checked_next_after;
///
/// // The smallest positive subnormal.
/// assert_eq!(checked_next_after(0.0_f64, 1.0)?, f64::from_bits(1));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn checked_next_after<S: RealScalar>(value: S, toward: S) -> Result<S, SciOxideError> {
    validate_finite(value, "next-after value must be finite")?;
    validate_finite(toward, "next-after target must be finite")?;
    Ok(value.next_after(toward))
}

/// Computes `value · 2^exponent`, correctly rounded.
///
/// This is the checked form of [`RealScalar::scalbn`]. A result that rounds to
/// a subnormal or zero is returned as is.
///
/// # Errors
///
/// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if `value` is NaN or
///   infinite.
/// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
///   ([`Overflow`](crate::NumericalIssue::Overflow)) if the result is outside the
///   finite range.
///
/// # Examples
///
/// ```
/// use scioxide_core::checked_scalbn;
///
/// assert_eq!(checked_scalbn(1.5_f64, 2)?, 6.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn checked_scalbn<S: RealScalar>(value: S, exponent: i32) -> Result<S, SciOxideError> {
    validate_finite(value, "power-of-two scaling input must be finite")?;
    checked_final_rounding(
        value.scalbn(exponent),
        "power-of-two scaled value is outside the finite scalar range",
    )
}

/// Computes `left − right`, correctly rounded.
///
/// Equal operands give `+0`.
///
/// # Errors
///
/// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if either operand is NaN or
///   infinite.
/// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
///   ([`Overflow`](crate::NumericalIssue::Overflow)) if the difference is outside the
///   finite range.
///
/// # Examples
///
/// ```
/// use scioxide_core::checked_difference;
///
/// assert_eq!(checked_difference(3.0_f64, 1.0)?, 2.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn checked_difference<S: RealScalar>(left: S, right: S) -> Result<S, SciOxideError> {
    validate_finite(left, "left difference operand must be finite")?;
    validate_finite(right, "right difference operand must be finite")?;
    checked_final_rounding(
        if left == right { S::ZERO } else { left - right },
        "difference is outside the finite scalar range",
    )
}

/// Returns whether `|left − right| ≤ limit`, without overflow when the operands
/// have opposite signs.
///
/// The inputs are not checked: they must be finite and `limit` nonnegative.
/// Equal values, including zeros of opposite sign, have difference zero. The
/// comparison is made in working precision and is not a rigorous bound.
///
/// # Examples
///
/// ```
/// use scioxide_core::backend::absolute_difference_le;
///
/// assert!(absolute_difference_le(-1.0_f64, 1.0, 2.0));
/// assert!(!absolute_difference_le(-f64::MAX, f64::MAX, f64::MAX));
/// ```
#[inline(always)]
pub fn absolute_difference_le<S: RealScalar>(left: S, right: S, limit: S) -> bool {
    if left == right {
        return true;
    }
    if !opposite_signs(left, right) {
        return (left - right).abs() <= limit;
    }

    let left_magnitude = left.abs();
    if left_magnitude > limit {
        return false;
    }
    right.abs() <= limit - left_magnitude
}

/// Computes `left · right`, correctly rounded.
///
/// An exact zero product is `+0`; a nonzero product that underflows keeps its
/// sign.
///
/// # Errors
///
/// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if either factor is NaN or
///   infinite.
/// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
///   ([`Overflow`](crate::NumericalIssue::Overflow)) if the product is outside the
///   finite range.
///
/// # Examples
///
/// ```
/// use scioxide_core::checked_float_product;
///
/// assert_eq!(checked_float_product(2.0_f64, 3.0)?, 6.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline]
pub fn checked_float_product<S: RealScalar>(left: S, right: S) -> Result<S, SciOxideError> {
    validate_finite(left, "left product factor must be finite")?;
    validate_finite(right, "right product factor must be finite")?;
    // RealScalar is sealed to IEEE binary32/binary64. A single multiplication
    // already rounds the exact product once, including gradual underflow.
    // Exact zero follows the accumulator's canonical cancellation convention;
    // a negative nonzero value rounded by underflow retains its negative zero.
    if left == S::ZERO || right == S::ZERO {
        return Ok(S::ZERO);
    }
    checked_final_rounding(left * right, "product is outside the finite scalar range")
}

/// Computes the product `Π factors` without overflow or underflow in
/// intermediate products.
///
/// Each factor is split into a significand and a binary exponent. The
/// significands are multiplied in order, the exponents are summed separately,
/// and the result is scaled once at the end. Cost is `O(n)`. The empty product
/// is `1`. A product with a zero factor is `+0`, but every factor is still
/// checked; a nonzero product that underflows keeps its sign.
///
/// # Accuracy
///
/// Let `P` be the exact product, `u = ε/2` the unit roundoff, `k = n − 1` for
/// `n ≥ 1` factors, and `η` the smallest positive subnormal. Then
///
/// ```text
/// |computed − P| ≤ ((1 + u)^k − 1) |P| + η/2 ≤ γ_k |P| + η/2,   γ_k = k u / (1 − k u),
/// ```
///
/// the second form holding for `k u < 1`. The `η/2` term arises only when the
/// result is subnormal. The error grows with the number of factors.
///
/// # Errors
///
/// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if a factor is NaN or
///   infinite.
/// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability) if
///   the intermediate exponent leaves the extended range or the product is
///   outside the finite range.
///
/// # Examples
///
/// ```
/// use scioxide_core::checked_float_product_factors;
///
/// // MAX · 2 overflows on its own, but the full product is MAX.
/// assert_eq!(checked_float_product_factors(&[f64::MAX, 2.0, 0.5])?, f64::MAX);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn checked_float_product_factors<S: RealScalar>(factors: &[S]) -> Result<S, SciOxideError> {
    // Validate the whole slice before a zero can absorb the remaining factors.
    let mut zero = false;
    for &factor in factors {
        validate_finite(factor, "product factor must be finite")?;
        zero |= factor == S::ZERO;
    }
    if zero {
        return Ok(S::ZERO);
    }
    factors
        .iter()
        .try_fold(Scaled::ONE, |product, &factor| {
            product.mul(Scaled::from_value(factor)?)
        })?
        .round()
}

/// Computes `left · right / denominator` without overflow or underflow in the
/// intermediate product.
///
/// The operands are split into significands and exponents, so only the final
/// quotient must be representable. An exact zero is `+0`; a nonzero result
/// that underflows keeps its sign.
///
/// # Accuracy
///
/// The result is within 4 ulps of the correctly rounded quotient, including
/// subnormal results. With `Q` the exact quotient, `u = ε/2` and `η` the
/// smallest positive subnormal,
///
/// ```text
/// |computed − Q| ≤ γ₂ |Q| + η/2,   γ₂ = 2u / (1 − 2u),
/// ```
///
/// where the `η/2` term arises only for subnormal results.
///
/// # Errors
///
/// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if an operand is NaN or
///   infinite.
/// - [`DomainError`](crate::SciOxideErrorKind::DomainError) if `denominator` is
///   zero.
/// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
///   ([`Overflow`](crate::NumericalIssue::Overflow)) if the quotient is outside the
///   finite range.
///
/// # Examples
///
/// ```
/// use scioxide_core::checked_product_quotient;
///
/// // MAX · 2 overflows on its own, but MAX · 2 / 2 = MAX.
/// assert_eq!(checked_product_quotient(f64::MAX, 2.0, 2.0)?, f64::MAX);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn checked_product_quotient<S: RealScalar>(
    left: S,
    right: S,
    denominator: S,
) -> Result<S, SciOxideError> {
    validate_finite(left, "left product factor must be finite")?;
    validate_finite(right, "right product factor must be finite")?;
    validate_finite(denominator, "denominator must be finite")?;
    if denominator == S::ZERO {
        return Err(SciOxideError::domain_error(
            crate::argument!("denominator"),
            DomainReason::Undefined,
            "denominator must be nonzero",
        ));
    }
    if left == S::ZERO || right == S::ZERO {
        return Ok(S::ZERO);
    }
    Scaled::from_value(left)?
        .mul(Scaled::from_value(right)?)?
        .div(Scaled::from_value(denominator)?)?
        .round()
}

/// Computes `numerator / denominator`, correctly rounded (to nearest, ties to
/// even), including subnormal results.
///
/// A zero numerator gives `+0`; a negative nonzero quotient that rounds to zero
/// is `-0`.
///
/// # Errors
///
/// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if either operand is NaN or
///   infinite.
/// - [`DomainError`](crate::SciOxideErrorKind::DomainError) if `denominator` is
///   zero.
/// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
///   ([`Overflow`](crate::NumericalIssue::Overflow)) if the quotient is outside the
///   finite range.
///
/// # Examples
///
/// ```
/// use scioxide_core::checked_float_quotient;
///
/// assert_eq!(checked_float_quotient(6.0_f64, 2.0)?, 3.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline]
pub fn checked_float_quotient<S: RealScalar>(
    numerator: S,
    denominator: S,
) -> Result<S, SciOxideError> {
    validate_finite(numerator, "numerator must be finite")?;
    validate_finite(denominator, "denominator must be finite")?;
    if denominator == S::ZERO {
        return Err(SciOxideError::domain_error(
            crate::argument!("denominator"),
            DomainReason::Undefined,
            "denominator must be nonzero",
        ));
    }
    checked_final_rounding(
        if numerator == S::ZERO {
            S::ZERO
        } else {
            numerator / denominator
        },
        "quotient is outside the finite scalar range",
    )
}

/// Computes `Σ values`, correctly rounded.
///
/// Longer inputs are first summed by compensated summation (Ogita, Rump and
/// Oishi's `Sum2`). If an error bound proves which floating-point number the
/// exact sum rounds to, that number is returned; otherwise the sum is
/// accumulated exactly and rounded once. Either way the result is the exact sum rounded to nearest,
/// ties to even, and intermediate overflow cannot occur. Cost is `O(n)`. An
/// empty slice or exact cancellation gives `+0`.
///
/// # Errors
///
/// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if a value is NaN or
///   infinite.
/// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
///   ([`Overflow`](crate::NumericalIssue::Overflow)) if the sum is outside the
///   finite range.
///
/// # Examples
///
/// ```
/// use scioxide_core::checked_sum;
///
/// // The exact sum is 1; naive summation returns 0.
/// assert_eq!(checked_sum(&[1.0e100_f64, 1.0, -1.0e100])?, 1.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline]
pub fn checked_sum<S: RealScalar>(values: &[S]) -> Result<S, SciOxideError> {
    match values {
        [] => return Ok(S::ZERO),
        [value] => {
            validate_finite(*value, "sum input must be finite")?;
            return Ok(if *value == S::ZERO { S::ZERO } else { *value });
        }
        [left, right] => {
            validate_finite(*left, "sum input must be finite")?;
            validate_finite(*right, "sum input must be finite")?;
            // One IEEE addition has the same rounding as the exact accumulator.
            let value = *left + *right;
            return checked_final_rounding(
                if value == S::ZERO { S::ZERO } else { value },
                "sum is outside the finite scalar range",
            );
        }
        // Bound every TwoSum intermediate away from overflow. Each transform
        // preserves its exact sum, so a+b+c = rounded + tail + remainder.
        [a, b, c]
            if values
                .iter()
                .all(|v| v.abs() <= S::MAX * (S::HALF * S::HALF)) =>
        {
            let (sum, low) = two_sum(*a, *b);
            let (sum, next_low) = two_sum(sum, *c);
            let (correction, remainder) = two_sum(low, next_low);
            let (rounded, tail) = two_sum(sum, correction);
            // For normal rounded values, |rounded|*eps/4 is no larger
            // than half the spacing to either neighbor (including powers
            // of two). Strict comparison leaves midpoint ties to the exact
            // accumulator. A rounded nonnegative bound below this margin
            // also bounds the exact |tail|+|remainder| below the margin.
            let margin = rounded.abs() * (S::EPSILON * (S::HALF * S::HALF));
            if tail == -remainder || (margin.is_normal() && tail.abs() + remainder.abs() < margin) {
                return Ok(if rounded == S::ZERO { S::ZERO } else { rounded });
            }
        }
        _ => {}
    }
    if values.len() > 3
        && let Some(value) = certified_long_sum(values)
    {
        return Ok(value);
    }
    let mut accumulator = RangeAccumulator::new();
    accumulator.add_weighted_values(S::ONE, values)?;
    accumulator.finish()
}

/// Certified compensated summation: Ogita, Rump and Oishi's `Sum2` in four
/// independent lanes. Returns the correctly rounded sum when an a-priori bound
/// on the compensation error proves it, and `None` for the exact accumulator.
///
/// Every TwoSum error is exact, so the true sum is `sum + Σe`. The compensation
/// sums the `m` errors with error at most `γ_m·Σ|e| ≤ (m·u)²·(1 + δ)·Σ|x|`; the
/// bound below uses four times `(m·u)²·Σ|x|` for `m·u < 1/64`. The enclosure
/// `rounded + tail ± bound` must lie strictly between the actual neighboring
/// midpoints, including their asymmetric distances at powers of two. Midpoint
/// ties and unproved rounding cells fall back to exact accumulation.
#[inline]
fn certified_long_sum<S: RealScalar>(values: &[S]) -> Option<S> {
    const LANES: usize = 4;
    let mut sums = [S::ZERO; LANES];
    let mut errors = [S::ZERO; LANES];
    let mut magnitudes = [S::ZERO; LANES];
    let mut add = |lane: usize, value: S| {
        let (sum, error) = two_sum(sums[lane], value);
        sums[lane] = sum;
        errors[lane] += error;
        magnitudes[lane] += value.abs();
    };
    let mut chunks = values.chunks_exact(LANES);
    for chunk in &mut chunks {
        for (lane, &value) in chunk.iter().enumerate() {
            add(lane, value);
        }
    }
    for (lane, &value) in chunks.remainder().iter().enumerate() {
        add(lane, value);
    }
    let (mut sum, mut error, mut magnitude) = (sums[0], errors[0], magnitudes[0]);
    for lane in 1..LANES {
        let (combined, low) = two_sum(sum, sums[lane]);
        sum = combined;
        error = error + low + errors[lane];
        magnitude += magnitudes[lane];
    }
    // Every partial |sum| is at most the magnitude, so no TwoSum overflowed;
    // non-finite inputs or overflow fail this test and use the accumulator.
    let bounded = magnitude.is_finite() && magnitude <= S::MAX * (S::HALF * S::HALF);
    if !bounded {
        return None;
    }
    let (rounded, tail) = two_sum(sum, error);
    let terms = S::try_from_usize_exact(values.len().checked_add(3 * LANES)?).ok()?;
    let scaled = terms * (S::EPSILON * S::HALF);
    if scaled >= S::ONE / S::try_from_usize_exact(64).ok()? {
        return None;
    }
    // The factor four covers reduction and bound-evaluation rounding; stepping
    // outward also covers the final multiplication's possible underflow.
    let bound = (S::TWO * S::TWO * scaled * scaled * magnitude).next_after(S::MAX);
    super::certified_rounding_cell(rounded, tail, bound)
}

/// Computes `left · right + addend`, correctly rounded (fused multiply-add).
///
/// Uses the IEEE fused multiply-add with `std` and exact accumulation without
/// it; both give the same result. An exact zero is `+0`; a nonzero result that
/// rounds to zero keeps its sign.
///
/// # Errors
///
/// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if an operand is NaN or
///   infinite.
/// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
///   ([`Overflow`](crate::NumericalIssue::Overflow)) if the result is outside the
///   finite range.
///
/// # Examples
///
/// ```
/// use scioxide_core::checked_mul_add;
///
/// assert_eq!(checked_mul_add(2.0_f64, 3.0, 1.0)?, 7.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline]
pub fn checked_mul_add<S: RealScalar>(left: S, right: S, addend: S) -> Result<S, SciOxideError> {
    #[cfg(feature = "std")]
    {
        validate_finite(left, "left factor must be finite")?;
        validate_finite(right, "right factor must be finite")?;
        validate_finite(addend, "addend must be finite")?;
        let value = left.mul_add(right, addend);
        if !value.is_finite() {
            return Err(overflow("fused expression is not representable"));
        }
        if value == S::ZERO {
            // The native result alone cannot distinguish exact cancellation
            // from negative underflow. Resolve only this boundary exactly.
            let mut accumulator = RangeAccumulator::new();
            accumulator.add_product(left, right)?;
            accumulator.add(addend)?;
            return accumulator.finish();
        }
        Ok(value)
    }
    #[cfg(not(feature = "std"))]
    {
        let mut accumulator = RangeAccumulator::new();
        accumulator.add_product(left, right)?;
        accumulator.add(addend)?;
        accumulator.finish()
    }
}

/// Computes the dot product `Σ leftᵢ · rightᵢ`, correctly rounded.
///
/// The exact sum of the exact products is rounded once to nearest, ties to
/// even, so the result does not depend on the order of the pairs, and
/// intermediate products and sums cannot overflow or underflow. Cost is `O(n)`.
/// Short inputs use error-free products with a rounding check, and fall back to
/// exact accumulation when the check is inconclusive; the result is the same
/// with or without a hardware FMA.
///
/// # Errors
///
/// - [`DimensionMismatch`](crate::SciOxideErrorKind::DimensionMismatch) if the
///   slices differ in length.
/// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if a value is NaN or
///   infinite.
/// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
///   ([`Overflow`](crate::NumericalIssue::Overflow)) if the result is outside the
///   finite range.
///
/// # Examples
///
/// ```
/// use scioxide_core::checked_sum_of_products;
///
/// // 2·4 + 3·5 = 23
/// assert_eq!(checked_sum_of_products(&[2.0_f64, 3.0], &[4.0, 5.0])?, 23.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn checked_sum_of_products<S: RealScalar>(left: &[S], right: &[S]) -> Result<S, SciOxideError> {
    if left.len() != right.len() {
        return Err(SciOxideError::dimension_mismatch(
            Dimensions::one(left.len()),
            Dimensions::one(right.len()),
        ));
    }
    if let ([a, c], [b, d]) = (left, right)
        && let Some(value) = super::certified_sum_two_products(*a, *b, *c, *d)
    {
        return Ok(value);
    }
    if (3..=4).contains(&left.len()) {
        let mut pairs = [(S::ZERO, S::ZERO); 4];
        for (pair, (&a, &b)) in pairs.iter_mut().zip(left.iter().zip(right)) {
            *pair = (a, b);
        }
        if let Some(value) = super::certified_short_sum_products(&pairs[..left.len()]) {
            return Ok(value);
        }
    }
    // Short stencils commonly contain only two nonzero dyadic products. A
    // represented product is exact iff its integer significand fits the
    // destination precision and its lowest bit is not below gradual underflow.
    // This proof works in both feature modes, without relying on an FMA whose
    // residual could itself underflow. Ambiguous cases retain the accumulator.
    if left.len() <= 4 {
        let minimum_exponent = crate::scaled::rounding::minimum_subnormal_exponent::<S>();
        let mut pairs = [None; 4];
        let mut products = [S::ZERO; 4];
        let mut count = 0;
        let mut exact = true;
        for (&a, &b) in left.iter().zip(right) {
            let ap = finite_parts(a, "left factor must be finite")?;
            let bp = finite_parts(b, "right factor must be finite")?;
            let (integer, exponent, _) = ap.product(bp);
            if integer == 0 {
                continue;
            }
            pairs[count] = Some((ap, bp));
            let low = integer.trailing_zeros();
            let width = u128::BITS - integer.leading_zeros() - low;
            let product = a * b;
            exact &= width <= S::MANTISSA_DIGITS
                && exponent + low as i32 >= minimum_exponent
                && product.is_finite();
            products[count] = product;
            count += 1;
        }
        if exact {
            let mut sum = S::ZERO;
            for (index, &product) in products[..count].iter().enumerate() {
                let next = sum + product;
                if index + 1 < count {
                    // Knuth TwoSum: every addition before the final one must
                    // have zero residual. Check every intermediate for overflow.
                    let recovered = next - sum;
                    let residual = (sum - (next - recovered)) + (product - recovered);
                    if !next.is_finite() || !recovered.is_finite() || residual != S::ZERO {
                        exact = false;
                        break;
                    }
                }
                sum = next;
            }
            if exact {
                let value = checked_final_rounding(
                    if sum == S::ZERO { S::ZERO } else { sum },
                    "sum is outside the finite scalar range",
                )?;
                return Ok(value);
            }
        }
        let mut accumulator = RangeAccumulator::new();
        pairs[..count]
            .iter()
            .flatten()
            .try_for_each(|&(a, b)| accumulator.update_finite_product(a, b, false))?;
        return accumulator.finish();
    }
    super::checked_product_sum(left.iter().copied().zip(right.iter().copied()))
}

/// Computes `left_a · left_b − right_a · right_b`, correctly rounded.
///
/// This is free of the cancellation error of the naive formula, for example in
/// 2×2 determinants and discriminants.
///
/// # Errors
///
/// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if an operand is NaN or
///   infinite.
/// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
///   ([`Overflow`](crate::NumericalIssue::Overflow)) if the result is outside the
///   finite range.
///
/// # Examples
///
/// ```
/// use scioxide_core::checked_difference_of_products;
///
/// // 3·4 − 2·5 = 2
/// assert_eq!(checked_difference_of_products(3.0_f64, 4.0, 2.0, 5.0)?, 2.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn checked_difference_of_products<S: RealScalar>(
    left_a: S,
    left_b: S,
    right_a: S,
    right_b: S,
) -> Result<S, SciOxideError> {
    let mut accumulator = RangeAccumulator::new();
    accumulator.add_product(left_a, left_b)?;
    accumulator.subtract_product(right_a, right_b)?;
    accumulator.finish()
}

/// Computes the linear interpolation `(1 − t) left + t right` for
/// `t = fraction ∈ [0, 1]`, correctly rounded.
///
/// `fraction = 0` and `fraction = 1` return `left` and `right` exactly, and every
/// result lies in the closed interval between the endpoints.
///
/// # Errors
///
/// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if an argument is NaN or
///   infinite.
/// - [`InvalidInput`](crate::SciOxideErrorKind::InvalidInput) (`OutOfRange`) if
///   `fraction` is outside `[0, 1]`.
///
/// # Examples
///
/// ```
/// use scioxide_core::checked_affine;
///
/// assert_eq!(checked_affine(1.0_f64, 3.0, 0.5)?, 2.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn checked_affine<S: RealScalar>(left: S, right: S, fraction: S) -> Result<S, SciOxideError> {
    validate_finite(left, "left affine endpoint must be finite")?;
    validate_finite(right, "right affine endpoint must be finite")?;
    validate_finite(fraction, "affine fraction must be finite")?;
    if fraction < S::ZERO || fraction > S::ONE {
        return Err(SciOxideError::invalid_input(
            crate::argument!("fraction"),
            InvalidInputReason::OutOfRange,
            "affine fraction must lie in [0, 1]",
        ));
    }
    if fraction == S::ZERO {
        return Ok(left);
    }
    if fraction == S::ONE {
        return Ok(right);
    }
    let mut accumulator = RangeAccumulator::new();
    accumulator.add(left)?;
    accumulator.subtract_product(fraction, left)?;
    accumulator.add_product(fraction, right)?;
    accumulator.finish()
}

/// Computes `((d − n) left + n right) / d` for integers `0 ≤ n ≤ d`, `d > 0`,
/// correctly rounded.
///
/// Here `n = numerator` and `d = denominator`. The integer weights are used
/// exactly, even when they have more bits than the significand of `S`, and the
/// result is rounded once to nearest, ties to even. It lies between the
/// endpoints, and `n = 0` and `n = d` return `left` and `right` bit for bit.
/// Use it for points `i / d` of a uniform grid on `[left, right]`.
///
/// # Errors
///
/// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if an endpoint is NaN or
///   infinite.
/// - [`InvalidInput`](crate::SciOxideErrorKind::InvalidInput) (`OutOfRange`) if
///   `denominator` is zero or `numerator > denominator`.
///
/// # Examples
///
/// ```
/// use scioxide_core::checked_affine_ratio;
///
/// // One quarter of the way from −1 to 1.
/// assert_eq!(checked_affine_ratio(-1.0_f32, 1.0, 1, 4)?, -0.5);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn checked_affine_ratio<S: RealScalar>(
    left: S,
    right: S,
    numerator: usize,
    denominator: usize,
) -> Result<S, SciOxideError> {
    validate_finite(left, "left affine endpoint must be finite")?;
    validate_finite(right, "right affine endpoint must be finite")?;
    if denominator == 0 || numerator > denominator {
        return Err(SciOxideError::invalid_input(
            if denominator == 0 {
                crate::argument!("denominator")
            } else {
                crate::argument!("numerator")
            },
            InvalidInputReason::OutOfRange,
            "affine ratio requires 0 <= numerator <= denominator and a nonzero denominator",
        ));
    }
    if numerator == 0 {
        return Ok(left);
    }
    if numerator == denominator {
        return Ok(right);
    }
    let mut sum = RangeAccumulator::new();
    sum.add_integer_product(left, denominator - numerator)?;
    sum.add_integer_product(right, numerator)?;
    sum.finish_integer_quotient(denominator)
}

#[cfg(test)]
mod tests {

    use super::super::certified_rounding_cell;
    use super::*;
    use crate::NumericalIssue;

    #[test]
    fn rounding_cells_use_both_signed_midpoint_distances() {
        fn check<S: RealScalar>() {
            let quarter_ulp = S::EPSILON * S::HALF * S::HALF;
            let small_bound = quarter_ulp * S::HALF * S::HALF;
            let wider_tail = quarter_ulp + quarter_ulp * S::HALF;
            for sign in [S::ONE, -S::ONE] {
                let candidate = sign;
                // At ±1, the cell extends twice as far away from zero.
                assert_eq!(
                    certified_rounding_cell(candidate, sign * wider_tail, small_bound),
                    Some(candidate)
                );
                assert_eq!(
                    certified_rounding_cell(candidate, -sign * wider_tail, small_bound),
                    None
                );
                // Equality at either midpoint, or an enclosure touching it,
                // must never certify, regardless of the candidate's parity.
                for rounded in [candidate, candidate.next_after(sign * S::MAX)] {
                    let below = (rounded - rounded.next_after(-S::MAX)) * S::HALF;
                    let above = (rounded.next_after(S::MAX) - rounded) * S::HALF;
                    for tail in [-below, above] {
                        assert_eq!(certified_rounding_cell(rounded, tail, S::ZERO), None);
                    }
                    assert_eq!(certified_rounding_cell(rounded, S::ZERO, below), None);
                    assert_eq!(certified_rounding_cell(rounded, S::ZERO, above), None);
                }
            }
            for rounded in [
                S::ZERO,
                -S::ZERO,
                S::MIN_SUBNORMAL,
                S::MIN_POSITIVE,
                S::MAX,
                S::INFINITY,
                S::NAN,
            ] {
                assert_eq!(certified_rounding_cell(rounded, S::ZERO, S::ZERO), None);
            }
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn long_sum_certificate_accepts_wide_tails_and_rejects_midpoints() {
        fn check<S: RealScalar>() {
            let quarter = S::HALF * S::HALF;
            let odd = S::ONE + S::EPSILON;
            for sign in [S::ONE, -S::ONE] {
                for head in [S::ONE, S::ONE + quarter] {
                    let tail = S::EPSILON * (quarter + quarter * S::HALF);
                    let values = [sign * head, sign * tail, S::ZERO, S::ZERO];
                    assert_eq!(certified_long_sum(&values), Some(sign * head));
                }
                for (head, tail) in [
                    (S::ONE, S::EPSILON * S::HALF),
                    (odd, S::EPSILON * S::HALF),
                    (S::ONE, -S::EPSILON * quarter),
                ] {
                    let values = [sign * head, sign * tail, S::ZERO, S::ZERO];
                    assert_eq!(certified_long_sum(&values), None);
                    let mut exact = RangeAccumulator::new();
                    exact.add_weighted_values(S::ONE, &values).unwrap();
                    assert_eq!(checked_sum(&values).unwrap(), exact.finish().unwrap());
                }
            }
            assert_eq!(
                certified_long_sum(&[S::ONE, -S::ONE, S::ZERO, S::ZERO]),
                None
            );
            assert_eq!(certified_long_sum(&[S::MIN_SUBNORMAL; 4]), None);
            assert_eq!(certified_long_sum(&[S::MAX; 4]), None);
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn three_term_sums_match_exact_accumulation_at_rounding_and_range_boundaries() {
        fn check<S: super::RealScalar>() {
            let compare = |values: [S; 3]| {
                let mut exact = super::RangeAccumulator::new();
                for value in values {
                    exact.add(value).unwrap();
                }
                let expected = exact.finish();
                let actual = super::checked_sum(&values);
                match expected {
                    Ok(expected) => assert_eq!(
                        actual.unwrap().into_f64().to_bits(),
                        expected.into_f64().to_bits()
                    ),
                    Err(expected) => assert_eq!(actual.unwrap_err().kind(), expected.kind()),
                }
            };
            let values = [
                -S::MAX,
                -S::ONE,
                -S::EPSILON * S::HALF,
                -S::MIN_SUBNORMAL,
                S::ZERO,
                S::MIN_SUBNORMAL,
                S::EPSILON * S::HALF,
                S::ONE,
                S::ONE + S::EPSILON,
                S::MAX,
            ];
            for a in values {
                for b in values {
                    for c in values {
                        compare([a, b, c]);
                    }
                }
            }
            let mut state = 957_u64;
            for i in 0..10_000 {
                let values = core::array::from_fn(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1);
                    let value = S::from_u64_half_open01(state).scalbn((state % 101) as i32 - 50);
                    if state & 1 == 0 { value } else { -value }
                });
                compare(values);
                if i % 2 == 0 {
                    compare([values[0], -values[0], values[1]]);
                }
            }
            for bad in [S::NAN, S::INFINITY, S::NEG_INFINITY] {
                for values in [
                    [bad, S::ONE, S::ONE],
                    [S::ONE, bad, S::ONE],
                    [S::ONE, S::ONE, bad],
                ] {
                    assert_eq!(
                        super::checked_sum(&values).unwrap_err().kind(),
                        crate::SciOxideErrorKind::NonFinite
                    );
                }
            }
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn single_operations_match_the_exact_accumulator_across_ieee_encodings() {
        macro_rules! check {
            ($scalar:ty, $bits:ty) => {{
                let mut bits = 0x4752_4144_4941_0206_u64;
                for _ in 0..2048 {
                    bits = bits.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let left = <$scalar>::from_bits(bits as $bits);
                    bits = bits.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let right = <$scalar>::from_bits(bits as $bits);
                    if !left.is_finite() || !right.is_finite() {
                        continue;
                    }
                    let mut exact_product = RangeAccumulator::new();
                    exact_product
                        .add_product(left, right)
                        .unwrap();
                    let mut exact_sum = RangeAccumulator::new();
                    exact_sum.add(left).unwrap();
                    exact_sum.add(right).unwrap();
                    for (actual, expected) in [
                        (
                            checked_float_product(left, right),
                            exact_product.finish(),
                        ),
                        (
                            checked_sum(&[left, right]),
                            exact_sum.finish(),
                        ),
                    ] {
                        match (actual, expected) {
                            (Ok(actual), Ok(expected)) => {
                                assert_eq!(actual.to_bits(), expected.to_bits())
                            }
                            (Err(actual), Err(expected)) if matches!(actual.view(), crate::SciOxideErrorView::NumericalInstability { .. }) && matches!(expected.view(), crate::SciOxideErrorView::NumericalInstability { .. }) => {}
                            pair => panic!("IEEE operation differs from exact oracle: {pair:?}"),
                        }
                    }
                }
                let tiny = <$scalar>::from_bits(1);
                assert_eq!(
                    checked_float_product(-tiny, 0.25)
                        .unwrap()
                        .to_bits(),
                    (-0.0 as $scalar).to_bits()
                );
                assert_eq!(
                    checked_float_product(-0.0 as $scalar, 1.0)
                        .unwrap()
                        .to_bits(),
                    (0.0 as $scalar).to_bits()
                );
                assert!(matches!(
                    checked_float_product(0.0 as $scalar, <$scalar>::NAN),
                    Err(error) if matches!(error.view(), crate::SciOxideErrorView::NonFinite { .. })
                ));
                assert!(matches!(
                    checked_sum(&[<$scalar>::NAN]),
                    Err(error) if matches!(error.view(), crate::SciOxideErrorView::NonFinite { .. })
                ));
            }};
        }
        check!(f32, u32);
        check!(f64, u64);
    }

    #[test]
    fn fused_expression_recovers_range_and_preserves_the_checked_contract() {
        macro_rules! check {
            ($scalar:ty) => {{
                let maximum = <$scalar>::MAX;
                assert_eq!(
                    checked_mul_add(maximum, 2.0, -maximum).unwrap(),
                    maximum
                );
                let tiny = <$scalar>::from_bits(1);
                assert_eq!(
                    checked_mul_add(tiny, 0.5, tiny).unwrap(),
                    tiny * 2.0
                );
                assert_eq!(
                    checked_mul_add(-0.0 as $scalar, 1.0, -0.0)
                        .unwrap()
                        .to_bits(),
                    0
                );
                assert!(matches!((checked_mul_add (maximum , 2.0 , 0.0 )).as_ref().unwrap_err().view(), crate :: SciOxideErrorView :: NumericalInstability { .. }));
                assert!(matches!(
                    checked_mul_add(<$scalar>::NAN, 1.0, 0.0),
                    Err(error) if matches!(error.view(), crate::SciOxideErrorView::NonFinite { .. })
                ));
            }};
        }
        check!(f32);
        check!(f64);
    }

    #[test]
    fn midpoint_and_difference_cover_extreme_signs() {
        assert_eq!(checked_midpoint(-f64::MAX, f64::MAX).unwrap(), 0.0);
        assert!(matches!(
            (checked_difference(f64::MAX, -f64::MAX))
                .as_ref()
                .unwrap_err()
                .view(),
            crate::SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Overflow,
                ..
            }
        ));
        assert!(!absolute_difference_le(f64::MAX, -f64::MAX, f64::MAX));
        assert!(absolute_difference_le(
            f64::MIN_POSITIVE,
            0.0,
            f64::MIN_POSITIVE
        ));
    }

    #[test]
    fn adjacent_and_power_of_two_helpers_cover_ieee_boundaries() {
        assert_eq!(
            checked_next_after(0.0_f64, -1.0).unwrap().to_bits(),
            (-(f64::from_bits(1))).to_bits()
        );
        assert_eq!(checked_scalbn(f64::MIN_POSITIVE, -52).unwrap().to_bits(), 1);
        assert!(matches!(
            (checked_scalbn(f64::MAX, 1)).as_ref().unwrap_err().view(),
            crate::SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Overflow,
                ..
            }
        ));
    }

    #[test]
    fn cross_scaled_products_and_subnormal_midpoints_round_once() {
        assert_eq!(
            checked_sum_of_products(
                &[f64::MAX, f64::MIN_POSITIVE],
                &[f64::MIN_POSITIVE, f64::MAX],
            )
            .unwrap()
            .to_bits(),
            0x401f_ffff_ffff_ffff
        );
        assert_eq!(
            checked_sum_of_products(
                &[f32::MAX, f32::MIN_POSITIVE],
                &[f32::MIN_POSITIVE, f32::MAX],
            )
            .unwrap()
            .to_bits(),
            0x40ff_ffff
        );
        let f64_u = f64::from_bits(1);
        let f32_u = f32::from_bits(1);
        assert_eq!(checked_midpoint(-f64_u, 3.0 * f64_u).unwrap().to_bits(), 1);
        assert_eq!(checked_midpoint(-f32_u, 3.0 * f32_u).unwrap().to_bits(), 1);
    }

    #[test]
    fn cancellation_does_not_reject_avoidable_product_overflow() {
        assert_eq!(
            checked_difference_of_products(f64::MAX, 2.0, f64::MAX, 2.0).unwrap(),
            0.0
        );
        assert_eq!(checked_sum(&[f64::MAX, -f64::MAX, 1.0]).unwrap(), 1.0);
    }

    #[test]
    fn factor_products_defer_range_checks_until_the_final_scale() {
        let recovered =
            checked_float_product_factors(&[f64::MAX, f64::MAX, 0.5, 2.0 / f64::MAX]).unwrap();
        assert!(f64::MAX.to_bits() - recovered.to_bits() <= 2);

        let recovered =
            checked_float_product_factors(&[f32::MAX, f32::MAX, 0.5, 2.0 / f32::MAX]).unwrap();
        assert!(f32::MAX.to_bits() - recovered.to_bits() <= 2);
        assert_eq!(checked_float_product_factors::<f64>(&[]).unwrap(), 1.0);
        assert!(matches!(
            (checked_float_product_factors(&[f64::MAX, 2.0]))
                .as_ref()
                .unwrap_err()
                .view(),
            crate::SciOxideErrorView::NumericalInstability {
                issue: NumericalIssue::Overflow,
                ..
            }
        ));
    }
}
