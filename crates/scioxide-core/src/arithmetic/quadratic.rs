//! Roots of real quadratics with an exactly accumulated discriminant when needed.

use super::accumulator::{ACCUMULATOR_LIMBS, ACCUMULATOR_MIN_EXPONENT};
#[cfg(feature = "std")]
use super::two_sum;
use super::{RangeAccumulator, checked_product_quotient, validate_finite};
use crate::scaled::rounding::{highest_bit, shifted_low_u64};
use crate::{Complex, InvalidInputReason, NumericalIssue, RealScalar, Scaled, SciOxideError};
use core::cmp::Ordering;
use core::ops::Neg;

fn scaled_sqrt<S: RealScalar>(
    words: &[u64; ACCUMULATOR_LIMBS],
) -> Result<Option<Scaled<S>>, SciOxideError> {
    let Some(top_bit) = highest_bit(words) else {
        return Ok(None);
    };
    // Keep the leading mantissa_digits bits of the exact value (truncation
    // costs under one ulp, and the square root halves the relative error). The
    // exponent is made even so that sqrt(2^e) = 2^(e/2) is exact; an odd
    // exponent folds its extra factor of two into the significand.
    let retained_shift = (top_bit + 1).saturating_sub(S::MANTISSA_DIGITS as usize);
    let significand = shifted_low_u64(words, retained_shift);
    let significand = S::try_from_f64_exact(significand as f64)?;
    let normalized = significand.scalbn(-(S::MANTISSA_DIGITS as i32 - 1));
    let exponent = ACCUMULATOR_MIN_EXPONENT + top_bit as i32;
    Ok(Some(
        Scaled::from_parts(normalized, i64::from(exponent))?.sqrt()?,
    ))
}

/// Computes the roots of `a x² + b x + c = 0` for finite real coefficients
/// with `a ≠ 0`.
///
/// The discriminant `b² − 4ac` is computed in working precision when its sign
/// and error can be certified, and otherwise accumulated exactly, with an
/// exponent-scaled square root. For real roots, the larger-magnitude root comes
/// from `q = −(b + sign(b) √(b² − 4ac)) / 2` and the other from `c / q`, which
/// avoids cancellation when the roots differ widely in magnitude. Avoidable
/// overflow in intermediate quantities is avoided. The cost is `O(1)`.
///
/// Both returned roots are finite complex numbers. Real roots are in
/// increasing order, a complex-conjugate pair has the root with positive
/// imaginary part first, and a repeated root appears twice.
///
/// # Accuracy
///
/// The roots of a quadratic can be arbitrarily ill-conditioned, so the bound is
/// on the residual, not on the forward error. For each returned root `z`, let
///
/// ```text
/// x = |Re z| + |Im z|,   T = |a| x² + |b| x + |c|.
/// ```
///
/// The 1-norm of the residual `a z² + b z + c` is at most
///
/// ```text
/// 32 ε T + (2 |a| x + |b|) η + |a| η²,
/// ```
///
/// where `ε` is the machine epsilon and `η` the smallest positive subnormal.
/// The `η` terms account for rounding the root components, including subnormal
/// and zero results. An exactly zero component is `+0`; a nonzero component that
/// underflows keeps its sign.
///
/// # Errors
///
/// - [`NonFinite`](crate::SciOxideErrorKind::NonFinite) if a coefficient is NaN or
///   infinite.
/// - [`InvalidInput`](crate::SciOxideErrorKind::InvalidInput) (`OutOfRange`) if
///   `a` is zero.
/// - [`NumericalInstability`](crate::SciOxideErrorKind::NumericalInstability)
///   ([`Overflow`](crate::NumericalIssue::Overflow)) if a root is outside the
///   finite range.
///
/// # Examples
///
/// ```
/// use scioxide_core::{Complex64, checked_quadratic_roots};
///
/// // x² − 3x + 2 = (x − 1)(x − 2)
/// let roots = checked_quadratic_roots(1.0_f64, -3.0, 2.0)?;
/// assert_eq!(roots, [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)]);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline]
pub fn checked_quadratic_roots<S: RealScalar>(
    a: S,
    b: S,
    c: S,
) -> Result<[Complex<S>; 2], SciOxideError> {
    validate_finite(a, "quadratic leading coefficient must be finite")?;
    validate_finite(b, "quadratic linear coefficient must be finite")?;
    validate_finite(c, "quadratic constant coefficient must be finite")?;
    if a == S::ZERO {
        return Err(SciOxideError::invalid_input(
            crate::argument!("leading_coefficient"),
            InvalidInputReason::OutOfRange,
            "quadratic leading coefficient must be nonzero",
        ));
    }

    #[cfg(feature = "std")]
    if let Some(roots) = certified_quadratic_roots(a, b, c) {
        return Ok(roots);
    }
    exact_quadratic_roots(a, b, c)
}

// Absence means the working-precision certificate could not establish the
// contract; the caller then uses exact arithmetic. No numerical failure is
// turned into absence at the public boundary.
#[cfg(feature = "std")]
#[inline]
fn certified_quadratic_roots<S: RealScalar>(a: S, b: S, c: S) -> Option<[Complex<S>; 2]> {
    let (a, b, c) = (a.into_f64(), b.into_f64(), c.into_f64());
    let (lower, upper) = if S::MANTISSA_DIGITS == 24 {
        (5.960_464_477_539_063e-8, 16_777_216.0) // 2^±24
    } else {
        (6.223_015_277_861_142e-61, 1.606_938_044_258_990_3e60) // 2^±200
    };
    if ![a, b, c]
        .into_iter()
        .all(|x| x.abs() >= lower && x.abs() <= upper)
    {
        return None;
    }
    // All products and their FMA residuals are normal in these bounds.
    // TwoSum and each product+residual are exact. Only the three low-part
    // additions and the final reconstruction round. With e=2^-52, their
    // absolute error is below 8e²*(|p|+|q|)+e*|d|. Inflate the computed
    // bound for its own rounding. Near cancellation is deliberately excluded.
    let p = b * b;
    let four_a = 4.0 * a;
    let q = four_a * c;
    let (high, low) = two_sum(p, -q);
    let correction = (b.mul_add(b, -p) - four_a.mul_add(c, -q)) + low;
    let d = high + correction;
    let mass = p.abs() + q.abs();
    let e = f64::EPSILON;
    let bound = (8.0 * e * e * mass + e * d.abs()) * (1.0 + 8.0 * e);
    if d.abs() <= 64.0 * S::EPSILON.into_f64() * mass || bound > 4.0 * e * d.abs() {
        return None;
    }
    // Thus sign(d)=sign(D) and |d-D|/|D| < 5e. Correct rounding of sqrt,
    // same-sign q addition and the two divisions give relative component
    // errors below 8e, including destination rounding. Substitution in the
    // quadratic gives < 32*destination_epsilon*T. Non-normal projected
    // components use the exact path, which also owns zero/underflow signs.
    let square_root = d.abs().sqrt();
    let roots = if d < 0.0 {
        let real = (-0.5 * b) / a;
        let imaginary = (0.5 * square_root) / a.abs();
        [(real, imaginary), (real, -imaginary)]
    } else {
        let q = -0.5 * (b + square_root.copysign(b));
        let first = q / a;
        let second = c / q;
        if first <= second {
            [(first, 0.0), (second, 0.0)]
        } else {
            [(second, 0.0), (first, 0.0)]
        }
    };
    let project = |(re, im): (f64, f64)| {
        let re = S::try_from_f64_lossy(re).ok()?;
        let im = S::try_from_f64_lossy(im).ok()?;
        if re.abs() < S::MIN_POSITIVE || (d < 0.0 && im.abs() < S::MIN_POSITIVE) {
            return None;
        }
        Some(Complex::new(re, im))
    };
    Some([project(roots[0])?, project(roots[1])?])
}

fn exact_quadratic_roots<S: RealScalar>(
    a: S,
    b: S,
    c: S,
) -> Result<[Complex<S>; 2], SciOxideError> {
    // Subtracting a * c four times keeps every term an exact pair product.
    // Forming 4 * a * c first could overflow or round. The exact sign of the
    // discriminant then chooses real, repeated, or complex roots reliably even
    // when b^2 and 4ac nearly cancel.
    let mut discriminant = RangeAccumulator::new();
    discriminant.add_product(b, b)?;
    for _ in 0..4 {
        discriminant.subtract_product(a, c)?;
    }
    let (sign, magnitude) = discriminant.signed_magnitude();
    if sign == Ordering::Equal {
        let root = checked_product_quotient(-b, S::HALF, a)?;
        return Ok([Complex::new(root, S::ZERO); 2]);
    }

    let Some(square_root) = scaled_sqrt::<S>(&magnitude)? else {
        return Err(SciOxideError::numerical_instability(
            NumericalIssue::Breakdown,
            "nonzero discriminant lost its scaled square root",
        ));
    };
    if sign == Ordering::Less {
        let real = checked_product_quotient(-b, S::HALF, a)?;
        let imaginary = square_root
            .shifted(-1)?
            .div(Scaled::from_value(a.abs())?)?
            .round()?;
        return Ok([
            Complex::new(real, imaginary),
            Complex::new(real, -imaginary),
        ]);
    }

    if b == S::ZERO {
        let root = square_root
            .shifted(-1)?
            .div(Scaled::from_value(a)?)?
            .round()?;
        let (first, second) = if -root <= root {
            (-root, root)
        } else {
            (root, -root)
        };
        return Ok([Complex::new(first, S::ZERO), Complex::new(second, S::ZERO)]);
    };
    // q = -(b + sign(b) sqrt(disc)) / 2 adds two terms of the same sign, so it
    // has no cancellation. The roots are q / a and c / q, which avoids the
    // catastrophic cancellation of the textbook formula for the smaller root.
    let magnitude = Scaled::from_value(b.abs())?.add(square_root)?.shifted(-1)?;
    let q = if b.is_sign_negative() {
        magnitude
    } else {
        magnitude.neg()
    };
    let first = q.div(Scaled::from_value(a)?)?.round()?;
    let second = if c == S::ZERO {
        S::ZERO
    } else {
        Scaled::from_value(c)?.div(q)?.round()?
    };
    let (first, second) = if first <= second {
        (first, second)
    } else {
        (second, first)
    };
    Ok([Complex::new(first, S::ZERO), Complex::new(second, S::ZERO)])
}

#[cfg(test)]
mod tests {

    use super::*;

    #[cfg(feature = "std")]
    #[test]
    fn quadratic_certificate_accepts_ordinary_inputs_and_rejects_boundaries() {
        fn check<S: crate::RealScalar>() {
            let one = S::ONE;
            let two = one + one;
            let three = two + one;
            for (a, b, c) in [(one, -three, two), (-one, three, -two), (one, one, one)] {
                assert!(super::certified_quadratic_roots(a, b, c).is_some());
            }
            let tiny = S::ZERO.next_after(one);
            for (a, b, c) in [
                (one, two, one),
                (one, two, one.next_after(S::ZERO)),
                (one, two, one.next_after(S::INFINITY)),
                (S::MAX, S::MAX, S::MAX),
                (one, -S::MAX, one),
                (tiny, tiny, tiny),
                (one, one, tiny),
                (one, S::ZERO, one),
                (one, one, S::ZERO),
            ] {
                assert!(super::certified_quadratic_roots(a, b, c).is_none());
                let expected = super::exact_quadratic_roots(a, b, c).unwrap();
                let actual = super::checked_quadratic_roots(a, b, c).unwrap();
                assert_eq!(actual, expected);
            }
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn quadratic_roots_preserve_widely_separated_results() {
        let roots = checked_quadratic_roots(1.0_f64, -1.0e200, 1.0).unwrap();
        assert!(roots[0].re > 0.0);
        assert!((roots[0].re * roots[1].re - 1.0).abs() <= 4.0 * f64::EPSILON);
        assert_eq!(roots[0].im, 0.0);
        assert_eq!(roots[1].im, 0.0);
    }
}
