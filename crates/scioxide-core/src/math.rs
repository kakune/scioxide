//! Owned IEEE-754 bit operations used by SciOxide's scalar contracts.

#[cfg(not(feature = "std"))]
use crate::scalar::FiniteParts;

pub(crate) mod dispatch;
mod elementary;
mod lanes;
pub(crate) use elementary::IPIO2;

pub(crate) fn reduce_radians(value: f64) -> (i32, f64, f64) {
    if value.abs() <= core::f64::consts::FRAC_PI_4 {
        (0, value, 0.0)
    } else {
        // Ordinary medium reduction needs only enough tail bits to round a
        // trig value. Cancellation-sensitive consumers need the full tail.
        let exponent = ((value.to_bits() >> 52) & 0x7ff) as i32 - 1023 - 23;
        let mut z =
            f64::from_bits((value.to_bits() & ((1_u64 << 52) - 1)) | ((1023_u64 + 23) << 52));
        let mut limbs = [0.0; 3];
        for limb in &mut limbs {
            *limb = z as u32 as f64;
            z = (z - *limb) * 16_777_216.0;
        }
        let mut remainder = [0.0; 2];
        let quadrant = elementary::rem_pio2_large_precise(&limbs, &mut remainder, exponent);
        if value.is_sign_negative() {
            (-quadrant, -remainder[0], -remainder[1])
        } else {
            (quadrant, remainder[0], remainder[1])
        }
    }
}

// The positive logarithmic certificate needs the owned kernels' one-ULP
// bounds, independently of the native elementary dispatch's four-ULP envelope.
#[inline]
pub(crate) fn positive_exp(hi: f64, lo: f64) -> f64 {
    elementary::positive_exp(hi, lo)
}

#[inline]
pub(crate) fn positive_log1p(value: f64) -> f64 {
    elementary::log1p(value)
}

// pi minus its nearest binary64 value, rounded to binary64. A reduced input
// has |x| <= 1/4; callers handle tiny arguments before multiplying.
pub(crate) const PI_TAIL: f64 = 1.224_646_799_147_353_2e-16;

/// `sin(pi * (x + quadrant / 2))` for a reduced `|x| <= 1/4`.
///
/// Both kernels are evaluated and the quadrant selects one by bit masks, so
/// uniformly distributed inputs do not pay a mispredicted branch per element.
/// The selected value and its sign are bit-identical to a branching selection.
#[inline]
pub(crate) fn reduced_pi_sine(x: f64, quadrant: i32) -> f64 {
    let angle = x * core::f64::consts::PI;
    let tail = crate::RealScalar::mul_add(x, core::f64::consts::PI, -angle) + x * PI_TAIL;
    let sine = elementary::k_sin(angle, tail, 1).to_bits();
    let cosine = elementary::k_cos(angle, tail).to_bits();
    let odd = 0_u64.wrapping_sub((quadrant & 1) as u64);
    let negative = (((quadrant >> 1) & 1) as u64) << 63;
    f64::from_bits(((sine & !odd) | (cosine & odd)) ^ negative)
}

// Exact axis and infinity limits are a SciOxide contract, independent of the
// host libm's approximation envelope. Widening binary32 inputs is exact, and
// these binary64 constants round to the documented bits in either destination.
pub(crate) fn atan2_special(y: f64, x: f64) -> Option<f64> {
    use core::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};
    if y.is_nan() || x.is_nan() {
        Some(f64::NAN)
    } else if y == 0.0 {
        Some(if x.is_sign_negative() {
            PI.copysign(y)
        } else {
            y
        })
    } else if x == 0.0 {
        Some(FRAC_PI_2.copysign(y))
    } else if y.is_infinite() {
        let magnitude = if x.is_infinite() {
            if x.is_sign_negative() {
                3.0 * FRAC_PI_4
            } else {
                FRAC_PI_4
            }
        } else {
            FRAC_PI_2
        };
        Some(magnitude.copysign(y))
    } else if x.is_infinite() {
        Some(if x.is_sign_negative() { PI } else { 0.0 }.copysign(y))
    } else {
        None
    }
}

#[cfg(not(feature = "std"))]
fn trunc(value: f64) -> f64 {
    // Bit-level truncation: |value| < 1 becomes a zero of the same sign,
    // values with no fractional mantissa bits are already integral, and
    // otherwise the fractional mantissa bits are cleared.
    let exponent = ((value.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    if exponent < 0 {
        return f64::from_bits(value.to_bits() & (1_u64 << 63));
    }
    if exponent >= 52 {
        return value;
    }
    let fractional_mask = (1_u64 << (52 - exponent)) - 1;
    f64::from_bits(value.to_bits() & !fractional_mask)
}

#[cfg(not(feature = "std"))]
fn floor(value: f64) -> f64 {
    let integral = trunc(value);
    if value.is_sign_negative() && integral != value {
        integral - 1.0
    } else {
        integral
    }
}

#[cfg(not(feature = "std"))]
fn ceil(value: f64) -> f64 {
    let integral = trunc(value);
    if value.is_sign_positive() && integral != value {
        integral + 1.0
    } else {
        integral
    }
}

#[cfg(not(feature = "std"))]
fn round(value: f64) -> f64 {
    let integral = trunc(value);
    // Halfway cases round away from zero, matching `f64::round`. The
    // subtraction is exact: it yields exactly the fractional bits that
    // `trunc` discarded.
    if (value - integral).abs() >= 0.5 {
        integral + 1.0_f64.copysign(value)
    } else {
        integral
    }
}

// For n integer and q=floor(sqrt(n)), round up iff n-q*q > q:
// the midpoint square is q*q + q + 1/4 and cannot equal an integer.
#[cfg(not(feature = "std"))]
fn sqrt_rounded(value: f64, precision: u32) -> f64 {
    if value == 0.0 || value == f64::INFINITY {
        return value;
    }
    if value < 0.0 || value.is_nan() {
        return f64::NAN;
    }
    let Ok(parts) = FiniteParts::decode(value) else {
        return f64::NAN;
    };
    let significand = parts.significand();
    let exponent = parts.exponent();
    let top = 63 - significand.leading_zeros();
    let exponent = i32::from(exponent) + top as i32;
    // Scale the significand into an integer radicand of 2(p - 1) or 2(p - 1) + 1
    // bits, the extra bit making the remaining exponent even. Its integer
    // square root then has exactly p bits, and the exponent halves exactly.
    let shift = 2 * (precision - 1) + exponent.rem_euclid(2) as u32;
    // Widened f32 significands have enough trailing zeros for this shift.
    let radicand = if shift >= top {
        u128::from(significand) << (shift - top)
    } else {
        u128::from(significand >> (top - shift))
    };
    let root = radicand.isqrt();
    let rounded = root + u128::from(radicand - root * root > root);
    scalbn_f64(
        rounded as f64,
        exponent.div_euclid(2) - (precision - 1) as i32,
    )
}

/// Correctly rounded `mul_add` without `std`. A statically enabled scalar FMA
/// instruction (x86 `fma`, AArch64 `neon`) is used when the target provides
/// one; otherwise the owned exact emulation rounds once.
#[cfg(not(feature = "std"))]
pub(crate) mod fused {
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "fma"
    ))]
    #[allow(
        unsafe_code,
        reason = "the target-feature cfg proves the scalar FMA instruction"
    )]
    #[inline]
    pub(crate) fn binary32(left: f32, right: f32, addend: f32) -> f32 {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::{_mm_cvtss_f32, _mm_fmadd_ss, _mm_set_ss};
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{_mm_cvtss_f32, _mm_fmadd_ss, _mm_set_ss};
        // SAFETY: the enclosing cfg proves `fma` (and hence `sse2`) is
        // statically enabled; the intrinsics take values only.
        unsafe {
            _mm_cvtss_f32(_mm_fmadd_ss(
                _mm_set_ss(left),
                _mm_set_ss(right),
                _mm_set_ss(addend),
            ))
        }
    }

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "fma"
    ))]
    #[allow(
        unsafe_code,
        reason = "the target-feature cfg proves the scalar FMA instruction"
    )]
    #[inline]
    pub(crate) fn binary64(left: f64, right: f64, addend: f64) -> f64 {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::{_mm_cvtsd_f64, _mm_fmadd_sd, _mm_set_sd};
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{_mm_cvtsd_f64, _mm_fmadd_sd, _mm_set_sd};
        // SAFETY: the enclosing cfg proves `fma` (and hence `sse2`) is
        // statically enabled; the intrinsics take values only.
        unsafe {
            _mm_cvtsd_f64(_mm_fmadd_sd(
                _mm_set_sd(left),
                _mm_set_sd(right),
                _mm_set_sd(addend),
            ))
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[allow(
        unsafe_code,
        reason = "the target-feature cfg proves the scalar FMA instruction"
    )]
    #[inline]
    pub(crate) fn binary32(left: f32, right: f32, addend: f32) -> f32 {
        use core::arch::aarch64::{vdup_n_f32, vfma_f32, vget_lane_f32};
        // SAFETY: the enclosing cfg proves `neon` is statically enabled; the
        // intrinsics take values only.
        unsafe {
            vget_lane_f32::<0>(vfma_f32(
                vdup_n_f32(addend),
                vdup_n_f32(left),
                vdup_n_f32(right),
            ))
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[allow(
        unsafe_code,
        reason = "the target-feature cfg proves the scalar FMA instruction"
    )]
    #[inline]
    pub(crate) fn binary64(left: f64, right: f64, addend: f64) -> f64 {
        use core::arch::aarch64::{vdup_n_f64, vfma_f64, vget_lane_f64};
        // SAFETY: the enclosing cfg proves `neon` is statically enabled; the
        // intrinsics take values only.
        unsafe {
            vget_lane_f64::<0>(vfma_f64(
                vdup_n_f64(addend),
                vdup_n_f64(left),
                vdup_n_f64(right),
            ))
        }
    }

    #[cfg(not(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    pub(crate) use emulated::{binary32, binary64};

    /// Software `mul_add` with one rounding. Error-free transformations and
    /// Boldo--Melquiond round-to-odd handle finite operands away from the
    /// exponent extremes; the exact accumulator covers the remaining cases.
    #[cfg(not(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "fma"
        ),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    mod emulated {
        use crate::scalar::FiniteParts;

        /// Error-free transformations need binary64 rounding; x87 extended
        /// precision would round twice.
        const X87_SAFE: bool = cfg!(any(not(target_arch = "x86"), target_feature = "sse2"));

        /// Binary32 products are exact in binary64, whose round-to-odd sum has
        /// enough guard bits for one correct rounding to binary32.
        #[inline]
        pub(crate) fn binary32(left: f32, right: f32, addend: f32) -> f32 {
            if !(X87_SAFE
                && left.is_finite()
                && right.is_finite()
                && addend.is_finite()
                && left != 0.0
                && right != 0.0)
            {
                return exact(left, right, addend);
            }
            odd_sum(f64::from(left) * f64::from(right), f64::from(addend)) as f32
        }

        /// Dekker's exact product, TwoSum with the addend, then
        /// `RN(high + RO(low + product_low))`. The exponent window keeps the
        /// splits, the product error and both sums free of overflow and underflow.
        #[inline]
        pub(crate) fn binary64(left: f64, right: f64, addend: f64) -> f64 {
            let exponent = |value: f64| ((value.to_bits() >> 52) & 0x7ff) as i32 - 1023;
            let (a, b, c) = (exponent(left), exponent(right), exponent(addend));
            let window = |e: i32| (-900..=990).contains(&e);
            if !(X87_SAFE
                && left != 0.0
                && right != 0.0
                && window(a)
                && window(b)
                && (-850..=990).contains(&(a + b))
                && (addend == 0.0 || (-850..=1000).contains(&c)))
            {
                return exact(left, right, addend);
            }
            let (high, low) = two_product(left, right);
            let (sum, error) = two_sum(addend, high);
            sum + odd_sum(error, low)
        }

        #[inline]
        fn two_sum(left: f64, right: f64) -> (f64, f64) {
            let sum = left + right;
            let right_part = sum - left;
            (sum, (left - (sum - right_part)) + (right - right_part))
        }

        #[inline]
        fn two_product(left: f64, right: f64) -> (f64, f64) {
            let split = |value: f64| {
                let scaled = 134_217_729.0 * value;
                let high = scaled - (scaled - value);
                (high, value - high)
            };
            let product = left * right;
            let ((lh, ll), (rh, rl)) = (split(left), split(right));
            (product, ((lh * rh - product) + lh * rl + ll * rh) + ll * rl)
        }

        /// Round `left + right` to odd: an inexact nearest sum with an even last
        /// bit moves one unit toward the exact value.
        #[inline]
        fn odd_sum(left: f64, right: f64) -> f64 {
            let (sum, error) = two_sum(left, right);
            if error == 0.0 || sum.to_bits() & 1 == 1 {
                sum
            } else if (error > 0.0) == (sum > 0.0) {
                f64::from_bits(sum.to_bits() + 1)
            } else {
                f64::from_bits(sum.to_bits() - 1)
            }
        }

        fn exact<S: crate::RealScalar>(left: S, right: S, addend: S) -> S {
            use crate::FloatDecomposition::Nan;
            match (
                FiniteParts::decode(left),
                FiniteParts::decode(right),
                FiniteParts::decode(addend),
            ) {
                (Err(Nan), _, _) | (_, Err(Nan), _) | (_, _, Err(Nan)) => S::NAN,
                (Err(_), _, _) | (_, Err(_), _) => left * right + addend,
                (_, _, Err(_)) => addend,
                (Ok(left_parts), Ok(right_parts), Ok(addend_parts)) => {
                    // A zero factor keeps the raw IEEE sign; exact reductions canonicalize zero.
                    if left_parts.significand() == 0 || right_parts.significand() == 0 {
                        left * right + addend
                    } else {
                        crate::arithmetic::finite_fused_mul_add(
                            left_parts,
                            right_parts,
                            addend_parts,
                        )
                    }
                }
            }
        }

        #[cfg(test)]
        mod tests {
            use super::*;

            /// Deterministic operands spanning the fast-path window, both
            /// window edges and near-cancelling addends.
            fn operands(count: usize) -> impl Iterator<Item = (f64, f64, f64)> {
                let mut state = 0x9e37_79b9_7f4a_7c15_u64;
                let mut next = move || {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    state
                };
                (0..count).map(move |i| {
                    let mut value = |range: i64| {
                        let bits = next();
                        let exponent = (bits >> 53) as i64 % (2 * range + 1) - range;
                        let significand = f64::from_bits(0x3ff0_0000_0000_0000 | (bits >> 12));
                        let sign = if bits & 1 == 0 { 1.0 } else { -1.0 };
                        sign * crate::math::scalbn_f64(significand, exponent as i32)
                    };
                    let (left, right) = (value(460), value(460));
                    let addend = match i % 4 {
                        0 => value(1000),
                        1 => -(left * right),
                        2 => crate::math::next_after_f64(-(left * right), 0.0),
                        _ => -(left * right) * (1.0 + f64::EPSILON * value(4)),
                    };
                    (left, right, addend)
                })
            }

            #[test]
            fn fast_paths_match_the_exact_accumulator() {
                for (left, right, addend) in operands(200_000) {
                    let expected = exact(left, right, addend);
                    assert_eq!(
                        binary64(left, right, addend).to_bits(),
                        expected.to_bits(),
                        "{left:e} * {right:e} + {addend:e}"
                    );
                    let (left, right, addend) = (left as f32, right as f32, addend as f32);
                    let expected = exact(left, right, addend);
                    let actual = binary32(left, right, addend);
                    assert!(
                        actual.to_bits() == expected.to_bits()
                            || (actual.is_nan() && expected.is_nan()),
                        "{left:e} * {right:e} + {addend:e}"
                    );
                }
            }

            #[test]
            fn window_edges_ties_and_special_values_round_once() {
                let tiny = f64::from_bits(1);
                for (left, right, addend) in [
                    (2.0_f64.powi(990), 2.0, 1.0),
                    (2.0_f64.powi(-900), 2.0_f64.powi(-900), 1.0),
                    (1.0 + f64::EPSILON, 1.0 - f64::EPSILON, -1.0),
                    (3.0, 5.0, tiny),
                    (f64::MAX, 2.0, -f64::MAX),
                    (0.0, -1.0, -0.0),
                    (f64::INFINITY, 1.0, -f64::INFINITY),
                    (1.0, 1.0, f64::NAN),
                ] {
                    let expected = exact(left, right, addend);
                    let actual = binary64(left, right, addend);
                    assert!(
                        actual.to_bits() == expected.to_bits()
                            || (actual.is_nan() && expected.is_nan()),
                        "{left:e} * {right:e} + {addend:e}"
                    );
                }
                assert_eq!(
                    binary64(1.0 + f64::EPSILON, 1.0 - f64::EPSILON, -1.0),
                    -f64::EPSILON * f64::EPSILON
                );
                // Nearest-even partial sums would round these just-below-half
                // tails up a second time; round-to-odd keeps them below.
                let odd = 1.0 + f64::EPSILON;
                assert_eq!(
                    binary64(odd, (1.0 - f64::EPSILON) / 2.0_f64.powi(53), odd),
                    odd
                );
                let odd = 2.0_f32.powi(30) + 2.0_f32.powi(7);
                let (left, right) = (1.0 + f32::EPSILON, 64.0 * (1.0 - f32::EPSILON));
                assert_eq!(binary32(left, right, odd), odd);
                assert_eq!(exact(left, right, odd), odd);
            }
        }
    }
}

#[cfg(not(feature = "std"))]
fn cbrt(value: f64) -> f64 {
    let Ok(parts) = FiniteParts::decode(value) else {
        return value;
    };
    if parts.significand() == 0 {
        return value;
    }
    let significand = parts.significand();
    let exponent = parts.exponent();
    let top = 63 - significand.leading_zeros();
    let exponent = i32::from(exponent) + top as i32;
    // Fold exponent mod 3 into the mantissa so the remaining exponent divides
    // by 3 exactly. Newton's iteration r <- (2r + m / r^2) / 3 from r = 2
    // approaches the cube root in [1, 2) from above and converges
    // quadratically, so a fixed count of eight steps is ample.
    let mantissa = scalbn_f64(significand as f64, -(top as i32) + exponent.rem_euclid(3));
    // Normalize to [1, 8): iteration cannot overflow or underflow.
    let root = (0..8).fold(2.0, |root, _| (2.0 * root + mantissa / (root * root)) / 3.0);
    scalbn_f64(root, exponent.div_euclid(3)).copysign(value)
}

// A complex power's phase can slightly exceed the real scalar exponent range.
// Reduce its decoded significand directly; never materialize the overflowing phase.
pub(crate) fn scaled_sin_cos(significand: f64, exponent: i32) -> (f64, f64) {
    let rounded = scalbn_f64(significand, exponent);
    if rounded.is_finite() {
        return crate::RealScalar::sin_cos(rounded);
    }
    // The Payne-Hanek reducer takes the argument as 24-bit limbs plus the
    // exponent of the first limb. The at most 53-bit significand is split
    // exactly into three limbs. |x| is reduced, and the sign is restored at
    // the end because sine is odd and cosine is even.
    let top = 63 - (significand.abs() as u64).leading_zeros();
    let mut value = scalbn_f64(significand.abs(), 23 - top as i32);
    let mut limbs = [0.0; 3];
    for limb in &mut limbs {
        *limb = value as u32 as f64;
        value = (value - *limb) * 16_777_216.0;
    }
    let mut remainder = [0.0; 2];
    let quadrant = elementary::rem_pio2_large(&limbs, &mut remainder, exponent + top as i32 - 23);
    let sin = elementary::k_sin(remainder[0], remainder[1], 1);
    let cos = elementary::k_cos(remainder[0], remainder[1]);
    let (sin, cos) = match quadrant & 3 {
        0 => (sin, cos),
        1 => (cos, -sin),
        2 => (-sin, -cos),
        _ => (-cos, sin),
    };
    (
        if significand.is_sign_negative() {
            -sin
        } else {
            sin
        },
        cos,
    )
}

/// Return the adjacent `f32` value in the direction of `toward`.
#[must_use]
pub(crate) const fn next_after_f32(value: f32, toward: f32) -> f32 {
    if value.is_nan() || toward.is_nan() {
        return value + toward;
    }
    if value == toward {
        return toward;
    }
    if value == 0.0 {
        return f32::from_bits(if toward.is_sign_negative() {
            0x8000_0001
        } else {
            1
        });
    }

    let bits = value.to_bits();
    let increment = (toward > value) == value.is_sign_positive();
    f32::from_bits(if increment { bits + 1 } else { bits - 1 })
}

/// Return the adjacent `f64` value in the direction of `toward`.
#[must_use]
pub(crate) const fn next_after_f64(value: f64, toward: f64) -> f64 {
    if value.is_nan() || toward.is_nan() {
        return value + toward;
    }
    if value == toward {
        return toward;
    }
    if value == 0.0 {
        return f64::from_bits(if toward.is_sign_negative() {
            0x8000_0000_0000_0001
        } else {
            1
        });
    }

    let bits = value.to_bits();
    let increment = (toward > value) == value.is_sign_positive();
    f64::from_bits(if increment { bits + 1 } else { bits - 1 })
}

/// Decompose a finite `f32` as `significand * 2^exponent`.
#[must_use]
#[inline]
pub(crate) const fn decompose_f32(value: f32) -> crate::FloatDecomposition {
    let bits = value.to_bits();
    let negative = bits >> 31 != 0;
    let raw_exponent = ((bits >> 23) & 0xff) as i16;
    let fraction = (bits & 0x7f_ffff) as u64;
    match raw_exponent {
        0xff if fraction == 0 => crate::FloatDecomposition::Infinite { negative },
        0xff => crate::FloatDecomposition::Nan,
        0 => crate::FloatDecomposition::Finite {
            significand: fraction,
            exponent: -149,
            negative,
        },
        _ => crate::FloatDecomposition::Finite {
            significand: fraction | (1 << 23),
            exponent: raw_exponent - 127 - 23,
            negative,
        },
    }
}

/// Decompose a finite `f64` as `significand * 2^exponent`.
#[must_use]
#[inline]
pub(crate) const fn decompose_f64(value: f64) -> crate::FloatDecomposition {
    let bits = value.to_bits();
    let negative = bits >> 63 != 0;
    let raw_exponent = ((bits >> 52) & 0x7ff) as i16;
    let fraction = bits & 0x000f_ffff_ffff_ffff;
    match raw_exponent {
        0x7ff if fraction == 0 => crate::FloatDecomposition::Infinite { negative },
        0x7ff => crate::FloatDecomposition::Nan,
        0 => crate::FloatDecomposition::Finite {
            significand: fraction,
            exponent: -1074,
            negative,
        },
        _ => crate::FloatDecomposition::Finite {
            significand: fraction | (1_u64 << 52),
            exponent: raw_exponent - 1023 - 52,
            negative,
        },
    }
}

/// Multiply `value` by `2^exponent` with one nearest-even rounding.
#[must_use]
pub(crate) fn scalbn_f32(value: f32, exponent: i32) -> f32 {
    f32::from_bits(scale_bits(u64::from(value.to_bits()), exponent, 23, 127) as u32)
}

/// Multiply `value` by `2^exponent` with one nearest-even rounding.
#[must_use]
pub(crate) fn scalbn_f64(value: f64, exponent: i32) -> f64 {
    f64::from_bits(scale_bits(value.to_bits(), exponent, 52, 1023))
}

// Decode independently of floating arithmetic, move the binary exponent, and
// round directly on the final subnormal grid. Chunked multiplication can round
// twice when a penultimate chunk is already subnormal.
fn scale_bits(bits: u64, exponent: i32, fraction: u32, bias: i32) -> u64 {
    let hidden = 1_u64 << fraction;
    let infinity = ((2 * bias + 1) as u64) << fraction;
    let sign_mask = 1_u64 << (infinity.ilog2() + 1);
    let sign = bits & sign_mask;
    let magnitude = bits & (sign_mask - 1);
    if magnitude == 0 || magnitude >= infinity {
        return bits;
    }
    let raw_exponent = (magnitude >> fraction) as i32;
    let significand = if raw_exponent == 0 {
        magnitude
    } else {
        (magnitude & (hidden - 1)) | hidden
    };
    let binary_exponent = i64::from(if raw_exponent == 0 { 1 } else { raw_exponent })
        - i64::from(bias)
        - i64::from(fraction)
        + i64::from(exponent);
    let top = significand.ilog2();
    let destination_exponent = binary_exponent + i64::from(top);
    if destination_exponent > i64::from(bias) {
        return sign | infinity;
    }
    if destination_exponent >= i64::from(1 - bias) {
        return sign
            | ((destination_exponent + i64::from(bias)) as u64) << fraction
            | ((significand << (fraction - top)) & (hidden - 1));
    }
    let shift = i64::from(1 - bias) - i64::from(fraction) - binary_exponent;
    let rounded = if shift <= 0 {
        significand << shift.unsigned_abs()
    } else if shift >= 64 {
        0
    } else {
        let integer = significand >> shift;
        let remainder = significand & ((1_u64 << shift) - 1);
        let halfway = 1_u64 << (shift - 1);
        integer + u64::from(remainder > halfway || (remainder == halfway && integer & 1 != 0))
    };
    // A carry into the hidden bit is precisely the minimum normal encoding.
    sign | rounded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "std"))]
    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected}"
        );
    }

    #[test]
    fn next_after_crosses_signed_zero_one_subnormal_at_a_time() {
        assert_eq!(next_after_f64(0.0, -1.0).to_bits(), 0x8000_0000_0000_0001);
        assert_eq!(next_after_f64(-0.0, 1.0).to_bits(), 1);
        assert_eq!(next_after_f32(1.0, 2.0).to_bits(), 1.0_f32.to_bits() + 1);
        assert_eq!(next_after_f32(-0.0, -1.0).to_bits(), 0x8000_0001);
        assert!(next_after_f64(f64::NAN, 1.0).is_nan());
        assert_eq!(next_after_f64(f64::INFINITY, 0.0), f64::MAX);
    }

    #[test]
    fn decomposition_reconstructs_normal_and_subnormal_values() {
        for value in [
            1.0_f64,
            -3.5,
            f64::from_bits(1),
            f64::from_bits(0x000f_ffff_ffff_ffff),
            f64::MIN_POSITIVE,
            f64::MAX,
        ] {
            let crate::FloatDecomposition::Finite {
                significand,
                exponent,
                negative,
                ..
            } = decompose_f64(value)
            else {
                unreachable!("caller established a finite scalar before decomposition");
            };
            let sign = if negative { -1_i8 } else { 1_i8 };
            let reconstructed = sign as f64 * scalbn_f64(significand as f64, i32::from(exponent));
            assert_eq!(reconstructed.to_bits(), value.to_bits());
        }
        for value in [
            1.0_f32,
            -3.5,
            f32::from_bits(1),
            f32::from_bits(0x007f_ffff),
            f32::MIN_POSITIVE,
            f32::MAX,
        ] {
            let crate::FloatDecomposition::Finite {
                significand,
                exponent,
                negative,
                ..
            } = decompose_f32(value)
            else {
                unreachable!("caller established a finite scalar before decomposition");
            };
            let sign = if negative { -1_i8 } else { 1_i8 };
            let reconstructed = sign as f32 * scalbn_f32(significand as f32, i32::from(exponent));
            assert_eq!(reconstructed.to_bits(), value.to_bits());
        }
    }

    #[test]
    fn scalbn_observes_normal_subnormal_and_overflow_boundaries() {
        assert_eq!(scalbn_f64(1.0, -1074).to_bits(), 1);
        assert_eq!(scalbn_f64(1.0, -1075), 0.0);
        assert_eq!(scalbn_f64(1.0, 1024), f64::INFINITY);
        assert_eq!(scalbn_f32(1.0, -149).to_bits(), 1);
        assert_eq!(scalbn_f64(-0.0, i32::MAX).to_bits(), (-0.0_f64).to_bits());
        assert_eq!(scalbn_f64(-1.0, i32::MAX), f64::NEG_INFINITY);
        assert_eq!(scalbn_f64(-1.0, i32::MIN).to_bits(), (-0.0_f64).to_bits());
    }

    #[cfg(not(feature = "std"))]
    #[test]
    fn no_std_math_covers_the_sealed_real_contract() {
        close(
            crate::RealScalar::sqrt(2.0_f64),
            core::f64::consts::SQRT_2,
            2.0e-15,
        );
        close(
            crate::RealScalar::exp(1.0_f64),
            core::f64::consts::E,
            2.0e-15,
        );
        close(crate::RealScalar::ln(core::f64::consts::E), 1.0, 2.0e-15);
        close(
            crate::RealScalar::sin(core::f64::consts::FRAC_PI_6),
            0.5,
            2.0e-15,
        );
        close(
            crate::RealScalar::cos(core::f64::consts::FRAC_PI_3),
            0.5,
            2.0e-15,
        );
        close(
            crate::RealScalar::atan(1.0_f64),
            core::f64::consts::FRAC_PI_4,
            2.0e-15,
        );
        close(crate::RealScalar::cbrt(27.0_f64), 3.0, 2.0e-15);
        close(crate::RealScalar::hypot(3.0_f64, 4.0), 5.0, 2.0e-15);
        close(
            crate::RealScalar::asinh(-2.0_f64),
            -1.443_635_475_178_810_3,
            2.0e-15,
        );
        assert_eq!(
            crate::RealScalar::trunc(-0.25_f64).to_bits(),
            (-0.0_f64).to_bits()
        );
        assert!(crate::RealScalar::sqrt(-1.0_f64).is_nan());
    }

    #[cfg(not(feature = "std"))]
    #[test]
    fn no_std_elementary_math_matches_offline_high_precision_references() {
        // Rounded binary64 references generated with MPFR 4.2.2 at 256-bit
        // precision in round-to-nearest, ties-to-even mode.
        let observations = [
            (crate::RealScalar::exp(0.75_f64), 2.117_000_016_612_675),
            (crate::RealScalar::ln(1.5_f64), 0.405_465_108_108_164_4),
            (crate::RealScalar::sin(1.0_f64), 0.841_470_984_807_896_5),
            (crate::RealScalar::cos(1.0_f64), 0.540_302_305_868_139_8),
            (crate::RealScalar::atan(0.75_f64), 0.643_501_108_793_284_4),
            (crate::RealScalar::asinh(0.75_f64), core::f64::consts::LN_2),
        ];
        observations.into_iter().for_each(|(actual, expected)| {
            close(
                actual,
                expected,
                16.0 * f64::EPSILON * expected.abs().max(1.0),
            );
        });
    }
}
