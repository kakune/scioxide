//! Power kernel: `powf` as `exp(b·ln a)` with a double-length logarithm.

use super::exp::{LN2_HI, LN2_LO, exp_split};
use super::log::reduce;
use super::{fast_two_sum, two_product};

/// Largest exponent magnitude evaluated; Veltkamp splitting of `b` cannot
/// overflow below it.
pub(crate) const EXPONENT_LIMIT: f64 = f64::from_bits(0x5ff0_0000_0000_0000);

// 2/3 as a double-length constant.
const TWO_THIRDS: f64 = 2.0 / 3.0;
const TWO_THIRDS_LO: f64 = f64::from_bits(0x3c85_5555_5555_5555);

// Taylor coefficients 2/(2n + 1), n = 2…12, of
// atanh(s)·2 = 2s + (2/3)s³ + s⁵·U(s²).
const U0: f64 = 2.0 / 5.0;
const U1: f64 = 2.0 / 7.0;
const U2: f64 = 2.0 / 9.0;
const U3: f64 = 2.0 / 11.0;
const U4: f64 = 2.0 / 13.0;
const U5: f64 = 2.0 / 15.0;
const U6: f64 = 2.0 / 17.0;
const U7: f64 = 2.0 / 19.0;
const U8: f64 = 2.0 / 21.0;
const U9: f64 = 2.0 / 23.0;
const U10: f64 = 2.0 / 25.0;

/// `ln a` as an unevaluated sum `high + low` with relative error of about
/// `2⁻⁶³` for positive normal finite `a`.
///
/// `a = 2ᵏ·m` with `m ∈ [√½, √2)`, `s = (m − 1)/(m + 1)` to double length by
/// an exact Dekker remainder, and `ln a = k ln 2 + 2s + (2/3)s³ + s⁵·U(s²)`
/// with the first three terms in double-length arithmetic and `U` by
/// Estrin's scheme.
#[inline(always)]
fn ln_split(a: f64) -> (f64, f64) {
    let (k, m) = reduce(a);
    let f = m - 1.0;
    let d = 2.0 + f;
    let d_low = f - (d - 2.0);
    let inverse = 1.0 / d;
    let s = f * inverse;
    let (product, product_error) = two_product(s, d);
    let s_low = (((f - product) - product_error) - s * d_low) * inverse;
    let (square, square_error) = two_product(s, s);
    let (cube, cube_error) = two_product(square, s);
    let cube_low = cube_error + square_error * s + 3.0 * square * s_low;
    let (third, third_error) = two_product(TWO_THIRDS, cube);
    let third_low = third_error + TWO_THIRDS_LO * cube + TWO_THIRDS * cube_low;
    let z = s * s;
    let z2 = z * z;
    let z4 = z2 * z2;
    let u = ((U0 + z * U1) + z2 * (U2 + z * U3))
        + z4 * (((U4 + z * U5) + z2 * (U6 + z * U7)) + z4 * ((U8 + z * U9) + z2 * U10));
    let tail = cube * z * u;
    let (high, low) = fast_two_sum(k * LN2_HI, 2.0 * s);
    let (high, low_third) = fast_two_sum(high, third);
    fast_two_sum(
        high,
        low + low_third + (third_low + tail + 2.0 * s_low + k * LN2_LO),
    )
}

/// `aᵇ` for positive normal finite `a` and `|b| ≤ EXPONENT_LIMIT`.
///
/// `b·ln a` is formed as a Dekker product and its double-length sum enters
/// the exponential's reduction.
#[inline(always)]
pub(crate) fn powf(a: f64, b: f64) -> f64 {
    let (high, low) = ln_split(a);
    let (product, error) = two_product(b, high);
    exp_split(product, error + b * low)
}
