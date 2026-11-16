//! Exponential kernels: `exp`, `exp_m1` and `tanh`.

use super::{SIGN, blend, mask, round_to_integer, scale, select};
use core::f64::consts::LOG2_E;

/// Cody–Waite split of `ln 2`: the high part has 32 significant bits, so
/// `k·LN2_HI` is exact for `|k| < 2²¹`.
pub(super) const LN2_HI: f64 = f64::from_bits(0x3fe6_2e42_fee0_0000);
pub(super) const LN2_LO: f64 = f64::from_bits(0x3dea_39ef_3579_3c76);

/// Largest magnitude evaluated: `e^746` overflows and `e^−746` rounds to zero,
/// so larger magnitudes select `+∞` or the limit at `−∞`.
const LIMIT: f64 = 746.0;

/// Largest magnitude whose exponential is scaled in one step: for
/// `|x| ≤ 708`, `|round(x/ln 2)| ≤ 1021` and `eˣ` is normal.
pub(crate) const NORMAL_LIMIT: f64 = 708.0;

// Taylor coefficients 1/n! of the degree-13 exponential polynomial.
const C2: f64 = 1.0 / 2.0;
const C3: f64 = 1.0 / 6.0;
const C4: f64 = 1.0 / 24.0;
const C5: f64 = 1.0 / 120.0;
const C6: f64 = 1.0 / 720.0;
const C7: f64 = 1.0 / 5_040.0;
const C8: f64 = 1.0 / 40_320.0;
const C9: f64 = 1.0 / 362_880.0;
const C10: f64 = 1.0 / 3_628_800.0;
const C11: f64 = 1.0 / 39_916_800.0;
const C12: f64 = 1.0 / 479_001_600.0;
const C13: f64 = 1.0 / 6_227_020_800.0;

// fdlibm `expm1` rational kernel coefficients.
const Q1: f64 = f64::from_bits(0xbfa1_1111_1111_10f4);
const Q2: f64 = f64::from_bits(0x3f5a_01a0_19fe_5585);
const Q3: f64 = f64::from_bits(0xbf14_ce19_9eaa_dbb7);
const Q4: f64 = f64::from_bits(0x3ed0_cfca_86e6_5239);
const Q5: f64 = f64::from_bits(0xbe8a_fdb7_6e09_c32d);

/// `(m, k)` with `exp(high + low) = m·2ᵏ` for `|low| ≤ ulp(high)`, `k` as
/// two's-complement bits.
///
/// `k = round(high/ln 2)`, `r = high − k ln 2 + low` by a two-constant
/// Cody–Waite reduction and `m = e^r = 1 + (r + r²·P(r))` with the Taylor
/// polynomial of degree 13 (`P` by Estrin's scheme), so `m ∈ [√½, √2]` up to
/// rounding.
#[inline(always)]
fn reduced(high: f64, low: f64) -> (f64, u64) {
    let (k, bits) = round_to_integer(high * LOG2_E);
    let r = ((high - k * LN2_HI) - k * LN2_LO) + low;
    let r2 = r * r;
    let r4 = r2 * r2;
    let p0 = (C2 + r * C3) + r2 * (C4 + r * C5);
    let p1 = (C6 + r * C7) + r2 * (C8 + r * C9);
    let p2 = (C10 + r * C11) + r2 * (C12 + r * C13);
    let tail = (r + r2 * p0) + (r2 * r4) * (p1 + r4 * p2);
    (1.0 + tail, bits)
}

/// `exp(high + low)` for `|low| ≤ ulp(high)`.
///
/// The [`reduced`] value is scaled by `2ᵏ` in two steps, so a subnormal result
/// is rounded once. Beyond `±LIMIT` the result is selected, `+∞` or `+0`.
#[inline(always)]
pub(super) fn exp_split(high: f64, low: f64) -> f64 {
    let (m, bits) = reduced(high, low);
    select(
        high > LIMIT,
        f64::INFINITY,
        select(high < -LIMIT, 0.0, scale(m, bits)),
    )
}

/// `eˣ` for `|x| ≤ NORMAL_LIMIT`; other lanes are recomputed by the caller.
///
/// The result is normal, so `k` is added to the exponent field of the
/// [`reduced`] value: the bits of [`exp_full`] without its two products and
/// two selections.
#[inline(always)]
pub(crate) fn exp(x: f64) -> f64 {
    // Adding −0 leaves every r unchanged, so the addition folds away.
    let (m, bits) = reduced(x, -0.0);
    f64::from_bits(m.to_bits().wrapping_add(bits << 52))
}

/// `eˣ` for every `x` except NaN, for single lanes: the limits beyond `±LIMIT`
/// return at once, and every other argument has the bits of [`exp_split`].
pub(crate) fn exp_full(x: f64) -> f64 {
    if x > LIMIT {
        f64::INFINITY
    } else if x < -LIMIT {
        0.0
    } else {
        exp_split(x, -0.0)
    }
}

/// `eˣ − 1` for `|x| ≤ LIMIT` (fdlibm `expm1` with every case selected).
#[inline(always)]
fn exp_m1_reduced(x: f64) -> f64 {
    let (k, bits) = round_to_integer(x * LOG2_E);
    let high = x - k * LN2_HI;
    let low = k * LN2_LO;
    let r = high - low;
    let c = (high - r) - low;
    let hfx = 0.5 * r;
    let hxs = r * hfx;
    let r1 = 1.0 + hxs * (Q1 + hxs * (Q2 + hxs * (Q3 + hxs * (Q4 + hxs * Q5))));
    let t = 3.0 - r1 * hfx;
    let e = hxs * ((r1 - t) / (6.0 - r * t));
    let unreduced = r - (r * e - hxs);
    let e = (r * (e - c) - c) - hxs;
    let half = 0.5 * (r - e) - 0.5;
    let double = select(r < -0.25, -2.0 * (e - (r + 0.5)), 1.0 + 2.0 * (r - e));
    // 2ᵏ and 2⁻ᵏ are meaningful only for 2 ≤ k ≤ 56, where they are selected.
    let power = f64::from_bits(bits.wrapping_add(1023) << 52);
    let reciprocal = f64::from_bits(1023_u64.wrapping_sub(bits) << 52);
    let near = (r - e + (1.0 - reciprocal)) * power;
    let far = (r - (e + reciprocal) + 1.0) * power;
    let wide = scale(r - e + 1.0, bits) - 1.0;
    // k = 0, −1 and 1 have their own forms; 2 ≤ k < 20 and 20 ≤ k ≤ 56 scale
    // by 2ᵏ, and every other k uses the wide form.
    let scaled = select(k > 56.0, wide, select(k < 20.0, near, far));
    select(
        k == 0.0,
        unreduced,
        select(
            k == -1.0,
            half,
            select(k == 1.0, double, select(k < -1.0, wide, scaled)),
        ),
    )
}

/// `eˣ − 1`; NaN lanes are recomputed by the caller.
#[inline(always)]
pub(crate) fn exp_m1(x: f64) -> f64 {
    select(
        x > LIMIT,
        f64::INFINITY,
        select(x < -LIMIT, -1.0, exp_m1_reduced(x)),
    )
}

/// `tanh x` through `exp_m1` with the fdlibm case split; NaN lanes are
/// recomputed by the caller.
#[inline(always)]
pub(crate) fn tanh(x: f64) -> f64 {
    let magnitude = f64::from_bits(x.to_bits() & !SIGN);
    // tanh rounds to one beyond 19.1, so |x| > 20 and infinity use 20.
    let a = select(magnitude < 20.0, magnitude, 20.0);
    // |x| > ln(3)/2 and |x| ≤ ln(5/3)/2, as fdlibm's high-word tests; the
    // small case negates the argument of expm1 and the quotient.
    let large = mask(a >= f64::from_bits(0x3fe1_93eb_0000_0000));
    let small = mask(a < f64::from_bits(0x3fd0_58af_0000_0000)) & SIGN;
    let e = exp_m1_reduced(f64::from_bits((2.0 * a).to_bits() ^ small));
    let quotient = blend(large, 2.0, e) / (e + 2.0);
    let t = blend(
        large,
        1.0 - quotient,
        f64::from_bits(quotient.to_bits() ^ small),
    );
    f64::from_bits(t.to_bits() | (x.to_bits() & SIGN))
}

#[cfg(test)]
mod tests {
    use core::f64::consts::LN_2;

    use super::{NORMAL_LIMIT, exp, exp_full};

    /// `x` and its neighbours within three bit steps, kept to the one-step range.
    fn around(x: f64) -> impl Iterator<Item = f64> {
        (-3_i64..=3)
            .map(move |step| f64::from_bits(x.to_bits().wrapping_add_signed(step)))
            .filter(|x| x.abs() <= NORMAL_LIMIT)
    }

    #[test]
    fn one_step_scaling_has_the_bits_of_two_step_scaling() {
        // Both ends of the range, every change of k, a dense sweep, and arguments
        // whose reduction is the identity.
        let ends = [NORMAL_LIMIT, -NORMAL_LIMIT];
        let switches = (0..=1021).flat_map(|k| {
            let x = (f64::from(k) + 0.5) * LN_2;
            [x, -x]
        });
        let sweep = (0..=400_000).map(|i| -NORMAL_LIMIT + f64::from(i) * 3.54e-3);
        let tiny = [0.0, -0.0, f64::MIN_POSITIVE, 5e-324, 1e-300, 5.5e-17, 1e-9];
        let tiny = tiny.into_iter().flat_map(|x| [x, -x]);
        let points = ends.into_iter().chain(switches).chain(sweep).chain(tiny);
        let mut checked = 0_u32;
        for x in points.flat_map(around) {
            let (one, two) = (exp(x), exp_full(x));
            assert!(one.is_normal(), "exp({x:e}) = {one:e}");
            assert_eq!(one.to_bits(), two.to_bits(), "exp({x:e})");
            checked += 1;
        }
        assert!(checked > 2_800_000);
    }

    #[test]
    fn the_full_range_form_returns_the_limits() {
        let below = f64::from_bits(NORMAL_LIMIT.to_bits() + 1);
        assert!(exp_full(below).is_normal() && exp_full(-below).is_normal());
        for above in [710.0, 746.0, 747.0, f64::MAX, f64::INFINITY] {
            assert_eq!(exp_full(above), f64::INFINITY, "exp({above:e})");
        }
        for below in [-746.0, -747.0, f64::MIN, f64::NEG_INFINITY] {
            assert_eq!(exp_full(below).to_bits(), 0, "exp({below:e})");
        }
        assert!(exp_full(-740.0) > 0.0 && !exp_full(-740.0).is_normal());
    }
}
