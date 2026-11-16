//! Binary64 kernels of the Faddeeva function on the coordinate axes.
//!
//! The scaled complementary error function `erfcx(y) = w(iy) = exp(y²) erfc(y)`
//! and Dawson's integral `F(x) = (√π/2) Im w(x) = exp(−x²) ∫₀ˣ exp(t²) dt` are
//! the only real erfcx and Dawson kernels in the crate. erfc tails, erfcx,
//! log erfc, the Gaussian tails, Dawson, Owen T and Faddeeva on the axes all
//! use them. Both kernels have fixed cost and stay within two ulps of the
//! checked-in reference values.
//!
//! # Algorithms
//!
//! - **Bins.** On `[1, 32)` (erfcx) and `[1/2, 8)` (Dawson) each binade has
//!   sixteen bins, keyed by the exponent and four leading mantissa bits. The
//!   local coordinate `t = x − centre` is exact (Sterbenz). Each bin holds a
//!   degree-9 Chebyshev interpolant in monomial form. Its value at the centre
//!   is kept in two parts, so the result has a single final rounding.
//! - **erfcx** uses 32 uniform bins on `[−1, 1)`. Beyond 32 it uses
//!   `erfcx(x) = (1 + u S(u))/(√π x)` with `u = x⁻²`, through a compensated
//!   division. Below −1 it uses `2 exp(x²) − erfcx(−x)`, with the square
//!   formed exactly. That reflection overflows to `+∞` once `2 exp(x²)` exceeds
//!   the `f64` range.
//! - **Dawson** is odd. On `|x| < 1/2` it uses `F = x (1 + v Q(v))` with
//!   `v = x²`, and beyond 8 it uses `F = (1 + u T(u))/(2x)`.
//!
//! The asymptotic forms also give the slopes `erfcx′ = 2u S(u)/√π` and
//! `F′ = −u T(u)` without the cancellation in `2x·erfcx(x) − 2/√π` and
//! `1 − 2x F(x)`. The tables come from
//! `generate_real_faddeeva.py`.

use crate::error_probability::real_faddeeva_coefficients::{
    Bin, DAWSON_ASYMPTOTIC, DAWSON_BINADES, DAWSON_CENTRE, ERFCX_ASYMPTOTIC, ERFCX_BINADES,
    ERFCX_CENTRE,
};
use crate::support::polynomial::{PolynomialPolicy, polynomial};
use crate::support::wide::two_product;
use core::f64::consts::FRAC_2_SQRT_PI;
#[cfg(all(not(feature = "std"), not(test)))]
#[allow(
    unused_imports,
    reason = "required when elementary methods use the no_std scalar trait"
)]
use scioxide_core::RealScalar;

/// Starts of the asymptotic forms. erfc's tail and the normal probabilities
/// stay within the erfcx bins.
const ERFCX_ASYMPTOTIC_START: f64 = 32.0;
const DAWSON_ASYMPTOTIC_START: f64 = 8.0;
/// `1/√π` in two parts.
const FRAC_1_SQRT_PI: (f64, f64) = (0.564_189_583_547_756_3, 7.667_729_806_582_94e-18);
/// Binade keys, `bits >> 48`, of 1 and 1/2.
const ONE: u64 = 0x3FF0;
const HALF: u64 = 0x3FE0;

/// The bin of `x` among sixteen per binade, counted from the binade with key
/// `first`, and `x − centre`. The centre shares the binade and leading bits
/// of `x`, so the difference is exact.
#[inline]
fn binade(x: f64, first: u64, bins: &[Bin]) -> (&Bin, f64) {
    let key = x.to_bits() >> 48;
    let centre = f64::from_bits((key << 48) | (1 << 47));
    (&bins[(key - first) as usize], x - centre)
}

/// A bin's interpolant at `t`. The remaining terms are small beside the
/// two-part constant, which absorbs their rounding.
#[inline(always)]
fn interpolate((constant, rest): &Bin, t: f64) -> f64 {
    constant[0] + (constant[1] + t * polynomial(t, rest, PolynomialPolicy::Grouped))
}

/// `u p(u)` with `u = x⁻²` for `inverse = 1/x`: the relative correction of an
/// asymptotic form.
#[inline]
fn correction<const N: usize>(inverse: f64, p: &[f64; N]) -> f64 {
    let u = inverse * inverse;
    u * polynomial(u, p, PolynomialPolicy::Grouped)
}

/// The asymptotic form `a (1 + u p(u))/x` for a two-part `a`. One reciprocal
/// serves both terms, and the exact remainder of the leading quotient
/// (Dekker) restores a single final rounding.
#[inline]
fn asymptotic<const N: usize>((hi, lo): (f64, f64), p: &[f64; N], x: f64) -> f64 {
    let inverse = 1.0 / x;
    let quotient = hi * inverse;
    let product = two_product(quotient, x);
    let remainder = (hi - product.hi) - product.lo;
    quotient + (remainder + (lo + hi * correction(inverse, p))) * inverse
}

/// `exp(x²) erfc(x)` for finite `x`; `+∞` where `2 exp(x²)` overflows.
pub(crate) fn erfcx(x: f64) -> f64 {
    if x >= ERFCX_ASYMPTOTIC_START {
        asymptotic(FRAC_1_SQRT_PI, &ERFCX_ASYMPTOTIC, x)
    } else if x >= 1.0 {
        let (bin, t) = binade(x, ONE, &ERFCX_BINADES);
        interpolate(bin, t)
    } else if x >= -1.0 {
        // Width-1/16 bins. An index rounded up as x approaches 1 extrapolates
        // the last bin by an ulp, which its fit interval covers.
        let index = (((x + 1.0) * 16.0) as usize).min(ERFCX_CENTRE.len() - 1);
        let centre = (2 * index + 1) as f64 / 32.0 - 1.0;
        interpolate(&ERFCX_CENTRE[index], x - centre)
    } else {
        // 2 exp(x²) − erfcx(−x), with x² = s + e exactly.
        let s = x * x;
        let twice = 2.0 * s.exp();
        if twice == f64::INFINITY {
            return twice;
        }
        twice + (twice * two_product(x, x).lo - erfcx(-x))
    }
}

/// `erfcx′(x) = 2x erfcx(x) − 2/√π` and its error bound, given
/// `value = erfcx(x)` within `error`.
pub(crate) fn erfcx_slope(x: f64, value: f64, error: f64) -> (f64, f64) {
    if x >= ERFCX_ASYMPTOTIC_START {
        let slope = FRAC_2_SQRT_PI * correction(x.recip(), &ERFCX_ASYMPTOTIC);
        (slope, 4.0 * f64::EPSILON * slope.abs())
    } else {
        let product = 2.0 * x * value;
        (
            product - FRAC_2_SQRT_PI,
            2.0 * x.abs() * error + 4.0 * f64::EPSILON * (product.abs() + FRAC_2_SQRT_PI),
        )
    }
}

/// Dawson's integral `exp(−x²) ∫₀ˣ exp(t²) dt` for finite `x`.
pub(crate) fn dawson(x: f64) -> f64 {
    let a = x.abs();
    let value = if a >= DAWSON_ASYMPTOTIC_START {
        asymptotic((0.5, 0.0), &DAWSON_ASYMPTOTIC, a)
    } else if a >= 0.5 {
        let (bin, t) = binade(a, HALF, &DAWSON_BINADES);
        interpolate(bin, t)
    } else {
        let v = a * a;
        a + a * (v * polynomial(v, &DAWSON_CENTRE, PolynomialPolicy::Grouped))
    };
    value.copysign(x)
}

/// `F′(x) = 1 − 2x F(x)` and its error bound, given `value = F(x)` within
/// `error`.
pub(crate) fn dawson_slope(x: f64, value: f64, error: f64) -> (f64, f64) {
    if x.abs() >= DAWSON_ASYMPTOTIC_START {
        let slope = -correction(x.abs().recip(), &DAWSON_ASYMPTOTIC);
        (slope, 4.0 * f64::EPSILON * slope.abs())
    } else {
        let product = 2.0 * x * value;
        (
            1.0 - product,
            2.0 * x.abs() * error + 4.0 * f64::EPSILON * (1.0 + product.abs()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every cutover and bin edge of both kernels.
    fn edges() -> impl Iterator<Item = f64> {
        let centre = (0..=32).map(|k| f64::from(k) / 16.0 - 1.0);
        let binades = (-1..5)
            .flat_map(|e: i32| (0..16).map(move |k| 2.0_f64.powi(e) * (1.0 + f64::from(k) / 16.0)));
        centre.chain(binades).chain([32.0])
    }

    fn close(a: f64, b: f64, relative: f64) -> bool {
        (a - b).abs() <= relative * a.abs().max(b.abs())
    }

    #[test]
    fn continuous_and_monotone_across_edges() {
        for edge in edges() {
            let (below, at, above) = (edge.next_down(), edge, edge.next_up());
            let [l, m, h] = [below, at, above].map(erfcx);
            assert!(l >= m && m >= h, "erfcx not decreasing at {edge}");
            assert!(close(l, m, 4.0 * f64::EPSILON), "erfcx jumps at {edge}");
            if edge > 0.0 && edge <= 8.0 {
                let [l, m, h] = [below, at, above].map(dawson);
                assert!(close(l, m, 4.0 * f64::EPSILON), "dawson jumps at {edge}");
                assert!(close(m, h, 4.0 * f64::EPSILON), "dawson jumps at {edge}");
            }
        }
        // The reflection joins the centre bins at -1.
        let [l, m] = [(-1.0_f64).next_down(), -1.0].map(erfcx);
        assert!(l > m && close(l, m, 8.0 * f64::EPSILON));
    }

    #[test]
    fn limits_and_symmetry() {
        assert!((erfcx(0.0) - 1.0).abs() <= f64::EPSILON);
        assert!(erfcx(-26.6).is_finite());
        assert_eq!(erfcx(-26.65), f64::INFINITY);
        assert_eq!(erfcx(-1e300), f64::INFINITY);
        // erfcx(x) → 1/(√π x): subnormal at the top of the range, never zero.
        let top = erfcx(f64::MAX);
        assert!(top > 0.0 && (top - FRAC_1_SQRT_PI.0 / f64::MAX).abs() <= 4.0 * f64::from_bits(1));
        assert_eq!(dawson(0.0).to_bits(), 0.0_f64.to_bits());
        assert_eq!(dawson(-0.0).to_bits(), (-0.0_f64).to_bits());
        let tiny = f64::from_bits(1);
        assert_eq!(dawson(tiny), tiny);
        assert!(dawson(f64::MAX) > 0.0);
        for x in [1e-200, 0.3, 0.75, 1.5, 3.0, 7.9, 8.0, 40.0, 1e8] {
            assert_eq!(dawson(-x).to_bits(), (-dawson(x)).to_bits());
        }
    }

    #[test]
    fn slopes_match_central_differences() {
        for x in [
            -3.0_f64, -0.6, 0.2, 0.9, 1.7, 5.3, 7.99, 8.0, 12.0, 31.9, 32.0, 300.0,
        ] {
            let h = 1e-5 * x.abs().max(1.0);
            let erfcx_difference = (erfcx(x + h) - erfcx(x - h)) / (2.0 * h);
            let (slope, _) = erfcx_slope(x, erfcx(x), 0.0);
            assert!(close(slope, erfcx_difference, 1e-8), "erfcx' at {x}");
            let dawson_difference = (dawson(x + h) - dawson(x - h)) / (2.0 * h);
            let (slope, _) = dawson_slope(x, dawson(x), 0.0);
            assert!(close(slope, dawson_difference, 1e-7), "F' at {x}");
        }
    }
}
