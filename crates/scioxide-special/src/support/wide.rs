//! Double-word arithmetic for cancellation-sensitive special-function kernels.
//!
//! Products use a native FMA only when a std target guarantees hardware support
//! (AArch64 or an enabled FMA target feature). Other builds use Dekker splitting;
//! no software FMA is invoked in the ordinary product path. Splitting rescales
//! extreme operands; gradual underflow can round a residual below the f64 lattice.
//! Additions and full products renormalize the pair. Raw product/scale primitives
//! keep an unnormalized low word for compensated Horner and recurrence kernels.
//! Logarithms require positive finite inputs established by the calling kernel.

use scioxide_core::RealScalar;

use super::constants::LN_2;

/// Internal two-term arithmetic for cancellation-sensitive special kernels.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Wide {
    pub(crate) hi: f64,
    pub(crate) lo: f64,
}

/// Nonnegative square root with one compensated Newton correction.
#[inline]
pub(crate) fn square_root(x: Wide) -> Wide {
    if x.hi > 0.0 && x.hi < f64::MIN_POSITIVE / f64::EPSILON {
        // Keep the square's residual above the subnormal lattice before
        // Newton correction. Both binary rescalings are exact.
        let root = square_root(Wide {
            hi: x.hi.scalbn(512),
            lo: x.lo.scalbn(512),
        });
        return Wide {
            hi: root.hi.scalbn(-256),
            lo: root.lo.scalbn(-256),
        };
    }
    let root = x.hi.sqrt();
    if root == 0.0 || !root.is_finite() {
        return Wide::from(root);
    }
    let square = two_product(root, root);
    Wide {
        hi: root,
        lo: ((x.hi - square.hi) - square.lo + x.lo) / (2.0 * root),
    }
}

/// Quarter-turn reduction followed by a two-word Taylor sum.
/// Each series update is counted by the caller; large high components use
/// core's full-range reduction and compose the low phase independently.
pub(crate) fn sin_cos_with_step<E>(
    angle: Wide,
    step: &mut impl FnMut() -> Result<(), E>,
) -> Result<(Wide, Wide), E> {
    if angle.hi.abs() > 1e14 {
        let (s, c) = angle.hi.sin_cos();
        let (sl, cl) = sin_cos_with_step(Wide::from(angle.lo), step)?;
        return Ok((cl.mul_f64(s) + sl.mul_f64(c), cl.mul_f64(c) - sl.mul_f64(s)));
    }
    let quadrant = (angle.hi / core::f64::consts::FRAC_PI_2).round();
    let reduced = angle - super::constants::FRAC_PI_2.mul_f64(quadrant);
    let square = -(reduced * reduced);
    let (mut st, mut ct) = (reduced, Wide::from(1.0));
    let (mut sine, mut cosine) = (st, ct);
    for k in 1..20 {
        step()?;
        st = (st * square).div_f64(f64::from((2 * k) * (2 * k + 1)));
        ct = (ct * square).div_f64(f64::from((2 * k - 1) * (2 * k)));
        sine = sine + st;
        cosine = cosine + ct;
        if st.hi.abs() + ct.hi.abs() <= f64::EPSILON * f64::EPSILON / 64.0 {
            break;
        }
    }
    Ok(match (quadrant as i64).rem_euclid(4) {
        0 => (sine, cosine),
        1 => (cosine, -sine),
        2 => (-sine, -cosine),
        _ => (-cosine, sine),
    })
}

impl Wide {
    pub(crate) const ZERO: Self = Self { hi: 0.0, lo: 0.0 };

    #[inline]
    pub(crate) const fn from(value: f64) -> Self {
        Self { hi: value, lo: 0.0 }
    }

    /// The same phase kernel for fixed-cost callers without an iterative allowance.
    pub(crate) fn sin_cos(self) -> (Self, Self) {
        match sin_cos_with_step(self, &mut || Ok::<(), core::convert::Infallible>(())) {
            Ok(value) => value,
            Err(never) => match never {},
        }
    }

    #[inline]
    pub(crate) fn add(self, other: Self) -> Self {
        let hi = self.hi + other.hi;
        let residual = if self.hi.abs() >= other.hi.abs() {
            (self.hi - hi) + other.hi
        } else {
            (other.hi - hi) + self.hi
        };
        let lo = residual + self.lo + other.lo;
        let sum = hi + lo;
        Self {
            hi: sum,
            lo: (hi - sum) + lo,
        }
    }

    #[inline]
    pub(crate) fn mul(self, other: Self) -> Self {
        let hi = self.hi * other.hi;
        let lo = two_product(self.hi, other.hi).lo
            + self.hi * other.lo
            + self.lo * other.hi
            + self.lo * other.lo;
        let product = hi + lo;
        Self {
            hi: product,
            lo: (hi - product) + lo,
        }
    }

    #[inline]
    pub(crate) fn mul_f64(self, factor: f64) -> Self {
        let hi = self.hi * factor;
        let lo = two_product(self.hi, factor).lo + self.lo * factor;
        let product = hi + lo;
        Self {
            hi: product,
            lo: (hi - product) + lo,
        }
    }

    #[inline]
    pub(crate) fn div_f64(self, divisor: f64) -> Self {
        self / Self::from(divisor)
    }

    #[inline]
    pub(crate) fn value(self) -> f64 {
        self.hi + self.lo
    }

    /// Compensated logarithm of a positive finite value.
    #[inline]
    pub(crate) fn ln(self) -> Self {
        ln(self)
    }

    /// Preserve a displacement too small to survive addition to one.
    #[inline]
    pub(crate) fn ln_1p(self) -> Self {
        // The quadratic term is below the binary64 subnormal lattice here;
        // forming x/(2+x) would unnecessarily lose the smallest inputs.
        if self.hi.abs() < f64::MIN_POSITIVE {
            return self;
        }
        if self.hi.abs() >= 0.5 {
            return (Self::from(1.0) + self).ln();
        }
        let z = self / (Self::from(2.0) + self);
        let square = z * z;
        let mut term = z;
        let mut sum = z;
        for index in 1..36 {
            term = term * square;
            sum = sum + term.div_f64(f64::from(2 * index + 1));
            if term.hi.abs() <= 1.0e-34 * sum.hi.abs() {
                break;
            }
        }
        sum.mul_f64(2.0)
    }

    /// Correct a representable leading exponential with the low component.
    /// Products needing binary range recovery use [`project`] instead.
    pub(crate) fn exp(self) -> f64 {
        let high = self.hi.exp();
        if !high.is_finite() {
            return high;
        }
        scioxide_core::fast_mul_add(high, self.lo.exp_m1(), high)
    }

    /// Two-word exponential for cancellation-sensitive connection formulas.
    /// The reduced expm1 shares the same series as small complementary tails.
    #[inline]
    pub(crate) fn exp_wide(self) -> Self {
        let exponent = (self.hi * core::f64::consts::LOG2_E).round();
        if exponent.abs() > 2200.0 {
            return Self::from(if exponent < 0.0 { 0.0 } else { f64::INFINITY });
        }
        let reduced = self - LN_2.mul_f64(exponent);
        let value = Self::from(1.0) + reduced.exp_m1();
        Self {
            hi: value.hi.scalbn(exponent as i32),
            lo: value.lo.scalbn(exponent as i32),
        }
    }

    #[inline]
    pub(crate) fn exp_m1(self) -> Self {
        // Complementary log tails reach ln(2). Rounding exp(x) before
        // subtracting one would discard the residual needed by small-shape
        // inverses. On |x|<=1, 34 terms leave less than 1/35! < 1e-40.
        if self.hi.abs() > 1.0 {
            return Self::from(self.exp()) - Self::from(1.0);
        }
        let mut term = self;
        let mut sum = term;
        for index in 2..34 {
            term = (term * self).div_f64(f64::from(index));
            sum = sum + term;
            if term.hi.abs() <= 1.0e-34 * sum.hi.abs() {
                break;
            }
        }
        sum
    }

    #[inline]
    pub(crate) fn powu(self, mut power: usize) -> Self {
        let mut factor = self;
        let mut value = Self::from(1.0);
        while power != 0 {
            if power & 1 != 0 {
                value = value * factor;
            }
            power >>= 1;
            if power != 0 {
                factor = factor * factor;
            }
        }
        value
    }

    #[inline]
    pub(crate) fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}

impl core::ops::Add for Wide {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.add(rhs)
    }
}
impl core::ops::Sub for Wide {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self.add(rhs.neg())
    }
}
impl core::ops::Mul for Wide {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.mul(rhs)
    }
}
impl core::ops::Div for Wide {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let pair = quotient(self, rhs);
        let hi = pair.hi + pair.lo;
        Self {
            hi,
            lo: pair.lo - (hi - pair.hi),
        }
    }
}
impl core::ops::Neg for Wide {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self.neg()
    }
}

/// Compensated logarithm for gamma ratios and tail normalization. Range
/// reduction gives |z| <= 0.172; the degree-47 atanh remainder is < 2e-39.
/// Only the low-degree terms need two-component coefficients and arithmetic.
pub(crate) fn ln(x: Wide) -> Wide {
    let x = if x.hi == 0.0 { Wide::from(x.lo) } else { x };
    let (mantissa, exponent) = ln_parts(x);
    mantissa + LN_2.mul_f64(f64::from(exponent))
}

#[inline]
fn ln_parts(x: Wide) -> (Wide, i32) {
    let (_, mut exponent) = finite_parts(x.hi);
    let mut reduced = Wide {
        hi: x.hi.scalbn(-exponent),
        lo: x.lo.scalbn(-exponent),
    };
    if reduced.hi > core::f64::consts::SQRT_2 {
        reduced = reduced.mul_f64(0.5);
        exponent += 1;
    }
    let z = (reduced - Wide::from(1.0)) / (reduced + Wide::from(1.0));
    let square = z * z;
    (
        (z + z * square * atanh_odd_series(square)).mul_f64(2.0),
        exponent,
    )
}

/// `sum_(k>=0) s^k/(2k+3)` at `s = z^2`, so that `atanh z = z + z^3` times
/// this series. On `|z| <= 0.172` the degree-47 remainder is below 2e-39;
/// only the low-degree terms need two-component coefficients and arithmetic.
#[inline]
fn atanh_odd_series(square: Wide) -> Wide {
    const LOW: [(f64, f64); 12] = [
        (0.3333333333333333, 1.850371707708594e-17),
        (0.2, -1.1102230246251566e-17),
        (0.14285714285714285, 7.93016446160826e-18),
        (0.1111111111111111, 6.1679056923619804e-18),
        (0.09090909090909091, -2.523234146875356e-18),
        (0.07692307692307693, -4.270088556250602e-18),
        (0.06666666666666667, 9.251858538542971e-19),
        (0.058823529411764705, 8.163404592832033e-19),
        (0.05263157894736842, 2.921639538487254e-18),
        (0.047619047619047616, 2.64338815386942e-18),
        (0.043478260869565216, 1.206764157201257e-18),
        (0.04, -8.326672684688674e-19),
    ];
    const HIGH: [f64; 11] = [
        1.0 / 27.0,
        1.0 / 29.0,
        1.0 / 31.0,
        1.0 / 33.0,
        1.0 / 35.0,
        1.0 / 37.0,
        1.0 / 39.0,
        1.0 / 41.0,
        1.0 / 43.0,
        1.0 / 45.0,
        1.0 / 47.0,
    ];
    let high = HIGH.iter().rev().fold(0.0, |sum, &c| sum * square.hi + c);
    super::polynomial::compensated(square, &LOW, high)
}

/// `ln(1+sigma) - sigma` for `sigma > -1`, without cancelling its terms.
/// With `t = sigma/(2+sigma)`, `ln(1+sigma) = 2 atanh t` and `sigma - 2t =
/// sigma t` exactly, so the result `2t^3 S(t^2) - sigma t` loses no leading
/// digits on the reduced range `|t| <= 0.172`. Outside it the direct
/// difference loses at most three of the two-word bits.
pub(crate) fn log1pmx(sigma: Wide) -> Wide {
    let t = sigma / (Wide::from(2.0) + sigma);
    if t.hi.abs() > 0.172 {
        return ln(Wide::from(1.0) + sigma) - sigma;
    }
    let square = t * t;
    (t * square * atanh_odd_series(square)).mul_f64(2.0) - sigma * t
}

/// Veltkamp's splitting factor, 2^27 + 1.
const SPLIT: f64 = 134_217_729.0;

#[inline]
fn split(a: f64) -> (f64, f64) {
    let scaled = SPLIT * a;
    let hi = scaled - (scaled - a);
    (hi, a - hi)
}

/// Rounded product and its residual; no intermediate split overflow.
#[inline]
pub(crate) fn two_product(a: f64, b: f64) -> Wide {
    if scioxide_core::HARDWARE_FMA {
        let hi = a * b;
        Wide {
            hi,
            lo: a.mul_add(b, -hi),
        }
    } else {
        dekker_product(a, b)
    }
}

#[inline]
fn dekker_product(a: f64, b: f64) -> Wide {
    // Even the product of two one-ULP low parts must stay normal: a factor
    // sixteen above 2^-1022 / eps^2 = 2^-918 leaves room for binade endpoints
    // and the leading rounding. Below 2^996 < f64::MAX / SPLIT the splitter
    // cannot overflow. One unsigned exponent test per value keeps zeros,
    // non-finite values and extreme exponents off the split path.
    const LOW: u64 = 1023 - 914;
    const SPAN: u64 = 1023 + 995 - LOW;
    let regular = |x: f64| ((x.to_bits() >> 52) & 0x7ff).wrapping_sub(LOW) <= SPAN;
    let hi = a * b;
    if !(regular(a) & regular(b) & regular(hi)) {
        return extreme_product(a, b, hi);
    }
    let (ah, al) = split(a);
    let (bh, bl) = split(b);
    Wide {
        hi,
        lo: ((ah * bh - hi) + ah * bl + al * bh) + al * bl,
    }
}

/// Zeros, non-finite products and exponent extremes: balance the exponents
/// without overflowing the splitter or erasing a small operand.
#[cold]
fn extreme_product(a: f64, b: f64, hi: f64) -> Wide {
    if !hi.is_finite() || a == 0.0 || b == 0.0 {
        return Wide { hi, lo: 0.0 };
    }
    let (_, ae) = finite_parts(a.abs());
    let (_, be) = finite_parts(b.abs());
    let a = a.scalbn(-ae);
    let b = b.scalbn(-be);
    let p = dekker_product(a, b);
    let exponent = ae + be;
    Wide {
        hi,
        lo: ((p.hi - hi.scalbn(-exponent)) + p.lo).scalbn(exponent),
    }
}

/// a*b as an unnormalized pair.
#[inline]
pub(crate) fn product(a: Wide, b: Wide) -> Wide {
    let leading = two_product(a.hi, b.hi);
    Wide {
        hi: leading.hi,
        lo: leading.lo + (a.hi * b.lo + a.lo * b.hi),
    }
}

/// a*factor as an unnormalized pair.
#[inline]
pub(crate) fn scale(a: Wide, factor: f64) -> Wide {
    let leading = two_product(a.hi, factor);
    Wide {
        hi: leading.hi,
        lo: leading.lo + a.lo * factor,
    }
}

/// Renormalized sum.
#[inline]
pub(crate) fn sum(a: Wide, b: Wide) -> Wide {
    a + b
}
/// Renormalized difference.
#[inline]
pub(crate) fn difference(a: Wide, b: Wide) -> Wide {
    a - b
}
/// Quotient and its correction, unnormalized for recurrence kernels.
#[inline]
pub(crate) fn quotient(numerator: Wide, denominator: Wide) -> Wide {
    let inverse = 1.0 / denominator.hi;
    if !inverse.is_finite() {
        return quotient_extreme(numerator, denominator);
    }
    let hi = numerator.hi * inverse;
    let residual = if scioxide_core::HARDWARE_FMA {
        (-hi).mul_add(denominator.hi, numerator.hi)
    } else {
        let back = two_product(hi, denominator.hi);
        if back.hi.is_finite() {
            (numerator.hi - back.hi) - back.lo
        } else {
            // A rounded quotient can make its reverse product overflow even
            // when the numerator is finite. Core owns the cold fused fallback.
            RealScalar::mul_add(-hi, denominator.hi, numerator.hi)
        }
    };
    let lo = (residual + numerator.lo - hi * denominator.lo) * inverse;
    Wide { hi, lo }
}

#[cold]
fn quotient_extreme(numerator: Wide, denominator: Wide) -> Wide {
    // Scaling both operands avoids an overflowing reciprocal of a subnormal.
    if denominator.hi == 0.0 || !numerator.hi.scalbn(54).is_finite() {
        return Wide::from(numerator.hi / denominator.hi);
    }
    quotient(
        Wide {
            hi: numerator.hi.scalbn(54),
            lo: numerator.lo.scalbn(54),
        },
        Wide {
            hi: denominator.hi.scalbn(54),
            lo: denominator.lo.scalbn(54),
        },
    )
}
/// Exact positive finite decomposition x = significand * 2^exponent,
/// with 1 <= significand < 2, including subnormals. Callers validate the domain.
#[inline]
pub(crate) fn finite_parts(x: f64) -> (f64, i32) {
    let scioxide_core::FloatDecomposition::Finite {
        significand,
        exponent,
        ..
    } = x.decompose()
    else {
        unreachable!("caller established a positive finite nonzero argument");
    };
    let leading = significand.ilog2();
    let fraction = (significand << (52 - leading)) & ((1_u64 << 52) - 1);
    let mantissa = f64::from_bits((1023_u64 << 52) | fraction);
    (mantissa, i32::from(exponent) + leading as i32)
}

/// Project y*exp(log) with binary scaling, keeping the logarithm's low word.
#[inline]
pub(crate) fn project(value: Wide, log: Wide) -> f64 {
    if value.hi == 0.0 && value.lo == 0.0 {
        return value.hi;
    }
    if log.hi == 0.0 && log.lo == 0.0 {
        return value.value();
    }
    let value = if value.hi == 0.0 {
        Wide::from(value.lo)
    } else {
        value
    };
    let exponent = (log.hi * core::f64::consts::LOG2_E).round();
    if exponent.abs() > 2200.0 {
        return if exponent < 0.0 {
            0.0_f64
        } else {
            f64::INFINITY
        }
        .copysign(value.hi);
    }
    let residual = log - LN_2.mul_f64(exponent);
    let (value, shift) = if value.hi.abs() < f64::MIN_POSITIVE || value.hi.abs() > f64::MAX * 0.5 {
        let (_, shift) = finite_parts(value.hi.abs());
        (
            Wide {
                hi: value.hi.scalbn(-shift),
                lo: value.lo.scalbn(-shift),
            },
            shift,
        )
    } else {
        (value, 0)
    };
    let scaled = value.mul_f64(residual.exp());
    let exponent = exponent as i32 + shift;
    let high = scaled.hi.scalbn(exponent);
    // Once the leading word overflows, the smaller correction cannot restore
    // range. Scaling both first could form opposite infinities and return NaN.
    if high.is_infinite() {
        high
    } else {
        // A sum of opposite-signed zeros is +0; keep the sign of a value that
        // rounds to zero.
        (high + scaled.lo.scalbn(exponent)).copysign(scaled.hi)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::RangeAccumulator;

    #[test]
    fn products_retain_residuals_across_exponent_boundaries() {
        for (a, b) in [
            (1.0 + f64::EPSILON, 1.0 - f64::EPSILON),
            (f64::MAX, 0.25),
            (1.0e308, 1.0e-300),
            (f64::MIN_POSITIVE, 1.5),
            (1.0e-160, 1.0e-160),
            (1.0e-150, 1.0e-150),
            (f64::from_bits(3), 0.5),
            (f64::from_bits(1), 2.0_f64.powi(1000)),
        ] {
            let mut exact = RangeAccumulator::new();
            exact.add_product(a, b).unwrap();
            for p in [two_product(a, b), dekker_product(a, b)] {
                assert_eq!(p.hi.to_bits(), exact.finish().unwrap().to_bits());
                let mut residual = exact.clone();
                residual.add(-p.hi).unwrap();
                assert_eq!(p.lo, residual.finish().unwrap());
            }
        }
    }

    #[test]
    fn quotients_preserve_subnormal_denominators_and_low_words() {
        let tiny = f64::from_bits(1);
        for (a, b, expected) in [(3.0 * tiny, tiny, 3.0), (tiny, tiny, 1.0), (1.0, 2.0, 0.5)] {
            assert_eq!((Wide::from(a) / Wide::from(b)).value(), expected);
        }
        let numerator = Wide {
            hi: 1.0,
            lo: 2.0_f64.powi(-80),
        };
        let quotient = numerator / Wide::from(2.0);
        assert_eq!(quotient.hi, 0.5);
        assert_eq!(quotient.lo, numerator.lo * 0.5);
    }

    #[test]
    fn cancellation_and_elementary_boundaries() {
        let a = Wide {
            hi: 1.0,
            lo: 2.0_f64.powi(-80),
        };
        assert_eq!((a - Wide::from(1.0)).value(), a.lo);
        assert_eq!((a * Wide::from(1.0)).lo, a.lo);
        assert!((a.ln().value() - a.lo).abs() <= a.lo * f64::EPSILON);
        for x in [
            f64::from_bits(1),
            f64::MIN_POSITIVE,
            0.5,
            1.0,
            2.0,
            f64::MAX,
        ] {
            let (mantissa, exponent) = finite_parts(x);
            assert!((1.0..2.0).contains(&mantissa));
            assert_eq!(mantissa.scalbn(exponent), x);
            assert!(
                (Wide::from(x).ln().value() - x.ln()).abs() <= 2.0 * f64::EPSILON * x.ln().abs()
            );
        }
        for x in [-f64::from_bits(1), f64::from_bits(1), 2.0_f64.powi(-80)] {
            assert_eq!(Wide::from(x).ln_1p().value(), x);
            assert_eq!(Wide::from(x).exp_m1().value(), x);
        }
    }

    #[test]
    fn projection_preserves_range_and_zero_sign() {
        for sign in [-1.0, 1.0] {
            let value = Wide {
                hi: sign,
                lo: -sign * 1e-18,
            };
            assert_eq!(
                project(value, Wide::from(1000.0)),
                f64::INFINITY.copysign(sign)
            );
        }
        assert_eq!(
            project(Wide::from(-0.0), LN_2).to_bits(),
            (-0.0_f64).to_bits()
        );
        assert_eq!(
            project(Wide::from(3.0), LN_2.mul_f64(-1074.0)),
            f64::from_bits(3)
        );
        assert_eq!(project(Wide::from(1.5), LN_2.mul_f64(5.0)), 48.0);
        assert_eq!(
            project(
                Wide {
                    hi: 0.0,
                    lo: f64::from_bits(1)
                },
                LN_2
            ),
            f64::from_bits(2)
        );
        let reduced_max = project(Wide::from(f64::MAX), Wide::from(-0.5));
        let expected = f64::MAX * (-0.5_f64).exp();
        assert!((reduced_max - expected).abs() <= f64::EPSILON * expected);
        assert_eq!(LN_2.mul_f64(-1074.0).exp(), f64::from_bits(1));
        for (value, log, expected) in [
            (2.0_f64.powi(-1000), LN_2.mul_f64(1500.0), 2.0_f64.powi(500)),
            (
                2.0_f64.powi(1000),
                LN_2.mul_f64(-1500.0),
                2.0_f64.powi(-500),
            ),
        ] {
            assert!((project(Wide::from(value), log) - expected).abs() <= f64::EPSILON * expected);
        }
    }

    #[test]
    fn log1pmx_retains_two_words_across_its_reduction() {
        // 60-digit mpmath references ln(1+s)-s, split into two words.
        for (sigma, hi, lo) in [
            (1e-10, -4.999999999666667e-21, 1.126787430617783e-37),
            (0.003, -4.4910202015211885e-06, -1.2022250943755557e-22),
            (-0.003, -4.509020298721814e-06, 1.8065030618310428e-22),
            (0.3, -0.03763573553250894, -2.189469401213996e-18),
            (0.414, -0.06757743252561899, 5.0787159976969565e-18),
            (0.416, -0.06816400472847198, 1.2647300292757584e-18),
            (-0.292, -0.05331118528841729, 3.3992645005532678e-18),
            (-0.294, -0.054140041488894966, -2.361055148983988e-18),
            (-0.5, -0.19314718055994531, 4.565107477165917e-18),
            (-0.999, -5.908755278982136, 9.611535470209617e-17),
            (5.0, -3.208240530771945, 4.349979825096335e-17),
        ] {
            let value = log1pmx(Wide::from(sigma));
            let difference = (value.hi - hi) + (value.lo - lo);
            assert!(
                difference.abs() <= 1e-30 * hi.abs(),
                "{sigma}: {value:?} differs by {difference:e}"
            );
        }
        assert_eq!(log1pmx(Wide::ZERO).value(), 0.0);
    }
}
