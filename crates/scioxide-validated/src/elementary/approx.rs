//! Two-component approximations with explicit absolute error bounds.
//!
//! TwoSum is exact (including gradual underflow). TwoProduct is used only
//! below the splitter overflow threshold; outside its exact-product range the
//! same algebra has an additional absolute underflow allowance. Every bound
//! operation is inflated by 16 epsilon and one minimum subnormal, covering
//! the at most eight positive operations used to propagate an error.
use scioxide_core::{
    RealScalar,
    backend::{two_product, two_sum},
};

#[derive(Clone, Copy, Debug)]
pub(super) struct A {
    pub hi: f64,
    pub lo: f64,
    pub error: f64,
}

const ETA: f64 = f64::from_bits(1);
const U2: f64 = 0.0000000000000000000000000000031554436208840472; // 2^-98

#[inline]
fn inflate(value: f64) -> f64 {
    value * (1.0 + 16.0 * f64::EPSILON) + ETA
}

/// A scalar polynomial tail with a proved roundoff allowance. Callers bound
/// coefficient rounding and Horner evaluation by gamma_k times the absolute
/// polynomial mass. A derivative bound accounts for the omitted input low
/// word and its uncertainty. The leading coefficients are evaluated separately
/// with two-component arithmetic, so this tail's error is attenuated by powers
/// of the reduced argument before reconstruction.
pub(super) fn polynomial_tail(
    x: A,
    coefficients: impl Iterator<Item = A>,
    roundoff: f64,
    derivative: f64,
) -> A {
    let hi = coefficients.fold(0.0, |p, c| p * x.hi + c.hi);
    A {
        hi,
        lo: 0.0,
        error: inflate(roundoff + derivative * (x.lo.abs() + x.error)),
    }
}

impl A {
    pub(crate) const ZERO: Self = Self::point(0.0);
    pub(crate) const ONE: Self = Self::point(1.0);
    #[inline]
    pub(crate) const fn point(hi: f64) -> Self {
        Self {
            hi,
            lo: 0.0,
            error: 0.0,
        }
    }
    /// Constants are supplied as nearest high/low words. The omitted tail is
    /// bounded by one ulp of the low word, independent of its sign.
    #[inline]
    pub(crate) const fn constant(hi: f64, lo: f64) -> Self {
        Self {
            hi,
            lo,
            error: if lo < 0.0 {
                -lo * f64::EPSILON
            } else {
                lo * f64::EPSILON
            },
        }
    }
    #[inline]
    pub(crate) fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
            error: self.error,
        }
    }
    #[inline]
    pub(crate) fn mass(self) -> f64 {
        inflate(self.hi.abs() + self.lo.abs() + self.error)
    }
    #[inline]
    pub(crate) fn add(self, rhs: Self) -> Self {
        let (s, e) = two_sum(self.hi, rhs.hi);
        let t = self.lo + rhs.lo;
        let (v, f) = two_sum(e, t);
        let (h, l) = two_sum(s, v);
        let low = l + f;
        let (hi, lo) = two_sum(h, low);
        Self {
            hi,
            lo,
            error: inflate(
                self.error + rhs.error + f64::EPSILON * (t.abs() + low.abs()) + 2.0 * ETA,
            ),
        }
    }
    #[inline]
    pub(crate) fn sub(self, rhs: Self) -> Self {
        self.add(rhs.neg())
    }
    #[inline]
    pub(crate) fn mul(self, rhs: Self) -> Self {
        let (p, e) = two_product(self.hi, rhs.hi);
        let cross = self.hi * rhs.lo + self.lo * rhs.hi;
        let low = (e + cross) + self.lo * rhs.lo;
        let (hi, lo) = two_sum(p, low);
        // For normalized pairs, the discarded cross-product rounding is
        // bounded by 2^-98 |hi1 hi2| (256 u²). 32 eta covers every
        // underflowing split, cross product and addition.
        let error =
            inflate(self.mass() * rhs.error + rhs.mass() * self.error + U2 * p.abs() + 32.0 * ETA);
        Self { hi, lo, error }
    }
    /// Compensated product plus addend with one final normalization. The
    /// exact high-product residual and TwoSum carry are kept; low-part
    /// rounding is bounded by 256 u² times the high operand mass and 48 eta.
    #[inline]
    pub(crate) fn mul_add(self, rhs: Self, addend: Self) -> Self {
        let (product, residual) = two_product(self.hi, rhs.hi);
        let (sum, carry) = two_sum(product, addend.hi);
        let cross = self.hi * rhs.lo + self.lo * rhs.hi;
        let low = ((carry + residual) + cross) + (self.lo * rhs.lo + addend.lo);
        let (hi, lo) = two_sum(sum, low);
        let error = inflate(
            self.mass() * rhs.error
                + rhs.mass() * self.error
                + addend.error
                + U2 * (product.abs() + addend.hi.abs())
                + 48.0 * ETA,
        );
        Self { hi, lo, error }
    }
    #[inline]
    pub(crate) fn div(self, rhs: Self) -> Self {
        let denominator = (rhs.hi.abs() - rhs.lo.abs() - rhs.error) * (1.0 - 4.0 * f64::EPSILON);
        let q = self.hi / rhs.hi;
        if denominator <= 0.0 || !q.is_finite() {
            return Self {
                error: f64::INFINITY,
                ..Self::ZERO
            };
        }
        let r = self.sub(rhs.mul(Self::point(q)));
        let low = (r.hi + r.lo) / rhs.hi;
        let (hi, lo) = two_sum(q, low);
        let candidate = Self { hi, lo, error: 0.0 };
        let residual = self.sub(rhs.mul(candidate));
        Self {
            error: inflate(residual.mass() / denominator),
            ..candidate
        }
    }
    pub(crate) fn sqrt(self) -> Self {
        if self.hi == 0.0 {
            return self;
        }
        let q = RealScalar::sqrt(self.hi);
        let r = self.sub(Self::point(q).mul(Self::point(q)));
        let low = (r.hi + r.lo) / (2.0 * q);
        let (hi, lo) = two_sum(q, low);
        let candidate = Self { hi, lo, error: 0.0 };
        let residual = self.sub(candidate.mul(candidate));
        // |sqrt(x)-q| = |x-q²|/(sqrt(x)+q) <= |x-q²|/q.
        Self {
            error: inflate(residual.mass() / (hi * (1.0 - f64::EPSILON))),
            ..candidate
        }
    }
    #[inline]
    pub(crate) fn with_tail(self, tail: f64) -> Self {
        Self {
            error: inflate(self.error + tail),
            ..self
        }
    }
    /// Power-of-two scaling within a finite working range. One eta covers
    /// each possible underflow of a component or its error bound.
    #[inline]
    pub(crate) fn scale(self, exponent: i32) -> Self {
        Self {
            hi: RealScalar::scalbn(self.hi, exponent),
            lo: RealScalar::scalbn(self.lo, exponent),
            error: inflate(RealScalar::scalbn(self.error, exponent) + 2.0 * ETA),
        }
    }
    /// Outward endpoints. Each side is decided against its own neighbor gap:
    /// the rounded high word, its adjacent float, or a directed general
    /// addition for wider bounds.
    pub(crate) fn endpoints(self) -> (f64, f64) {
        if !self.error.is_finite() || !self.hi.is_finite() {
            return (f64::NEG_INFINITY, f64::INFINITY);
        }
        let (h, l) = two_sum(self.hi, self.lo);
        if self.error == 0.0 && l == 0.0 {
            return (h, h);
        }
        // The value lies in [h + l - error, h + l + error] because TwoSum is
        // exact. Gaps between adjacent floats are exact, and next_up of a
        // nearest-rounded distance bounds the exact distance from above, so a
        // distance that fits a gap keeps that side on the adjacent float. At a
        // power of two the two gaps differ by a factor of two, so testing both
        // sides against the smaller gap would widen them.
        let down = RealScalar::next_after(h, f64::NEG_INFINITY);
        let up = RealScalar::next_after(h, f64::INFINITY);
        // Usually both distances fit the smaller gap, and the per-side tests
        // below would select the same endpoints.
        let margin = RealScalar::next_after(l.abs() + self.error, f64::INFINITY);
        if margin <= (h - down).min(up - h) {
            return (
                if l < self.error { down } else { h },
                if l > -self.error { up } else { h },
            );
        }
        let lower = if l >= self.error {
            h
        } else if RealScalar::next_after(self.error - l, f64::INFINITY) <= h - down {
            down
        } else {
            RealScalar::next_after(
                h + RealScalar::next_after(l - self.error, f64::NEG_INFINITY),
                f64::NEG_INFINITY,
            )
        };
        let upper = if l <= -self.error {
            h
        } else if RealScalar::next_after(l + self.error, f64::INFINITY) <= up - h {
            up
        } else {
            RealScalar::next_after(
                h + RealScalar::next_after(l + self.error, f64::INFINITY),
                f64::INFINITY,
            )
        };
        (lower, upper)
    }
}

#[cfg(test)]
mod tests {
    use crate::elementary::constants::{EXP_TABLE, FACTORIAL, LN2, LN10, ODD_RECIPROCAL, PI};
    use alloc::vec::Vec;
    #[test]
    fn split_constants_contain_directed_mpfr_residuals() {
        let mut count = 0;
        for line in include_str!("../../tests/data/elementary-constants.txt")
            .lines()
            .filter(|s| !s.starts_with('#'))
        {
            let fields: Vec<_> = line.split_whitespace().collect();
            let index: usize = fields[1].parse().unwrap();
            let value = match fields[0] {
                "0" => LN2,
                "1" => LN10,
                "2" => PI,
                "3" => EXP_TABLE[index],
                "4" => FACTORIAL[index],
                _ => ODD_RECIPROCAL[index],
            };
            let decode = |word| f64::from_bits(u64::from_str_radix(word, 16).unwrap());
            assert_eq!(value.hi.to_bits(), decode(fields[2]).to_bits());
            // Directed exact-zero residuals may have either zero sign.
            assert_eq!(value.lo, decode(fields[3]));
            assert!(value.error >= decode(fields[4]).abs().max(decode(fields[5]).abs()));
            count += 1;
        }
        assert_eq!(count, 104);
    }
}
