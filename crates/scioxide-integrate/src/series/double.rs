//! Double-word real and complex values for ill-conditioned transform weights.
//!
//! A value is the unevaluated sum `high + low` with `|low|` at most about
//! `ulp(high)`, carrying about twice the working precision; accumulations
//! gather their rounding errors unnormalized (Ogita–Rump–Oishi Sum2) until
//! renormalized. General products use the error-free `two_product` only in
//! the range where it is exact, and drop the residual elsewhere, where the
//! operands are negligible or the result is about to be rejected as nonfinite.
//! The accumulating product and the quotient by a working-precision divisor are
//! unchecked: their callers scale the operands into that range and reject
//! nonfinite results. No operation allocates or fails. Real scalars skip every
//! imaginary operation.

use scioxide_core::backend::{two_product, two_sum};
use scioxide_core::{AlgebraicScalar, NumericScalar, RealScalar};

/// A real double-word value.
#[derive(Clone, Copy)]
pub(super) struct Double<R: RealScalar> {
    pub(super) high: R,
    pub(super) low: R,
}

/// Error-free product where `two_product` is exact, otherwise the rounded product.
fn product<R: RealScalar>(a: R, b: R) -> (R, R) {
    let digits = i32::try_from(R::MANTISSA_DIGITS).unwrap_or(i32::MAX);
    let large = R::MAX.scalbn(-digits);
    let small = R::MIN_POSITIVE.scalbn(2 * digits + 2);
    let rounded = a * b;
    if a.abs() <= large && b.abs() <= large && (rounded == R::ZERO || rounded.abs() >= small) {
        two_product(a, b)
    } else {
        (rounded, R::ZERO)
    }
}

impl<R: RealScalar> Double<R> {
    pub(super) const fn new(high: R) -> Self {
        Self { high, low: R::ZERO }
    }

    fn normalized(high: R, low: R) -> Self {
        let (high, low) = two_sum(high, low);
        Self { high, low }
    }

    /// The nearest double word to a nonnegative integer, exact below
    /// `2^(2 * MANTISSA_DIGITS)`.
    pub(super) fn integer(value: u128) -> Self {
        #[allow(clippy::cast_precision_loss, reason = "the residual is kept")]
        let high = R::try_from_f64_lossy(value as f64).unwrap_or(R::INFINITY);
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the leading part is integral"
        )]
        let rest = value.cast_signed() - high.into_f64() as i128;
        #[allow(clippy::cast_precision_loss, reason = "beyond double-word precision")]
        let low = R::try_from_f64_lossy(rest as f64).unwrap_or(R::NAN);
        Self { high, low }
    }

    /// `base^exponent` of a working-precision base, by squaring.
    pub(super) fn power(base: R, mut exponent: u32) -> Self {
        let mut result = Self::new(R::ONE);
        let mut factor = Self::new(base);
        while exponent > 0 {
            if exponent & 1 == 1 {
                result = result.mul(factor);
            }
            exponent >>= 1;
            if exponent > 0 {
                factor = factor.mul(factor);
            }
        }
        result
    }

    pub(super) fn add(self, other: Self) -> Self {
        let (high, low) = two_sum(self.high, other.high);
        Self::normalized(high, low + self.low + other.low)
    }

    /// `self + value` for a working-precision `value`.
    fn add_value(self, value: R) -> Self {
        let (high, low) = two_sum(self.high, value);
        Self::normalized(high, low + self.low)
    }

    pub(super) fn neg(self) -> Self {
        Self {
            high: -self.high,
            low: -self.low,
        }
    }

    pub(super) fn sub(self, other: Self) -> Self {
        self.add(other.neg())
    }

    pub(super) fn mul(self, other: Self) -> Self {
        let (high, low) = product(self.high, other.high);
        Self::normalized(high, low + (self.high * other.low + self.low * other.high))
    }

    pub(super) fn div(self, other: Self) -> Self {
        let quotient = self.high / other.high;
        let remainder = self.sub(other.mul(Self::new(quotient)));
        Self::normalized(quotient, remainder.high / other.high)
    }

    /// `self / divisor` for a working-precision `divisor`: a faithful quotient
    /// through the reciprocal plus its exact remainder over the divisor, at
    /// most a few units in the quotient's last place. Unchecked: the quotient
    /// and divisor must lie in the range where `two_product` is exact.
    fn div_value(self, divisor: R) -> Self {
        let reciprocal = R::ONE / divisor;
        let quotient = self.high * reciprocal;
        let (product, error) = two_product(quotient, divisor);
        // Within a few units of `self.high`, the product cancels exactly.
        let remainder = ((self.high - product) - error) + self.low;
        Self {
            high: quotient,
            low: remainder * reciprocal,
        }
    }

    /// `self + other` without renormalization (Ogita–Rump–Oishi Sum2): the low
    /// part gathers every rounding error until [`Self::renormalized`].
    fn accumulate(self, other: Self) -> Self {
        let (high, error) = two_sum(self.high, other.high);
        Self {
            high,
            low: self.low + (error + other.low),
        }
    }

    /// `self + a * b` without renormalization, with an error-free leading
    /// product. Unchecked: the leading parts must lie in the range where
    /// `two_product` is exact.
    fn accumulate_product(self, a: Self, b: Self) -> Self {
        let (product, error) = two_product(a.high, b.high);
        let (high, rounding) = two_sum(self.high, product);
        let tail = error + (a.high * b.low + a.low * b.high);
        Self {
            high,
            low: self.low + (rounding + tail),
        }
    }

    fn renormalized(self) -> Self {
        Self::normalized(self.high, self.low)
    }

    pub(super) fn is_finite(self) -> bool {
        self.high.is_finite() && self.low.is_finite()
    }
}

/// A double-word real or complex value of scalar type `T`; real scalars keep
/// a zero imaginary part and skip its arithmetic.
#[derive(Clone, Copy)]
pub(super) struct Wide<T: NumericScalar> {
    re: Double<T::Real>,
    im: Double<T::Real>,
}

impl<T: NumericScalar> Wide<T> {
    pub(super) const ZERO: Self = Self {
        re: Double::new(T::Real::ZERO),
        im: Double::new(T::Real::ZERO),
    };

    pub(super) fn of(value: T) -> Self {
        Self {
            re: Double::new(value.real_component()),
            im: Double::new(value.imaginary_component()),
        }
    }

    /// The rounded working-precision value.
    pub(super) fn value(self) -> T {
        T::from_components(self.re.high + self.re.low, self.im.high + self.im.low)
    }

    /// The leading working-precision parts.
    pub(super) fn leading(self) -> T {
        T::from_components(self.re.high, self.im.high)
    }

    pub(super) fn add(self, other: Self) -> Self {
        Self {
            re: self.re.add(other.re),
            im: if T::IS_REAL {
                self.im
            } else {
                self.im.add(other.im)
            },
        }
    }

    /// `self + value` for a working-precision `value`.
    pub(super) fn add_value(self, value: T) -> Self {
        Self {
            re: self.re.add_value(value.real_component()),
            im: if T::IS_REAL {
                self.im
            } else {
                self.im.add_value(value.imaginary_component())
            },
        }
    }

    pub(super) fn neg(self) -> Self {
        Self {
            re: self.re.neg(),
            im: self.im.neg(),
        }
    }

    /// Multiply every part by a power of two, exactly unless a part becomes
    /// subnormal.
    pub(super) fn scaled(self, power: T::Real) -> Self {
        let scale = |value: Double<T::Real>| Double {
            high: value.high * power,
            low: value.low * power,
        };
        Self {
            re: scale(self.re),
            im: if T::IS_REAL { self.im } else { scale(self.im) },
        }
    }

    /// `self + other` without renormalization; see [`Self::renormalized`].
    pub(super) fn accumulate(self, other: Self) -> Self {
        Self {
            re: self.re.accumulate(other.re),
            im: if T::IS_REAL {
                self.im
            } else {
                self.im.accumulate(other.im)
            },
        }
    }

    /// `self + a * b` without renormalization; see [`Self::renormalized`].
    pub(super) fn accumulate_product(self, a: Self, b: Self) -> Self {
        if T::IS_REAL {
            return Self {
                re: self.re.accumulate_product(a.re, b.re),
                im: self.im,
            };
        }
        Self {
            re: self
                .re
                .accumulate_product(a.re, b.re)
                .accumulate_product(a.im.neg(), b.im),
            im: self
                .im
                .accumulate_product(a.re, b.im)
                .accumulate_product(a.im, b.re),
        }
    }

    /// Fold the gathered rounding errors of an accumulation into a double word.
    pub(super) fn renormalized(self) -> Self {
        Self {
            re: self.re.renormalized(),
            im: self.im.renormalized(),
        }
    }

    /// `numerator / divisor` for a real double-word numerator and a
    /// working-precision divisor.
    pub(super) fn quotient(numerator: Double<T::Real>, divisor: T) -> Self {
        if T::IS_REAL {
            return Self {
                re: numerator.div_value(divisor.real_component()),
                im: Double::new(T::Real::ZERO),
            };
        }
        Self {
            re: numerator,
            im: Double::new(T::Real::ZERO),
        }
        .div(Self::of(divisor))
    }

    /// Binary exponent of the larger leading component (zero for zero).
    fn exponent(self) -> i32 {
        scioxide_core::backend::standard::exponent(self.re.high.abs().max(self.im.high.abs()))
            .unwrap_or(0)
    }

    /// Multiply by `2^exponent`, exactly unless a part becomes subnormal.
    fn shifted(self, exponent: i32) -> Self {
        let shift = |value: Double<T::Real>| Double {
            high: value.high.scalbn(exponent),
            low: value.low.scalbn(exponent),
        };
        Self {
            re: shift(self.re),
            im: shift(self.im),
        }
    }

    /// Quotient; a complex quotient first normalizes both operands by exact
    /// powers of two, so neither the squared divisor magnitude nor the
    /// product overflows.
    pub(super) fn div(self, other: Self) -> Self {
        if T::IS_REAL {
            return Self {
                re: self.re.div(other.re),
                im: self.im,
            };
        }
        let (top, bottom) = (self.exponent(), other.exponent());
        let (numerator, divisor) = (self.shifted(-top), other.shifted(-bottom));
        let norm = divisor.re.mul(divisor.re).add(divisor.im.mul(divisor.im));
        let product = Self {
            re: numerator
                .re
                .mul(divisor.re)
                .add(numerator.im.mul(divisor.im)),
            im: numerator
                .im
                .mul(divisor.re)
                .sub(numerator.re.mul(divisor.im)),
        };
        Self {
            re: product.re.div(norm),
            im: product.im.div(norm),
        }
        .shifted(top.saturating_sub(bottom))
    }

    pub(super) fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}
