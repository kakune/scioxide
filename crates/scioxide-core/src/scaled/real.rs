//! Real values with an extended binary exponent.
use crate::{
    DomainReason, InvalidInputReason, NonFiniteOrigin, NumericalIssue, RealScalar, SciOxideError,
};

// Error-free sum followed by a compensated sum of two two-term expansions.
// All operands here are bounded by 2^65, so no intermediate can overflow.
use crate::arithmetic::two_sum as sum2;
fn add2(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    let (hi, error) = sum2(a.0, b.0);
    sum2(hi, error + a.1 + b.1)
}
fn times_log_two(value: f64) -> (f64, f64) {
    const LOW: f64 = 2.319_046_813_846_299_6e-17;
    let hi = value * core::f64::consts::LN_2;
    sum2(
        hi,
        RealScalar::mul_add(value, core::f64::consts::LN_2, -hi) + value * LOW,
    )
}
fn log_remainder(value: f64, exponent: i128) -> (f64, f64) {
    // The high integer is rounded to binary64; the remaining integer has at
    // most eleven bits and is exact. Never convert k+1 through a scalar floor.
    let high = exponent as f64;
    let low = (exponent - high as i128) as f64;
    let a = times_log_two(-high);
    let b = times_log_two(-low);
    add2(add2((value, 0.0), a), b)
}
use super::rounding::{maximum_normal_exponent, minimum_subnormal_exponent};
use crate::scalar::{FiniteParts, finite_parts};
use core::cmp::Ordering;

fn exponent_error() -> SciOxideError {
    SciOxideError::numerical_instability(
        NumericalIssue::ExponentOutOfRange,
        "the normalized binary exponent is outside i64",
    )
}

/// A real number `m · 2ᵉ` with `1 ≤ |m| < 2` and a 64-bit exponent `e`.
///
/// The significand `m` has the precision of `S`, but the exponent range is that
/// of `i64`, so intermediate results do not overflow or underflow. Zero keeps its
/// sign and has exponent zero.
///
/// Each arithmetic operation rounds its exact result once to the precision of
/// `S`. Exact zero follows the IEEE sign rules, and cancellation of opposite
/// nonzero values gives `+0`. For an exactly rounded sum of many terms, use
/// [`RangeAccumulator`](crate::RangeAccumulator) instead. Convert back with
/// [`Self::round`], which allows gradual underflow, or [`Self::try_value`], which
/// rejects a nonzero value that would round to zero.
///
/// # Examples
///
/// ```
/// use scioxide_core::Scaled;
///
/// // 1.5 · 2²⁰⁰⁰ is far outside the f64 range.
/// let large = Scaled::from_parts(1.5_f64, 2000)?;
/// let ratio = large.div(large)?;
/// assert_eq!(ratio.try_value()?, 1.0);
/// assert!(large.add(-large)?.is_zero());
/// assert!(large.abs_cmp(ratio).is_gt());
/// assert!((-large).cmp_value(ratio).is_lt());
/// assert!(large.round().is_err());
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scaled<S: RealScalar> {
    mantissa: S,
    exponent: i64,
}

// A public operation combines a fixed number of i64 exponents. The widest
// expression, a complex quotient of product differences, stays within
// 8 * 2^63 + 4096. These temporaries let cancellation precede range checks.
#[derive(Clone, Copy)]
pub(super) struct Wide<S: RealScalar> {
    pub(super) mantissa: S,
    exponent: i128,
}

impl<S: RealScalar> From<Scaled<S>> for Wide<S> {
    fn from(v: Scaled<S>) -> Self {
        Self {
            mantissa: v.mantissa,
            exponent: i128::from(v.exponent),
        }
    }
}

impl<S: RealScalar> Wide<S> {
    // Normalized operations have finite intermediate significands of magnitude
    // at most 8. Only decoded finite parts enter the normalization kernel.
    pub(super) fn normalize(value: S, exponent: i128) -> Result<Self, SciOxideError> {
        let parts = finite_parts(value, "scaled component must be finite")?;
        Ok(Self::normalize_parts(parts, exponent))
    }

    fn normalize_parts(parts: FiniteParts, exponent: i128) -> Self {
        let bits = parts.significand();
        if bits == 0 {
            return Self {
                mantissa: if parts.negative() {
                    S::NEG_ZERO
                } else {
                    S::ZERO
                },
                exponent: 0,
            };
        }
        let leading = bits.leading_zeros();
        // All decoded bits fit the destination precision. This u64 conversion
        // is exact on both pointer widths and never substitutes a sentinel.
        let magnitude = S::from_u64_half_open01(bits << leading).scalbn(1);
        Self {
            mantissa: if parts.negative() {
                -magnitude
            } else {
                magnitude
            },
            exponent: exponent + i128::from(parts.exponent()) + i128::from(63 - leading),
        }
    }
    pub(super) fn finish(self) -> Result<Scaled<S>, SciOxideError> {
        Ok(Scaled {
            mantissa: self.mantissa,
            exponent: i64::try_from(self.exponent).map_err(|_| exponent_error())?,
        })
    }
    pub(super) fn neg(self) -> Self {
        Self {
            mantissa: -self.mantissa,
            ..self
        }
    }
    pub(super) fn abs(self) -> Self {
        Self {
            mantissa: self.mantissa.abs(),
            ..self
        }
    }
    pub(super) fn shifted(self, shift: i128) -> Self {
        if self.mantissa.is_zero() {
            self
        } else {
            Self {
                exponent: self.exponent + shift,
                ..self
            }
        }
    }
    pub(super) fn abs_cmp(self, other: Self) -> Ordering {
        match (self.mantissa.is_zero(), other.mantissa.is_zero()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => self.exponent.cmp(&other.exponent).then_with(|| {
                if self.mantissa.abs() < other.mantissa.abs() {
                    Ordering::Less
                } else if self.mantissa.abs() > other.mantissa.abs() {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            }),
        }
    }
    pub(super) fn add(self, other: Self) -> Result<Self, SciOxideError> {
        if self.mantissa.is_zero() && other.mantissa.is_zero() {
            return Self::normalize(self.mantissa + other.mantissa, 0);
        }
        if self.mantissa.is_zero() {
            return Ok(other);
        }
        if other.mantissa.is_zero() {
            return Ok(self);
        }
        let (large, small) = if self.exponent >= other.exponent {
            (self, other)
        } else {
            (other, self)
        };
        let gap = large.exponent - small.exponent;
        if gap > i128::from(S::MANTISSA_DIGITS + 2) {
            return Ok(large);
        }
        Self::normalize(
            large.mantissa + small.mantissa.scalbn(-(gap as i32)),
            large.exponent,
        )
    }
    pub(super) fn sub(self, other: Self) -> Result<Self, SciOxideError> {
        self.add(other.neg())
    }
    pub(super) fn mul(self, other: Self) -> Result<Self, SciOxideError> {
        Self::normalize(
            self.mantissa * other.mantissa,
            self.exponent + other.exponent,
        )
    }
    pub(super) fn div(self, other: Self) -> Result<Self, SciOxideError> {
        if other.mantissa.is_zero() {
            return Err(SciOxideError::domain_error(
                crate::argument!("denominator"),
                DomainReason::Undefined,
                "division by zero",
            ));
        }
        Self::normalize(
            self.mantissa / other.mantissa,
            self.exponent - other.exponent,
        )
    }
    pub(super) fn difference_of_products(
        self,
        b: Self,
        c: Self,
        d: Self,
    ) -> Result<Self, SciOxideError> {
        if (self.mantissa.is_zero() || b.mantissa.is_zero())
            && (c.mantissa.is_zero() || d.mantissa.is_zero())
        {
            return Self::normalize(self.mantissa * b.mantissa - c.mantissa * d.mantissa, 0);
        }
        if self.mantissa.is_zero() || b.mantissa.is_zero() {
            return Ok(c.mul(d)?.neg());
        }
        if c.mantissa.is_zero() || d.mantissa.is_zero() {
            return self.mul(b);
        }
        let left = self.exponent + b.exponent;
        let right = c.exponent + d.exponent;
        let common = left.max(right);
        // Products contain at most 2p bits. Beyond 2p+4 bits only the smaller
        // product's sign can affect a midpoint tie. Retain that sticky term.
        let limit = i128::from(2 * S::MANTISSA_DIGITS + 4);
        let a = self.mantissa.scalbn(-((common - left).min(limit) as i32));
        let c = c.mantissa.scalbn(-((common - right).min(limit) as i32));
        Self::normalize(
            crate::checked_difference_of_products(a, b.mantissa, c, d.mantissa)?,
            common,
        )
    }
    pub(super) fn sqrt(self) -> Result<Self, SciOxideError> {
        if self.mantissa.is_zero() {
            return Ok(self);
        }
        if self.mantissa.is_sign_negative() {
            return Err(SciOxideError::domain_error(
                crate::argument!("radicand"),
                DomainReason::OutsideDomain,
                "negative real square root",
            ));
        }
        let value = self.mantissa.scalbn(self.exponent.rem_euclid(2) as i32);
        Self::normalize(value.sqrt(), self.exponent.div_euclid(2))
    }
    pub(super) fn round_ieee(self) -> S {
        self.round_to_ieee()
    }
    fn round_to_ieee<R: RealScalar>(self) -> R {
        let negative = self.mantissa.is_sign_negative();
        let signed = |value: R| if negative { -value } else { value };
        if self.mantissa.is_zero() {
            return signed(R::ZERO);
        }
        let maximum = i128::from(maximum_normal_exponent::<R>());
        if self.exponent > maximum {
            return signed(R::INFINITY);
        }
        let minimum = i128::from(minimum_subnormal_exponent::<R>());
        if self.exponent < minimum - 1 {
            return signed(R::ZERO);
        }
        let Ok(parts) = FiniteParts::decode(self.mantissa) else {
            return R::NAN;
        };
        // The IEEE window plus <= 53 decoded significand bits fits i32.
        super::rounding::rounded_words_at_exponent::<R>(
            &[parts.significand()],
            negative,
            (self.exponent + i128::from(parts.exponent())) as i32,
            false,
        )
    }
    pub(super) fn hypot(self, other: Self) -> Result<Self, SciOxideError> {
        if self.mantissa.is_zero() {
            return Ok(other.abs());
        }
        if other.mantissa.is_zero() {
            return Ok(self.abs());
        }
        // Binary alignment is exact wherever a term can influence the norm.
        // At least one magnitude is in [1,2), both are below 2, and a term
        // too small for this bounded frame contributes below the norm's ulp.
        // This avoids rounding two arbitrary ratios and renormalizing them.
        let common = self.exponent.max(other.exponent);
        let align = |value: Self| {
            value
                .mantissa
                .scalbn((value.exponent - common).max(i128::from(i32::MIN)) as i32)
        };
        Self::normalize(align(self).hypot(align(other)), common)
    }
}

impl<S: RealScalar> core::ops::Neg for Scaled<S> {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            mantissa: -self.mantissa,
            ..self
        }
    }
}

impl<S: RealScalar> Scaled<S> {
    // Only representation containers populated from checked Scaled values use
    // this constructor. Their private fields retain exactly these invariants.
    pub(crate) fn from_normalized_parts(mantissa: S, exponent: i64) -> Self {
        Self { mantissa, exponent }
    }
    /// Returns `sign · e^(log_magnitude)` without the IEEE range limit.
    ///
    /// `sign` must be `1` or `−1`. If the result is a normal number of `S`, it is
    /// within 4 ulps. Otherwise the error in the logarithm is at most
    /// `16 ε + 64 |k| (ε₆₄ / 2)²`, where `k` is the binary exponent of the result,
    /// `ε` the machine epsilon of `S` and `ε₆₄` that of `f64`. Errors in
    /// `log_magnitude` itself add to this.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `sign` or `log_magnitude` is not finite.
    /// - `InvalidInput` if `sign` is not `±1`.
    /// - `NumericalInstability` (`ExponentOutOfRange`) if the exponent leaves the
    ///   `i64` range.
    pub fn from_log(sign: S, log_magnitude: S) -> Result<Self, SciOxideError> {
        if !sign.is_finite() || !log_magnitude.is_finite() {
            return Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "sign and log magnitude must be finite",
            ));
        }
        if sign != S::ONE && sign != -S::ONE {
            return Err(SciOxideError::invalid_input(
                crate::argument!("sign"),
                InvalidInputReason::OutOfRange,
                "sign must be -1 or 1",
            ));
        }
        let x = log_magnitude.into_f64();
        // A coarse guard before conversion; normalized i64 range is checked
        // only after quotient correction. It includes both endpoint neighbors.
        if x.abs() > 8.0e18 {
            return Err(exponent_error());
        }
        let ordinary = log_magnitude.exp();
        if ordinary.is_normal() {
            return Self::from_parts(sign * ordinary, 0);
        }
        let approximate = RealScalar::round(x * core::f64::consts::LOG2_E) as i128;
        let r = log_remainder(x, approximate);
        let correction = RealScalar::round((r.0 + r.1) * core::f64::consts::LOG2_E) as i128;
        let exponent = approximate + correction;
        let r = log_remainder(x, exponent);
        let hi = RealScalar::exp(r.0);
        let mantissa = RealScalar::mul_add(hi, RealScalar::exp_m1(r.1), hi);
        Wide::normalize(sign * S::try_from_f64_lossy(mantissa)?, exponent)?.finish()
    }
    /// Positive zero.
    pub const ZERO: Self = Self {
        mantissa: S::ZERO,
        exponent: 0,
    };
    /// One.
    pub const ONE: Self = Self {
        mantissa: S::ONE,
        exponent: 0,
    };
    /// One half.
    pub const HALF: Self = Self {
        mantissa: S::ONE,
        exponent: -1,
    };
    /// Converts a finite value exactly, keeping signed zeros and subnormals.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `value` is NaN or infinite.
    pub fn from_value(value: S) -> Result<Self, SciOxideError> {
        Self::from_parts(value, 0)
    }
    /// Returns `value · 2^exponent`, normalized before its range is checked.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `value` is NaN or infinite.
    /// - `NumericalInstability` (`ExponentOutOfRange`) if the normalized exponent
    ///   leaves the `i64` range.
    pub fn from_parts(value: S, exponent: i64) -> Result<Self, SciOxideError> {
        Wide::normalize(value, i128::from(exponent))?.finish()
    }
    /// Returns the significand `m`, with `1 ≤ |m| < 2`, or a signed zero.
    pub fn significand(self) -> S {
        self.mantissa
    }
    /// Returns the binary exponent `e`; zero has exponent zero.
    pub fn exponent(self) -> i64 {
        self.exponent
    }
    /// Returns whether the value is zero.
    pub fn is_zero(self) -> bool {
        self.mantissa.is_zero()
    }
    /// Returns whether the sign is negative, including `−0`.
    pub fn is_sign_negative(self) -> bool {
        self.mantissa.is_sign_negative()
    }
    /// Returns the absolute value; zero becomes `+0`.
    pub fn abs(self) -> Self {
        Self {
            mantissa: self.mantissa.abs(),
            ..self
        }
    }
    /// Compares magnitudes exactly; both zeros compare equal.
    pub fn abs_cmp(self, other: Self) -> Ordering {
        Wide::from(self).abs_cmp(other.into())
    }
    /// Returns whether `|self| ≤ |other|`.
    pub fn abs_le(self, other: Self) -> bool {
        self.abs_cmp(other) != Ordering::Greater
    }
    /// Compares values; `+0` and `−0` compare equal.
    pub fn cmp_value(self, other: Self) -> Ordering {
        if self.is_zero() && other.is_zero() {
            return Ordering::Equal;
        }
        match (self.is_sign_negative(), other.is_sign_negative()) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => self.abs_cmp(other),
            (true, true) => other.abs_cmp(self),
        }
    }
    /// Returns `self + other`, rounded once.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`ExponentOutOfRange`) if the exponent leaves the
    ///   `i64` range.
    #[allow(
        clippy::should_implement_trait,
        reason = "checked arithmetic is fallible"
    )]
    pub fn add(self, other: Self) -> Result<Self, SciOxideError> {
        Wide::from(self).add(other.into())?.finish()
    }
    /// Returns `self − other`, rounded once.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`ExponentOutOfRange`) if the exponent leaves the
    ///   `i64` range.
    #[allow(
        clippy::should_implement_trait,
        reason = "checked arithmetic is fallible"
    )]
    pub fn sub(self, other: Self) -> Result<Self, SciOxideError> {
        Wide::from(self).sub(other.into())?.finish()
    }
    /// Returns `self · other`, rounded once.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`ExponentOutOfRange`) if the exponent leaves the
    ///   `i64` range.
    #[allow(
        clippy::should_implement_trait,
        reason = "checked arithmetic is fallible"
    )]
    pub fn mul(self, other: Self) -> Result<Self, SciOxideError> {
        Wide::from(self).mul(other.into())?.finish()
    }
    /// Returns `self²`, rounded once.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`ExponentOutOfRange`) if the exponent leaves the
    ///   `i64` range.
    pub fn square(self) -> Result<Self, SciOxideError> {
        self.mul(self)
    }
    /// Returns `ln|self|` as a value of `S`.
    ///
    /// The absolute error is at most `16 ε (1 + |ln|self||)`, where `ε` is the
    /// machine epsilon of `S`. This is not a relative error bound near `|self| = 1`.
    ///
    /// # Errors
    ///
    /// - `DomainError` if `self` is zero.
    pub fn ln_abs(self) -> Result<S, SciOxideError> {
        if self.is_zero() {
            return Err(SciOxideError::domain_error(
                crate::argument!("operand"),
                DomainReason::Undefined,
                "zero has no finite logarithm",
            ));
        }
        let exponent = S::try_from_f64_lossy(self.exponent as f64)?;
        Ok(self.mantissa.abs().ln() + exponent * S::try_from_f64_lossy(core::f64::consts::LN_2)?)
    }
    /// Returns `self / other`, rounded once.
    ///
    /// # Errors
    ///
    /// - `DomainError` if `other` is zero.
    /// - `NumericalInstability` (`ExponentOutOfRange`) if the exponent leaves the
    ///   `i64` range.
    #[allow(
        clippy::should_implement_trait,
        reason = "checked arithmetic is fallible"
    )]
    pub fn div(self, other: Self) -> Result<Self, SciOxideError> {
        Wide::from(self).div(other.into())?.finish()
    }
    /// Returns `self · b − c · d`, rounded once from the exact value.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`ExponentOutOfRange`) if the exponent leaves the
    ///   `i64` range.
    pub fn difference_of_products(self, b: Self, c: Self, d: Self) -> Result<Self, SciOxideError> {
        Wide::from(self)
            .difference_of_products(b.into(), c.into(), d.into())?
            .finish()
    }
    /// Returns `√self`, rounded once.
    ///
    /// # Errors
    ///
    /// - `DomainError` if `self` is negative and nonzero.
    /// - `NumericalInstability` (`ExponentOutOfRange`) if the exponent leaves the
    ///   `i64` range.
    pub fn sqrt(self) -> Result<Self, SciOxideError> {
        Wide::from(self).sqrt()?.finish()
    }
    /// Returns `self · 2^exponent` exactly.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`ExponentOutOfRange`) if the exponent leaves the
    ///   `i64` range.
    pub fn shifted(self, exponent: i64) -> Result<Self, SciOxideError> {
        Wide::from(self).shifted(i128::from(exponent)).finish()
    }
    // The finite scalar products used by complex power bound this phase to the
    // reducer's table range. Unsupported internal inputs propagate raw IEEE NaN.
    pub(crate) fn sin_cos_ieee(self) -> (S, S) {
        let rounded = self.round_ieee();
        if rounded.is_finite() {
            return rounded.sin_cos();
        }
        let Ok(parts) = FiniteParts::decode(self.mantissa) else {
            return (S::NAN, S::NAN);
        };
        let total = i128::from(self.exponent) + i128::from(parts.exponent());
        let Ok(total) = i32::try_from(total) else {
            return (S::NAN, S::NAN);
        };
        let (sin, cos) = crate::math::scaled_sin_cos(
            if parts.negative() {
                -(parts.significand() as f64)
            } else {
                parts.significand() as f64
            },
            total,
        );
        match (S::try_from_f64_lossy(sin), S::try_from_f64_lossy(cos)) {
            (Ok(sin), Ok(cos)) => (sin, cos),
            _ => (S::NAN, S::NAN),
        }
    }
    pub(crate) fn round_ieee(self) -> S {
        Wide::from(self).round_ieee()
    }
    // Representation summaries may project relative to a frame without storing
    // its intermediate exponent. The caller classifies IEEE range loss.
    pub(crate) fn round_shifted(self, shift: i128) -> S {
        Wide::from(self).shifted(shift).round_ieee()
    }
    // Final component projection must not first narrow a temporary exponent or
    // round a product significand. The bounded IEEE window fits the existing
    // exact accumulator; outside it the class is known without an integer cast.
    pub(crate) fn mul_round(self, other: Self) -> Result<S, SciOxideError> {
        if self.is_zero() || other.is_zero() {
            return Ok(self.mantissa * other.mantissa);
        }
        let exponent = i128::from(self.exponent) + i128::from(other.exponent);
        let minimum = i128::from(minimum_subnormal_exponent::<S>());
        let maximum = i128::from(maximum_normal_exponent::<S>());
        if exponent < minimum - 2 {
            return Ok(S::ZERO.copysign(self.mantissa * other.mantissa));
        }
        if exponent > maximum {
            return Err(SciOxideError::numerical_instability(
                NumericalIssue::Overflow,
                "final product component exceeds the IEEE finite range",
            ));
        }
        // minimum - 2 <= exponent <= maximum lies in [-1076, 1023].
        let mut product = crate::RangeAccumulator::<S>::new();
        product.add_scaled_product(self.mantissa, other.mantissa, exponent as i32)?;
        product.finish()
    }
    /// Converts to `S` with one rounding, allowing gradual underflow to a signed zero.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Overflow`) if the rounded value is infinite.
    pub fn round(self) -> Result<S, SciOxideError> {
        self.round_to()
    }
    /// Converts to another type `R` with one round-to-nearest-even step.
    ///
    /// The value is rounded directly to `R`, also at the subnormal boundary of `R`.
    /// Signed zeros and the signs of values that underflow to zero are kept.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Overflow`) if the rounded value is infinite.
    ///
    /// # Examples
    ///
    /// ```
    /// use scioxide_core::Scaled;
    ///
    /// // 1.5 · 2⁻¹⁴⁹ rounds to 2 · 2⁻¹⁴⁹, the second subnormal of f32.
    /// let x = Scaled::from_parts(1.5_f64, -149)?;
    /// assert_eq!(x.round_to::<f32>()?.to_bits(), 2);
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    pub fn round_to<R: RealScalar>(self) -> Result<R, SciOxideError> {
        let value: R = Wide::from(self).round_to_ieee();
        if value.is_finite() {
            Ok(value)
        } else {
            Err(SciOxideError::numerical_instability(
                NumericalIssue::Overflow,
                "final component exceeds the IEEE finite range",
            ))
        }
    }
    /// Converts to `S` with one rounding, rejecting underflow to zero.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Overflow`) if the rounded value is infinite.
    /// - `NumericalInstability` (`Underflow`) if a nonzero value rounds to zero.
    pub fn try_value(self) -> Result<S, SciOxideError> {
        self.try_value_to()
    }
    /// Converts to another type `R` with one rounding, rejecting underflow to zero.
    ///
    /// Signed zeros are kept.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Overflow`) if the rounded value is infinite.
    /// - `NumericalInstability` (`Underflow`) if a nonzero value rounds to zero.
    pub fn try_value_to<R: RealScalar>(self) -> Result<R, SciOxideError> {
        let value: R = self.round_to()?;
        if value.is_zero() && !self.is_zero() {
            Err(SciOxideError::numerical_instability(
                NumericalIssue::Underflow,
                "nonzero component rounds to zero",
            ))
        } else {
            Ok(value)
        }
    }
    /// Returns `e^self` as a value of `S`, allowing gradual underflow.
    ///
    /// # Errors
    ///
    /// - `NumericalInstability` (`Overflow`) if the result is infinite.
    pub fn exp(self) -> Result<S, SciOxideError> {
        let value = self.round_ieee().exp();
        if value.is_finite() {
            Ok(value)
        } else {
            Err(SciOxideError::numerical_instability(
                NumericalIssue::Overflow,
                "exponential result exceeds the IEEE finite range",
            ))
        }
    }
}
