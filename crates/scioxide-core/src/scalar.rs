//! Scalar traits: [`AlgebraicScalar`] for any field-like value, [`RealScalar`]
//! for `f32` and `f64`, and [`NumericScalar`] for those and their complex
//! counterparts.

use core::fmt::Debug;
use core::iter::{Product, Sum};
use core::num::FpCategory;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign};

use crate::complex::Complex;
use crate::error::{ScalarConversionReason, SciOxideError};

mod private {
    pub trait RealScalarSealed {}
    pub trait NumericScalarSealed {}
}

/// Exact sign, significand and exponent of a finite value; zero keeps its sign.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FiniteParts {
    significand: u64,
    exponent: i16,
    negative: bool,
}

impl FiniteParts {
    pub(crate) fn normalized(self) -> Self {
        if self.significand == 0 {
            return self;
        }
        let zeros = self.significand.trailing_zeros();
        Self {
            significand: self.significand >> zeros,
            exponent: self.exponent + zeros as i16,
            negative: self.negative,
        }
    }
    /// Decode once; raw IEEE callers handle the nonfinite classification directly.
    #[inline]
    pub(crate) fn decode<S: RealScalar>(value: S) -> Result<Self, FloatDecomposition> {
        match value.decompose() {
            FloatDecomposition::Finite {
                significand,
                exponent,
                negative,
            } => Ok(Self {
                significand,
                exponent,
                negative,
            }),
            nonfinite => Err(nonfinite),
        }
    }

    pub(crate) fn significand(self) -> u64 {
        self.significand
    }
    pub(crate) fn exponent(self) -> i16 {
        self.exponent
    }
    pub(crate) fn negative(self) -> bool {
        self.negative
    }

    pub(crate) fn is_one(self) -> bool {
        !self.negative
            && self.significand.is_power_of_two()
            && i32::from(self.exponent) == -(self.significand.trailing_zeros() as i32)
    }

    /// Exact pair product and its combined sign, without an IEEE intermediate.
    #[inline]
    pub(crate) fn product(self, other: Self) -> (u128, i32, bool) {
        (
            u128::from(self.significand) * u128::from(other.significand),
            i32::from(self.exponent) + i32::from(other.exponent),
            self.negative ^ other.negative,
        )
    }
}

/// Decodes a finite value, or returns a `NonFinite` input error with `detail`.
#[inline]
pub(crate) fn finite_parts<S: RealScalar>(
    value: S,
    detail: &'static str,
) -> Result<FiniteParts, SciOxideError> {
    FiniteParts::decode(value)
        .map_err(|_| SciOxideError::non_finite(crate::NonFiniteOrigin::Input, detail))
}

fn integer_is_exact_in_binary_float(magnitude: u128, mantissa_digits: u32) -> bool {
    if magnitude == 0 {
        return true;
    }
    // An integer is representable exactly when every bit that does not fit in
    // the significand is a trailing zero, i.e. the value is a representable
    // significand times a power of two.
    let significant_bits = u128::BITS - magnitude.leading_zeros();
    let discarded_bits = significant_bits.saturating_sub(mantissa_digits);
    discarded_bits == 0 || magnitude.trailing_zeros() >= discarded_bits
}

/// Exact decomposition of a floating-point value into sign, integer significand and binary exponent.
///
/// A finite value equals `(−1)^negative · significand · 2^exponent`. The
/// significands returned by [`RealScalar::decompose`] have at most 24 bits for
/// `f32` and 53 bits for `f64`, including the implicit leading bit of normal
/// values. Zero and subnormals use the minimum exponent, −149 for `f32` and
/// −1074 for `f64`, and zero keeps its sign. Infinities and NaN have their
/// own variants; the sign and payload of NaN are not reported.
///
/// ```rust
/// use scioxide_core::{FloatDecomposition, RealScalar};
///
/// // 1.5 = 12_582_912 · 2⁻²³
/// assert_eq!(
///     1.5_f32.decompose(),
///     FloatDecomposition::Finite { significand: 12_582_912, exponent: -23, negative: false }
/// );
/// assert_eq!(f64::INFINITY.decompose(), FloatDecomposition::Infinite { negative: false });
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatDecomposition {
    /// A finite value, including both zeros and subnormals.
    Finite {
        /// Integer significand, including the implicit bit of a normal value.
        significand: u64,
        /// Binary exponent applied to the integer significand.
        exponent: i16,
        /// Whether the sign bit is set, also for zero.
        negative: bool,
    },
    /// An infinity.
    Infinite {
        /// Whether the sign bit is set.
        negative: bool,
    },
    /// Any quiet or signaling NaN; its sign and payload are not reported.
    Nan,
}

/// Values with field arithmetic (`+`, `−`, `×`, `÷`), zero and one.
///
/// Unlike [`RealScalar`] and [`NumericScalar`], other crates may implement this
/// trait, for example for dual numbers in automatic differentiation. It does
/// not require IEEE 754 classification or elementary functions.
///
/// ```
/// use scioxide_core::AlgebraicScalar;
///
/// fn two<S: AlgebraicScalar>() -> S {
///     S::ONE + S::ONE
/// }
/// assert_eq!(two::<f64>(), 2.0);
/// ```
pub trait AlgebraicScalar:
    Copy
    + Clone
    + Debug
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Sum<Self>
    + Product<Self>
    + 'static
{
    /// Additive identity.
    const ZERO: Self;

    /// Multiplicative identity.
    const ONE: Self;

    /// Returns whether the value equals the additive identity.
    #[inline]
    #[must_use]
    fn is_zero(self) -> bool {
        self == Self::ZERO
    }
}

/// IEEE 754 real floating-point types: `f32` and `f64`.
///
/// The trait is sealed; no other type can implement it. It provides
/// precision and mathematical constants, elementary functions and checked
/// conversions. The arithmetic and compound-assignment operators and the
/// elementary-function methods follow IEEE 754 and do not check their arguments, so they can return NaN
/// or infinity; see [non-finite values](crate#non-finite-values). The checked
/// functions of this crate, such as [`crate::checked_sum`], do check.
///
/// # Elementary accuracy
///
/// For finite results, including subnormals, both scalar widths use the following
/// error envelopes. An ulp is the destination spacing at the exact result.
/// These are tested accuracy guarantees, not per-call certified error bounds.
///
/// | Methods | Maximum error |
/// |---|---|
/// | `sqrt`, `floor`, `ceil`, `round`, `trunc`, `fract`, `mul_add`, `scalbn` | 0 ulps (correct rounding) |
/// | `powi`, `powf`, `cbrt`, `hypot`, `exp`, `exp2`, `exp10`, `exp_m1`, `ln`, `log2`, `log10`, `ln_1p`, trigonometric and inverse trigonometric functions, hyperbolic and inverse hyperbolic functions, `atan2`, `sin_cos` | 4 ulps |
/// | `log` with an arbitrary base | 16 ulps |
///
/// `copysign` and `next_after` manipulate bits exactly; `to_degrees` and
/// `to_radians` use a scaling multiplication with no ULP guarantee.
/// Native and portable implementations may differ within these envelopes.
/// Qualify calls with the trait, such as `RealScalar::sin(x)`, to use SciOxide's
/// dispatch; inherent methods such as `f64::sin` do not use it.
///
/// ```
/// use scioxide_core::RealScalar;
///
/// fn half_turn<S: RealScalar>() -> S {
///     S::TAU * S::HALF
/// }
/// assert_eq!(half_turn::<f64>(), core::f64::consts::PI);
/// ```
///
/// Other types cannot implement the trait:
///
/// ```compile_fail
/// use scioxide_core::RealScalar;
///
/// struct ExternalFloat(f64);
///
/// fn require_real<S: RealScalar>() {}
/// fn rejected() {
///     require_real::<ExternalFloat>();
/// }
/// ```
pub trait RealScalar:
    private::RealScalarSealed
    + crate::backend::RealBackend
    + NumericScalar<Real = Self>
    + PartialOrd
    + Rem<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Send
    + Sync
{
    /// Returns the square root, correctly rounded (to nearest, ties to even).
    ///
    /// Subnormals and the sign of zero are kept. A negative nonzero input
    /// returns NaN.
    #[must_use]
    fn sqrt(self) -> Self;

    /// Returns whether the value is finite.
    #[must_use]
    fn is_finite(self) -> bool;

    /// Quiet NaN.
    const NAN: Self;
    /// Positive infinity.
    const INFINITY: Self;
    /// Negative infinity.
    const NEG_INFINITY: Self;
    /// Negative zero.
    const NEG_ZERO: Self;
    /// Lowest finite value.
    const MIN: Self;
    /// Smallest positive normal value.
    const MIN_POSITIVE: Self;
    /// Largest finite value.
    const MAX: Self;
    /// Number of explicit and implicit significand bits.
    const MANTISSA_DIGITS: u32;
    /// Converts to `f64` exactly, including NaN, infinities and the sign of zero.
    #[must_use]
    fn into_f64(self) -> f64;
    /// Machine epsilon `ε`, the distance from 1 to the next larger value.
    const EPSILON: Self;
    /// `√ε`, rounded to nearest.
    const SQRT_EPSILON: Self;

    /// `∛ε`, rounded once to nearest, ties to even.
    const CBRT_EPSILON: Self;
    /// Smallest positive subnormal value.
    const MIN_SUBNORMAL: Self;
    /// The value 2.
    const TWO: Self;
    /// The value 10.
    const TEN: Self;
    /// Natural logarithm of two.
    const LN_2: Self;
    /// Natural logarithm of ten.
    const LN_10: Self;
    /// Base-two logarithm of e.
    const LOG2_E: Self;
    /// Base-ten logarithm of e.
    const LOG10_E: Self;

    /// Mathematical constant π.
    const PI: Self;
    /// Mathematical constant 2π.
    const TAU: Self;
    /// The value 0.5.
    const HALF: Self;

    /// Returns whether the value is NaN.
    #[must_use]
    fn is_nan(self) -> bool;
    /// Returns whether the value is infinite.
    #[must_use]
    fn is_infinite(self) -> bool;
    /// Returns whether the value is normal (not zero, subnormal, infinite or NaN).
    #[must_use]
    fn is_normal(self) -> bool;
    /// Returns the IEEE 754 category of the value.
    #[must_use]
    fn classify(self) -> FpCategory;
    /// Returns whether the sign bit is clear.
    #[must_use]
    fn is_sign_positive(self) -> bool;
    /// Returns whether the sign bit is set.
    #[must_use]
    fn is_sign_negative(self) -> bool;

    /// Rounds toward −∞ to an integer.
    #[must_use]
    fn floor(self) -> Self;
    /// Rounds toward +∞ to an integer.
    #[must_use]
    fn ceil(self) -> Self;
    /// Rounds to the nearest integer, with halfway cases away from zero.
    #[must_use]
    fn round(self) -> Self;
    /// Rounds toward zero to an integer.
    #[must_use]
    fn trunc(self) -> Self;
    /// Returns the fractional part, `self − self.trunc()`.
    #[must_use]
    fn fract(self) -> Self;
    /// Returns `1` or `−1` with the sign of the value, or NaN for NaN.
    #[must_use]
    fn signum(self) -> Self;
    /// Computes `self · factor + addend` with one rounding (fused multiply-add).
    ///
    /// The result is rounded to nearest, ties to even. Subnormal results are
    /// kept, and overflow gives a signed infinity.
    #[must_use]
    fn mul_add(self, factor: Self, addend: Self) -> Self;
    /// Returns `1 / self`.
    #[must_use]
    fn recip(self) -> Self;
    /// Raises to an integer power.
    #[must_use]
    fn powi(self, exponent: i32) -> Self;
    /// Raises to a floating-point power.
    #[must_use]
    fn powf(self, exponent: Self) -> Self;
    /// Returns the cube root.
    #[must_use]
    fn cbrt(self) -> Self;
    /// Returns `√(self² + other²)` without intermediate overflow.
    #[must_use]
    fn hypot(self, other: Self) -> Self;
    /// Returns `eˣ`.
    #[must_use]
    fn exp(self) -> Self;
    /// Returns `2ˣ`.
    #[must_use]
    fn exp2(self) -> Self;
    /// Returns `10ˣ`, with IEEE 754 overflow, underflow and special values.
    #[must_use]
    fn exp10(self) -> Self;
    /// Returns `eˣ − 1`, accurate for small `x`.
    #[must_use]
    fn exp_m1(self) -> Self;
    /// Returns the natural logarithm.
    #[must_use]
    fn ln(self) -> Self;
    /// Returns the logarithm to base `base`.
    #[must_use]
    fn log(self, base: Self) -> Self;
    /// Returns the base-2 logarithm.
    #[must_use]
    fn log2(self) -> Self;
    /// Returns the base-10 logarithm.
    #[must_use]
    fn log10(self) -> Self;
    /// Returns `ln(1 + x)`, accurate for small `x`.
    #[must_use]
    fn ln_1p(self) -> Self;
    /// Returns the larger value; if one operand is NaN, returns the other.
    #[must_use]
    fn max(self, other: Self) -> Self;
    /// Returns the smaller value; if one operand is NaN, returns the other.
    #[must_use]
    fn min(self, other: Self) -> Self;
    /// Restricts the value to the closed interval `[minimum, maximum]`.
    ///
    /// Returns NaN if a bound is NaN or `minimum > maximum`; never panics.
    #[must_use]
    fn clamp(self, minimum: Self, maximum: Self) -> Self;
    /// Returns the value with the sign bit of `sign`.
    #[must_use]
    fn copysign(self, sign: Self) -> Self;

    /// Returns the sine (argument in radians).
    #[must_use]
    fn sin(self) -> Self;
    /// Returns the cosine (argument in radians).
    #[must_use]
    fn cos(self) -> Self;
    /// Returns the tangent (argument in radians).
    #[must_use]
    fn tan(self) -> Self;
    /// Returns the arcsine, in `[−π/2, π/2]`.
    #[must_use]
    fn asin(self) -> Self;
    /// Returns the arccosine, in `[0, π]`.
    #[must_use]
    fn acos(self) -> Self;
    /// Returns the arctangent, in `[−π/2, π/2]`.
    #[must_use]
    fn atan(self) -> Self;
    /// Returns the four-quadrant arctangent of `self / other`, in `[−π, π]`.
    #[must_use]
    fn atan2(self, other: Self) -> Self;
    /// Returns `(sin x, cos x)`.
    #[must_use]
    fn sin_cos(self) -> (Self, Self);
    /// Returns the hyperbolic sine.
    #[must_use]
    fn sinh(self) -> Self;
    /// Returns the hyperbolic cosine.
    #[must_use]
    fn cosh(self) -> Self;
    /// Returns the hyperbolic tangent.
    #[must_use]
    fn tanh(self) -> Self;
    /// Returns the inverse hyperbolic sine.
    #[must_use]
    fn asinh(self) -> Self;
    /// Returns the inverse hyperbolic cosine.
    #[must_use]
    fn acosh(self) -> Self;
    /// Returns the inverse hyperbolic tangent.
    #[must_use]
    fn atanh(self) -> Self;
    /// Converts radians to degrees.
    #[must_use]
    fn to_degrees(self) -> Self;
    /// Converts degrees to radians.
    #[must_use]
    fn to_radians(self) -> Self;

    /// Returns the adjacent representable value in the direction of `toward`.
    #[must_use]
    fn next_after(self, toward: Self) -> Self;
    /// Returns `self · 2^exponent`.
    #[must_use]
    fn scalbn(self, exponent: i32) -> Self;
    /// Returns the exact sign, integer significand and binary exponent; see [`FloatDecomposition`].
    #[must_use]
    fn decompose(self) -> FloatDecomposition;

    /// Converts a finite `f64` exactly.
    ///
    /// # Errors
    ///
    /// - `ScalarConversion` if `value` is NaN, infinite, out of range, or not
    ///   exactly representable in `Self`.
    fn try_from_f64_exact(value: f64) -> Result<Self, SciOxideError>;

    /// Converts a finite `f64`, rounding to nearest if it is not representable.
    ///
    /// The sign of zero is kept.
    ///
    /// # Errors
    ///
    /// - `ScalarConversion` if `value` is NaN, infinite or out of range.
    fn try_from_f64_lossy(value: f64) -> Result<Self, SciOxideError>;

    /// Converts a `usize` exactly.
    ///
    /// # Errors
    ///
    /// - `ScalarConversion` if `value` is not exactly representable in `Self`.
    fn try_from_usize_exact(value: usize) -> Result<Self, SciOxideError>;

    /// Converts an `i64` exactly.
    ///
    /// # Errors
    ///
    /// - `ScalarConversion` if `value` is not exactly representable in `Self`.
    fn try_from_i64_exact(value: i64) -> Result<Self, SciOxideError>;

    /// Converts a finite value to `f64` exactly, keeping the sign of zero.
    ///
    /// Unlike [`Self::into_f64`], this rejects NaN and infinities.
    ///
    /// # Errors
    ///
    /// - `ScalarConversion` ([`ScalarConversionReason::NonFiniteInput`]) for
    ///   NaN or an infinity.
    ///
    /// ```
    /// use scioxide_core::RealScalar;
    ///
    /// assert_eq!(0.5_f32.try_to_f64_exact()?, 0.5_f64);
    /// assert_eq!((-0.0_f32).try_to_f64_exact()?.to_bits(), (-0.0_f64).to_bits());
    /// assert!(f64::INFINITY.try_to_f64_exact().is_err());
    /// # Ok::<(), scioxide_core::SciOxideError>(())
    /// ```
    fn try_to_f64_exact(self) -> Result<f64, SciOxideError>;

    /// Converts to `i64` exactly.
    ///
    /// # Errors
    ///
    /// - `ScalarConversion` if the value is NaN, infinite, not an integer, or out of range.
    fn try_to_i64_exact(self) -> Result<i64, SciOxideError>;

    /// Converts to `usize` exactly.
    ///
    /// # Errors
    ///
    /// - `ScalarConversion` if the value is NaN, infinite, not an integer, or out of range.
    fn try_to_usize_exact(self) -> Result<usize, SciOxideError>;
}

use crate::math::dispatch::scalar_methods;

macro_rules! impl_real_scalar {
    ($ty:ident, $mantissa:expr, $pi:expr, $tau:expr, $next_after:path, $scalbn:path, $decompose:path, $sqrt_epsilon:expr, $cbrt_epsilon:expr, $dispatch:ident) => {
        impl private::RealScalarSealed for $ty {}

        impl AlgebraicScalar for $ty {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
        }

        impl RealScalar for $ty {
            crate::math::dispatch::registry!(scalar_methods, $ty, $dispatch, $next_after, $scalbn);
            const CBRT_EPSILON: Self = <$ty>::from_bits($cbrt_epsilon);
            const MIN_SUBNORMAL: Self = <$ty>::from_bits(1);
            const TWO: Self = 2.0;
            const TEN: Self = 10.0;
            const LN_2: Self = core::$ty::consts::LN_2;
            const LN_10: Self = core::$ty::consts::LN_10;
            const LOG2_E: Self = core::$ty::consts::LOG2_E;
            const LOG10_E: Self = core::$ty::consts::LOG10_E;
            #[inline]
            fn is_finite(self) -> bool {
                <$ty>::is_finite(self)
            }

            const NAN: Self = <$ty>::NAN;
            const INFINITY: Self = <$ty>::INFINITY;
            const NEG_INFINITY: Self = <$ty>::NEG_INFINITY;
            const NEG_ZERO: Self = -0.0;
            const MIN: Self = <$ty>::MIN;
            const MIN_POSITIVE: Self = <$ty>::MIN_POSITIVE;
            const MAX: Self = <$ty>::MAX;
            const MANTISSA_DIGITS: u32 = $mantissa;
            #[inline]
            fn into_f64(self) -> f64 {
                self as f64
            }
            const EPSILON: Self = <$ty>::EPSILON;
            const SQRT_EPSILON: Self = <$ty>::from_bits($sqrt_epsilon);
            const PI: Self = $pi;
            const TAU: Self = $tau;
            const HALF: Self = 0.5;
            #[inline]
            fn is_nan(self) -> bool {
                <$ty>::is_nan(self)
            }
            #[inline]
            fn is_infinite(self) -> bool {
                <$ty>::is_infinite(self)
            }
            #[inline]
            fn is_normal(self) -> bool {
                <$ty>::is_normal(self)
            }
            #[inline]
            fn classify(self) -> FpCategory {
                <$ty>::classify(self)
            }
            #[inline]
            fn is_sign_positive(self) -> bool {
                <$ty>::is_sign_positive(self)
            }
            #[inline]
            fn is_sign_negative(self) -> bool {
                <$ty>::is_sign_negative(self)
            }
            #[inline]
            fn decompose(self) -> FloatDecomposition {
                $decompose(self)
            }

            fn try_from_f64_exact(value: f64) -> Result<Self, SciOxideError> {
                let scalar_kind: crate::ConversionType =
                    <Self as crate::backend::Scalar>::kind().into();
                if !value.is_finite() {
                    return Err(SciOxideError::scalar_conversion(
                        crate::ConversionType::F64,
                        scalar_kind,
                        ScalarConversionReason::NonFiniteInput,
                    ));
                }
                let converted = value as Self;
                if !converted.is_finite() {
                    return Err(SciOxideError::scalar_conversion(
                        crate::ConversionType::F64,
                        scalar_kind,
                        ScalarConversionReason::OutOfRange,
                    ));
                }
                if (converted as f64).to_bits() != value.to_bits() {
                    return Err(SciOxideError::scalar_conversion(
                        crate::ConversionType::F64,
                        scalar_kind,
                        ScalarConversionReason::Inexact,
                    ));
                }
                Ok(converted)
            }

            #[inline]
            fn try_from_f64_lossy(value: f64) -> Result<Self, SciOxideError> {
                let scalar_kind: crate::ConversionType =
                    <Self as crate::backend::Scalar>::kind().into();
                if !value.is_finite() {
                    return Err(SciOxideError::scalar_conversion(
                        crate::ConversionType::F64,
                        scalar_kind,
                        ScalarConversionReason::NonFiniteInput,
                    ));
                }
                let converted = value as Self;
                if !converted.is_finite() {
                    return Err(SciOxideError::scalar_conversion(
                        crate::ConversionType::F64,
                        scalar_kind,
                        ScalarConversionReason::OutOfRange,
                    ));
                }
                if value == 0.0 && value.is_sign_negative() != converted.is_sign_negative() {
                    return Err(SciOxideError::scalar_conversion(
                        crate::ConversionType::F64,
                        scalar_kind,
                        ScalarConversionReason::Inexact,
                    ));
                }
                Ok(converted)
            }

            #[inline]
            fn try_from_usize_exact(value: usize) -> Result<Self, SciOxideError> {
                if !integer_is_exact_in_binary_float(value as u128, $mantissa) {
                    return Err(SciOxideError::scalar_conversion(
                        crate::ConversionType::Usize,
                        <Self as crate::backend::Scalar>::kind().into(),
                        ScalarConversionReason::Inexact,
                    ));
                }
                Ok(value as Self)
            }

            fn try_from_i64_exact(value: i64) -> Result<Self, SciOxideError> {
                if !integer_is_exact_in_binary_float(u128::from(value.unsigned_abs()), $mantissa) {
                    return Err(SciOxideError::scalar_conversion(
                        crate::ConversionType::I64,
                        <Self as crate::backend::Scalar>::kind().into(),
                        ScalarConversionReason::Inexact,
                    ));
                }
                Ok(value as Self)
            }

            #[inline]
            fn try_to_f64_exact(self) -> Result<f64, SciOxideError> {
                if !self.is_finite() {
                    return Err(SciOxideError::scalar_conversion(
                        <Self as crate::backend::Scalar>::kind().into(),
                        crate::ConversionType::F64,
                        ScalarConversionReason::NonFiniteInput,
                    ));
                }
                Ok(self.into_f64())
            }

            fn try_to_i64_exact(self) -> Result<i64, SciOxideError> {
                let scalar_kind: crate::ConversionType =
                    <Self as crate::backend::Scalar>::kind().into();
                if !self.is_finite() {
                    return Err(SciOxideError::scalar_conversion(
                        scalar_kind,
                        crate::ConversionType::I64,
                        ScalarConversionReason::NonFiniteInput,
                    ));
                }
                let value = self as f64;
                if RealScalar::fract(value) != 0.0 {
                    return Err(SciOxideError::scalar_conversion(
                        scalar_kind,
                        crate::ConversionType::I64,
                        ScalarConversionReason::Inexact,
                    ));
                }
                // `as` saturates instead of failing, so the range must be checked first.
                if !((i64::MIN as f64)..-(i64::MIN as f64)).contains(&value) {
                    return Err(SciOxideError::scalar_conversion(
                        scalar_kind,
                        crate::ConversionType::I64,
                        ScalarConversionReason::OutOfRange,
                    ));
                }
                let converted = self as i64;
                if converted as Self != self {
                    return Err(SciOxideError::scalar_conversion(
                        scalar_kind,
                        crate::ConversionType::I64,
                        ScalarConversionReason::Inexact,
                    ));
                }
                Ok(converted)
            }

            #[inline]
            fn try_to_usize_exact(self) -> Result<usize, SciOxideError> {
                let scalar_kind: crate::ConversionType =
                    <Self as crate::backend::Scalar>::kind().into();
                let parts = FiniteParts::decode(self).map_err(|_| {
                    SciOxideError::scalar_conversion(
                        scalar_kind,
                        crate::ConversionType::Usize,
                        ScalarConversionReason::NonFiniteInput,
                    )
                })?;
                let value = self as f64;
                let upper_bound = (1_u128 << usize::BITS) as f64;
                if value < 0.0 || value >= upper_bound {
                    // Preserve fractional-input error precedence outside the
                    // range without a libm truncation call. Inside the range,
                    // the round trip below already proves exact integrality.
                    let reason = if parts.significand() != 0
                        && i32::from(parts.exponent())
                            + (parts.significand().trailing_zeros() as i32)
                            < 0
                    {
                        ScalarConversionReason::Inexact
                    } else {
                        ScalarConversionReason::OutOfRange
                    };
                    return Err(SciOxideError::scalar_conversion(
                        scalar_kind,
                        crate::ConversionType::Usize,
                        reason,
                    ));
                }
                let converted = self as usize;
                if converted as Self != self {
                    return Err(SciOxideError::scalar_conversion(
                        scalar_kind,
                        crate::ConversionType::Usize,
                        ScalarConversionReason::Inexact,
                    ));
                }
                Ok(converted)
            }
        }
    };
}

impl_real_scalar!(
    f32,
    24,
    core::f32::consts::PI,
    core::f32::consts::TAU,
    crate::math::next_after_f32,
    crate::math::scalbn_f32,
    crate::math::decompose_f32,
    0x39b5_04f3,
    0x3ba1_4518,
    binary32
);
impl_real_scalar!(
    f64,
    53,
    core::f64::consts::PI,
    core::f64::consts::TAU,
    crate::math::next_after_f64,
    crate::math::scalbn_f64,
    crate::math::decompose_f64,
    0x3e50_0000_0000_0000,
    0x3ed9_65fe_a53d_6e3d,
    binary64
);

/// Real or complex floating-point types: `f32`, `f64`, [`Complex<f32>`] and [`Complex<f64>`].
///
/// The trait is sealed. Use it only where an algorithm supports complex
/// arithmetic; otherwise use [`RealScalar`].
///
/// ```
/// use scioxide_core::{Complex, NumericScalar};
///
/// assert!(<f64 as NumericScalar>::IS_REAL);
/// assert!(!<Complex<f64> as NumericScalar>::IS_REAL);
/// ```
pub trait NumericScalar:
    private::NumericScalarSealed + crate::backend::Scalar + AlgebraicScalar + Send + Sync
{
    /// Whether the type is real (`f32` or `f64`).
    const IS_REAL: bool;
    /// The real type of the components.
    type Real: RealScalar;
    /// Returns the complex conjugate; real values are returned unchanged.
    #[must_use]
    fn conj(self) -> Self;
    /// Returns the absolute value or complex modulus `|z|`.
    #[must_use]
    fn abs(self) -> Self::Real;
    /// Returns `|z|²`.
    #[must_use]
    fn norm_sqr(self) -> Self::Real;
    /// Returns the real part.
    #[must_use]
    fn real_component(self) -> Self::Real;
    /// Returns the imaginary part, or `+0` for a real type.
    #[must_use]
    fn imaginary_component(self) -> Self::Real;
    /// Creates a value from its real and imaginary parts.
    ///
    /// Real types ignore `imaginary`.
    #[must_use]
    fn from_components(real: Self::Real, imaginary: Self::Real) -> Self;
    /// Returns whether the real part and, for complex types, the imaginary part are finite.
    #[must_use]
    fn has_finite_components(self) -> bool;
    /// Converts a real value, with zero imaginary part for complex types.
    #[must_use]
    fn from_real(value: Self::Real) -> Self;
}

macro_rules! impl_numeric_real {
    ($ty:ty) => {
        impl private::NumericScalarSealed for $ty {}
        impl NumericScalar for $ty {
            const IS_REAL: bool = true;
            type Real = $ty;
            #[inline]
            fn conj(self) -> Self {
                self
            }
            #[inline]
            fn abs(self) -> Self::Real {
                self.abs()
            }
            #[inline]
            fn norm_sqr(self) -> Self::Real {
                self * self
            }
            #[inline]
            fn real_component(self) -> Self::Real {
                self
            }
            #[inline]
            fn imaginary_component(self) -> Self::Real {
                0.0
            }
            #[inline]
            fn from_components(real: Self::Real, imaginary: Self::Real) -> Self {
                let _ = imaginary;
                real
            }
            #[inline]
            fn has_finite_components(self) -> bool {
                self.is_finite()
            }
            #[inline]
            fn from_real(value: Self::Real) -> Self {
                value
            }
        }
    };
}

impl_numeric_real!(f32);
impl_numeric_real!(f64);

impl<R: RealScalar> AlgebraicScalar for Complex<R> {
    const ZERO: Self = Self::new(R::ZERO, R::ZERO);
    const ONE: Self = Self::new(R::ONE, R::ZERO);
}

impl<R: RealScalar> private::NumericScalarSealed for Complex<R> {}
impl<R: RealScalar> NumericScalar for Complex<R> {
    const IS_REAL: bool = false;
    type Real = R;
    #[inline]
    fn conj(self) -> Self {
        self.conj()
    }
    #[inline]
    fn abs(self) -> Self::Real {
        self.norm()
    }
    #[inline]
    fn norm_sqr(self) -> Self::Real {
        self.norm_sqr()
    }
    #[inline]
    fn real_component(self) -> Self::Real {
        self.re
    }
    #[inline]
    fn imaginary_component(self) -> Self::Real {
        self.im
    }
    #[inline]
    fn from_components(real: Self::Real, imaginary: Self::Real) -> Self {
        Self::new(real, imaginary)
    }
    #[inline]
    fn has_finite_components(self) -> bool {
        self.is_finite()
    }
    #[inline]
    fn from_real(value: Self::Real) -> Self {
        Self::new(value, R::ZERO)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn numeric_scalar_representation_kind() {
        assert!(core::hint::black_box(
            <f32 as super::NumericScalar>::IS_REAL
        ));
        assert!(core::hint::black_box(
            <f64 as super::NumericScalar>::IS_REAL
        ));
        assert!(!core::hint::black_box(
            <super::Complex<f32> as super::NumericScalar>::IS_REAL
        ));
        assert!(!core::hint::black_box(
            <super::Complex<f64> as super::NumericScalar>::IS_REAL
        ));
    }

    use super::*;

    mod atan2_boundaries {
        include!("../tests/support/atan2_cases.rs");
    }

    #[test]
    fn supported_real_constants_are_exact_primitive_constants() {
        assert_eq!(
            <f32 as RealScalar>::PI.to_bits(),
            core::f32::consts::PI.to_bits()
        );
        assert_eq!(
            <f64 as RealScalar>::TAU.to_bits(),
            core::f64::consts::TAU.to_bits()
        );
        assert_eq!(<f64 as RealScalar>::EPSILON, f64::EPSILON);
    }

    #[test]
    fn exact_and_lossy_conversions_have_distinct_results() {
        assert!(<f32 as RealScalar>::try_from_f64_exact(0.1).is_err());
        assert_eq!(
            <f32 as RealScalar>::try_from_f64_lossy(0.1).unwrap(),
            0.1_f32
        );
        assert!(<f64 as RealScalar>::try_from_f64_lossy(f64::INFINITY).is_err());
    }

    #[test]
    fn conversions_preserve_signed_zero() {
        let narrow = <f32 as RealScalar>::try_from_f64_exact(-0.0).unwrap();
        assert!(narrow.is_sign_negative());
        let wide = <f32 as RealScalar>::try_to_f64_exact(-0.0).unwrap();
        assert!(wide.is_sign_negative());
    }

    #[test]
    fn exact_widening_preserves_finite_representations_and_rejects_nonfinite() {
        fn check<S: RealScalar>() {
            for value in [
                S::ZERO,
                S::NEG_ZERO,
                S::ONE,
                -S::ONE,
                S::MIN,
                S::MAX,
                S::MIN_POSITIVE,
                S::ZERO.next_after(S::ONE),
                -S::ZERO.next_after(S::ONE),
            ] {
                assert_eq!(
                    value.try_to_f64_exact().unwrap().to_bits(),
                    value.into_f64().to_bits(),
                );
            }
            for value in [S::NAN, S::INFINITY, S::NEG_INFINITY] {
                assert!(matches!(
                    value.try_to_f64_exact().unwrap_err().view(),
                    crate::SciOxideErrorView::ScalarConversion {
                        from_type,
                        to_type: crate::ConversionType::F64,
                        reason: ScalarConversionReason::NonFiniteInput,
                    } if from_type == <S as crate::backend::Scalar>::kind().into()
                ));
            }
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn exact_count_conversion_preserves_boundary_error_precedence() {
        fn check<S: RealScalar>() {
            for value in [-0.5, 0.5, -1.5] {
                assert!(matches!(
                    (S::try_from_f64_lossy(value).unwrap().try_to_usize_exact())
                        .as_ref()
                        .unwrap_err()
                        .view(),
                    crate::SciOxideErrorView::ScalarConversion {
                        reason: ScalarConversionReason::Inexact,
                        ..
                    }
                ));
            }
            for value in [-1.0, 2.0_f64.powi(usize::BITS as i32)] {
                assert!(matches!(
                    (S::try_from_f64_lossy(value).unwrap().try_to_usize_exact())
                        .as_ref()
                        .unwrap_err()
                        .view(),
                    crate::SciOxideErrorView::ScalarConversion {
                        reason: ScalarConversionReason::OutOfRange,
                        ..
                    }
                ));
            }
            let subnormal = S::MIN_POSITIVE * <S as RealScalar>::EPSILON;
            assert!(matches!(
                (subnormal.try_to_usize_exact())
                    .as_ref()
                    .unwrap_err()
                    .view(),
                crate::SciOxideErrorView::ScalarConversion {
                    reason: ScalarConversionReason::Inexact,
                    ..
                }
            ));
            assert_eq!(
                S::try_from_f64_lossy(-0.0)
                    .unwrap()
                    .try_to_usize_exact()
                    .unwrap(),
                0
            );
            assert_eq!(
                S::try_from_f64_lossy(4096.0)
                    .unwrap()
                    .try_to_usize_exact()
                    .unwrap(),
                4096
            );
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn conversion_failures_are_machine_readable() {
        let inexact = <f32 as RealScalar>::try_from_f64_exact(0.1).unwrap_err();
        assert!(matches!(
            (inexact).view(),
            crate::SciOxideErrorView::ScalarConversion {
                reason: ScalarConversionReason::Inexact,
                ..
            }
        ));
        let out_of_range = <f32 as RealScalar>::try_from_f64_lossy(f64::MAX).unwrap_err();
        assert!(matches!(
            (out_of_range).view(),
            crate::SciOxideErrorView::ScalarConversion {
                reason: ScalarConversionReason::OutOfRange,
                ..
            }
        ));
        let non_finite = <f64 as RealScalar>::try_from_f64_lossy(f64::NAN).unwrap_err();
        assert!(matches!(
            (non_finite).view(),
            crate::SciOxideErrorView::ScalarConversion {
                reason: ScalarConversionReason::NonFiniteInput,
                ..
            }
        ));
    }

    #[test]
    fn numeric_complex_contract_uses_owned_representation() {
        let value = Complex::new(3.0_f64, 4.0);
        assert_eq!(<Complex<f64> as NumericScalar>::abs(value), 5.0);
        assert_eq!(
            <Complex<f64> as NumericScalar>::conj(value),
            Complex::new(3.0, -4.0)
        );
    }

    #[test]
    fn real_scalar_widening_is_exact() {
        for value in [0.0_f32, 0.5, 1.0 - f32::EPSILON] {
            assert_eq!(value.into_f64(), f64::from(value));
        }
        for value in [0.0_f64, 0.5, 1.0 - f64::EPSILON] {
            assert_eq!(value.into_f64(), value);
        }
    }

    #[test]
    fn real_scalar_next_after_matches_primitive_ordering() {
        assert_eq!(
            <f32 as RealScalar>::next_after(1.0, f32::NEG_INFINITY),
            1.0_f32.next_down()
        );
        assert_eq!(
            <f64 as RealScalar>::next_after(1.0, f64::NEG_INFINITY),
            1.0_f64.next_down()
        );
        assert_eq!(
            <f32 as RealScalar>::next_after(-0.0, f32::NEG_INFINITY).to_bits(),
            (-0.0_f32).next_down().to_bits()
        );
        assert_eq!(
            <f64 as RealScalar>::next_after(-0.0, f64::NEG_INFINITY).to_bits(),
            (-0.0_f64).next_down().to_bits()
        );
    }
}
