//! Complex numbers with public components and a `#[repr(C)]` layout.

use core::fmt;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::{AlgebraicScalar, NonFiniteOrigin, RealScalar, SciOxideError};

/// A complex number `re + i·im` with components of type `T`.
///
/// The layout is `#[repr(C)]`: the real component followed by the imaginary
/// one. Arithmetic follows IEEE 754 for the component type and does not check
/// for NaN or infinity; use [`Complex::try_new`] where finite values are
/// required. Logarithms, roots and inverse functions return principal values.
///
/// # Accuracy
///
/// For finite arguments with representable results, each component of `sqrt`,
/// `ln`, `tan`, `tanh` and the inverse trigonometric and hyperbolic functions
/// is within `16 ε max(|Re w|, |Im w|)` of the exact result `w`, where `ε` is
/// the machine epsilon. For `exp`, `sin`, `cos`, `sinh` and `cosh` the bound is
/// multiplied by `1 + max(|Re z|, |Im z|)`, and for `powf` and `powc` (`zʷ`) by
/// `1 + |w| |ln z|`. Integer powers round once per multiplication, without
/// intermediate overflow. Division has the bound documented on its `Div`
/// implementation. Components that underflow are compared on the subnormal grid.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[repr(C)]
pub struct Complex<T> {
    /// Real component.
    pub re: T,
    /// Imaginary component.
    pub im: T,
}

/// A single-precision complex number.
pub type Complex32 = Complex<f32>;

/// A double-precision complex number.
pub type Complex64 = Complex<f64>;

// Scaled logarithmic magnitude, with a log1p branch around the unit circle.
pub(crate) fn log_hypot<T: RealScalar>(x: T, y: T) -> T {
    let a = x.abs().max(y.abs());
    let b = x.abs().min(y.abs());
    if x.is_infinite() || y.is_infinite() {
        return T::INFINITY;
    }
    if x.is_nan() || y.is_nan() {
        return T::NAN;
    }
    if a == T::ZERO {
        return T::NEG_INFINITY;
    }
    if b == T::ZERO {
        return a.ln();
    }
    // Near |z| = 1, ln|z| = ln1p(a^2 - 1 + b^2) / 2, and writing a^2 - 1 as
    // (a - 1)(a + 1) avoids the cancellation in a^2 - 1. Elsewhere,
    // ln a + ln1p((b/a)^2) / 2 never forms a^2 + b^2, which could overflow or
    // underflow.
    if a >= T::HALF && a <= T::ONE {
        // Near the unit circle, rounding either square before subtracting one
        // erases a representable logarithm. Three bounded exact products retain
        // the residual even when the ordinary modulus rounds to exactly one.
        // All inputs are finite and <= 1; the result lies in [-3/4, 1], and
        // three terms fit the fixed accumulator by construction.
        return match crate::checked_sum_of_products(&[a, b, -T::ONE], &[a, b, T::ONE]) {
            Ok(residual) => residual.ln_1p() * T::HALF,
            Err(_) => T::NAN, // Raw IEEE boundary; the proved finite path cannot fail.
        };
    }
    let ratio = b / a;
    a.ln() + (ratio * ratio).ln_1p() * T::HALF
}

// Exponent `e` of a finite value in `[2^e, 2^(e+1))`; zero uses exponent zero.
fn binary_exponent<T: RealScalar>(value: T) -> i32 {
    match value.decompose() {
        crate::FloatDecomposition::Finite {
            significand,
            exponent,
            ..
        } if significand != 0 => i32::from(exponent) + 63 - significand.leading_zeros() as i32,
        _ => 0,
    }
}

// Split exp before multiplication: a component may be finite even when exp(x) is not.
fn exp_times<T: RealScalar>(x: T, factor: T) -> T {
    if factor == T::ZERO && !x.is_nan() {
        return factor;
    }
    let half = (x * T::HALF).exp();
    (half * factor) * half
}
fn cosh_times<T: RealScalar>(x: T, factor: T) -> T {
    if factor == T::ZERO && !x.is_nan() {
        return factor;
    }
    // Past ln(MAX) / 2, cosh(x) = e^|x| / 2 to working precision and may
    // overflow even though cosh(x) * factor is finite, so the exponential is
    // split into two halves around the multiplication.
    if x.abs() > T::MAX.ln() * T::HALF {
        // Halve after the first multiplication, so a subnormal factor survives.
        let half_exp = (x.abs() * T::HALF).exp();
        ((half_exp * factor) * T::HALF) * half_exp
    } else {
        x.cosh() * factor
    }
}
fn sinh_times<T: RealScalar>(x: T, factor: T) -> T {
    // In the same range, sinh(x) = sign(x) cosh(x) to working precision.
    if x.abs() > T::MAX.ln() * T::HALF {
        cosh_times(x, factor) * T::ONE.copysign(x)
    } else {
        x.sinh() * factor
    }
}

// Hull-style distance formulation. Square roots of A-1 and A-x are formed
// without subtracting nearly equal distances or squaring tiny imaginary parts.
fn inverse_distances<T: RealScalar>(x: T, y: T) -> (T, T, T) {
    let one = T::ONE;
    let two = T::TWO;
    let r = (x + one).hypot(y);
    let s = (x - one).hypot(y);
    let a = r * T::HALF + s * T::HALF;
    let first = y / (two * (r + x + one)).sqrt();
    let second_den = two * (s + (x - one).abs());
    let second = if second_den == T::ZERO {
        T::ZERO
    } else {
        y / second_den.sqrt()
    };
    let correction = first.hypot(second);
    if x <= one {
        (a, correction, (one - x).sqrt().hypot(correction))
    } else {
        (a, (x - one).sqrt().hypot(correction), correction)
    }
}
fn inverse_sine_parts<T: RealScalar>(x: T, y: T) -> (T, T) {
    if x.is_nan() || y.is_nan() {
        return (T::NAN, T::NAN);
    }
    // For |z| > 1/eps, asin z = -i ln(2 i z) + O(1/|z|^2), and the dropped terms
    // are below rounding. On the real axis, |x| <= 1 is the real asin and
    // |x| > 1 lies on the branch cut: pi/2 + i acosh(x).
    if x > T::ONE / T::EPSILON || y > T::ONE / T::EPSILON {
        return (x.atan2(y), log_hypot(x, y) + T::LN_2);
    }
    if y == T::ZERO {
        return if x <= T::ONE {
            (x.asin(), T::ZERO)
        } else {
            (T::PI * T::HALF, x.acosh())
        };
    }
    let (a, root_am1, root_amx) = inverse_distances(x, y);
    let real = x.atan2(root_amx * (a + x).sqrt());
    let imaginary = (root_am1 * (root_am1 + (a + T::ONE).sqrt())).ln_1p();
    (real, imaginary)
}

impl<T> Complex<T> {
    /// Creates a complex number from its components without checking them.
    #[inline]
    pub const fn new(re: T, im: T) -> Self {
        Self { re, im }
    }
}

impl<T: RealScalar> Complex<T> {
    /// Creates a complex number from two finite components.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if either component is NaN or infinite.
    pub fn try_new(re: T, im: T) -> Result<Self, SciOxideError> {
        if re.is_finite() && im.is_finite() {
            Ok(Self::new(re, im))
        } else {
            Err(SciOxideError::non_finite(
                NonFiniteOrigin::Input,
                "real and imaginary components must be finite",
            ))
        }
    }

    /// Returns the imaginary unit.
    #[inline]
    #[must_use]
    pub fn i() -> Self {
        Self::new(T::ZERO, T::ONE)
    }

    /// Returns the complex conjugate.
    #[inline]
    #[must_use]
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    /// Returns the squared Euclidean norm.
    #[inline]
    #[must_use]
    pub fn norm_sqr(self) -> T {
        self.re * self.re + self.im * self.im
    }

    /// Returns the modulus `|z| = √(re² + im²)`, rescaled to avoid
    /// intermediate overflow and underflow.
    #[inline]
    #[must_use]
    pub fn norm(self) -> T {
        self.re.hypot(self.im)
    }

    /// Returns whether both components are finite.
    #[inline]
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    /// Returns whether either component is NaN.
    #[inline]
    #[must_use]
    pub fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }

    /// Returns whether neither component is NaN and at least one is infinite.
    #[inline]
    #[must_use]
    pub fn is_infinite(self) -> bool {
        !self.is_nan() && (self.re.is_infinite() || self.im.is_infinite())
    }

    /// Returns the principal argument `atan2(im, re)` in `[−π, π]`.
    #[inline]
    #[must_use]
    pub fn arg(self) -> T {
        self.im.atan2(self.re)
    }

    /// Returns the polar form `(|z|, arg z)`.
    #[inline]
    #[must_use]
    pub fn to_polar(self) -> (T, T) {
        (self.norm(), self.arg())
    }

    /// Creates `radius · e^(i·phase)`.
    #[inline]
    #[must_use]
    pub fn from_polar(radius: T, phase: T) -> Self {
        Self::new(radius * phase.cos(), radius * phase.sin())
    }

    /// Multiplies both components by a real `factor`.
    #[inline]
    #[must_use]
    pub fn scale(self, factor: T) -> Self {
        Self::new(self.re * factor, self.im * factor)
    }

    /// Divides both components by a real `factor`.
    #[inline]
    #[must_use]
    pub fn unscale(self, factor: T) -> Self {
        Self::new(self.re / factor, self.im / factor)
    }

    /// Returns the multiplicative inverse.
    #[inline]
    #[must_use]
    pub fn inv(self) -> Self {
        Self::ONE / self
    }

    /// Computes `zⁿ` for an unsigned integer `n`.
    #[must_use]
    pub fn powu(self, exponent: u32) -> Self {
        self.integer_power(exponent, false)
    }

    /// Computes `zⁿ` for a signed integer `n`.
    #[inline]
    #[must_use]
    pub fn powi(self, exponent: i32) -> Self {
        self.integer_power(exponent.unsigned_abs(), exponent < 0)
    }

    // Reuse the per-component extended-range arithmetic. A common complex
    // scale would discard tiny components whose powers can later be finite.
    fn integer_power(self, exponent: u32, reciprocal: bool) -> Self {
        use crate::ScaledValue;
        if exponent == 0 {
            return Self::ONE;
        }
        // Non-finite and zero bases follow the ordinary IEEE special-value algebra
        // of repeated squaring, which the extended-range path cannot represent.
        if !self.is_finite() || (self.re == T::ZERO && self.im == T::ZERO) {
            let mut result = Self::ONE;
            let mut base = if reciprocal { self.inv() } else { self };
            let mut power = exponent;
            while power != 0 {
                if power & 1 != 0 {
                    result *= base;
                }
                power >>= 1;
                if power != 0 {
                    base *= base;
                }
            }
            return result;
        }
        let calculate = || -> Result<ScaledValue<crate::Complex<T>>, SciOxideError> {
            let one = ScaledValue::from_value(crate::Complex::new(
                T::ONE,
                <_ as crate::AlgebraicScalar>::ZERO,
            ))?;
            let mut base = ScaledValue::from_value(self)?;
            if reciprocal {
                base = one.div(base)?;
            }
            let mut result = one;
            let mut power = exponent;
            while power != 0 {
                if power & 1 != 0 {
                    result = result.mul(base)?;
                }
                power >>= 1;
                if power != 0 {
                    base = base.mul(base)?;
                }
            }
            Ok(result)
        };
        // Scalar exponent ranges times u32::MAX fit i64, so finite inputs
        // cannot exhaust the extended exponent; keep internal errors explicit.
        match calculate() {
            Ok(value) => Self::new(value.real().round_ieee(), value.imaginary().round_ieee()),
            Err(_) => Self::new(T::NAN, T::NAN),
        }
    }

    /// Returns the complex exponential.
    #[must_use]
    pub fn exp(self) -> Self {
        // C99 Annex G special values for cexp, where exp(re) is infinite or zero
        // and the generic product would produce NaN.
        let Self { re, mut im } = self;
        if re.is_infinite() {
            if re.is_sign_negative() && !im.is_finite() {
                return Self::new(T::ZERO, T::ZERO);
            }
            if re.is_sign_positive() && (im == T::ZERO || !im.is_finite()) {
                if im.is_infinite() {
                    im = T::NAN;
                }
                return Self::new(re, im);
            }
        } else if re.is_nan() && im == T::ZERO {
            return self;
        }

        let (sin, cos) = im.sin_cos();
        Self::new(exp_times(re, cos), exp_times(re, sin))
    }

    /// Computes the principal natural logarithm `ln|z| + i arg z`.
    #[inline]
    #[must_use]
    pub fn ln(self) -> Self {
        Self::new(log_hypot(self.re, self.im), self.arg())
    }

    /// Returns the principal square root.
    #[must_use]
    pub fn sqrt(self) -> Self {
        if self.im.is_infinite() {
            return Self::new(T::INFINITY, self.im);
        }
        if self.re.is_infinite() {
            return if self.re.is_sign_positive() {
                Self::new(
                    self.re,
                    if self.im.is_nan() {
                        self.im
                    } else {
                        T::ZERO.copysign(self.im)
                    },
                )
            } else {
                Self::new(
                    if self.im.is_nan() { self.im } else { T::ZERO },
                    T::INFINITY.copysign(self.im),
                )
            };
        }
        if self.is_nan() {
            return Self::new(T::NAN, T::NAN);
        }
        if self.im == T::ZERO {
            return if self.re >= T::ZERO {
                Self::new(self.re.abs().sqrt(), self.im)
            } else {
                Self::new(T::ZERO, (-self.re).sqrt().copysign(self.im))
            };
        }
        // Normalize with one ratio. Its square is bounded by one, and
        // sqrt(scale) is outside the bounded expression, even for subnormals.
        // The smaller root component comes from a division, not cancellation.
        let x = self.re.abs();
        let y = self.im.abs();
        let root = if x >= y {
            let ratio = y / x;
            x.sqrt() * ((T::ONE + (T::ONE + ratio * ratio).sqrt()) * T::HALF).sqrt()
        } else {
            let ratio = x / y;
            y.sqrt() * ((ratio + (T::ONE + ratio * ratio).sqrt()) * T::HALF).sqrt()
        };
        let small = (self.im.abs() / root) * T::HALF;
        if self.re >= T::ZERO {
            Self::new(root, small.copysign(self.im))
        } else {
            Self::new(small, root.copysign(self.im))
        }
    }

    /// Returns the complex sine.
    #[inline]
    #[must_use]
    pub fn sin(self) -> Self {
        let (sin, cos) = self.re.sin_cos();
        Self::new(cosh_times(self.im, sin), sinh_times(self.im, cos))
    }

    /// Returns the complex cosine.
    #[inline]
    #[must_use]
    pub fn cos(self) -> Self {
        let (sin, cos) = self.re.sin_cos();
        Self::new(cosh_times(self.im, cos), -sinh_times(self.im, sin))
    }

    /// Returns the complex tangent.
    #[inline]
    #[must_use]
    pub fn tan(self) -> Self {
        let (sin, cos) = self.re.sin_cos();
        let y = self.im.abs();
        if y < T::ONE {
            let sh = self.im.sinh();
            let den = cos * cos + sh * sh;
            return Self::new(sin * cos / den, sh * self.im.cosh() / den);
        }
        // For |im| >= 1, sinh and cosh grow exponentially. Multiplying the
        // numerator and the denominator cos^2 + sinh^2(y) by 4t with t = e^(-2|y|)
        // gives (4t sin cos + i (1 - t^2)) / ((1 - t)^2 + 4t cos^2), in which every
        // term is bounded.
        let t = (-(y + y)).exp();
        let one_minus_t = T::ONE - t;
        let four_t = (t + t) + (t + t);
        let den = one_minus_t * one_minus_t + four_t * cos * cos;
        if y.is_infinite() {
            return Self::new(T::ZERO.copysign(sin * cos), T::ONE.copysign(self.im));
        }
        Self::new(
            four_t * sin * cos / den,
            ((T::ONE - t * t) / den).copysign(self.im),
        )
    }

    /// Returns the complex hyperbolic sine.
    #[inline]
    #[must_use]
    pub fn sinh(self) -> Self {
        // sinh z = -i sin(i z), cosh z = cos(i z), and tanh z = -i tan(i z); the
        // hyperbolic forms reuse the overflow-safe trigonometric kernels.
        let value = Self::new(-self.im, self.re).sin();
        Self::new(value.im, -value.re)
    }

    /// Returns the complex hyperbolic cosine.
    #[inline]
    #[must_use]
    pub fn cosh(self) -> Self {
        Self::new(-self.im, self.re).cos()
    }

    /// Returns the complex hyperbolic tangent.
    #[inline]
    #[must_use]
    pub fn tanh(self) -> Self {
        let value = Self::new(-self.im, self.re).tan();
        Self::new(value.im, -value.re)
    }

    /// Returns the principal inverse sine.
    #[must_use]
    pub fn asin(self) -> Self {
        // asin is odd and conjugate-symmetric, so it is evaluated in the first
        // quadrant and the component signs are restored (this also places signed
        // zeros on the correct side of the branch cuts).
        let (real, imaginary) = inverse_sine_parts(self.re.abs(), self.im.abs());
        Self::new(real.copysign(self.re), imaginary.copysign(self.im))
    }

    /// Returns the principal inverse cosine.
    #[inline]
    #[must_use]
    pub fn acos(self) -> Self {
        // Compute the small endpoint angle directly, rather than pi/2 - asin.
        let x = self.re.abs();
        let y = self.im.abs();
        let (_, imaginary) = inverse_sine_parts(x, y);
        let angle = if y == T::ZERO && x <= T::ONE {
            x.acos()
        } else if x > T::ONE / T::EPSILON || y > T::ONE / T::EPSILON {
            y.atan2(x)
        } else {
            let (a, _, root_amx) = inverse_distances(x, y);
            (root_amx * (a + x).sqrt()).atan2(x)
        };
        let real = if self.re.is_sign_negative() {
            T::PI - angle
        } else {
            angle
        };
        Self::new(real, -imaginary.copysign(self.im))
    }

    /// Returns the principal inverse tangent.
    #[must_use]
    pub fn atan(self) -> Self {
        let value = Self::new(-self.im, self.re).atanh();
        Self::new(value.im, -value.re)
    }

    /// Returns the principal inverse hyperbolic sine.
    #[inline]
    #[must_use]
    pub fn asinh(self) -> Self {
        let value = Self::new(-self.im, self.re).asin();
        Self::new(value.im, -value.re)
    }

    /// Returns the principal inverse hyperbolic cosine.
    #[inline]
    #[must_use]
    pub fn acosh(self) -> Self {
        let value = self.acos();
        Self::new(value.im.abs(), value.re.copysign(self.im))
    }

    /// Returns the principal inverse hyperbolic tangent.
    #[inline]
    #[must_use]
    pub fn atanh(self) -> Self {
        let x = self.re.abs();
        let y = self.im.abs();
        let one = T::ONE;
        let two = T::TWO;
        if x.is_infinite() || y.is_infinite() {
            return Self::new(
                T::ZERO.copysign(self.re),
                (T::PI * T::HALF).copysign(self.im),
            );
        }
        if self.is_nan() {
            return Self::new(T::NAN, T::NAN);
        }
        // Far from the origin, atanh z = 1/z + i pi/2 + O(1/|z|^3), so the real part
        // is x / |z|^2, formed with scaled components to avoid overflow.
        let scale = x.max(y);
        if scale > one / T::EPSILON {
            let a = x / scale;
            let b = y / scale;
            let real = (a / (a * a + b * b)) / scale;
            return Self::new(real.copysign(self.re), (T::PI * T::HALF).copysign(self.im));
        }
        // Re atanh z = ln(|1 + z| / |1 - z|) / 2. Next to the singularity at 1, the
        // logarithms are taken separately; elsewhere, the ratio of squared
        // distances is rewritten as 1 + 4x / |1 - z|^2 for ln1p. The imaginary part
        // is arg(1 - z^2) / 2, with 1 - x^2 factored to avoid cancellation.
        let distance = (one - x).hypot(y);
        let real = if distance < T::SQRT_EPSILON {
            (log_hypot(one + x, y) - distance.ln()) * T::HALF
        } else {
            let ratio = ((x / distance) / distance) * (two + two);
            ratio.ln_1p() * T::HALF * T::HALF
        };
        let imaginary = (two * y).atan2((one - x) * (one + x) - y * y) * T::HALF;
        Self::new(real.copysign(self.re), imaginary.copysign(self.im))
    }

    /// Computes the principal power `zʷ = e^(w ln z)` for a real exponent `w`.
    #[inline]
    #[must_use]
    pub fn powf(self, exponent: T) -> Self {
        if exponent == T::ZERO {
            return Self::ONE;
        }
        self.power_from_log(Self::new(exponent, T::ZERO))
    }

    /// Computes the principal power `zʷ = e^(w ln z)` for a complex exponent `w`.
    #[inline]
    #[must_use]
    pub fn powc(self, exponent: Self) -> Self {
        if exponent.re == T::ZERO && exponent.im == T::ZERO {
            return Self::ONE;
        }
        self.power_from_log(exponent)
    }

    fn power_from_log(self, exponent: Self) -> Self {
        use crate::ScaledValue;
        let logarithm = self.ln();
        let product = if exponent.im == T::ZERO {
            logarithm.scale(exponent.re)
        } else {
            logarithm * exponent
        };
        // z^w = exp(w ln z). If w ln z overflows even though both factors are
        // finite, the true result may still be finite (for example, with a
        // decaying magnitude or a phase needing reduction). Recompute the product
        // in extended range, reduce the phase from its exact scaled value, and
        // split the exponential around the multiplication.
        if product.is_finite() || !logarithm.is_finite() || !exponent.is_finite() {
            return product.exp();
        }
        match ScaledValue::from_value(logarithm)
            .and_then(|value| value.mul(ScaledValue::from_value(exponent)?))
        {
            Ok(value) => {
                let (sin, cos) = value.imaginary().sin_cos_ieee();
                let growth = value.real().round_ieee();
                Self::new(exp_times(growth, cos), exp_times(growth, sin))
            }
            Err(_) => Self::new(T::NAN, T::NAN),
        }
    }

    /// Converts both components to `R` exactly.
    ///
    /// # Errors
    ///
    /// - `ScalarConversion` if a component is not finite, is out of range, or is
    ///   not exactly representable in `R`.
    pub fn try_cast_exact<R: RealScalar>(self) -> Result<Complex<R>, SciOxideError> {
        // Every supported real converts to f64 exactly, so f64 is a lossless hub
        // and only the final conversion decides exactness.
        let re = self.re.try_to_f64_exact()?;
        let im = self.im.try_to_f64_exact()?;
        Ok(Complex::new(
            R::try_from_f64_exact(re)?,
            R::try_from_f64_exact(im)?,
        ))
    }

    /// Converts both components to `R`, rounding if necessary.
    ///
    /// # Errors
    ///
    /// - `ScalarConversion` if a component is not finite or is outside the
    ///   finite range of `R`.
    pub fn try_cast_lossy<R: RealScalar>(self) -> Result<Complex<R>, SciOxideError> {
        let re = self.re.try_to_f64_exact()?;
        let im = self.im.try_to_f64_exact()?;
        Ok(Complex::new(
            R::try_from_f64_lossy(re)?,
            R::try_from_f64_lossy(im)?,
        ))
    }
}

impl<T: RealScalar + fmt::Display> fmt::Display for Complex<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im.is_sign_negative() {
            write!(formatter, "{}{}i", self.re, self.im)
        } else {
            write!(formatter, "{}+{}i", self.re, self.im)
        }
    }
}

impl<T: Default> From<T> for Complex<T> {
    #[inline]
    fn from(value: T) -> Self {
        Self::new(value, T::default())
    }
}

impl<T: Add<Output = T>> Add for Complex<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl<T: Sub<Output = T>> Sub for Complex<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl<T> Mul for Complex<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl<T: RealScalar> Div for Complex<T> {
    type Output = Self;

    /// Computes `(a + bi) / (c + di)` without spurious overflow or underflow.
    ///
    /// For finite operands and a nonzero divisor, the computed quotient `q̂`
    /// satisfies `|q̂ − q| ≤ 8 ε |q| + 4 η`, where `q` is the exact quotient, `ε`
    /// the machine epsilon and `η` the smallest positive subnormal. If the
    /// components of `q` are representable, no overflow, underflow or NaN
    /// occurs. The Cartesian formula `((ac + bd) + (bc − ad)i) / (c² + d²)` is
    /// used when its intermediates stay in range; otherwise both operands are
    /// first rescaled by exact powers of two. Infinite, NaN and zero divisors
    /// follow C99 Annex G.
    ///
    /// ```
    /// use scioxide_core::Complex64;
    ///
    /// // (c² + d²) = 1e-320 would underflow in the textbook formula.
    /// let q = Complex64::new(1.0, 0.0) / Complex64::new(1e-160, 0.0);
    /// assert_eq!(q, Complex64::new(1e160, 0.0));
    /// let q = Complex64::new(f64::MAX, 1.0) / Complex64::new(2.0, 0.0);
    /// assert_eq!(q, Complex64::new(f64::MAX / 2.0, 0.5));
    /// ```
    fn div(self, rhs: Self) -> Self::Output {
        let numerator = self.re.abs().max(self.im.abs());
        let divisor = rhs.re.abs().max(rhs.im.abs());
        let norm_sqr = rhs.norm_sqr();
        let real = self.re * rhs.re + self.im * rhs.im;
        let imaginary = self.im * rhs.re - self.re * rhs.im;
        // The direct formula is accurate when the denominator is normal, the
        // numerator sums are finite and the dominant product lies far above the
        // subnormal range, so subnormal product rounding is below EPSILON
        // relative to the result. Its rounding is retained for these operands.
        if norm_sqr.is_normal()
            && real.is_finite()
            && imaginary.is_finite()
            && (numerator == T::ZERO || numerator * divisor >= T::MIN_POSITIVE / T::EPSILON)
        {
            return Self::new(real / norm_sqr, imaginary / norm_sqr);
        }

        if self.is_finite() && rhs.is_finite() && divisor != T::ZERO {
            // Exact power-of-two scaling places both largest components in
            // [1, 2). Smaller components may lose only bits below EPSILON of
            // their operand's magnitude, and the final scaling rounds once.
            let left = binary_exponent(numerator);
            let right = binary_exponent(divisor);
            let (a, b) = (self.re.scalbn(-left), self.im.scalbn(-left));
            let (c, d) = (rhs.re.scalbn(-right), rhs.im.scalbn(-right));
            let denominator = c * c + d * d;
            let shift = left - right;
            return Self::new(
                ((a * c + b * d) / denominator).scalbn(shift),
                ((b * c - a * d) / denominator).scalbn(shift),
            );
        }

        // C99 Annex G recovery: replace each infinite component by +/-1 and each
        // finite one by a signed 0 so that the direction survives. finite/inf is
        // then a correctly signed zero, inf/finite a correctly signed infinity,
        // and nonzero/0 an infinity.
        if rhs.is_infinite() && self.is_finite() {
            let real = if rhs.re.is_infinite() {
                T::ONE.copysign(rhs.re)
            } else {
                T::ZERO.copysign(rhs.re)
            };
            let imaginary = if rhs.im.is_infinite() {
                T::ONE.copysign(rhs.im)
            } else {
                T::ZERO.copysign(rhs.im)
            };
            let zero = T::ZERO;
            return Self::new(
                zero * (self.re * real + self.im * imaginary),
                zero * (self.im * real - self.re * imaginary),
            );
        }

        if self.is_infinite() && rhs.is_finite() {
            let real = if self.re.is_infinite() {
                T::ONE.copysign(self.re)
            } else {
                T::ZERO.copysign(self.re)
            };
            let imaginary = if self.im.is_infinite() {
                T::ONE.copysign(self.im)
            } else {
                T::ZERO.copysign(self.im)
            };
            let infinity = T::INFINITY;
            return Self::new(
                infinity * (real * rhs.re + imaginary * rhs.im),
                infinity * (imaginary * rhs.re - real * rhs.im),
            );
        }

        if divisor == T::ZERO && !self.is_nan() {
            let infinity = T::INFINITY.copysign(rhs.re);
            return Self::new(infinity * self.re, infinity * self.im);
        }

        // NaN-containing operands retain component IEEE behavior. The finite
        // and recoverable infinite paths above own scaling and sign recovery.
        let denominator = rhs.norm_sqr();
        Self::new(
            (self.re * rhs.re + self.im * rhs.im) / denominator,
            (self.im * rhs.re - self.re * rhs.im) / denominator,
        )
    }
}

impl<T: Neg<Output = T>> Neg for Complex<T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::new(-self.re, -self.im)
    }
}

impl<T: AddAssign> AddAssign for Complex<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl<T: SubAssign> SubAssign for Complex<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}

impl<T> MulAssign for Complex<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T: RealScalar> DivAssign for Complex<T> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<T: Copy + Mul<Output = T>> Mul<T> for Complex<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl<T: Copy + Div<Output = T>> Div<T> for Complex<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        Self::new(self.re / rhs, self.im / rhs)
    }
}

impl<T: Copy + MulAssign> MulAssign<T> for Complex<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        self.re *= rhs;
        self.im *= rhs;
    }
}

impl<T: Copy + DivAssign> DivAssign<T> for Complex<T> {
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        self.re /= rhs;
        self.im /= rhs;
    }
}

impl<T: RealScalar> Sum for Complex<T> {
    fn sum<I: Iterator<Item = Self>>(values: I) -> Self {
        values.fold(Self::new(T::ZERO, T::ZERO), Add::add)
    }
}

impl<T: RealScalar> Product for Complex<T> {
    fn product<I: Iterator<Item = Self>>(values: I) -> Self {
        values.fold(Self::ONE, Mul::mul)
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;

    use super::*;

    #[test]
    fn layout_is_two_adjacent_components() {
        assert_eq!(
            std::mem::size_of::<Complex<f32>>(),
            2 * std::mem::size_of::<f32>()
        );
        assert_eq!(
            std::mem::align_of::<Complex<f32>>(),
            std::mem::align_of::<f32>()
        );
    }

    #[test]
    fn checked_construction_rejects_non_finite_components() {
        assert!(Complex::try_new(1.0, -2.0).is_ok());
        assert!(Complex::try_new(f64::NAN, 0.0).is_err());
        assert!(Complex::try_new(0.0, f64::INFINITY).is_err());
    }

    #[test]
    fn division_avoids_intermediate_overflow() {
        let value = Complex::new(f64::MAX, f64::MAX) / Complex::new(f64::MAX, f64::MAX);
        assert_eq!(value, Complex::new(1.0, 0.0));
    }

    // Normwise relative error of `q` against a reference known exactly.
    fn normwise_error<T: RealScalar>(q: Complex<T>, exact: Complex<T>) -> f64 {
        let error =
            (q.re.into_f64() - exact.re.into_f64()).hypot(q.im.into_f64() - exact.im.into_f64());
        error / exact.re.into_f64().hypot(exact.im.into_f64())
    }

    #[test]
    fn division_is_range_safe_for_representable_quotients() {
        fn check<T: RealScalar>() {
            let c = |re: f64, im: f64| {
                Complex::new(
                    T::try_from_f64_lossy(re).unwrap(),
                    T::try_from_f64_lossy(im).unwrap(),
                )
            };
            let max_exponent = if T::MANTISSA_DIGITS == 24 { 127 } else { 1023 };
            let min_normal = T::MIN_POSITIVE.into_f64();
            let tiny = 2.0_f64.powi(-(max_exponent / 2 + 10));
            let huge = T::MAX.into_f64();
            // (quotient, numerator, divisor) with exactly known quotients.
            let cases = [
                (c(1.0 / tiny, 0.0), c(1.0, 0.0), c(tiny, 0.0)),
                (c(1.0 / tiny, 0.0), c(1.0, 1.0), c(tiny, tiny)),
                (c(huge / 2.0, 0.5), c(huge, 1.0), c(2.0, 0.0)),
                (c(huge / 4.0, huge / 4.0), c(0.0, huge), c(2.0, 2.0)),
                (c(min_normal * 4.0, 0.0), c(min_normal, 0.0), c(0.25, 0.0)),
                (c(2.0, -3.0), c(2.0 * tiny, -3.0 * tiny), c(tiny, 0.0)),
                (
                    c(0.5, -0.5),
                    c(T::MIN_SUBNORMAL.into_f64(), 0.0),
                    c(T::MIN_SUBNORMAL.into_f64(), T::MIN_SUBNORMAL.into_f64()),
                ),
            ];
            for (exact, numerator, divisor) in cases {
                let q = numerator / divisor;
                assert!(q.is_finite(), "{numerator:?}/{divisor:?} = {q:?}");
                let error = normwise_error(q, exact);
                assert!(
                    error <= 8.0 * T::EPSILON.into_f64(),
                    "{numerator:?}/{divisor:?} = {q:?}, want {exact:?} ({error:e})"
                );
                let mut assigned = numerator;
                assigned /= divisor;
                assert_eq!(assigned, q);
            }
            let inverse = c(tiny, 0.0).inv();
            assert_eq!(inverse, c(1.0 / tiny, 0.0));
            // Zero numerators keep zero quotients over subnormal divisors.
            let zero = c(0.0, 0.0) / Complex::new(T::MIN_SUBNORMAL, T::ZERO);
            assert_eq!(zero, c(0.0, 0.0));
        }
        check::<f32>();
        check::<f64>();
    }

    #[test]
    fn division_commutes_with_exact_binary_scaling() {
        fn check<T: RealScalar>(span: i32) {
            let pairs = [
                (Complex::new(T::ONE, T::TWO), Complex::new(T::TEN, -T::ONE)),
                (
                    Complex::new(-T::TEN, T::HALF),
                    Complex::new(T::HALF, T::TWO),
                ),
                (Complex::new(T::TWO, T::ZERO), Complex::new(T::ONE, T::TEN)),
            ];
            for (z, w) in pairs {
                let reference = z / w;
                for k in (-span..=span).step_by(7) {
                    for m in (-span..=span).step_by(11) {
                        let scaled = Complex::new(z.re.scalbn(k), z.im.scalbn(k))
                            / Complex::new(w.re.scalbn(m), w.im.scalbn(m));
                        let expected =
                            Complex::new(reference.re.scalbn(k - m), reference.im.scalbn(k - m));
                        // Only compare results that stay in the normal range,
                        // where binary scaling commutes with rounding exactly.
                        if expected.re.is_normal() && expected.im.is_normal() {
                            assert_eq!(scaled, expected, "k={k}, m={m}, {z:?}/{w:?}");
                        }
                    }
                }
            }
        }
        check::<f32>(120);
        check::<f64>(1000);
    }

    #[test]
    fn square_root_observes_the_negative_real_branch_cut() {
        let above = Complex::new(-4.0_f64, 0.0).sqrt();
        let below = Complex::new(-4.0_f64, -0.0).sqrt();
        assert_eq!(above, Complex::new(0.0, 2.0));
        assert_eq!(below, Complex::new(0.0, -2.0));
        assert!(below.im.is_sign_negative());
    }

    #[test]
    fn exact_and_lossy_casts_have_distinct_contracts() {
        let value = Complex::new(0.1_f64, -0.0);
        assert!(value.try_cast_exact::<f32>().is_err());
        let narrowed = value.try_cast_lossy::<f32>().unwrap();
        assert_eq!(narrowed.re, 0.1_f32);
        assert!(narrowed.im.is_sign_negative());
    }

    #[test]
    fn display_is_stable_and_explicit_about_the_sign() {
        assert_eq!(Complex::new(1.5_f64, 2.0).to_string(), "1.5+2i");
        assert_eq!(Complex::new(1.5_f64, -2.0).to_string(), "1.5-2i");
        assert_eq!(Complex::new(1.5_f64, -0.0).to_string(), "1.5-0i");
    }

    #[test]
    fn algebraic_and_elementary_identities_hold() {
        let value = Complex::new(0.75_f64, -1.25);
        let other = Complex::new(-0.5, 2.0);
        let product = value * value.conj();
        assert_eq!(product.im.to_bits(), 0.0_f64.to_bits());
        assert!((product.re - value.norm_sqr()).abs() <= 2.0 * f64::EPSILON);
        assert_eq!((value + other) - other, value);
        assert_eq!(value.powu(0), Complex::new(1.0, 0.0));

        let round_trip = value.ln().exp();
        let tolerance = 128.0 * f64::EPSILON * value.norm().max(1.0);
        assert!((round_trip.re - value.re).abs() <= tolerance);
        assert!((round_trip.im - value.im).abs() <= tolerance);

        let square_root = value.sqrt();
        let reconstructed = square_root * square_root;
        assert!((reconstructed.re - value.re).abs() <= tolerance);
        assert!((reconstructed.im - value.im).abs() <= tolerance);

        let quotient = value / other;
        let reconstructed = quotient * other;
        assert!((reconstructed.re - value.re).abs() <= tolerance);
        assert!((reconstructed.im - value.im).abs() <= tolerance);
    }

    #[test]
    fn signed_zero_nan_and_infinity_are_classified_and_recovered() {
        let upper_cut = Complex::new(-1.0_f64, 0.0).arg();
        let lower_cut = Complex::new(-1.0_f64, -0.0).arg();
        assert_eq!(upper_cut, core::f64::consts::PI);
        assert_eq!(lower_cut, -core::f64::consts::PI);

        assert!(Complex::new(f64::NAN, f64::INFINITY).is_nan());
        assert!(!Complex::new(f64::NAN, f64::INFINITY).is_infinite());
        assert!(Complex::new(f64::INFINITY, 0.0).is_infinite());

        let quotient = Complex::new(3.0_f64, -4.0) / Complex::new(f64::INFINITY, 0.0);
        assert_eq!(quotient.re.to_bits(), 0.0_f64.to_bits());
        assert_eq!(quotient.im.to_bits(), (-0.0_f64).to_bits());

        let root = Complex::new(1.0_f64, f64::NEG_INFINITY).sqrt();
        assert_eq!(root.re, f64::INFINITY);
        assert_eq!(root.im, f64::NEG_INFINITY);
    }
}
