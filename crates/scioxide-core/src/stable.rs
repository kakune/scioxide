//! Elementary functions that avoid cancellation and overflow.

use crate::{DomainReason, NonFiniteOrigin, RealScalar, SciOxideError};
use core::f64::consts::PI;

/// Computes `log1mexp(x) = ln(1 − eˣ)` for `x ≤ 0` without cancellation.
///
/// Uses `ln(−expm1(x))` for `x ≥ −ln 2` and `ln1p(−eˣ)` below it, in O(1)
/// time and storage. Both signed zeros return `−∞` (the
/// logarithm of zero). `−∞` returns `−0` (the logarithm of one), and very
/// negative finite inputs may also round to `−0`.
///
/// # Accuracy
///
/// For finite `x < 0`, absolute error is at most `32 ε |f(x)| + 8 η`, where
/// `f(x) = ln(1 − eˣ)`, `ε` is scalar machine epsilon and `η` is its smallest
/// positive subnormal. This bound includes gradual underflow in the left tail.
/// The switch keeps each subtraction inside `expm1` or `log1p`; composing
/// their elementary error bounds does not require a relative bound on `1 − eˣ`.
///
/// # Errors
///
/// - `NonFinite` if `x` is NaN or `+∞`.
/// - `DomainError` if `x > 0`.
///
/// # References
///
/// M. Mächler, *Accurately Computing log(1 − exp(−|a|))*, 2012,
/// [logarithmic-complement note](https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf).
///
/// # Examples
///
/// ```
/// // ln(1 − e^(−ln 2)) = ln(1/2) = −ln 2
/// let value = scioxide_core::log1mexp(-core::f64::consts::LN_2)?;
/// assert!((value + core::f64::consts::LN_2).abs() < 1e-15);
/// assert_eq!(scioxide_core::log1mexp(0.0_f64)?, f64::NEG_INFINITY);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline(always)]
pub fn log1mexp<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    if x == S::NEG_INFINITY {
        return Ok(-S::ZERO);
    }
    if !x.is_finite() {
        return Err(SciOxideError::non_finite(
            NonFiniteOrigin::Input,
            "logarithmic complement input must not be NaN or positive infinity",
        ));
    }
    if x > S::ZERO {
        return Err(domain(crate::argument!("x")));
    }
    if x == S::ZERO {
        return Ok(S::NEG_INFINITY);
    }
    Ok(if x < -S::LN_2 {
        (-x.exp()).ln_1p()
    } else {
        (-x.exp_m1()).ln()
    })
}

#[inline]
pub(super) fn input<S: RealScalar>(x: S) -> Result<f64, SciOxideError> {
    if x.is_finite() {
        Ok(x.into_f64())
    } else {
        Err(SciOxideError::non_finite(
            NonFiniteOrigin::Input,
            "transform input must be finite",
        ))
    }
}
pub(super) fn domain(argument: crate::Argument) -> SciOxideError {
    SciOxideError::domain_error(
        argument,
        DomainReason::OutsideDomain,
        "argument is outside the transform domain",
    )
}
// All transforms evaluate in binary64 and round once to the destination.
// Finiteness is checked both before narrowing and after it, because a
// finite binary64 result can still overflow binary32.
#[inline]
pub(super) fn output<S: RealScalar>(x: f64) -> Result<S, SciOxideError> {
    crate::backend::checked_final_rounding(x, "transform result exceeds scalar range")?;
    let result = S::from_f64_ieee(x);
    crate::backend::checked_final_rounding(result, "transform result exceeds scalar range")
}

fn softplus_value(x: f64) -> f64 {
    // ln(1 + e^x) = max(x, 0) + ln1p(e^-|x|): the exponential argument is never
    // positive, so nothing overflows, and ln1p keeps the tail accurate when
    // e^-|x| is tiny.
    x.max(0.0) + RealScalar::ln_1p(RealScalar::exp(-x.abs()))
}
fn sigmoid_value(x: f64) -> f64 {
    // With e = e^-|x| <= 1, the negative side uses e / (1 + e) rather than
    // 1 - 1 / (1 + e), keeping full relative accuracy in the far left tail.
    let e = RealScalar::exp(-x.abs());
    if x < 0.0 {
        e / (1.0 + e)
    } else {
        1.0 / (1.0 + e)
    }
}

fn real_softplus<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    let x = input(x)?;
    output(softplus_value(x))
}

fn real_sigmoid<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    let x = input(x)?;
    output(sigmoid_value(x))
}

fn real_log_sigmoid<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    let x = input(x)?;
    output(-softplus_value(-x))
}

#[inline]
fn real_logit<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    let x = input(x)?;
    output({
        if x <= 0.0 || x >= 1.0 {
            return Err(domain(crate::argument!("x")));
        }
        if (0.4375..=0.5625).contains(&x) {
            // 2*x is exact; subtracting 1 is exact by Sterbenz on this
            // interval. Forming the small correction before division
            // preserves the destination lattice next to one half.
            let u = 2.0 * x - 1.0;
            RealScalar::ln_1p(u / (1.0 - x))
        } else {
            // Outside this interval |logit(x)| >= ln(9/7) > 1/4.
            // Subtraction/division rounding cannot cause cancellation
            // and fits the transform envelope with one logarithm.
            RealScalar::ln(x / (1.0 - x))
        }
    })
}

fn real_cosm1<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    let x = input(x)?;
    output({
        // cos x - 1 = -2 sin^2(x / 2) has no cancellation near zero.
        let s = RealScalar::sin(x * 0.5);
        -2.0 * s * s
    })
}

fn real_sinc<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    let x = input(x)?;
    output(if x == 0.0 {
        1.0
    } else {
        RealScalar::sin(x) / x
    })
}

fn real_exprel<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    let x = input(x)?;
    output(if x == 0.0 {
        1.0
    // Near ln(f64::MAX) ~ 709.8, e^x overflows while (e^x - 1) / x ~ e^x / x may
    // not, so the exponential is split in halves around the division. The -1
    // is far below rounding here.
    } else if x > 700.0 {
        let e = RealScalar::exp(x * 0.5);
        (e / x) * e
    } else {
        RealScalar::exp_m1(x) / x
    })
}

#[inline]
fn real_logaddexp<S: RealScalar>(x: S, y: S) -> Result<S, SciOxideError> {
    crate::logarithmic::two_term_logarithm(x, y)
}

#[inline]
fn real_xlogy<S: RealScalar>(x: S, y: S) -> Result<S, SciOxideError> {
    let x = input(x)?;
    let y = input(y)?;
    // 0 * ln(y) is defined as 0 for every y, including y <= 0, which is the
    // point of xlogy.
    if x == 0.0 {
        return output(x);
    }
    if y <= 0.0 {
        return Err(domain(crate::argument!("y")));
    }
    output(x * RealScalar::ln(y))
}

#[inline]
fn real_xlog1py<S: RealScalar>(x: S, y: S) -> Result<S, SciOxideError> {
    let x = input(x)?;
    let y = input(y)?;
    if x == 0.0 {
        return output(x);
    }
    if y <= -1.0 {
        return Err(domain(crate::argument!("y")));
    }
    // For |y| >= 1/4, rounding 1+y perturbs its logarithm by at most
    // four destination ULPs in binary64. Including the elementary four-ULP
    // envelope and multiplication stays inside the transform's 16 ULPs.
    // Small y still uses log1p, preserving zero signs and subnormal tails.
    let logarithm = if y.abs() >= 0.25 {
        RealScalar::ln(1.0 + y)
    } else {
        RealScalar::ln_1p(y)
    };
    output(x * logarithm)
}

#[inline]
fn real_powm1<S: RealScalar>(x: S, y: S) -> Result<S, SciOxideError> {
    let x = input(x)?;
    let y = input(y)?;
    if y == 0.0 || x == 1.0 {
        return output(0.0);
    }
    if x == 0.0 {
        return if y > 0.0 {
            output(-1.0)
        } else {
            Err(domain(crate::argument!("y")))
        };
    }
    if x < 0.0 && RealScalar::trunc(y) != y {
        return Err(domain(crate::argument!("y")));
    }
    // For a in [0.5, 1.5], a - 1 is exact (Sterbenz), so ln1p gives a small
    // logarithm with full relative accuracy. expm1(y ln|x|) then avoids the
    // cancellation of x^y - 1 when the result is near zero. A negative base
    // with an odd exponent gives a negative power, for which the expm1 form
    // has the wrong sign. For |z| > 1/2, |x^y - 1| >= 1 - e^(-1/2), so pow - 1
    // has no harmful cancellation.
    let a = x.abs();
    // Integrating 1/t gives |ln(a)| >= |a-1|/max(a,1). The slack
    // above 1/2 covers all three rounded operations in this lower estimate.
    // Odd negative powers cannot cancel against +1 either.
    if (x < 0.0 && y % 2.0 != 0.0)
        || y.abs() * ((a - 1.0).abs() / a.max(1.0)) > 0.5 * (1.0 + 8.0 * f64::EPSILON)
    {
        return output(RealScalar::powf(x, y) - 1.0);
    }
    let log = if (0.5..=1.5).contains(&a) {
        RealScalar::ln_1p(a - 1.0)
    } else {
        RealScalar::ln(a)
    };
    let z = y * log;
    let result = if z.abs() > 0.5 {
        RealScalar::powf(x, y) - 1.0
    } else {
        RealScalar::exp_m1(z)
    };
    output(result)
}

// Finite input, in multiples of pi or degrees. The common reduction uses
// nearest-even integer extraction, without libm round or fmod. Quarter-turn
// products are exact within the bound below; subtraction is exact by Sterbenz.
#[inline]
fn reduced_phase(x: f64, half_turn: f64) -> (i32, f64) {
    let a = x.abs();
    if half_turn == 1.0 && a >= 1_125_899_906_842_624.0 {
        // Above 2^50 the lattice is at least 1/4; above 2^53 all inputs
        // are even integers. Extract parity without overflowing a conversion.
        if a >= 9_007_199_254_740_992.0 {
            return (0, 0.0);
        }
        let q = (2.0 * a) as u64;
        let residual = a - 0.5 * q as f64;
        let sign = if x.is_sign_negative() { -1 } else { 1 };
        return (sign * (q & 3) as i32, residual.copysign(x));
    }
    // Large degree inputs require exact modular reduction: 360 is not a
    // power of two. The remainder retains every bit of a small residual.
    let r = if half_turn != 1.0 && a >= 35_184_372_088_832.0 {
        x % 360.0
    } else {
        x
    };
    let quarter = half_turn * 0.5;
    let count = r.abs() / quarter;
    let q = ((count + 4_503_599_627_370_496.0) - 4_503_599_627_370_496.0).copysign(r);
    ((q as i64 & 3) as i32, r - q * quarter)
}

// Reduction is performed in the input's units, before multiplication by pi.
#[inline]
pub(super) fn phase(x: f64, half_turn: f64) -> (f64, f64) {
    let (q, residual) = reduced_phase(x, half_turn);
    let (s, c) = RealScalar::sin_cos(radians(residual, half_turn));
    let (s, c) = match q & 3 {
        1 => (c, -s),
        2 => (-s, -c),
        3 => (-c, s),
        _ => (s, c),
    };
    (
        if s == 0.0 { 0.0_f64.copysign(x) } else { s },
        if c == 0.0 { 0.0 } else { c },
    )
}

#[inline]
fn phase_component<const COSINE: bool>(x: f64, half_turn: f64) -> f64 {
    let (q, residual) = reduced_phase(x, half_turn);
    let quadrant = (q + i32::from(COSINE)) & 3;
    let angle = radians(residual, half_turn);
    let magnitude = if quadrant & 1 == 0 {
        RealScalar::sin(angle)
    } else {
        RealScalar::cos(angle)
    };
    if magnitude == 0.0 {
        if COSINE { 0.0 } else { 0.0_f64.copysign(x) }
    } else if quadrant & 2 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

// Conversion happens before destination projection, rather than dividing a
// tiny residual by 180 first. Scaling to the normal range avoids intermediate
// subnormal rounding. Both real phase reduction and complex imaginary phases
// use this conversion owner.
#[inline]
pub(super) fn radians(x: f64, half_turn: f64) -> f64 {
    let factor = PI / half_turn;
    if x != 0.0 && x.abs() < f64::MIN_POSITIVE * 256.0 {
        RealScalar::scalbn(RealScalar::scalbn(x, 512) * factor, -512)
    } else {
        x * factor
    }
}

#[inline]
fn real_sincpi<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    let x = input(x)?;
    output(if x.abs() < 3.725_290_298_461_914e-9 {
        // |sinc(pi*x)-1| < pi²*x²/6 < 1/2 ULP at one.
        1.0
    } else if x.abs() < 0.25 {
        let z = PI * x;
        RealScalar::sin(z) / z
    } else {
        let (q, residual) = reduced_phase(x, 1.0);
        let sine = crate::math::reduced_pi_sine(residual, q);
        if sine == 0.0 {
            0.0
        } else {
            // Noninteger binary64 inputs have |x| < 2^52, so the
            // denominator and its product residual are always finite.
            let denominator = PI * x;
            let tail = RealScalar::mul_add(PI, x, -denominator) + crate::math::PI_TAIL * x;
            let quotient = sine / denominator;
            let residual = RealScalar::mul_add(-quotient, denominator, sine) - quotient * tail;
            quotient + residual / denominator
        }
    })
}

#[inline]
fn real_sin_cos_pi<S: RealScalar>(x: S) -> Result<(S, S), SciOxideError> {
    let (s, c) = phase(input(x)?, 1.0);
    Ok((output(s)?, output(c)?))
}

#[inline]
fn real_sin_pi<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    let s = phase_component::<false>(input(x)?, 1.0);
    output(s)
}

#[inline]
fn real_cos_pi<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    let c = phase_component::<true>(input(x)?, 1.0);
    output(c)
}

#[inline]
fn real_tan_pi<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    let (s, c) = phase(input(x)?, 1.0);
    output({
        if c == 0.0 {
            return Err(domain(crate::argument!("x")));
        }
        s / c
    })
}

#[inline]
fn real_sin_cos_degrees<S: RealScalar>(x: S) -> Result<(S, S), SciOxideError> {
    let (s, c) = phase(input(x)?, 180.0);
    Ok((output(s)?, output(c)?))
}

#[inline]
fn real_sin_degrees<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    let s = phase_component::<false>(input(x)?, 180.0);
    output(s)
}

#[inline]
fn real_cos_degrees<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    let c = phase_component::<true>(input(x)?, 180.0);
    output(c)
}

#[inline]
fn real_tan_degrees<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    let (s, c) = phase(input(x)?, 180.0);
    output({
        if c == 0.0 {
            return Err(domain(crate::argument!("x")));
        }
        s / c
    })
}

// Shared projections own real/complex width selection; numerical kernels stay explicit.
macro_rules! scalar_dispatch {
    (unary $t:ident, $x:expr, $real:path, $complex:expr) => {{
        use crate::backend::ScalarValue;
        match $t::value($x) {
            ScalarValue::F32(x) => $t::from_value(ScalarValue::F32($real(x)?)),
            ScalarValue::F64(x) => $t::from_value(ScalarValue::F64($real(x)?)),
            ScalarValue::Complex32(x) => $t::from_value(ScalarValue::Complex32(($complex)(x)?)),
            ScalarValue::Complex64(x) => $t::from_value(ScalarValue::Complex64(($complex)(x)?)),
        }
    }};
    (binary $t:ident, $x:expr, $y:expr, $real:path, $complex:expr) => {{
        use crate::backend::ScalarValue;
        let y = $t::value($y);
        match $t::value($x) {
            ScalarValue::F32(x) => $t::from_value(ScalarValue::F32($real(
                x,
                <f32 as crate::backend::Scalar>::from_value(y)?,
            )?)),
            ScalarValue::F64(x) => $t::from_value(ScalarValue::F64($real(
                x,
                <f64 as crate::backend::Scalar>::from_value(y)?,
            )?)),
            ScalarValue::Complex32(x) => $t::from_value(ScalarValue::Complex32(($complex)(
                x,
                <crate::Complex32 as crate::backend::Scalar>::from_value(y)?,
            )?)),
            ScalarValue::Complex64(x) => $t::from_value(ScalarValue::Complex64(($complex)(
                x,
                <crate::Complex64 as crate::backend::Scalar>::from_value(y)?,
            )?)),
        }
    }};
    (pair $t:ident, $x:expr, $real:path, $complex:expr) => {{
        use crate::backend::ScalarValue;
        match $t::value($x) {
            ScalarValue::F32(x) => {
                let (a, b) = $real(x)?;
                Ok((
                    $t::from_value(ScalarValue::F32(a))?,
                    $t::from_value(ScalarValue::F32(b))?,
                ))
            }
            ScalarValue::F64(x) => {
                let (a, b) = $real(x)?;
                Ok((
                    $t::from_value(ScalarValue::F64(a))?,
                    $t::from_value(ScalarValue::F64(b))?,
                ))
            }
            ScalarValue::Complex32(x) => {
                let (a, b) = ($complex)(x)?;
                Ok((
                    $t::from_value(ScalarValue::Complex32(a))?,
                    $t::from_value(ScalarValue::Complex32(b))?,
                ))
            }
            ScalarValue::Complex64(x) => {
                let (a, b) = ($complex)(x)?;
                Ok((
                    $t::from_value(ScalarValue::Complex64(a))?,
                    $t::from_value(ScalarValue::Complex64(b))?,
                ))
            }
        }
    }};
}

/// Computes the softplus function `softplus(x) = ln(1 + eˣ)`.
///
/// Real arguments use `max(x, 0) + ln1p(e^(−|x|))`, so no exponential
/// overflows. Complex arguments use the principal logarithm of `1 + eᶻ`.
///
/// Arguments are evaluated in `f64` and the result is rounded once to the
/// output type.
///
/// # Accuracy
///
/// Real results are within 16 ulps. For complex `z`, each component of the
/// result is within `32 ε m + 8 η` of the exact value, where `m` is
/// the larger component magnitude of the exact value, `ε` is the machine
/// epsilon and `η` the smallest positive subnormal.
///
/// # Errors
///
/// - `NonFinite` if an input component is NaN or infinite.
/// - `NumericalInstability` if a result component overflows, including at the
///   singularities of the complex function, where `1 + eᶻ = 0`.
///
/// # References
///
/// NIST DLMF, [§4.2, logarithms and exponentials](https://dlmf.nist.gov/4.2),
/// and [§4.6, power series](https://dlmf.nist.gov/4.6).
///
/// # Examples
///
/// ```
/// // softplus(0) = ln 2
/// let y = scioxide_core::softplus(0.0_f64)?;
/// assert!((y - core::f64::consts::LN_2).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn softplus<T: crate::NumericScalar>(x: T) -> Result<T, SciOxideError> {
    scalar_dispatch!(unary T, x,  real_softplus, crate::stable_complex::softplus)
}

/// Computes the logistic function `σ(x) = 1 / (1 + e^(−x))`.
///
/// Real arguments exponentiate only `−|x|`, so neither tail overflows and the
/// left tail keeps full relative accuracy. Complex arguments evaluate the
/// analytic continuation, which has poles where `1 + e^(−z) = 0`.
///
/// Arguments are evaluated in `f64` and the result is rounded once to the
/// output type.
///
/// # Accuracy
///
/// Real results are within 16 ulps. For complex `z`, each component of the
/// result is within `32 ε m + 8 η` of the exact value, where `m` is
/// the larger component magnitude of the exact value, `ε` is the machine
/// epsilon and `η` the smallest positive subnormal.
///
/// # Errors
///
/// - `NonFinite` if an input component is NaN or infinite.
/// - `NumericalInstability` if a result component overflows, including at the
///   poles of the complex function.
///
/// # References
///
/// NIST DLMF, [§4.2, logarithms and exponentials](https://dlmf.nist.gov/4.2),
/// and [§4.6, power series](https://dlmf.nist.gov/4.6).
///
/// # Examples
///
/// ```
/// // σ(0) = 1/2
/// assert_eq!(scioxide_core::sigmoid(0.0_f64)?, 0.5);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn sigmoid<T: crate::NumericScalar>(x: T) -> Result<T, SciOxideError> {
    scalar_dispatch!(unary T, x,  real_sigmoid, crate::stable_complex::sigmoid)
}

/// Computes `ln σ(x) = −ln(1 + e^(−x))`, the logarithm of the logistic function.
///
/// Real arguments keep full relative accuracy in both tails. Complex arguments
/// return the principal logarithm.
///
/// Arguments are evaluated in `f64` and the result is rounded once to the
/// output type.
///
/// # Accuracy
///
/// Real results are within 16 ulps. For complex `z`, each component of the
/// result is within `32 ε m + 8 η` of the exact value, where `m` is
/// the larger component magnitude of the exact value, `ε` is the machine
/// epsilon and `η` the smallest positive subnormal.
///
/// # Errors
///
/// - `NonFinite` if an input component is NaN or infinite.
/// - `NumericalInstability` if a result component overflows.
///
/// # References
///
/// NIST DLMF, [§4.2, logarithms and exponentials](https://dlmf.nist.gov/4.2),
/// and [§4.6, power series](https://dlmf.nist.gov/4.6).
///
/// # Examples
///
/// ```
/// // ln σ(0) = −ln 2
/// let y = scioxide_core::log_sigmoid(0.0_f64)?;
/// assert!((y + core::f64::consts::LN_2).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn log_sigmoid<T: crate::NumericScalar>(x: T) -> Result<T, SciOxideError> {
    scalar_dispatch!(unary T, x,  real_log_sigmoid, crate::stable_complex::log_sigmoid)
}

/// Computes the logit function `logit(x) = ln(x / (1 − x))`, the inverse of [`sigmoid`].
///
/// Real arguments must satisfy `0 < x < 1`. Near `x = 1/2` the function is
/// evaluated without cancellation, and `logit(1/2)` is `+0`. Complex arguments
/// use the principal logarithms `ln z − ln(1 − z)`.
///
/// Arguments are evaluated in `f64` and the result is rounded once to the
/// output type.
///
/// # Accuracy
///
/// Real results are within 16 ulps. For complex `z`, each component of the
/// result is within `32 ε m + 8 η` of the exact value, where `m` is
/// the larger component magnitude of the exact value, `ε` is the machine
/// epsilon and `η` the smallest positive subnormal.
///
/// # Errors
///
/// - `NonFinite` if an input component is NaN or infinite.
/// - `DomainError` if a real `x` is outside `(0, 1)`, or if a complex `z` is 0 or 1.
/// - `NumericalInstability` if a result component overflows.
///
/// # References
///
/// NIST DLMF, [§4.2, logarithms and exponentials](https://dlmf.nist.gov/4.2),
/// and [§4.6, power series](https://dlmf.nist.gov/4.6).
///
/// # Examples
///
/// ```
/// assert_eq!(scioxide_core::logit(0.5_f64)?, 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline]
pub fn logit<T: crate::NumericScalar>(x: T) -> Result<T, SciOxideError> {
    scalar_dispatch!(unary T, x,  real_logit, crate::stable_complex::logit)
}

/// Computes `cos(x) − 1` without cancellation near zero.
///
/// Real arguments use `−2 sin²(x/2)`.
///
/// Arguments are evaluated in `f64` and the result is rounded once to the
/// output type.
///
/// # Accuracy
///
/// Real results are within 16 ulps. For complex `z`, each component of the
/// result is within `32 ε m g + 8 η` of the exact value, where `m` is
/// the larger component magnitude of the exact value, `ε` is the machine
/// epsilon and `η` the smallest positive subnormal,
/// and `g = 1 + |Im z|`.
///
/// # Errors
///
/// - `NonFinite` if an input component is NaN or infinite.
/// - `NumericalInstability` if a result component overflows.
///
/// # References
///
/// NIST DLMF, [§4.21, trigonometric identities](https://dlmf.nist.gov/4.21),
/// and [§4.45, methods of computation](https://dlmf.nist.gov/4.45).
///
/// # Examples
///
/// ```
/// // cos(x) − 1 ≈ −x²/2 for small x
/// let y = scioxide_core::cosm1(1e-10_f64)?;
/// assert!((y + 5e-21).abs() < 1e-33);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn cosm1<T: crate::NumericScalar>(x: T) -> Result<T, SciOxideError> {
    scalar_dispatch!(unary T, x,  real_cosm1, crate::stable_complex::cosm1)
}

/// Computes the unnormalized sinc function `sinc(x) = sin(x) / x`, with `sinc(0) = 1`.
///
/// For the normalized function `sin(πx) / (πx)`, use [`sincpi`].
///
/// Arguments are evaluated in `f64` and the result is rounded once to the
/// output type.
///
/// # Accuracy
///
/// Real results are within 16 ulps. For complex `z`, each component of the
/// result is within `32 ε m g + 8 η` of the exact value, where `m` is
/// the larger component magnitude of the exact value, `ε` is the machine
/// epsilon and `η` the smallest positive subnormal,
/// and `g = 1 + |Im z|`.
///
/// # Errors
///
/// - `NonFinite` if an input component is NaN or infinite.
/// - `NumericalInstability` if a result component overflows.
///
/// # References
///
/// NIST DLMF, [§4.21, trigonometric identities](https://dlmf.nist.gov/4.21),
/// and [§4.45, methods of computation](https://dlmf.nist.gov/4.45).
///
/// # Examples
///
/// ```
/// assert_eq!(scioxide_core::sinc(0.0_f64)?, 1.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn sinc<T: crate::NumericScalar>(x: T) -> Result<T, SciOxideError> {
    scalar_dispatch!(unary T, x,  real_sinc, crate::stable_complex::sinc)
}

/// Computes the relative exponential `exprel(x) = (eˣ − 1) / x`, with `exprel(0) = 1`.
///
/// Real arguments near `ln(f64::MAX)` split the exponential, so the result is
/// finite whenever `eˣ / x` is.
///
/// Arguments are evaluated in `f64` and the result is rounded once to the
/// output type.
///
/// # Accuracy
///
/// Real results are within 16 ulps. For complex `z`, each component of the
/// result is within `32 ε m g + 8 η` of the exact value, where `m` is
/// the larger component magnitude of the exact value, `ε` is the machine
/// epsilon and `η` the smallest positive subnormal,
/// and `g = 1 + |Re z|`.
///
/// # Errors
///
/// - `NonFinite` if an input component is NaN or infinite.
/// - `NumericalInstability` if a result component overflows.
///
/// # References
///
/// NIST DLMF, [§4.2, logarithms and exponentials](https://dlmf.nist.gov/4.2),
/// and [§4.6, power series](https://dlmf.nist.gov/4.6).
///
/// # Examples
///
/// ```
/// assert_eq!(scioxide_core::exprel(0.0_f64)?, 1.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
pub fn exprel<T: crate::NumericScalar>(x: T) -> Result<T, SciOxideError> {
    scalar_dispatch!(unary T, x,  real_exprel, crate::stable_complex::exprel)
}

/// Computes `x ln(y)`, defined as zero when `x = 0`.
///
/// A zero `x` returns zero for every finite `y`, including `y ≤ 0`. Complex
/// arguments use the principal logarithm.
///
/// Arguments are evaluated in `f64` and the result is rounded once to the
/// output type.
///
/// # Accuracy
///
/// Real results are within 16 ulps. For complex `z`, each component of the
/// result is within `32 ε m + 8 η` of the exact value, where `m` is
/// the larger component magnitude of the exact value, `ε` is the machine
/// epsilon and `η` the smallest positive subnormal.
///
/// # Errors
///
/// - `NonFinite` if an input component is NaN or infinite.
/// - `DomainError` if `x ≠ 0` and a real `y ≤ 0`, or a complex `y = 0`.
/// - `NumericalInstability` if a result component overflows.
///
/// # References
///
/// NIST DLMF, [§4.2, logarithms and exponentials](https://dlmf.nist.gov/4.2),
/// and [§4.6, power series](https://dlmf.nist.gov/4.6).
///
/// # Examples
///
/// ```
/// assert_eq!(scioxide_core::xlogy(0.0_f64, 0.0)?, 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline]
pub fn xlogy<T: crate::NumericScalar>(x: T, y: T) -> Result<T, SciOxideError> {
    scalar_dispatch!(binary T, x, y,  real_xlogy, crate::stable_complex::xlogy)
}

/// Computes `x ln(1 + y)`, defined as zero when `x = 0`.
///
/// A zero `x` returns zero for every finite `y`. Small `y` keeps full relative
/// accuracy. Complex arguments use the principal logarithm.
///
/// Arguments are evaluated in `f64` and the result is rounded once to the
/// output type.
///
/// # Accuracy
///
/// Real results are within 16 ulps. For complex `z`, each component of the
/// result is within `32 ε m + 8 η` of the exact value, where `m` is
/// the larger component magnitude of the exact value, `ε` is the machine
/// epsilon and `η` the smallest positive subnormal.
///
/// # Errors
///
/// - `NonFinite` if an input component is NaN or infinite.
/// - `DomainError` if `x ≠ 0` and a real `y ≤ −1`, or a complex `y = −1`.
/// - `NumericalInstability` if a result component overflows.
///
/// # References
///
/// NIST DLMF, [§4.2, logarithms and exponentials](https://dlmf.nist.gov/4.2),
/// and [§4.6, power series](https://dlmf.nist.gov/4.6).
///
/// # Examples
///
/// ```
/// assert_eq!(scioxide_core::xlog1py(0.0_f64, -2.0)?, 0.0);
/// // 2 ln(1 + 10⁻²⁰) ≈ 2 × 10⁻²⁰
/// let tiny = scioxide_core::xlog1py(2.0_f64, 1e-20)?;
/// assert!((tiny - 2e-20).abs() < 1e-33);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline]
pub fn xlog1py<T: crate::NumericScalar>(x: T, y: T) -> Result<T, SciOxideError> {
    scalar_dispatch!(binary T, x, y,  real_xlog1py, crate::stable_complex::xlog1py)
}

/// Computes `xʸ − 1` without cancellation when `xʸ` is close to one.
///
/// For real arguments, `xʸ − 1 = 0` when `y = 0` or `x = 1`. A zero base needs
/// `y > 0`, and a negative base needs an integer `y`. Complex arguments use the
/// principal logarithm, `xʸ = e^(y ln x)`.
///
/// Arguments are evaluated in `f64` and the result is rounded once to the
/// output type.
///
/// # Accuracy
///
/// Real results are within 16 ulps. For complex `z`, each component of the
/// result is within `32 ε m g + 8 η` of the exact value, where `m` is
/// the larger component magnitude of the exact value, `ε` is the machine
/// epsilon and `η` the smallest positive subnormal,
/// and `g = 1 + |y| |ln x|`.
///
/// # Errors
///
/// - `NonFinite` if an input component is NaN or infinite.
/// - `DomainError` for a real negative `x` with non-integer `y`, or for `x = 0`
///   with `y < 0` (complex: `Re y ≤ 0` and `y ≠ 0`).
/// - `NumericalInstability` if a result component overflows.
///
/// # References
///
/// NIST DLMF, [§4.2, logarithms and exponentials](https://dlmf.nist.gov/4.2),
/// and [§4.6, power series](https://dlmf.nist.gov/4.6).
///
/// # Examples
///
/// ```
/// // 4^0.5 − 1 = 1
/// let y = scioxide_core::powm1(4.0_f64, 0.5)?;
/// assert!((y - 1.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline]
pub fn powm1<T: crate::NumericScalar>(x: T, y: T) -> Result<T, SciOxideError> {
    scalar_dispatch!(binary T, x, y,  real_powm1, crate::stable_complex::powm1)
}

/// Computes `(sin(πx), cos(πx))`.
///
/// The argument is reduced exactly, so integer arguments give exact zeros of
/// the sine and half-integer arguments give exact zeros of the cosine.
///
/// Arguments are evaluated in `f64` and the result is rounded once to the
/// output type.
///
/// # Accuracy
///
/// Real results are within 16 ulps. For complex `z`, each component of the
/// result is within `32 ε m g + 8 η` of the exact value, where `m` is
/// the larger component magnitude of the exact value, `ε` is the machine
/// epsilon and `η` the smallest positive subnormal,
/// and `g = 1 + π |Im z|`.
///
/// # Errors
///
/// - `NonFinite` if an input component is NaN or infinite.
/// - `NumericalInstability` if a result component overflows.
///
/// # References
///
/// NIST DLMF, [§4.21, trigonometric identities](https://dlmf.nist.gov/4.21),
/// and [§4.45, methods of computation](https://dlmf.nist.gov/4.45).
///
/// # Examples
///
/// ```
/// let (s, c) = scioxide_core::sin_cos_pi(0.5_f64)?;
/// assert_eq!((s, c), (1.0, 0.0));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline]
pub fn sin_cos_pi<T: crate::NumericScalar>(x: T) -> Result<(T, T), SciOxideError> {
    scalar_dispatch!(pair T, x,  real_sin_cos_pi, crate::stable_complex::sin_cos_pi)
}

/// Computes `sin(πx)`.
///
/// The argument is reduced exactly, so integer arguments give exact zeros of
/// the sine and half-integer arguments give exact zeros of the cosine.
///
/// Arguments are evaluated in `f64` and the result is rounded once to the
/// output type.
///
/// # Accuracy
///
/// Real results are within 16 ulps. For complex `z`, each component of the
/// result is within `32 ε m g + 8 η` of the exact value, where `m` is
/// the larger component magnitude of the exact value, `ε` is the machine
/// epsilon and `η` the smallest positive subnormal,
/// and `g = 1 + π |Im z|`.
///
/// # Errors
///
/// - `NonFinite` if an input component is NaN or infinite.
/// - `NumericalInstability` if a result component overflows.
///
/// # References
///
/// NIST DLMF, [§4.21, trigonometric identities](https://dlmf.nist.gov/4.21),
/// and [§4.45, methods of computation](https://dlmf.nist.gov/4.45).
///
/// # Examples
///
/// ```
/// assert_eq!(scioxide_core::sin_pi(0.5_f64)?, 1.0);
/// // Every f64 above 2⁵³ is an even integer.
/// assert_eq!(scioxide_core::sin_pi(1e30_f64)?, 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline]
pub fn sin_pi<T: crate::NumericScalar>(x: T) -> Result<T, SciOxideError> {
    scalar_dispatch!(unary T, x,  real_sin_pi, crate::stable_complex::sin_pi)
}

/// Computes `cos(πx)`.
///
/// The argument is reduced exactly, so integer arguments give exact zeros of
/// the sine and half-integer arguments give exact zeros of the cosine.
///
/// Arguments are evaluated in `f64` and the result is rounded once to the
/// output type.
///
/// # Accuracy
///
/// Real results are within 16 ulps. For complex `z`, each component of the
/// result is within `32 ε m g + 8 η` of the exact value, where `m` is
/// the larger component magnitude of the exact value, `ε` is the machine
/// epsilon and `η` the smallest positive subnormal,
/// and `g = 1 + π |Im z|`.
///
/// # Errors
///
/// - `NonFinite` if an input component is NaN or infinite.
/// - `NumericalInstability` if a result component overflows.
///
/// # References
///
/// NIST DLMF, [§4.21, trigonometric identities](https://dlmf.nist.gov/4.21),
/// and [§4.45, methods of computation](https://dlmf.nist.gov/4.45).
///
/// # Examples
///
/// ```
/// assert_eq!(scioxide_core::cos_pi(1.0_f64)?, -1.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline]
pub fn cos_pi<T: crate::NumericScalar>(x: T) -> Result<T, SciOxideError> {
    scalar_dispatch!(unary T, x,  real_cos_pi, crate::stable_complex::cos_pi)
}

/// Computes `tan(πx)`.
///
/// The argument is reduced exactly, so integer arguments give exact zeros of
/// the sine and half-integer arguments give exact zeros of the cosine.
///
/// Arguments are evaluated in `f64` and the result is rounded once to the
/// output type.
///
/// # Accuracy
///
/// Real results are within 16 ulps. For complex `z`, each component of the
/// result is within `32 ε m g + 8 η` of the exact value, where `m` is
/// the larger component magnitude of the exact value, `ε` is the machine
/// epsilon and `η` the smallest positive subnormal,
/// and `g = 1 + π |Im z|`.
///
/// # Errors
///
/// - `NonFinite` if an input component is NaN or infinite.
/// - `DomainError` at a pole: a half-integer real `x`, or a complex `z` where
///   `cos(πz) = 0`.
/// - `NumericalInstability` if a result component overflows.
///
/// # References
///
/// NIST DLMF, [§4.21, trigonometric identities](https://dlmf.nist.gov/4.21),
/// and [§4.45, methods of computation](https://dlmf.nist.gov/4.45).
///
/// # Examples
///
/// ```
/// // tan(π/4) = 1
/// let t = scioxide_core::tan_pi(0.25_f64)?;
/// assert!((t - 1.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline]
pub fn tan_pi<T: crate::NumericScalar>(x: T) -> Result<T, SciOxideError> {
    scalar_dispatch!(unary T, x,  real_tan_pi, crate::stable_complex::tan_pi)
}

/// Computes the sine and cosine of an angle in degrees.
///
/// The argument is reduced exactly, so multiples of 180° give exact zeros of
/// the sine and odd multiples of 90° give exact zeros of the cosine.
///
/// Arguments are evaluated in `f64` and the result is rounded once to the
/// output type.
///
/// # Accuracy
///
/// Real results are within 16 ulps. For complex `z`, each component of the
/// result is within `32 ε m g + 8 η` of the exact value, where `m` is
/// the larger component magnitude of the exact value, `ε` is the machine
/// epsilon and `η` the smallest positive subnormal,
/// and `g = 1 + (π/180) |Im z|`.
///
/// # Errors
///
/// - `NonFinite` if an input component is NaN or infinite.
/// - `NumericalInstability` if a result component overflows.
///
/// # References
///
/// NIST DLMF, [§4.21, trigonometric identities](https://dlmf.nist.gov/4.21),
/// and [§4.45, methods of computation](https://dlmf.nist.gov/4.45).
///
/// # Examples
///
/// ```
/// let (s, c) = scioxide_core::sin_cos_degrees(90.0_f64)?;
/// assert_eq!((s, c), (1.0, 0.0));
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline]
pub fn sin_cos_degrees<T: crate::NumericScalar>(x: T) -> Result<(T, T), SciOxideError> {
    scalar_dispatch!(pair T, x,  real_sin_cos_degrees, crate::stable_complex::sin_cos_degrees)
}

/// Computes the sine of an angle in degrees.
///
/// The argument is reduced exactly, so multiples of 180° give exact zeros of
/// the sine and odd multiples of 90° give exact zeros of the cosine.
///
/// Arguments are evaluated in `f64` and the result is rounded once to the
/// output type.
///
/// # Accuracy
///
/// Real results are within 16 ulps. For complex `z`, each component of the
/// result is within `32 ε m g + 8 η` of the exact value, where `m` is
/// the larger component magnitude of the exact value, `ε` is the machine
/// epsilon and `η` the smallest positive subnormal,
/// and `g = 1 + (π/180) |Im z|`.
///
/// # Errors
///
/// - `NonFinite` if an input component is NaN or infinite.
/// - `NumericalInstability` if a result component overflows.
///
/// # References
///
/// NIST DLMF, [§4.21, trigonometric identities](https://dlmf.nist.gov/4.21),
/// and [§4.45, methods of computation](https://dlmf.nist.gov/4.45).
///
/// # Examples
///
/// ```
/// assert_eq!(scioxide_core::sin_degrees(90.0_f64)?, 1.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline]
pub fn sin_degrees<T: crate::NumericScalar>(x: T) -> Result<T, SciOxideError> {
    scalar_dispatch!(unary T, x,  real_sin_degrees, crate::stable_complex::sin_degrees)
}

/// Computes the cosine of an angle in degrees.
///
/// The argument is reduced exactly, so multiples of 180° give exact zeros of
/// the sine and odd multiples of 90° give exact zeros of the cosine.
///
/// Arguments are evaluated in `f64` and the result is rounded once to the
/// output type.
///
/// # Accuracy
///
/// Real results are within 16 ulps. For complex `z`, each component of the
/// result is within `32 ε m g + 8 η` of the exact value, where `m` is
/// the larger component magnitude of the exact value, `ε` is the machine
/// epsilon and `η` the smallest positive subnormal,
/// and `g = 1 + (π/180) |Im z|`.
///
/// # Errors
///
/// - `NonFinite` if an input component is NaN or infinite.
/// - `NumericalInstability` if a result component overflows.
///
/// # References
///
/// NIST DLMF, [§4.21, trigonometric identities](https://dlmf.nist.gov/4.21),
/// and [§4.45, methods of computation](https://dlmf.nist.gov/4.45).
///
/// # Examples
///
/// ```
/// assert_eq!(scioxide_core::cos_degrees(180.0_f64)?, -1.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline]
pub fn cos_degrees<T: crate::NumericScalar>(x: T) -> Result<T, SciOxideError> {
    scalar_dispatch!(unary T, x,  real_cos_degrees, crate::stable_complex::cos_degrees)
}

/// Computes the tangent of an angle in degrees.
///
/// The argument is reduced exactly, so multiples of 180° give exact zeros of
/// the sine and odd multiples of 90° give exact zeros of the cosine.
///
/// Arguments are evaluated in `f64` and the result is rounded once to the
/// output type.
///
/// # Accuracy
///
/// Real results are within 16 ulps. For complex `z`, each component of the
/// result is within `32 ε m g + 8 η` of the exact value, where `m` is
/// the larger component magnitude of the exact value, `ε` is the machine
/// epsilon and `η` the smallest positive subnormal,
/// and `g = 1 + (π/180) |Im z|`.
///
/// # Errors
///
/// - `NonFinite` if an input component is NaN or infinite.
/// - `DomainError` at a pole: an odd multiple of 90° for real `x`, or a complex
///   `z` where the cosine is zero.
/// - `NumericalInstability` if a result component overflows.
///
/// # References
///
/// NIST DLMF, [§4.21, trigonometric identities](https://dlmf.nist.gov/4.21),
/// and [§4.45, methods of computation](https://dlmf.nist.gov/4.45).
///
/// # Examples
///
/// ```
/// // tan 45° = 1
/// let t = scioxide_core::tan_degrees(45.0_f64)?;
/// assert!((t - 1.0).abs() < 1e-14);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline]
pub fn tan_degrees<T: crate::NumericScalar>(x: T) -> Result<T, SciOxideError> {
    scalar_dispatch!(unary T, x,  real_tan_degrees, crate::stable_complex::tan_degrees)
}

/// Computes the normalized sinc function `sin(πx) / (πx)`, with value 1 at zero.
///
/// Nonzero integer arguments return exactly zero. For the unnormalized
/// function `sin(x) / x`, use [`sinc`].
///
/// Arguments are evaluated in `f64` and the result is rounded once to the
/// output type.
///
/// # Accuracy
///
/// Real results, including subnormal ones, are within one ulp of the correctly
/// rounded value. For complex `z`, each component of the result is within
/// `32 ε m (1 + π |Im z|) + 8 η` of the exact value, where `m` is the larger
/// component magnitude of the exact value, `ε` is the machine epsilon and `η`
/// the smallest positive subnormal.
///
/// # Errors
///
/// - `NonFinite` if an input component is NaN or infinite.
/// - `NumericalInstability` if a result component overflows.
///
/// # References
///
/// NIST DLMF, [§4.21, trigonometric identities](https://dlmf.nist.gov/4.21),
/// and [§4.45, methods of computation](https://dlmf.nist.gov/4.45).
///
/// # Examples
///
/// ```
/// assert_eq!(scioxide_core::sincpi(0.0_f64)?, 1.0);
/// assert_eq!(scioxide_core::sincpi(2.0_f64)?, 0.0);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline]
pub fn sincpi<T: crate::NumericScalar>(x: T) -> Result<T, SciOxideError> {
    scalar_dispatch!(unary T, x,  real_sincpi, crate::stable_complex::sincpi)
}

/// Computes `logaddexp(x, y) = ln(eˣ + eʸ)` without overflow.
///
/// A `−∞` argument is a masked term with `e^(−∞) = 0`: one masked argument
/// returns the other, and two masked arguments return `−∞`. Complex arguments use
/// the principal logarithm. Results are within 16 ulps, also when the result is
/// close to zero or subnormal.
///
/// # Errors
///
/// - `NonFinite` if an argument is NaN or `+∞`.
/// - `NumericalInstability` if the result overflows.
///
/// # References
///
/// NIST DLMF, [§4.2, logarithms and exponentials](https://dlmf.nist.gov/4.2),
/// and [§4.6, power series](https://dlmf.nist.gov/4.6).
///
/// # Examples
///
/// ```
/// // The f64 nearest to −ln 2 is slightly above it, so the exact sum
/// // 2 e^x is slightly above 1 and its logarithm is tiny but nonzero.
/// let x = -core::f64::consts::LN_2;
/// let y = scioxide_core::logaddexp(x, x)?;
/// assert_eq!(y.to_bits(), 0x3c7a_bc9e_3b39_803f);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[inline]
pub fn logaddexp<T: crate::NumericScalar>(x: T, y: T) -> Result<T, SciOxideError> {
    scalar_dispatch!(binary T, x, y,  real_logaddexp, |x, y| crate::logarithmic::logsumexp(&[x, y]))
}
