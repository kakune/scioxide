//! Shared scalar gamma kernels used by the gamma, hypergeometric, and zeta families.

use crate::support::gamma_real as real;
use crate::support::{project_real, require_finite_param};
use crate::util::{is_exact, sin_pi};
use scioxide_core::{RealScalar, SciOxideError};

/// Envelope of the fixed gamma-family kernels on the tested range, in ulps of the result.
pub(crate) const GAMMA_ULPS: f64 = 4.0;

/// Check if x is a non-positive integer (pole of gamma).
pub(crate) fn is_nonpositive_integer<S: RealScalar>(x: S) -> bool {
    x <= S::ZERO && is_exact(x, x.floor())
}

pub(crate) fn validate_gamma_argument<S: RealScalar>(x: S) -> Result<(), SciOxideError> {
    require_finite_param(x, "gamma parameter x must be finite")?;
    if is_nonpositive_integer(x) {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("x"),
            scioxide_core::DomainReason::Pole,
            "gamma-family argument is a nonpositive integer",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Lanczos-13 coefficients shared with real/complex gamma and beta
// ---------------------------------------------------------------------------

pub(crate) const LANCZOS_G: f64 = real::G;
pub(crate) const SMALL_FACTORIALS_F64: [f64; 21] = [
    1.0,
    1.0,
    2.0,
    6.0,
    24.0,
    120.0,
    720.0,
    5_040.0,
    40_320.0,
    362_880.0,
    3_628_800.0,
    39_916_800.0,
    479_001_600.0,
    6_227_020_800.0,
    87_178_291_200.0,
    1_307_674_368_000.0,
    20_922_789_888_000.0,
    355_687_428_096_000.0,
    6_402_373_705_728_000.0,
    121_645_100_408_832_000.0,
    2_432_902_008_176_640_000.0,
];
pub(crate) const LANCZOS_COEFFS: [f64; 13] = real::PARTIAL;

/// ln |Γ(x)|.
pub(crate) fn ln_gamma_kernel<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    validate_gamma_argument(x)?;

    let argument = x.into_f64();
    let value = if argument > 0.0 {
        real::ln_positive(argument)
    } else {
        real::ln_negative(argument)
    };
    project_real(value)
}

/// Γ(x) in `f64` within `GAMMA_ULPS`: ±∞ beyond the range and signed zeros below it.
/// Callers exclude the poles.
pub(crate) fn gamma_value(x: f64) -> Result<f64, SciOxideError> {
    if x > 0.0 {
        real::gamma_positive(x)
    } else {
        real::gamma_negative(x)
    }
}

pub(crate) fn gamma_log_parts<S: RealScalar>(x: S) -> Result<(S, S), SciOxideError> {
    validate_gamma_argument(x)?;

    let sign = if x > S::ZERO || sin_pi(x.into_f64()) > 0.0 {
        S::ONE
    } else {
        -S::ONE
    };
    Ok((ln_gamma_kernel(x)?, sign))
}

/// Digamma.
#[inline]
pub(crate) fn digamma_kernel<S: RealScalar>(x: S) -> Result<S, SciOxideError> {
    validate_gamma_argument(x)?;

    project_real(real::digamma(x.into_f64()))
}

/// Shared complex Lanczos logarithm and its analytic derivative. The caller owns
/// work accounting; reflection uses core's exact pi-phase reduction.
pub(crate) fn complex_log_gamma(
    z: scioxide_core::Complex<f64>,
    step: &mut impl FnMut() -> Result<(), SciOxideError>,
) -> Result<(scioxide_core::Complex<f64>, scioxide_core::Complex<f64>), SciOxideError> {
    complex_log_gamma_kernel::<true>(z, step)
}

/// The same logarithm and work charges, without computing its derivative.
pub(crate) fn complex_log_gamma_value(
    z: scioxide_core::Complex<f64>,
    step: &mut impl FnMut() -> Result<(), SciOxideError>,
) -> Result<scioxide_core::Complex<f64>, SciOxideError> {
    complex_log_gamma_kernel::<false>(z, step).map(|(value, _)| value)
}

fn complex_log_gamma_kernel<const DERIVATIVE: bool>(
    z: scioxide_core::Complex<f64>,
    step: &mut impl FnMut() -> Result<(), SciOxideError>,
) -> Result<(scioxide_core::Complex<f64>, scioxide_core::Complex<f64>), SciOxideError> {
    use scioxide_core::Complex;
    let r = |x| Complex::new(x, 0.0);
    if z.re < 0.5 {
        step()?;
        // An unwrapped log(sin(pi*z)) is required here: taking the principal
        // logarithm of sin loses whole turns of log-Gamma on the negative axis.
        // Factoring exp(pi*|Im z|) also avoids overflowing the reflection seed.
        let side = if z.im.is_sign_negative() { -1.0 } else { 1.0 };
        let sine = scioxide_core::sin_pi(z.re)?;
        let cosine = scioxide_core::cos_pi(z.re)?;
        let decay = -2.0 * core::f64::consts::PI * z.im.abs();
        let attenuation = decay.exp();
        let difference = Complex::new(
            -decay.exp_m1() + 2.0 * attenuation * sine * sine,
            -side * 2.0 * attenuation * sine * cosine,
        );
        if difference == r(0.0) {
            return Err(SciOxideError::domain_error(
                scioxide_core::argument!("z"),
                scioxide_core::DomainReason::Pole,
                "gamma has a pole at a nonpositive integer",
            ));
        }
        let log_sine = difference.ln()
            + Complex::new(
                core::f64::consts::PI * z.im.abs() - core::f64::consts::LN_2,
                side * core::f64::consts::PI * (0.5 - z.re),
            );
        let (value, derivative) = complex_log_gamma_kernel::<DERIVATIVE>(r(1.0) - z, step)?;
        return Ok((
            r(core::f64::consts::PI.ln()) - log_sine - value,
            if DERIVATIVE {
                let cotangent = Complex::new(0.0, -side) * (r(2.0) - difference) / difference;
                derivative - r(core::f64::consts::PI) * cotangent
            } else {
                r(0.0)
            },
        ));
    }
    let y = z - r(1.0);
    let mut sum = r(LANCZOS_COEFFS[0]);
    let mut derivative = r(0.0);
    for (j, &coefficient) in LANCZOS_COEFFS.iter().enumerate().skip(1) {
        step()?;
        let denominator = y + r(j as f64);
        let term = r(coefficient) / denominator;
        sum += term;
        if DERIVATIVE {
            derivative -= term / denominator;
        }
    }
    let t = y + r(LANCZOS_G + 0.5);
    let log_t = t.ln();
    Ok((
        r(0.5 * (2.0 * core::f64::consts::PI).ln()) + (y + r(0.5)) * log_t - t + sum.ln(),
        if DERIVATIVE {
            log_t + (y + r(0.5)) / t - r(1.0) + derivative / sum
        } else {
            r(0.0)
        },
    ))
}
