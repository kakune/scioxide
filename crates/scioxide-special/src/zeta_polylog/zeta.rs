//! Riemann zeta, Dirichlet eta, and Hurwitz zeta functions.
//!
//! `f32` and `f64` arguments use generated piecewise Chebyshev kernels on the
//! positive axis, direct Chebyshev kernels on `[-1,0]`, and a compensated
//! logarithmic functional equation below -1. Negative `f32` calls round the
//! `f64` kernel once with checked range conversion. Hurwitz zeta uses
//! a normalized Euler--Maclaurin sum for s>1, and centred Taylor/Fourier
//! continuation with derived absolute errors below the pole.

use crate::support::Work;
use crate::support::check_finite;
use crate::support::gamma_core::{digamma_kernel, ln_gamma_kernel};
use crate::support::wide::Wide;
use crate::util::sin_pi;
use scioxide_core::{AlgorithmComponent, RealScalar, SciOxideError, algorithm_component};

mod coefficients {
    include!("zeta_coefficients.rs");
}

#[inline]
fn overflow() -> SciOxideError {
    SciOxideError::numerical_instability(
        scioxide_core::NumericalIssue::Overflow,
        "the finite mathematical result is outside the scalar range",
    )
}

#[inline]
fn breakdown() -> SciOxideError {
    SciOxideError::numerical_instability(
        scioxide_core::NumericalIssue::Breakdown,
        "the certified remainder bound was not reached",
    )
}

#[inline]
fn chebyshev_native<S: RealScalar>(
    x: S,
    coefficients: &[S],
    work: &mut Work,
) -> Result<S, SciOxideError> {
    let result = crate::support::polynomial::chebyshev(
        x,
        coefficients,
        false,
        false,
        Ok::<_, core::convert::Infallible>,
        work,
    );
    work.check()?;
    if !work.complete {
        return Err(work.limit_error());
    }
    match result {
        Ok((value, _)) => Ok(value),
        Err(never) => match never {},
    }
}

#[inline]
fn mapped_f64(value: f64, lower: f64, upper: f64) -> f64 {
    (2.0 * value - (lower + upper)) / (upper - lower)
}

#[inline]
fn mapped_f32(value: f32, lower: f32, upper: f32) -> f32 {
    (2.0 * value - (lower + upper)) / (upper - lower)
}

#[inline]
fn is_negative_even_f64(value: f64) -> bool {
    value < 0.0 && value == value.floor() && value * 0.5 == (value * 0.5).floor()
}

#[inline]
fn exp_signed_f64(sign: f64, logarithm: Wide) -> Result<f64, SciOxideError> {
    if logarithm.value() > f64::MAX.ln() {
        return Err(overflow());
    }
    // |lo| < 2^-44 throughout the representable exponent range, so omitting
    // the quadratic correction in exp(lo) costs less than 2^-89 relatively.
    // A magnitude below the range rounds to a signed zero.
    let high = logarithm.hi.exp();
    check_finite(sign * high.mul_add(logarithm.lo, high))
}

#[inline]
fn reflected_f32(value: f64) -> Result<f32, SciOxideError> {
    let result = value as f32;
    if !result.is_finite() {
        return Err(overflow());
    }
    Ok(result)
}

fn riemann_positive_f64(
    s: f64,
    work: &mut Work,
) -> Result<(f64, AlgorithmComponent, usize), SciOxideError> {
    Ok(if s <= 2.0 {
        let value = chebyshev_native(
            mapped_f64(s, 0.0, 2.0),
            &coefficients::ZETA_POLE_PRODUCT_F64,
            work,
        )? / (s - 1.0);
        (
            value,
            algorithm_component!("primitive_central"),
            coefficients::ZETA_POLE_PRODUCT_F64.len(),
        )
    } else if s <= 4.0 {
        let residual = chebyshev_native(
            mapped_f64(s, 2.0, 4.0),
            &coefficients::ZETA_MINUS_ONE_2_4_F64,
            work,
        )?;
        (
            1.0 + residual,
            algorithm_component!("primitive_positive_2_4"),
            coefficients::ZETA_MINUS_ONE_2_4_F64.len(),
        )
    } else if s <= 8.0 {
        let residual = chebyshev_native(
            mapped_f64(s, 4.0, 8.0),
            &coefficients::ZETA_MINUS_ONE_4_8_F64,
            work,
        )?;
        (
            1.0 + residual,
            algorithm_component!("primitive_positive_4_8"),
            coefficients::ZETA_MINUS_ONE_4_8_F64.len(),
        )
    } else if s <= 16.0 {
        let residual = chebyshev_native(
            mapped_f64(s, 8.0, 16.0),
            &coefficients::ZETA_MINUS_ONE_8_16_F64,
            work,
        )?;
        (
            1.0 + residual,
            algorithm_component!("primitive_positive_8_16"),
            coefficients::ZETA_MINUS_ONE_8_16_F64.len(),
        )
    } else if s <= 24.0 {
        let residual = chebyshev_native(
            mapped_f64(s, 16.0, 24.0),
            &coefficients::ZETA_MINUS_ONE_16_24_F64,
            work,
        )?;
        (
            1.0 + residual,
            algorithm_component!("primitive_positive_16_24"),
            coefficients::ZETA_MINUS_ONE_16_24_F64.len(),
        )
    } else {
        // The omitted tail is <= 4^(1-s)/(s-1) < 2^-50 for s>24.
        let sum = (2_u32..=4).try_fold(1.0, |sum, term| {
            work.try_step()?;
            Ok::<_, SciOxideError>(sum + f64::from(term).powf(-s))
        })?;
        (sum, algorithm_component!("primitive_tail"), 3)
    })
}

fn riemann_positive_f32(
    s: f32,
    work: &mut Work,
) -> Result<(f32, AlgorithmComponent, usize), SciOxideError> {
    Ok(if s <= 2.0 {
        let value = chebyshev_native(
            mapped_f32(s, 0.0, 2.0),
            &coefficients::ZETA_POLE_PRODUCT_F32,
            work,
        )? / (s - 1.0);
        (
            value,
            algorithm_component!("primitive_central"),
            coefficients::ZETA_POLE_PRODUCT_F32.len(),
        )
    } else if s <= 4.0 {
        let residual = chebyshev_native(
            mapped_f32(s, 2.0, 4.0),
            &coefficients::ZETA_MINUS_ONE_2_4_F32,
            work,
        )?;
        (
            1.0 + residual,
            algorithm_component!("primitive_positive_2_4"),
            coefficients::ZETA_MINUS_ONE_2_4_F32.len(),
        )
    } else if s <= 8.0 {
        let residual = chebyshev_native(
            mapped_f32(s, 4.0, 8.0),
            &coefficients::ZETA_MINUS_ONE_4_8_F32,
            work,
        )?;
        (
            1.0 + residual,
            algorithm_component!("primitive_positive_4_8"),
            coefficients::ZETA_MINUS_ONE_4_8_F32.len(),
        )
    } else if s <= 16.0 {
        let residual = chebyshev_native(
            mapped_f32(s, 8.0, 16.0),
            &coefficients::ZETA_MINUS_ONE_8_16_F32,
            work,
        )?;
        (
            1.0 + residual,
            algorithm_component!("primitive_positive_8_16"),
            coefficients::ZETA_MINUS_ONE_8_16_F32.len(),
        )
    } else {
        let sum = (2_u16..=4).try_fold(1.0_f32, |sum, term| {
            work.try_step()?;
            Ok::<_, SciOxideError>(sum + f32::from(term).powf(-s))
        })?;
        (sum, algorithm_component!("primitive_tail"), 3)
    })
}

fn negative_zeta_log_f64(s: f64, work: &mut Work) -> Result<(f64, Wide), SciOxideError> {
    // Below -300 every nonzero representable reflection result overflows:
    // even a one-ULP displacement from an even integer cannot offset Γ(1-s).
    if s < -300.0 {
        return Err(overflow());
    }
    work.try_step()?;
    let reflected = Wide::from(1.0) - Wide::from(s);
    let sine = sin_pi(s * 0.5);
    let (zeta, _, _) = riemann_positive_f64(reflected.hi, work)?;
    // Moderate logarithms need compensation but not wide transcendental
    // evaluation. Larger exponents require the low gamma/log components too.
    let (sine_log, gamma_log, zeta_log) = if s >= -16.0 {
        let mut gamma_log = Wide::from(ln_gamma_kernel(reflected.hi)?);
        if reflected.lo != 0.0 {
            // log Γ(x+δ) = log Γ(x) + ψ(x)δ + O(δ²). Here x>=2,
            // |δ|<=2^-49 and ψ'(x)<1, making the remainder <2^-99.
            gamma_log = gamma_log + Wide::from(digamma_kernel(reflected.hi)?).mul_f64(reflected.lo);
        }
        (
            Wide::from(sine.abs().ln()),
            gamma_log,
            Wide::from(zeta.ln()),
        )
    } else {
        (
            Wide::from(sine.abs()).ln(),
            crate::support::gamma_real::ln_positive_wide(reflected),
            Wide::from(zeta).ln(),
        )
    };
    let logarithm = LN_2.mul_f64(s) - reflected * LN_PI_WIDE + sine_log + gamma_log + zeta_log;
    Ok((sine.signum(), logarithm))
}

fn riemann_primitive_f64(
    s: f64,
    work: &mut Work,
) -> Result<(f64, AlgorithmComponent, usize), SciOxideError> {
    if s == 0.0 {
        return Ok((-0.5, algorithm_component!("exact_zero"), 0));
    }
    if s > 0.0 {
        return riemann_positive_f64(s, work);
    }
    if is_negative_even_f64(s) {
        return Ok((0.0, algorithm_component!("trivial_zero"), 0));
    }
    if s >= -1.0 {
        return Ok((
            chebyshev_native(
                mapped_f64(s, -1.0, 0.0),
                &coefficients::ZETA_NEGATIVE_ORIGIN_F64,
                work,
            )?,
            algorithm_component!("negative_origin"),
            coefficients::ZETA_NEGATIVE_ORIGIN_F64.len(),
        ));
    }
    let (sign, logarithm) = negative_zeta_log_f64(s, work)?;
    Ok((
        exp_signed_f64(sign, logarithm)?,
        algorithm_component!("primitive_reflection"),
        1,
    ))
}

fn riemann_primitive_f32(
    s: f32,
    work: &mut Work,
) -> Result<(f32, AlgorithmComponent, usize), SciOxideError> {
    if s == 0.0 {
        return Ok((-0.5, algorithm_component!("exact_zero"), 0));
    }
    if s > 0.0 {
        return riemann_positive_f32(s, work);
    }
    let (value, branch, iterations) = riemann_primitive_f64(f64::from(s), work)?;
    Ok((reflected_f32(value)?, branch, iterations))
}

pub(crate) fn riemann_zeta_evaluate<S: RealScalar>(
    s: S,
    work: &mut Work,
) -> Result<(S, AlgorithmComponent, usize), SciOxideError> {
    crate::support::require_finite_param(s, "zeta parameter s must be finite")?;
    if s == S::ONE {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("s"),
            scioxide_core::DomainReason::Pole,
            "s = 1 is a pole of the Riemann zeta function",
        ));
    }
    if let Ok(input) = f64::try_from(s.value()) {
        let (value, branch, iterations) = riemann_primitive_f64(input, work)?;
        return Ok((S::try_from_f64_lossy(value)?, branch, iterations));
    }
    if let Ok(input) = f32::try_from(s.value()) {
        let (value, branch, iterations) = riemann_primitive_f32(input, work)?;
        return Ok((S::try_from_f64_lossy(f64::from(value))?, branch, iterations));
    }
    Err(SciOxideError::scalar_conversion(
        <S as scioxide_core::backend::Scalar>::kind().into(),
        scioxide_core::ConversionType::F64,
        scioxide_core::ScalarConversionReason::Unsupported,
    ))
}

// Two-term logarithmic constants rounded once at 240 decimal digits.
use crate::support::constants::LN_2;

use crate::support::constants::LN_PI as LN_PI_WIDE;

fn log_expm1_positive_f64(value: Wide) -> Wide {
    if value.hi > 1.0 {
        value + Wide::from((-(-value).exp()).ln_1p())
    } else {
        value.exp_m1().ln()
    }
}

fn eta_primitive_f64(
    s: f64,
    work: &mut Work,
) -> Result<(f64, AlgorithmComponent, usize), SciOxideError> {
    if s == 0.0 {
        return Ok((0.5, algorithm_component!("exact_zero"), 0));
    }
    if s == 1.0 {
        return Ok((
            core::f64::consts::LN_2,
            algorithm_component!("exact_one"),
            0,
        ));
    }
    if s > 0.0 && s <= 2.0 {
        let value = chebyshev_native(
            mapped_f64(s, 0.0, 2.0),
            &coefficients::ETA_CENTRAL_F64,
            work,
        )?;
        return Ok((
            value,
            algorithm_component!("primitive_central"),
            coefficients::ETA_CENTRAL_F64.len(),
        ));
    }
    if s > 2.0 {
        let (zeta, _, _) = riemann_positive_f64(s, work)?;
        work.try_step()?;
        let factor = -((1.0 - s) * core::f64::consts::LN_2).exp_m1();
        return Ok((factor * zeta, algorithm_component!("primitive_relation"), 1));
    }
    if is_negative_even_f64(s) {
        return Ok((0.0, algorithm_component!("trivial_zero"), 0));
    }
    if s >= -1.0 {
        return Ok((
            chebyshev_native(
                mapped_f64(s, -1.0, 0.0),
                &coefficients::ETA_NEGATIVE_ORIGIN_F64,
                work,
            )?,
            algorithm_component!("negative_origin"),
            coefficients::ETA_NEGATIVE_ORIGIN_F64.len(),
        ));
    }
    let (zeta_sign, zeta_logarithm) = negative_zeta_log_f64(s, work)?;
    let exponent = (Wide::from(1.0) - Wide::from(s)) * LN_2;
    let logarithm = zeta_logarithm + log_expm1_positive_f64(exponent);
    Ok((
        exp_signed_f64(-zeta_sign, logarithm)?,
        algorithm_component!("primitive_reflection"),
        1,
    ))
}

fn eta_primitive_f32(
    s: f32,
    work: &mut Work,
) -> Result<(f32, AlgorithmComponent, usize), SciOxideError> {
    if s == 0.0 {
        return Ok((0.5, algorithm_component!("exact_zero"), 0));
    }
    if s == 1.0 {
        return Ok((
            core::f32::consts::LN_2,
            algorithm_component!("exact_one"),
            0,
        ));
    }
    if s > 0.0 && s <= 2.0 {
        let value = chebyshev_native(
            mapped_f32(s, 0.0, 2.0),
            &coefficients::ETA_CENTRAL_F32,
            work,
        )?;
        return Ok((
            value,
            algorithm_component!("primitive_central"),
            coefficients::ETA_CENTRAL_F32.len(),
        ));
    }
    if s > 2.0 {
        let (zeta, _, _) = riemann_positive_f32(s, work)?;
        work.try_step()?;
        let factor = -((1.0 - s) * core::f32::consts::LN_2).exp_m1();
        return Ok((factor * zeta, algorithm_component!("primitive_relation"), 1));
    }
    let (value, branch, iterations) = eta_primitive_f64(f64::from(s), work)?;
    Ok((reflected_f32(value)?, branch, iterations))
}

pub(crate) fn dirichlet_eta_evaluate<S: RealScalar>(
    s: S,
    work: &mut Work,
) -> Result<(S, AlgorithmComponent, usize), SciOxideError> {
    crate::support::require_finite_param(s, "zeta parameter s must be finite")?;
    if let Ok(input) = f64::try_from(s.value()) {
        let (value, branch, iterations) = eta_primitive_f64(input, work)?;
        return Ok((S::try_from_f64_lossy(value)?, branch, iterations));
    }
    if let Ok(input) = f32::try_from(s.value()) {
        let (value, branch, iterations) = eta_primitive_f32(input, work)?;
        return Ok((S::try_from_f64_lossy(f64::from(value))?, branch, iterations));
    }
    Err(SciOxideError::scalar_conversion(
        <S as scioxide_core::backend::Scalar>::kind().into(),
        scioxide_core::ConversionType::F64,
        scioxide_core::ScalarConversionReason::Unsupported,
    ))
}

/// DLMF 25.11.10 about a=1, with |delta| <= 1/2. The pole product
/// (t-1) zeta(t) is evaluated directly when a rising-factor zero meets t=1.
/// This also supplies the last Bernoulli-polynomial term at integer orders.
fn hurwitz_shift_series(
    s: f64,
    delta: Wide,
    work: &mut Work,
) -> Result<(Wide, f64), SciOxideError> {
    let (initial, _, _) = riemann_primitive_f64(s, work)?;
    let mut sum = Wide::from(initial);
    let mut coefficient = Wide::from(1.0);
    let mut magnitude = initial.abs();
    let mut error = 64.0 * f64::EPSILON * magnitude;
    if delta.hi == 0.0 && delta.lo == 0.0 {
        return Ok((sum, error));
    }
    for k in 1usize.. {
        work.try_step()?;
        let partial = coefficient * (-delta).div_f64(k as f64);
        let factor = Wide::from(s) + Wide::from((k - 1) as f64);
        coefficient = partial * factor;
        let argument = factor + Wide::from(1.0);
        let term = if argument.hi > 0.0 && argument.hi < 2.0 {
            // Mapping [0,2] to [-1,1] is exactly t-1. Retain that
            // coordinate even if forming t would round a neighbour to 1.
            partial.mul_f64(chebyshev_native(
                factor.value(),
                &coefficients::ZETA_POLE_PRODUCT_F64,
                work,
            )?)
        } else {
            coefficient.mul_f64(riemann_primitive_f64(argument.value(), work)?.0)
        };
        sum = sum + term;
        magnitude += term.hi.abs();
        error += 64.0 * f64::EPSILON * term.hi.abs();
        if !sum.hi.is_finite() || !coefficient.hi.is_finite() {
            return Err(overflow());
        }
        if factor.hi == 0.0 && factor.lo == 0.0 {
            return Ok((sum, error));
        }
        if argument.hi >= 2.0 {
            // For s<1 the subsequent coefficient ratios are <= |delta|,
            // and zeta(t)<=zeta(2)<2 for t>=2.
            let ratio = delta.value().abs();
            let tail = 2.0 * coefficient.hi.abs() * ratio / (1.0 - ratio);
            if tail <= f64::EPSILON * magnitude.max(f64::MIN_POSITIVE) {
                return Ok((sum, error + tail));
            }
        }
    }
    Err(breakdown())
}

fn hurwitz_continuation(
    s: f64,
    a: f64,
    work: &mut Work,
) -> Result<(f64, f64, AlgorithmComponent), SciOxideError> {
    // At large shifts the integral dominates, so no cancelling prefix is
    // necessary. Scale by a^(1-s), keeping both the base and integral finite.
    if s > -23.0 && a >= 16.0_f64.max(2.0 * s.abs()) {
        let q = Wide::from(a);
        let minimum = ((1.0 - s) * 0.5).ceil().max(1.0) as usize;
        if let Some((sum, tail)) = crate::support::hurwitz::tail(
            Wide::from(s),
            q,
            Wide::from(1.0) / q,
            (Wide::ZERO, Wide::ZERO),
            f64::EPSILON,
            minimum,
            &mut || work.try_step(),
        )? {
            let logarithm = q.ln() * (Wide::from(1.0) - Wide::from(s));
            let scale = logarithm.exp_wide();
            let projected = (scale * sum).value();
            let value = if projected.is_finite() {
                projected
            } else {
                let magnitude = if sum.hi < 0.0 { -sum } else { sum };
                (logarithm + magnitude.ln()).exp().copysign(sum.hi)
            };
            let value = check_finite(value)?;
            return Ok((
                value,
                (tail / sum.hi.abs() + 64.0 * f64::EPSILON) * value.abs(),
                algorithm_component!("hurwitz_large_shift"),
            ));
        }
    }
    if s <= -8.0 {
        return hurwitz_fourier(s, a, work)
            .map(|(value, error)| (value, error, algorithm_component!("hurwitz_fourier")));
    }
    // No enormous prefix is subtracted from an integral. Only the exact
    // Hurwitz shift recurrence is applied to the centred Taylor value.
    let shifts = if a < 0.5 { 0.0 } else { (a - 0.5).floor() };
    if shifts > work.remaining as f64 {
        work.steps(work.remaining + 1)?;
    }
    let centre = if a < 0.5 {
        Wide::from(a) + Wide::from(1.0)
    } else {
        Wide::from(a) - Wide::from(shifts)
    };
    let delta = if a < 0.5 {
        Wide::from(a)
    } else {
        centre - Wide::from(1.0)
    };
    let (mut value, mut error) = hurwitz_shift_series(s, delta, work)?;
    let mut magnitude = value.hi.abs();
    for index in 0..shifts as usize {
        work.try_step()?;
        let argument = centre + Wide::from(index as f64);
        let term = argument.ln().mul_f64(-s).exp_wide();
        value = value - term;
        magnitude += term.hi.abs();
        error += 8.0 * f64::EPSILON * term.hi.abs();
    }
    if a < 0.5 {
        work.try_step()?;
        let term = Wide::from(a).ln().mul_f64(-s).exp_wide();
        value = value + term;
        magnitude += term.hi.abs();
        error += 8.0 * f64::EPSILON * term.hi.abs();
    }
    let result = check_finite(value.value())?;
    error += 4.0 * f64::EPSILON * result.abs() + 32.0 * f64::EPSILON * f64::EPSILON * magnitude;
    Ok((result, error, algorithm_component!("hurwitz_shift_series")))
}

/// DLMF 25.11.9. Below -8 the absolutely convergent Fourier tail reaches
/// `f64` rounding in fewer than 128 terms. Unlike a Taylor polynomial,
/// it keeps the exact symmetry zeros at integral and half-integral shifts.
fn hurwitz_fourier(s: f64, a: f64, work: &mut Work) -> Result<(f64, f64), SciOxideError> {
    let shifts = if a <= 1.0 { 0.0 } else { a.ceil() - 1.0 };
    if shifts > work.remaining as f64 {
        work.steps(work.remaining + 1)?;
    }
    let q = a - shifts;
    let (mut value, mut error) = if is_negative_even_f64(s) && (q == 0.5 || q == 1.0) {
        work.try_step()?;
        (Wide::ZERO, 0.0)
    } else {
        let reflected = Wide::from(1.0) - Wide::from(s);
        let logarithm = crate::support::gamma_real::ln_positive_wide(reflected)
            - reflected * (LN_2 + LN_PI_WIDE)
            + LN_2;
        let scale = logarithm.exp_wide();
        if !scale.hi.is_finite() {
            return Err(overflow());
        }
        let (sine, cosine) = scioxide_core::sin_cos_pi(s * 0.5)?;
        let mut sum = Wide::ZERO;
        let mut magnitude = 0.0;
        let mut n = 0usize;
        let tail = loop {
            work.try_step()?;
            n += 1;
            let phase = Wide::from(q).mul_f64(2.0 * n as f64);
            let (sn, cs) = scioxide_core::sin_cos_pi(phase.hi)?;
            let angle = core::f64::consts::PI * phase.lo;
            let sine_term = angle.mul_add(cs, sn);
            let cosine_term = (-angle).mul_add(sn, cs);
            let weight = (n as f64).powf(s - 1.0);
            let term = (Wide::from(sine).mul_f64(cosine_term)
                + Wide::from(cosine).mul_f64(sine_term))
            .mul_f64(weight);
            sum = sum + term;
            magnitude += weight;
            let bound = (n as f64).powf(s) / (-s);
            if bound <= f64::EPSILON {
                break bound;
            }
        };
        let value = scale * sum;
        let error = scale.hi.abs() * (tail + 32.0 * f64::EPSILON * magnitude);
        (value, error)
    };
    for k in 0..shifts as usize {
        work.try_step()?;
        let term = (Wide::from(q) + Wide::from(k as f64))
            .ln()
            .mul_f64(-s)
            .exp_wide();
        value = value - term;
        error += 8.0 * f64::EPSILON * term.hi.abs();
    }
    let value = check_finite(value.value())?;
    Ok((value, error + 4.0 * f64::EPSILON * value.abs()))
}

/// A completed Hurwitz value; `None` selects the measured envelope for `s > 1`.
/// Continuation carries its absolute derived error even at a zero.
pub(crate) struct HurwitzEvaluation<S> {
    pub(crate) value: S,
    pub(crate) method: AlgorithmComponent,
    pub(crate) error: Option<S>,
}

/// Hurwitz zeta function `ζ(s,a)` on the real branch `s != 1`, `a > 0`.
///
/// Positive orders use a normalized Euler--Maclaurin sum. Below the pole,
/// a centred Taylor continuation and the shift recurrence avoid cancellation
/// between a large finite prefix and its integral. Bounds include the absolute
/// conditioning of those sums; zeros do not require relative convergence.
///
/// # Errors
///
/// Returns [`scioxide_core::SciOxideErrorKind::NonFinite`] for non-finite input,
/// [`scioxide_core::SciOxideErrorKind::DomainError`] outside the implemented branch or at a pole,
/// and [`scioxide_core::SciOxideErrorKind::NumericalInstability`] for an unrepresentable result or
/// an uncertified remainder.
pub(crate) fn hurwitz_zeta_evaluate<S: RealScalar>(
    s: S,
    a: S,
    work: &mut Work,
) -> Result<HurwitzEvaluation<S>, SciOxideError> {
    crate::support::require_finite_param(s, "zeta parameter s must be finite")?;
    crate::support::require_finite_param(a, "zeta parameter a must be finite")?;
    let one = S::ONE;
    if s == one {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("s"),
            scioxide_core::DomainReason::Pole,
            "s = 1 is a pole of the Hurwitz zeta function",
        ));
    }
    if a == S::ZERO {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("a"),
            scioxide_core::DomainReason::Pole,
            "a = 0 makes the leading series term singular",
        ));
    }
    if a < S::ZERO {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("a"),
            scioxide_core::DomainReason::OutsideDomain,
            "the implemented real branch requires a > 0",
        ));
    }

    if a == one {
        let (value, method, _) = riemann_zeta_evaluate(s, work)?;
        return Ok(HurwitzEvaluation {
            value,
            method,
            error: None,
        });
    }
    if s == S::ZERO || s == -one {
        let argument = Wide::from(a.into_f64());
        let value = if s == S::ZERO {
            Wide::from(0.5) - argument
        } else {
            -(argument * (argument - Wide::from(1.0)) + Wide::from(1.0) / Wide::from(6.0))
                .mul_f64(0.5)
        };
        if !value.hi.is_finite() {
            return Err(overflow());
        }
        let rounded = crate::support::project_real::<S>(value.value())?;
        let magnitude = if s == S::ZERO {
            0.5 + a.into_f64()
        } else {
            1.0 + a.into_f64() * a.into_f64()
        };
        let error = (rounded.into_f64() - value.hi).abs()
            + value.lo.abs()
            + 8.0 * f64::EPSILON * f64::EPSILON * magnitude;
        return Ok(HurwitzEvaluation {
            value: check_finite(rounded)?,
            method: algorithm_component!("hurwitz_bernoulli_polynomial"),
            error: Some(S::try_from_f64_lossy(error)?.max(S::MIN_SUBNORMAL)),
        });
    }

    if s > one {
        let (scaled, _) = crate::support::hurwitz::sum(s, a, work)?;
        let leading = a.powf(-s);
        let value = leading * scaled;
        // Only range-boundary cases need a logarithmic reconstruction. In the
        // ordinary range powf retains its split logarithm through exponentiation.
        let value = if !value.is_finite() || leading < S::MIN_POSITIVE {
            let logarithm = Wide::from(a.into_f64()).ln().mul_f64(-s.into_f64())
                + Wide::from(scaled.into_f64()).ln();
            let projected = logarithm.exp();
            if !projected.is_finite() {
                return Err(overflow());
            }
            crate::support::project_real::<S>(projected)?
        } else {
            value
        };
        return Ok(HurwitzEvaluation {
            value: check_finite(value)?,
            method: algorithm_component!("hurwitz_scaled_euler_maclaurin"),
            error: None,
        });
    }
    let (value, error, method) = hurwitz_continuation(s.into_f64(), a.into_f64(), work)?;
    let rounded = crate::support::project_real::<S>(value)?;
    let error = error + (rounded.into_f64() - value).abs();
    Ok(HurwitzEvaluation {
        value: rounded,
        method,
        error: Some(S::try_from_f64_lossy(error.min(S::MAX.into_f64()))?.max(S::MIN_SUBNORMAL)),
    })
}

/// Positive Hurwitz series normalized by q^-s, s>1 and q>0.
/// The leading q^-s is factored out before any power is formed. The positive
/// series and Euler--Maclaurin remainder use relative stopping in that scale.
/// This is shared by ordinary positive Hurwitz evaluation and count-law tails.
/// Logarithmic projection of the same positive scaled sum as the ordinary API.
pub(crate) fn hurwitz_positive_log<S: RealScalar>(
    s: S,
    q: S,
    work: &mut Work,
) -> Result<(S, usize), SciOxideError> {
    let (sum, count) = crate::support::hurwitz::sum(s, q, work)?;
    let logarithm =
        Wide::from(q.into_f64()).ln().mul_f64(-s.into_f64()) + Wide::from(sum.into_f64()).ln();
    let value = logarithm.value();
    if !value.is_finite() {
        return Err(overflow());
    }
    Ok((crate::support::project_real::<S>(value)?, count))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use crate::{dirichlet_eta, hurwitz_zeta, riemann_zeta};

    #[test]
    fn known_riemann_values() {
        let zeta_two = riemann_zeta(2.0_f64).unwrap();
        assert!((zeta_two - std::f64::consts::PI.powi(2) / 6.0).abs() < 1.0e-12);
        assert!((riemann_zeta(3.0_f64).unwrap() - 1.202_056_903_159_594_3).abs() < 1.0e-12);
        assert!((riemann_zeta(-1.0_f64).unwrap() + 1.0 / 12.0).abs() < 1.0e-12);
        assert_eq!(riemann_zeta(-2.0_f64).unwrap(), 0.0);
        assert_eq!(riemann_zeta(0.0_f64).unwrap(), -0.5);
    }

    #[test]
    fn known_eta_values() {
        assert_eq!(dirichlet_eta(1.0_f64).unwrap(), std::f64::consts::LN_2);
        assert_eq!(dirichlet_eta(0.0_f64).unwrap(), 0.5);
        assert!(
            (dirichlet_eta(2.0_f64).unwrap() - std::f64::consts::PI.powi(2) / 12.0).abs() < 1.0e-12
        );
    }

    #[test]
    fn hurwitz_identities() {
        let zeta = hurwitz_zeta(2.0_f64, 1.0).unwrap();
        assert!((zeta - riemann_zeta(2.0_f64).unwrap()).abs() < 1.0e-12);
        let shifted = hurwitz_zeta(2.5_f64, 2.5).unwrap() + 1.5_f64.powf(-2.5);
        assert!((hurwitz_zeta(2.5_f64, 1.5).unwrap() - shifted).abs() < 1.0e-12);
        assert!((hurwitz_zeta(0.0_f64, 0.25).unwrap() - 0.25).abs() < 1.0e-14);
        assert!(
            (hurwitz_zeta(-1.0_f64, 0.25).unwrap() - 0.010_416_666_666_666_666).abs() < 1.0e-14
        );
    }

    #[test]
    fn public_error_precedence_is_preserved() {
        {
            let outcome = &(riemann_zeta(f64::NAN));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::NonFinite { .. }
            ));
        }
        {
            let outcome = &(riemann_zeta(1.0_f64));
            let err = outcome.as_ref().expect_err("expected error");
            assert!(matches!(
                err.view(),
                scioxide_core::SciOxideErrorView::DomainError { .. }
            ));
        }
        assert!(
            matches!(hurwitz_zeta(f64::NAN, f64::NAN), Err(error) if matches!(error.view(), scioxide_core::SciOxideErrorView::NonFinite { origin: scioxide_core::NonFiniteOrigin::Input, .. }))
        );
        assert!(
            matches!(hurwitz_zeta(1.0_f64, -1.0), Err(error) if match error.view() { scioxide_core::SciOxideErrorView::DomainError {  argument: _view_argument, reason: _view_reason, detail: _view_detail } => _view_argument == scioxide_core::argument!("s"), _ => false })
        );
        assert!(
            matches!(hurwitz_zeta(2.0_f64, 0.0), Err(error) if match error.view() { scioxide_core::SciOxideErrorView::DomainError {  argument: _view_argument, reason: _view_reason, detail: _view_detail } => _view_argument == scioxide_core::argument!("a") && _view_reason == scioxide_core::DomainReason::Pole, _ => false })
        );
    }
}
