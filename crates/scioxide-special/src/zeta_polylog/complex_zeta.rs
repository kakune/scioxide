//! Complex zeta continuation shared with Lerch and polygamma.
use crate::ZetaOptions;
use crate::support::TRANSCENDENTAL_ERRORS as ERRORS;
use crate::support::TranscendentalEstimate as Estimate;
use crate::support::wide::Wide;
use crate::support::widen_numeric as widen;
use crate::support::{C, ONE, ZERO, real};
use crate::support::{Jet, Work};
use core::f64::consts::PI;
use scioxide_core::algorithm_component;
use scioxide_core::{
    Complex, DomainReason, NumericScalar, RealScalar, SciOxideError, SolveContext,
};
/// Hurwitz zeta estimate.
pub(crate) fn hurwitz(s: C, a: C, work: &mut Work) -> Result<Estimate, SciOxideError> {
    if s == ONE {
        return Err(ERRORS.domain(scioxide_core::argument!("s"), DomainReason::Pole));
    }
    if a == ONE {
        return zeta(s, work);
    }
    if s.im == 0.0 && a.im == 0.0 && a.re > 0.0 {
        let e = super::zeta::hurwitz_zeta_evaluate(s.re, a.re, work)?;
        // The positive-order envelope keeps one quantum for a value below the range.
        return Ok(Estimate::new(
            real(e.value),
            e.error
                .unwrap_or_else(|| (64.0 * f64::EPSILON * e.value.abs()).max(f64::from_bits(1))),
            e.method,
        ));
    }
    if s.re <= 0.0 {
        // Centre the real coordinate before evaluating DLMF 25.11.10.
        // Unlike an Euler--Maclaurin prefix, this recurrence does not
        // subtract an enormous integral from equally enormous powers.
        let shifts = (a.re - 0.5).floor();
        let centre = C::new(a.re - shifts, a.im);
        let delta = a - real(shifts + 1.0);
        if delta.norm() < 0.75 {
            if shifts.abs() > work.remaining as f64 {
                work.steps(work.remaining)?;
                work.try_step()?;
            }
            let mut e = if delta == ZERO {
                zeta(s, work)?
            } else if delta == real(-0.5) {
                // ζ(s,1/2) = (2^s-1)ζ(s) retains the symmetry zeros;
                // a Taylor sum would cancel to scalar rounding there.
                let mut e = zeta(s, work)?;
                let exponent = s * real(core::f64::consts::LN_2);
                let factor = crate::support::complex_expm1(exponent);
                e.value *= factor;
                e.error = factor.norm() * e.error
                    + 16.0 * f64::EPSILON * (1.0 + exponent.norm()) * e.value.norm();
                e
            } else {
                shifted_series(s, delta, work)?
            };
            let (mut re, mut im) = (Wide::from(e.value.re), Wide::from(e.value.im));
            for k in 0..shifts.abs() as usize {
                work.try_step()?;
                let argument = if shifts < 0.0 { a } else { centre } + real(k as f64);
                let (exponent, term) = if argument == ZERO {
                    // Lerch admits the polynomial continuation at a=0.
                    (ZERO, shift_power(s, argument))
                } else {
                    let exponent = -s * argument.ln();
                    (exponent, exponent.exp())
                };
                let sign = if shifts < 0.0 { 1.0 } else { -1.0 };
                re = re + Wide::from(sign * term.re);
                im = im + Wide::from(sign * term.im);
                e.error += f64::EPSILON * (16.0 + 8.0 * exponent.norm()) * term.norm();
            }
            e.value = C::new(re.value(), im.value());
            e.error += 4.0 * f64::EPSILON * e.value.norm();
            return Ok(e);
        }
        // Off the real axis, shifting a into an Euler--Maclaurin tail cancels
        // catastrophically once Re(s)<=0; Hermite's integral does not.
        return hermite(s, a, work);
    }
    euler_maclaurin(s, a, None, work)
}

/// Hermite's representation (DLMF 25.11.29) at b = a + shifts:
/// ζ(s,b) = b^{-s}/2 + b^{1-s}/(s-1) + i∫₀^∞ ((b+it)^{-s}-(b-it)^{-s})/(e^{2πt}-1) dt.
///
/// Re(b)>=1 keeps the branch points ±ib off the integration path. Below
/// |a|=|s|/π the leading powers exceed ζ(s,a) by orders of magnitude when
/// Re(s)<0, so a is then also shifted back to Re(b)<2; the recurrence terms
/// stay within a small factor of the result.
fn hermite(s: C, a: C, work: &mut Work) -> Result<Estimate, SciOxideError> {
    let shifts = if a.re < 1.0 {
        (1.0 - a.re).ceil()
    } else if a.norm() < s.norm() / PI {
        -(a.re - 1.0).floor()
    } else {
        0.0
    };
    if shifts.abs() > work.remaining as f64 {
        work.steps(work.remaining)?;
        work.try_step()?;
    }
    let b = a + real(shifts);
    let mut value = ZERO;
    let mut error = 0.0;
    let mut magnitude = 0.0;
    for k in 0..shifts.abs() as usize {
        work.try_step()?;
        let (start, sign) = if shifts > 0.0 { (a, 1.0) } else { (b, -1.0) };
        let exponent = -s * (start + real(k as f64)).ln();
        let term = exponent.exp();
        value += term * real(sign);
        magnitude += term.norm();
        error += f64::EPSILON * (8.0 + 2.0 * exponent.norm()) * term.norm();
    }
    let exponent = -s * b.ln();
    let power = exponent.exp();
    let pole = power * b / (s - ONE);
    value += power * real(0.5) + pole;
    let leading = 0.5 * power.norm() + pole.norm();
    magnitude += leading;
    error += f64::EPSILON * (8.0 + 2.0 * exponent.norm()) * leading;
    let (integral, integral_error, integral_magnitude) = hermite_integral(s, b, magnitude, work)?;
    value += integral;
    magnitude += integral_magnitude;
    if !value.is_finite() {
        return Err(ERRORS.unstable());
    }
    Ok(Estimate::new(
        value,
        error + integral_error + 8.0 * f64::EPSILON * magnitude,
        algorithm_component!("hurwitz_hermite_integral"),
    ))
}

/// Adaptive Gauss--Legendre panels of Hermite's integrand for Re(b)>0.
/// Returns the integral, its quadrature, rounding and tail bound, and the
/// integral of the integrand modulus.
fn hermite_integral(
    s: C,
    b: C,
    scale: f64,
    work: &mut Work,
) -> Result<(C, f64, f64), SciOxideError> {
    let gauss = |left: f64, right: f64, work: &mut Work| -> Result<_, SciOxideError> {
        let half = 0.5 * (right - left);
        let mid = left + half;
        let (mut sum, mut magnitude, mut rounding) = (ZERO, 0.0, 0.0);
        for &(node, weight) in &crate::support::GL32 {
            work.try_step()?;
            let (value, error) = hermite_integrand(s, b, mid + half * node);
            sum += value * real(weight * half);
            magnitude += value.norm() * weight * half;
            rounding += error * weight * half;
        }
        if !sum.is_finite() || !rounding.is_finite() {
            return Err(ERRORS.unstable());
        }
        Ok((sum, magnitude, rounding + 4.0 * f64::EPSILON * magnitude))
    };
    // |b±it| >= max(Re b, t-|Im b|), so past `reach` |s|/|b±it| <= π:
    // neither power grows faster than e^{πt} while the Bose factor decays at
    // least like e^{-2πt}.
    let reach = if b.re * PI >= s.norm() {
        0.0
    } else {
        b.im.abs() + s.norm() / PI
    };
    let (mut total, mut error, mut magnitude) = (ZERO, 0.0, 0.0);
    let mut left = 0.0;
    loop {
        let right = left + 2.0;
        let mut stack = [(0.0, 0.0, ZERO); 48];
        stack[0] = (left, right, gauss(left, right, work)?.0);
        let mut count = 1;
        while count > 0 {
            count -= 1;
            let (lo, hi, coarse) = stack[count];
            let mid = 0.5 * (lo + hi);
            let (l, lm, lr) = gauss(lo, mid, work)?;
            let (r, rm, rr) = gauss(mid, hi, work)?;
            let fine = l + r;
            let delta = (fine - coarse).norm();
            if delta <= 1e-14 * (lm + rm) + 4.0 * (lr + rr) {
                total += fine;
                error += delta + lr + rr;
                magnitude += lm + rm;
            } else {
                if count + 2 > stack.len() || mid == lo || mid == hi {
                    return Err(ERRORS.unstable());
                }
                stack[count] = (mid, hi, r);
                stack[count + 1] = (lo, mid, l);
                count += 2;
            }
        }
        left = right;
        if left >= reach {
            let bose = bose_log(left);
            let tail = ([C::new(0.0, left), C::new(0.0, -left)]
                .into_iter()
                .map(|shift| ((-s * (b + shift).ln()).re - bose).exp())
                .sum::<f64>())
                / PI;
            if tail <= 0.125 * f64::EPSILON * (scale + magnitude) {
                return Ok((total, error + tail, magnitude));
            }
        }
    }
}

/// ln(e^{2πt}-1) without overflow or cancellation.
fn bose_log(t: f64) -> f64 {
    let x = 2.0 * PI * t;
    if x < 1.0 {
        x.exp_m1().ln()
    } else {
        x + (-(-x).exp()).ln_1p()
    }
}

/// i((b+it)^{-s}-(b-it)^{-s})/(e^{2πt}-1) and a bound on its rounding.
/// The difference is formed as a power times expm1(-s ln((b+it)/(b-it))),
/// which keeps it as t→0. Both b±it lie in the right half-plane, so the
/// principal logarithm of their ratio is the difference of their logarithms.
fn hermite_integrand(s: C, b: C, t: f64) -> (C, f64) {
    let lower = b - C::new(0.0, t);
    let ratio = C::new(0.0, 2.0 * t) / lower;
    let log_ratio = if ratio.norm() < 0.5 {
        // Componentwise log1p; Re retains |1+w|²-1 without cancellation.
        let delta = ratio.re * (2.0 + ratio.re) + ratio.im * ratio.im;
        C::new(0.5 * delta.ln_1p(), ratio.im.atan2(1.0 + ratio.re))
    } else {
        ((b + C::new(0.0, t)) / lower).ln()
    };
    let x = -s * log_ratio;
    let base = -s * lower.ln() - real(bose_log(t));
    // Factor out the larger power so neither exponential underflows first.
    let value = if x.re > 0.0 {
        C::new(0.0, -1.0) * (base + x).exp() * crate::support::complex_expm1(-x)
    } else {
        C::new(0.0, 1.0) * base.exp() * crate::support::complex_expm1(x)
    };
    let largest = (base.re + x.re.max(0.0)).exp();
    let rounding = f64::EPSILON
        * ((8.0 + 2.0 * (base.norm() + x.norm())) * value.norm() + 8.0 * x.norm() * largest);
    (value, rounding)
}

// An optional exact displacement from s=1 evaluates (s-1)ζ(s), with
// its integral pole removed before arithmetic. This retains displacements
// smaller than an ULP of one in the centred Hurwitz coefficients.
fn euler_maclaurin(
    s: C,
    a: C,
    pole_offset: Option<C>,
    work: &mut Work,
) -> Result<Estimate, SciOxideError> {
    if pole_offset == Some(ZERO) {
        return Ok(Estimate::new(
            ONE,
            0.0,
            algorithm_component!("zeta_pole_product"),
        ));
    }
    let factor = pole_offset.unwrap_or(ONE);
    let mut target = (16.0 + 2.0 * s.norm()).max(a.im.abs());
    loop {
        let mut x = a;
        let mut value = ZERO;
        let mut magnitude = 0.0;
        // A rounded exponent w=-s ln(x) perturbs exp(w) by about ε|w|
        // relative; at large |s| this dominates the summation rounding.
        let mut exponents = 0.0;
        while x.re < target {
            work.try_step()?;
            let exponent = -s * x.ln();
            let term = factor * exponent.exp();
            value += term;
            magnitude += term.norm();
            exponents += term.norm() * exponent.norm();
            let next = C::new(x.re + 1.0, x.im);
            if next == x {
                return Err(ERRORS.unstable());
            }
            x = next;
        }
        let exponent = -s * x.ln();
        let power = exponent.exp();
        let integral = if let Some(offset) = pole_offset {
            (-offset * x.ln()).exp()
        } else {
            power * x / (s - ONE)
        };
        value += integral + factor * power * real(0.5);
        magnitude += integral.norm() + factor.norm() * power.norm() * 0.5;
        let inverse = ONE / x;
        let mut rising = s;
        let mut xpower = power * inverse;
        let mut error = f64::MAX;
        for (j, &coefficient) in crate::support::gamma_series::BERNOULLI_OVER_FACTORIAL
            .iter()
            .enumerate()
        {
            work.try_step()?;
            let term = factor * rising * xpower * real(coefficient.hi);
            if j == 12 {
                error = term.norm();
                break;
            }
            value += term;
            magnitude += term.norm();
            rising *= (s + real((2 * j + 1) as f64)) * (s + real((2 * j + 2) as f64));
            xpower *= inverse * inverse;
        }
        if error <= 2e-15 * value.norm().max(1.0) {
            // Every tail term carries the factor exp(-s ln x).
            return Ok(Estimate::new(
                value,
                error
                    + f64::EPSILON
                        * (32.0 * magnitude + 2.0 * (exponents + exponent.norm() * magnitude)),
                algorithm_component!("complex_euler_maclaurin"),
            ));
        }
        target *= 2.0;
        if !target.is_finite() {
            return Err(ERRORS.unstable());
        }
    }
}
// DLMF 25.11.10, expanded about a=1. This avoids subtracting enormous
// Euler--Maclaurin integral terms when the requested exponent is negative.
fn shifted_series(s: C, delta: C, work: &mut Work) -> Result<Estimate, SciOxideError> {
    let mut coefficient = ONE;
    let mut sum = ZERO;
    let mut correction = ZERO;
    let mut magnitude = 0.0;
    let mut error = 0.0;
    for k in 0usize.. {
        work.try_step()?;
        let mut terminal = false;
        let (e, weight) = if k == 0 {
            (zeta(s, work)?, coefficient)
        } else {
            let partial = coefficient * (-delta) / real(k as f64);
            let factor = s + real((k - 1) as f64);
            coefficient = partial * factor;
            terminal = factor == ZERO;
            if factor.norm() < 0.5 {
                (
                    euler_maclaurin(ONE + factor, ONE, Some(factor), work)?,
                    partial,
                )
            } else {
                (zeta(s + real(k as f64), work)?, coefficient)
            }
        };
        let term = weight * e.value;
        let adjusted = term - correction;
        let next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
        magnitude += term.norm();
        error += weight.norm() * e.error + 16.0 * (k + 1) as f64 * f64::EPSILON * term.norm();
        if terminal {
            return Ok(Estimate::new(
                sum,
                error + 32.0 * f64::EPSILON * magnitude,
                algorithm_component!("hurwitz_shift_series"),
            ));
        }
        let ratio = delta.norm() * ((k as f64 + s.norm()) / (k + 1) as f64).max(1.0);
        if s.re + k as f64 > 2.0 && ratio < 1.0 {
            let tail = 2.0 * coefficient.norm() * ratio / (1.0 - ratio);
            if tail < 2e-15 * sum.norm().max(1e-300) {
                return Ok(Estimate::new(
                    sum,
                    error + tail + 32.0 * f64::EPSILON * magnitude,
                    algorithm_component!("hurwitz_shift_series"),
                ));
            }
        }
        if !coefficient.is_finite() || !sum.is_finite() {
            return Err(ERRORS.unstable());
        }
    }
    Err(ERRORS.unstable())
}
/// Riemann zeta estimate.
pub(crate) fn zeta(s: C, work: &mut Work) -> Result<Estimate, SciOxideError> {
    if s == ONE {
        return Err(ERRORS.domain(scioxide_core::argument!("s"), DomainReason::Pole));
    }
    if s.im == 0.0 && s.re < 0.0 && (s.re * 0.5).floor() == s.re * 0.5 {
        return Ok(Estimate::new(
            ZERO,
            0.0,
            algorithm_component!("zeta_trivial_zero"),
        ));
    }
    if s.re < -0.5 {
        // Reflection avoids large cancellation in the Euler--Maclaurin sum.
        let positive = zeta(ONE - s, work)?;
        let gamma_log =
            crate::support::gamma_core::complex_log_gamma_value(ONE - s, &mut || work.try_step())?;
        let two_log = s * real(core::f64::consts::LN_2);
        let pi_log = (s - ONE) * real(PI.ln());
        let logarithm = two_log + pi_log + gamma_log;
        let sine = scioxide_core::sin_cos_pi(s * real(0.5))?.0;
        let multiplier = logarithm.exp() * sine;
        let value = multiplier * positive.value;
        // A rounded log perturbs exp(log) by its absolute, rather than
        // relative, error. Include the magnitudes before their cancellation,
        // the Lanczos log-gamma rounding and the final products. Pi-phase
        // reduction preserves exact half periods even next to trivial zeros.
        let relative =
            f64::EPSILON * (64.0 + 16.0 * (two_log.norm() + pi_log.norm() + gamma_log.norm()));
        return Ok(Estimate::new(
            value,
            multiplier.norm() * positive.error + relative * value.norm(),
            algorithm_component!("zeta_reflection"),
        ));
    }
    if s.im == 0.0 && s.re > 1.0 {
        let value = real(crate::support::hurwitz::native(s.re, 1.0, &mut || {
            work.try_step()
        })?);
        return Ok(Estimate::new(
            value,
            64.0 * f64::EPSILON * value.norm(),
            algorithm_component!("hurwitz_scaled_euler_maclaurin"),
        ));
    }
    euler_maclaurin(s, ONE, None, work)
}

pub(crate) fn shift_power(s: C, a: C) -> C {
    if s == ZERO {
        ONE
    } else if a == ZERO && s.im == 0.0 && s.re < 0.0 && s.re == s.re.floor() {
        ZERO
    } else {
        (-s * a.ln()).exp()
    }
}

fn run<S: RealScalar>(
    s: Complex<S>,
    a: Option<Complex<S>>,
    eta: bool,
    options: ZetaOptions,
    context: Option<&mut SolveContext<'_, '_>>,
) -> Result<crate::complex_support::ScaledEvaluation<Complex<S>>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    use crate::support::{JetEstimate as ScaledEstimate, Work};
    let s = widen(s, "zeta parameter s must be finite")?;
    let a = a
        .map(|a| widen(a, "zeta parameter a must be finite"))
        .transpose()?;
    if let Some(a) = a
        && crate::support::pole(a)
    {
        return Err(ERRORS.domain(scioxide_core::argument!("a"), DomainReason::Pole));
    }
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    fn zeta_estimate(
        eta: bool,
        s: C,
        a: Option<C>,
        work: &mut Work<'_, '_, '_>,
    ) -> Result<ScaledEstimate, SciOxideError> {
        work.check()?;
        let e = if eta && s == ONE {
            Estimate::new(
                real(core::f64::consts::LN_2),
                0.0,
                algorithm_component!("eta_removable_pole"),
            )
        } else {
            let mut e = if let Some(a) = a {
                hurwitz(s, a, work)?
            } else {
                zeta(s, work)?
            };
            if eta {
                let exponent = (ONE - s) * real(core::f64::consts::LN_2);
                // expm1 avoids cancellation near the removable pole at s=1.
                let factor = -crate::support::complex_expm1(exponent);
                e.value *= factor;
                e.error *= factor.norm();
                e.error += 16.0 * f64::EPSILON * e.value.norm();
            }
            e
        };
        Ok(ScaledEstimate::from(crate::support::JetData {
            derivative_error: None,
            y: Jet::constant(e.value),
            scale: 0.0.into(),
            error: e.error,
            method: e.method,
        }))
    }
    let value = zeta_estimate(eta, s, a, &mut work);
    crate::complex_support::report(value, false, &work, algorithm_component!("zeta"))
}
pub(crate) fn riemann_zeta_complex<S: RealScalar>(
    s: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    run(s, None, false, ZetaOptions::default(), None)?.value()
}

#[inline]
pub(crate) fn riemann_zeta_complex_evaluation<S: RealScalar>(
    s: Complex<S>,
    options: ZetaOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    run(s, None, false, options, context)
}

pub(crate) fn dirichlet_eta_complex<S: RealScalar>(
    s: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    run(s, None, true, ZetaOptions::default(), None)?.value()
}

#[inline]
pub(crate) fn dirichlet_eta_complex_evaluation<S: RealScalar>(
    s: Complex<S>,
    options: ZetaOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    run(s, None, true, options, context)
}

pub(crate) fn hurwitz_zeta_complex<S: RealScalar>(
    s: Complex<S>,
    a: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    run(s, Some(a), false, ZetaOptions::default(), None)?.value()
}

#[inline]
pub(crate) fn hurwitz_zeta_complex_evaluation<S: RealScalar>(
    s: Complex<S>,
    a: Complex<S>,
    options: ZetaOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    run(s, Some(a), false, options, context)
}
