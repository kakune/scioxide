//! Complex gamma and beta continuation, sharing logarithmic and pFq/U kernels.
use crate::complex_support::ScaledEvaluation;
use crate::support::Jet;
use crate::support::JetEstimate as Estimate;
use crate::support::Work;
use crate::support::widen_numeric as widen;
use crate::support::{C, ONE, ZERO};

use crate::hypergeometric_family::hypergeometric_kernel::{self as kernel};
use crate::{BetaOptions, GammaOptions};
use scioxide_core::AlgorithmComponent;
use scioxide_core::algorithm_component;
use scioxide_core::{
    Complex, DomainReason, NumericScalar, RealScalar, SciOxideError, SolveContext,
};

fn pole(z: C) -> Result<(), SciOxideError> {
    if crate::support::pole(z) {
        Err(SciOxideError::domain_error(
            scioxide_core::argument!("z"),
            DomainReason::Pole,
            "nonpositive integral gamma parameter",
        ))
    } else {
        Ok(())
    }
}
fn log_gamma<const DERIVATIVE: bool>(z: C, w: &mut Work<'_, '_, '_>) -> Result<Jet, SciOxideError> {
    pole(z)?;
    let (v, d) = if DERIVATIVE {
        crate::support::gamma_core::complex_log_gamma(z, &mut || w.try_step())?
    } else {
        (
            crate::support::gamma_core::complex_log_gamma_value(z, &mut || w.try_step())?,
            ZERO,
        )
    };
    // A logarithmic value can remain finite when its derivative overflows near
    // a pole. The shared report independently marks that derivative unavailable.
    if !v.is_finite() {
        return Err(crate::support::GAMMA_ERRORS.unstable());
    }
    Ok(Jet { v, d })
}
fn estimate(y: Jet, scale: Jet, error: f64, method: AlgorithmComponent) -> Estimate {
    Estimate::from(crate::support::JetData {
        derivative_error: None,
        y,
        scale,
        error,
        method,
    })
}
#[derive(Clone, Copy)]
enum GammaKind {
    Gamma,
    Log,
    Psi,
    Lower,
    Upper,
    P,
    Q,
}

/// γ(a,z) = z^a exp(-z) / a * sum z^n/(a+1)_n (DLMF 8.7.3).
/// Specializing the recurrence avoids the generic 1F1 continuation's
/// ill-scaled seeds when both the shape and the argument are large.
fn incomplete_lower_series(
    a: C,
    z: C,
    w: &mut Work<'_, '_, '_>,
) -> Result<Estimate, SciOxideError> {
    // Kummer's transformation on the left: M(1,a+1,z)=exp(z)M(a,a+1,-z).
    // Its terms are a*(-z)^n/(n!*(a+n)); this avoids cancelling exp(|z|)
    // sized terms on and near the negative-real cut.
    // Compare the saddle growth of the two series before selecting Kummer's
    // transformation. Merely testing Re(z)<0 loses many digits near the
    // imaginary axis, where the untransformed terms can already decrease.
    let radius = z.norm();
    let shape = a.norm();
    let growth = if radius > shape {
        radius - shape - shape * (radius / shape).ln()
    } else {
        0.0
    };
    let transformed = z.re < 0.0
        && (if a.re > 0.0 {
            growth > radius + z.re
        } else {
            z.re < -z.im.abs()
        });
    let mut term = ONE;
    let mut sum = ONE;
    let mut correction = ZERO;
    let mut magnitude = 1.0;
    let mut scale = 0.0;
    for n in 1usize.. {
        w.try_step()?;
        let index = C::new(n as f64, 0.0);
        term *= if transformed {
            (-z / index) * ((a + index - ONE) / (a + index))
        } else {
            z / (a + index)
        };
        let adjusted = term - correction;
        let next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
        magnitude += term.norm();
        if !sum.is_finite() || !magnitude.is_finite() {
            return Err(crate::support::GAMMA_ERRORS.unstable());
        }
        if magnitude > 1e100 {
            // Leave room for subsequent products and the accumulated error;
            // the external logarithmic scale retains the removed magnitude.
            term /= C::new(1e100, 0.0);
            sum /= C::new(1e100, 0.0);
            correction /= C::new(1e100, 0.0);
            magnitude /= 1e100;
            scale += 1e100_f64.ln();
        }
        let denominator = a + C::new((n + 1) as f64, 0.0);
        let ratio = if transformed {
            // |a+k-1| <= |a+k| once Re(a)+k >= 1/2.
            z.norm() / (n + 1) as f64
        } else {
            z.norm() / denominator.norm()
        };
        if denominator.re >= 0.5 && ratio < 1.0 {
            let remainder = term.norm() * ratio / (1.0 - ratio);
            if remainder <= 2.0 * f64::EPSILON * sum.norm() {
                return Ok(estimate(
                    Jet::constant(sum),
                    Jet::constant(C::new(scale, 0.0) - a.ln() + if transformed { z } else { ZERO }),
                    remainder + 8.0 * f64::EPSILON * (n as f64 + 1.0) * magnitude,
                    algorithm_component!("incomplete_gamma_lower_series"),
                ));
            }
        }
    }
    Err(crate::support::GAMMA_ERRORS.unstable())
}

/// Legendre's upper fraction, with modified Lentz updates (DLMF 8.9.2).
fn incomplete_upper_fraction(
    a: C,
    z: C,
    w: &mut Work<'_, '_, '_>,
) -> Result<Estimate, SciOxideError> {
    let tiny = f64::MIN_POSITIVE / f64::EPSILON;
    let floor = |value: C| {
        if value.norm() < tiny {
            C::new(tiny, 0.0)
        } else {
            value
        }
    };
    let mut b = z + ONE - a;
    let mut c = C::new(1.0 / tiny, 0.0);
    let mut d = ONE / floor(b);
    let mut value = d;
    let mut roundoff = 8.0 * f64::EPSILON * (1.0 + (z.norm() + 1.0 + a.norm()) / floor(b).norm());
    let mut quiet = 0;
    for n in 1usize.. {
        w.try_step()?;
        let index = C::new(n as f64, 0.0);
        let numerator = index * (a - index);
        b += C::new(2.0, 0.0);
        let d_term = numerator * d;
        let c_term = numerator / c;
        let d_denominator = floor(b + d_term);
        let c_denominator = floor(b + c_term);
        // Each denominator addition amplifies the preceding multiply/divide
        // errors by its cancellation factor. Counting steps alone misses this
        // conditioning for shapes far from the positive real axis.
        roundoff += 16.0
            * f64::EPSILON
            * (1.0
                + (b.norm() + d_term.norm()) / d_denominator.norm()
                + (b.norm() + c_term.norm()) / c_denominator.norm());
        d = ONE / d_denominator;
        c = c_denominator;
        let delta = c * d;
        value *= delta;
        if !value.is_finite() {
            return Err(crate::support::GAMMA_ERRORS.unstable());
        }
        let change = (delta - ONE).norm();
        quiet = if change <= 4.0 * f64::EPSILON {
            quiet + 1
        } else {
            0
        };
        if quiet == 3 {
            return Ok(Estimate::new(
                Jet::constant(value),
                (4.0 * change + roundoff) * value.norm(),
                algorithm_component!("incomplete_gamma_upper_fraction"),
            ));
        }
    }
    Err(crate::support::GAMMA_ERRORS.unstable())
}

/// One directly evaluated tail, in the requested normalization. Constant jets
/// keep the value's stopping rule independent of a potentially ill-scaled
/// argument derivative. The latter has a closed form and is attached below.
fn incomplete_tail(
    a: C,
    z: C,
    upper: bool,
    regularized: bool,
    logarithm: C,
    w: &mut Work<'_, '_, '_>,
) -> Result<Estimate, SciOxideError> {
    let mut e = if upper && z.re > 0.0 && (z.norm() > 1.0 || a.norm() > 1.0) {
        incomplete_upper_fraction(a, z, w)?
    } else if upper {
        kernel::hyperu(1.0.into(), Jet::constant(ONE + a), Jet::constant(z), w)?
    } else {
        incomplete_lower_series(a, z, w)?
    };
    e.scale = e.scale + Jet::constant(a * z.ln() - z);
    if regularized {
        e.scale = e.scale - Jet::constant(logarithm);
    }
    // Bound the rounded power and any gamma normalization, including their
    // cancellation in the final scale rather than only its final magnitude.
    let magnitude =
        (a * z.ln()).norm() + z.norm() + if regularized { logarithm.norm() } else { 0.0 };
    e.error += 32.0 * f64::EPSILON * (1.0 + magnitude) * e.y.v.norm();
    Ok(e)
}

/// Subtract in a common logarithmic scale, accepting only a cancellation
/// factor at most two. `None` means the other tail needs direct evaluation.
fn incomplete_complement(tail: Estimate, total_log: C) -> Option<Estimate> {
    let magnitude = tail.y.v.norm();
    let tail_log = tail.scale.v
        + C::new(
            if magnitude == 0.0 {
                0.0
            } else {
                magnitude.ln()
            },
            0.0,
        );
    let shift = if magnitude == 0.0 {
        total_log.re
    } else {
        total_log.re.max(tail_log.re)
    };
    let total = (total_log - C::new(shift, 0.0)).exp();
    let part = if magnitude == 0.0 {
        ZERO
    } else {
        (tail.y.v / C::new(magnitude, 0.0)) * (tail_log - C::new(shift, 0.0)).exp()
    };
    let value = total - part;
    let operands = total.norm() + part.norm();
    if operands > 2.0 * value.norm() {
        return None;
    }
    let propagated = if tail.error == 0.0 {
        0.0
    } else {
        (tail.error.ln() + tail.scale.v.re - shift).exp()
    };
    Some(estimate(
        Jet::constant(value),
        Jet::from(shift),
        propagated + 32.0 * f64::EPSILON * (1.0 + total_log.norm() + tail_log.norm()) * operands,
        algorithm_component!("incomplete_gamma_complement"),
    ))
}

fn evaluate<const DERIVATIVE: bool>(
    kind: GammaKind,
    a: C,
    z: C,
    w: &mut Work<'_, '_, '_>,
) -> Result<Estimate, SciOxideError> {
    w.check()?;
    if matches!(kind, GammaKind::Gamma | GammaKind::Log | GammaKind::Psi) {
        let l = if matches!(kind, GammaKind::Psi) {
            log_gamma::<true>(a, w)?
        } else {
            log_gamma::<DERIVATIVE>(a, w)?
        };
        return Ok(match kind {
            GammaKind::Gamma => estimate(
                1.0.into(),
                l,
                32.0 * f64::EPSILON * (1.0 + l.v.norm()),
                algorithm_component!("complex_gamma_logarithmic"),
            ),
            GammaKind::Log => estimate(
                l,
                0.0.into(),
                32.0 * f64::EPSILON * (1.0 + l.v.norm()),
                algorithm_component!("analytic_log_gamma"),
            ),
            _ => estimate(
                Jet::constant(l.d),
                0.0.into(),
                64.0 * f64::EPSILON * (1.0 + l.d.norm()),
                algorithm_component!("complex_digamma"),
            ),
        });
    }
    let regularized = matches!(kind, GammaKind::P | GammaKind::Q);
    let upper = matches!(kind, GammaKind::Upper | GammaKind::Q);
    if z == ZERO {
        if a.re <= 0.0 {
            if a == ZERO && matches!(kind, GammaKind::Upper) {
                return Err(crate::support::GAMMA_ERRORS
                    .domain(scioxide_core::argument!("z"), DomainReason::Pole));
            }
            return Err(crate::support::GAMMA_ERRORS
                .domain(scioxide_core::argument!("z"), DomainReason::Undefined));
        }
        return if upper {
            if regularized {
                Ok(estimate(
                    1.0.into(),
                    0.0.into(),
                    0.0,
                    algorithm_component!("gamma_origin"),
                ))
            } else {
                evaluate::<DERIVATIVE>(GammaKind::Gamma, a, ZERO, w)
            }
        } else {
            Ok(estimate(
                0.0.into(),
                0.0.into(),
                0.0,
                algorithm_component!("gamma_origin"),
            ))
        };
    }
    if regularized && crate::support::pole(a) {
        return Ok(estimate(
            if upper { 0.0.into() } else { 1.0.into() },
            0.0.into(),
            0.0,
            algorithm_component!("reciprocal_gamma_zero"),
        ));
    }
    let gamma_pole = crate::support::pole(a);
    let logarithm = if gamma_pole && upper {
        ZERO
    } else {
        log_gamma::<true>(a, w)?.v
    };
    // The lower series covers small arguments and the left half-plane up to
    // the U asymptotic region. The upper fraction handles the right half-plane
    // and transition; U handles small-shape and cut continuation. No hand-off
    // tests whether Im(a) is exactly zero.
    let primary_upper = gamma_pole
        || !(z.norm() <= a.norm() + 1.0 || z.re <= 0.0 && z.norm() <= 40.0 + 2.0 * a.norm());
    let primary = incomplete_tail(a, z, primary_upper, regularized, logarithm, w)?;
    let mut e = if primary_upper == upper {
        primary
    } else if let Some(complement) =
        incomplete_complement(primary, if regularized { ZERO } else { logarithm })
    {
        complement
    } else {
        incomplete_tail(a, z, upper, regularized, logarithm, w)?
    };
    // dγ/dz = z^(a-1) exp(-z), dΓ/dz = -dγ/dz. In particular, do not
    // recover this derivative by cancelling two large U-jet contributions.
    let power = (a - ONE) * z.ln();
    let normalization = if regularized { logarithm } else { ZERO };
    e.scale.d = ZERO;
    e.y.d =
        (power - z - normalization - e.scale.v).exp() * C::new(if upper { -1.0 } else { 1.0 }, 0.0);
    e.derivative_error = Some(if e.y.d == ZERO {
        // A nonzero analytic derivative lost while moving into the value's
        // scale is unavailable, not an exact zero derivative.
        f64::INFINITY
    } else {
        32.0 * f64::EPSILON
            * (1.0 + power.norm() + z.norm() + normalization.norm() + e.scale.v.norm())
            * e.y.d.norm()
    });
    Ok(e)
}
fn run<const DERIVATIVE: bool, S: RealScalar>(
    kind: GammaKind,
    a: Complex<S>,
    z: Complex<S>,
    options: GammaOptions,
    context: Option<&mut SolveContext<'_, '_>>,
) -> Result<ScaledEvaluation<Complex<S>>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let a = widen(a, "gamma parameter a must be finite")?;
    let z = widen(z, "gamma parameter z must be finite")?;
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    let e = evaluate::<DERIVATIVE>(kind, a, z, &mut work);
    // Only log-gamma is unscaled. Gamma always needs the canonical base-two
    // projection, so it has no reason to pass through the unscaled shortcut.
    if !DERIVATIVE && matches!(kind, GammaKind::Log) {
        return crate::complex_support::report_plain(e, &work, algorithm_component!("gamma"));
    }
    crate::complex_support::report(
        e,
        !matches!(kind, GammaKind::Psi) && z != ZERO
            || matches!(kind, GammaKind::Gamma | GammaKind::Log),
        &work,
        algorithm_component!("gamma"),
    )
}
pub(crate) fn gamma_complex<S: RealScalar>(
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    run::<{ matches!(GammaKind::Gamma, GammaKind::Psi) }, S>(
        GammaKind::Gamma,
        z,
        Complex::new(S::ZERO, S::ZERO),
        GammaOptions::default(),
        None,
    )?
    .value()
}

#[inline]
pub(crate) fn gamma_complex_evaluation<S: RealScalar>(
    z: Complex<S>,
    options: GammaOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    run::<true, S>(
        GammaKind::Gamma,
        z,
        Complex::new(S::ZERO, S::ZERO),
        options,
        context,
    )
}

pub(crate) fn ln_gamma_complex<S: RealScalar>(
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    run::<{ matches!(GammaKind::Log, GammaKind::Psi) }, S>(
        GammaKind::Log,
        z,
        Complex::new(S::ZERO, S::ZERO),
        GammaOptions::default(),
        None,
    )?
    .value()
}

#[inline]
pub(crate) fn ln_gamma_complex_evaluation<S: RealScalar>(
    z: Complex<S>,
    options: GammaOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    run::<true, S>(
        GammaKind::Log,
        z,
        Complex::new(S::ZERO, S::ZERO),
        options,
        context,
    )
}

pub(crate) fn digamma_complex<S: RealScalar>(
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    run::<{ matches!(GammaKind::Psi, GammaKind::Psi) }, S>(
        GammaKind::Psi,
        z,
        Complex::new(S::ZERO, S::ZERO),
        GammaOptions::default(),
        None,
    )?
    .value()
}

#[inline]
pub(crate) fn digamma_complex_evaluation<S: RealScalar>(
    z: Complex<S>,
    options: GammaOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    run::<true, S>(
        GammaKind::Psi,
        z,
        Complex::new(S::ZERO, S::ZERO),
        options,
        context,
    )
}

pub(crate) fn gamma_lower_complex<S: RealScalar>(
    a: Complex<S>,
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    run::<true, S>(GammaKind::Lower, a, z, GammaOptions::default(), None)?.value()
}

#[inline]
pub(crate) fn gamma_lower_complex_evaluation<S: RealScalar>(
    a: Complex<S>,
    z: Complex<S>,
    options: GammaOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    run::<true, S>(GammaKind::Lower, a, z, options, context)
}

pub(crate) fn gamma_upper_complex<S: RealScalar>(
    a: Complex<S>,
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    run::<true, S>(GammaKind::Upper, a, z, GammaOptions::default(), None)?.value()
}

#[inline]
pub(crate) fn gamma_upper_complex_evaluation<S: RealScalar>(
    a: Complex<S>,
    z: Complex<S>,
    options: GammaOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    run::<true, S>(GammaKind::Upper, a, z, options, context)
}

pub(crate) fn gamma_p_complex<S: RealScalar>(
    a: Complex<S>,
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    run::<true, S>(GammaKind::P, a, z, GammaOptions::default(), None)?.value()
}

#[inline]
pub(crate) fn gamma_p_complex_evaluation<S: RealScalar>(
    a: Complex<S>,
    z: Complex<S>,
    options: GammaOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    run::<true, S>(GammaKind::P, a, z, options, context)
}

pub(crate) fn gamma_q_complex<S: RealScalar>(
    a: Complex<S>,
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    run::<true, S>(GammaKind::Q, a, z, GammaOptions::default(), None)?.value()
}

#[inline]
pub(crate) fn gamma_q_complex_evaluation<S: RealScalar>(
    a: Complex<S>,
    z: Complex<S>,
    options: GammaOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    run::<true, S>(GammaKind::Q, a, z, options, context)
}

fn beta_evaluate(
    a: C,
    b: C,
    logarithmic: bool,
    w: &mut Work<'_, '_, '_>,
) -> Result<Estimate, SciOxideError> {
    let aa = log_gamma::<true>(a, w)?;
    let bb = log_gamma::<true>(b, w)?;
    if crate::support::pole(a + b) {
        return if logarithmic {
            Err(crate::support::GAMMA_ERRORS
                .domain(scioxide_core::argument!("b"), DomainReason::Pole))
        } else {
            Ok(estimate(
                0.0.into(),
                0.0.into(),
                0.0,
                algorithm_component!("beta_zero"),
            ))
        };
    }
    let ab = log_gamma::<true>(a + b, w)?;
    let l = Jet {
        v: aa.v + bb.v - ab.v,
        d: aa.d - ab.d,
    };
    let error = 64.0 * f64::EPSILON * (1.0 + aa.v.norm() + bb.v.norm() + ab.v.norm());
    Ok(if logarithmic {
        estimate(
            l,
            0.0.into(),
            error,
            algorithm_component!("complex_log_beta"),
        )
    } else {
        estimate(
            1.0.into(),
            l,
            error,
            algorithm_component!("complex_beta_logarithmic"),
        )
    })
}

fn beta_run<S: RealScalar>(
    a: Complex<S>,
    b: Complex<S>,
    options: BetaOptions,
    log: bool,
) -> Result<ScaledEvaluation<Complex<S>>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let (a, b) = (
        widen(a, "beta parameter a must be finite")?,
        widen(b, "beta parameter b must be finite")?,
    );
    pole(a)?;
    pole(b)?;
    let mut w = Work::new(options.iteration_budget().get());
    let e = beta_evaluate(a, b, log, &mut w);
    crate::complex_support::report(e, true, &w, algorithm_component!("beta"))
}
pub(crate) fn beta_complex<S: RealScalar>(
    a: Complex<S>,
    b: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options = <BetaOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    beta_run(a, b, options, false)?.value()
}

#[inline]
pub(crate) fn beta_complex_evaluation<S: RealScalar>(
    a: Complex<S>,
    b: Complex<S>,
    options: BetaOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    beta_run(a, b, options, false)
}

pub(crate) fn ln_beta_complex<S: RealScalar>(
    a: Complex<S>,
    b: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options = <BetaOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    beta_run(a, b, options, true)?.value()
}

#[inline]
pub(crate) fn ln_beta_complex_evaluation<S: RealScalar>(
    a: Complex<S>,
    b: Complex<S>,
    options: BetaOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    beta_run(a, b, options, true)
}

pub(crate) fn polygamma_complex<S: RealScalar>(
    n: usize,
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    polygamma_run::<false, S>(n, z, GammaOptions::default(), None)?.value()
}

#[inline]
pub(crate) fn polygamma_complex_evaluation<S: RealScalar>(
    n: usize,
    z: Complex<S>,
    options: GammaOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    polygamma_run::<true, S>(n, z, options, context)
}

fn polygamma_run<const DERIVATIVE: bool, S: RealScalar>(
    n: usize,
    z: Complex<S>,
    options: GammaOptions,
    context: Option<&mut SolveContext<'_, '_>>,
) -> Result<ScaledEvaluation<Complex<S>>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let order = n
        .checked_add(1)
        .ok_or_else(|| SciOxideError::size_overflow(scioxide_core::quantity!("polygamma_order")))?;
    let s = C::new(f64::try_from_usize_exact(order)?, 0.0);
    let z = widen(z, "gamma parameter z must be finite")?;
    pole(z)?;
    let mut w = Work::with_context(options.iteration_budget().get(), context);
    fn polygamma_estimate<const DERIVATIVE: bool>(
        n: usize,
        s: C,
        z: C,
        w: &mut Work<'_, '_, '_>,
    ) -> Result<Estimate, SciOxideError> {
        if n == 0 {
            let mut value = evaluate::<false>(GammaKind::Psi, z, ZERO, w)?;
            if DERIVATIVE {
                let derivative =
                    crate::zeta_polylog::complex_zeta::hurwitz(C::new(2.0, 0.0), z, w)?;
                value.y.d = derivative.value;
                value.derivative_error = Some(derivative.error);
            }
            return Ok(value);
        }
        let value = crate::zeta_polylog::complex_zeta::hurwitz(s, z, w)?;
        let derivative = if DERIVATIVE {
            let derivative = crate::zeta_polylog::complex_zeta::hurwitz(s + ONE, z, w)?;
            (derivative.value, derivative.error)
        } else {
            (ZERO, 0.0)
        };
        let l = log_gamma::<false>(s, w)?;
        let sign = if n.is_multiple_of(2) { -ONE } else { ONE };
        // A real sign scales each component, so a rounded-zero ζ keeps the
        // sign of the real path; a complex product by ±1 can cancel it.
        let mut result = estimate(
            Jet {
                v: C::new(sign.re * value.value.re, sign.re * value.value.im),
                d: -sign * s * derivative.0,
            },
            Jet::constant(l.v),
            value.error + 32.0 * f64::EPSILON * value.value.norm(),
            algorithm_component!("polygamma_hurwitz"),
        );
        if DERIVATIVE {
            result.derivative_error =
                Some(s.norm() * derivative.1 + 32.0 * f64::EPSILON * result.y.d.norm());
        }
        Ok(result)
    }
    let e = polygamma_estimate::<DERIVATIVE>(n, s, z, &mut w);
    crate::complex_support::report(e, DERIVATIVE, &w, algorithm_component!("polygamma"))
}
pub(crate) fn beta_inc_complex<S: RealScalar>(
    a: Complex<S>,
    b: Complex<S>,
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    beta_inc_run(a, b, z, BetaOptions::default(), None)?.value()
}

pub(crate) fn beta_inc_run<S: RealScalar>(
    a: Complex<S>,
    b: Complex<S>,
    z: Complex<S>,
    options: BetaOptions,
    context: Option<&mut SolveContext<'_, '_>>,
) -> Result<ScaledEvaluation<Complex<S>>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let (a, b, z) = (
        widen(a, "beta parameter a must be finite")?,
        widen(b, "beta parameter b must be finite")?,
        widen(z, "beta parameter z must be finite")?,
    );
    let mut w = Work::with_context(options.iteration_budget().get(), context);
    fn incomplete_beta_estimate(
        a: C,
        b: C,
        z: C,
        w: &mut Work<'_, '_, '_>,
    ) -> Result<Estimate, SciOxideError> {
        w.check()?;
        if a == ZERO && b == ZERO {
            return Err(SciOxideError::domain_error(
                scioxide_core::argument!("a"),
                DomainReason::Undefined,
                "regularized beta has no joint limit when both shapes are zero",
            ));
        }
        if z == ZERO || z == ONE {
            if (z == ZERO && a.re <= 0.0) || (z == ONE && b.re <= 0.0) {
                return Err(crate::support::GAMMA_ERRORS
                    .domain(scioxide_core::argument!("z"), DomainReason::Undefined));
            }
            return Ok(estimate(
                Jet::constant(z),
                0.0.into(),
                0.0,
                algorithm_component!("beta_endpoint"),
            ));
        }
        if a == ZERO && b != ZERO && !crate::support::pole(b) {
            return Ok(estimate(
                1.0.into(),
                0.0.into(),
                0.0,
                algorithm_component!("beta_zero_shape_limit"),
            ));
        }
        if crate::support::pole(b) && !crate::support::pole(a + b) && !crate::support::pole(a) {
            return Ok(estimate(
                0.0.into(),
                0.0.into(),
                0.0,
                algorithm_component!("reciprocal_beta_zero"),
            ));
        }
        let ab = log_gamma::<true>(a + b, w)?;
        let bb = log_gamma::<true>(b, w)?;
        let mut e = kernel::pfq(
            &[Jet::constant(a), Jet::constant(ONE - b)],
            &[Jet::constant(ONE + a)],
            Jet::variable(z),
            true,
            w,
        )?;
        e.scale = e.scale
            + Jet {
                v: a * z.ln() + ab.v - bb.v,
                d: a / z,
            };
        e.error += 64.0 * f64::EPSILON * (1.0 + ab.v.norm() + bb.v.norm()) * e.y.v.norm();
        Ok(e)
    }
    let e = incomplete_beta_estimate(a, b, z, &mut w);
    crate::complex_support::report(e, z != ZERO && z != ONE, &w, algorithm_component!("beta"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod polygamma_tests {
    use super::*;

    fn check<S: RealScalar>()
    where
        Complex<S>: NumericScalar<Real = S>,
    {
        for n in [0, 1, 5, 20] {
            for (re, im) in [(0.75, 0.25), (-3.5, -0.125), (20.0, 1.0)] {
                let z = Complex::new(
                    S::try_from_f64_lossy(re).unwrap(),
                    S::try_from_f64_lossy(im).unwrap(),
                );
                let plain = polygamma_run::<false, S>(n, z, GammaOptions::default(), None).unwrap();
                let detailed = polygamma_run::<true, S>(n, z, GammaOptions::default(), None)
                    .unwrap()
                    .publish()
                    .unwrap();
                let value = plain.value().unwrap();
                let expected = detailed.try_value().unwrap();
                assert_eq!(
                    value.re.into_f64().to_bits(),
                    expected.re.into_f64().to_bits()
                );
                assert_eq!(
                    value.im.into_f64().to_bits(),
                    expected.im.into_f64().to_bits()
                );
                let plain = plain.publish().unwrap();
                assert!(plain.counts().iterations() < detailed.counts().iterations());
                assert!(detailed.derivative().is_some());
                let options = GammaOptions::default().with_iteration_budget(
                    scioxide_core::Budget::try_from(plain.counts().iterations()).unwrap(),
                );
                assert_eq!(
                    polygamma_run::<false, S>(n, z, options, None)
                        .unwrap()
                        .value()
                        .unwrap(),
                    value
                );
                assert!(
                    polygamma_run::<true, S>(n, z, options, None)
                        .and_then(|evaluation| evaluation.value())
                        .is_err()
                );
            }
        }
    }

    #[test]
    fn plain_skips_derivative_work_without_changing_value() {
        check::<f64>();
        check::<f32>();
    }
}
