//! Entire Airy continuation and explicit principal/damped scaling conventions.
use crate::bessel_airy::airy_kernel::{self as kernel, Kind};
use crate::support::Jet;
use crate::support::JetEstimate as Estimate;
use crate::support::Work;
use crate::support::{C, real};

use crate::AiryOptions;
use crate::result::Either;
use scioxide_core::ScaledValue as Z;
use scioxide_core::{AlgorithmComponent, algorithm_component};
use scioxide_core::{Complex, NumericScalar, RealScalar, SciOxideError, SolveContext};
fn phase(x: f64) -> C {
    let (s, c) = RealScalar::sin_cos(x);
    C::new(c, s)
}
fn argument(z: C) -> Result<C, SciOxideError> {
    Z::from_value(z)?
        .mul(Z::from_value(z.sqrt())?)?
        .mul(Z::from_value(scioxide_core::Complex::new(2.0 / 3.0, 0.0))?)?
        .round()
}
fn delta(z: C) -> Result<C, SciOxideError> {
    let base = if z.re < 0.0 {
        C::new(0.0, (-z.re).sqrt().copysign(z.im))
    } else {
        real(z.re.sqrt())
    };
    let root = z.sqrt();
    let factor = Z::from_value(scioxide_core::Complex::new(z.re, 0.0))?
        .div(Z::from_value(root + base)?)?
        .add(Z::from_value(root)?)?;
    Z::from_value(C::new(0.0, z.im))?
        .mul(factor)?
        .mul(Z::from_value(scioxide_core::Complex::new(2.0 / 3.0, 0.0))?)?
        .round()
}
fn negative_phase(theta: f64, lower: bool) -> C {
    let unit = phase(-theta)
        * C::new(
            core::f64::consts::FRAC_1_SQRT_2,
            core::f64::consts::FRAC_1_SQRT_2,
        );
    if lower { unit.conj() } else { unit }
}
fn rescale(mut e: Estimate, log: f64, unit: C, uncertainty: f64) -> Estimate {
    e.y.v *= unit;
    e.y.d *= unit;
    e.scale.v.re += log;
    e.error += (uncertainty + 8.0 * f64::EPSILON) * e.y.v.norm();
    e.derivative_error =
        Some(e.derivative_error.unwrap_or(0.0) + (uncertainty + 8.0 * f64::EPSILON) * e.y.d.norm());
    e
}
fn near_real(z: C, kind: Kind, scaled: bool, work: &mut Work) -> Result<Estimate, SciOxideError> {
    let seed = kernel::real_seed(z.re, kind, z.re > 0.0, work)?;
    if !work.complete {
        return Ok(crate::support::interrupted());
    }
    let e = seed.estimate;
    let h = C::new(0.0, z.im);
    let frequency = z.re.abs().sqrt();
    let travel = z.im.abs() * frequency;
    let a = Jet {
        v: real(-(z.im * frequency).powi(2) * z.re.signum()),
        d: C::new(0.0, 2.0 * z.re * z.im),
    };
    let b = Jet {
        v: C::new(0.0, -z.im * z.im * z.im),
        d: real(-3.0 * z.im * z.im),
    };
    let mut terms = [
        Jet::constant(e.y.v),
        Jet {
            v: h * e.y.d,
            d: e.y.d,
        },
        Jet::constant(real(0.0)),
    ];
    let mut sum = terms[0] + terms[1];
    let mut magnitude = terms[0].v.norm() + terms[1].v.norm();
    let mut dmagnitude = e.y.d.norm();
    let mut recent = [(0.0_f64, 0.0_f64); 3];
    let mut n = 1_usize;
    let (tail, dtail) = loop {
        if !work.poll_step()? {
            return Ok(crate::support::interrupted());
        }
        n += 1;
        let mut term = a * terms[(n - 2) % 3];
        if n >= 3 {
            term = term + b * terms[(n - 3) % 3];
        }
        term = term / Jet::from(n as f64 * (n as f64 - 1.0));
        terms[n % 3] = term;
        sum = sum + term;
        magnitude += term.v.norm();
        dmagnitude += term.d.norm();
        recent[n % 3] = (term.v.norm(), term.d.norm());
        let tail = recent.iter().map(|x| x.0).fold(0.0, f64::max);
        let dtail = recent.iter().map(|x| x.1).fold(0.0, f64::max);
        if n >= 6 && tail <= f64::EPSILON * magnitude && dtail <= f64::EPSILON * dmagnitude {
            break (tail, dtail);
        }
    };
    let seed_error =
        (e.error + e.derivative_error.unwrap_or(0.0) / frequency) * (2.0 * travel).exp();
    let mut result = Estimate::from(crate::support::JetData {
        y: sum,
        scale: e.scale,
        error: seed_error + 8.0 * tail + 16.0 * f64::EPSILON * (n + 1) as f64 * magnitude,
        derivative_error: Some(
            frequency * seed_error
                + 8.0 * dtail
                + 16.0 * f64::EPSILON * (n + 1) as f64 * dmagnitude,
        ),
        method: algorithm_component!("airy_near_real_ode"),
    });
    if scaled {
        let difference = delta(z)?;
        let uncertainty =
            64.0 * f64::EPSILON * (1.0 + difference.re.abs().max(difference.im.abs()));
        if z.re > 0.0 {
            result = if matches!(kind, Kind::Ai) {
                rescale(result, difference.re, phase(difference.im), uncertainty)
            } else {
                rescale(result, -difference.re, real(1.0), uncertainty)
            };
        } else if matches!(kind, Kind::Ai) {
            let (theta, error) = if let Some(p) = seed.phase {
                p
            } else {
                let Some(p) = kernel::phase(-z.re, work)? else {
                    return Ok(crate::support::interrupted());
                };
                p
            };
            result = rescale(
                result,
                difference.re,
                negative_phase(theta, z.im.is_sign_negative()) * phase(difference.im),
                uncertainty + error,
            );
        } else {
            result = rescale(result, -difference.re.abs(), real(1.0), uncertainty);
        }
    } else if z.re > 0.0 {
        let exponent = ((2.0 / 3.0) * z.re) * frequency;
        result = rescale(
            result,
            if matches!(kind, Kind::Ai) {
                -exponent
            } else {
                exponent
            },
            real(1.0),
            16.0 * f64::EPSILON * (1.0 + exponent),
        );
    }
    Ok(result)
}
/// DLMF 9.7: generate the two Olver waves, keeping only exponentially relevant
/// ones. Near the negative axis the phase is reduced from the exact real seed;
/// elsewhere an explicit argument-rounding estimate decides acceptance.
fn asymptotic(z: C, kind: Kind, scaled: bool, work: &mut Work) -> Result<Estimate, SciOxideError> {
    let root = z.sqrt();
    let quarter = root.sqrt();
    let inverse = (real(1.5) / z) / root;
    let a = real(0.5 / core::f64::consts::PI.sqrt()) / quarter;
    let b = quarter * (0.5 / core::f64::consts::PI.sqrt());
    let negative = z.re < 0.0 && z.im.abs() <= 0.25 * z.re.abs();
    let travel = z.im.abs() * z.re.abs().sqrt();
    let Some(minus) = kernel::coefficients(-inverse, work)? else {
        return Ok(crate::support::interrupted());
    };
    let primary = Estimate::from(crate::support::JetData {
        y: Jet {
            v: a * minus.u,
            d: -b * minus.v,
        },
        scale: Jet::constant(real(0.0)),
        error: a.norm() * minus.error,
        derivative_error: Some(b.norm() * minus.derivative_error),
        method: algorithm_component!("airy_olver_complex"),
    });
    // The exponential cancels analytically in scaled Ai. Do not form its phase
    // (or its logarithm, which can exceed even binary64) just to cancel it again.
    if scaled && matches!(kind, Kind::Ai) && (!negative || travel > 32.0) {
        return Ok(primary);
    }
    let (exponent, unit, uncertainty) = if negative {
        let Some((theta, error)) = kernel::phase(-z.re, work)? else {
            return Ok(crate::support::interrupted());
        };
        let difference = delta(z)?;
        (
            difference.re,
            negative_phase(theta, z.im.is_sign_negative()) * phase(difference.im),
            error + 64.0 * f64::EPSILON * (1.0 + difference.norm()),
        )
    } else if scaled && matches!(kind, Kind::Bi) && z.re > 32.0 && z.im.abs() <= 0.25 * z.re {
        // Re(zeta) is positive here. The negative wave is negligible; computing
        // only the phase difference also supports an unrepresentable real zeta.
        let difference = delta(z)?;
        let Some(plus) = kernel::coefficients(inverse, work)? else {
            return Ok(crate::support::interrupted());
        };
        let e = Estimate::from(crate::support::JetData {
            y: Jet {
                v: a * plus.u * 2.0,
                d: b * plus.v * 2.0,
            },
            scale: Jet::constant(real(0.0)),
            error: 2.0 * a.norm() * plus.error,
            derivative_error: Some(2.0 * b.norm() * plus.derivative_error),
            method: algorithm_component!("airy_olver_complex"),
        });
        return Ok(rescale(
            e,
            0.0,
            phase(difference.im),
            64.0 * f64::EPSILON * (1.0 + difference.norm()),
        ));
    } else {
        let exponent = argument(z)?;
        (
            exponent.re,
            phase(exponent.im),
            32.0 * f64::EPSILON * (1.0 + exponent.norm()),
        )
    };
    if matches!(kind, Kind::Ai) && !negative {
        return Ok(rescale(primary, -exponent, unit.conj(), uncertainty));
    }
    let Some(plus) = kernel::coefficients(inverse, work)? else {
        return Ok(crate::support::interrupted());
    };
    let side = C::new(0.0, if z.im.is_sign_negative() { -1.0 } else { 1.0 });
    let (left, right, log) = if matches!(kind, Kind::Ai) {
        if scaled {
            (real(1.0), side * unit * unit * (2.0 * exponent).exp(), 0.0)
        } else {
            (unit.conj(), side * unit * (2.0 * exponent).exp(), -exponent)
        }
    } else {
        let common = exponent.abs();
        (
            side * unit.conj() * (-exponent - common).exp(),
            unit * (exponent - common).exp() * if negative { 1.0 } else { 2.0 },
            if scaled { 0.0 } else { common },
        )
    };
    let v = a * (left * minus.u + right * plus.u);
    let d = b * (-left * minus.v + right * plus.v);
    let magnitude = a.norm() * (left.norm() * minus.u.norm() + right.norm() * plus.u.norm());
    let dmagnitude = b.norm() * (left.norm() * minus.v.norm() + right.norm() * plus.v.norm());
    let rounding = uncertainty + 16.0 * f64::EPSILON;
    Ok(Estimate::from(crate::support::JetData {
        y: Jet { v, d },
        scale: Jet::constant(real(log)),
        error: a.norm() * (left.norm() * minus.error + right.norm() * plus.error)
            + rounding * magnitude,
        derivative_error: Some(
            b.norm()
                * (left.norm() * minus.derivative_error + right.norm() * plus.derivative_error)
                + rounding * dmagnitude,
        ),
        method: algorithm_component!("airy_olver_complex_waves"),
    }))
}
// Entire ODE transport along the oscillatory Bi rays. Short Taylor steps avoid
// cancellation in the origin series and fractional-Bessel connections near zeros.
fn oscillatory_transport(z: C, kind: Kind, work: &mut Work) -> Result<Estimate, SciOxideError> {
    let mut result = kernel::series(real(0.0), kind, work)?;
    let steps = (2.0 * z.norm() * z.norm().sqrt()).ceil() as usize;
    let h = z / steps as f64;
    let h2 = h * h;
    let h3 = h2 * h;
    let mut point = real(0.0);
    for _ in 0..steps {
        let mut terms = [result.y.v, result.y.d * h, real(0.0)];
        let mut value = terms[0] + terms[1];
        let mut derivative = result.y.d;
        let mut absolute = terms[0].norm() + terms[1].norm();
        let mut dabsolute = derivative.norm();
        let mut recent = [f64::INFINITY; 3];
        let mut n = 2usize;
        loop {
            if !work.poll_step()? {
                return Ok(crate::support::interrupted());
            }
            let mut term = point * h2 * terms[(n - 2) % 3];
            if n >= 3 {
                term += h3 * terms[(n - 3) % 3];
            }
            term /= (n * (n - 1)) as f64;
            terms[n % 3] = term;
            value += term;
            let dterm = term * (n as f64) / h;
            derivative += dterm;
            absolute += term.norm();
            dabsolute += dterm.norm();
            recent[n % 3] = term.norm().max(dterm.norm());
            if n >= 6
                && recent
                    .iter()
                    .all(|v| *v <= f64::EPSILON * absolute.max(dabsolute))
            {
                break;
            }
            n += 1;
        }
        result.y = Jet {
            v: value,
            d: derivative,
        };
        result.error += 4.0 * f64::EPSILON * absolute;
        result.derivative_error =
            Some(result.derivative_error.unwrap_or(0.0) + 4.0 * f64::EPSILON * dabsolute);
        point += h;
    }
    result.method = algorithm_component!("airy_oscillatory_ODE_transport");
    Ok(result)
}
fn raw(z: C, kind: Kind, scaled: bool, work: &mut Work) -> Result<Estimate, SciOxideError> {
    if z.im == 0.0 {
        let seed = kernel::real_seed(z.re, kind, scaled && z.re > 0.0, work)?;
        if !work.complete {
            return Ok(crate::support::interrupted());
        }
        let mut e = seed.estimate;
        e.y.v.im = e.y.d.re * z.im;
        e.y.d.im = (z.re * e.y.v.re) * z.im;
        if scaled && z.re < 0.0 && matches!(kind, Kind::Ai) {
            let (theta, error) = if let Some(p) = seed.phase {
                p
            } else {
                let Some(p) = kernel::phase(-z.re, work)? else {
                    return Ok(crate::support::interrupted());
                };
                p
            };
            e = rescale(
                e,
                0.0,
                negative_phase(theta, z.im.is_sign_negative()),
                error,
            );
        }
        return Ok(e);
    }
    if !work.poll_step()? {
        return Ok(crate::support::interrupted());
    }
    if z.norm() > 32.0 && scaled && matches!(kind, Kind::Ai) && z.re > 0.0 {
        return asymptotic(z, kind, scaled, work);
    }
    if z.re.abs() >= 4.0 && z.im.abs() * z.re.abs().sqrt() <= 0.25 {
        return near_real(z, kind, scaled, work);
    }
    if z.norm() > 32.0 {
        return asymptotic(z, kind, scaled, work);
    }
    let mut e = if matches!(kind, Kind::Bi)
        && z.re > 0.0
        && (z.im.abs() / z.re - 3.0_f64.sqrt()).abs() < 0.6
        && z.norm() > 4.0
    {
        oscillatory_transport(z, kind, work)?
    } else if z.norm() <= 4.0 {
        kernel::series(z, kind, work)?
    } else {
        kernel::bessel_pair(z, kind, work)?
    };
    if !work.complete {
        return Ok(crate::support::interrupted());
    }
    if scaled {
        let exponent = argument(z)?;
        e = if matches!(kind, Kind::Ai) {
            rescale(
                e,
                exponent.re,
                phase(exponent.im),
                16.0 * f64::EPSILON * (1.0 + exponent.norm()),
            )
        } else {
            rescale(
                e,
                -exponent.re.abs(),
                real(1.0),
                16.0 * f64::EPSILON * (1.0 + exponent.norm()),
            )
        };
    }
    Ok(e)
}
fn evaluate<S: RealScalar>(
    z: Complex<S>,
    kind: Kind,
    scaled: bool,
    options: AiryOptions,
    family: AlgorithmComponent,
    parent: Option<&mut SolveContext<'_, '_>>,
) -> Result<crate::bessel_airy::airy::AiryEvaluation<Complex<S>>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    if !z.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "Airy coordinate must be finite",
        ));
    }
    let z = C::new(z.re.try_to_f64_exact()?, z.im.try_to_f64_exact()?);
    let mut work = Work::with_context(options.iteration_budget().get(), parent);
    if !scaled && z.re > 0.0 && z.im.abs() * z.re.sqrt() <= 0.25 {
        let exponent = ((2.0 / 3.0) * z.re) * z.re.sqrt();
        if !exponent.is_finite() || exponent > S::MAX.try_to_f64_exact()? {
            if work.poll_step()? {
                return if matches!(kind, Kind::Ai) {
                    Ok(crate::bessel_airy::airy::underflow(work.used))
                } else {
                    Err(crate::support::instability(
                        "Airy value and logarithm exceed the output scalar range",
                    ))
                };
            }
            return Ok(crate::complex_support::report_with_status(
                Ok(crate::support::interrupted()),
                false,
                family,
                work.used,
                true,
            )?
            .map_form(Either::Left));
        }
    }
    let result = raw(z, kind, scaled, &mut work);
    work.check()?;
    Ok(
        crate::complex_support::report_with_status(
            result,
            true,
            family,
            work.used,
            !work.complete,
        )?
        .map_form(Either::Left),
    )
}
pub(crate) fn airy_ai_complex<S: RealScalar>(
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    evaluate(
        z,
        Kind::Ai,
        false,
        AiryOptions::default(),
        algorithm_component!(stringify!(airy_ai_complex)),
        None,
    )?
    .value()
}

#[inline]
pub(crate) fn airy_ai_complex_evaluation<S: RealScalar>(
    z: Complex<S>,
    options: AiryOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    evaluate(
        z,
        Kind::Ai,
        false,
        options,
        algorithm_component!(stringify!(airy_ai_complex)),
        context,
    )
}

pub(crate) fn airy_bi_complex<S: RealScalar>(
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    evaluate(
        z,
        Kind::Bi,
        false,
        AiryOptions::default(),
        algorithm_component!(stringify!(airy_bi_complex)),
        None,
    )?
    .value()
}

#[inline]
pub(crate) fn airy_bi_complex_evaluation<S: RealScalar>(
    z: Complex<S>,
    options: AiryOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    evaluate(
        z,
        Kind::Bi,
        false,
        options,
        algorithm_component!(stringify!(airy_bi_complex)),
        context,
    )
}

pub(crate) fn airy_ai_complex_scaled<S: RealScalar>(
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    evaluate(
        z,
        Kind::Ai,
        true,
        AiryOptions::default(),
        algorithm_component!(stringify!(airy_ai_complex_scaled)),
        None,
    )?
    .value()
}

#[inline]
pub(crate) fn airy_ai_complex_scaled_evaluation<S: RealScalar>(
    z: Complex<S>,
    options: AiryOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    evaluate(
        z,
        Kind::Ai,
        true,
        options,
        algorithm_component!(stringify!(airy_ai_complex_scaled)),
        context,
    )
}

pub(crate) fn airy_bi_complex_scaled<S: RealScalar>(
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    evaluate(
        z,
        Kind::Bi,
        true,
        AiryOptions::default(),
        algorithm_component!(stringify!(airy_bi_complex_scaled)),
        None,
    )?
    .value()
}

#[inline]
pub(crate) fn airy_bi_complex_scaled_evaluation<S: RealScalar>(
    z: Complex<S>,
    options: AiryOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    evaluate(
        z,
        Kind::Bi,
        true,
        options,
        algorithm_component!(stringify!(airy_bi_complex_scaled)),
        context,
    )
}
