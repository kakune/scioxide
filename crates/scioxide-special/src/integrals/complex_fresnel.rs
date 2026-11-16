//! Normalized entire Fresnel integrals; DLMF 7.5.8 and origin series.

use crate::IntegralOptions;
use crate::support::Jet;
use crate::support::JetEstimate as Estimate;
use crate::support::Work;
use crate::support::{C, real};
use scioxide_core::algorithm_component;
use scioxide_core::{Complex, NumericScalar, RealScalar, SciOxideError, SolveContext};

fn estimate(z: C, sine: bool, work: &mut Work) -> Result<Estimate, SciOxideError> {
    let phase = z * z * core::f64::consts::FRAC_PI_2;
    let derivative = if sine { phase.sin() } else { phase.cos() };
    if !phase.is_finite() || !derivative.is_finite() {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Overflow,
            "Fresnel phase or derivative exceeds the working range",
        ));
    }
    if z.norm() <= 1.5 {
        let mut term = if sine { z * phase / 3.0 } else { z };
        let mut value = term;
        let mut absolute = term.norm();
        let mut n = 0usize;
        let tail = loop {
            if !work.step() {
                return Ok(crate::support::interrupted());
            }
            let k = n as f64;
            let offset = if sine { 3.0 } else { 1.0 };
            let a = if sine { 2.0 } else { 1.0 };
            term = -term
                * phase
                * phase
                * ((4.0 * k + offset)
                    / ((2.0 * k + a) * (2.0 * k + a + 1.0) * (4.0 * k + offset + 4.0)));
            value += term;
            absolute += term.norm();
            n += 1;
            if term.norm() <= f64::EPSILON * absolute {
                break term.norm();
            }
        };
        return Ok(Estimate::from(crate::support::JetData {
            y: Jet {
                v: value,
                d: derivative,
            },
            scale: Jet::constant(real(0.0)),
            error: tail + 8.0 * f64::EPSILON * absolute,
            derivative_error: Some(16.0 * f64::EPSILON * (1.0 + phase.norm()) * derivative.norm()),
            method: algorithm_component!("Fresnel_origin_series"),
        }));
    }
    let factor = core::f64::consts::PI.sqrt() * 0.5;
    let a = crate::error_probability::complex_error::base(
        z * C::new(factor, -factor),
        crate::error_probability::complex_error::Kind::Erf,
        work,
    )?;
    let b = crate::error_probability::complex_error::base(
        z * C::new(factor, factor),
        crate::error_probability::complex_error::Kind::Erf,
        work,
    )?;
    let scale = a.scale.v.re.max(b.scale.v.re).max(0.0);
    let project = |e: Estimate, f: C| {
        let multiplier = (e.scale.v - real(scale)).exp() * f;
        (e.y.v * multiplier, e.error * multiplier.norm())
    };
    let (a, ae) = project(a, C::new(0.5, 0.5));
    let (b, be) = project(b, C::new(0.5, -0.5));
    let value = if sine {
        (a - b) * C::new(0.0, -0.5)
    } else {
        (a + b) * 0.5
    };
    Ok(Estimate::from(crate::support::JetData {
        y: Jet {
            v: value,
            d: derivative * (-scale).exp(),
        },
        scale: Jet::constant(real(scale)),
        error: (ae + be) * 0.5 + 16.0 * f64::EPSILON * (a.norm() + b.norm()),
        derivative_error: Some(
            16.0 * f64::EPSILON * (1.0 + phase.norm()) * derivative.norm() * (-scale).exp(),
        ),
        method: algorithm_component!("Fresnel_erf_connection"),
    }))
}
fn evaluate<S: RealScalar>(
    z: Complex<S>,
    sine: bool,
    options: IntegralOptions,
    parent: Option<&mut SolveContext<'_, '_>>,
) -> Result<crate::complex_support::ScaledEvaluation<Complex<S>>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    if !z.is_finite() {
        return Err(SciOxideError::non_finite(
            scioxide_core::NonFiniteOrigin::Input,
            "finite input required",
        ));
    }
    let z = C::new(z.re.try_to_f64_exact()?, z.im.try_to_f64_exact()?);
    let mut work = Work::with_context(options.iteration_budget().get(), parent);
    let result = estimate(z, sine, &mut work);
    work.check()?;
    crate::complex_support::report_with_status(
        result,
        true,
        algorithm_component!("complex-Fresnel"),
        work.used,
        !work.complete,
    )
}
pub(crate) fn fresnel_c_complex<S: RealScalar>(
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    evaluate(z, false, IntegralOptions::default(), None)?.value()
}

#[inline]
pub(crate) fn fresnel_c_complex_evaluation<S: RealScalar>(
    z: Complex<S>,
    options: IntegralOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    evaluate(z, false, options, context)
}

pub(crate) fn fresnel_s_complex<S: RealScalar>(
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    evaluate(z, true, IntegralOptions::default(), None)?.value()
}

#[inline]
pub(crate) fn fresnel_s_complex_evaluation<S: RealScalar>(
    z: Complex<S>,
    options: IntegralOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    evaluate(z, true, options, context)
}
