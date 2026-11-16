//! Polylogarithms, dilogarithm, Spence and Lerch public entry points.
use crate::support::TRANSCENDENTAL_ERRORS as ERRORS;
use crate::support::TranscendentalEstimate as Estimate;
use crate::support::Work;
use crate::support::widen_numeric as widen;
use crate::support::{C, ONE, ZERO, real};
use crate::zeta_polylog::transcendental::{
    RealEvaluation, RealKernelEvaluation, TranscendentalEvaluation, project_real, real_report,
    report,
};
use crate::{LerchOptions, PolylogOptions};
use core::f64::consts::PI;
use scioxide_core::algorithm_component;
use scioxide_core::{
    Complex, DomainReason, NumericScalar, RealScalar, SciOxideError, SolveContext,
};

#[path = "polylog_real.rs"]
mod real_kernel;

fn one_minus(z: C) -> C {
    C::new(1.0 - z.re, -z.im)
}
fn log_one_minus(z: C) -> C {
    // Re(log(1-z)) needs log1p when subtraction would round away tiny z.
    if z.norm() < 0.25 {
        C::new(
            0.5 * ((-2.0 * z.re + z.re * z.re) + z.im * z.im).ln_1p(),
            (-z.im).atan2(1.0 - z.re),
        )
    } else {
        one_minus(z).ln()
    }
}
fn dilog_kernel(z: C, work: &mut Work<'_, '_, '_>) -> Result<Estimate, SciOxideError> {
    if z == ZERO {
        let mut e = Estimate::new(z, 0.0, algorithm_component!("dilog_zero"));
        e.derivative = Some(ONE);
        return Ok(e);
    }
    if z == ONE {
        return Ok(Estimate::new(
            real(PI * PI / 6.0),
            4.0 * f64::EPSILON,
            algorithm_component!("dilog_one"),
        ));
    }
    work.try_step()?;
    let mut e = if z.norm() > 1.0 {
        let mut inverse = ONE / z;
        if z.im == 0.0 {
            inverse.im = -z.im;
        }
        let r = dilog_kernel(inverse, work)?;
        let correction = real(PI * PI / 6.0) + (-z).ln().powu(2) * real(0.5);
        Estimate::new(
            -r.value - correction,
            r.error + 16.0 * f64::EPSILON * correction.norm(),
            algorithm_component!("dilog_inversion"),
        )
    } else if z.re > 0.5 {
        let r = dilog_kernel(one_minus(z), work)?;
        let correction = real(PI * PI / 6.0) - z.ln() * log_one_minus(z);
        Estimate::new(
            correction - r.value,
            r.error + 16.0 * f64::EPSILON * correction.norm(),
            algorithm_component!("dilog_reflection"),
        )
    } else {
        // Integral of w/(exp(w)-1), w=-log(1-z). The preceding transforms
        // keep this inside the Bernoulli series' disk |w|<2*pi.
        let w = -log_one_minus(z);
        let mut value = w - w * w * real(0.25);
        let mut power = w * w * w;
        let mut magnitude = w.norm() + (w * w).norm() * 0.25;
        let mut tail = 0.0;
        for (j, &b) in crate::support::gamma_series::BERNOULLI_OVER_FACTORIAL
            .iter()
            .enumerate()
        {
            work.try_step()?;
            let term = power * real(b.hi / ((2 * j + 3) as f64));
            value += term;
            magnitude += term.norm();
            tail = term.norm();
            power *= w * w;
        }
        Estimate::new(
            value,
            tail * (w.norm() / (2.0 * PI)).powi(2) + 16.0 * f64::EPSILON * magnitude,
            algorithm_component!("dilog_bernoulli"),
        )
    };
    e.derivative = Some(-log_one_minus(z) / z);
    Ok(e)
}
fn polylog_kernel(s: C, z: C, work: &mut Work<'_, '_, '_>) -> Result<Estimate, SciOxideError> {
    if s == real(2.0) {
        return dilog_kernel(z, work);
    }
    if z == ZERO {
        return Ok(Estimate::new(z, 0.0, algorithm_component!("polylog_zero")));
    }
    if z == ONE {
        return crate::zeta_polylog::lerch::kernel(z, s, ONE, work);
    }
    if s == ONE {
        let value = -log_one_minus(z);
        return Ok(Estimate::new(
            value,
            8.0 * f64::EPSILON * value.norm(),
            algorithm_component!("polylog_logarithm"),
        ));
    }
    if s.im == 0.0 && s.re <= 0.0 && s.re == s.re.floor() {
        return super::negative::integer(-s.re, z, work);
    }
    if s.im == 0.0 && s.re < 0.0 && z.norm() >= 0.95 {
        return super::negative::continued(s.re, z, work);
    }
    let mut e = crate::zeta_polylog::lerch::kernel(z, s, ONE, work)?;
    e.value *= z;
    e.error = e.error * z.norm() + 4.0 * f64::EPSILON * e.value.norm();
    Ok(e)
}

fn unary_evaluate<S: RealScalar>(
    z: Complex<S>,
    options: PolylogOptions,
    spence: bool,
) -> Result<TranscendentalEvaluation<S>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let z = widen(z, "polylogarithm parameter z must be finite")?;
    let mut work = Work::new(options.iteration_budget().get());
    let e = dilog_kernel(if spence { one_minus(z) } else { z }, &mut work).map(|mut e| {
        if spence {
            e.derivative = e.derivative.map(|d| -d);
        }
        e
    });
    report(e, &work)
}
pub(crate) fn dilog_complex<S: RealScalar>(
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options = <PolylogOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    unary_evaluate(z, options, false)?.value()
}

#[inline]
pub(crate) fn dilog_complex_evaluation<S: RealScalar>(
    z: Complex<S>,
    options: PolylogOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    unary_evaluate(z, options, false)
}

pub(crate) fn spence_complex<S: RealScalar>(
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options = <PolylogOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    unary_evaluate(z, options, true)?.value()
}

#[inline]
pub(crate) fn spence_complex_evaluation<S: RealScalar>(
    z: Complex<S>,
    options: PolylogOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    unary_evaluate(z, options, true)
}

pub(crate) fn dilog<S: RealScalar>(x: S) -> Result<S, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options = <PolylogOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    dilog_evaluate(x, options)?.value()
}

#[doc = "Evaluate before projecting the value or forming the detailed result."]
pub(crate) fn dilog_evaluate<S: RealScalar>(
    x: S,
    options: PolylogOptions,
) -> Result<RealKernelEvaluation<S>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    crate::support::require_finite_param(x, "polylogarithm parameter x must be finite")?;

    let mut work = Work::new(options.iteration_budget().get());
    let estimate = real_kernel::kernel(x.try_to_f64_exact()?, false, &mut work);
    real_report(estimate, &work)
}
pub(crate) fn spence<S: RealScalar>(x: S) -> Result<S, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options = <PolylogOptions>::default();
    let context: Option<&mut scioxide_core::SolveContext<'_, '_>> = None;
    let _ = &context;
    spence_evaluate(x, options)?.value()
}

#[doc = "Evaluate before projecting the value or forming the detailed result."]
pub(crate) fn spence_evaluate<S: RealScalar>(
    x: S,
    options: PolylogOptions,
) -> Result<RealKernelEvaluation<S>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    crate::support::require_finite_param(x, "polylogarithm parameter x must be finite")?;
    if x < S::ZERO {
        return Err(ERRORS.domain(scioxide_core::argument!("x"), DomainReason::OutsideDomain));
    }
    let mut work = Work::new(options.iteration_budget().get());
    let estimate = real_kernel::kernel(x.try_to_f64_exact()?, true, &mut work);
    real_report(estimate, &work)
}
pub(crate) fn polylog_complex<S: RealScalar>(
    s: Complex<S>,
    z: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options: PolylogOptions = PolylogOptions::default();

    poly_evaluate(s, z, options, None)?.value()
}

pub(crate) fn poly_evaluate<S: RealScalar>(
    s: Complex<S>,
    z: Complex<S>,
    options: PolylogOptions,
    context: Option<&mut SolveContext<'_, '_>>,
) -> Result<TranscendentalEvaluation<S>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let s = widen(s, "polylogarithm parameter s must be finite")?;
    let z = widen(z, "polylogarithm parameter z must be finite")?;
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    let e = polylog_kernel(s, z, &mut work);
    report(e, &work)
}
pub(crate) fn polylog<S: RealScalar>(s: S, z: S) -> Result<S, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    polylog_evaluate(s, z, PolylogOptions::default(), None)?.value()
}

/// Evaluate before projecting the value or forming the detailed result.
pub(crate) fn polylog_evaluate<S: RealScalar>(
    s: S,
    z: S,
    options: PolylogOptions,
    context: Option<&mut SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<
        S,
        crate::result::Either<
            crate::zeta_polylog::transcendental::RealEstimateForm,
            crate::zeta_polylog::transcendental::RealPart,
        >,
    >,
    SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    real_domain(s, z, None)?;
    if s == S::TWO {
        let mut work = Work::with_context(options.iteration_budget().get(), context);
        let estimate = real_kernel::kernel(z.try_to_f64_exact()?, false, &mut work);
        return Ok(real_report(estimate, &work)?.map_form(crate::result::Either::Left));
    }
    Ok(project_real(
        poly_evaluate(
            Complex::new(s, S::ZERO),
            Complex::new(z, S::ZERO),
            options,
            context,
        )?,
        false,
    )?
    .map_form(crate::result::Either::Right))
}
pub(crate) fn lerch_phi_complex<S: RealScalar>(
    z: Complex<S>,
    s: Complex<S>,
    a: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let options: LerchOptions = LerchOptions::default();

    lerch_evaluate(z, s, a, options, None)?.value()
}

pub(crate) fn lerch_evaluate<S: RealScalar>(
    z: Complex<S>,
    s: Complex<S>,
    a: Complex<S>,
    options: LerchOptions,
    context: Option<&mut SolveContext<'_, '_>>,
) -> Result<TranscendentalEvaluation<S>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let z = widen(z, "polylogarithm parameter z must be finite")?;
    let s = widen(s, "polylogarithm parameter s must be finite")?;
    let a = widen(a, "polylogarithm parameter a must be finite")?;
    let mut work = Work::with_context(options.iteration_budget().get(), context);
    let e = crate::zeta_polylog::lerch::kernel(z, s, a, &mut work);
    report(e, &work)
}
pub(crate) fn lerch_phi<S: RealScalar>(z: S, s: S, a: S) -> Result<S, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    lerch_phi_evaluate(z, s, a, LerchOptions::default(), None)?.value()
}

/// Evaluate before projecting the value or forming the detailed result.
pub(crate) fn lerch_phi_evaluate<S: RealScalar>(
    z: S,
    s: S,
    a: S,
    options: LerchOptions,
    context: Option<&mut SolveContext<'_, '_>>,
) -> Result<RealEvaluation<S>, SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    real_domain(s, z, Some(a))?;
    project_real(
        lerch_evaluate(
            Complex::new(z, S::ZERO),
            Complex::new(s, S::ZERO),
            Complex::new(a, S::ZERO),
            options,
            context,
        )?,
        false,
    )
}

/// Real-branch validation.
fn real_domain<S: RealScalar>(s: S, z: S, a: Option<S>) -> Result<(), SciOxideError> {
    crate::support::require_finite_param(s, "polylogarithm parameter s must be finite")?;
    crate::support::require_finite_param(z, "polylogarithm parameter z must be finite")?;
    if let Some(a) = a {
        crate::support::require_finite_param(a, "polylogarithm parameter a must be finite")?;
    }
    let integer = s == s.floor();
    if z > S::ONE && !(integer && s <= S::ZERO || a.is_none() && s == S::TWO) {
        return Err(ERRORS.domain(scioxide_core::argument!("z"), DomainReason::OutsideDomain));
    }
    if a.is_some_and(|a| a <= S::ZERO && !integer) {
        return Err(ERRORS.domain(scioxide_core::argument!("a"), DomainReason::OutsideDomain));
    }
    Ok(())
}
