//! Principal complex Legendre integrals, sharing Carlson duplication and scaling.
use crate::EllipticOptions;
use crate::elliptic_family::elliptic_kernel::{self as kernel, Companion, duplication};
use crate::support::ELLIPTIC_ERRORS as ERRORS;
use crate::support::EllipticEstimate as Estimate;
use crate::support::Work;
use crate::support::Z;
use crate::support::widen_numeric as widen;
use crate::support::{C, ONE, ZERO};
use core::ops::Neg;
use scioxide_core::algorithm_component;
use scioxide_core::{
    Complex, DomainReason, NumericScalar, RealScalar, SciOxideError, SolveContext,
};
#[derive(Clone, Copy)]
pub(crate) enum Kind {
    K,
    E,
    F,
    Einc,
    Pi,
}
fn complement(z: C) -> C {
    C::new(1.0 - z.re, -z.im)
}
fn combine(a: Estimate, b: Estimate) -> Result<Estimate, SciOxideError> {
    let mut e = crate::elliptic_family::carlson::combine(
        [a.value, b.value, Z::ZERO],
        Z::from_value(scioxide_core::Complex::new(1.0, 0.0))?,
        a.relative.max(b.relative) + 8.0 * f64::EPSILON,
    )?;
    e.absolute += a.absolute + b.absolute;
    Ok(e)
}
fn weighted(mut e: Estimate, factor: Z) -> Result<Estimate, SciOxideError> {
    if e.absolute != 0.0 {
        e.absolute = scioxide_core::Scaled::from_value(e.absolute)?
            .mul(factor.magnitude()?)?
            .round()?;
    }
    e.value = e.value.mul(factor)?;
    e.relative += 8.0 * f64::EPSILON;
    Ok(e)
}
pub(crate) fn complete(kind: Kind, m: C, n: C, work: &mut Work) -> Result<Estimate, SciOxideError> {
    work.try_step()?;
    if m == ONE {
        return if matches!(kind, Kind::E) {
            Estimate::exact(ONE)
        } else {
            Err(ERRORS.domain(scioxide_core::argument!("m"), DomainReason::Pole))
        };
    }
    let delta = complement(m);
    if matches!(kind, Kind::K | Kind::E) && delta.norm() <= 1e-6 {
        let logarithm = C::new(4.0_f64.ln(), 0.0) - delta.ln() * 0.5;
        let (k, e) = kernel::endpoint_series(delta, logarithm)?;
        let value = if matches!(kind, Kind::K) { k } else { e };
        let mut estimate = Estimate::new(Z::from_value(value)?, 16.0 * f64::EPSILON);
        estimate.absolute = 2.0 * delta.norm().powi(3) * (1.0 + logarithm.norm());
        return Ok(estimate);
    }
    if matches!(kind, Kind::Pi) && n == ONE {
        return Err(ERRORS.domain(scioxide_core::argument!("n"), DomainReason::Pole));
    }
    if m == ZERO && (!matches!(kind, Kind::Pi) || n == ZERO) {
        return Ok(Estimate::new(
            Z::from_value(scioxide_core::Complex::new(
                core::f64::consts::FRAC_PI_2,
                0.0,
            ))?,
            f64::EPSILON,
        ));
    }
    if matches!(kind, Kind::Pi) && n == m {
        return weighted(
            complete(Kind::E, m, ZERO, work)?,
            Z::from_value(scioxide_core::Complex::new(1.0, 0.0))?
                .div(Z::from_value(complement(m))?)?,
        );
    }
    if matches!(kind, Kind::Pi) && m == ZERO {
        return Ok(Estimate::new(
            Z::from_value(scioxide_core::Complex::new(
                core::f64::consts::FRAC_PI_2,
                0.0,
            ))?
            .div(Z::from_value(complement(n))?.sqrt()?)?,
            8.0 * f64::EPSILON,
        ));
    }
    if matches!(kind, Kind::Pi) && n != ZERO && (m.re > 1.0 || n.re >= 1.0) {
        let rf = duplication(
            Z::ZERO,
            Z::from_value(complement(m))?,
            Z::from_value(scioxide_core::Complex::new(1.0, 0.0))?,
            Companion::None,
            work,
        )?
        .rf;
        let rj = crate::elliptic_family::carlson::complex_rj(
            [ZERO, complement(m), ONE, complement(n)],
            work,
        )?;
        return combine(
            rf,
            weighted(
                rj,
                Z::from_value(n)?
                    .mul(Z::from_value(scioxide_core::Complex::new(1.0 / 3.0, 0.0))?)?,
            )?,
        );
    }
    let y = Z::from_value(complement(m))?;
    let companion = match kind {
        Kind::E => Companion::Rd,
        Kind::Pi if n != ZERO => Companion::Rj(Z::from_value(complement(n))?),
        _ => Companion::None,
    };
    let pair = duplication(
        Z::ZERO,
        y,
        Z::from_value(scioxide_core::Complex::new(1.0, 0.0))?,
        companion,
        work,
    )?;
    match kind {
        Kind::E => combine(
            pair.rf,
            weighted(
                pair.companion,
                Z::from_value(m)?
                    .mul(Z::from_value(scioxide_core::Complex::new(-1.0 / 3.0, 0.0))?)?,
            )?,
        ),
        Kind::Pi if n != ZERO => combine(
            pair.rf,
            weighted(
                pair.companion,
                Z::from_value(n)?
                    .mul(Z::from_value(scioxide_core::Complex::new(1.0 / 3.0, 0.0))?)?,
            )?,
        ),
        _ => Ok(pair.rf),
    }
}
fn sin_cos(z: C) -> Result<(Z, Z), SciOxideError> {
    if z.im.abs() < 350.0 {
        return Ok((Z::from_value(z.sin())?, Z::from_value(z.cos())?));
    }
    let a = kernel::scaled_exp(C::new(-z.im, z.re))?;
    let b = kernel::scaled_exp(C::new(z.im, -z.re))?;
    Ok((
        a.sub(b)?.mul(Z::from_value(C::new(0.0, -0.5))?)?,
        a.add(b)?
            .mul(Z::from_value(scioxide_core::Complex::new(0.5, 0.0))?)?,
    ))
}
fn incomplete(kind: Kind, phi: C, m: C, work: &mut Work) -> Result<Estimate, SciOxideError> {
    work.try_step()?;
    if phi == ZERO || m == ZERO {
        return Estimate::exact(phi);
    }
    let periods = (phi.re / core::f64::consts::PI).round();
    let local = C::new((-periods).mul_add(core::f64::consts::PI, phi.re), phi.im);
    let negative = local.re.is_sign_negative();
    let local = if negative { -local } else { local };
    let (s, c) = sin_cos(local)?;
    if matches!(kind, Kind::Einc) && m == ONE {
        let local = if negative { s.neg() } else { s };
        return Ok(Estimate::new(
            local.add(
                Z::from_value(scioxide_core::Complex::new(periods, 0.0))?
                    .mul(Z::from_value(scioxide_core::Complex::new(2.0, 0.0))?)?,
            )?,
            16.0 * f64::EPSILON * (1.0 + phi.re.abs()),
        ));
    }
    let square = s.mul(s)?;
    let y = if local.im == 0.0 && m.im == 0.0 {
        Z::from_value(C::new(1.0 - m.re * local.re.sin().powi(2), -m.im))?
    } else {
        Z::from_value(scioxide_core::Complex::new(1.0, 0.0))?.sub(Z::from_value(m)?.mul(square)?)?
    };
    let pair = duplication(
        c.mul(c)?,
        y,
        Z::from_value(scioxide_core::Complex::new(1.0, 0.0))?,
        if matches!(kind, Kind::Einc) {
            Companion::Rd
        } else {
            Companion::None
        },
        work,
    )?;
    let mut e = weighted(pair.rf, s)?;
    if matches!(kind, Kind::Einc) {
        e = combine(
            e,
            weighted(
                pair.companion,
                square
                    .mul(s)?
                    .mul(Z::from_value(m)?)?
                    .mul(Z::from_value(scioxide_core::Complex::new(-1.0 / 3.0, 0.0))?)?,
            )?,
        )?;
    }
    if negative {
        e.value = e.value.neg();
    }
    if periods != 0.0 {
        let full = complete(
            if matches!(kind, Kind::Einc) {
                Kind::E
            } else {
                Kind::K
            },
            m,
            ZERO,
            work,
        )?;
        e = combine(
            e,
            weighted(
                full,
                Z::from_value(scioxide_core::Complex::new(periods, 0.0))?
                    .mul(Z::from_value(scioxide_core::Complex::new(2.0, 0.0))?)?,
            )?,
        )?;
    }
    e.relative += 16.0 * f64::EPSILON * (1.0 + phi.re.abs());
    Ok(e)
}
/// Validate Legendre arguments.
fn validate(kind: Kind, first: C, m: C) -> Result<(), SciOxideError> {
    crate::support::require_finite_param(first, "elliptic parameter first must be finite")?;
    crate::support::require_finite_param(m, "elliptic parameter m must be finite")?;
    if matches!(kind, Kind::K | Kind::Pi) && m == ONE {
        return Err(ERRORS.domain(scioxide_core::argument!("m"), DomainReason::Pole));
    }
    if matches!(kind, Kind::Pi) && first == ONE {
        return Err(ERRORS.domain(scioxide_core::argument!("n"), DomainReason::Pole));
    }
    if matches!(kind, Kind::Pi)
        && first.re > 1.0
        && m.re > 1.0
        && first.im == 0.0
        && m.im == 0.0
        && first.im.is_sign_negative() != m.im.is_sign_negative()
    {
        return Err(SciOxideError::domain_error(
            scioxide_core::argument!("n"),
            DomainReason::OutsideDomain,
            "incompatible signed cut limits",
        ));
    }
    Ok(())
}
pub(crate) fn evaluate<T: NumericScalar>(
    kind: Kind,
    first: T,
    m: T,
    options: EllipticOptions,
    context: Option<&mut SolveContext<'_, '_>>,
) -> Result<kernel::EllipticEvaluation<T>, SciOxideError> {
    let first = widen(first, "elliptic parameter first must be finite")?;
    let m = widen(m, "elliptic parameter m must be finite")?;
    validate(kind, first, m)?;
    let mut work = Work::elliptic(
        options.iteration_budget().get(),
        kernel::family_target::<T::Real>(),
        context,
    );
    let e = if matches!(kind, Kind::F | Kind::Einc) {
        incomplete(kind, first, m, &mut work)
    } else {
        complete(kind, m, first, &mut work)
    };
    work.check()?;
    kernel::report(
        e,
        &work,
        match kind {
            Kind::K => algorithm_component!("complex_K"),
            Kind::E => algorithm_component!("complex_E"),
            Kind::F => algorithm_component!("complex_F"),
            Kind::Einc => algorithm_component!("complex_E_inc"),
            Kind::Pi => algorithm_component!("complex_Pi"),
        },
    )
}
pub(crate) fn elliptic_k_complex<S: RealScalar>(
    m: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    evaluate(
        Kind::K,
        Complex::new(S::ZERO, S::ZERO),
        m,
        EllipticOptions::default(),
        None,
    )?
    .value()
}

#[inline]
pub(crate) fn elliptic_k_complex_evaluation<S: RealScalar>(
    m: Complex<S>,
    options: EllipticOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    evaluate(Kind::K, Complex::new(S::ZERO, S::ZERO), m, options, context)
}

pub(crate) fn elliptic_e_complex<S: RealScalar>(
    m: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    evaluate(
        Kind::E,
        Complex::new(S::ZERO, S::ZERO),
        m,
        EllipticOptions::default(),
        None,
    )?
    .value()
}

#[inline]
pub(crate) fn elliptic_e_complex_evaluation<S: RealScalar>(
    m: Complex<S>,
    options: EllipticOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    evaluate(Kind::E, Complex::new(S::ZERO, S::ZERO), m, options, context)
}

pub(crate) fn elliptic_f_complex<S: RealScalar>(
    phi: Complex<S>,
    m: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    evaluate(Kind::F, phi, m, EllipticOptions::default(), None)?.value()
}

#[inline]
pub(crate) fn elliptic_f_complex_evaluation<S: RealScalar>(
    phi: Complex<S>,
    m: Complex<S>,
    options: EllipticOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    evaluate(Kind::F, phi, m, options, context)
}

pub(crate) fn elliptic_e_inc_complex<S: RealScalar>(
    phi: Complex<S>,
    m: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    evaluate(Kind::Einc, phi, m, EllipticOptions::default(), None)?.value()
}

#[inline]
pub(crate) fn elliptic_e_inc_complex_evaluation<S: RealScalar>(
    phi: Complex<S>,
    m: Complex<S>,
    options: EllipticOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    evaluate(Kind::Einc, phi, m, options, context)
}

pub(crate) fn elliptic_pi_complex<S: RealScalar>(
    n: Complex<S>,
    m: Complex<S>,
) -> Result<Complex<S>, scioxide_core::SciOxideError>
where
    Complex<S>: NumericScalar<Real = S>,
{
    evaluate(Kind::Pi, n, m, EllipticOptions::default(), None)?.value()
}

#[inline]
pub(crate) fn elliptic_pi_complex_evaluation<S: RealScalar>(
    n: Complex<S>,
    m: Complex<S>,
    options: EllipticOptions,
    context: Option<&mut scioxide_core::SolveContext<'_, '_>>,
) -> Result<
    crate::result::Evaluation<Complex<S>, impl crate::result::Publish<Complex<S>> + use<S>>,
    scioxide_core::SciOxideError,
>
where
    Complex<S>: NumericScalar<Real = S>,
{
    let _ = &context;
    evaluate(Kind::Pi, n, m, options, context)
}
