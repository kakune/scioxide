//! Family-local, scaled quadrature for the defining Lamé integrals.
use super::*;

use crate::elliptic_family::elliptic_quadrature::integrate;

pub(super) fn second<R: RealScalar>(
    plan: &EllipsoidalHarmonicPlan<R>,
    s: C,
    work: &mut Work<'_, '_, '_>,
) -> Result<LameEstimate, SciOxideError> {
    let e = plan.evaluate(s, false, work)?;
    let z = Z::from_value(s)?;
    let k_over_s = Z::from_value(scioxide_core::Complex::new(plan.k, 0.0))?.div(z)?;
    let integral = integrate(
        |t, work| {
            let coordinate = z.div(Z::from_value(scioxide_core::Complex::new(t, 0.0))?)?;
            let at = plan.evaluate_scaled(coordinate, false, work)?.value;
            let u2 = k_over_s.mul(Z::from_value(scioxide_core::Complex::new(t, 0.0))?)?;
            let u2 = u2.mul(u2)?;
            let one = Z::from_value(scioxide_core::Complex::new(1.0, 0.0))?;
            let denominator = one.sub(u2)?.sqrt()?.mul(
                one.sub(u2.mul(Z::from_value(scioxide_core::Complex::new(plan.ratio, 0.0))?)?)?
                    .sqrt()?,
            )?;
            Ok(Estimate::new(
                one.div(at.value.mul(at.value)?)?.div(denominator)?,
                2.0 * at.relative + 32.0 * f64::EPSILON,
            ))
        },
        1.0,
        work,
    )?;
    let integral_value = integral.value.div(z)?;
    let multiplier = Z::from_value(scioxide_core::Complex::new(
        (2 * plan.degree + 1) as f64,
        0.0,
    ))?;
    let value = e.value.value.mul(integral_value)?.mul(multiplier)?;
    let h = Z::from_value(scioxide_core::Complex::new(plan.k * plan.ratio.sqrt(), 0.0))?;
    let k = Z::from_value(scioxide_core::Complex::new(plan.k, 0.0))?;
    let radical = z
        .sub(h)?
        .sqrt()?
        .mul(z.add(h)?.sqrt()?)?
        .mul(z.sub(k)?.sqrt()?)?
        .mul(z.add(k)?.sqrt()?)?;
    let derivative = if let Some(d) = e.derivative {
        let first = d.value.mul(integral_value)?;
        let second = Z::from_value(scioxide_core::Complex::new(1.0, 0.0))?
            .div(e.value.value.mul(radical)?)?;
        let value = first.sub(second)?.mul(multiplier)?;
        let condition = first
            .magnitude()?
            .add(second.magnitude()?)?
            .mul(multiplier.magnitude()?)?
            .div(value.magnitude()?)?
            .round()?;
        Some(Estimate::new(
            value,
            (d.relative + integral.relative + e.value.relative) * condition,
        ))
    } else {
        None
    };
    Ok(LameEstimate {
        value: Estimate::new(value, e.value.relative + integral.relative),
        derivative,
    })
}

pub(super) fn normalization<R: RealScalar>(
    plan: &EllipsoidalHarmonicPlan<R>,
    work: &mut Work<'_, '_, '_>,
) -> Result<Estimate, SciOxideError> {
    // ν=h sn(u|m), μ=h/dn(v|1-m); both angular measures become du/k.
    // γ=8(A0 Bgap + Agap B0), with nonnegative gaps measured from h².
    let m = plan.ratio;
    let mc =
        (plan.k2.try_to_f64_exact()? - plan.h2.try_to_f64_exact()?) / plan.k2.try_to_f64_exact()?;
    let complete = |parameter: f64, work: &mut Work<'_, '_, '_>| -> Result<f64, SciOxideError> {
        Ok(crate::elliptic_family::carlson::evaluate(
            crate::elliptic_family::carlson::Family::Rf,
            [real(0.0), real(parameter), real(1.0), real(0.0)],
            true,
            work,
        )?
        .value
        .round()?
        .re)
    };
    let a_end = complete(mc, work)?;
    let b_end = complete(m, work)?;
    let mut moments = [Estimate::exact(real(0.0))?; 4];
    for (index, moment) in moments.iter_mut().enumerate() {
        let first = index < 2;
        let gap = index % 2 == 1;
        *moment = integrate(
            |u, work| {
                let component = if first {
                    crate::elliptic_family::jacobi::Component::Sn
                } else {
                    crate::elliptic_family::jacobi::Component::Dn
                };
                let v = crate::elliptic_family::jacobi::evaluate(
                    component,
                    real(u),
                    if first { m } else { mc },
                    true,
                    work,
                )?;
                let v = v.value.round()?.re;
                let x = if first { m.sqrt() * v } else { m.sqrt() / v };
                let e = if !first && let Some(expansion) = &plan.complementary {
                    let cn = crate::elliptic_family::jacobi::evaluate(
                        crate::elliptic_family::jacobi::Component::Cn,
                        real(u),
                        mc,
                        true,
                        work,
                    )?
                    .value
                    .round()?
                    .re;
                    let t = plan.k * mc.sqrt() * cn / v;
                    plan.evaluate_expansion(
                        Z::from_value(scioxide_core::Complex::new(t, 0.0))?,
                        true,
                        Some(expansion),
                        work,
                    )?
                    .value
                } else {
                    plan.evaluate(real(plan.k * x), true, work)?.value
                };
                let factor = if gap {
                    if first {
                        let cn = crate::elliptic_family::jacobi::evaluate(
                            crate::elliptic_family::jacobi::Component::Cn,
                            real(u),
                            m,
                            true,
                            work,
                        )?
                        .value
                        .round()?
                        .re;
                        m * cn * cn
                    } else {
                        let sn = crate::elliptic_family::jacobi::evaluate(
                            crate::elliptic_family::jacobi::Component::Sn,
                            real(u),
                            mc,
                            true,
                            work,
                        )?
                        .value
                        .round()?
                        .re;
                        m * mc * sn * sn / (v * v)
                    }
                } else {
                    1.0
                };
                Ok(Estimate::new(
                    e.value
                        .mul(e.value)?
                        .mul(Z::from_value(scioxide_core::Complex::new(factor, 0.0))?)?,
                    2.0 * e.relative + 128.0 * f64::EPSILON,
                ))
            },
            if first { a_end } else { b_end },
            work,
        )?;
    }
    let [a, agap, b, bgap] = moments;
    // The k² in each dimensional gap cancels the two measure factors 1/k.
    let value = a
        .value
        .mul(bgap.value)?
        .add(agap.value.mul(b.value)?)?
        .mul(Z::from_value(scioxide_core::Complex::new(8.0, 0.0))?)?;
    Ok(Estimate::new(
        value,
        a.relative.max(agap.relative) + b.relative.max(bgap.relative) + 16.0 * f64::EPSILON,
    ))
}
