//! Jacobi theta quotients with modular reduction; DLMF 20.7 and 22.2.
use crate::elliptic_family::complex_elliptic::{Kind, complete};
use crate::elliptic_family::elliptic_kernel::{Component, principal_amplitude};
use crate::support::ELLIPTIC_ERRORS as ERRORS;
use crate::support::EllipticEstimate as Estimate;
use crate::support::Work;
use crate::support::Z;
use crate::support::{C, ONE, ZERO};
#[cfg(all(not(feature = "std"), not(test)))]
#[allow(unused_imports)]
// Required in standalone no_std; a std-linked consumer has inherent f64 methods.
use scioxide_core::RealScalar;
use scioxide_core::{DomainReason, SciOxideError};
const PI: f64 = core::f64::consts::PI;
fn unit(angle: f64) -> C {
    let (s, c) = angle.sin_cos();
    C::new(c, s)
}
struct Theta {
    values: [C; 4],
    errors: [f64; 4],
    dv: [f64; 4],
    dtau: [f64; 4],
}
/// Evaluate q^(a²) sin/cos(2*a*v) without forming opposing large and small
/// exponentials. expm1 preserves the odd part at tiny imaginary arguments.
fn term(a: f64, v: C, tau: C, quarter: bool) -> (C, C) {
    let power = a * a - if quarter { 0.25 } else { 0.0 };
    let angle = 2.0 * a * v.re;
    let height = 2.0 * a * v.im;
    let damp = (-2.0 * height.abs()).exp();
    let odd = -(-2.0 * height.abs()).exp_m1() * height.signum();
    let factor = unit(PI * tau.re * power) * (((-PI * tau.im * power + height.abs()).exp()) * 0.5);
    (
        factor * C::new(angle.sin() * (1.0 + damp), angle.cos() * odd),
        factor * C::new(angle.cos() * (1.0 + damp), -angle.sin() * odd),
    )
}
fn theta(v: C, tau: C, work: &mut Work) -> Result<Theta, SciOxideError> {
    let mut values = [ZERO, ZERO, ONE, ONE];
    let mut magnitudes = [0.0, 0.0, 1.0, 1.0];
    let mut dv = [0.0; 4];
    let mut dtau = [0.0; 4];
    for n in 0_usize.. {
        work.try_step()?;
        let (s, c) = term(n as f64 + 0.5, v, tau, true);
        let (es, even) = term(n as f64 + 1.0, v, tau, false);
        let terms = [
            s * if n.is_multiple_of(2) { 2.0 } else { -2.0 },
            c * 2.0,
            even * 2.0,
            even * if n.is_multiple_of(2) { -2.0 } else { 2.0 },
        ];
        let a = n as f64 + 0.5;
        let b = n as f64 + 1.0;
        let vd = [
            4.0 * a * c.norm(),
            4.0 * a * s.norm(),
            4.0 * b * es.norm(),
            4.0 * b * es.norm(),
        ];
        for i in 0..4 {
            values[i] += terms[i];
            magnitudes[i] += terms[i].norm();
            dv[i] += vd[i];
            dtau[i] += PI * if i < 2 { a * a - 0.25 } else { b * b } * terms[i].norm();
        }
        if values.iter().any(|z| !z.is_finite()) {
            return Err(ERRORS.unstable());
        }
        if n >= 2 && (0..4).all(|i| terms[i].norm() <= f64::EPSILON * magnitudes[i]) {
            return Ok(Theta {
                values,
                errors: core::array::from_fn(|i| {
                    4.0 * terms[i].norm() + 32.0 * f64::EPSILON * (n + 1) as f64 * magnitudes[i]
                }),
                dv,
                dtau,
            });
        }
    }
    Err(ERRORS.unstable())
}
/// Local analytic ODE series for sn(u*t)/u, cn(u*t), dn(u*t). Scaling the
/// coefficients by u² keeps tiny orthogonal components without theta ratios.
fn local(component: Component, u: C, m: C, work: &mut Work) -> Result<Estimate, SciOxideError> {
    let square = u * u;
    let parameter = m * square;
    let mut a = [ZERO; 32];
    let mut b = a;
    let mut d = a;
    a[0] = ONE;
    b[0] = ONE;
    d[0] = ONE;
    let mut values = [ONE, ONE, ONE];
    let mut magnitude = [1.0, 1.0, 1.0];
    for n in 1..32 {
        // A(t)=sn(u*t)/u has A(0)=0 and A'(0)=1. Store A(t)/t,
        // so the convolution for its nth coefficient uses B*D at index n.
        let mut bd = ZERO;
        let mut ad = ZERO;
        let mut ab = ZERO;
        for k in 0..n {
            work.try_step()?;
            if n >= 2 && k <= n - 2 {
                ad += a[k] * d[n - 2 - k];
                ab += a[k] * b[n - 2 - k];
            }
        }
        b[n] = -square * ad / (n as f64);
        d[n] = -parameter * ab / (n as f64);
        for k in 0..=n {
            work.try_step()?;
            bd += b[k] * d[n - k];
        }
        a[n] = bd / (n as f64 + 1.0);
        let terms = [a[n], b[n], d[n]];
        for i in 0..3 {
            values[i] += terms[i];
            magnitude[i] += terms[i].norm();
        }
        if n >= 6
            && n.is_multiple_of(2)
            && (0..3).all(|i| terms[i].norm() <= f64::EPSILON * magnitude[i])
        {
            let sn = u * values[0];
            let cn = values[1];
            let dn = values[2];
            let value = match component {
                Component::Sn => sn,
                Component::Cn => cn,
                Component::Dn => dn,
                Component::Amplitude => principal_amplitude(sn, cn)?,
            };
            return Ok(Estimate::new(
                Z::from_value(value)?,
                64.0 * f64::EPSILON * (n + 1) as f64,
            ));
        }
    }
    Err(ERRORS.unstable())
}
fn amplitude(sn: Estimate, cn: Estimate) -> Result<Estimate, SciOxideError> {
    let s = sn.value.round()?;
    let c = cn.value.round()?;
    let value = principal_amplitude(s, c)?;
    let denominator = (c + C::new(0.0, 1.0) * s)
        .norm()
        .max((c - C::new(0.0, 1.0) * s).norm());
    let mut e = Estimate::new(Z::from_value(value)?, 16.0 * f64::EPSILON);
    e.absolute =
        (sn.relative * s.norm() + cn.relative * c.norm() + sn.absolute + cn.absolute) / denominator;
    Ok(e)
}
pub(crate) fn evaluate(
    component: Component,
    u: C,
    m: C,
    work: &mut Work,
) -> Result<Estimate, SciOxideError> {
    work.check()?;
    if u == ZERO {
        work.try_step()?;
        return Estimate::exact(match component {
            Component::Sn | Component::Amplitude => u,
            _ => ONE,
        });
    }
    if m.norm() > 1.0 {
        work.try_step()?;
        let root = Z::from_value(m)?.sqrt()?;
        let point = Z::from_value(u)?.mul(root)?.round()?;
        let reciprocal = Z::from_value(scioxide_core::Complex::new(1.0, 0.0))?
            .div(Z::from_value(m)?)?
            .round()?;
        let mut e = match component {
            Component::Sn => {
                let mut e = evaluate(Component::Sn, point, reciprocal, work)?;
                e.value = e.value.div(root)?;
                e
            }
            Component::Cn => evaluate(Component::Dn, point, reciprocal, work)?,
            Component::Dn => evaluate(Component::Cn, point, reciprocal, work)?,
            Component::Amplitude => {
                let mut sn = evaluate(Component::Sn, point, reciprocal, work)?;
                sn.value = sn.value.div(root)?;
                let cn = evaluate(Component::Dn, point, reciprocal, work)?;
                amplitude(sn, cn)?
            }
        };
        e.relative += 16.0 * f64::EPSILON;
        return Ok(e);
    }
    if u.norm() * (1.0 + m.norm().sqrt()) <= 0.25 {
        return local(component, u, m, work);
    }
    let k = complete(Kind::K, m, ZERO, work)?;
    let kp = crate::elliptic_family::elliptic_kernel::duplication(
        Z::ZERO,
        Z::from_value(m)?,
        Z::from_value(scioxide_core::Complex::new(1.0, 0.0))?,
        crate::elliptic_family::elliptic_kernel::Companion::None,
        work,
    )?
    .rf;
    let mut tau = kp
        .value
        .div(k.value)?
        .mul(Z::from_value(C::new(0.0, 1.0))?)?
        .round()?;
    let mut v = Z::from_value(u)?
        .div(k.value)?
        .mul(Z::from_value(scioxide_core::Complex::new(PI / 2.0, 0.0))?)?
        .round()?;
    let mut tau_error = tau.norm() * (k.relative + kp.relative + 16.0 * f64::EPSILON);
    let mut v_error = v.norm() * (k.relative + 16.0 * f64::EPSILON);
    let mut index = [0, 1, 2, 3];
    let mut phase = [ONE; 4];
    loop {
        work.try_step()?;
        if !tau.is_finite() || tau.im <= 0.0 || !v.is_finite() {
            return Err(ERRORS.unstable());
        }
        let turns = tau.re.round();
        let (s, c) = scioxide_core::sin_cos_pi(turns * 0.25)?;
        let factor = C::new(c, s);
        for i in 0..4 {
            if index[i] < 2 {
                phase[i] *= factor;
            } else if turns % 2.0 != 0.0 {
                index[i] = 5 - index[i];
            }
        }
        tau.re -= turns;
        tau_error += 8.0 * f64::EPSILON * (1.0 + turns.abs());
        if tau.norm() >= 1.0 {
            break;
        }
        let next_v = v / tau;
        let next_tau = -ONE / tau;
        v_error = v_error / tau.norm()
            + next_v.norm() * tau_error / tau.norm()
            + 8.0 * f64::EPSILON * next_v.norm();
        tau_error = tau_error / tau.norm_sqr() + 8.0 * f64::EPSILON * next_tau.norm();
        v = next_v;
        tau = next_tau;
        for i in 0..4 {
            match index[i] {
                0 => phase[i] *= C::new(0.0, 1.0),
                1 => index[i] = 3,
                3 => index[i] = 1,
                _ => {}
            }
        }
    }
    let vertical = (v.im / (PI * tau.im)).round();
    let horizontal = (v.re / PI - vertical * tau.re).round();
    let reduced = C::new(
        (-horizontal).mul_add(PI, (-vertical).mul_add(PI * tau.re, v.re)),
        (-vertical).mul_add(PI * tau.im, v.im),
    );
    v_error += PI * vertical.abs() * tau_error
        + 16.0 * f64::EPSILON * (v.norm() + PI * (horizontal.abs() + vertical.abs() * tau.norm()));
    let signs = [
        if (horizontal + vertical) % 2.0 == 0.0 {
            1.0
        } else {
            -1.0
        },
        if horizontal % 2.0 == 0.0 { 1.0 } else { -1.0 },
        1.0,
        if vertical % 2.0 == 0.0 { 1.0 } else { -1.0 },
    ];
    let mut at = theta(reduced, tau, work)?;
    let mut origin = theta(ZERO, tau, work)?;
    for i in 0..4 {
        at.errors[i] += at.dv[i] * v_error + at.dtau[i] * tau_error;
        origin.errors[i] += origin.dtau[i] * tau_error;
    }
    let values =
        core::array::from_fn::<_, 4, _>(|i| at.values[index[i]] * phase[i] * signs[index[i]]);
    let constants = core::array::from_fn::<_, 4, _>(|i| origin.values[index[i]] * phase[i]);
    if values[3] == ZERO {
        return Err(ERRORS.domain(scioxide_core::argument!("u"), DomainReason::Pole));
    }
    let quotient = |top: usize, bottom: usize, which: usize| -> Result<Estimate, SciOxideError> {
        if constants[bottom] == ZERO {
            return Err(ERRORS.unstable());
        }
        let mut value = Z::from_value(constants[top])?
            .div(Z::from_value(constants[bottom])?)?
            .mul(Z::from_value(values[which])?)?
            .div(Z::from_value(values[3])?)?;
        let quarter = |i: usize| if index[i] < 2 { 1.0 } else { 0.0 };
        let power = quarter(top) + quarter(which) - quarter(bottom) - quarter(3);
        if power != 0.0 {
            value = value.mul(crate::elliptic_family::elliptic_kernel::scaled_exp(
                C::new(-tau.im, tau.re) * (power * PI / 4.0),
            )?)?;
        }
        let relative = power.abs() * PI * tau_error / 4.0
            + origin.errors[index[top]] / constants[top].norm()
            + origin.errors[index[bottom]] / constants[bottom].norm()
            + at.errors[index[which]] / values[which].norm()
            + at.errors[index[3]] / values[3].norm();
        Ok(Estimate::new(value, relative + 32.0 * f64::EPSILON))
    };
    match component {
        Component::Sn => quotient(2, 1, 0),
        Component::Cn => quotient(3, 1, 1),
        Component::Dn => quotient(3, 2, 2),
        Component::Amplitude => {
            let sn = quotient(2, 1, 0)?;
            let cn = quotient(3, 1, 1)?;
            amplitude(sn, cn)
        }
    }
}
