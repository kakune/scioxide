//! Root-factored J0/J1/Y0/Y1 and shared Hankel phase. All arguments are positive.
//! Compensated fallbacks keep the relative envelope next to oscillatory zeros.
//! No recurrence in order is needed.
// The rational tables follow Boost.Math 1.90.
use super::bessel_low_tables::{j0, j1, y0, y1};
use crate::support::{
    Work,
    polynomial::{PolynomialPolicy, polynomial},
};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use scioxide_core::RealScalar;
use scioxide_core::SciOxideError;

#[inline]
fn rational(x: f64, p: &[f64], q: &[f64], w: &mut Work) -> Result<f64, SciOxideError> {
    w.steps(p.len() + q.len())?;
    Ok(polynomial(x, p, PolynomialPolicy::Fused) / polynomial(x, q, PolynomialPolicy::Fused))
}

#[inline]
fn small_j(x: f64, one: bool, w: &mut Work) -> Result<f64, SciOxideError> {
    if x < f64::EPSILON.sqrt() {
        w.try_step()?;
        return Ok(if one { x * 0.5 } else { 1.0 });
    }
    let upper = x > 4.0;
    let (p, q, roots): (&[f64], &[f64], &[(f64, f64)]) = match (one, upper) {
        (false, false) => (&j0::P1, &j0::Q1, &j0::ROOTS),
        (false, true) => (&j0::P2, &j0::Q2, &j0::ROOTS),
        (true, false) => (&j1::P1, &j1::Q1, &j1::ROOTS),
        (true, true) => (&j1::P2, &j1::Q2, &j1::ROOTS),
    };
    let (hi, lo) = roots[usize::from(upper)];
    let z = if upper && !one {
        1.0 - x * x / 64.0
    } else {
        x * x
    };
    let factor = (x + hi) * ((x - hi) - lo);
    Ok(factor * rational(z, p, q, w)? * if one { x } else { 1.0 })
}

#[inline]
fn small_y(x: f64, one: bool, w: &mut Work) -> Result<f64, SciOxideError> {
    let (p, q, root): (&[f64], &[f64], (f64, f64)) = if one {
        if x <= 4.0 {
            (&y1::P1, &y1::Q1, y1::ROOTS[0])
        } else {
            (&y1::P2, &y1::Q2, y1::ROOTS[1])
        }
    } else if x <= 3.0 {
        (&y0::P1, &y0::Q1, y0::ROOTS[0])
    } else if x <= 5.5 {
        (&y0::P2, &y0::Q2, y0::ROOTS[1])
    } else {
        (&y0::P3, &y0::Q3, y0::ROOTS[2])
    };
    let offset = (x - root.0) - root.1;
    // log(x/root), without rounding x/root to one near a Y zero or zero
    // near the origin. The root's low part also survives in the rational term.
    let logarithm = if offset.abs() < 0.5 * root.0 {
        (offset / root.0).ln_1p()
    } else {
        x.ln() - root.0.ln()
    };
    let factor = (x + root.0) * offset;
    let correction = factor * rational(x * x, p, q, w)?;
    let correction = if one { correction / x } else { correction };
    Ok(scioxide_core::fast_mul_add(
        core::f64::consts::FRAC_2_PI * logarithm,
        small_j(x, one, w)?,
        correction,
    ))
}

/// Value and, when requested, the adjacent order. The returned pair is [F0,F1].
#[inline]
pub(super) fn pair(
    x: f64,
    second: bool,
    order: usize,
    derivative: bool,
    w: &mut Work,
) -> Result<[f64; 2], SciOxideError> {
    let mut out = [0.0; 2];
    let mut phase = None;
    for (n, value) in out.iter_mut().enumerate() {
        if !derivative && n != order {
            continue;
        }
        if let Some(local) = near_zero(x, second, n, w)? {
            *value = local;
            continue;
        }
        if x > 8.0 {
            let (s, c) = if let Some(phase) = phase {
                phase
            } else {
                let computed = super::bessel_kernel::hankel_phase(0.0, x)?;
                phase = Some(computed);
                computed
            };
            let (s, c) = if n == 0 { (s, c) } else { (-c, s) };
            // Detect cancellation before evaluating the ordinary rational
            // tables, so a zero solve never pays for two approximations.
            if x >= 256.0 && (if second { s } else { c }).abs() < 0.0625 {
                *value = super::bessel_kernel::low_cancellation(n as f64, x, second, w)?;
                continue;
            }
            let (p, q, r, t): (&[f64], &[f64], &[f64], &[f64]) = if n == 0 {
                (&j0::PC, &j0::QC, &j0::PS, &j0::QS)
            } else {
                (&j1::PC, &j1::QC, &j1::PS, &j1::QS)
            };
            let inverse = 8.0 / x;
            let rc = rational(inverse * inverse, p, q, w)?;
            let rs = inverse * rational(inverse * inverse, r, t, w)?;
            let oscillation = if second {
                rc * s + rs * c
            } else {
                rc * c - rs * s
            };
            *value = (core::f64::consts::FRAC_2_PI / x).sqrt() * oscillation;
            if oscillation.abs() < (rc.abs() + rs.abs()) / 32.0 {
                *value = super::bessel_kernel::low_cancellation(n as f64, x, second, w)?;
            }
        } else {
            *value = if second {
                small_y(x, n == 1, w)?
            } else {
                small_j(x, n == 1, w)?
            };
        }
    }
    Ok(out)
}

#[inline]
fn near_zero(
    x: f64,
    second: bool,
    order: usize,
    w: &mut Work,
) -> Result<Option<f64>, SciOxideError> {
    if !(8.0..256.0).contains(&x) {
        return Ok(None);
    }
    use super::bessel_zero_polynomials as zeros;
    let tables = match (second, order) {
        (false, 0) => &zeros::J0,
        (false, _) => &zeros::J1,
        (true, 0) => &zeros::Y0,
        (true, _) => &zeros::Y1,
    };
    let shift = if second { 0.75 } else { 0.25 } - 0.5 * order as f64;
    let index = (x / core::f64::consts::PI + shift).round() as usize;
    let Some((root, coefficients)) = tables.get(index - 1) else {
        return Ok(None);
    };
    let offset = (x - root.0) - root.1;
    if offset.abs() > 0.125 {
        return Ok(None);
    }
    w.steps(coefficients.len())?;
    Ok(Some(
        offset * polynomial(offset, coefficients, PolynomialPolicy::Fused),
    ))
}

#[inline]
pub(super) fn value(order: f64, x: f64, second: bool, w: &mut Work) -> Result<f64, SciOxideError> {
    let n = usize::from(order == 1.0);
    Ok(pair(x, second, n, false, w)?[n])
}
