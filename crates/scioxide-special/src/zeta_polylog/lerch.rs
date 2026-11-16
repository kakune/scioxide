//! Principal Lerch continuation and internal complex gamma/zeta ingredients.
use crate::support::TRANSCENDENTAL_ERRORS as ERRORS;
use crate::support::TranscendentalEstimate as Estimate;
use crate::support::Work;
use crate::support::{C, ONE, ZERO, real};
use core::f64::consts::PI;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)] // A sibling may enable core/std through feature unification.
use scioxide_core::RealScalar;
use scioxide_core::algorithm_component;
use scioxide_core::{DomainReason, SciOxideError};

fn log_gamma_positive(z: C, work: &mut Work<'_, '_, '_>) -> Result<C, SciOxideError> {
    Ok(crate::support::gamma_core::complex_log_gamma(z, &mut || work.try_step())?.0)
}
fn gamma_positive(z: C, work: &mut Work<'_, '_, '_>) -> Result<C, SciOxideError> {
    Ok(log_gamma_positive(z, work)?.exp())
}

fn hurwitz(s: C, a: C, work: &mut Work<'_, '_, '_>) -> Result<Estimate, SciOxideError> {
    crate::zeta_polylog::complex_zeta::hurwitz(s, a, work)
}
fn zeta(s: C, work: &mut Work<'_, '_, '_>) -> Result<Estimate, SciOxideError> {
    crate::zeta_polylog::complex_zeta::zeta(s, work)
}

pub(super) fn series(
    z: C,
    s: C,
    a: C,
    work: &mut Work<'_, '_, '_>,
) -> Result<Estimate, SciOxideError> {
    let mut value = ZERO;
    let mut compensation = ZERO;
    let mut power = ONE;
    let mut n = 0usize;
    let mut magnitude = 0.0;
    // Each term is exp(w) with w = -s ln(a+n): the rounded exponent
    // perturbs the term by about 2ε|w| relative.
    let mut exponent_rounding = 0.0;
    let mut previous = f64::MAX;
    let mut scale = 0.0;
    let log_z = z.ln();
    // Half the binary64 exponent range leaves headroom for products and sums.
    const EXPONENT_HEADROOM: f64 = 512.0 * core::f64::consts::LN_2;
    loop {
        work.try_step()?;
        let exponent = -s * (a + real(n as f64)).ln();
        let term = if scale == 0.0 && exponent.re.abs() < EXPONENT_HEADROOM && power != ZERO {
            power * exponent.exp()
        } else {
            let log_term = exponent + real(n as f64) * log_z;
            if log_term.re > scale {
                let factor = (scale - log_term.re).exp();
                value *= factor;
                compensation *= factor;
                magnitude *= factor;
                exponent_rounding *= factor;
                previous *= factor;
                scale = log_term.re;
            }
            (log_term - real(scale)).exp()
        };
        if !term.is_finite() {
            return Err(ERRORS.unstable());
        }
        let corrected = term - compensation;
        let next = value + corrected;
        compensation = (next - value) - corrected;
        value = next;
        magnitude += term.norm();
        exponent_rounding += term.norm()
            * (exponent.norm()
                + if scale == 0.0 {
                    0.0
                } else {
                    n as f64 * log_z.norm()
                });
        // For every later real t>=n, |d log((a+t)^(-s))/dt|
        // <= |s|/(Re(a)+n). This bounds ALL future ratios, including complex
        // orders whose term magnitudes decrease before a later maximum.
        let ratio = (z.norm() * (s.norm() / (a.re + n as f64)).exp()).max(term.norm() / previous);
        if n > 8 && ratio < 1.0 {
            let tail = term.norm() * ratio / (1.0 - ratio);
            if tail < 2e-15 * value.norm().max(1e-300) {
                let mut estimate = Estimate::new(
                    value,
                    // Each scaled term and partial sum may round below the
                    // normal range with an absolute error of one quantum.
                    tail + f64::EPSILON * (16.0 * magnitude + 2.0 * exponent_rounding)
                        + 2.0 * (n + 1) as f64 * f64::from_bits(1),
                    algorithm_component!("lerch_compensated_series"),
                );
                estimate.scale = scale;
                return Ok(estimate);
            }
        }
        previous = term.norm();
        power *= z;
        n += 1;
    }
}

pub(super) fn logarithmic(
    z: C,
    s: C,
    a: C,
    work: &mut Work<'_, '_, '_>,
) -> Result<Estimate, SciOxideError> {
    let mu = z.ln();
    // Re(s)<1/2 excludes colliding gamma/zeta poles. Positive and near-positive
    // integer orders use the regularized Mellin path instead.
    let initial = gamma_positive(ONE - s, work)? * ((s - ONE) * (-mu).ln()).exp();
    let mut value = initial;
    let mut magnitude = initial.norm();
    let mut error = 0.0;
    let mut factor = ONE;
    let mut n = 0usize;
    loop {
        work.try_step()?;
        let h = if a == ONE {
            zeta(s - real(n as f64), work)?
        } else {
            hurwitz(s - real(n as f64), a, work)?
        };
        let term = h.value * factor;
        value += term;
        magnitude += term.norm();
        error += factor.norm() * h.error;
        if n > 8 && term.norm() < 2e-15 * value.norm().max(1e-300) {
            let scale = (-a * mu).exp();
            return Ok(Estimate::new(
                value * scale,
                scale.norm() * (error + term.norm() + 32.0 * f64::EPSILON * magnitude),
                algorithm_component!("lerch_logarithmic_series"),
            ));
        }
        n += 1;
        factor *= mu / real(n as f64);
        if !value.is_finite() || !factor.is_finite() {
            return Err(ERRORS.unstable());
        }
    }
}

// Fixed-storage adaptive Gauss quadrature. The 32-point rule is shared with
// probability kernels. A panel is accepted only after comparison with its two
// children; each integrand evaluation is admitted before execution.
fn gauss(
    f: &impl Fn(f64) -> C,
    left: f64,
    right: f64,
    work: &mut Work<'_, '_, '_>,
) -> Result<(C, f64), SciOxideError> {
    let half = (right - left) * 0.5;
    let mid = left + half;
    let mut value = ZERO;
    let mut magnitude = 0.0;
    for &(node, weight) in &crate::support::GL32 {
        work.try_step()?;
        let term = f(mid + half * node) * real(weight * half);
        if !term.is_finite() {
            return Err(ERRORS.unstable());
        }
        value += term;
        magnitude += term.norm();
    }
    Ok((value, magnitude))
}
fn integrate(
    f: &impl Fn(f64) -> C,
    left: f64,
    right: f64,
    work: &mut Work<'_, '_, '_>,
) -> Result<(C, f64), SciOxideError> {
    let mut stack = [(0.0, 0.0, ZERO, 0.0); 64];
    let (v, m) = gauss(f, left, right, work)?;
    stack[0] = (left, right, v, m);
    let mut count = 1usize;
    let mut total = ZERO;
    let mut error = 0.0;
    while count > 0 {
        count -= 1;
        let (lo, hi, coarse, _) = stack[count];
        let mid = lo + (hi - lo) * 0.5;
        let (l, lm) = gauss(f, lo, mid, work)?;
        let (r, rm) = gauss(f, mid, hi, work)?;
        let fine = l + r;
        let delta = (fine - coarse).norm();
        if delta <= 2e-13 * fine.norm().max(1e-30) {
            total += fine;
            error += delta + 16.0 * f64::EPSILON * (lm + rm);
        } else {
            if count + 2 > stack.len() || mid == lo || mid == hi {
                return Err(ERRORS.unstable());
            }
            stack[count] = (mid, hi, r, rm);
            stack[count + 1] = (lo, mid, l, lm);
            count += 2;
        }
    }
    Ok((total, error))
}

fn mellin(z: C, s: C, a: C, work: &mut Work<'_, '_, '_>) -> Result<Estimate, SciOxideError> {
    let mu = z.ln();
    // Rotate away from the selected z-cut lip, but never past another pole
    // log(z)+2*pi*i*k, or out of the Laplace decay sector for complex a.
    let pole_angle = (2.0 * PI - mu.im.abs()).atan2(mu.re.max(0.0));
    let angle = 0.35_f64
        .min(0.4 * pole_angle)
        .min(0.4 * a.re / (1.0 + a.im.abs()));
    let theta = if z.im.is_sign_negative() {
        angle
    } else {
        -angle
    };
    let direction = C::new(theta.cos(), theta.sin());
    let decay = (a * direction).re;
    if decay <= 0.0 {
        return Err(ERRORS.unstable());
    }
    let shifts = (1.0 - s.re).ceil().max(0.0);
    if shifts > 64.0 {
        return Err(ERRORS.unstable());
    }
    let shifts = shifts as usize;
    let inv_shifted = (-log_gamma_positive(s + real(shifts as f64), work)?).exp();
    let mut inv_gamma = inv_shifted;
    for j in 0..shifts {
        work.try_step()?;
        inv_gamma *= s + real(j as f64);
    }
    // A local Taylor integral supplies analytic continuation in s. Keeping
    // the reciprocal gamma factor separate removes the removable s=-j poles.
    let radius = 0.4_f64.min(0.25 * mu.norm()).min(0.5 / a.norm());
    if radius == 0.0 || !radius.is_finite() {
        return Err(ERRORS.unstable());
    }
    let dr = direction * real(radius);
    let mut denominator = [ZERO; 96];
    let mut coefficients = [ZERO; 96];
    denominator[0] = ONE - z;
    let mut exp_term = ONE;
    let mut numerator = ONE;
    let count = shifts + 32;
    let mut sum = ZERO;
    let mut magnitude = 0.0;
    for n in 0..count {
        work.try_step()?;
        if n > 0 {
            exp_term *= (-dr) / real(n as f64);
            denominator[n] = -z * exp_term;
            numerator *= (-a * dr) / real(n as f64);
        }
        let mut convolution = ZERO;
        for j in 1..=n {
            work.try_step()?;
            convolution += denominator[j] * coefficients[n - j];
        }
        coefficients[n] = (numerator - convolution) / denominator[0];
        let ratio = if n < shifts {
            let mut r = inv_shifted;
            for j in 0..shifts {
                if j != n {
                    work.try_step()?;
                    r *= s + real(j as f64);
                }
            }
            r
        } else {
            inv_gamma / (s + real(n as f64))
        };
        let term = coefficients[n] * ratio;
        sum += term;
        magnitude += term.norm();
    }
    let scale = (s * (real(radius.ln()) + C::new(0.0, theta))).exp();
    let mut value = sum * scale;
    let mut error = scale.norm()
        * (coefficients[count - 1].norm() * inv_gamma.norm() + 32.0 * f64::EPSILON * magnitude);
    if inv_gamma == ZERO {
        return Ok(Estimate::new(
            value,
            error,
            algorithm_component!("lerch_integer_continuation"),
        ));
    }
    let factor = (s * C::new(0.0, theta)).exp() * inv_gamma;
    let f = |u: f64| {
        let t = direction * real(u);
        // expm1-style denominator near zero is evaluated from its logarithm.
        let exponent = mu - t;
        let den = if exponent.norm() < 0.5 {
            -complex_expm1(exponent)
        } else {
            ONE - exponent.exp()
        };
        factor * ((s - ONE) * real(u.ln()) - a * t).exp() / den
    };
    let mut lo = radius;
    let mut quiet = 0;
    loop {
        let hi = (lo * 2.0).min(lo + 8.0 / decay);
        if hi == lo || !hi.is_finite() {
            return Err(ERRORS.unstable());
        }
        let (panel, pe) = integrate(&f, lo, hi, work)?;
        value += panel;
        error += pe;
        if lo * decay > (s.re.max(0.0) + s.im.abs() + mu.re.max(0.0) + 32.0)
            && panel.norm() < 2e-15 * value.norm().max(1e-300)
        {
            quiet += 1;
        } else {
            quiet = 0;
        }
        if quiet == 2 {
            return Ok(Estimate::new(
                value,
                error + panel.norm() * 2.0,
                algorithm_component!("lerch_regularized_mellin"),
            ));
        }
        lo = hi;
    }
}
use crate::support::complex_expm1;

pub(crate) fn kernel(
    z: C,
    s: C,
    a: C,
    work: &mut Work<'_, '_, '_>,
) -> Result<Estimate, SciOxideError> {
    let polynomial_order = s.im == 0.0 && s.re <= 0.0 && s.re == s.re.floor();
    if !polynomial_order && a.im == 0.0 && a.re <= 0.0 && a.re == a.re.floor() {
        return Err(ERRORS.domain(scioxide_core::argument!("a"), DomainReason::Pole));
    }
    if z == ZERO {
        let value = shift_power(s, a);
        // Only a = 0 gives an exact zero; an exponential below the range
        // rounds to zero within one quantum.
        let rounding = if value == ZERO && a != ZERO {
            f64::from_bits(1)
        } else {
            0.0
        };
        return Ok(Estimate::new(
            value,
            8.0 * f64::EPSILON * value.norm() + rounding,
            algorithm_component!("lerch_zero"),
        ));
    }
    if z == ONE {
        return if a == ONE {
            zeta(s, work)
        } else {
            hurwitz(s, a, work)
        };
    }
    if a == ONE && (polynomial_order || s.im == 0.0 && s.re < 0.0 && z.norm() >= 0.95) {
        let mut e = if polynomial_order {
            super::negative::integer(-s.re, z, work)?
        } else {
            super::negative::continued(s.re, z, work)?
        };
        let radius = z.norm();
        e.value *= (z / real(radius)).conj();
        e.error += 4.0 * f64::EPSILON * e.value.norm();
        e.scale -= radius.ln();
        return Ok(e);
    }
    if s == ZERO {
        let value = ONE / (ONE - z);
        return Ok(Estimate::new(
            value,
            8.0 * f64::EPSILON * value.norm(),
            algorithm_component!("lerch_geometric"),
        ));
    }
    let mut shifted = a;
    let mut prefix = ZERO;
    let mut power = ONE;
    let mut prefix_mag = 0.0;
    while shifted.re < 1.0 {
        work.try_step()?;
        let term = power * shift_power(s, shifted);
        prefix += term;
        prefix_mag += term.norm();
        power *= z;
        let next = C::new(shifted.re + 1.0, shifted.im);
        if next == shifted || !power.is_finite() {
            return Err(ERRORS.unstable());
        }
        shifted = next;
    }
    let e = if z.norm() < 0.95 {
        series(z, s, shifted, work)?
    } else if s.re < 0.5
        && !(s.im == 0.0 && s.re == s.re.floor())
        && z.ln().norm() < 1.0
        // The logarithmic expansion requests arbitrarily negative Hurwitz
        // orders. Use it only in the cancellation-resistant shift-series
        // region; Euler--Maclaurin loses digits as those orders decrease.
        && (shifted - ONE).norm() < 0.75
    {
        logarithmic(z, s, shifted, work)?
    } else {
        mellin(z, s, shifted, work)?
    };
    let mut combined = Estimate::new(
        prefix * (-e.scale).exp() + power * e.value,
        power.norm() * e.error + 32.0 * f64::EPSILON * prefix_mag * (-e.scale).exp(),
        e.method,
    );
    combined.scale = e.scale;
    Ok(combined)
}

use crate::zeta_polylog::complex_zeta::shift_power;
