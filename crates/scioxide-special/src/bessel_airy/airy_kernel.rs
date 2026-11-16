//! Shared Airy ODE coefficients, Bessel reductions and full-range real seeds.
//!
//! Real seeds use fixed ODE polynomials for |x| <= 16, with compensated leading
//! coefficients and a bounded `f64` tail. Cancellation triggers a fully
//! compensated evaluation. Beyond that band, Olver's expansions keep a
//! two-term phase on the oscillatory side. Values and derivatives share work.
use crate::bessel_airy::complex_bessel::{self as bessel, ExponentialEstimate as Cylinder};
use crate::support::Jet;
use crate::support::JetEstimate as Estimate;
use crate::support::Work;
use crate::support::{C, real};

use scioxide_core::AlgorithmComponent;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)] // A sibling can enable core/std through feature unification.
use scioxide_core::RealScalar;
use scioxide_core::SciOxideError;
use scioxide_core::algorithm_component;
const AI0: f64 = 0.355_028_053_887_817_2;
const AIP0: f64 = -0.258_819_403_792_806_8;
const BI0: f64 = 0.614_926_627_446_000_7;
const BIP0: f64 = 0.448_288_357_353_826_4;
#[derive(Clone, Copy)]
pub(crate) enum Kind {
    Ai,
    Bi,
}

fn constants(kind: Kind) -> (f64, f64) {
    match kind {
        Kind::Ai => (AI0, AIP0),
        Kind::Bi => (BI0, BIP0),
    }
}

use crate::bessel_airy::bessel_base::{self as wide, Wide};

#[inline]
fn accurate_sin_cos(phase: Wide) -> (Wide, Wide) {
    let (sin, cos) = phase.hi.sin_cos();
    if sin.abs().min(cos.abs()) > 1.0e-8 {
        // |phase.lo| <= 2^-43 on the direct path and is smaller after fixed
        // reduction. The omitted quadratic correction is below one ULP here.
        return (
            wide::sum(Wide::from(sin), wide::two_product(phase.lo, cos)),
            wide::sum(Wide::from(cos), wide::two_product(-phase.lo, sin)),
        );
    }
    phase.sin_cos()
}

#[inline]
fn exponent(x: f64) -> Wide {
    let root = x.sqrt();
    let square = wide::two_product(root, root);
    let correction = ((x - square.hi) - square.lo) / (2.0 * root);
    let fraction = crate::support::constants::TWO_THIRDS;
    wide::scale(
        Wide {
            hi: root,
            lo: correction,
        },
        x,
    ) * fraction
}

/// Fixed ODE polynomial, evaluated with compensated Horner. The high/low
/// coefficient words avoid rebuilding a recurrence and keep accuracy under
/// cancellation next to zeros of either the value or its derivative.
fn real_taylor(
    x: f64,
    kind: Kind,
    scaled: bool,
    work: &mut Work,
) -> Result<Estimate, SciOxideError> {
    let anchor = x.round().clamp(-16.0, 16.0) as i32;
    let index = usize::from(matches!(kind, Kind::Bi));
    let coefficients = &super::airy_polynomials::COEFFICIENTS[(anchor + 16) as usize][index];
    // A narrow strip around the rounded zeros selects full precision before
    // any evaluation. This avoids repeated work at the final Newton iterate;
    // the error-based retry below still covers every other cancellation.
    let zeros = super::airy_polynomials::CANCELLATION_ZEROS[index];
    let position = zeros.partition_point(|&zero| zero < x);
    let near_zero = zeros
        .get(position)
        .into_iter()
        .chain(position.checked_sub(1).and_then(|i| zeros.get(i)))
        .any(|&zero| (x - zero).abs() <= 64.0 * f64::EPSILON.sqrt());
    let h = x - f64::from(anchor);
    work.steps(coefficients.len())?;
    let evaluate = |head: usize| {
        let (mut value, mut derivative) = (0.0_f64, 0.0_f64);
        let (mut magnitude, mut dmagnitude) = (0.0, 0.0);
        for &(hi, _) in coefficients[head..].iter().rev() {
            derivative = derivative * h + value;
            value = value * h + hi;
            dmagnitude = dmagnitude * h.abs() + magnitude;
            magnitude = magnitude * h.abs() + hi.abs();
        }
        let rounding = 2.0 * (coefficients.len() - head) as f64 * f64::EPSILON;
        let error = rounding * magnitude * h.abs().powi(head as i32);
        let derror = rounding
            * (dmagnitude * h.abs().powi(head as i32)
                + head as f64 * magnitude * h.abs().powi(head as i32 - 1));
        let (mut value, mut derivative) = (Wide::from(value), Wide::from(derivative));
        let anchors =
            &super::airy_anchors::ANCHORS[(anchor + 16) as usize][2 * index..2 * index + 2];
        for &(hi, lo) in coefficients[2..head]
            .iter()
            .rev()
            .chain(anchors.iter().rev())
        {
            derivative = crate::support::polynomial::compensated_step(
                derivative,
                Wide::from(h),
                value.hi,
                value.lo,
            );
            value = crate::support::polynomial::compensated_step(value, Wide::from(h), hi, lo);
            dmagnitude = dmagnitude * h.abs() + magnitude;
            magnitude = magnitude * h.abs() + hi.abs();
        }
        (value, derivative, magnitude, dmagnitude, error, derror)
    };
    let (mut value, mut derivative, mut magnitude, mut dmagnitude, mut error, mut derror) =
        evaluate(if near_zero { coefficients.len() } else { 8 });
    if error > f64::EPSILON * value.hi.abs() / 16.0
        || derror > f64::EPSILON * derivative.hi.abs() / 16.0
    {
        work.steps(coefficients.len())?;
        (value, derivative, magnitude, dmagnitude, error, derror) = evaluate(16);
        if error > f64::EPSILON * value.hi.abs() / 16.0
            || derror > f64::EPSILON * derivative.hi.abs() / 16.0
        {
            work.steps(coefficients.len())?;
            (value, derivative, magnitude, dmagnitude, error, derror) =
                evaluate(coefficients.len());
        }
    }
    let n = coefficients.len();
    let distance = h.abs();
    let q = (f64::from(anchor).abs() + distance) * distance * distance / (n * (n - 1)) as f64;
    let last = coefficients[n - 3..]
        .iter()
        .enumerate()
        .map(|(i, c)| c.0.abs() * distance.powi((n - 3 + i) as i32))
        .sum::<f64>();
    let tail = q * last / (1.0 - q);
    let dtail = if distance == 0.0 {
        0.0
    } else {
        tail * (n as f64 + 3.0 / (1.0 - q)) / distance
    };
    let multiplier = if scaled && x > 0.0 {
        let z = exponent(x);
        let sign = if matches!(kind, Kind::Ai) { 1.0 } else { -1.0 };
        (sign * z.hi).exp() * (sign * z.lo).exp()
    } else {
        1.0
    };
    value = wide::scale(value, multiplier);
    derivative = wide::scale(derivative, multiplier);
    let rounding = 1024.0 * f64::EPSILON * f64::EPSILON;
    Ok(Estimate::from(crate::support::JetData {
        y: Jet {
            v: real(value.value()),
            d: real(derivative.value()),
        },
        scale: Jet::constant(real(0.0)),
        error: 64.0 * f64::EPSILON * value.hi.abs()
            + multiplier * (tail + rounding * magnitude + error),
        derivative_error: Some(
            64.0 * f64::EPSILON * derivative.hi.abs()
                + multiplier * (dtail + rounding * dmagnitude + derror),
        ),
        method: algorithm_component!("airy_fixed_ode_polynomial"),
    }))
}
pub(crate) fn series(z: C, kind: Kind, work: &mut Work) -> Result<Estimate, SciOxideError> {
    let (a, b) = constants(kind);
    let cube = Jet::variable(z) * Jet::variable(z) * Jet::variable(z);
    let mut f = Jet::constant(real(1.0));
    let mut g = Jet::variable(z);
    let mut ft = f;
    let mut gt = g;
    let mut magnitude = a.abs() + b.abs() * z.norm();
    let mut dmagnitude = b.abs();
    let mut k = 0_usize;
    let (tail, dtail) = loop {
        if !work.poll_step()? {
            return Ok(crate::support::interrupted());
        }
        k += 1;
        let n = 3.0 * k as f64;
        ft = ft * cube / Jet::from(n * (n - 1.0));
        gt = gt * cube / Jet::from(n * (n + 1.0));
        f = f + ft;
        g = g + gt;
        let size = a.abs() * ft.v.norm() + b.abs() * gt.v.norm();
        let dsize = a.abs() * ft.d.norm() + b.abs() * gt.d.norm();
        magnitude += size;
        dmagnitude += dsize;
        let ratio = cube.v.norm() / ((n + 3.0) * (n + 2.0));
        let dratio = ratio * (n + 3.0) / n;
        if dratio < 1.0 {
            let tail = size * ratio / (1.0 - ratio);
            let dtail = dsize * dratio / (1.0 - dratio);
            if tail <= f64::EPSILON * magnitude && dtail <= f64::EPSILON * dmagnitude {
                break (tail, dtail);
            }
        }
    };
    Ok(Estimate::from(crate::support::JetData {
        y: Jet::from(a) * f + Jet::from(b) * g,
        scale: Jet::constant(real(0.0)),
        error: tail + 16.0 * f64::EPSILON * (k + 1) as f64 * magnitude,
        derivative_error: Some(dtail + 16.0 * f64::EPSILON * (k + 1) as f64 * dmagnitude),
        method: algorithm_component!("airy_ode_series"),
    }))
}
pub(crate) struct Coefficients {
    pub u: C,
    pub v: C,
    pub error: f64,
    pub derivative_error: f64,
}
/// Olver u/v coefficients are generated in their scaled term recurrence. No
/// coefficient array or fixed truncation cap is needed.
pub(crate) fn coefficients(
    ratio: C,
    work: &mut Work,
) -> Result<Option<Coefficients>, SciOxideError> {
    let mut u = real(1.0);
    let mut v = u;
    let mut term = u;
    let mut magnitude = 1.0;
    let mut dmagnitude = 1.0;
    let mut previous = f64::MAX;
    let mut k = 0_usize;
    let (tail, dtail) = loop {
        if !work.poll_step()? {
            return Ok(None);
        }
        k += 1;
        let n = k as f64;
        let next = term * ratio * ((6.0 * n - 1.0) * (6.0 * n - 5.0) / (72.0 * n));
        let dv = next * ((6.0 * n + 1.0) / (1.0 - 6.0 * n));
        let size = next.norm();
        let dsize = dv.norm();
        if size > previous {
            break (size, dsize);
        }
        term = next;
        u += term;
        v += dv;
        magnitude += size;
        dmagnitude += dsize;
        if size <= f64::EPSILON * magnitude && dsize <= f64::EPSILON * dmagnitude {
            break (size, dsize);
        }
        previous = size;
    };
    Ok(Some(Coefficients {
        u,
        v,
        error: tail + 16.0 * f64::EPSILON * (k + 1) as f64 * magnitude,
        derivative_error: dtail + 16.0 * f64::EPSILON * (k + 1) as f64 * dmagnitude,
    }))
}
/// Olver u/v sums for a real argument as (real, imaginary) parts. With a real
/// ratio, or -i times a positive one on the oscillatory side, every term of
/// [`coefficients`] is purely real or imaginary, so the sums run in real
/// arithmetic with the same roundings.
struct RealCoefficients {
    u: (f64, f64),
    v: (f64, f64),
    error: f64,
    derivative_error: f64,
}

fn real_coefficients(
    ratio: f64,
    oscillatory: bool,
    work: &mut Work,
) -> Result<Option<RealCoefficients>, SciOxideError> {
    let (mut u, mut v) = ((1.0, 0.0), (1.0, 0.0));
    let mut term = 1.0_f64;
    let mut magnitude = 1.0;
    let mut dmagnitude = 1.0;
    let mut previous = f64::MAX;
    let mut k = 0_usize;
    let (tail, dtail) = loop {
        if !work.poll_step()? {
            return Ok(None);
        }
        k += 1;
        let n = k as f64;
        let next = term * ratio * ((6.0 * n - 1.0) * (6.0 * n - 5.0) / (72.0 * n));
        let dv = next * ((6.0 * n + 1.0) / (1.0 - 6.0 * n));
        let size = next.abs();
        let dsize = dv.abs();
        if size > previous {
            break (size, dsize);
        }
        term = next;
        // (-i)^k cycles -i, -1, i, 1 on the oscillatory side.
        let (part, sign) = if oscillatory {
            match k % 4 {
                1 => (1, -1.0),
                2 => (0, -1.0),
                3 => (1, 1.0),
                _ => (0, 1.0),
            }
        } else {
            (0, 1.0)
        };
        if part == 0 {
            u.0 += sign * term;
            v.0 += sign * dv;
        } else {
            u.1 += sign * term;
            v.1 += sign * dv;
        }
        magnitude += size;
        dmagnitude += dsize;
        if size <= f64::EPSILON * magnitude && dsize <= f64::EPSILON * dmagnitude {
            break (size, dsize);
        }
        previous = size;
    };
    Ok(Some(RealCoefficients {
        u,
        v,
        error: tail + 16.0 * f64::EPSILON * (k + 1) as f64 * magnitude,
        derivative_error: dtail + 16.0 * f64::EPSILON * (k + 1) as f64 * dmagnitude,
    }))
}

pub(crate) fn pack(value: Cylinder, derivative: Cylinder, method: AlgorithmComponent) -> Estimate {
    let common = value.scale.max(derivative.scale);
    let a = (value.scale - common).exp();
    let b = (derivative.scale - common).exp();
    Estimate::from(crate::support::JetData {
        y: Jet {
            v: value.value * a,
            d: derivative.value * b,
        },
        scale: Jet::constant(real(common)),
        error: value.error * a,
        derivative_error: Some(derivative.error * b),
        method,
    })
}
pub(crate) fn bessel_pair(z: C, kind: Kind, work: &mut Work) -> Result<Estimate, SciOxideError> {
    let left = z.re < 0.0;
    let point = if left { -z } else { z };
    let root = point.sqrt();
    let argument = (point * (2.0 / 3.0)) * root;
    if !argument.is_finite() || argument == real(0.0) {
        return Err(crate::support::instability(
            "Airy Bessel argument is unavailable",
        ));
    }
    let inv_sqrt3 = 1.0 / 3.0_f64.sqrt();
    let (value, derivative) = if left {
        let j = bessel::kernel(bessel::Kind::J, 1.0 / 3.0, argument, work)?;
        let y = bessel::kernel(bessel::Kind::Y, 1.0 / 3.0, argument, work)?;
        let jd = bessel::kernel(bessel::Kind::J, 2.0 / 3.0, argument, work)?;
        let yd = bessel::kernel(bessel::Kind::Y, 2.0 / 3.0, argument, work)?;
        match kind {
            Kind::Ai => (
                bessel::add(
                    bessel::multiply(j, root * 0.5),
                    bessel::multiply(y, root * (-0.5 * inv_sqrt3)),
                ),
                bessel::add(
                    bessel::multiply(jd, point * 0.5),
                    bessel::multiply(yd, point * (0.5 * inv_sqrt3)),
                ),
            ),
            Kind::Bi => (
                bessel::add(
                    bessel::multiply(j, root * (-0.5 * inv_sqrt3)),
                    bessel::multiply(y, root * (-0.5)),
                ),
                bessel::add(
                    bessel::multiply(jd, point * (0.5 * inv_sqrt3)),
                    bessel::multiply(yd, point * (-0.5)),
                ),
            ),
        }
    } else {
        let k = bessel::kernel(bessel::Kind::K, 1.0 / 3.0, argument, work)?;
        let kd = bessel::kernel(bessel::Kind::K, 2.0 / 3.0, argument, work)?;
        match kind {
            Kind::Ai => (
                bessel::multiply(k, root * (inv_sqrt3 / core::f64::consts::PI)),
                bessel::multiply(kd, z * (-inv_sqrt3 / core::f64::consts::PI)),
            ),
            Kind::Bi => {
                let i = bessel::kernel(bessel::Kind::I, 1.0 / 3.0, argument, work)?;
                let id = bessel::kernel(bessel::Kind::I, 2.0 / 3.0, argument, work)?;
                (
                    bessel::add(
                        bessel::multiply(i, root * (2.0 * inv_sqrt3)),
                        bessel::multiply(k, root / core::f64::consts::PI),
                    ),
                    bessel::add(
                        bessel::multiply(id, z * (2.0 * inv_sqrt3)),
                        bessel::multiply(kd, z / core::f64::consts::PI),
                    ),
                )
            }
        }
    };
    let mut e = pack(
        value,
        derivative,
        algorithm_component!("airy_bessel_reduction"),
    );
    let uncertainty = 16.0 * f64::EPSILON * (1.0 + argument.norm());
    e.error += uncertainty * e.y.v.norm();
    e.derivative_error = Some(e.derivative_error.unwrap_or(0.0) + uncertainty * e.y.d.norm());
    Ok(e)
}
pub(crate) fn phase(x: f64, work: &mut Work) -> Result<Option<(f64, f64)>, SciOxideError> {
    if !work.poll_step()? {
        return Ok(None);
    }
    Ok(Some((real_phase(x)?.value(), 16.0 * f64::EPSILON)))
}

#[inline]
fn real_phase(x: f64) -> Result<Wide, SciOxideError> {
    let zeta = exponent(x);
    if zeta.hi <= 1024.0 {
        Ok(zeta + crate::support::constants::FRAC_PI_4)
    } else {
        let (hi, lo) = crate::bessel_airy::phase::airy_oscillatory_phase_parts(x)?;
        Ok(Wide { hi, lo })
    }
}
pub(crate) fn real_pair(
    x: f64,
    kind: Kind,
    scaled: bool,
    work: &mut Work,
) -> Result<Estimate, SciOxideError> {
    real_seed(x, kind, scaled, work).map(|seed| seed.estimate)
}
pub(crate) struct RealSeed {
    pub estimate: Estimate,
    pub phase: Option<(f64, f64)>,
}
impl From<Estimate> for RealSeed {
    fn from(estimate: Estimate) -> Self {
        Self {
            estimate,
            phase: None,
        }
    }
}
pub(crate) fn real_seed(
    x: f64,
    kind: Kind,
    scaled: bool,
    work: &mut Work,
) -> Result<RealSeed, SciOxideError> {
    if !work.poll_step()? {
        return Ok(crate::support::interrupted().into());
    }
    if x == 0.0 {
        let (v, d) = constants(kind);
        return Ok(Estimate::from(crate::support::JetData {
            y: Jet {
                v: real(v),
                d: real(d),
            },
            scale: Jet::constant(real(0.0)),
            error: f64::EPSILON * v.abs(),
            derivative_error: Some(f64::EPSILON * d.abs()),
            method: algorithm_component!("airy_origin"),
        })
        .into());
    }
    let magnitude = x.abs();
    if magnitude <= 16.0 {
        return Ok(real_taylor(x, kind, scaled, work)?.into());
    }
    let root = magnitude.sqrt();
    let quarter = root.sqrt();
    let inverse = (1.5 / magnitude) / root;
    let prefactor = 1.0 / (core::f64::consts::PI.sqrt() * quarter);
    let dprefactor = quarter / core::f64::consts::PI.sqrt();
    if x < 0.0 {
        let Some(c) = real_coefficients(inverse, true, work)? else {
            return Ok(crate::support::interrupted().into());
        };
        if !work.poll_step()? {
            return Ok(crate::support::interrupted().into());
        }
        let theta = real_phase(magnitude)?;
        let (sin, cos) = accurate_sin_cos(theta);
        let phase_error = 16.0 * f64::EPSILON;
        let combine = |first: Wide, a: f64, second: Wide, b: f64, scale: f64| {
            wide::scale(wide::scale(first, a) + wide::scale(second, b), scale).value()
        };
        let (v, d) = match kind {
            Kind::Ai => (
                combine(sin, c.u.0, cos, c.u.1, prefactor),
                combine(cos, c.v.0, sin, -c.v.1, -dprefactor),
            ),
            Kind::Bi => (
                combine(cos, c.u.0, sin, -c.u.1, prefactor),
                combine(sin, c.v.0, cos, c.v.1, dprefactor),
            ),
        };
        Ok(RealSeed {
            estimate: Estimate::from(crate::support::JetData {
                y: Jet {
                    v: real(v),
                    d: real(d),
                },
                scale: Jet::constant(real(0.0)),
                error: prefactor * (2.0 * c.error + phase_error * c.u.0.hypot(c.u.1)),
                derivative_error: Some(
                    dprefactor * (2.0 * c.derivative_error + phase_error * c.v.0.hypot(c.v.1)),
                ),
                method: algorithm_component!("airy_olver_fixed_phase"),
            }),
            phase: Some((theta.value(), phase_error)),
        })
    } else {
        let sign = if matches!(kind, Kind::Ai) { -1.0 } else { 1.0 };
        let Some(c) = real_coefficients(sign * inverse, false, work)? else {
            return Ok(crate::support::interrupted().into());
        };
        let factor = if matches!(kind, Kind::Ai) { 0.5 } else { 1.0 };
        let zeta = if scaled { Wide::ZERO } else { exponent(x) };
        if !scaled && !zeta.hi.is_finite() {
            return Err(crate::support::instability(
                "Airy logarithmic exponent is unavailable",
            ));
        }
        if !scaled && zeta.hi < 750.0 {
            // Split the exponential around the prefactor: neither a growing
            // exponential nor a decaying one loses range before the final value.
            let half = (0.5 * sign * zeta.hi).exp() * (0.5 * sign * zeta.lo).exp();
            let value = ((c.u.0 * factor * prefactor) * half) * half;
            let derivative = ((c.v.0 * sign * factor * dprefactor) * half) * half;
            if value > 0.0 && value.is_finite() {
                return Ok(Estimate::from(crate::support::JetData {
                    y: Jet {
                        v: real(value),
                        d: real(derivative),
                    },
                    scale: Jet::constant(real(0.0)),
                    error: value * (c.error / c.u.0.abs() + 16.0 * f64::EPSILON),
                    derivative_error: Some(
                        derivative.abs() * (c.derivative_error / c.v.0.abs() + 16.0 * f64::EPSILON),
                    ),
                    method: algorithm_component!("airy_olver_real"),
                })
                .into());
            }
        }
        let scale = if scaled { 0.0 } else { sign * zeta.value() };
        let uncertainty = 16.0 * f64::EPSILON * (1.0 + scale.abs());
        Ok(Estimate::from(crate::support::JetData {
            y: Jet {
                v: real(c.u.0 * (factor * prefactor)),
                d: real(c.v.0 * (sign * factor * dprefactor)),
            },
            scale: Jet::constant(real(scale)),
            error: factor * prefactor * (c.error + uncertainty * c.u.0.abs()),
            derivative_error: Some(
                factor * dprefactor * (c.derivative_error + uncertainty * c.v.0.abs()),
            ),
            method: algorithm_component!("airy_olver_real"),
        })
        .into())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn low_phase_survives_a_near_axis_projection() {
        // sin(high + low), independently rounded with mpmath at 160/240 digits.
        let (sine, cosine) = accurate_sin_cos(Wide {
            hi: core::f64::consts::PI,
            lo: 1.000_122_464_679_914_7e-12,
        });
        let expected = f64::from_bits(0xbd71_9799_812d_ea11);
        assert!((sine.value() - expected).abs() <= 2.0e-28);
        assert_eq!(cosine.value(), -1.0);
    }

    /// The anchored Taylor series with every coefficient and Horner step in
    /// two-term arithmetic, the reference for the mixed evaluation.
    fn two_term_taylor(x: f64, kind: Kind) -> f64 {
        let anchor = x.round().clamp(-16.0, 16.0);
        let row = crate::bessel_airy::airy_anchors::ANCHORS[(anchor as i32 + 16) as usize];
        let index = if matches!(kind, Kind::Ai) { 0 } else { 2 };
        let mut coefficients = [Wide::ZERO; 48];
        coefficients[0] = Wide {
            hi: row[index].0,
            lo: row[index].1,
        };
        coefficients[1] = Wide {
            hi: row[index + 1].0,
            lo: row[index + 1].1,
        };
        for n in 0..coefficients.len() - 2 {
            let prior = if n == 0 {
                Wide::ZERO
            } else {
                coefficients[n - 1]
            };
            coefficients[n + 2] =
                (coefficients[n].mul_f64(anchor) + prior) / Wide::from(((n + 2) * (n + 1)) as f64);
        }
        coefficients
            .iter()
            .rev()
            .fold(Wide::ZERO, |value, &coefficient| {
                value.mul_f64(x - anchor) + coefficient
            })
            .value()
    }

    #[test]
    fn mixed_precision_taylor_matches_the_two_term_series() {
        // Anchors at both ends, the first zeros of Ai and Bi and the origin.
        for x in [
            -16.4,
            -15.7,
            -4.087_949_444_130_97,
            -2.338_107_410_459_767,
            -1.173_713_222_709_127_9,
            -0.3,
            1.0e-4,
            0.49,
            3.7,
            15.6,
            16.2,
        ] {
            for kind in [Kind::Ai, Kind::Bi] {
                let estimate = real_taylor(x, kind, false, &mut Work::new(1 << 16)).unwrap();
                let reference = two_term_taylor(x, kind);
                let value = estimate.y.v.re;
                assert!(
                    (value - reference).abs() <= f64::EPSILON * reference.abs(),
                    "{x}: {value:e} {reference:e}"
                );
                assert!((value - reference).abs() <= estimate.error);
            }
        }
    }

    #[test]
    fn real_olver_sums_match_the_complex_recurrence() {
        for (ratio, oscillatory) in [
            (0.023, true),
            (0.004, true),
            (0.023, false),
            (-0.017, false),
        ] {
            let real = real_coefficients(ratio, oscillatory, &mut Work::new(1 << 16))
                .unwrap()
                .unwrap();
            let complex_ratio = if oscillatory {
                C::new(0.0, -ratio)
            } else {
                crate::support::real(ratio)
            };
            let complex = coefficients(complex_ratio, &mut Work::new(1 << 16))
                .unwrap()
                .unwrap();
            for (left, right) in [
                (real.u.0, complex.u.re),
                (real.u.1, complex.u.im),
                (real.v.0, complex.v.re),
                (real.v.1, complex.v.im),
                (real.error, complex.error),
                (real.derivative_error, complex.derivative_error),
            ] {
                assert_eq!(left.abs(), right.abs(), "{ratio} {oscillatory}");
                assert!(left == right || left == 0.0);
            }
        }
    }
}
