//! Nonpositive integral polylogarithms, shared with Lerch and thermal integrals.
//! Eulerian rational functions use reciprocal arguments outside the disk.
//! Differentiated Mittag-Leffler poles avoid high-degree polynomial cancellation.
use crate::support::wide::Wide;
use crate::support::{C, ONE, TranscendentalEstimate as Estimate, Work, ZERO, real};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use scioxide_core::RealScalar;
use scioxide_core::{SciOxideError, algorithm_component};

/// Negative nonintegral real orders use the convergent logarithmic disk or
/// Jonquiere inversion (DLMF 25.12.12--13), including the selected cut lip.
pub(super) fn continued(s: f64, z: C, work: &mut Work) -> Result<Estimate, SciOxideError> {
    if z.im.is_sign_negative() {
        let mut e = continued(s, z.conj(), work)?;
        e.value = e.value.conj();
        return Ok(e);
    }
    if z.ln().norm() <= 3.5 {
        let mut e = super::lerch::logarithmic(z, real(s), ONE, work)?;
        e.value *= z;
        e.error = e.error * z.norm() + 4.0 * f64::EPSILON * e.value.norm();
        if z.im == 0.0 && z.re < 0.0 {
            e.value.im = 0.0;
        }
        return Ok(e);
    }
    let inverse = ONE / z;
    let mut inner = super::lerch::series(inverse, real(s), ONE, work)?;
    inner.value *= inverse;
    inner.error = inner.error * inverse.norm() + 4.0 * f64::EPSILON * inner.value.norm();
    let mu = z.ln();
    let q = mu / C::new(0.0, core::f64::consts::TAU);
    let h = super::complex_zeta::hurwitz(real(1.0 - s), q, work)?;
    let log_gamma = crate::support::gamma_series::gamma_log_shifted(Wide::from(-s), work)?;
    let log_factor = log_gamma + Wide::from(core::f64::consts::TAU).ln().mul_f64(s)
        - Wide::from(core::f64::consts::PI).ln();
    let factor = log_factor.exp() * scioxide_core::sin_pi(s)?;
    let (sin, cos) = scioxide_core::sin_cos_pi(s * 0.5)?;
    let coefficient = C::new(cos, sin) * real(factor);
    let (sin, cos) = scioxide_core::sin_cos_pi(s)?;
    let first = coefficient * h.value;
    let second = C::new(cos, sin) * inner.value;
    let mut value = first - second;
    let error = coefficient.norm() * h.error
        + inner.error
        + 64.0
            * f64::EPSILON
            * (1.0 + s.abs() + log_factor.value().abs())
            * (first.norm() + second.norm());
    if z.im == 0.0 && z.re < 0.0 {
        value.im = 0.0;
    }
    Ok(Estimate::new(
        value,
        error,
        algorithm_component!("polylog_negative_inversion"),
    ))
}

/// Li_-n(z), apart from the separately documented zeta-assigned endpoint z=1.
pub(crate) fn integer(n: f64, z: C, work: &mut Work) -> Result<Estimate, SciOxideError> {
    work.try_step()?;
    if z == ZERO {
        return Ok(Estimate::new(
            z,
            0.0,
            algorithm_component!("polylog_integer_zero"),
        ));
    }
    if n == 0.0 {
        let value = if z.norm() > 2.0 {
            -ONE / (ONE - ONE / z)
        } else {
            z / (ONE - z)
        };
        return Ok(Estimate::new(
            value,
            8.0 * f64::EPSILON * value.norm(),
            algorithm_component!("polylog_geometric"),
        ));
    }
    if z == real(-1.0) && n % 2.0 == 0.0 {
        return Ok(Estimate::new(
            ZERO,
            0.0,
            algorithm_component!("polylog_integer_zero"),
        ));
    }
    // Keep the original displacement near z=1: rounding 1/z there would
    // amplify reciprocal rounding by 1/|1-z| in the pole denominator.
    let inverse = z.norm() > 2.0;
    let argument = if inverse { ONE / z } else { z };
    let mut result = if n >= 10.0 && argument.ln().norm() <= n {
        poles(n, argument.ln(), work)?
    } else if n <= 100.0 {
        rational(n as usize, argument, work)?
    } else {
        let mut e = super::lerch::series(argument, real(-n), ONE, work)?;
        e.value *= argument;
        e.error = e.error * argument.norm() + 4.0 * f64::EPSILON * e.value.norm();
        e
    };
    if inverse && n % 2.0 == 0.0 {
        result.value = -result.value;
    }
    Ok(result)
}

/// Evaluate -Li_-n(-exp(x)) without forming exp(x) at large thermal arguments.
pub(crate) fn fermi(n: usize, x: f64, work: &mut Work) -> Result<Estimate, SciOxideError> {
    if x == 0.0 && n.is_multiple_of(2) {
        return integer(n as f64, real(-1.0), work);
    }
    let mut e = if n >= 10 && x.abs() <= n as f64 {
        poles(n as f64, C::new(x, core::f64::consts::PI), work)?
    } else {
        let mut e = integer(n as f64, real(-(-x.abs()).exp()), work)?;
        if x > 0.0 && n.is_multiple_of(2) {
            e.value = -e.value;
        }
        // exp(-|x|) can round to zero while its logarithm remains useful.
        if e.value == ZERO && x.abs() > 700.0 {
            e.value = real(-1.0);
            e.scale = -x.abs();
            if x > 0.0 && n.is_multiple_of(2) {
                e.value = -e.value;
            }
            e.error = 8.0 * f64::EPSILON;
        }
        e
    };
    e.value = -e.value;
    Ok(e)
}

fn rational(n: usize, z: C, work: &mut Work) -> Result<Estimate, SciOxideError> {
    let mut coefficients = [Wide::ZERO; 101];
    coefficients[0] = Wide::from(1.0);
    for degree in 2..=n {
        for k in (1..degree).rev() {
            work.try_step()?;
            coefficients[k] = coefficients[k].mul_f64((k + 1) as f64)
                + coefficients[k - 1].mul_f64((degree - k) as f64);
        }
    }
    let mut re = Wide::ZERO;
    let mut im = Wide::ZERO;
    let mut magnitude = 0.0;
    for coefficient in coefficients[..n].iter().rev() {
        work.try_step()?;
        let next = re.mul_f64(z.re) - im.mul_f64(z.im) + *coefficient;
        im = re.mul_f64(z.im) + im.mul_f64(z.re);
        re = next;
        magnitude = magnitude * z.norm() + coefficient.value().abs();
    }
    let denominator = ONE - z;
    let radius = denominator.norm();
    let phase = (denominator / real(radius)).powu((n + 1) as u32).conj();
    let polynomial = C::new(re.value(), im.value());
    let value = polynomial * z * phase;
    let scale = -(n as f64 + 1.0) * radius.ln();
    let mut result = Estimate::new(
        value,
        32.0 * f64::EPSILON
            * (n as f64 + 1.0 + scale.abs())
            * (magnitude * z.norm()).max(value.norm()),
        algorithm_component!("polylog_eulerian_rational"),
    );
    result.scale = scale;
    Ok(result)
}

fn poles(n: f64, mu: C, work: &mut Work) -> Result<Estimate, SciOxideError> {
    let m = n + 1.0;
    let radius = mu.norm();
    let term = |k: f64| {
        let base = real(radius) / (C::new(0.0, core::f64::consts::TAU * k) - mu);
        if m <= f64::from(u32::MAX) {
            base.powu(m as u32)
        } else {
            (base.ln() * real(m)).exp()
        }
    };
    let mut value = term(0.0);
    let mut compensation = ZERO;
    let mut magnitude = value.norm();
    let mut k = 0usize;
    let remainder = loop {
        work.try_step()?;
        k += 1;
        let pair = term(k as f64) + term(-(k as f64));
        let corrected = pair - compensation;
        let next = value + corrected;
        compensation = (next - value) - corrected;
        value = next;
        magnitude += term(k as f64).norm() + term(-(k as f64)).norm();
        let distance = core::f64::consts::TAU * (k + 1) as f64 - mu.im.abs();
        let tail =
            2.0 * (radius / distance).powf(m) * (1.0 + distance / (core::f64::consts::TAU * n));
        if tail <= 2.0 * f64::EPSILON * value.norm().max(1e-300) {
            break tail;
        }
    };
    if mu.im.abs() == core::f64::consts::PI || mu.im == 0.0 {
        value.im = 0.0;
    }
    let log_factorial = crate::support::gamma_series::gamma_log_shifted(Wide::from(n), work)?;
    let scale = log_factorial - Wide::from(radius).ln().mul_f64(m);
    let mut estimate = Estimate::new(
        value,
        remainder + 64.0 * f64::EPSILON * (m + scale.value().abs() + 1.0) * magnitude,
        algorithm_component!("polylog_integer_poles"),
    );
    estimate.scale = scale.value();
    Ok(estimate)
}
