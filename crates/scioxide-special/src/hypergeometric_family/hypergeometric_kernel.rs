//! Pure complex hypergeometric recurrences with explicit work accounting.
use crate::support::HYPERGEOMETRIC_ERRORS as ERRORS;
use crate::support::JetEstimate as Estimate;
use crate::support::Work;
use crate::support::{C, Jet, ONE, ZERO, pole, real};
#[allow(unused_imports)]
// Required in standalone no_std; a std-linked consumer has inherent f64 methods.
use scioxide_core::RealScalar;
use scioxide_core::algorithm_component;
use scioxide_core::{DomainReason, SciOxideError};

use crate::support::norm;

/// Real parts of at most two parameters of a named real family.
pub(crate) fn real_parts(x: &[Jet]) -> [f64; 2] {
    let mut out = [0.0; 2];
    for (out, x) in out.iter_mut().zip(x) {
        *out = x.v.re;
    }
    out
}

fn log_gamma(z: Jet, w: &mut Work<'_, '_, '_>) -> Result<Jet, SciOxideError> {
    let (v, d) = crate::support::gamma_core::complex_log_gamma(z.v, &mut || w.try_step())?;
    Ok(Jet {
        v,
        d: if z.d == ZERO { ZERO } else { d * z.d },
    })
}

fn rgamma(z: Jet, w: &mut Work<'_, '_, '_>) -> Result<Jet, SciOxideError> {
    if pole(z.v) {
        if z.d == ZERO {
            return Ok(0.0.into());
        }
        let n = -z.v.re;
        let factorial = log_gamma(Jet::from(n + 1.0), w)?.exp().v;
        return Ok(Jet {
            v: ZERO,
            d: z.d * factorial * real(if n % 2.0 == 0.0 { 1.0 } else { -1.0 }),
        });
    }
    let value = (-log_gamma(z, w)?).exp();
    if value.v == ZERO {
        return Err(ERRORS.unstable());
    }
    Ok(value)
}

/// Rejects a denominator parameter at a pole that the series reaches before it terminates;
/// `poles` names the denominator parameters in the error.
pub(crate) fn validate(
    a: &[Jet],
    b: &[Jet],
    regularized: bool,
    poles: scioxide_core::Argument,
) -> Result<(), SciOxideError> {
    let end = a
        .iter()
        .filter(|x| pole(x.v) && x.d == ZERO)
        .map(|x| -x.v.re)
        .fold(f64::INFINITY, f64::min);
    for x in b {
        if !regularized && pole(x.v) && end >= -x.v.re {
            return Err(ERRORS.domain(poles, DomainReason::Pole));
        }
    }
    Ok(())
}

/// Rebuild a regularized coefficient only at denominator-zero crossings.
fn coefficient(
    a: &[Jet],
    b: &[Jet],
    n: usize,
    w: &mut Work<'_, '_, '_>,
) -> Result<Jet, SciOxideError> {
    let mut value = Jet::from(1.0);
    for &x in a {
        for k in 0..n {
            w.try_step()?;
            value = value * (x + Jet::from(k as f64));
        }
    }
    for &x in b {
        value = value * rgamma(x + Jet::from(n as f64), w)?;
    }
    for k in 1..=n {
        w.try_step()?;
        value = value / Jet::from(k as f64);
    }
    Ok(value)
}

/// Forward recurrence, Kahan accumulation and an eventual ratio majorant.
fn series(
    a: &[Jet],
    b: &[Jet],
    z: Jet,
    regularized: bool,
    w: &mut Work<'_, '_, '_>,
) -> Result<Estimate, SciOxideError> {
    validate(a, b, regularized, scioxide_core::argument!("denominator"))?;
    let mut term = if regularized {
        coefficient(a, b, 0, w)?
    } else {
        Jet::from(1.0)
    };
    let mut sum = term;
    let mut correction = Jet::from(0.0);
    let mut magnitude = term.norm();
    let end = a
        .iter()
        .filter(|x| pole(x.v) && x.d == ZERO)
        .map(|x| -x.v.re)
        .fold(f64::INFINITY, f64::min);
    let last_pole = b
        .iter()
        .filter(|x| pole(x.v))
        .map(|x| -x.v.re)
        .fold(-1.0, f64::max);
    let mut power = Jet::from(1.0);
    for n in 0usize.. {
        if n as f64 == end {
            return Ok(Estimate::new(
                sum,
                4.0 * f64::EPSILON * magnitude,
                algorithm_component!("terminating_polynomial"),
            ));
        }
        w.try_step()?;
        let k = Jet::from(n as f64);
        if regularized && n as f64 <= last_pole {
            power = power * z;
        }
        if regularized && b.iter().any(|x| (x.v + real(n as f64)) == ZERO) {
            term = coefficient(a, b, n + 1, w)? * power;
        } else {
            term = term * z / Jet::from((n + 1) as f64);
            for &x in a {
                w.try_step()?;
                term = term * (x + k);
            }
            for &x in b {
                w.try_step()?;
                term = term / (x + k);
            }
        }
        let adjusted = term - correction;
        let next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
        magnitude += term.norm();
        if !sum.v.is_finite() || !sum.d.is_finite() || !magnitude.is_finite() {
            return Err(ERRORS.unstable());
        }
        // A majorant is needed only for a possible stopping point. Small
        // intermediate terms still cannot terminate before the eventual-ratio
        // test below (including every pending denominator pole) succeeds.
        if (a.len() > b.len() + 1 && end.is_finite())
            || n <= 8
            || term.norm() > 8.0 * f64::EPSILON * sum.norm().max(1e-300)
        {
            continue;
        }
        // For k>=n+1, |k+b| >= k+Re(b). A paired factor
        // (k+|a|)/(k+Re(b)) is bounded by its current value or its
        // limit 1. This also qualifies huge positive denominator parameters
        // without requiring an iteration index larger than |b|.
        let k = (n + 1) as f64;
        let paired = a.len().min(b.len());
        let past = b.iter().all(|x| k + x.v.re > 0.0) && k > last_pole + 1.0;
        let mut ratio = norm(z.v);
        for (x, y) in a.iter().zip(b) {
            ratio *= ((k + norm(x.v)) / (k + y.v.re)).max(1.0);
        }
        if a.len() <= b.len() {
            ratio /= k + 1.0;
            for x in &b[paired..] {
                ratio /= k + x.v.re;
            }
        } else {
            ratio *= ((k + norm(a[paired].v)) / (k + 1.0)).max(1.0);
        }
        // Sensitivities have an additional polynomial/log factor. The factor
        // two reserves its eventual growth; tests cover each selector.
        if past && (0.0..1.0).contains(&ratio) && n > 8 {
            let tail = 2.0 * term.norm() * ratio / (1.0 - ratio);
            if tail <= 2.0 * f64::EPSILON * sum.norm().max(1e-300) {
                return Ok(Estimate::new(
                    sum,
                    tail + 8.0 * f64::EPSILON * magnitude,
                    algorithm_component!("convergent_series"),
                ));
            }
        }
    }
    unreachable_loop()
}
// The unbounded Rust range may theoretically end at usize::MAX; callers cannot
// admit that much work without first exhausting the checked core budget.
fn unreachable_loop<T>() -> Result<T, SciOxideError> {
    Err(ERRORS.unstable())
}

/// The shared scaled two-term continuation handles real and complex coordinates.
fn continue_ode(
    a: &[Jet],
    b: &[Jet],
    target: Jet,
    regularized: bool,
    w: &mut Work<'_, '_, '_>,
) -> Result<Estimate, SciOxideError> {
    super::hypergeometric_complex::continuation(a, b[0], target, regularized, w)
}

pub(crate) fn asymptotic(
    a: Jet,
    b: Jet,
    inverse: Jet,
    w: &mut Work<'_, '_, '_>,
) -> Result<Option<Estimate>, SciOxideError> {
    super::hypergeometric_shared::asymptotic(a, b, inverse, w).map(|result| {
        result
            .map(|(y, error)| Estimate::new(y, error, algorithm_component!("asymptotic_expansion")))
    })
}

pub(crate) fn pfq(
    a: &[Jet],
    b: &[Jet],
    z: Jet,
    regularized: bool,
    w: &mut Work<'_, '_, '_>,
) -> Result<Estimate, SciOxideError> {
    validate(a, b, regularized, scioxide_core::argument!("denominator"))?;
    let logarithmic_identity = a.len() == 2
        && b.len() == 1
        && a.iter().all(|x| x.v == ONE && x.d == ZERO)
        && b[0].v == real(2.0)
        && b[0].d == ZERO;
    let real_valued =
        z.v.im == 0.0 && z.d.im == 0.0 && a.iter().chain(b).all(|x| x.v.im == 0.0 && x.d.im == 0.0);
    if real_valued
        && let Some(kind) =
            crate::hypergeometric_family::hypergeometric_value::Kind::pfq(a.len(), b.len())
    {
        if a.iter().chain(b).all(|x| x.d == ZERO) {
            // Real values and argument derivatives: the binary64 kernels.
            let (numerator, denominator) = (real_parts(a), real_parts(b));
            if let Some(mut e) = crate::hypergeometric_family::hypergeometric_value::estimate(
                kind,
                &numerator[..a.len()],
                &denominator[..b.len()],
                z.v.re,
                regularized,
                z.d != ZERO,
                w,
            )? {
                e.y.d *= z.d;
                e.derivative_error = e.derivative_error.map(|error| error * z.d.re.abs());
                return Ok(e);
            }
        } else if !regularized
            && b.len() == 1
            // Preserve the canonical exact-origin and exponential identities below.
            && z.v != ZERO
            && !(a.len() == 1 && a[0].v == b[0].v && a[0].d == b[0].d)
            && !logarithmic_identity
            && (a.len() < 2 || z.v.re <= 1.0 || a.iter().any(|x| pole(x.v) && x.d == ZERO))
        {
            // A parameter direction: the compensated two-term kernels.
            return crate::hypergeometric_family::hypergeometric_real::evaluate(a, b[0], z, w);
        }
    }
    if regularized && b.iter().all(|x| !pole(x.v)) {
        let mut e = pfq(a, b, z, false, w)?;
        for &x in b {
            e.scale = e.scale - log_gamma(x, w)?;
        }
        return Ok(e);
    }
    let terminating = a.iter().any(|x| pole(x.v) && x.d == ZERO);
    if a.len() > b.len() + 1 && !(a.len() == 2 && b.is_empty()) && !terminating {
        return Err(ERRORS.domain(scioxide_core::argument!("z"), DomainReason::OutsideDomain));
    }
    if z.v == ZERO && !regularized {
        let mut d = Jet::from(1.0);
        if z.d != ZERO {
            for &x in a {
                w.try_step()?;
                d = d * x;
            }
            for &x in b {
                w.try_step()?;
                d = d / x;
            }
        }
        return Ok(Estimate::new(
            Jet {
                v: ONE,
                d: if z.d == ZERO { ZERO } else { d.v * z.d },
            },
            0.0,
            algorithm_component!("zero_identity"),
        ));
    }
    if z.v == ZERO || terminating {
        return series(a, b, z, regularized, w);
    }
    if a.len() == 2 && b.is_empty() {
        return two_f_zero(a[0], a[1], z, w);
    }
    if a.len() == 2 && b.len() == 1 {
        if z.v == ONE {
            let gap = b[0] - a[0] - a[1];
            if gap.v.re <= 0.0 || z.d != ZERO && gap.v.re <= 1.0 {
                return Err(
                    ERRORS.domain(scioxide_core::argument!("z"), DomainReason::OutsideDomain)
                );
            }
            let mut y =
                log_gamma(gap, w)?.exp() * rgamma(b[0] - a[0], w)? * rgamma(b[0] - a[1], w)?;
            if !regularized {
                y = y * log_gamma(b[0], w)?.exp();
            }
            if z.d != ZERO {
                y.d = y.v * a[0].v * a[1].v / (gap.v - ONE);
            }
            return Ok(Estimate::new(
                y,
                64.0 * f64::EPSILON * y.norm(),
                algorithm_component!("gauss_endpoint"),
            ));
        }
        if a[0].v == b[0].v && a[0].d == b[0].d || a[1].v == b[0].v && a[1].d == b[0].d {
            let exponent = if a[0].v == b[0].v && a[0].d == b[0].d {
                a[1]
            } else {
                a[0]
            };
            let mut e = Estimate::new(
                Jet::from(1.0),
                16.0 * f64::EPSILON,
                algorithm_component!("parameter_identity"),
            );
            let mut complement = Jet::from(1.0) - z;
            if z.v.im == 0.0 {
                complement.v.im = -z.v.im;
            }
            e.scale = -exponent * complement.ln();
            if regularized {
                e.y = e.y * rgamma(b[0], w)?;
            }
            return Ok(e);
        }
        if logarithmic_identity && z.v.im == 0.0 && z.v.re < 1.0 {
            let x = z.v.re;
            let value = -<f64 as scioxide_core::RealScalar>::ln_1p(-x) / x;
            let derivative = if x.abs() < 1e-6 {
                0.5 + 2.0 * x / 3.0 + 0.75 * x * x
            } else {
                (1.0 / (1.0 - x) - value) / x
            };
            return Ok(Estimate::new(
                Jet {
                    v: real(value),
                    d: z.d * real(derivative),
                },
                8.0 * f64::EPSILON * value.abs(),
                algorithm_component!("logarithmic_identity"),
            ));
        }
        if (ONE - z.v).norm() <= 0.1 {
            return super::hypergeometric_complex::near_one(a[0], a[1], b[0], z, regularized, w);
        }
        // Reciprocal connection avoids loss of 1/z for remote arguments.
        let difference = a[1] - a[0];
        if z.v.norm() > 4.0 && scioxide_core::sin_pi(difference.v)?.norm() > 0.01 {
            return super::hypergeometric_complex::reciprocal_connection(
                a[0],
                a[1],
                b[0],
                z,
                regularized,
                w,
            );
        }
        if z.v.norm() < 0.7 {
            return series(a, b, z, regularized, w);
        }
        return continue_ode(a, b, z, regularized, w);
    }
    if a.len() == 1 && b.len() == 1 {
        if a[0].v == b[0].v && a[0].d == b[0].d {
            let mut e = Estimate::new(
                Jet::from(1.0),
                0.0,
                algorithm_component!("exponential_identity"),
            );
            e.scale = z;
            if regularized {
                e.y = rgamma(b[0], w)?;
            }
            return Ok(e);
        }
        if z.v.norm() > 40.0 + 2.0 * (a[0].v.norm() + b[0].v.norm())
            && let Some(e) = super::hypergeometric_complex::kummer(a[0], b[0], z, regularized, w)?
        {
            return Ok(e);
        }
        if z.v.re < -8.0 {
            let mut e = pfq(&[b[0] - a[0]], b, -z, regularized, w)?;
            e.scale = e.scale + z;
            e.method = algorithm_component!("kummer_transformation");
            return Ok(e);
        }
        if z.v.norm() < 8.0 || z.v.re >= 0.0 && z.v.im.abs() < 0.5 * z.v.re {
            return series(a, b, z, regularized, w);
        }
        return continue_ode(a, b, z, regularized, w);
    }
    if a.is_empty() && b.len() == 1 {
        if z.v.re < -8.0 || z.v.im.abs() > 12.0 {
            return continue_ode(a, b, z, regularized, w);
        }
        return series(a, b, z, regularized, w);
    }
    if a.len() > b.len() + 1 || a.len() == b.len() + 1 && z.v.norm() >= 1.0 {
        return Err(ERRORS.domain(scioxide_core::argument!("z"), DomainReason::OutsideDomain));
    }
    series(a, b, z, regularized, w)
}

/// Trapezoidal quadrature of the analytic logarithmic U integral on a rotated
/// positive ray. Refinement and both tails are checked independently.
fn u_integral(a: Jet, b: Jet, z: Jet, w: &mut Work<'_, '_, '_>) -> Result<Estimate, SciOxideError> {
    let angle = -z.v.arg();
    let ray = Jet::constant(C::new(0.0, angle));
    let log_norm = -log_gamma(a, w)?;
    // Center the logarithmic integral near its positive-ray saddle. Normalizing
    // before exponentiation preserves values beyond ordinary scalar range.
    let magnitude_z = z.v.norm();
    let linear = magnitude_z - b.v.re + 1.0;
    let discriminant = linear.hypot(2.0 * magnitude_z.sqrt() * a.v.re.sqrt());
    let saddle = if linear >= 0.0 {
        2.0 * a.v.re / (discriminant + linear)
    } else {
        (discriminant - linear) / (2.0 * magnitude_z)
    };
    let center = saddle.ln();
    let log_t = Jet::from(center) + ray;
    let t = log_t.exp();
    let scale =
        -z * t + a * log_t + (b - a - Jet::from(1.0)) * (Jet::from(1.0) + t).ln() + log_norm;
    let mut old = Jet::from(0.0);
    let mut h = 0.5;
    for level in 0.. {
        let mut sum = Jet::from(0.0);
        let mut magnitude = 0.0;
        let mut tail = 0.0;
        for side in [-1.0, 1.0] {
            let mut quiet = 0;
            for n in 0usize.. {
                if side > 0.0 && n == 0 {
                    continue;
                }
                w.try_step()?;
                let s = center + side * n as f64 * h;
                let log_t = Jet::from(s) + ray;
                let t = log_t.exp();
                let exponent = -z * t
                    + a * log_t
                    + (b - a - Jet::from(1.0)) * (Jet::from(1.0) + t).ln()
                    + log_norm
                    - scale;
                let slope = (-z * t + a + (b - a - Jet::from(1.0)) * t / (Jet::from(1.0) + t))
                    .v
                    .re;
                let term = if exponent.v.re < -740.0 {
                    Jet::from(0.0)
                } else {
                    exponent.exp()
                };
                if !term.v.is_finite() || !term.d.is_finite() {
                    return Err(ERRORS.unstable());
                }
                sum = sum + term;
                magnitude += term.norm();
                if n > 8 && side * slope < 0.0 && term.norm() < 1e-18 * sum.norm().max(1e-300) {
                    quiet += 1;
                } else {
                    quiet = 0;
                }
                if quiet >= 4 {
                    tail += term.norm() * h / (a.v.re * h).min(1.0);
                    break;
                }
            }
        }
        sum = sum * Jet::from(h);
        let difference = (sum - old).norm();
        if level > 0 && difference < 2e-14 * sum.norm().max(1e-300) {
            if sum.v == ZERO {
                return Err(ERRORS.unstable());
            }
            let mut e = Estimate::new(
                sum,
                4.0 * difference + tail + 32.0 * f64::EPSILON * h * magnitude,
                algorithm_component!("U_rotated_integral"),
            );
            e.scale = scale;
            return Ok(e);
        }
        old = sum;
        h *= 0.5;
    }
    unreachable_loop()
}

pub(crate) fn hyperu(
    a: Jet,
    b: Jet,
    z: Jet,
    w: &mut Work<'_, '_, '_>,
) -> Result<Estimate, SciOxideError> {
    if pole(a.v) && a.d == ZERO {
        let n = -a.v.re;
        let mut prev = Jet::from(1.0);
        if n == 0.0 {
            return Ok(Estimate::new(
                prev,
                0.0,
                algorithm_component!("U_terminating_polynomial"),
            ));
        }
        let mut y = z - b;
        let mut error = 8.0 * f64::EPSILON * y.norm();
        let mut previous_error = 0.0;
        let mut scale = 0.0;
        for k in 1usize.. {
            if k as f64 >= n {
                let mut result =
                    Estimate::new(y, error, algorithm_component!("U_terminating_polynomial"));
                result.scale = Jet::from(scale);
                return Ok(result);
            }
            w.try_step()?;
            let magnitude = prev.norm().max(y.norm());
            if magnitude > 1e100 {
                prev = prev / Jet::from(magnitude);
                y = y / Jet::from(magnitude);
                error /= magnitude;
                previous_error /= magnitude;
                scale += magnitude.ln();
            }
            let k = Jet::from(k as f64);
            let p = z - b - Jet::from(2.0) * k;
            let q = k * (k + b - Jet::from(1.0));
            let next = p * y - q * prev;
            let next_error = error * p.norm()
                + previous_error * q.norm()
                + 8.0 * f64::EPSILON * (p.norm() * y.norm() + q.norm() * prev.norm());
            previous_error = error;
            error = next_error;
            prev = y;
            y = next;
        }
    }
    if a.d == ZERO
        && b.d == ZERO
        && a.v.im == 0.0
        && b.v.im == 0.0
        && z.v.im == 0.0
        && z.d.im == 0.0
        && z.v.re >= 0.0
        && let Some(mut e) = crate::hypergeometric_family::hypergeometric_value::estimate(
            crate::hypergeometric_family::hypergeometric_value::Kind::U,
            &[a.v.re],
            &[b.v.re],
            z.v.re,
            false,
            z.d != ZERO,
            w,
        )?
    {
        e.y.d *= z.d;
        e.derivative_error = e.derivative_error.map(|error| error * z.d.re.abs());
        return Ok(e);
    }
    if z.v == ZERO {
        if b.v.re >= 1.0 {
            return Err(ERRORS.domain(scioxide_core::argument!("z"), DomainReason::OutsideDomain));
        }
        let y = log_gamma(Jet::from(1.0) - b, w)?.exp() * rgamma(Jet::from(1.0) + a - b, w)?;
        if z.d != ZERO {
            return Err(ERRORS.domain(scioxide_core::argument!("z"), DomainReason::OutsideDomain));
        }
        return Ok(Estimate::new(
            y,
            32.0 * f64::EPSILON * y.norm(),
            algorithm_component!("U_origin_limit"),
        ));
    }
    if a.v == ZERO && a.d == ZERO {
        return Ok(Estimate::new(
            Jet::from(1.0),
            0.0,
            algorithm_component!("U_zero_parameter"),
        ));
    }
    if z.v.norm() > 40.0 + 2.0 * a.v.norm() * (Jet::from(1.0) + a - b).v.norm()
        && let Some(mut e) = asymptotic(a, Jet::from(1.0) + a - b, -Jet::from(1.0) / z, w)?
    {
        e.scale = -a * z.ln();
        return Ok(e);
    }
    if z.v.re <= 0.0 {
        let center = C::new(
            1.0,
            if z.v.im.is_sign_negative() {
                -0.25
            } else {
                0.25
            },
        );
        let first = hyperu(a, b, Jet::constant(center), w)?;
        let second = hyperu(
            a + Jet::from(1.0),
            b + Jet::from(1.0),
            Jet::constant(center),
            w,
        )?;
        let y = first.y * first.scale.exp();
        let dy = -a * second.y * second.scale.exp();
        let mut e = super::hypergeometric_complex::propagate(
            &[a],
            &[b],
            z,
            center,
            y,
            dy,
            first.error * first.scale.v.re.exp()
                + second.error * second.scale.v.re.exp() * a.norm(),
            w,
        )?;
        e.method = algorithm_component!("U_taylor_continuation");
        return Ok(e);
    }
    // Shift into a well-decaying defining integral, then use the contiguous
    // relation downward. Differentiate the recurrence, including at a=-n.
    let shifts = (1.0 - a.v.re).ceil().max(0.0);
    let n = shifts.try_to_usize_exact()?;
    let aa = a + Jet::from(shifts);
    let first = u_integral(aa, b, z, w)?;
    if n == 0 {
        return Ok(first);
    }
    let second = u_integral(aa + Jet::from(1.0), b, z, w)?;
    let scale = Jet::constant(real(first.scale.v.re.max(second.scale.v.re)));
    let p1 = (first.scale - scale).exp();
    let p2 = (second.scale - scale).exp();
    let mut next = first.y * p1;
    let mut next2 = second.y * p2;
    let mut error = first.error * p1.norm() + second.error * p2.norm();
    for k in (0..n).rev() {
        w.try_step()?;
        let x = a + Jet::from(f64::try_from_usize_exact(k)?);
        let p = z + Jet::from(2.0) * x + Jet::from(2.0) - b;
        let q = (x + Jet::from(1.0)) * (x + Jet::from(2.0) - b);
        let y = p * next - q * next2;
        error = error * (p.norm() + q.norm())
            + 16.0 * f64::EPSILON * (p.norm() * next.norm() + q.norm() * next2.norm());
        next2 = next;
        next = y;
    }
    let mut e = Estimate::new(next, error, algorithm_component!("U_contiguous_recurrence"));
    e.scale = scale;
    Ok(e)
}
fn two_f_zero(a: Jet, b: Jet, z: Jet, w: &mut Work<'_, '_, '_>) -> Result<Estimate, SciOxideError> {
    let mut inverse = -Jet::from(1.0) / z;
    if z.v.im == 0.0 {
        inverse.v.im = z.v.im;
    }
    let mut e = hyperu(a, Jet::from(1.0) + a - b, inverse, w)?;
    e.scale = e.scale + a * inverse.ln();
    e.method = algorithm_component!("2F0_U_connection");
    Ok(e)
}

impl super::hypergeometric_shared::Arithmetic for Jet {
    const ROUNDING: f64 = f64::EPSILON;
    const TAIL: f64 = 2.0 * f64::EPSILON;
    fn constant(z: C) -> Self {
        Self::constant(z)
    }
    fn point(self) -> C {
        self.v
    }
    fn value_norm(self) -> f64 {
        norm(self.v)
    }
    fn direction_norm(self) -> f64 {
        norm(self.d)
    }
    fn finite(self) -> bool {
        self.v.is_finite() && self.d.is_finite()
    }
    fn scale(self, x: f64) -> Self {
        self * Self::from(x)
    }
    fn small(self, other: Self, factor: f64) -> bool {
        self.norm() <= factor * other.norm().max(1e-300)
    }
}
