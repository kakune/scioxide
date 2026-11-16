//! Compensated positive-radius Coulomb kernels (DLMF 33.6, 33.8, 33.23).
use super::kernel::State;
use crate::support::COULOMB_ERRORS as ERRORS;
use crate::support::Work;
use crate::support::wide::Wide as D;
use crate::support::wide::project;
use crate::support::{C, real};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)] // std-linked tests can supply the inherent float methods.
use scioxide_core::RealScalar;
use scioxide_core::SciOxideError;

// Three successive small terms guard accidental zeros in the recurrences.
// Two-term updates retain about 100 significant bits; these tails leave
// guard digits before the final binary64 projection.
const EPS: f64 = 1e-30;
const ONE: D = D { hi: 1.0, lo: 0.0 };

#[derive(Clone, Copy)]
struct Radial {
    y: D,
    d: D,
    log: D,
    error: f64,
}
impl Radial {
    fn normalize(&mut self) -> Result<(), SciOxideError> {
        let magnitude = self.y.hi.abs().max(self.d.hi.abs());
        if !magnitude.is_finite() || !self.log.hi.is_finite() || !self.error.is_finite() {
            return Err(ERRORS.unstable());
        }
        if magnitude > 1e100 || (magnitude > 0.0 && magnitude < 1e-100) {
            // A power of two rescales both components without rounding.
            let shift = if magnitude > 1.0 { 256 } else { -256 };
            let factor = 2.0_f64.powi(-shift);
            self.y = self.y.mul_f64(factor);
            self.d = self.d.mul_f64(factor);
            self.error *= factor;
            self.log = self.log + crate::support::constants::LN_2.mul_f64(f64::from(shift));
        }
        Ok(())
    }
    fn finish(self) -> State {
        let y = project(self.y, self.log);
        let d = project(self.d, self.log);
        let materialized = y.is_finite()
            && d.is_finite()
            && (y != 0.0 || self.y.hi == 0.0)
            && (d != 0.0 || self.d.hi == 0.0);
        if materialized {
            State {
                y: real(y),
                d: real(d),
                log: 0.0,
                log_lo: 0.0,
                error: project(D::from(self.error), self.log).abs()
                    + 8.0 * f64::EPSILON * (y.abs() + d.abs()),
            }
        } else {
            State {
                y: real(self.y.value()),
                d: real(self.d.value()),
                log: self.log.hi,
                log_lo: self.log.lo,
                error: self.error + 8.0 * f64::EPSILON * (self.y.hi.abs() + self.d.hi.abs()),
            }
        }
    }
}
use crate::support::wide::square_root as sqrt;

/// DLMF 33.8.1 evaluated by homogeneous numerator/denominator continuants.
/// Their common power-of-two scaling preserves the sign of the regular solution.
fn cf1(l: D, eta: D, x: D, w: &mut Work<'_, '_, '_>) -> Result<(D, f64), SciOxideError> {
    let mut numerators = [ONE, (l + ONE) / x + eta / (l + ONE)];
    let mut denominators = [D::ZERO, ONE];
    let mut previous = numerators[1];
    let mut stable = 0;
    let mut order = l + ONE;
    loop {
        w.try_step()?;
        let charge = eta / order;
        let numerator = -(ONE + charge * charge);
        let denominator = (order.mul_f64(2.0) + ONE) * (ONE / x + charge / (order + ONE));
        let top = denominator * numerators[1] + numerator * numerators[0];
        let bottom = denominator * denominators[1] + numerator * denominators[0];
        numerators = [numerators[1], top];
        denominators = [denominators[1], bottom];
        if !top.hi.is_finite() || !bottom.hi.is_finite() {
            return Err(ERRORS.unstable());
        }
        if bottom.hi != 0.0 {
            let value = top / bottom;
            if (value - previous).hi.abs() <= EPS * value.hi.abs().max(1.0) {
                stable += 1;
                if stable == 3 {
                    return Ok((value, bottom.hi.signum()));
                }
            } else {
                stable = 0;
            }
            previous = value;
        }
        let magnitude = numerators
            .iter()
            .chain(&denominators)
            .fold(0.0_f64, |v, z| v.max(z.hi.abs()));
        let scale = if magnitude > 1e100 {
            2.0_f64.powi(-256)
        } else if magnitude < 1e-100 {
            2.0_f64.powi(256)
        } else {
            1.0
        };
        for value in numerators.iter_mut().chain(&mut denominators) {
            *value = value.mul_f64(scale);
        }
        let next = order + ONE;
        if next.hi == order.hi {
            return Err(ERRORS.unstable());
        }
        order = next;
    }
}

/// Two-component complex arithmetic for the denominator of DLMF 33.8.2.
#[derive(Clone, Copy)]
struct FractionComplex {
    re: D,
    im: D,
}
impl FractionComplex {
    const ONE: Self = Self {
        re: ONE,
        im: D::ZERO,
    };
    const ZERO: Self = Self {
        re: D::ZERO,
        im: D::ZERO,
    };
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
    fn reciprocal(self) -> Result<Self, SciOxideError> {
        let scale = self.re.hi.abs().max(self.im.hi.abs());
        if !scale.is_finite() || scale == 0.0 {
            return Err(ERRORS.unstable());
        }
        let a = self.re.div_f64(scale);
        let b = self.im.div_f64(scale);
        let norm = a * a + b * b;
        Ok(Self {
            re: (a / norm).div_f64(scale),
            im: (-b / norm).div_f64(scale),
        })
    }
}

/// Modified Lentz evaluation of the complex continued fraction in DLMF 33.8.2.
fn cf2(l: D, eta: D, x: D, w: &mut Work<'_, '_, '_>) -> Result<(D, D, f64), SciOxideError> {
    let factor = |n: D| FractionComplex {
        re: n * (n - ONE) - l * (l + ONE) - eta * eta,
        im: eta * (n.mul_f64(2.0) - ONE),
    };
    let denominator = |n: D| FractionComplex {
        re: (x - eta).mul_f64(2.0),
        im: n.mul_f64(2.0),
    };
    let first = factor(ONE);
    let mut value = denominator(ONE);
    let mut c = value;
    let mut d = FractionComplex::ZERO;
    let mut n = ONE;
    let mut stable = 0;
    loop {
        w.try_step()?;
        n = n + ONE;
        let a = factor(n);
        let b = denominator(n);
        d = b.add(a.mul(d)).reciprocal()?;
        c = b.add(a.mul(c.reciprocal()?));
        let ratio = c.mul(d);
        value = value.mul(ratio);
        if !value.re.hi.is_finite() || !value.im.hi.is_finite() {
            return Err(ERRORS.unstable());
        }
        if (ratio.re - FractionComplex::ONE.re).hi.abs() + ratio.im.hi.abs() <= EPS {
            stable += 1;
            if stable == 3 {
                let quotient = first.mul(value.reciprocal()?);
                return Ok((-quotient.im / x, ONE - eta / x + quotient.re / x, 0.0));
            }
        } else {
            stable = 0;
        }
    }
}
fn steed(
    l: f64,
    eta: f64,
    x: f64,
    w: &mut Work<'_, '_, '_>,
) -> Result<(Radial, Radial), SciOxideError> {
    let (u, sign) = cf1(D::from(l), D::from(eta), D::from(x), w)?;
    let (p, q, rounding) = cf2(D::from(l), D::from(eta), D::from(x), w)?;
    if q.hi <= 64.0 * f64::EPSILON * p.hi.abs() {
        return Err(ERRORS.unstable());
    }
    let alpha = u - p;
    let f = sqrt(q / (alpha * alpha + q * q)).mul_f64(sign);
    let fp = u * f;
    let g = alpha / q * f;
    let gp = p * g - q * f;
    let state = |y: D, d: D| Radial {
        y,
        d,
        log: D::ZERO,
        error: (EPS + rounding) * (y.hi.abs() + d.hi.abs()),
    };
    Ok((state(f, fp), state(g, gp)))
}

fn propagate(
    l: f64,
    eta: f64,
    mut center: f64,
    target: f64,
    mut s: Radial,
    w: &mut Work<'_, '_, '_>,
) -> Result<Radial, SciOxideError> {
    let l = D::from(l);
    let eta = D::from(eta);
    let angular = l * (l + ONE);
    while center != target {
        w.try_step()?;
        let c = D::from(center);
        let c2 = c * c;
        let potential = ONE - eta.mul_f64(2.0) / c - angular / c2;
        let radius = (0.5 * center).min(4.0 / (1.0 + potential.hi.abs().sqrt()));
        let delta = target - center;
        let next = if delta.abs() <= radius {
            target
        } else {
            center + radius.copysign(delta)
        };
        if next == center {
            return Err(ERRORS.unstable());
        }
        // The exact represented endpoint displacement, including subtraction's residual.
        let h = D::from(next) - c;
        let ratio = h / c;
        let ratio2 = ratio * ratio;
        let p = c2 - eta.mul_f64(2.0) * c - angular;
        let q = (c - eta).mul_f64(2.0);
        let p2 = p * ratio2;
        let q3 = q * ratio2 * h;
        let h4 = ratio2 * h * h;
        let inverse_h = ONE / h;
        let mut terms = [D::ZERO, D::ZERO, s.y, h * s.d];
        let mut y = s.y + h * s.d;
        let mut d = s.d;
        let mut magnitude = y.hi.abs() + d.hi.abs();
        let mut small = 0;
        for n in 0usize.. {
            w.try_step()?;
            let nf = n as f64;
            let term = -(ratio.mul_f64(2.0 * nf * (nf + 1.0)) * terms[3]
                + (ratio2.mul_f64(nf * (nf - 1.0)) + p2) * terms[2]
                + q3 * terms[1]
                + h4 * terms[0])
                .div_f64((nf + 2.0) * (nf + 1.0));
            let dt = (term * inverse_h).mul_f64(nf + 2.0);
            y = y + term;
            d = d + dt;
            magnitude += term.hi.abs() + dt.hi.abs();
            if term.hi.abs() <= EPS * y.hi.abs() && dt.hi.abs() <= EPS * d.hi.abs() {
                small += 1;
            } else {
                small = 0;
            }
            if small == 3 {
                break;
            }
            if !y.hi.is_finite() || !d.hi.is_finite() {
                return Err(ERRORS.unstable());
            }
            terms = [terms[1], terms[2], terms[3], term];
        }
        let growth = (y.hi.abs() + d.hi.abs()) / (s.y.hi.abs() + s.d.hi.abs()).max(1e-300);
        s.error = s.error * growth + 8.0 * EPS * magnitude;
        s.y = y;
        s.d = d;
        s.normalize()?;
        center = next;
    }
    Ok(s)
}
fn regular_state(
    l: f64,
    eta: f64,
    x: f64,
    w: &mut Work<'_, '_, '_>,
) -> Result<Radial, SciOxideError> {
    // The series majorant is below exp(rho + 2 sqrt(2|eta| rho)). Its largest
    // radius with exponent 16 bounds cancellation by 2^23, leaving more than
    // 70 of the two-term sum's bits; outward continuation covers the rest.
    let limit = 8.0 * core::f64::consts::SQRT_2 / ((eta.abs() + 8.0).sqrt() + eta.abs().sqrt());
    let start = x.min(limit * limit);
    let z = D::from(start);
    let order = D::from(l) + ONE;
    let mut previous = D::ZERO;
    let mut term = ONE;
    let mut sum = ONE;
    let mut derivative = order;
    let (mut magnitude, mut derivative_magnitude) = (1.0, order.hi.abs());
    let mut small = 0;
    for n in 1usize.. {
        w.try_step()?;
        let nf = n as f64;
        let next = (D::from(eta).mul_f64(2.0) * z * term - z * z * previous)
            / (D::from(nf) * (D::from(nf + 1.0) + D::from(l).mul_f64(2.0)));
        previous = term;
        term = next;
        sum = sum + term;
        let dt = term * (D::from(nf) + order);
        derivative = derivative + dt;
        magnitude += term.hi.abs();
        derivative_magnitude += dt.hi.abs();
        if term.hi.abs() <= EPS * magnitude && dt.hi.abs() <= EPS * derivative_magnitude {
            small += 1;
        } else {
            small = 0;
        }
        if small == 3 {
            break;
        }
        if !sum.hi.is_finite() {
            return Err(ERRORS.unstable());
        }
    }
    // The Frobenius scale is arbitrary: normalize with the independently
    // anchored irregular solution, after stable outward continuation when the
    // series starts short of x.
    let seed = Radial {
        y: sum,
        d: derivative / z,
        log: order * logarithm(z, w)?,
        error: EPS * (magnitude + derivative_magnitude / start),
    };
    propagate(l, eta, start, x, seed, w)
}

pub(super) fn regular(
    l: f64,
    eta: f64,
    x: f64,
    w: &mut Work<'_, '_, '_>,
) -> Result<State, SciOxideError> {
    if l == 0.0 && eta == 0.0 {
        return Ok(waves(l, eta, x, w)?.0);
    }
    let mut f = regular_state(l, eta, x, w)?;
    let norm = normalization(l, eta, w)?;
    f.log = f.log + norm.log;
    f.error += norm.error * (f.y.hi.abs() + f.d.hi.abs());
    Ok(f.finish())
}
pub(super) fn waves(
    l: f64,
    eta: f64,
    x: f64,
    w: &mut Work<'_, '_, '_>,
) -> Result<(State, State), SciOxideError> {
    if l == 0.0 && eta == 0.0 {
        w.try_step()?;
        let (f, g) = x.sin_cos();
        let state = |y, d| State {
            y: real(y),
            d: real(d),
            log: 0.0,
            log_lo: 0.0,
            error: 2.0 * f64::EPSILON * (f.abs() + g.abs()),
        };
        return Ok((state(f, g), state(g, -f)));
    }
    if let Some(pair) = outgoing(l, eta, x, w)? {
        return Ok(pair);
    }
    // Steed's fractions converge on the allowed side of the turning point
    // eta + sqrt(eta^2 + l(l+1)) < eta + hypot(eta, l + 1/2). Below it, G is
    // anchored one unit outside and continued inward, where it dominates.
    let anchor = x.max(eta + eta.hypot(l + 0.5) + 1.0);
    let (f, g) = steed(l, eta, anchor, w)?;
    if anchor == x {
        return Ok((f.finish(), g.finish()));
    }
    let g = propagate(l, eta, anchor, x, g, w)?;
    let mut f = regular_state(l, eta, x, w)?;
    let wronskian = f.d * g.y - f.y * g.d;
    if wronskian.hi <= 0.0 || !wronskian.hi.is_finite() {
        return Err(ERRORS.unstable());
    }
    f.y = f.y / wronskian;
    f.d = f.d / wronskian;
    f.error /= wronskian.hi;
    f.log = -g.log;
    Ok((f.finish(), g.finish()))
}

// The shared two-term logarithm uses at most 24 atanh terms. Charge its
// maximum work before entering the bounded, allocation-free primitive.
fn logarithm(x: D, w: &mut Work<'_, '_, '_>) -> Result<D, SciOxideError> {
    for _ in 0..24 {
        w.try_step()?;
    }
    Ok(x.ln())
}

use crate::support::constants::FRAC_PI_2 as PI_2;
use crate::support::constants::LN_2;
type Pair = (D, D);
fn product(a: Pair, b: Pair) -> Pair {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
fn phase(angle: D, w: &mut Work<'_, '_, '_>) -> Result<Pair, SciOxideError> {
    let (sine, cosine) = crate::support::wide::sin_cos_with_step(angle, &mut || w.try_step())?;
    Ok((cosine, sine))
}

fn atan(mut x: D, w: &mut Work<'_, '_, '_>) -> Result<D, SciOxideError> {
    let negative = x.hi < 0.0;
    if negative {
        x = -x;
    }
    let reciprocal = x.hi > 1.0;
    if reciprocal {
        x = ONE / x;
    }
    let shifted = x.hi > 0.414;
    if shifted {
        x = (x - ONE) / (x + ONE);
    }
    let square = -x * x;
    let mut term = x;
    let mut sum = x;
    for n in 1..=45 {
        w.try_step()?;
        term = term * square;
        let delta = term.div_f64(f64::from(2 * n + 1));
        sum = sum + delta;
        if delta.hi.abs() <= EPS * sum.hi.abs() {
            break;
        }
    }
    if shifted {
        sum = sum + PI_2.mul_f64(0.5);
    }
    if reciprocal {
        sum = PI_2 - sum;
    }
    Ok(if negative { -sum } else { sum })
}

/// Stirling phase at Re(z)>=32, divided by the recurrence product's phase.
/// The shared B_(2k)/(2k(2k-1)) tail leaves < 5e-32 at |z|=32.
fn gamma_seed(l: f64, eta: f64, w: &mut Work<'_, '_, '_>) -> Result<(D, D, Pair), SciOxideError> {
    if eta == 0.0 {
        for _ in 0..128 {
            w.try_step()?;
        }
        return Ok((
            D::ZERO,
            crate::gamma_beta::gamma::real::ln_positive_wide(D::from(l) + ONE),
            (ONE, D::ZERO),
        ));
    }
    let mut a = D::from(l) + ONE;
    let b = D::from(eta);
    let mut recurrence = (ONE, D::ZERO);
    let mut recurrence_scale = D::ZERO;
    while a.hi < 32.0 {
        w.try_step()?;
        let scale = a.hi.max(eta.abs());
        if scale > 1e100 {
            recurrence = product(recurrence, (a.div_f64(scale), b.div_f64(scale)));
            recurrence_scale = recurrence_scale + logarithm(D::from(scale), w)?;
        } else {
            recurrence = product(recurrence, (a, b));
        }
        if recurrence.0.hi.abs().max(recurrence.1.hi.abs()) > 1e100 {
            let factor = 2.0_f64.powi(-256);
            recurrence = (recurrence.0.mul_f64(factor), recurrence.1.mul_f64(factor));
            recurrence_scale = recurrence_scale + LN_2.mul_f64(256.0);
        }
        a = a + ONE;
    }
    let scale = a.hi.max(eta.abs());
    let (ar, br) = (a.div_f64(scale), b.div_f64(scale));
    let norm = ar * ar + br * br;
    let log_abs = logarithm(D::from(scale), w)? + logarithm(norm, w)?.mul_f64(0.5);
    let argument = atan(b / a, w)?;
    let inverse = ((ar / norm).div_f64(scale), (-br / norm).div_f64(scale));
    let square = product(inverse, inverse);
    let correction = crate::support::gamma_series::stirling(
        inverse,
        square,
        12,
        (D::ZERO, D::ZERO),
        |sum, square, [hi, lo, _]| {
            w.try_step()?;
            let sum = product(sum, square);
            Ok::<_, SciOxideError>((sum.0 + D { hi, lo }, sum.1))
        },
        |a, b| Ok(product(a, b)),
    )?;
    let sigma = b * (log_abs - ONE) + (a - D::from(0.5)) * argument + correction.1;
    let magnitude = sqrt(recurrence.0 * recurrence.0 + recurrence.1 * recurrence.1);
    let log_gamma = (a - D::from(0.5)) * log_abs - b * argument - a
        + crate::support::constants::LOG_SQRT_TWO_PI
        + correction.0
        - logarithm(magnitude, w)?
        - recurrence_scale;
    Ok((
        sigma,
        log_gamma,
        (recurrence.0 / magnitude, -recurrence.1 / magnitude),
    ))
}

/// One compensated normalization for scalar values, origin limits and wave seeds.
pub(super) fn normalization(
    l: f64,
    eta: f64,
    w: &mut Work<'_, '_, '_>,
) -> Result<super::kernel::Normalization, SciOxideError> {
    let (sigma, gamma, recurrence) = gamma_seed(l, eta, w)?;
    for _ in 0..128 {
        w.try_step()?;
    }
    let denominator =
        crate::gamma_beta::gamma::real::ln_positive_wide((D::from(l) + ONE).mul_f64(2.0));
    let a = LN_2.mul_f64(l);
    let b = PI_2.mul_f64(eta);
    let log = a - b + gamma - denominator;
    let carrier = product(phase(sigma, w)?, recurrence);
    // Include the Stirling remainder, compensated recurrence/log rounding and
    // cancellation of the original terms. The final IEEE projection is separate.
    let magnitude =
        1.0 + a.hi.abs() + b.hi.abs() + gamma.hi.abs() + denominator.hi.abs() + sigma.hi.abs();
    let error = 512.0 * EPS * magnitude;
    if !log.hi.is_finite() || !error.is_finite() {
        return Err(ERRORS.unstable());
    }
    Ok(super::kernel::Normalization {
        log,
        phase: C::new(carrier.0.value(), carrier.1.value()),
        error,
    })
}

/// The outgoing expansion H+ = e^(iθ) Σ (l+1+iη)_n (−l+iη)_n / (n! (2ix)^n)
/// (DLMF 33.11.1), accepted by its own computed tails rather than a fixed
/// radius. Terms first grow by about (l²+η²)/(2nx) per step, so the sum is
/// accepted only when its value/derivative tails converge before the least
/// term and the two-word rounding exposure of the growth stays below one
/// `f64` unit of the result. Otherwise `None` hands off to Steed.
fn outgoing(
    l: f64,
    eta: f64,
    x: f64,
    w: &mut Work<'_, '_, '_>,
) -> Result<Option<(State, State)>, SciOxideError> {
    // The least term, near n = 2x, is about exp(-2x): smaller radii cannot
    // reach the `EPS` tail.
    if x < 32.0 {
        return Ok(None);
    }
    // Two-word products and sums round by at most a few 2^-104 each. Term n
    // inherits about 8n such roundings from the running product; every
    // update also rounds the partial sums.
    let unit = 16.0 * f64::EPSILON * f64::EPSILON;
    let mut term = (ONE, D::ZERO);
    let mut sum = term;
    let mut derivative = (D::ZERO, D::ZERO);
    let mut small = 0;
    let mut previous = 1.0;
    let mut decreasing = false;
    let mut exposure = 1.0;
    let mut tail = 0.0;
    for n in 1usize.. {
        w.try_step()?;
        let n = n as f64;
        let a = (D::from(l) + D::from(n), D::from(eta));
        let b = (D::from(n - 1.0) - D::from(l), D::from(eta));
        let coefficient = product(a, b);
        let factor = (
            coefficient.1.div_f64(2.0 * n).div_f64(x),
            -coefficient.0.div_f64(2.0 * n).div_f64(x),
        );
        term = product(term, factor);
        sum = (sum.0 + term.0, sum.1 + term.1);
        derivative = (
            derivative.0 - term.0.mul_f64(n).div_f64(x),
            derivative.1 - term.1.mul_f64(n).div_f64(x),
        );
        let size = term.0.hi.abs() + term.1.hi.abs();
        let derivative_size = size * n / x;
        let amplitude = sum.0.hi.abs() + sum.1.hi.abs();
        tail = size + derivative_size;
        exposure += (n + 1.0) * tail + amplitude + derivative.0.hi.abs() + derivative.1.hi.abs();
        if !exposure.is_finite() || decreasing && size > previous {
            return Ok(None);
        }
        decreasing |= size < previous;
        // Past the largest term, once terms halve, the remaining value tail is
        // at most about the current term. Reject growth whose rounding already
        // exceeds what the final test could accept, without summing the rest.
        if decreasing
            && size <= 0.5 * previous
            && unit * exposure > f64::EPSILON * (amplitude + 2.0 * size)
        {
            return Ok(None);
        }
        if tail <= EPS * amplitude {
            small += 1;
        } else {
            small = 0;
        }
        if small == 3 {
            break;
        }
        previous = size;
    }
    let rounding = unit * exposure;
    if rounding > f64::EPSILON * (sum.0.hi.abs() + sum.1.hi.abs()) {
        return Ok(None);
    }
    // Delay the phase until the expansion has proved applicable, so a rejected
    // candidate does not pay for gamma/logarithm reduction.
    let (sigma, _, recurrence) = gamma_seed(l, eta, w)?;
    let theta = sigma - PI_2.mul_f64(l) - (LN_2 + logarithm(D::from(x), w)?).mul_f64(eta);
    let carrier = if x <= 1e14 {
        phase(D::from(x) + theta, w)?
    } else {
        product(phase(D::from(x), w)?, phase(theta, w)?)
    };
    let carrier = product(carrier, recurrence);
    let phase_derivative = ONE - D::from(eta) / D::from(x);
    derivative = (
        derivative.0 - phase_derivative * sum.1,
        derivative.1 + phase_derivative * sum.0,
    );
    // The two-word phase, including quarter-turn reduction of x, rotates a
    // unimodular carrier: its error scales the summed amplitudes.
    let amplitude = sum.0.hi.abs() + sum.1.hi.abs() + derivative.0.hi.abs() + derivative.1.hi.abs();
    let carrier_error = 512.0 * EPS * (1.0 + x + sigma.hi.abs() + theta.hi.abs()) * amplitude;
    let y = product(sum, carrier);
    let d = product(derivative, carrier);
    let state = |y: D, d: D| {
        Radial {
            y,
            d,
            log: D::ZERO,
            error: 4.0 * tail + rounding + carrier_error + 8.0 * EPS * (y.hi.abs() + d.hi.abs()),
        }
        .finish()
    };
    Ok(Some((state(y.1, d.1), state(y.0, d.0))))
}
