//! Two-term complex coordinates for cancellation-sensitive hypergeometric kernels.
use super::hypergeometric_shared::{self as shared, Arithmetic, Propagated};
use crate::support::wide::Wide as D;
use crate::support::{C, HYPERGEOMETRIC_ERRORS as ERRORS, Jet, JetEstimate as Estimate, Work};
use core::ops::{Add, Div, Mul, Neg, Sub};
#[allow(unused_imports)]
use scioxide_core::RealScalar;
use scioxide_core::{SciOxideError, algorithm_component};

#[derive(Clone, Copy, Debug)]
struct Pair {
    re: D,
    im: D,
}
impl Pair {
    const ZERO: Self = Self {
        re: D::ZERO,
        im: D::ZERO,
    };
    fn from(z: C) -> Self {
        Self {
            re: D::from(z.re),
            im: D::from(z.im),
        }
    }
    fn real(x: f64) -> Self {
        Self::from(C::new(x, 0.0))
    }
    fn value(self) -> C {
        let component = |x: D| {
            if x.hi == 0.0 && x.lo == 0.0 {
                x.hi
            } else {
                x.value()
            }
        };
        C::new(component(self.re), component(self.im))
    }
    fn norm(self) -> f64 {
        self.re.hi.hypot(self.im.hi)
    }
    fn zero(self) -> bool {
        self.re.hi == 0.0 && self.re.lo == 0.0 && self.im.hi == 0.0 && self.im.lo == 0.0
    }
    fn scale(self, x: f64) -> Self {
        Self {
            re: self.re.mul_f64(x),
            im: self.im.mul_f64(x),
        }
    }
    fn finite(self) -> bool {
        [self.re.hi, self.re.lo, self.im.hi, self.im.lo]
            .iter()
            .all(|x| x.is_finite())
    }
    fn ln(self) -> Self {
        let pi = crate::support::constants::PI;
        let x = if self.re.hi < 0.0 { -self.re } else { self.re };
        let y = if self.im.hi < 0.0 { -self.im } else { self.im };
        let swap = y.hi > x.hi;
        let ratio = if swap { x / y } else { y / x };
        let reduced = ratio.hi > core::f64::consts::SQRT_2 - 1.0;
        let t = if reduced {
            (ratio - D::from(1.0)) / (ratio + D::from(1.0))
        } else {
            ratio
        };
        let square = t * t;
        let mut term = t;
        let mut angle = t;
        for k in 1..48 {
            term = term * (-square);
            angle = angle + term.div_f64(f64::from(2 * k + 1));
        }
        if reduced {
            angle = angle + pi.mul_f64(0.25);
        }
        if swap {
            angle = pi.mul_f64(0.5) - angle;
        }
        if self.re.hi < 0.0 {
            angle = pi - angle;
        }
        if self.im.hi.is_sign_negative() {
            angle = -angle;
        }
        let magnitude = if swap { y.ln() } else { x.ln() } + (ratio * ratio).ln_1p().mul_f64(0.5);
        Self {
            re: magnitude,
            im: angle,
        }
    }
    fn exp(self) -> Self {
        let (sin, cos) = self.im.sin_cos();
        let magnitude = self.re.exp_wide();
        Self {
            re: magnitude * cos,
            im: magnitude * sin,
        }
    }
}
impl Add for Pair {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self {
            re: self.re + b.re,
            im: self.im + b.im,
        }
    }
}
impl Sub for Pair {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        self + (-b)
    }
}
impl Neg for Pair {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}
impl Mul for Pair {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self {
            re: self.re * b.re - self.im * b.im,
            im: self.re * b.im + self.im * b.re,
        }
    }
}
impl Div for Pair {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        let (a, b) = if b.re.hi.abs().max(b.im.hi.abs()) > f64::MAX * 0.25 {
            (self.scale(0.5), b.scale(0.5))
        } else {
            (self, b)
        };
        if b.re.hi.abs() >= b.im.hi.abs() {
            let ratio = b.im / b.re;
            let denominator = b.re + b.im * ratio;
            Self {
                re: (a.re + a.im * ratio) / denominator,
                im: (a.im - a.re * ratio) / denominator,
            }
        } else {
            let ratio = b.re / b.im;
            let denominator = b.im + b.re * ratio;
            Self {
                re: (a.re * ratio + a.im) / denominator,
                im: (a.im * ratio - a.re) / denominator,
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Q {
    v: Pair,
    d: Pair,
}
impl From<f64> for Q {
    fn from(x: f64) -> Self {
        Self {
            v: Pair::real(x),
            d: Pair::ZERO,
        }
    }
}
impl From<Jet> for Q {
    fn from(x: Jet) -> Self {
        Self {
            v: Pair::from(x.v),
            d: Pair::from(x.d),
        }
    }
}
impl Arithmetic for Q {
    fn constant(value: C) -> Self {
        Self {
            v: Pair::from(value),
            d: Pair::ZERO,
        }
    }
    fn point(self) -> C {
        self.v.value()
    }
    fn value_norm(self) -> f64 {
        self.v.norm()
    }
    fn direction_norm(self) -> f64 {
        self.d.norm()
    }
    fn finite(self) -> bool {
        self.v.finite() && self.d.finite()
    }
    fn scale(self, x: f64) -> Self {
        Self {
            v: self.v.scale(x),
            d: self.d.scale(x),
        }
    }
}
impl Q {
    fn ln(self) -> Self {
        Self {
            v: self.v.ln(),
            d: if self.d.zero() {
                Pair::ZERO
            } else {
                self.d / self.v
            },
        }
    }
    fn exp(self) -> Self {
        let v = self.v.exp();
        Self {
            v,
            d: if self.d.zero() {
                Pair::ZERO
            } else {
                self.d * v
            },
        }
    }
    fn jet(self) -> Jet {
        Jet {
            v: self.v.value(),
            d: self.d.value(),
        }
    }
    fn pole(self) -> bool {
        let z = self.point();
        z.im == 0.0
            && self.v.im.lo == 0.0
            && self.v.re.lo == 0.0
            && z.re <= 0.0
            && z.re.fract() == 0.0
    }
}
impl Add for Q {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self {
            v: self.v + b.v,
            d: self.d + b.d,
        }
    }
}
impl Sub for Q {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self {
            v: self.v - b.v,
            d: self.d - b.d,
        }
    }
}
impl Neg for Q {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            v: -self.v,
            d: -self.d,
        }
    }
}
impl Mul for Q {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self {
            v: self.v * b.v,
            d: self.d * b.v + self.v * b.d,
        }
    }
}
impl Div for Q {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        let v = self.v / b.v;
        Self {
            v,
            d: (self.d - v * b.d) / b.v,
        }
    }
}

/// Gamma's shared Bernoulli correction and its derivative. The outer pair
/// differentiates the same Horner evaluation; each Q keeps the caller's
/// analytic direction, so digamma sensitivities need no coefficient copy.
fn stirling_correction(x: Q, w: &mut Work<'_, '_, '_>) -> Result<(Q, Q), SciOxideError> {
    let inverse = Q::from(1.0) / x;
    let square = inverse * inverse;
    crate::support::gamma_series::stirling(
        (inverse, -square),
        (square, -(square * inverse).scale(2.0)),
        16,
        (Q::from(0.0), Q::from(0.0)),
        |(sum, derivative), (square, dsquare), [hi, lo, _]| {
            w.try_step()?;
            let coefficient = Q {
                v: Pair {
                    re: D { hi, lo },
                    im: D::ZERO,
                },
                d: Pair::ZERO,
            };
            Ok((
                sum * square + coefficient,
                derivative * square + sum * dsquare,
            ))
        },
        |(inverse, dinverse), (sum, derivative)| {
            Ok((inverse * sum, dinverse * sum + inverse * derivative))
        },
    )
}

/// Stirling's expansion after recurrence into Re(z)>=32. The shared
/// Bernoulli coefficients keep two terms; its remainder is <1e-32
/// for this half-plane. Recurrence logarithms preserve analytic directions.
fn log_gamma(z: Q, w: &mut Work<'_, '_, '_>) -> Result<Q, SciOxideError> {
    if z.pole() {
        return Err(ERRORS.unstable());
    }
    let mut x = z;
    let mut correction = Q::from(0.0);
    while x.point().re < 32.0 {
        w.try_step()?;
        correction = correction + x.ln();
        x = x + Q::from(1.0);
    }
    w.steps(48)?;
    let correction_series = stirling_correction(x, w)?.0;
    let constant = Q {
        v: Pair {
            re: crate::support::constants::LOG_SQRT_TWO_PI,
            im: D::ZERO,
        },
        d: Pair::ZERO,
    };
    Ok((x - Q::from(0.5)) * x.ln() - x + constant + correction_series - correction)
}
fn rgamma(z: Q, w: &mut Work<'_, '_, '_>) -> Result<Q, SciOxideError> {
    if z.pole() {
        if z.d.zero() {
            return Ok(Q::from(0.0));
        }
        let n = -z.point().re;
        let factorial = log_gamma(Q::from(n + 1.0), w)?.exp();
        return Ok(Q {
            v: Pair::ZERO,
            d: z.d * factorial.v.scale(if n % 2.0 == 0.0 { 1.0 } else { -1.0 }),
        });
    }
    Ok((-log_gamma(z, w)?).exp())
}

#[derive(Clone, Copy)]
struct State {
    y: Q,
    log: Q,
    error: f64,
}
impl State {
    fn finish(self, method: scioxide_core::AlgorithmComponent) -> Result<Estimate, SciOxideError> {
        if !self.y.finite() || !self.log.finite() || !self.error.is_finite() {
            return Err(ERRORS.unstable());
        }
        let mut e = Estimate::new(
            self.y.jet(),
            self.error + 8.0 * f64::EPSILON * self.y.value_norm(),
            method,
        );
        e.scale = self.log.jet();
        Ok(e)
    }
}
fn combine(a: State, b: State) -> State {
    let shift = if a.y.value_norm() == 0.0 {
        b.log.point().re
    } else if b.y.value_norm() == 0.0 {
        a.log.point().re
    } else {
        a.log.point().re.max(b.log.point().re)
    };
    let log = Q::from(shift);
    let p = if a.y.value_norm() == 0.0 && a.y.direction_norm() == 0.0 {
        Q::from(0.0)
    } else {
        (a.log - log).exp()
    };
    let q = if b.y.value_norm() == 0.0 && b.y.direction_norm() == 0.0 {
        Q::from(0.0)
    } else {
        (b.log - log).exp()
    };
    State {
        y: a.y * p + b.y * q,
        log,
        error: a.error * p.value_norm()
            + b.error * q.value_norm()
            + 32e-30 * (a.y.value_norm() * p.value_norm() + b.y.value_norm() * q.value_norm()),
    }
}

/// Local series in two-term coordinates; all callers use |z|<=0.3.
fn series(a: &[Q], b: Q, z: Q, w: &mut Work<'_, '_, '_>) -> Result<(Q, f64), SciOxideError> {
    let mut term = Q::from(1.0);
    let mut sum = term;
    let mut magnitude = 1.0;
    let mut quiet = 0;
    for n in 0usize.. {
        w.try_step()?;
        let k = Q::from(n as f64);
        term = term * z / ((b + k) * Q::from((n + 1) as f64));
        for &x in a {
            w.try_step()?;
            term = term * (x + k);
        }
        sum = sum + term;
        magnitude += term.value_norm();
        if !sum.finite() {
            return Err(ERRORS.unstable());
        }
        quiet = if term.small(sum, 1e-30) { quiet + 1 } else { 0 };
        if quiet == 3 {
            return Ok((sum, 4.0 * term.value_norm() + 32e-30 * magnitude));
        }
    }
    Err(ERRORS.unstable())
}

pub(super) fn continuation(
    a: &[Jet],
    b: Jet,
    target: Jet,
    regularized: bool,
    w: &mut Work<'_, '_, '_>,
) -> Result<Estimate, SciOxideError> {
    let aa = core::array::from_fn::<_, 2, _>(|i| Q::from(a.get(i).copied().unwrap_or_default()));
    let a = &aa[..a.len()];
    let b = Q::from(b);
    let target = Q::from(target);
    let center = if target.point().re < 0.0 {
        C::new(-0.25, 0.125_f64.copysign(target.point().im))
    } else {
        C::new(0.25, 0.125_f64.copysign(target.point().im))
    };
    let seed = Q::constant(center);
    let (mut y, mut error) = series(a, b, seed, w)?;
    let shifted = core::array::from_fn::<_, 2, _>(|i| {
        a.get(i).copied().unwrap_or(Q::from(0.0)) + Q::from(1.0)
    });
    let prefactor = a.iter().fold(Q::from(1.0) / b, |p, &x| p * x);
    let (dy, de) = series(&shifted[..a.len()], b + Q::from(1.0), seed, w)?;
    let mut slope = dy * prefactor;
    let mut slope_error = de * prefactor.value_norm();
    if regularized {
        let factor = rgamma(b, w)?;
        y = y * factor;
        slope = slope * factor;
        error *= factor.value_norm();
        slope_error *= factor.value_norm();
    }
    let state = shared::propagate(
        a,
        b,
        target,
        center,
        Propagated {
            y,
            slope,
            error,
            slope_error,
            log: 0.0,
        },
        w,
    )?;
    State {
        y: state.y,
        log: Q::from(state.log),
        error: state.error,
    }
    .finish(algorithm_component!("taylor_continuation"))
}

/// Both contributions of DLMF 13.7.2. Choosing -z for the algebraic
/// logarithm makes the side of the phase explicit, including signed zero.
pub(super) fn kummer(
    a: Jet,
    b: Jet,
    z: Jet,
    regularized: bool,
    w: &mut Work<'_, '_, '_>,
) -> Result<Option<Estimate>, SciOxideError> {
    let (a, b, z) = (Q::from(a), Q::from(b), Q::from(z));
    let one = Q::from(1.0);
    let Some((exponential, e1)) = shared::asymptotic(b - a, one - a, one / z, w)? else {
        return Ok(None);
    };
    let Some((algebraic, e2)) = shared::asymptotic(a, one + a - b, -one / z, w)? else {
        return Ok(None);
    };
    let common = if regularized {
        Q::from(0.0)
    } else {
        log_gamma(b, w)?
    };
    let first = rgamma(a, w)?;
    let second = rgamma(b - a, w)?;
    let result = combine(
        State {
            y: first * exponential,
            log: z + (a - b) * z.ln() + common,
            error: e1 * first.value_norm(),
        },
        State {
            y: second * algebraic,
            log: -a * (-z).ln() + common,
            error: e2 * second.value_norm(),
        },
    );
    result
        .finish(algorithm_component!("kummer_two_contributions"))
        .map(Some)
}

/// Noninteger 1-z connection in two-term arithmetic.
fn connection(
    a: Q,
    b: Q,
    c: Q,
    z: Q,
    regularized: bool,
    w: &mut Work<'_, '_, '_>,
) -> Result<State, SciOxideError> {
    let one = Q::from(1.0);
    let gap = c - a - b;
    let t = complement(z);
    let (s1, e1) = series(&[a, b], one - gap, t, w)?;
    let (s2, e2) = series(&[c - a, c - b], one + gap, t, w)?;
    let common = if regularized {
        Q::from(0.0)
    } else {
        log_gamma(c, w)?
    };
    let p = rgamma(c - a, w)? * rgamma(c - b, w)?;
    let q = rgamma(a, w)? * rgamma(b, w)?;
    Ok(combine(
        State {
            y: p * s1,
            log: common + log_gamma(gap, w)?,
            error: e1 * p.value_norm(),
        },
        State {
            y: q * s2,
            log: common + log_gamma(-gap, w)? + gap * t.ln(),
            error: e2 * q.value_norm(),
        },
    ))
}

pub(super) fn near_one(
    a: Jet,
    b: Jet,
    c: Jet,
    z: Jet,
    regularized: bool,
    w: &mut Work<'_, '_, '_>,
) -> Result<Estimate, SciOxideError> {
    let (a, b, c, z) = (Q::from(a), Q::from(b), Q::from(c), Q::from(z));
    let gap = c - a - b;
    let integer = gap.point().im == 0.0
        && gap.v.im.lo == 0.0
        && gap.v.re.lo == 0.0
        && gap.point().re.fract() == 0.0;
    let near_integer =
        (gap.point() - C::new(gap.point().re.round(), 0.0)).norm() < f64::EPSILON.sqrt();
    if near_integer && (!integer || !gap.d.zero()) {
        // Almost-coalescing connection coefficients magnify arithmetic errors,
        // quadratically for parameter directions. The exact integral limit is
        // analytic below; nearby gaps use the common two-term ODE engine.
        return continuation(&[a.jet(), b.jet()], c.jet(), z.jet(), regularized, w);
    }
    let result = if integer && gap.d.zero() {
        let m = gap.point().re.abs();
        let m = m.try_to_usize_exact()?;
        if gap.point().re < 0.0 {
            let mut state = integer_connection(c - a, c - b, c, z, m, regularized, w)?;
            state.log = state.log + gap * complement(z).ln();
            state
        } else {
            integer_connection(a, b, c, z, m, regularized, w)?
        }
    } else {
        connection(a, b, c, z, regularized, w)?
    };
    result.finish(algorithm_component!("gauss_complement_connection"))
}

/// Continue a supplied value/slope pair (also used by Tricomi U).
#[allow(clippy::too_many_arguments)]
pub(super) fn propagate(
    a: &[Jet],
    b: &[Jet],
    target: Jet,
    center: C,
    y: Jet,
    slope: Jet,
    error: f64,
    w: &mut Work<'_, '_, '_>,
) -> Result<Estimate, SciOxideError> {
    let aa = core::array::from_fn::<_, 2, _>(|i| Q::from(a.get(i).copied().unwrap_or_default()));
    let state = shared::propagate(
        &aa[..a.len()],
        Q::from(b[0]),
        Q::from(target),
        center,
        Propagated {
            y: Q::from(y),
            slope: Q::from(slope),
            error,
            slope_error: error,
            log: 0.0,
        },
        w,
    )?;
    State {
        y: state.y,
        log: Q::from(state.log),
        error: state.error,
    }
    .finish(algorithm_component!("taylor_continuation"))
}

fn digamma(z: Q, w: &mut Work<'_, '_, '_>) -> Result<Q, SciOxideError> {
    let mut x = z;
    let mut correction = Q::from(0.0);
    while x.point().re < 32.0 {
        w.try_step()?;
        correction = correction - Q::from(1.0) / x;
        x = x + Q::from(1.0);
    }
    w.steps(32)?;
    let inverse = Q::from(1.0) / x;
    let derivative_correction = stirling_correction(x, w)?.1;
    Ok(x.ln() - inverse.scale(0.5) + derivative_correction + correction)
}

/// The analytic logarithmic limit (DLMF 15.8.10), after Euler reflection
/// when the integral gap is negative. No finite parameter differences enter.
fn integer_connection(
    a: Q,
    b: Q,
    c: Q,
    z: Q,
    m: usize,
    regularized: bool,
    w: &mut Work<'_, '_, '_>,
) -> Result<State, SciOxideError> {
    let one = Q::from(1.0);
    let mf = Q::from(m as f64);
    let t = complement(z);
    let common = if regularized {
        Q::from(0.0)
    } else {
        log_gamma(c, w)?
    };
    let finite = if m == 0 {
        State {
            y: Q::from(0.0),
            log: Q::from(0.0),
            error: 0.0,
        }
    } else {
        let mut term = one;
        let mut sum = one;
        let mut magnitude = 1.0;
        for n in 0..m - 1 {
            w.try_step()?;
            let k = Q::from(f64::try_from_usize_exact(n)?);
            term = term * (a + k) * (b + k) * t
                / ((one - mf + k) * Q::from(f64::try_from_usize_exact(n + 1)?));
            sum = sum + term;
            magnitude += term.value_norm();
        }
        State {
            y: sum * rgamma(a + mf, w)? * rgamma(b + mf, w)?,
            log: common + log_gamma(mf, w)?,
            error: 32e-30 * magnitude,
        }
    };
    let mut term = one;
    let mut bracket = digamma(one, w)? + digamma(one + mf, w)?
        - digamma(a + mf, w)?
        - digamma(b + mf, w)?
        - t.ln();
    let mut sum = bracket;
    let mut magnitude = sum.value_norm();
    let mut quiet = 0;
    for n in 0usize.. {
        w.try_step()?;
        let k = Q::from(n as f64);
        let next = Q::from((n + 1) as f64);
        term = term * (a + mf + k) * (b + mf + k) * t / (next * (next + mf));
        bracket =
            bracket + one / next + one / (next + mf) - one / (a + mf + k) - one / (b + mf + k);
        let addition = term * bracket;
        sum = sum + addition;
        magnitude += addition.value_norm();
        quiet = if addition.small(sum, 1e-30) {
            quiet + 1
        } else {
            0
        };
        if !sum.finite() {
            return Err(ERRORS.unstable());
        }
        if quiet == 3 {
            let prefactor = rgamma(a, w)? * rgamma(b, w)?;
            let sign = if m.is_multiple_of(2) { 1.0 } else { -1.0 };
            let log = if m == 0 {
                common
            } else {
                common + mf * t.ln() - log_gamma(one + mf, w)?
            };
            return Ok(combine(
                finite,
                State {
                    y: sum * prefactor.scale(sign),
                    log,
                    error: (4.0 * addition.value_norm() + 256e-30 * magnitude)
                        * prefactor.value_norm(),
                },
            ));
        }
    }
    Err(ERRORS.unstable())
}
fn complement(z: Q) -> Q {
    let mut t = Q::from(1.0) - z;
    if z.v.im.hi == 0.0 && z.v.im.lo == 0.0 {
        t.v.im = D::from(-z.v.im.hi);
    }
    t
}

impl shared::ConnectionArithmetic for Q {
    fn logarithm(self, w: &mut Work<'_, '_, '_>) -> Result<Self, SciOxideError> {
        w.steps(48)?;
        Ok(self.ln())
    }
    fn exponential(self) -> Self {
        self.exp()
    }
}
pub(super) fn reciprocal_connection(
    a: Jet,
    b: Jet,
    c: Jet,
    z: Jet,
    regularized: bool,
    w: &mut Work<'_, '_, '_>,
) -> Result<Estimate, SciOxideError> {
    let (a, b, c, z) = (Q::from(a), Q::from(b), Q::from(c), Q::from(z));
    let result = shared::reciprocal_connection(
        [a, b],
        c,
        z,
        regularized,
        w,
        |a, b, z, w| {
            let (y, error) = series(a, b, z, w)?;
            Ok(shared::Contribution {
                y,
                log: Q::from(0.0),
                error,
            })
        },
        |a, b, w| {
            let mut log = Q::from(0.0);
            let mut y = Q::from(1.0);
            for &x in a {
                log = log + log_gamma(x, w)?;
            }
            for &x in b {
                if x.pole() {
                    y = y * rgamma(x, w)?;
                } else {
                    log = log - log_gamma(x, w)?;
                }
            }
            Ok(shared::Contribution { y, log, error: 0.0 })
        },
    )?;
    State {
        y: result.y,
        log: result.log,
        error: result.error,
    }
    .finish(algorithm_component!("reciprocal_connection"))
}
