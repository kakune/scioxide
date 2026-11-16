//! Compensated real 0F1/1F1/2F1 series and differential-equation continuation. DLMF 13.2, 13.7,
//! 13.29, 15.8 and 15.10.
//!
//! Two-term arithmetic throughout, with one analytic direction carried
//! alongside every value. It serves real parameter sensitivities, and the
//! value regions the `f64` kernels (`hypergeometric_value`) hand over:
//! continuation, the reciprocal and Gauss-endpoint connections, and the 1F1
//! asymptotic expansions.
use crate::support::HYPERGEOMETRIC_ERRORS as ERRORS;
use crate::support::Jet;
use crate::support::JetEstimate as Estimate;
use crate::support::Work;
use crate::support::real;
use crate::support::wide::Wide as D;
use crate::support::wide::project;
use core::ops::{Add, Div, Mul, Neg, Sub};
#[allow(unused_imports)] // std-linked tests can supply the inherent float methods.
use scioxide_core::RealScalar;
use scioxide_core::{AlgorithmComponent, algorithm_component};
use scioxide_core::{DomainReason, SciOxideError};

// Two-term arithmetic has approximately 100 useful bits. The tail target
// retains guard digits for cancellation before the final scalar projection.
const EPS: f64 = 1e-30;
use crate::support::constants::LN_2;
const ONE: R = R {
    v: D { hi: 1.0, lo: 0.0 },
    d: D::ZERO,
};
const ZERO: R = R {
    v: D::ZERO,
    d: D::ZERO,
};

/// One analytic direction, including parameter sensitivities at termination.
#[derive(Clone, Copy)]
struct R {
    v: D,
    d: D,
}
impl From<f64> for R {
    fn from(v: f64) -> Self {
        Self {
            v: D::from(v),
            d: D::ZERO,
        }
    }
}
impl From<Jet> for R {
    fn from(x: Jet) -> Self {
        Self {
            v: D::from(x.v.re),
            d: D::from(x.d.re),
        }
    }
}
impl R {
    fn finite(self) -> bool {
        self.v.hi.is_finite()
            && self.v.lo.is_finite()
            && self.d.hi.is_finite()
            && self.d.lo.is_finite()
    }
    fn constant(self) -> bool {
        self.d.hi == 0.0 && self.d.lo == 0.0
    }
    fn zero(self) -> bool {
        self.v.hi == 0.0 && self.v.lo == 0.0 && self.constant()
    }
    fn small_relative_to(self, other: Self, factor: f64) -> bool {
        self.v.hi.abs() <= factor * other.v.hi.abs() && self.d.hi.abs() <= factor * other.d.hi.abs()
    }
    fn mul_f64(self, x: f64) -> Self {
        Self {
            v: self.v.mul_f64(x),
            d: self.d.mul_f64(x),
        }
    }
    fn ln(self, w: &mut Work<'_, '_, '_>) -> Result<Self, SciOxideError> {
        w.steps(24)?;
        Ok(Self {
            v: self.v.ln(),
            d: if self.constant() {
                D::ZERO
            } else {
                self.d / self.v
            },
        })
    }
    fn exp(self) -> Self {
        let value = D::from(self.v.exp());
        Self {
            v: value,
            d: if self.constant() {
                D::ZERO
            } else {
                self.d * value
            },
        }
    }
}
impl Add for R {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self {
            v: self.v + b.v,
            d: self.d + b.d,
        }
    }
}
impl Sub for R {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self {
            v: self.v - b.v,
            d: self.d - b.d,
        }
    }
}
impl Neg for R {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            v: -self.v,
            d: -self.d,
        }
    }
}
impl Mul for R {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self {
            v: self.v * b.v,
            d: if self.constant() && b.constant() {
                D::ZERO
            } else {
                self.d * b.v + self.v * b.d
            },
        }
    }
}
impl Div for R {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        let v = self.v / b.v;
        Self {
            v,
            d: if self.constant() && b.constant() {
                D::ZERO
            } else {
                (self.d - v * b.d) / b.v
            },
        }
    }
}

#[derive(Clone, Copy)]
struct State {
    y: R,
    log: R,
    error: f64,
    method: AlgorithmComponent,
}
impl State {
    fn new(y: R, error: f64, method: AlgorithmComponent) -> Self {
        Self {
            y,
            log: ZERO,
            error,
            method,
        }
    }
    fn finish(self) -> Result<Estimate, SciOxideError> {
        if !self.y.v.hi.is_finite()
            || !self.y.v.lo.is_finite()
            || !self.log.v.hi.is_finite()
            || !self.log.v.lo.is_finite()
            || !self.error.is_finite()
        {
            return Err(ERRORS.unstable());
        }
        let value = project(self.y.v, self.log.v);
        let derivative = project(self.y.d + self.y.v * self.log.d, self.log.v);
        // Preserve ordinary values independently of derivative range. Reporting
        // an unavailable derivative must not destroy an accurate finite value.
        let materialized = value.is_finite() && (value != 0.0 || self.y.v.hi == 0.0);
        let derivative_available = derivative.is_finite()
            && (derivative != 0.0 || (self.y.d + self.y.v * self.log.d).hi == 0.0);
        let (y, scale, error) = if materialized {
            (
                Jet {
                    v: real(value),
                    d: real(if derivative_available {
                        derivative
                    } else {
                        0.0
                    }),
                },
                Jet::constant(real(0.0)),
                project(D::from(self.error), self.log.v).abs() + 4.0 * f64::EPSILON * value.abs(),
            )
        } else {
            (
                Jet {
                    v: real(self.y.v.value()),
                    d: real(self.y.d.value()),
                },
                Jet {
                    v: real(self.log.v.value()),
                    d: real(self.log.d.value()),
                },
                self.error + 4.0 * f64::EPSILON * self.y.v.hi.abs(),
            )
        };
        Ok(Estimate::from(crate::support::JetData {
            y,
            scale,
            error,
            method: self.method,
            derivative_error: if materialized && !derivative_available {
                Some(f64::INFINITY)
            } else {
                None
            },
        }))
    }
}

fn pole(x: R) -> bool {
    x.v.lo == 0.0 && x.v.hi <= 0.0 && x.v.hi.fract() == 0.0
}

/// Every term and parameter update is charged to the same parent work budget.
fn series(a: &[R], b: R, z: R, w: &mut Work<'_, '_, '_>) -> Result<State, SciOxideError> {
    let end = a
        .iter()
        .filter(|x| pole(**x) && x.constant())
        .map(|x| -x.v.hi)
        .fold(f64::INFINITY, f64::min);
    let mut term = ONE;
    let mut sum = term;
    let mut magnitude = 1.0;
    let mut log = ZERO;
    for n in 0usize.. {
        if n as f64 == end {
            return Ok(State {
                log,
                ..State::new(
                    sum,
                    8.0 * EPS * magnitude,
                    algorithm_component!("real_terminating_polynomial"),
                )
            });
        }
        w.try_step()?;
        let k = R::from(n as f64);
        term = term * z / R::from((n + 1) as f64);
        for &x in a {
            w.try_step()?;
            term = term * (x + k);
        }
        w.try_step()?;
        term = term / (b + k);
        sum = sum + term;
        magnitude += term.v.hi.abs();
        if !sum.finite() || !term.finite() || !magnitude.is_finite() {
            return Err(ERRORS.unstable());
        }
        if magnitude > 1e100 {
            let factor = 2.0_f64.powi(-256);
            sum = sum.mul_f64(factor);
            term = term.mul_f64(factor);
            magnitude *= factor;
            log.v = log.v + LN_2.mul_f64(256.0);
        }
        if n < 8 || !term.small_relative_to(sum, EPS) {
            continue;
        }
        // The same eventual ratio majorant as the generalized series. It must
        // hold past all denominator sign changes; derivative tails reserve a
        // factor two for their additional polynomial/logarithmic growth.
        let k = (n + 1) as f64;
        if k + b.v.hi <= 0.0 {
            continue;
        }
        let mut ratio = z.v.hi.abs();
        if let Some(first) = a.first() {
            ratio *= ((k + first.v.hi.abs()) / (k + b.v.hi)).max(1.0);
        } else {
            ratio /= k + b.v.hi;
        }
        if a.len() < 2 {
            ratio /= k + 1.0;
        } else {
            ratio *= ((k + a[1].v.hi.abs()) / (k + 1.0)).max(1.0);
        }
        if !(0.0..1.0).contains(&ratio) {
            continue;
        }
        let tail = term.mul_f64(2.0 * ratio / (1.0 - ratio));
        if tail.small_relative_to(sum, EPS) {
            let value_tail = 2.0 * term.v.hi.abs() * ratio / (1.0 - ratio);
            return Ok(State {
                log,
                ..State::new(
                    sum,
                    value_tail + 8.0 * EPS * magnitude,
                    algorithm_component!("real_compensated_series"),
                )
            });
        }
    }
    Err(ERRORS.unstable())
}

impl super::hypergeometric_shared::Arithmetic for R {
    fn constant(z: crate::support::C) -> Self {
        Self::from(z.re)
    }
    fn point(self) -> crate::support::C {
        real(self.v.value())
    }
    fn value_norm(self) -> f64 {
        self.v.hi.abs()
    }
    fn direction_norm(self) -> f64 {
        self.d.hi.abs()
    }
    fn finite(self) -> bool {
        self.finite()
    }
    fn scale(self, x: f64) -> Self {
        self.mul_f64(x)
    }
}

impl super::hypergeometric_shared::ConnectionArithmetic for R {
    fn logarithm(self, w: &mut Work<'_, '_, '_>) -> Result<Self, SciOxideError> {
        self.ln(w)
    }
    fn exponential(self) -> Self {
        self.exp()
    }
}

fn continuation(
    a: &[R],
    b: R,
    target: R,
    w: &mut Work<'_, '_, '_>,
) -> Result<State, SciOxideError> {
    let center = 0.25_f64.copysign(target.v.hi);
    let seed = R::from(center);
    let initial = series(a, b, seed, w)?;
    let mut shifted = [ZERO; 2];
    let mut prefactor = ONE / b;
    for (dest, &x) in shifted.iter_mut().zip(a) {
        *dest = x + ONE;
        prefactor = prefactor * x;
    }
    let derivative = series(&shifted[..a.len()], b + ONE, seed, w)?;
    let state = super::hypergeometric_shared::propagate(
        a,
        b,
        target,
        real(center),
        super::hypergeometric_shared::Propagated {
            y: initial.y,
            slope: derivative.y * prefactor,
            error: initial.error,
            slope_error: derivative.error * prefactor.v.hi.abs(),
            log: 0.0,
        },
        w,
    )?;
    Ok(State {
        y: state.y,
        log: initial.log + R::from(state.log),
        error: state.error,
        method: algorithm_component!("real_taylor_continuation"),
    })
}

fn log_gamma(x: R, w: &mut Work<'_, '_, '_>) -> Result<(f64, R), SciOxideError> {
    if pole(x) {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "hypergeometric continuation encountered a singular gamma factor",
        ));
    }
    // Reserve 256 units: at most 31 recurrence factors, 40 coefficients,
    // two bounded logarithms, reflection's two logarithms and the bounded
    // digamma recurrence/rational evaluation. Admit all nested work first.
    w.steps(256)?;
    let (sign, v) = if x.v.hi > 0.0 {
        (1.0, crate::support::gamma_real::ln_positive_wide(x.v))
    } else {
        let pi = crate::support::constants::PI;
        let sine =
            D::from(crate::util::sin_pi(x.v.hi)) + pi.mul_f64(x.v.lo * crate::util::cos_pi(x.v.hi));
        let sign = sine.hi.signum();
        (
            sign,
            pi.ln()
                - sine.mul_f64(sign).ln()
                - crate::support::gamma_real::ln_positive_wide(D::from(1.0) - x.v),
        )
    };
    let d = if x.constant() {
        D::ZERO
    } else {
        x.d.mul_f64(crate::support::gamma_core::digamma_kernel(x.v.hi)?)
    };
    Ok((sign, R { v, d }))
}

fn gamma_ratio(
    numerator: &[R],
    denominator: &[R],
    w: &mut Work<'_, '_, '_>,
) -> Result<State, SciOxideError> {
    if let ([upper], [lower]) = (numerator, denominator) {
        let shift = *upper - *lower;
        if shift.constant()
            && shift.v.lo == 0.0
            && shift.v.hi.fract() == 0.0
            && shift.v.hi.abs() <= 32.0
            && !pole(*upper)
            && !pole(*lower)
        {
            let mut product = ONE;
            let start = if shift.v.hi >= 0.0 { *lower } else { *upper };
            for k in 0..shift.v.hi.abs() as usize {
                w.try_step()?;
                product = product * (start + R::from(k as f64));
            }
            let y = if shift.v.hi >= 0.0 {
                product
            } else {
                ONE / product
            };
            if y.finite() && !y.zero() {
                return Ok(State::new(
                    y,
                    8.0 * EPS * y.v.hi.abs(),
                    algorithm_component!("real_gamma_ratio_recurrence"),
                ));
            }
        }
    }
    // A denominator gamma pole is an analytic zero of its reciprocal. The
    // parameter derivative there follows 1/Gamma(-n+e)=(-1)^n*n!*e+O(e^2).
    let mut result = State::new(ONE, 0.0, algorithm_component!("real_gamma_ratio"));
    for &x in numerator {
        let (sign, log) = log_gamma(x, w)?;
        result.y = result.y.mul_f64(sign);
        result.log = result.log + log;
    }
    for &x in denominator {
        if pole(x) {
            let mut zero = ZERO;
            if !x.constant() {
                let (_, factorial) = log_gamma(ONE - x, w)?;
                let sign = if (-x.v.hi) % 2.0 == 0.0 { 1.0 } else { -1.0 };
                zero.d = x.d.mul_f64(sign);
                result.log = result.log + factorial;
            }
            result.y = result.y * zero;
        } else {
            let (sign, log) = log_gamma(x, w)?;
            result.y = result.y.mul_f64(sign);
            result.log = result.log - log;
        }
    }
    Ok(result)
}

/// Digamma with two-word recurrence and the differentiated Stirling series.
/// At x >= 32 the first omitted term is below 2e-35.
fn digamma_wide(mut x: D, w: &mut Work<'_, '_, '_>) -> Result<D, SciOxideError> {
    let one = D::from(1.0);
    let mut correction = D::ZERO;
    while x.hi < 32.0 {
        w.try_step()?;
        correction = correction - one / x;
        x = x + one;
    }
    let inverse = one / x;
    let square = inverse * inverse;
    let mut power = square;
    let mut sum = D::ZERO;
    for (index, (numerator, denominator)) in [
        (1.0, 6.0),
        (-1.0, 30.0),
        (1.0, 42.0),
        (-1.0, 30.0),
        (5.0, 66.0),
        (-691.0, 2730.0),
        (7.0, 6.0),
        (-3617.0, 510.0),
        (43867.0, 798.0),
        (-174611.0, 330.0),
    ]
    .into_iter()
    .enumerate()
    {
        w.try_step()?;
        sum = sum + power * (D::from(numerator) / D::from(denominator * (2 * index + 2) as f64));
        power = power * square;
    }
    Ok(x.ln() - inverse.mul_f64(0.5) - sum + correction)
}

/// Entire pair (1/Gamma(x), psi(x)/Gamma(x)), including gamma poles.
fn reciprocal_gamma_pair(x: D, w: &mut Work<'_, '_, '_>) -> Result<(D, D), SciOxideError> {
    let r = R { v: x, d: D::ZERO };
    if pole(r) {
        let (_, factorial) = log_gamma(R::from(1.0 - x.hi), w)?;
        let sign = if (-x.hi) % 2.0 == 0.0 { -1.0 } else { 1.0 };
        return Ok((D::ZERO, factorial.v.exp_wide().mul_f64(sign)));
    }
    let (sign, logarithm) = log_gamma(r, w)?;
    let reciprocal = (-logarithm.v).exp_wide().mul_f64(sign);
    Ok((reciprocal, reciprocal * digamma_wide(x, w)?))
}

/// Integer-gap reciprocal connection, DLMF 15.8.8. The logarithmic
/// coefficient is propagated as a product, so reciprocal-gamma zeros cancel
/// digamma poles analytically, even when a series coefficient becomes zero.
pub(super) fn integer_reciprocal(
    mut a: D,
    mut b: D,
    c: D,
    z: f64,
    w: &mut Work<'_, '_, '_>,
) -> Result<Estimate, SciOxideError> {
    if b.hi < a.hi {
        core::mem::swap(&mut a, &mut b);
    }
    let gap = b - a;
    let m = gap.hi.try_to_usize_exact()?;
    let one = D::from(1.0);
    let mf = D::from(m as f64);
    let logarithm = D::from(-z).ln();
    let constant = |v| R { v, d: D::ZERO };
    let mut finite = State::new(ZERO, 0.0, algorithm_component!("real_integer_reciprocal"));
    if m > 0 {
        let mut term = one;
        let mut sum = term;
        let mut magnitude = 1.0;
        for n in 0..m - 1 {
            w.try_step()?;
            let k = D::from(f64::try_from_usize_exact(n)?);
            term = term * (a + k) * (one - c + a + k)
                / ((one - mf + k) * D::from(f64::try_from_usize_exact(n + 1)?) * D::from(z));
            sum = sum + term;
            magnitude += term.hi.abs();
        }
        finite = gamma_ratio(
            &[constant(c), constant(mf)],
            &[constant(b), constant(c - a)],
            w,
        )?;
        finite.error = 32.0 * EPS * magnitude;
        finite.y = finite.y * constant(sum);
        finite.log = finite.log - constant(a * logarithm);
    }
    let x = c - b;
    let (mut term, psi_product) = reciprocal_gamma_pair(x, w)?;
    let bracket =
        logarithm + digamma_wide(one, w)? + digamma_wide(one + mf, w)? - digamma_wide(b, w)?;
    let mut product = term * bracket - psi_product;
    let mut sum = product;
    let mut magnitude = product.hi.abs();
    let mut quiet = 0;
    for n in 0usize.. {
        w.try_step()?;
        let k = D::from(n as f64);
        let next = D::from((n + 1) as f64);
        let upper = b + k;
        let lower = one - x + k;
        let denominator = next * (next + mf) * D::from(z);
        let ratio = upper * lower / denominator;
        product = ratio * product
            + term * (upper * lower * (one / next + one / (next + mf)) - upper - lower)
                / denominator;
        term = term * ratio;
        sum = sum + product;
        magnitude += product.hi.abs();
        if !sum.hi.is_finite() || !magnitude.is_finite() {
            return Err(ERRORS.unstable());
        }
        quiet = if product.hi.abs() <= EPS * sum.hi.abs() {
            quiet + 1
        } else {
            0
        };
        if quiet >= 3 && n as f64 > (-b.hi).max(x.hi).max(0.0) {
            let mut infinite = gamma_ratio(&[constant(c)], &[constant(a), constant(one + mf)], w)?;
            infinite.log = infinite.log - constant(b * logarithm);
            let sign = if m.is_multiple_of(2) { 1.0 } else { -1.0 };
            infinite.error = 4.0 * product.hi.abs() + 256.0 * EPS * magnitude;
            infinite.y = infinite.y * constant(sum.mul_f64(sign));
            let scale =
                if infinite.y.zero() || !finite.y.zero() && finite.log.v.hi > infinite.log.v.hi {
                    finite.log
                } else {
                    infinite.log
                };
            let left = if finite.y.zero() {
                D::ZERO
            } else {
                (finite.log.v - scale.v).exp_wide()
            };
            let right = if infinite.y.zero() {
                D::ZERO
            } else {
                (infinite.log.v - scale.v).exp_wide()
            };
            return State {
                y: constant(finite.y.v * left + infinite.y.v * right),
                log: scale,
                error: finite.error * left.hi.abs()
                    + infinite.error * right.hi.abs()
                    + 64.0
                        * EPS
                        * (finite.y.v.hi.abs() * left.hi.abs()
                            + infinite.y.v.hi.abs() * right.hi.abs()),
                method: algorithm_component!("real_integer_reciprocal"),
            }
            .finish();
        }
    }
    Err(ERRORS.unstable())
}

fn asymptotic(
    a: R,
    b: R,
    z: R,
    exponential: bool,
    w: &mut Work<'_, '_, '_>,
) -> Result<Option<State>, SciOxideError> {
    if let Some((sum, expansion_error)) =
        super::hypergeometric_shared::asymptotic(b - a, ONE - a, ONE / z, w)?
    {
        let mut state = gamma_ratio(&[b], &[a], w)?;
        let logarithm = z.ln(w)?;
        state.log = state.log + (a - b) * logarithm;
        if exponential {
            state.log = state.log + z;
        }
        let prefactor = state.y.v.hi.abs();
        state.error = state.error * sum.v.hi.abs() + expansion_error * prefactor;
        state.y = state.y * sum;
        // On the positive ray the real algebraic contribution has the
        // multiplier cos(pi*a). Check its estimated size as well as the
        // exponential-series remainder, including after Kummer scaling.
        let cosine = R {
            v: D::from(crate::util::cos_pi(a.v.hi)),
            d: a.d
                .mul_f64(-core::f64::consts::PI * crate::util::sin_pi(a.v.hi)),
        };
        if !cosine.zero() {
            let ratio = (a.v.hi * (1.0 + a.v.hi - b.v.hi) / z.v.hi).abs();
            if ratio > 0.5 {
                return Ok(None);
            }
            let mut other = gamma_ratio(&[b], &[b - a], w)?;
            other.log = other.log - a * logarithm;
            if !exponential {
                other.log = other.log - z;
            }
            let displacement = other.log - state.log;
            let amplitude = other.y * cosine;
            let omitted = project(amplitude.v, displacement.v).abs() * 2.0;
            let omitted_d =
                project(amplitude.d + amplitude.v * displacement.d, displacement.v).abs() * 2.0;
            if omitted > f64::EPSILON * state.y.v.hi.abs()
                || omitted_d > f64::EPSILON * (state.y.d + state.y.v * state.log.d).hi.abs()
            {
                return Ok(None);
            }
            state.error += omitted;
        }
        state.method = algorithm_component!("real_asymptotic_expansion");
        return Ok(Some(state));
    }
    Ok(None)
}

pub(crate) fn evaluate(
    a: &[Jet],
    b: Jet,
    z: Jet,
    w: &mut Work<'_, '_, '_>,
) -> Result<Estimate, SciOxideError> {
    let mut numerator = [ZERO; 2];
    for (out, &x) in numerator.iter_mut().zip(a) {
        *out = R::from(x);
    }
    let estimate = evaluate_real(&numerator[..a.len()], R::from(b), R::from(z), w)?.finish()?;
    let parameter_direction = a.iter().chain(core::iter::once(&b)).any(|x| x.d.re != 0.0);
    if parameter_direction
        && a.len() <= 1
        && estimate.y.v.re != 0.0
        && estimate.error >= estimate.y.v.re.abs()
    {
        // This is the rounding model of the compensated algorithm, not a
        // condition-number bound. A direction cannot publish a base value
        // whose defining series has lost every digit to cancellation.
        return Err(ERRORS.unstable());
    }
    if parameter_direction && a.is_empty() && z.v.re < -128.0 {
        // Taylor propagation can amplify the unwanted origin solution for
        // negative b. An order sensitivity retains that analytic engine, but
        // must agree with the independently stable Bessel base value.
        let Some(base) = super::hypergeometric_value::estimate(
            super::hypergeometric_value::Kind::F01,
            &[],
            &[b.v.re],
            z.v.re,
            false,
            false,
            w,
        )?
        else {
            return Err(ERRORS.unstable());
        };
        let displacement = D::from(base.scale.v.re) - D::from(estimate.scale.v.re);
        let value = project(D::from(base.y.v.re), displacement);
        let error = project(D::from(base.error), displacement).abs();
        if !value.is_finite() || (estimate.y.v.re - value).abs() > estimate.error + error {
            return Err(ERRORS.unstable());
        }
    }
    Ok(estimate)
}

fn evaluate_real(a: &[R], b: R, z: R, w: &mut Work<'_, '_, '_>) -> Result<State, SciOxideError> {
    if a.iter().any(|x| pole(*x) && x.constant()) {
        return series(a, b, z, w);
    }
    if a.len() == 1 {
        if z.v.hi < -8.0 {
            // Cancel exp(z)*exp(-z) symbolically before forming the logarithm;
            // even twofold arithmetic cannot subtract 1e150 and retain log(z).
            if !pole(b - a[0])
                && -z.v.hi > 40.0 + 2.0 * (a[0].v.hi.abs() + b.v.hi.abs())
                && let Some(mut state) = asymptotic(b - a[0], b, -z, false, w)?
            {
                state.method = algorithm_component!("real_algebraic_asymptotic");
                return Ok(state);
            }
            let mut state = evaluate_real(&[b - a[0]], b, -z, w)?;
            state.log = state.log + z;
            state.method = algorithm_component!("real_kummer_transformation");
            return Ok(state);
        }
        // The dispatch margin is only a candidate selection: both asymptotic
        // contributions and the estimated remainder are checked above.
        if a[0].v.hi > 1e-6
            && z.v.hi > 40.0 + 2.0 * (a[0].v.hi.abs() + b.v.hi.abs())
            && let Some(state) = asymptotic(a[0], b, z, true, w)?
        {
            return Ok(state);
        }
        return series(a, b, z, w);
    }
    if a.len() == 2 {
        if z.v.hi == 1.0 {
            let gap = b - a[0] - a[1];
            if gap.v.hi <= 0.0 || !z.constant() && gap.v.hi <= 1.0 {
                return Err(
                    ERRORS.domain(scioxide_core::argument!("z"), DomainReason::OutsideDomain)
                );
            }
            let mut result = gamma_ratio(&[b, gap], &[b - a[0], b - a[1]], w)?;
            if !z.constant() {
                result.y.d =
                    result.y.d + result.y.v * a[0].v * a[1].v / (gap.v - D::from(1.0)) * z.d;
            }
            result.method = algorithm_component!("real_gauss_endpoint");
            return Ok(result);
        }
        if (a[0] - b).zero() || (a[1] - b).zero() {
            let exponent = if (a[0] - b).zero() { a[1] } else { a[0] };
            return Ok(State {
                log: -exponent * (ONE - z).ln(w)?,
                ..State::new(ONE, 0.0, algorithm_component!("parameter_identity"))
            });
        }
        let difference = a[1] - a[0];
        if z.v.hi < -4.0 && crate::util::sin_pi(difference.v.hi).abs() > 0.01 {
            let result = super::hypergeometric_shared::reciprocal_connection(
                [a[0], a[1]],
                b,
                z,
                false,
                w,
                |a, b, z, w| {
                    series(a, b, z, w).map(|s| super::hypergeometric_shared::Contribution {
                        y: s.y,
                        log: s.log,
                        error: s.error,
                    })
                },
                |a, b, w| {
                    gamma_ratio(a, b, w).map(|s| super::hypergeometric_shared::Contribution {
                        y: s.y,
                        log: s.log,
                        error: s.error,
                    })
                },
            )?;
            return Ok(State {
                y: result.y,
                log: result.log,
                error: result.error,
                method: algorithm_component!("real_reciprocal_connection"),
            });
        }
        if z.v.hi.abs() >= 0.7 {
            return continuation(a, b, z, w);
        }
    } else if z.v.hi < -8.0 {
        return continuation(a, b, z, w);
    }
    series(a, b, z, w)
}
