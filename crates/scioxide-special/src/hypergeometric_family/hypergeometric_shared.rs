//! Coordinate-independent Taylor propagation and connection building blocks.
use crate::support::{C, HYPERGEOMETRIC_ERRORS as ERRORS, Work};
use core::ops::{Add, Div, Mul, Neg, Sub};
#[allow(unused_imports)]
use scioxide_core::RealScalar;
use scioxide_core::SciOxideError;

/// Private arithmetic protocol: a value and one analytic direction. Implementations
/// keep two terms; this is deliberately separate from core's sealed scalar types.
pub(super) trait Arithmetic:
    Copy
    + From<f64>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    const ROUNDING: f64 = 1e-30;
    const TAIL: f64 = 0.125 * f64::EPSILON;
    fn constant(value: C) -> Self;
    fn point(self) -> C;
    fn value_norm(self) -> f64;
    fn direction_norm(self) -> f64;
    fn finite(self) -> bool;
    fn scale(self, factor: f64) -> Self;
    fn small(self, other: Self, factor: f64) -> bool {
        self.value_norm() <= factor * other.value_norm()
            && self.direction_norm() <= factor * other.direction_norm()
    }
}

/// Value, slope, their absolute errors, and an exact common binary rescaling.
pub(super) struct Propagated<T> {
    pub y: T,
    pub slope: T,
    pub error: f64,
    pub slope_error: f64,
    pub log: f64,
}

/// Scaled Taylor terms of A(z)y''+B(z)y'+Cy=0, including both unit solutions.
/// Step coordinates are subtracted in the working arithmetic, so a rounded
/// `f64` center never silently changes the destination of a Taylor step.
pub(super) fn propagate<T: Arithmetic>(
    a: &[T],
    b: T,
    target: T,
    center: C,
    mut state: Propagated<T>,
    w: &mut Work<'_, '_, '_>,
) -> Result<Propagated<T>, SciOxideError> {
    let one = T::from(1.0);
    let zero = T::from(0.0);
    let mut c = T::constant(center);
    loop {
        w.try_step()?;
        let distance = if a.len() == 2 {
            c.value_norm().min((one - c).value_norm())
        } else {
            c.value_norm()
        };
        let cap = if a.is_empty() {
            c.value_norm().sqrt().max(0.25)
        } else {
            4.0
        };
        let radius = if a.len() == 2 {
            0.35 * distance
        } else {
            (0.35 * distance).min(cap)
        };
        let delta = target - c;
        let last = delta.value_norm() <= radius;
        let h = if last {
            delta
        } else {
            T::constant(delta.point() * C::new(radius / delta.value_norm(), 0.0))
        };
        if h.value_norm() == 0.0 {
            if last {
                state.y = state.y + state.slope * h;
                return Ok(state);
            }
            return Err(ERRORS.unstable());
        }
        let (a0, a1, a2, b0, b1, c0) = if a.len() == 2 {
            (
                c * (one - c),
                one - c.scale(2.0),
                -one,
                b - (a[0] + a[1] + one) * c,
                -(a[0] + a[1] + one),
                -a[0] * a[1],
            )
        } else {
            (
                c,
                one,
                zero,
                if a.is_empty() { b } else { b - c },
                T::from(if a.is_empty() { 0.0 } else { -1.0 }),
                if a.is_empty() { -one } else { -a[0] },
            )
        };
        let square = h * h;
        // [previous scaled coefficient, current, value, h*slope, absolute sums].
        let mut units = [[one, zero, one, zero], [zero, h, h, h]];
        let mut previous = state.y;
        let mut current = state.slope * h;
        let mut y = previous + current;
        let mut moment = current;
        let mut magnitude = previous.value_norm() + current.value_norm();
        let mut weighted = current.value_norm();
        let mut quiet = 0;
        for n in 0usize.. {
            w.try_step()?;
            let k = n as f64;
            let p = (a1.scale(k) + b0).scale(k + 1.0) * h / a0.scale((k + 2.0) * (k + 1.0));
            let q = (a2.scale(k * (k - 1.0)) + b1.scale(k) + c0) * square
                / a0.scale((k + 2.0) * (k + 1.0));
            let term = -(p * current + q * previous);
            let dterm = term.scale(k + 2.0);
            y = y + term;
            moment = moment + dterm;
            magnitude += term.value_norm();
            weighted += dterm.value_norm();
            // Unit solutions need values only, not the selected analytic direction.
            let (p, q) = (T::constant(p.point()), T::constant(q.point()));
            for unit in &mut units {
                let next = -(p * unit[1] + q * unit[0]);
                *unit = [unit[1], next, unit[2] + next, unit[3] + next.scale(k + 2.0)];
            }
            if !y.finite() || !moment.finite() {
                return Err(ERRORS.unstable());
            }
            quiet = if term.small(y, 1e-30) && dterm.small(moment, 1e-30) {
                quiet + 1
            } else {
                0
            };
            if quiet == 3 {
                let [[_, _, u, du], [_, _, v, dv]] = units;
                let (e, de) = (state.error, state.slope_error);
                state.error = u.value_norm() * e
                    + v.value_norm() * de
                    + 4.0 * term.value_norm()
                    + 32e-30 * magnitude;
                state.slope_error = (du.value_norm() * e
                    + dv.value_norm() * de
                    + 4.0 * dterm.value_norm()
                    + 32e-30 * weighted)
                    / h.value_norm();
                break;
            }
            (previous, current) = (current, term);
        }
        state.y = y;
        state.slope = moment / h;
        if last {
            return Ok(state);
        }
        if state.y.value_norm().max(state.slope.value_norm()) > 1e100 {
            let factor = 2.0_f64.powi(-256);
            state.y = state.y.scale(factor);
            state.slope = state.slope.scale(factor);
            state.error *= factor;
            state.slope_error *= factor;
            state.log += 256.0 * core::f64::consts::LN_2;
        }
        c = c + h;
    }
}

/// The common 2F0 asymptotic sum, stopping before divergence. The first
/// neglected term and arithmetic accumulation both contribute to the bound.
pub(super) fn asymptotic<T: Arithmetic>(
    a: T,
    b: T,
    inverse: T,
    w: &mut Work<'_, '_, '_>,
) -> Result<Option<(T, f64)>, SciOxideError> {
    let mut term = T::from(1.0);
    let mut sum = term;
    let mut magnitude = 1.0;
    let mut previous = f64::INFINITY;
    for n in 0usize.. {
        w.try_step()?;
        let k = T::from(n as f64);
        term = term * (a + k) * (b + k) * inverse / T::from((n + 1) as f64);
        let size = term.value_norm().max(term.direction_norm());
        if size > previous || !term.finite() {
            return Ok(None);
        }
        sum = sum + term;
        magnitude += term.value_norm();
        if term.small(sum, T::TAIL) {
            return Ok(Some((
                sum,
                4.0 * term.value_norm() + 16.0 * T::ROUNDING * magnitude,
            )));
        }
        previous = size;
    }
    Err(ERRORS.unstable())
}

/// Reciprocal Gauss connection: transformed numerator/denominator parameters.
pub(super) fn reciprocal<T: Arithmetic>(a: T, b: T, c: T) -> [([T; 2], T); 2] {
    let one = T::from(1.0);
    [
        ([a, one - c + a], one - b + a),
        ([b, one - c + b], one - a + b),
    ]
}

pub(super) trait ConnectionArithmetic: Arithmetic {
    fn logarithm(self, w: &mut Work<'_, '_, '_>) -> Result<Self, SciOxideError>;
    fn exponential(self) -> Self;
}
/// A normalized connection contribution, with its mantissa's absolute error.
pub(super) struct Contribution<T> {
    pub y: T,
    pub log: T,
    pub error: f64,
}

/// Common reciprocal connection, including gamma ratios, transformed series,
/// logarithmic normalization and cancellation-aware error assembly.
pub(super) fn reciprocal_connection<T: ConnectionArithmetic>(
    [a, b]: [T; 2],
    c: T,
    z: T,
    regularized: bool,
    w: &mut Work<'_, '_, '_>,
    series: impl Fn(&[T], T, T, &mut Work<'_, '_, '_>) -> Result<Contribution<T>, SciOxideError>,
    gamma_ratio: impl Fn(&[T], &[T], &mut Work<'_, '_, '_>) -> Result<Contribution<T>, SciOxideError>,
) -> Result<Contribution<T>, SciOxideError> {
    let [(p, d1), (q, d2)] = reciprocal(a, b, c);
    let inverse = T::from(1.0) / z;
    let logarithm = (-z).logarithm(w)?;
    let first_numerator = [b - a, c];
    let second_numerator = [a - b, c];
    let count = if regularized { 1 } else { 2 };
    let mut first = gamma_ratio(&first_numerator[..count], &[b, c - a], w)?;
    let mut second = gamma_ratio(&second_numerator[..count], &[a, c - b], w)?;
    let s1 = series(&p, d1, inverse, w)?;
    let s2 = series(&q, d2, inverse, w)?;
    first.error = first.error * s1.y.value_norm() + s1.error * first.y.value_norm();
    second.error = second.error * s2.y.value_norm() + s2.error * second.y.value_norm();
    first.y = first.y * s1.y;
    second.y = second.y * s2.y;
    first.log = first.log - a * logarithm + s1.log;
    second.log = second.log - b * logarithm + s2.log;
    let scale = if first.y.value_norm() == 0.0 {
        second.log.point().re
    } else if second.y.value_norm() == 0.0 {
        first.log.point().re
    } else {
        first.log.point().re.max(second.log.point().re)
    };
    let log = T::from(scale);
    let factor = |x: &Contribution<T>| {
        if x.y.value_norm() == 0.0 && x.y.direction_norm() == 0.0 {
            T::from(0.0)
        } else {
            (x.log - log).exponential()
        }
    };
    let p = factor(&first);
    let q = factor(&second);
    Ok(Contribution {
        y: first.y * p + second.y * q,
        log,
        error: (first.error + 32.0 * f64::EPSILON * first.y.value_norm()) * p.value_norm()
            + (second.error + 32.0 * f64::EPSILON * second.y.value_norm()) * q.value_norm(),
    })
}
