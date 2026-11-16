//! Two-term (double-double) real Bessel kernels for cancellation regions.
//!
//! `bessel_kernel` calls these only where `f64` rounding is amplified:
//! Steed's CF1 in the oscillatory band and the order recurrences of its
//! normalization, CF2 near zeros of Y, Temme sums near zeros of Y and for K
//! near x = 2, and series whose prefix leaves the `f64` range. Debye
//! uniform expansions for orders >= 32 keep their exponent in two terms.
use crate::support::Work;
// Adapted from Boost.Math 1.89 bessel_jy.hpp and bessel_ik.hpp,
// copyright Xiaogang Zhang 2006 and Matt Borland 2024, Boost Software License 1.0.
// Attribution and license: THIRD_PARTY_NOTICES.md.
pub(super) use crate::support::wide::{
    Wide, difference, product, quotient, scale, sum, two_product,
};
use scioxide_core::{RealScalar, SciOxideError};

/// Y_u,Y_(u+1), or K_u,K_(u+1), for |u|<=1/2 and 0<x<=2.
pub(super) fn temme(
    u: f64,
    x: f64,
    modified: bool,
    w: &mut Work,
) -> Result<(f64, f64), SciOxideError> {
    w.try_step()?;
    let u = D::from(u);
    let square = u * u;
    let (g1, g2) = crate::support::gamma_series::reciprocal_gamma_temme(u, square, w)?;
    let one = D::from(1.0);
    let a = logarithm(D::from(x), w)? - LN_2;
    let sigma = -u * a;
    let power = exponential(-sigma, w)?;
    let inverse_power = one / power;
    let cosh = (power + inverse_power).mul_f64(0.5);
    let sinhc = if sigma.hi.abs() < 1e-5 {
        let square = sigma * sigma;
        one + square * (D::from(1.0 / 6.0) + square.mul_f64(1.0 / 120.0))
    } else {
        (inverse_power - power) / sigma.mul_f64(2.0)
    };
    // Gamma(1+u) Gamma(1-u) = 1/sinc(pi*u).
    let plus = g2 - u * g1;
    let minus = g2 + u * g1;
    let sinc = plus * minus;
    let mut f = (g1 * cosh - g2 * a * sinhc) / sinc;
    let mut p = one / (plus * power).mul_f64(2.0);
    let mut q = power / minus.mul_f64(2.0);
    let e = if modified {
        D::ZERO
    } else {
        // 2 sin(pi*u/2)^2/u, with the analytic zero-order limit.
        let z = PI_WIDE * u.mul_f64(0.5);
        let z2 = z * z;
        let mut sinc = one;
        let mut term = one;
        for k in 1..=16 {
            w.try_step()?;
            term = -term * z2 / D::from((2 * k * (2 * k + 1)) as f64);
            sinc = sinc + term;
            if term.hi.abs() < 1e-33 {
                break;
            }
        }
        PI_WIDE * z * sinc * sinc
    };
    if !modified {
        let factor = D::from(2.0) / PI_WIDE;
        f = f * factor;
        p = p * factor;
        q = q * factor;
    }
    let mut sum = f + e * q;
    let mut sum1 = p;
    let mut coefficient = one;
    let xd = D::from(x);
    for k in 1..usize::MAX {
        w.try_step()?;
        let k = D::from(k as f64);
        f = quotient(product(k, f) + p + q, product(k, k) - square);
        p = quotient(p, k - u);
        q = quotient(q, k + u);
        let g = f + product(e, q);
        let h = p - product(k, g);
        coefficient = quotient(
            product(product(coefficient, xd), xd),
            scale(k, if modified { 4.0 } else { -4.0 }),
        );
        let term = product(coefficient, g);
        let term1 = product(coefficient, h);
        sum = sum + term;
        sum1 = sum1 + term1;
        if term.hi.abs() <= 1e-30 * sum.hi.abs() && term1.hi.abs() <= 1e-30 * sum1.hi.abs() {
            break;
        }
    }
    let sign = if modified { 1.0 } else { -1.0 };
    Ok((sum.value() * sign, (sum1.mul_f64(sign * 2.0) / xd).value()))
}

// Shared two-word primitives retain corrections until scalar projection.
type D = Wide;

use crate::support::constants::LN_2;
use crate::support::constants::PI as PI_WIDE;

// These bounded support primitives each consume one work unit. Recurrences
// and polynomial coefficients retain their own per-step admission.
fn logarithm(value: D, w: &mut Work) -> Result<D, SciOxideError> {
    w.try_step()?;
    Ok(value.ln())
}

fn exponential(value: D, w: &mut Work) -> Result<D, SciOxideError> {
    w.try_step()?;
    Ok(value.exp_wide())
}

/// Real and imaginary parts of H'_u/H_u, for x>2.
pub(super) fn jy_fraction(u: D, x: D, w: &mut Work) -> Result<(D, D), SciOxideError> {
    let one = D::from(1.0);
    let mut fr = D::from(-0.5) / x;
    let mut fi = one;
    let a = (D::from(0.25) - u * u) / x;
    let br = x.mul_f64(2.0);
    let mut bi = D::from(2.0);
    let mut cr = br + a / (fr * fr + one);
    let mut ci = bi + a * fr / (fr * fr + one);
    let mut dr = br / (br * br + bi * bi);
    let mut di = -bi / (br * br + bi * bi);
    let mut delta_r = cr * dr - ci * di;
    let mut delta_i = ci * dr + cr * di;
    (fr, fi) = (fr * delta_r - fi * delta_i, fr * delta_i + fi * delta_r);
    for k in 2..usize::MAX {
        w.try_step()?;
        let a = D::from((k as f64 - 0.5).powi(2)) - u * u;
        bi = bi + D::from(2.0);
        let denominator = cr * cr + ci * ci;
        (cr, ci) = (br + a * cr / denominator, bi - a * ci / denominator);
        (dr, di) = (br + a * dr, bi + a * di);
        let denominator = dr * dr + di * di;
        (dr, di) = (dr / denominator, -di / denominator);
        delta_r = cr * dr - ci * di;
        delta_i = ci * dr + cr * di;
        (fr, fi) = (fr * delta_r - fi * delta_i, fr * delta_i + fi * delta_r);
        if (delta_r - one).hi.abs() + delta_i.hi.abs() <= 1e-30 {
            break;
        }
    }
    Ok((fr, fi))
}

/// Y_nu for 0 < x <= 2 from two-term Temme seeds and forward recurrence.
pub(super) fn y_small(order: f64, x: f64, w: &mut Work) -> Result<f64, SciOxideError> {
    w.try_step()?;
    let index = order.round().max(0.0);
    if index > w.remaining as f64 {
        w.complete = false;
        w.check()?;
    }
    let n = index.try_to_usize_exact()?;
    let u = order - index;
    let (y, y1) = temme(u, x, false, w)?;
    Ok(forward_y(u, x, n, (D::from(y), 0.0), (D::from(y1), 0.0), w)?.0)
}

/// Y_(u+n) from Y_u and Y_(u+1) by two-term forward recurrence, carrying
/// first-order absolute errors of the seeds; the recurrence adds none.
pub(super) fn forward_y(
    u: f64,
    x: f64,
    n: usize,
    (mut y, mut error): (D, f64),
    (mut y1, mut error1): (D, f64),
    w: &mut Work,
) -> Result<(f64, f64), SciOxideError> {
    let recurrence_step = quotient(D::from(2.0), D::from(x));
    for k in 1..=n {
        w.try_step()?;
        // Y_nu is y1 at the last step. Its successor is unused and can exceed
        // the scalar range even when the requested value is representable.
        crate::support::check_finite(y1.value())?;
        if k == n {
            return Ok((y1.value(), error1));
        }
        let factor = product(sum(D::from(u), D::from(k as f64)), recurrence_step);
        if !(factor.hi * y1.hi).is_finite() {
            return Err(overflow());
        }
        let next = difference(product(factor, y1), y);
        if !next.value().is_finite() {
            return Err(overflow());
        }
        (y, y1) = (y1, next);
        (error, error1) = (
            error1,
            scioxide_core::fast_mul_add(factor.hi.abs(), error1, error),
        );
    }
    Ok((y.value(), error))
}

/// Overflow of the Y recurrence.
pub(super) fn overflow() -> SciOxideError {
    SciOxideError::numerical_instability(
        scioxide_core::NumericalIssue::Overflow,
        "Bessel Y recurrence exceeds the scalar range",
    )
}

pub(super) use crate::support::wide::square_root;

/// Modified Debye value with its exponent kept as two terms.
pub(super) fn uniform_log(
    order: f64,
    x: f64,
    second: bool,
    scaled: bool,
    w: &mut Work,
) -> Result<D, SciOxideError> {
    let large = order.max(x);
    let nu = D::from(order);
    let xd = D::from(x);
    let one = D::from(1.0);
    let a = nu / D::from(large);
    let b = xd / D::from(large);
    let q = square_root(a * a + b * b);
    let t = a / q;
    let ratio = order / x;
    let asinh = if ratio < 1e-8 {
        let r = nu / xd;
        r - r * r * r / D::from(6.0)
    } else if order <= x {
        logarithm(a + q, w)?
    } else {
        logarithm(a + q, w)? + logarithm(D::from(large), w)? - logarithm(xd, w)?
    };
    let exponent = nu * (a / (q + b) - asinh);
    let mut sum = one;
    let mut inverse_power = one;
    let mut small_terms = 0;
    let square = t * t;
    let step = t / nu;
    for (k, coefficients) in crate::bessel_airy::bessel_debye::DEBYE
        .iter()
        .enumerate()
        .skip(1)
    {
        let mut value = D::ZERO;
        for &coefficient in coefficients[k..].iter().step_by(2).rev() {
            w.try_step()?;
            value = value * square + D::from(coefficient);
        }
        inverse_power = inverse_power * step;
        let term = value * inverse_power;
        sum = sum + if second && k % 2 != 0 { -term } else { term };
        small_terms = if term.hi.abs() <= (f64::EPSILON / 256.0) * sum.hi.abs() {
            small_terms + 1
        } else {
            0
        };
        if small_terms >= 2 {
            break;
        }
    }
    let log_prefactor = if second {
        logarithm(PI_WIDE.mul_f64(0.5), w)?
    } else {
        -logarithm(PI_WIDE.mul_f64(2.0), w)?
    };
    let log_prefactor =
        (log_prefactor - logarithm(D::from(large), w)? - logarithm(q, w)?).mul_f64(0.5);
    let mut result = logarithm(sum, w)? + log_prefactor + if second { -exponent } else { exponent };
    if !scaled {
        result = result + D::from(if second { -x } else { x });
    }
    Ok(result)
}

pub(super) fn project_log(value: D, w: &mut Work) -> Result<f64, SciOxideError> {
    if value.hi > 710.0 {
        return Ok(f64::INFINITY);
    }
    if value.hi < -746.0 {
        return Ok(0.0);
    }
    // Split the scale so the two-term correction survives until the final
    // binary64/subnormal rounding, without overflowing an intermediate exp.
    let half = exponential(value.mul_f64(0.5), w)?;
    if half.hi > f64::MAX / half.hi {
        return Ok(f64::INFINITY);
    }
    let product = half * half;
    Ok(product.value())
}

/// Defining series with an accurately normalized prefactor. J uses x<=2;
/// long I series beyond the `f64` kernel's argument limit.
pub(super) fn power(
    order: f64,
    x: f64,
    modified: bool,
    scaled: bool,
    w: &mut Work,
) -> Result<f64, SciOxideError> {
    w.try_step()?;
    let nu = D::from(order);
    let half_x = D::from(x).mul_f64(0.5);
    let mut prefix = nu * (logarithm(D::from(x), w)? - LN_2)
        - crate::support::gamma_series::gamma_log_shifted(nu, w)?;
    if scaled {
        prefix = prefix - D::from(x);
    }
    let mut term = D::from(1.0);
    let mut sum = term;
    let factor = (half_x * half_x).mul_f64(if modified { 1.0 } else { -1.0 });
    for k in 1..usize::MAX {
        w.try_step()?;
        let k = D::from(k as f64);
        let ratio = factor / (k * (nu + k));
        if term.hi.abs() > 1e150 / ratio.hi.abs().max(1.0) {
            let scale = f64::from_bits(511_u64 << 52);
            term = term.mul_f64(scale);
            sum = sum.mul_f64(scale);
            prefix = prefix + LN_2.mul_f64(512.0);
        }
        term = term * ratio;
        sum = sum + term;
        if term.hi.abs() <= 1e-30 * sum.hi.abs() {
            break;
        }
    }
    if sum.hi == 0.0 {
        return Ok(0.0);
    }
    let sign = sum.hi.signum();
    Ok(project_log(prefix + logarithm(sum.mul_f64(sign), w)?, w)? * sign)
}
