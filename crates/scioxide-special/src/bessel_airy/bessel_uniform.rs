//! Large real orders: Debye expansions away from x=nu and Olver's Airy
//! expansion through the turning point (DLMF 10.19.3, 10.19.6, 10.20.4--5).
//! All phases and exponential scales keep two words until projection.
use super::bessel::RealEstimate;
use super::bessel_base::square_root;
use crate::support::{
    Work,
    wide::{Wide as D, project},
};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use scioxide_core::RealScalar;
use scioxide_core::SciOxideError;

use crate::support::constants::PI;

fn finish(value: D, scale: D, absolute: f64) -> RealEstimate {
    let sign = value.value().signum();
    RealEstimate {
        value: project(value, scale),
        error: project(D::from(absolute), scale).min(f64::MAX),
        logarithm: (value.value() != 0.0).then(|| scale + value.mul_f64(sign).ln()),
        sign,
        relative: absolute / value.value().abs(),
    }
}

pub(super) fn evaluate(
    nu: f64,
    x: f64,
    second: bool,
    work: &mut Work,
) -> Result<RealEstimate, SciOxideError> {
    work.try_step()?;
    let ratio = D::from(x) / D::from(nu);
    let w = (D::from(1.0) - ratio) * (D::from(1.0) + ratio);
    if w.hi.abs() <= 0.5 {
        turning(nu, w, second, work)
    } else {
        debye(nu, x, w, second, work)
    }
}

fn polynomial(x: D, coefficients: &[(f64, f64)], work: &mut Work) -> Result<D, SciOxideError> {
    coefficients
        .iter()
        .rev()
        .try_fold(D::ZERO, |value, &(hi, lo)| {
            work.try_step()?;
            Ok(value * x + D { hi, lo })
        })
}

fn turning(nu: f64, w: D, second: bool, work: &mut Work) -> Result<RealEstimate, SciOxideError> {
    // Analytic at w=0. The 128-term geometric tail on |w|<=1/2 is
    // below 2^-128, including the two-word phase at nu<=1e9.
    let mut g = D::ZERO;
    for k in (0..128).rev() {
        work.try_step()?;
        g = g * w + D::from(3.0).div_f64((2 * k + 3) as f64);
    }
    let zeta_ratio = (g.mul_f64(0.5).ln() * D::from(2.0).div_f64(3.0)).exp_wide();
    let root = D::from(nu.cbrt());
    let root = root + (D::from(nu) - root * root * root) / (root * root).mul_f64(3.0);
    let argument = root * root * w * zeta_ratio;
    let prefactor = square_root(square_root(zeta_ratio.mul_f64(4.0))) / root;
    let mut a = D::ZERO;
    let mut b = D::ZERO;
    let mut power = D::from(1.0);
    let mut omitted = 0.0;
    for k in 0..5 {
        let av = polynomial(w, &super::bessel_debye::OLVER_A[k], work)? * power;
        let bv = polynomial(w, &super::bessel_debye::OLVER_B[k], work)? * power;
        if k == 4 {
            omitted = av.value().abs() + bv.value().abs();
        }
        a = a + av;
        b = b + bv;
        power = power.div_f64(nu).div_f64(nu);
    }
    let cube_root_two = crate::support::constants::CBRT_2;
    b = b * cube_root_two / (root * D::from(nu));
    let kind = if second {
        super::airy_kernel::Kind::Bi
    } else {
        super::airy_kernel::Kind::Ai
    };
    let airy = super::airy_kernel::real_pair(argument.hi, kind, true, work)?;
    work.check()?;
    // Taylor transport by the low argument word using y''=x*y. Keeping the
    // leading phase from the exact ratio avoids O(nu*epsilon) phase errors.
    let v = D::from(airy.y.v.re);
    let d = D::from(airy.y.d.re);
    let low = D::from(argument.lo);
    let transported = v + low
        * (d + low
            * (v.mul_f64(argument.hi * 0.5) + low * (v + d.mul_f64(argument.hi)).div_f64(6.0)));
    let slope =
        d + low * (v.mul_f64(argument.hi) + low * (v + d.mul_f64(argument.hi)).mul_f64(0.5));
    let sign = if second { -1.0 } else { 1.0 };
    let value = (a * transported + b * slope) * prefactor.mul_f64(sign);
    let scale = if argument.hi > 0.0 {
        let high = D::from(argument.hi);
        (high * square_root(high))
            .mul_f64(if second { 2.0 } else { -2.0 })
            .div_f64(3.0)
    } else {
        D::ZERO
    };
    let amplitude = prefactor.value().abs()
        * (a.value().abs() * transported.value().abs() + b.value().abs() * slope.value().abs());
    let propagated = prefactor.value().abs()
        * (a.value().abs() * airy.error + b.value().abs() * airy.derivative_error.unwrap_or(0.0));
    Ok(finish(
        value,
        scale,
        propagated + (64.0 * f64::EPSILON + omitted) * amplitude,
    ))
}

fn atan(mut x: D, work: &mut Work) -> Result<D, SciOxideError> {
    let reciprocal = x.hi > 1.0;
    if reciprocal {
        x = D::from(1.0) / x;
    }
    let shifted = x.hi > 0.414;
    if shifted {
        x = (x - D::from(1.0)) / (x + D::from(1.0));
    }
    let square = -(x * x);
    let mut term = x;
    let mut value = x;
    for k in 1..50 {
        work.try_step()?;
        term = term * square;
        value = value + term.div_f64((2 * k + 1) as f64);
    }
    if shifted {
        value = value + PI.mul_f64(0.25);
    }
    Ok(if reciprocal {
        PI.mul_f64(0.5) - value
    } else {
        value
    })
}

fn debye(
    nu: f64,
    x: f64,
    w: D,
    second: bool,
    work: &mut Work,
) -> Result<RealEstimate, SciOxideError> {
    let oscillatory = w.hi < 0.0;
    let q = square_root(if oscillatory { -w } else { w });
    let t = D::from(1.0) / q;
    let square = (t * t).mul_f64(if oscillatory { -1.0 } else { 1.0 });
    let mut even = D::from(1.0);
    let mut odd = D::ZERO;
    let mut power = D::from(1.0);
    let mut tail = 0.0;
    for (k, coefficients) in super::bessel_debye::DEBYE.iter().enumerate().skip(1) {
        let mut polynomial = D::ZERO;
        for &coefficient in coefficients[k..].iter().step_by(2).rev() {
            work.try_step()?;
            polynomial = polynomial * square + D::from(coefficient);
        }
        power = power * t.div_f64(nu);
        let term = polynomial * power;
        tail = term.value().abs();
        if oscillatory {
            let sign = if (k / 2).is_multiple_of(2) { 1.0 } else { -1.0 };
            if k.is_multiple_of(2) {
                even = even + term.mul_f64(sign);
            } else {
                odd = odd + term.mul_f64(sign);
            }
        } else {
            even = even
                + term.mul_f64(if second && !k.is_multiple_of(2) {
                    -1.0
                } else {
                    1.0
                });
        }
        if k >= 4 && tail < 1e-32 {
            break;
        }
    }
    if oscillatory {
        let phase = (q - atan(q, work)?).mul_f64(nu) - PI.mul_f64(0.25);
        let (s, c) = phase.hi.sin_cos();
        let (sl, cl) = phase.lo.sin_cos();
        let sin = D::from(s).mul_f64(cl) + D::from(c).mul_f64(sl);
        let cos = D::from(c).mul_f64(cl) - D::from(s).mul_f64(sl);
        let prefactor = square_root(D::from(2.0) / (PI * q).mul_f64(nu));
        let value = prefactor
            * if second {
                sin * even - cos * odd
            } else {
                cos * even + sin * odd
            };
        let error = prefactor.value().abs()
            * (tail + 64.0 * f64::EPSILON * (even.value().abs() + odd.value().abs()));
        Ok(finish(value, D::ZERO, error))
    } else {
        let alpha = D::from(nu).ln() + (D::from(1.0) + q).ln() - D::from(x).ln();
        let exponent = (q - alpha).mul_f64(if second { -nu } else { nu });
        let prefactor = if second {
            (PI.mul_f64(0.5) * q).ln()
        } else {
            (PI.mul_f64(2.0) * q).ln()
        };
        let scale = exponent - (prefactor + D::from(nu).ln()).mul_f64(0.5);
        Ok(finish(
            even.mul_f64(if second { -1.0 } else { 1.0 }),
            scale,
            tail + 64.0 * f64::EPSILON * even.value().abs(),
        ))
    }
}
