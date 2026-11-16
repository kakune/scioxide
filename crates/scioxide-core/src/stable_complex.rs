//! Complex arguments of the functions in `stable`.
//!
//! On their real domains, the complex forms of sigmoid, softplus, log_sigmoid
//! and logit stay within the real 16-ulp bound in each component.
use crate::ScaledValue;
use crate::stable::{domain, input, output, phase, radians};
use crate::{Complex, RealScalar, SciOxideError};
use core::f64::consts::PI;

fn checked<S: RealScalar>(z: Complex<S>) -> Result<Complex<f64>, SciOxideError> {
    Ok(Complex::new(input(z.re)?, input(z.im)?))
}
fn finish<S: RealScalar>(z: Complex<f64>) -> Result<Complex<S>, SciOxideError> {
    Ok(Complex::new(output(z.re)?, output(z.im)?))
}
fn expm1(z: Complex<f64>) -> Complex<f64> {
    // Re(e^z - 1) = expm1(re) cos(im) + (cos(im) - 1), and
    // cos(im) - 1 = -2 sin^2(im / 2), so neither part cancels near z = 0. Past
    // re ~ 700 the -1 is below rounding.
    if z.re > 700.0 {
        return z.exp() - <crate::Complex<f64> as crate::AlgebraicScalar>::ONE;
    }
    let s = RealScalar::sin(z.im * 0.5);
    Complex::new(
        RealScalar::exp_m1(z.re) * RealScalar::cos(z.im) - 2.0 * s * s,
        z.exp().im,
    )
}
fn log1p(z: Complex<f64>) -> Result<Complex<f64>, SciOxideError> {
    // Sum 2*re + re^2 + im^2 before rounding. Forming re*(2+re)
    // first loses the correction on the |1+z|=1 cancellation surface.
    if z.norm() < 0.5 {
        let delta = crate::checked_sum_of_products(&[z.re, z.re, z.im], &[2.0, z.re, z.im])?;
        Ok(Complex::new(
            0.5 * RealScalar::ln_1p(delta),
            RealScalar::atan2(z.im, 1.0 + z.re),
        ))
    } else {
        Ok((<crate::Complex<f64> as crate::AlgebraicScalar>::ONE + z).ln())
    }
}
fn multiply(x: Complex<f64>, y: Complex<f64>) -> Result<Complex<f64>, SciOxideError> {
    if !x.re.is_finite() || !x.im.is_finite() || !y.re.is_finite() || !y.im.is_finite() {
        return Err(SciOxideError::non_finite(
            crate::NonFiniteOrigin::Intermediate,
            "complex transform factor exceeds scalar range",
        ));
    }
    ScaledValue::from_value(x)?
        .mul(ScaledValue::from_value(y)?)?
        .round()
}
// For nonpositive real parts, retain 1+cos(theta) and exp(re)-1
// separately near an odd-pi phase. Ordinary exp rounds away tiny real parts.
fn one_plus_exp(z: Complex<f64>) -> Complex<f64> {
    let (sin, cos) = RealScalar::sin_cos(z.im);
    // Re(1 + e^z) = (1 + cos) + expm1(re) cos, with 1 + cos = 2 cos^2(im / 2).
    if z.re > -0.5 && cos < -0.5 {
        let half_cos = RealScalar::cos(z.im * 0.5);
        Complex::new(
            2.0 * half_cos * half_cos + RealScalar::exp_m1(z.re) * cos,
            RealScalar::exp(z.re) * sin,
        )
    } else {
        <crate::Complex<f64> as crate::AlgebraicScalar>::ONE + z.exp()
    }
}
fn log_one_plus_exp(z: Complex<f64>) -> Result<Complex<f64>, SciOxideError> {
    if z.re > -0.5 && RealScalar::cos(z.im) < -0.5 {
        Ok(one_plus_exp(z).ln())
    } else {
        log1p(z.exp())
    }
}
fn softplus_value(z: Complex<f64>) -> Result<Complex<f64>, SciOxideError> {
    if z.re <= 0.0 {
        log_one_plus_exp(z)
    } else {
        // Principal log of 1+exp(z), rather than an unwrapped analytic log.
        // ln(1 + e^z) = z + ln(1 + e^-z) for the real part. The imaginary part is
        // taken directly as arg(1 + e^z) = atan2(sin, cos + e^-re), since
        // 1 + e^z = e^re (e^(i im) + e^-re), which keeps it principal.
        let correction = log_one_plus_exp(-z)?;
        let (s, c) = RealScalar::sin_cos(z.im);
        let real = if z.re < 0.5 && c < -0.5 {
            let half_cos = RealScalar::cos(z.im * 0.5);
            2.0 * half_cos * half_cos + RealScalar::exp_m1(-z.re)
        } else {
            c + RealScalar::exp(-z.re)
        };
        Ok(Complex::new(
            z.re + correction.re,
            RealScalar::atan2(s, real),
        ))
    }
}

pub(super) fn softplus<S: RealScalar>(z: Complex<S>) -> Result<Complex<S>, SciOxideError> {
    let z = checked(z)?;
    finish(softplus_value(z)?)
}

pub(super) fn sigmoid<S: RealScalar>(z: Complex<S>) -> Result<Complex<S>, SciOxideError> {
    let z = checked(z)?;
    finish(
        // Exponentiate only arguments with nonpositive real part, mirroring the
        // real sigmoid.
        if z.re >= 0.0 {
            <crate::Complex<f64> as crate::AlgebraicScalar>::ONE / one_plus_exp(-z)
        } else {
            let e = z.exp();
            e / one_plus_exp(z)
        },
    )
}

pub(super) fn log_sigmoid<S: RealScalar>(z: Complex<S>) -> Result<Complex<S>, SciOxideError> {
    let z = checked(z)?;
    finish({
        let mut r = -softplus_value(-z)?;
        // Negating softplus can move the imaginary part to +pi; rewrap it to the
        // principal range.
        r.im = RealScalar::atan2(RealScalar::sin(r.im), RealScalar::cos(r.im));
        r
    })
}

pub(super) fn logit<S: RealScalar>(z: Complex<S>) -> Result<Complex<S>, SciOxideError> {
    let z = checked(z)?;
    if z == <crate::Complex<f64> as crate::AlgebraicScalar>::ZERO
        || z == <crate::Complex<f64> as crate::AlgebraicScalar>::ONE
    {
        return Err(domain(crate::argument!("z")));
    }
    let left = z.ln();
    let right = log1p(-z)?;
    // |z|^2-|1-z|^2 = 2*re-1. Reconstruct the logarithmic
    // ratio directly when the magnitudes are close, avoiding subtraction of
    // rounded logarithms. Divide in stages to avoid squaring a large norm.
    let norm = (<crate::Complex<f64> as crate::AlgebraicScalar>::ONE - z).norm();
    let delta = if z.re.abs() <= f64::MAX * 0.5 {
        ((2.0 * z.re - 1.0) / norm) / norm
    } else {
        2.0 * ((z.re / norm) / norm) - (1.0 / norm) / norm
    };
    let real = if delta.abs() < 0.5 {
        0.5 * RealScalar::ln_1p(delta)
    } else {
        left.re - right.re
    };
    finish(Complex::new(real, left.im - right.im))
}

pub(super) fn cosm1<S: RealScalar>(z: Complex<S>) -> Result<Complex<S>, SciOxideError> {
    let z = checked(z)?;
    finish({
        let s = (z * 0.5).sin();
        multiply(s, s)? * -2.0
    })
}

pub(super) fn sinc<S: RealScalar>(z: Complex<S>) -> Result<Complex<S>, SciOxideError> {
    let z = checked(z)?;
    finish(
        if z == <crate::Complex<f64> as crate::AlgebraicScalar>::ZERO {
            <crate::Complex<f64> as crate::AlgebraicScalar>::ONE
        // Taylor series through z^6; the first omitted term, |z|^8 / 9!, is far
        // below rounding for |z| < 0.01.
        } else if z.norm() < 0.01 {
            let q = multiply(z, z)?;
            <crate::Complex<f64> as crate::AlgebraicScalar>::ONE
                - q * (<crate::Complex<f64> as crate::AlgebraicScalar>::ONE / 6.0
                    - q * (<crate::Complex<f64> as crate::AlgebraicScalar>::ONE / 120.0
                        - q * (<crate::Complex<f64> as crate::AlgebraicScalar>::ONE / 5040.0)))
        } else {
            sinc_quotient(z, false)?
        },
    )
}

pub(super) fn exprel<S: RealScalar>(z: Complex<S>) -> Result<Complex<S>, SciOxideError> {
    let z = checked(z)?;
    finish(
        if z == <crate::Complex<f64> as crate::AlgebraicScalar>::ZERO {
            <crate::Complex<f64> as crate::AlgebraicScalar>::ONE
        // sum_k z^k / (k + 1)!, truncated after z^8, where the remainder is
        // negligible for |z| < 0.01.
        } else if z.norm() < 0.01 {
            let mut term = <crate::Complex<f64> as crate::AlgebraicScalar>::ONE;
            let mut sum = <crate::Complex<f64> as crate::AlgebraicScalar>::ONE;
            for k in 1..=8 {
                term = term * z / f64::from(k + 1);
                sum += term;
            }
            sum
        } else if z.re > 700.0 {
            let e = (z * 0.5).exp();
            multiply(e / z, e)?
        } else {
            expm1(z) / z
        },
    )
}

pub(super) fn xlogy<S: RealScalar>(
    x: Complex<S>,
    y: Complex<S>,
) -> Result<Complex<S>, SciOxideError> {
    let x = checked(x)?;
    let y = checked(y)?;
    if x != <crate::Complex<f64> as crate::AlgebraicScalar>::ZERO
        && y == <crate::Complex<f64> as crate::AlgebraicScalar>::ZERO
    {
        return Err(domain(crate::argument!("y")));
    }
    finish(
        if x == <crate::Complex<f64> as crate::AlgebraicScalar>::ZERO {
            <crate::Complex<f64> as crate::AlgebraicScalar>::ZERO
        } else {
            multiply(x, y.ln())?
        },
    )
}

pub(super) fn xlog1py<S: RealScalar>(
    x: Complex<S>,
    y: Complex<S>,
) -> Result<Complex<S>, SciOxideError> {
    let x = checked(x)?;
    let y = checked(y)?;
    if x != <crate::Complex<f64> as crate::AlgebraicScalar>::ZERO
        && y == -<crate::Complex<f64> as crate::AlgebraicScalar>::ONE
    {
        return Err(domain(crate::argument!("y")));
    }
    finish(
        if x == <crate::Complex<f64> as crate::AlgebraicScalar>::ZERO {
            <crate::Complex<f64> as crate::AlgebraicScalar>::ZERO
        } else {
            multiply(x, log1p(y)?)?
        },
    )
}

pub(super) fn powm1<S: RealScalar>(
    x: Complex<S>,
    y: Complex<S>,
) -> Result<Complex<S>, SciOxideError> {
    let x = checked(x)?;
    let y = checked(y)?;
    if x == <crate::Complex<f64> as crate::AlgebraicScalar>::ZERO
        && y != <crate::Complex<f64> as crate::AlgebraicScalar>::ZERO
    {
        return if y.re > 0.0 {
            finish(-<crate::Complex<f64> as crate::AlgebraicScalar>::ONE)
        } else {
            Err(domain(crate::argument!("y")))
        };
    }
    finish(
        if y == <crate::Complex<f64> as crate::AlgebraicScalar>::ZERO
            || x == <crate::Complex<f64> as crate::AlgebraicScalar>::ONE
        {
            <crate::Complex<f64> as crate::AlgebraicScalar>::ZERO
        } else {
            // Integer powers permit exact quarter-turn extraction. Reducing
            // near -1 or +/-i to a base near +1 avoids rounding away the small
            // phase before multiplying a principal logarithm by the exponent.
            let (base, quadrant) = if y.im == 0.0 && RealScalar::trunc(y.re) == y.re {
                let q = if x.re.abs() >= x.im.abs() {
                    if x.re >= 0.0 { 0 } else { 2 }
                } else if x.im >= 0.0 {
                    1
                } else {
                    3
                };
                let base = match q {
                    0 => x,
                    1 => Complex::new(x.im, -x.re),
                    2 => -x,
                    _ => Complex::new(-x.im, x.re),
                };
                (base, ((y.re % 4.0) as i32 * q).rem_euclid(4))
            } else {
                (x, 0)
            };
            let log = if (base - <crate::Complex<f64> as crate::AlgebraicScalar>::ONE).norm() < 0.5
            {
                log1p(base - <crate::Complex<f64> as crate::AlgebraicScalar>::ONE)?
            } else {
                base.ln()
            };
            let delta = match multiply(y, log) {
                Ok(exponent) => expm1(exponent),
                Err(_) => base.powc(y) - <crate::Complex<f64> as crate::AlgebraicScalar>::ONE,
            };
            // x^y = i^k (1 + delta) with k = q y mod 4, so x^y - 1 is the rotation of
            // 1 + delta by k quarter turns, minus 1, formed without re-rounding delta.
            match quadrant {
                0 => delta,
                1 => Complex::new(-delta.im - 1.0, delta.re + 1.0),
                2 => Complex::new(-delta.re - 2.0, -delta.im),
                _ => Complex::new(delta.im - 1.0, -delta.re - 1.0),
            }
        },
    )
}

fn scaled_phase(z: Complex<f64>, half_turn: f64) -> (Complex<f64>, Complex<f64>) {
    let (s, c) = phase(z.re, half_turn);
    let y = radians(z.im, half_turn);
    // Split exponentials retain representable small components near real phases.
    let e = RealScalar::exp(y.abs() * 0.5);
    let decay = RealScalar::exp(-y.abs());
    let ch = |a| ((a * e) * 0.5) * e + a * 0.5 * decay;
    let sh = |a: f64| (((a * e) * 0.5) * e - a * 0.5 * decay) * y.signum();
    // sin(a + i y) = sin a cosh y + i cos a sinh y. For |y| >= 1, the
    // exponential factors are applied in halves to each coefficient, so a tiny
    // sin or cos survives against a huge cosh.
    if y.abs() < 1.0 {
        (
            Complex::new(s * RealScalar::cosh(y), c * RealScalar::sinh(y)),
            Complex::new(c * RealScalar::cosh(y), -s * RealScalar::sinh(y)),
        )
    } else {
        (Complex::new(ch(s), sh(c)), Complex::new(ch(c), -sh(s)))
    }
}

pub(super) fn sin_cos_pi<S: RealScalar>(
    z: Complex<S>,
) -> Result<(Complex<S>, Complex<S>), SciOxideError> {
    let (s, c) = scaled_phase(checked(z)?, 1.0);
    Ok((finish(s)?, finish(c)?))
}

pub(super) fn sin_pi<S: RealScalar>(z: Complex<S>) -> Result<Complex<S>, SciOxideError> {
    let z = checked(z)?;
    finish(scaled_phase(z, 1.0).0)
}

pub(super) fn cos_pi<S: RealScalar>(z: Complex<S>) -> Result<Complex<S>, SciOxideError> {
    let z = checked(z)?;
    finish(scaled_phase(z, 1.0).1)
}

pub(super) fn tan_pi<S: RealScalar>(z: Complex<S>) -> Result<Complex<S>, SciOxideError> {
    let z = checked(z)?;
    finish({
        let (s, c) = phase(z.re, 1.0);
        let y = radians(z.im, 1.0);
        if y == 0.0 && c == 0.0 {
            return Err(domain(crate::argument!("z")));
        }
        if y.abs() < 1.0 {
            let (s, c) = scaled_phase(z, 1.0);
            s / c
        } else {
            // The bounded t = e^(-2|y|) form of tan, as in `Complex::tan`.
            let e = RealScalar::exp(-2.0 * y.abs());
            let d = (1.0 - e) * (1.0 - e) + 4.0 * c * c * e;
            Complex::new(4.0 * s * c * e / d, (1.0 - e * e) * y.signum() / d)
        }
    })
}

pub(super) fn sin_cos_degrees<S: RealScalar>(
    z: Complex<S>,
) -> Result<(Complex<S>, Complex<S>), SciOxideError> {
    let (s, c) = scaled_phase(checked(z)?, 180.0);
    Ok((finish(s)?, finish(c)?))
}

pub(super) fn sin_degrees<S: RealScalar>(z: Complex<S>) -> Result<Complex<S>, SciOxideError> {
    let z = checked(z)?;
    finish(scaled_phase(z, 180.0).0)
}

pub(super) fn cos_degrees<S: RealScalar>(z: Complex<S>) -> Result<Complex<S>, SciOxideError> {
    let z = checked(z)?;
    finish(scaled_phase(z, 180.0).1)
}

pub(super) fn tan_degrees<S: RealScalar>(z: Complex<S>) -> Result<Complex<S>, SciOxideError> {
    let z = checked(z)?;
    finish({
        let (s, c) = phase(z.re, 180.0);
        let y = radians(z.im, 180.0);
        if y == 0.0 && c == 0.0 {
            return Err(domain(crate::argument!("z")));
        }
        if y.abs() < 1.0 {
            let (s, c) = scaled_phase(z, 180.0);
            s / c
        } else {
            // The bounded t = e^(-2|y|) form of tan, as in `Complex::tan`.
            let e = RealScalar::exp(-2.0 * y.abs());
            let d = (1.0 - e) * (1.0 - e) + 4.0 * c * c * e;
            Complex::new(4.0 * s * c * e / d, (1.0 - e * e) * y.signum() / d)
        }
    })
}

pub(super) fn sincpi<S: RealScalar>(z: Complex<S>) -> Result<Complex<S>, SciOxideError> {
    let z = checked(z)?;
    if z == <crate::Complex<f64> as crate::AlgebraicScalar>::ZERO {
        return finish(<crate::Complex<f64> as crate::AlgebraicScalar>::ONE);
    }
    if z.norm() < 0.25 {
        return sinc(finish(z * PI)?);
    }
    finish(sinc_quotient(z, true)?)
}

// Form sin(z) as 2 sin(z/2) cos(z/2) in extended range, divide before
// projecting. This preserves finite sinc values even when sin(z) overflows.
fn sinc_quotient(z: Complex<f64>, normalized: bool) -> Result<Complex<f64>, SciOxideError> {
    let half = z * 0.5;
    let (s, c) = if normalized {
        scaled_phase(half, 1.0)
    } else {
        (half.sin(), half.cos())
    };
    checked(s)?;
    checked(c)?;
    let product = ScaledValue::from_value(s)?
        .mul(ScaledValue::from_value(c)?)?
        .mul(ScaledValue::from_value(crate::Complex::new(
            2.0,
            <_ as crate::AlgebraicScalar>::ZERO,
        ))?)?
        .div(ScaledValue::from_value(z)?)?;
    let product = if normalized {
        product.div(ScaledValue::from_value(crate::Complex::new(
            PI,
            <_ as crate::AlgebraicScalar>::ZERO,
        ))?)?
    } else {
        product
    };
    product.round()
}
