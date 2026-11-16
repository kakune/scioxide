//! Real arithmetic on the pi/4 ray: DLMF 10.31 and 10.40/10.67.
//! Series keep two words through cancellation. Asymptotic value/derivative
//! sums share powers, phase and exponential scaling, including the recessive
//! contribution to I in the overlap region.
use crate::support::{C, Jet, JetData, JetEstimate, Work, real, wide::Wide};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use scioxide_core::RealScalar;
use scioxide_core::{SciOxideError, algorithm_component};

const INV_SQRT_2: Wide = Wide {
    hi: core::f64::consts::FRAC_1_SQRT_2,
    lo: -4.833646656726457e-17,
};

// Normalized asymptotic coefficients have modulus below two, so ordinary
// Cartesian multiplication cannot overflow and needs no scaled division path.
#[inline]
fn multiply(a: C, b: C) -> C {
    C::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

#[inline]
fn magnitude(a: C) -> f64 {
    a.re.abs() + a.im.abs()
}

fn pack(v: C, d: C, error: f64, derror: f64, scale: f64) -> [JetEstimate; 2] {
    core::array::from_fn(|i| {
        JetEstimate::from(JetData {
            y: Jet {
                v: real(if i == 0 { v.re } else { v.im }),
                d: real(if i == 0 { d.re } else { d.im }),
            },
            scale: Jet::constant(real(scale)),
            error,
            derivative_error: Some(derror),
            method: algorithm_component!("Kelvin_real_pair"),
        })
    })
}

fn series(
    a: f64,
    second: bool,
    scaled: bool,
    w: &mut Work,
) -> Result<[JetEstimate; 2], SciOxideError> {
    let x = Wide::from(a);
    let q = x * x.mul_f64(0.25);
    let mut term = Wide::from(1.0);
    let mut dterm = x.mul_f64(0.5);
    let mut sum = [Wide::from(1.0), Wide::ZERO];
    let mut dsum = [Wide::ZERO; 2];
    let mut harmonic = Wide::ZERO;
    let mut hsum = [Wide::ZERO; 2];
    let mut hdsum = [Wide::ZERO; 2];
    let mut magnitude = 1.0;
    let mut dmagnitude = 0.0;
    let mut k = 0usize;
    loop {
        w.try_step()?;
        k += 1;
        let n = k as f64;
        term = term * q / Wide::from(n * n);
        if k > 1 {
            dterm = dterm * q / Wide::from(n * (n - 1.0));
        }
        let part = k % 2;
        let sign = if (k / 2).is_multiple_of(2) { 1.0 } else { -1.0 };
        sum[part] = sum[part] + term.mul_f64(sign);
        dsum[part] = dsum[part] + dterm.mul_f64(sign);
        if second {
            harmonic = harmonic + Wide::from(1.0) / Wide::from(n);
            hsum[part] = hsum[part] + (term * harmonic).mul_f64(sign);
            hdsum[part] = hdsum[part] + (dterm * harmonic).mul_f64(sign);
        }
        magnitude += term.hi.abs() * (1.0 + harmonic.hi.abs());
        dmagnitude += dterm.hi.abs() * (1.0 + harmonic.hi.abs());
        if k > 4 && term.hi.abs() <= 1e-33 * (sum[0].hi.abs() + sum[1].hi.abs()) {
            break;
        }
    }
    let (v, d) = if second {
        w.try_step()?;
        let log = x.ln() - crate::support::constants::LN_2 + crate::support::constants::EULER_GAMMA;
        let angle = crate::support::constants::FRAC_PI_4;
        (
            [
                hsum[0] - log * sum[0] + angle * sum[1],
                hsum[1] - log * sum[1] - angle * sum[0],
            ],
            [
                hdsum[0] - log * dsum[0] + angle * dsum[1] - sum[0] / x,
                hdsum[1] - log * dsum[1] - angle * dsum[0] - sum[1] / x,
            ],
        )
    } else {
        (sum, dsum)
    };
    let s = (x * INV_SQRT_2).value();
    let factor = if scaled {
        (if second { s } else { -s }).exp()
    } else {
        1.0
    };
    let v = C::new(v[0].value() * factor, v[1].value() * factor);
    let d = C::new(d[0].value() * factor, d[1].value() * factor);
    let rounding = 128.0 * k as f64 * f64::EPSILON * f64::EPSILON;
    let relative = 16.0 * f64::EPSILON * (1.0 + if scaled { s } else { 0.0 });
    Ok(pack(
        v,
        d,
        rounding * magnitude * factor + relative * v.norm(),
        rounding * dmagnitude * factor + relative * d.norm(),
        0.0,
    ))
}

fn rotation(angle: Wide) -> C {
    let (s, c) = angle.hi.sin_cos();
    if angle.lo.abs() < 1e-8 {
        C::new(c - angle.lo * s, s + angle.lo * c)
    } else {
        let (sl, cl) = angle.lo.sin_cos();
        C::new(c * cl - s * sl, s * cl + c * sl)
    }
}

/// a >= 20; compose both phase words before restoring the real exponential.
fn asymptotic(
    a: f64,
    second: bool,
    scaled: bool,
    w: &mut Work,
) -> Result<[JetEstimate; 2], SciOxideError> {
    let inverse = INV_SQRT_2.hi / a;
    let mut terms = [real(1.0); 2];
    let mut growing = terms;
    let mut recessive = terms;
    let mut magnitudes = [1.0; 2];
    let mut previous = f64::MAX;
    let mut k = 0usize;
    let tail = loop {
        w.try_step()?;
        k += 1;
        let odd = (2 * k - 1) as f64;
        for (n, term) in terms.iter_mut().enumerate() {
            let factor = inverse * ((4.0 * n as f64 - odd * odd) / (8.0 * k as f64));
            *term = C::new((term.re + term.im) * factor, (term.im - term.re) * factor);
        }
        let size = magnitude(terms[0]).max(magnitude(terms[1]));
        if size > previous {
            break size;
        }
        for n in 0..2 {
            recessive[n] += terms[n];
            growing[n] += terms[n].scale(if k.is_multiple_of(2) { 1.0 } else { -1.0 });
            magnitudes[n] += magnitude(terms[n]);
        }
        if size <= f64::EPSILON * 0.0625 {
            break size;
        }
        previous = size;
    };
    let phase = Wide::from(a) * INV_SQRT_2;
    let eighth = crate::support::constants::FRAC_PI_4.mul_f64(0.5);
    let kr = rotation(-phase - eighth).scale((core::f64::consts::PI / 2.0).sqrt() / a.sqrt());
    let ray = C::new(INV_SQRT_2.hi, INV_SQRT_2.hi);
    let (mut v, mut d) = if second {
        (
            multiply(kr, recessive[0]),
            -multiply(ray, multiply(kr, recessive[1])),
        )
    } else {
        let ir =
            rotation(phase - eighth).scale(1.0 / (2.0 * core::f64::consts::PI).sqrt() / a.sqrt());
        // I0 has +i K0/pi, I1 has -i K1/pi on this ray.
        let small = if a < 28.0 {
            (-2.0 * phase.hi).exp() / core::f64::consts::PI
        } else {
            0.0
        };
        (
            multiply(ir, growing[0]) + multiply(C::new(0.0, small), multiply(kr, recessive[0])),
            multiply(
                ray,
                multiply(ir, growing[1]) - multiply(C::new(0.0, small), multiply(kr, recessive[1])),
            ),
        )
    };
    let exponent = if scaled {
        0.0
    } else if second {
        -phase.value()
    } else {
        phase.value()
    };
    let factor = exponent.exp();
    // Preserve the unrounded scale when exp itself enters the subnormal
    // lattice; multiplying that rounded factor would lose significant bits.
    let expand = factor.is_finite() && factor >= f64::MIN_POSITIVE;
    let factor = if expand { factor } else { 1.0 };
    let scale = if expand { 0.0 } else { exponent };
    // Remainder and accumulated coefficient/phase/scale rounding, in pair units.
    let relative = 16.0 * tail
        + 32.0 * f64::EPSILON * (k as f64 + 1.0) * magnitudes[0].max(magnitudes[1])
        + 8.0 * f64::EPSILON * (1.0 + phase.hi.abs());
    v *= factor;
    d *= factor;
    Ok(pack(v, d, relative * v.norm(), relative * d.norm(), scale))
}

pub(super) fn pair(
    a: f64,
    second: bool,
    scaled: bool,
    w: &mut Work,
) -> Result<[JetEstimate; 2], SciOxideError> {
    if a < 20.0 {
        series(a, second, scaled, w)
    } else {
        asymptotic(a, second, scaled, w)
    }
}
