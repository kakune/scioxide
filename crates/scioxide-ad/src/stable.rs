//! First and second derivatives of the stable transforms.
//!
//! The fourteen unary transforms are `map_scalar` rules `[f(x), f′(x), f″(x)]`:
//! the value comes from the `scioxide_core::backend::real_*` kernel, and both
//! derivatives are evaluated in `f64` and rounded once to `S`. A kernel error
//! (a non-finite argument or a point outside the domain) gives NaN.

use scioxide_core::backend::{self, ScalarKernelError};
use scioxide_core::{NumericScalar, RealScalar};

mod sinc;

/// A unary stable transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stable {
    Softplus,
    Sigmoid,
    LogSigmoid,
    Logit,
    Exprel,
    Cosm1,
    Sinc,
    Sincpi,
    SinPi,
    CosPi,
    TanPi,
    SinDegrees,
    CosDegrees,
    TanDegrees,
}

/// Returns the value of `function` at `x`, NaN on a kernel error.
#[inline]
pub(crate) fn value<S: RealScalar>(function: Stable, x: S) -> S {
    let value = match function {
        Stable::Softplus => backend::real_softplus(x),
        Stable::Sigmoid => backend::real_sigmoid(x),
        Stable::LogSigmoid => backend::real_log_sigmoid(x),
        Stable::Logit => backend::real_logit(x),
        Stable::Exprel => backend::real_exprel(x),
        Stable::Cosm1 => backend::real_cosm1(x),
        Stable::Sinc => backend::real_sinc(x),
        Stable::Sincpi => backend::real_sincpi(x),
        Stable::SinPi => backend::real_sin_pi(x),
        Stable::CosPi => backend::real_cos_pi(x),
        Stable::TanPi => backend::real_tan_pi(x),
        Stable::SinDegrees => backend::real_sin_degrees(x),
        Stable::CosDegrees => backend::real_cos_degrees(x),
        Stable::TanDegrees => backend::real_tan_degrees(x),
    };
    value.unwrap_or(S::NAN)
}

/// Returns `[f(x), f′(x), f″(x)]` for `function`; failed components are NaN.
#[inline]
pub(crate) fn rule<S: RealScalar>(function: Stable, x: S) -> [S; 3] {
    let (first, second) = derivatives(function, x.into_f64()).unwrap_or((f64::NAN, f64::NAN));
    [
        value(function, x),
        S::from_f64_ieee(first),
        S::from_f64_ieee(second),
    ]
}

/// Returns `σ′(x) = e^{−|x|}/(1 + e^{−|x|})²`, which never overflows.
pub(crate) fn logistic_first(x: f64) -> f64 {
    let e = RealScalar::exp(-NumericScalar::abs(x));
    (e / (1.0 + e)) / (1.0 + e)
}

fn derivatives(function: Stable, x: f64) -> Result<(f64, f64), ScalarKernelError> {
    Ok(match function {
        Stable::Softplus => (backend::real_sigmoid(x)?, logistic_first(x)),
        // σ″ = σ′(1 − 2σ) and 1 − 2σ(x) = −tanh(x/2), which avoids cancellation near 0.
        Stable::Sigmoid => {
            let d = logistic_first(x);
            (d, -RealScalar::tanh(x * 0.5) * d)
        }
        // d/dx ln σ(x) = σ(−x); second derivative −σ′(x).
        Stable::LogSigmoid => (backend::real_sigmoid(-x)?, -logistic_first(x)),
        Stable::Logit => {
            // Outside (0, 1) the kernel error makes the derivatives NaN too.
            backend::real_logit(x)?;
            // The small numerator is formed before division to keep the curvature near
            // one half; sequential division avoids premature underflow.
            let a = 1.0 / x;
            let b = 1.0 / (1.0 - x);
            (a + b, (((2.0 * x - 1.0) / x) / (1.0 - x) / x) / (1.0 - x))
        }
        Stable::Cosm1 => (-RealScalar::sin(x), -RealScalar::cos(x)),
        Stable::Exprel => exprel(x)?,
        Stable::Sinc | Stable::Sincpi => {
            let normalized = function == Stable::Sincpi;
            let scale = if normalized {
                core::f64::consts::PI
            } else {
                1.0
            };
            let z = x * scale;
            if NumericScalar::abs(z) < 1.0 {
                sinc_series(z, scale)
            } else {
                sinc::derivatives(x, normalized)?
            }
        }
        Stable::SinPi | Stable::CosPi | Stable::TanPi => {
            // Exact reduction in units of π keeps the zeros at integers.
            let (s, c) = backend::real_sin_cos_pi(x)?;
            scaled_trigonometric(function, core::f64::consts::PI, s, c)
        }
        Stable::SinDegrees | Stable::CosDegrees | Stable::TanDegrees => {
            let (s, c) = backend::real_sin_cos_degrees(x)?;
            scaled_trigonometric(function, core::f64::consts::PI / 180.0, s, c)
        }
    })
}

fn scaled_trigonometric(function: Stable, scale: f64, s: f64, c: f64) -> (f64, f64) {
    match function {
        Stable::SinPi | Stable::SinDegrees => (scale * c, -scale * scale * s),
        Stable::CosPi | Stable::CosDegrees => (-scale * s, -scale * scale * c),
        // tan′ = scale/cos² and tan″ = 2·scale·tan·tan′; dividing by c twice avoids
        // squaring first.
        _ => {
            let d = (scale / c) / c;
            (d, 2.0 * scale * (s / c) * d)
        }
    }
}

fn exprel(x: f64) -> Result<(f64, f64), ScalarKernelError> {
    Ok(if NumericScalar::abs(x) < 2.0 {
        // exprel′ = Σ xᵏ/(k!(k + 2)) and exprel″ = Σ xᵏ/(k!(k + 3)); 24 terms put the
        // truncation error far below ε for |x| < 2.
        let mut term = 1.0;
        let mut d = 0.5;
        let mut dd = 1.0 / 3.0;
        for k in 1..=24 {
            term *= x / f64::from(k);
            d += term / f64::from(k + 2);
            dd += term / f64::from(k + 3);
        }
        (d, dd)
    } else if x < -50.0 {
        // eˣ is negligible: exprel ≈ −1/x with derivatives 1/x² and −2/x³.
        let r = 1.0 / x;
        (r * r, -2.0 * r * r * r)
    } else if x <= -2.0 {
        // Direct exponential numerators avoid cancelling O(1/|x|) terms.
        let e = RealScalar::exp(x);
        let r = 1.0 / x;
        (
            (1.0 + e * (x - 1.0)) * r * r,
            (e * (x * x - 2.0 * x + 2.0) - 2.0) * r * r * r,
        )
    } else {
        // eˣ = 1 + x·exprel(x) gives exprel′ = (eˣ − exprel)/x and
        // exprel″ = (eˣ − 2 exprel′)/x in terms of v = exprel(x) and r = 1/x.
        let v = backend::real_exprel(x)?;
        let r = 1.0 / x;
        let d = v * (1.0 - r) + r;
        (d, v - 2.0 * d * r + r)
    })
}

fn sinc_series(z: f64, scale: f64) -> (f64, f64) {
    // Nine terms through z¹⁷ and z¹⁶ bound both tails below ε/4 for |z| < 1.
    const FIRST: [f64; 9] = [
        -1.0 / 3.0,
        1.0 / 30.0,
        -1.0 / 840.0,
        1.0 / 45360.0,
        -1.0 / 3991680.0,
        1.0 / 518918400.0,
        -1.0 / 93405312000.0,
        1.0 / 22230464256000.0,
        -1.0 / 6758061133824000.0,
    ];
    const SECOND: [f64; 9] = [
        -1.0 / 3.0,
        1.0 / 10.0,
        -1.0 / 168.0,
        1.0 / 6480.0,
        -1.0 / 443520.0,
        1.0 / 47174400.0,
        -1.0 / 7185024000.0,
        1.0 / 1482030950400.0,
        -1.0 / 397533007872000.0,
    ];
    let q = z * z;
    let d = z * FIRST.iter().rev().fold(0.0, |v, c| c + q * v);
    let dd = SECOND.iter().rev().fold(0.0, |v, c| c + q * v);
    (scale * d, scale * scale * dd)
}
