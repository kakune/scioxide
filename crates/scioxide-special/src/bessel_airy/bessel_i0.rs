//! Dedicated scaled I₀ for positive finite `f64` arguments.
//!
//! Chebyshev expansions on [0,8] and on the inverted interval [8,∞), with
//! the 64-ulp real Bessel error envelope.

// Cephes 2.8 Chebyshev tables, Copyright 1984, 1987, 2000 Stephen L.
// Moshier; adapted from SciPy xsf `include/xsf/cephes/i0.h` (2024).
// Attribution and permission: `THIRD_PARTY_NOTICES.md`.
use crate::support::{Work, polynomial::clenshaw};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use scioxide_core::RealScalar;
use scioxide_core::SciOxideError;

const A: [f64; 30] = [
    0.6767952744094761,
    -0.3046826723431984,
    0.17162090152220877,
    -0.09490109704804764,
    0.04930528423967071,
    -0.02373741480589947,
    0.010546460394594998,
    -0.004324309995050576,
    0.0016394756169413357,
    -0.0005763755745385824,
    0.00018850288509584165,
    -5.754195010082104e-05,
    1.6448448070728896e-05,
    -4.4167383584587505e-06,
    1.1173875391201037e-06,
    -2.670793853940612e-07,
    6.046995022541919e-08,
    -1.300025009986248e-08,
    2.6598237246823866e-09,
    -5.189795601635263e-10,
    9.675809035373237e-11,
    -1.726826291441556e-11,
    2.95505266312964e-12,
    -4.856446783111929e-13,
    7.676185498604936e-14,
    -1.1685332877993451e-14,
    1.715391285555133e-15,
    -2.431279846547955e-16,
    3.3307945188222384e-17,
    -4.4153416464793395e-18,
];

const B: [f64; 25] = [
    0.8044904110141088,
    0.0033691164782556943,
    6.889758346916825e-05,
    2.8913705208347567e-06,
    2.0489185894690638e-07,
    2.266668990498178e-08,
    3.3962320257083865e-09,
    4.94060238822497e-10,
    1.1889147107846439e-11,
    -3.1499165279632416e-11,
    -1.3215811840447713e-11,
    -1.7941785315068062e-12,
    7.180124451383666e-13,
    3.8527783827421426e-13,
    1.54008621752141e-14,
    -4.150569347287222e-14,
    -9.554846698828307e-15,
    3.8116806693526224e-15,
    1.7725601330565263e-15,
    -3.425485619677219e-16,
    -2.8276239805165836e-16,
    3.461222867697461e-17,
    4.46562142029676e-17,
    -4.830504485944182e-18,
    -7.233180487874754e-18,
];

/// `exp(-x) I₀(x)`, without ever forming the growing exponential.
/// Each coefficient counts one work unit before evaluation.
pub(super) fn scaled(x: f64, work: &mut Work) -> Result<f64, SciOxideError> {
    let (z, coefficients, divisor): (f64, &[f64], f64) = if x <= 8.0 {
        (x * 0.25 - 1.0, &A, 1.0)
    } else {
        (16.0 / x - 1.0, &B, x.sqrt())
    };
    work.steps(coefficients.len())?;
    let (first, second) = clenshaw(coefficients.iter().skip(1), 0.0, |a, b, &c| {
        Ok::<_, SciOxideError>((2.0 * z) * a - b + c)
    })?;
    Ok((z * first - second + coefficients[0] * 0.5) / divisor)
}
