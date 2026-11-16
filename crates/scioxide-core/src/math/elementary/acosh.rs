//! acosh kernel. See the crate's THIRD_PARTY_NOTICES.md.
use super::{log, log1p, sqrt};

const LN2: f64 = core::f64::consts::LN_2; /* 0x3fe62e42,  0xfefa39ef*/

/// Inverse hyperbolic cosine (f64)
///
/// Calculates the inverse hyperbolic cosine of `x`.
/// Is defined as `log(x + sqrt(x*x-1))`.
/// `x` must be a number greater than or equal to 1.
pub(in crate::math) fn acosh(x: f64) -> f64 {
    if x < 1.0 {
        return f64::NAN;
    }
    let u = x.to_bits();
    let e = ((u >> 52) as usize) & 0x7ff;

    /* x < 1 domain error is handled in the called functions */

    if e < 0x3ff + 1 {
        /* |x| < 2, up to 2ulp error in [1,1.125] */
        return log1p(x - 1.0 + sqrt((x - 1.0) * (x - 1.0) + 2.0 * (x - 1.0)));
    }
    if e < 0x3ff + 26 {
        /* |x| < 0x1p26 */
        return log(2.0 * x - 1.0 / (x + sqrt(x * x - 1.0)));
    }
    /* |x| >= 0x1p26 or nan */
    log(x) + LN2
}
