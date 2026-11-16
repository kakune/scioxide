//! Symmetrized laws `±Y`: a fair sign times a nonnegative magnitude `Y`.
//!
//! The density is `f_Y(|x|)/2`, the tail beyond `x` on the side away from zero
//! is `S_Y(|x|)/2`, and the other tail is its complement. Inverses use the
//! magnitude's inverse survival at `2 min(p, 1-p)`, so no `1 - 2p` cancels.
use crate::Rng;
use crate::law::{finite, log1mexp};
use scioxide_core::{RealScalar, SciOxideError};

/// Logarithmic tail of `±Y` at `x`, in the working precision of `log_sf`, the
/// logarithmic survival function of the magnitude.
pub(super) fn log_tail<P: RealScalar>(
    x: P,
    upper: bool,
    log_sf: impl FnOnce(P) -> Result<P, SciOxideError>,
) -> Result<P, SciOxideError> {
    let x = if upper { -x } else { x };
    let small = log_sf(x.abs())? - P::LN_2;
    if x < P::ZERO {
        Ok(small)
    } else {
        log1mexp(small)
    }
}

/// Lower (or upper) inverse of `±Y` from the magnitude's inverse survival
/// `isf`; zero is the exact median and the endpoints are infinite.
pub(super) fn inverse<S: RealScalar>(
    p: S,
    upper: bool,
    isf: impl FnOnce(S) -> Result<S, SciOxideError>,
) -> Result<S, SciOxideError> {
    if upper {
        return Ok(-inverse(p, false, isf)?);
    }
    if p == S::ZERO {
        return Ok(S::NEG_INFINITY);
    }
    if p == S::ONE {
        return Ok(S::INFINITY);
    }
    if p == S::HALF {
        return Ok(S::ZERO);
    }
    let lower = p < S::HALF;
    let tail = S::TWO * if lower { p } else { S::ONE - p };
    let magnitude = isf(tail)?;
    finite(if lower { -magnitude } else { magnitude })
}

/// One draw: a fair sign from one open uniform, then the magnitude.
pub(super) fn draw<S: RealScalar, R: Rng<S>>(
    rng: &mut R,
    magnitude: impl FnOnce(&mut R) -> Result<S, SciOxideError>,
) -> Result<S, SciOxideError> {
    let sign = if crate::util::validated_open_float(rng)? < S::HALF {
        -S::ONE
    } else {
        S::ONE
    };
    finite(sign * magnitude(rng)?)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // ±Exp(1) is the standard Laplace law: F(x) = exp(x)/2 for x < 0.
    #[test]
    fn exponential_magnitude_is_laplace() {
        for x in [-30.0_f64, -1.5, -1e-9, 0.0, 2.0, 40.0] {
            let lower = log_tail(x, false, |y| Ok(-y)).unwrap();
            let upper = log_tail(x, true, |y| Ok(-y)).unwrap();
            let expected = if x < 0.0 {
                x - core::f64::consts::LN_2
            } else {
                (1.0 - 0.5 * (-x).exp()).ln()
            };
            assert!(
                (lower - expected).abs() <= 1e-15 * (1.0 + expected.abs()),
                "{x}"
            );
            assert!((lower.exp() + upper.exp() - 1.0).abs() < 1e-15);
        }
        for p in [1e-300_f64, 0.2, 0.5, 0.75, 1.0 - 1e-12] {
            let x = inverse(p, false, |t: f64| Ok(-t.ln())).unwrap();
            let back = log_tail(x, false, |y| Ok(-y)).unwrap().exp();
            assert!(
                (back - p).abs() <= 1e-12 * p.min(1.0 - p).max(1e-300),
                "{p}"
            );
            assert_eq!(inverse(p, true, |t: f64| Ok(-t.ln())).unwrap(), -x);
        }
        assert_eq!(
            inverse(0.0, false, |t: f64| Ok(-t.ln())).unwrap(),
            f64::NEG_INFINITY
        );
        assert_eq!(
            inverse(1.0, false, |t: f64| Ok(-t.ln())).unwrap(),
            f64::INFINITY
        );
        assert_eq!(inverse(0.5, false, |t: f64| Ok(-t.ln())).unwrap(), 0.0);
    }
}
