//! Unit deviances for GLM families and information measures.
use super::Arithmetic;
use crate::error_helpers::{finite, number};
use scioxide_core::{RealScalar, SciOxideError};

// For |t| <= 1/8 evaluate the coefficient of t^2 in
// (1+t)ln(1+t)-t (Poisson/KL), or t-ln(1+t) (Gamma). Factoring
// out t^2 preserves both near-equality accuracy and subnormal-scale results.
// The geometric tail after 64 terms is below 2^-180 for either scalar width.
pub(crate) fn deviance_series<S: RealScalar>(t: S, poisson: bool) -> Result<S, SciOxideError> {
    let mut sum = S::ZERO;
    let mut power = S::ONE;
    for n in 2..=64 {
        let term = power / number::<S>(f64::from(n))?;
        let term = if poisson {
            term / number::<S>(f64::from(n - 1))?
        } else {
            term
        };
        sum += term;
        if term.abs() <= S::EPSILON * sum.abs() * number::<S>(0.25)? {
            break;
        }
        power *= -t;
    }
    finite(sum)
}

/// Returns the Poisson (generalized Kullback–Leibler) unit deviance `y ln(y/μ) − δ`.
///
/// Here `μ > 0`, `y ≥ 0` and `δ = y − μ` is computed by the caller from the
/// original values. A zero response contributes `μ`. When `|δ| ≤ μ/8` the
/// factored series `δ (δ/μ) c(δ/μ)` avoids cancellation between the logarithm
/// and `δ`.
///
/// # Errors
///
/// Returns range errors of the selected arithmetic.
#[inline]
pub(crate) fn unit_deviance<N: Arithmetic>(y: N, mu: N, delta: N) -> Result<N, SciOxideError> {
    if y.is_zero() {
        return Ok(mu);
    }
    if delta.abs_le(mu.mul(N::number(0.125)?)?) {
        let ratio = delta.div(mu)?;
        delta
            .mul(ratio)?
            .mul(N::from_real(deviance_series(ratio.real()?, true)?)?)
    } else {
        y.mul(N::from_real(y.ln_ratio(mu)?)?)?.sub(delta)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use scioxide_core::Scaled;

    #[test]
    fn native_and_extended_unit_deviances_agree_on_both_branches() {
        for (y, mu) in [(1.05_f64, 1.0), (3.0, 1.0), (0.0, 2.5), (1e-3, 2.0)] {
            let native = unit_deviance(y, mu, y - mu).unwrap();
            let delta = Scaled::from_value(y)
                .unwrap()
                .sub(Scaled::from_value(mu).unwrap());
            let extended = unit_deviance(
                Scaled::from_value(y).unwrap(),
                Scaled::from_value(mu).unwrap(),
                delta.unwrap(),
            )
            .unwrap()
            .round()
            .unwrap();
            assert!((extended - native).abs() <= 4.0 * f64::EPSILON * native.abs());
            if (y - mu).abs() > mu / 8.0 {
                // Away from equality the direct form is well conditioned.
                let direct = if y == 0.0 {
                    mu
                } else {
                    y * (y / mu).ln() - (y - mu)
                };
                assert!((native - direct).abs() <= 4.0 * f64::EPSILON * direct.abs());
            }
        }
        // Near equality the series keeps relative accuracy where the direct
        // form cancels completely.
        let (y, mu) = (1.0 + 2.0_f64.powi(-40), 1.0);
        let value = unit_deviance(y, mu, y - mu).unwrap();
        let t = 2.0_f64.powi(-40);
        assert!((value - 0.5 * t * t).abs() <= 1e-12 * value);
    }
}
