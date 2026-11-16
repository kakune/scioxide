//! GLM families with their canonical links, means, variances and unit deviances.
use crate::arithmetic::deviance_series;
use crate::regression::{finite, invalid, number};
use scioxide_core::{NumericalIssue, RealScalar, Scaled, SciOxideError};

/// Selects the exponential-dispersion family of a generalized linear model.
///
/// Each family fixes the response domain, the variance function `V(μ)`, the
/// link `η = g(μ)` (see [`Self::link`]) and the unit deviance `d(y, μ)`:
///
/// | Family | Domain | `V(μ)` | `g(μ)` | `d(y, μ)` |
/// |---|---|---|---|---|
/// | Gaussian | `y ∈ ℝ` | `1` | `μ` | `(y − μ)²` |
/// | Binomial | `0 ≤ y ≤ 1` | `μ(1 − μ)` | `ln(μ/(1 − μ))` | `2[y ln(y/μ) + (1 − y) ln((1 − y)/(1 − μ))]` |
/// | Poisson | `y ≥ 0` | `μ` | `ln μ` | `2[y ln(y/μ) − (y − μ)]` |
/// | Gamma | `y > 0` | `μ²` | `1/μ` | `2[(y − μ)/μ − ln(y/μ)]` |
/// | InverseGaussian | `y > 0` | `μ³` | `1/μ²` | `(y − μ)² / (y μ²)` |
///
/// with `0 ln 0 = 0`. A binomial response is the proportion of successes in
/// `m` trials, where `m` is set per row at preparation. The Gamma and inverse
/// Gaussian links need `η > 0`. Unit deviances are computed in an extended
/// exponent range, so they do not overflow before weighting, and with series
/// expansions when `y` is close to `μ`, to avoid cancellation.
///
/// # References
///
/// P. McCullagh and J. A. Nelder, *Generalized Linear Models*, 2nd ed.,
/// Chapman & Hall, 1989, ch. 2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlmFamily {
    /// Normal response with constant variance and identity link.
    Gaussian,
    /// Binomial proportion, including Bernoulli responses, with logit link.
    Binomial,
    /// Poisson counts with log link.
    Poisson,
    /// Gamma response with reciprocal link.
    Gamma,
    /// Inverse Gaussian response with reciprocal-square link.
    InverseGaussian,
}

/// Names the link function `g` in `η = g(μ)`.
///
/// Each [`GlmFamily`] uses one fixed link, returned by [`GlmFamily::link`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlmLink {
    /// `η = μ`, used by the Gaussian family.
    Identity,
    /// `η = ln(μ / (1 − μ))`, used by the binomial family.
    Logit,
    /// `η = ln μ`, used by the Poisson family.
    Log,
    /// `η = 1/μ`, used by the Gamma family.
    Inverse,
    /// `η = 1/μ²`, used by the inverse Gaussian family.
    InverseSquared,
}

impl GlmFamily {
    /// Returns the canonical link of this family, up to sign.
    pub const fn link(self) -> GlmLink {
        match self {
            Self::Gaussian => GlmLink::Identity,
            Self::Binomial => GlmLink::Logit,
            Self::Poisson => GlmLink::Log,
            Self::Gamma => GlmLink::Inverse,
            Self::InverseGaussian => GlmLink::InverseSquared,
        }
    }

    pub(super) fn valid_response<S: RealScalar>(self, y: S) -> bool {
        y.is_finite()
            && match self {
                Self::Gaussian => true,
                Self::Binomial => y >= S::ZERO && y <= S::ONE,
                Self::Poisson => y >= S::ZERO,
                Self::Gamma | Self::InverseGaussian => y > S::ZERO,
            }
    }

    pub(super) fn mean<S: RealScalar>(self, eta: S) -> Option<(S, S, S)> {
        if !eta.is_finite() {
            return None;
        }
        let (mu, derivative, variance) = match self {
            Self::Gaussian => (eta, S::ONE, S::ONE),
            Self::Binomial => {
                let mu = if eta >= S::ZERO {
                    S::ONE / (S::ONE + (-eta).exp())
                } else {
                    let z = eta.exp();
                    z / (S::ONE + z)
                };
                let v = mu * (S::ONE - mu);
                (mu, v, v)
            }
            Self::Poisson => {
                let mu = eta.exp();
                (mu, mu, mu)
            }
            Self::Gamma => {
                if eta <= S::ZERO {
                    return None;
                }
                let mu = eta.recip();
                (mu, -mu * mu, mu * mu)
            }
            Self::InverseGaussian => {
                if eta <= S::ZERO {
                    return None;
                }
                let mu = eta.sqrt().recip();
                (mu, -mu * mu * mu / number::<S>(2.0).ok()?, mu * mu * mu)
            }
        };
        if !mu.is_finite()
            || !derivative.is_finite()
            || !variance.is_finite()
            || derivative == S::ZERO
            || variance <= S::ZERO
            || (self == Self::Binomial && (mu <= S::ZERO || mu >= S::ONE))
            || (self == Self::Poisson && mu <= S::ZERO)
        {
            return None;
        }
        Some((mu, derivative, variance))
    }

    pub(super) fn link_value<S: RealScalar>(self, mu: S) -> Result<S, SciOxideError> {
        let value = match self {
            Self::Gaussian => mu,
            Self::Binomial => (mu / (S::ONE - mu)).ln(),
            Self::Poisson => mu.ln(),
            Self::Gamma => mu.recip(),
            Self::InverseGaussian => (mu * mu).recip(),
        };
        finite(value)
    }

    // Keep the unit deviance extended until weights and objective normalization
    // have been applied. Even an overflowing unit deviance can give a finite
    // weighted objective. These pure kernels use fixed storage only.
    pub(super) fn deviance<S: RealScalar>(self, y: S, mu: S) -> Result<Scaled<S>, SciOxideError> {
        let ys = Scaled::from_value(y)?;
        let ms = Scaled::from_value(mu)?;
        if !self.valid_response(y)
            || (self != Self::Gaussian && mu <= S::ZERO)
            || (self == Self::Binomial && mu >= S::ONE)
        {
            return Err(invalid(
                scioxide_core::argument!("response"),
                scioxide_core::InvalidInputReason::Inconsistent,
                "GLM deviance domain",
            ));
        }
        if y == mu {
            return Ok(Scaled::ZERO);
        }
        let two = Scaled::from_value(number::<S>(2.0)?)?;
        let value = match self {
            Self::Gaussian => ys.sub(ms)?.square()?,
            Self::Binomial => {
                let delta = ys.sub(ms)?;
                // Reuse the original difference: subtracting rounded complements
                // can erase an adjacent response just below one half.
                poisson_deviance_part(ys, ms, delta)?
                    .add(poisson_deviance_part(
                        Scaled::ONE.sub(ys)?,
                        Scaled::ONE.sub(ms)?,
                        -delta,
                    )?)?
                    .mul(two)?
            }
            Self::Poisson => poisson_deviance_part(ys, ms, ys.sub(ms)?)?.mul(two)?,
            Self::Gamma => {
                let delta = ys.sub(ms)?;
                if close_means(delta, ms)? {
                    let t = delta.div(ms)?;
                    t.square()?
                        .mul(Scaled::from_value(deviance_series(t.round()?, false)?)?)?
                } else {
                    let ratio = ys.div(ms)?;
                    ratio
                        .sub(Scaled::ONE)?
                        .sub(Scaled::from_value(finite(ratio.ln_abs()?)?)?)?
                }
                .mul(two)?
            }
            Self::InverseGaussian => ys.sub(ms)?.square()?.div(ys)?.div(ms.square()?)?,
        };
        nonnegative_deviance(value)
    }

    // Canonical score d(log L)/d eta; simplifying the derivative/variance
    // quotient avoids multiplying large residuals by tiny derivatives first.
    pub(super) fn score<S: RealScalar>(self, y: S, mu: S) -> Result<Scaled<S>, SciOxideError> {
        let residual = Scaled::from_value(y)?.sub(Scaled::from_value(mu)?)?;
        match self {
            Self::Gaussian | Self::Binomial | Self::Poisson => Ok(residual),
            Self::Gamma => Ok(-residual),
            Self::InverseGaussian => residual.mul(Scaled::HALF).map(core::ops::Neg::neg),
        }
    }
}

pub(super) fn nonnegative_deviance<S: RealScalar>(
    value: Scaled<S>,
) -> Result<Scaled<S>, SciOxideError> {
    if value.cmp_value(Scaled::ZERO).is_lt() {
        return Err(SciOxideError::numerical_instability(
            NumericalIssue::Breakdown,
            "negative GLM deviance",
        ));
    }
    Ok(value)
}

fn close_means<S: RealScalar>(delta: Scaled<S>, mu: Scaled<S>) -> Result<bool, SciOxideError> {
    Ok(delta
        .abs_cmp(mu.mul(Scaled::from_value(number::<S>(0.125)?)?)?)
        .is_le())
}

fn poisson_deviance_part<S: RealScalar>(
    y: Scaled<S>,
    mu: Scaled<S>,
    delta: Scaled<S>,
) -> Result<Scaled<S>, SciOxideError> {
    nonnegative_deviance(crate::arithmetic::unit_deviance(y, mu, delta)?)
}
