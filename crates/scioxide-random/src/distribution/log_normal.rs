//! Log-normal distribution.

use scioxide_core::backend::checked_final_rounding;
use scioxide_core::{RealScalar, SciOxideError};
use scioxide_core::{checked_float_product, checked_sum};

use crate::distribution::normal::Normal;
use crate::traits::{Cdf, Density, Distribution, Mean, Quantile, Rng, Variance};

/// Log-normal distribution with parameters `μ` and `σ`: the law of `e^X` for
/// `X ~ Normal(μ, σ)`.
///
/// ```text
/// f(x) = exp(−(ln x − μ)² / (2σ²)) / (xσ √(2π)),   x > 0, σ > 0.
/// ```
///
/// The quantiles at `p = 0` and `p = 1` are `0` and `+∞`. Every other quantile
/// and every sample is positive and finite.
///
/// # Examples
///
/// ```
/// use scioxide_random::{Cdf, Distribution, LogNormal, Mean, Quantile, SeedableRng};
/// use scioxide_random::{SplitMix64, Support, Variance};
/// let law = LogNormal::new(0.0_f64, 0.5)?;
/// assert!((law.cdf(1.0)? - 0.5).abs() < 1e-16); // median exp(μ)
/// assert!((law.mean()? - 0.125_f64.exp()).abs() < 1e-15);
/// assert!((law.variance()? - 0.25_f64.exp_m1() * 0.25_f64.exp()).abs() < 1e-15);
/// assert!((law.cdf(law.quantile(0.3)?)? - 0.3).abs() < 1e-15);
/// let mut rng = SplitMix64::from_seed(24);
/// assert!(law.contains(law.sample(&mut rng)?)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct LogNormal<S> {
    mu: S,
    sigma: S,
    normal: Normal<S>,
}

impl<S: RealScalar> LogNormal<S> {
    /// Creates a log-normal distribution with parameters `mu` and `sigma`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `mu` or `sigma` is NaN or infinite.
    /// - `DomainError` if `sigma ≤ 0`.
    pub fn new(mu: S, sigma: S) -> Result<Self, SciOxideError> {
        crate::law::argument(mu, "mu")?;
        crate::law::positive(sigma, scioxide_core::argument!("sigma"))?;
        let normal = Normal::new(mu, sigma)?;
        Ok(Self { mu, sigma, normal })
    }
}

impl<S: RealScalar> Distribution<S> for LogNormal<S> {
    fn sample<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        let x: S = self.normal.sample(rng)?;
        positive_log_normal_result(x.exp())
    }
}

fn positive_log_normal_result<S: RealScalar>(value: S) -> Result<S, SciOxideError> {
    let value = checked_final_rounding(value, "log-normal result exceeds the finite scalar range")?;
    if value == S::ZERO {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Underflow,
            "positive log-normal result rounded to zero",
        ));
    }
    Ok(value)
}

impl<S: RealScalar> Density<S> for LogNormal<S> {
    fn pdf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        if x <= S::ZERO {
            return Ok(S::ZERO);
        }
        let two = S::TWO;
        let sqrt_2pi = S::try_from_f64_lossy(core::f64::consts::TAU.sqrt())?;
        let ln_x = x.ln();
        let z = (ln_x - self.mu) / self.sigma;
        let log_pdf = -(z * z) / two - ln_x - self.sigma.ln() - sqrt_2pi.ln();
        if log_pdf.is_nan() {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "log-normal log-density became indeterminate",
            ));
        }
        checked_final_rounding(
            log_pdf.exp(),
            "log-normal density exceeds the finite scalar range",
        )
    }
}

impl<S: RealScalar> Cdf<S> for LogNormal<S> {
    fn cdf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        if x <= S::ZERO {
            return Ok(S::ZERO);
        }
        self.normal.cdf(x.ln())
    }
}

impl<S: RealScalar> Quantile<S> for LogNormal<S> {
    fn quantile(&self, p: S) -> Result<S, SciOxideError> {
        crate::law::probability(p, scioxide_core::argument!("p"))?;
        if p == S::ZERO {
            return Ok(S::ZERO);
        }
        if p == S::ONE {
            return Ok(S::INFINITY);
        }
        // quantile = exp(μ + σ * Φ⁻¹(p))
        let exponent = self.normal.quantile(p)?;
        positive_log_normal_result(exponent.exp())
    }
}

impl<S: RealScalar> Mean<S> for LogNormal<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        let mut exponent = scioxide_core::RangeAccumulator::new();
        exponent.add(self.mu)?;
        exponent.add_triple_product(S::HALF, self.sigma, self.sigma)?;
        let exponent = exponent.finish()?;
        positive_log_normal_result(exponent.exp())
    }
}

impl<S: RealScalar> Variance<S> for LogNormal<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        let sig2 = checked_float_product(self.sigma, self.sigma)?;
        // Var = (exp(σ²) - 1) * exp(2μ + σ²)
        // ln(e^{s} - 1) without overflow or cancellation, so the whole
        // variance can be assembled in log space.
        let log_factor = if sig2 < S::EPSILON {
            S::TWO * self.sigma.ln()
        } else {
            crate::law::log_expm1(sig2)?
        };
        let log_variance = checked_sum(&[log_factor, self.mu, self.mu, sig2])?;
        let variance = checked_final_rounding(
            log_variance.exp(),
            "log-normal variance exceeds the finite scalar range",
        )?;
        if variance == S::ZERO {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Underflow,
                "positive log-normal variance rounded to zero",
            ));
        }
        Ok(variance)
    }
}

impl<S: RealScalar> crate::Support<S> for LogNormal<S> {
    fn support(&self) -> Result<scioxide_core::ExtendedInterval<S>, SciOxideError> {
        scioxide_core::ExtendedInterval::new(Some(S::ZERO), None)
    }
}

impl<S: RealScalar> crate::LogCdf<S> for LogNormal<S> {
    fn log_cdf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        if x <= S::ZERO {
            return Ok(S::NEG_INFINITY);
        }
        crate::LogCdf::log_cdf(&self.normal, x.ln())
    }
}
impl<S: RealScalar> crate::LogSurvival<S> for LogNormal<S> {
    fn log_sf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        if x <= S::ZERO {
            return Ok(S::ZERO);
        }
        crate::LogSurvival::log_sf(&self.normal, x.ln())
    }
}
impl<S: RealScalar> crate::Survival<S> for LogNormal<S> {
    fn sf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        if x <= S::ZERO {
            return Ok(S::ONE);
        }
        crate::Survival::sf(&self.normal, x.ln())
    }
}
impl<S: RealScalar> crate::LogDensity<S> for LogNormal<S> {
    fn log_pdf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        if x <= S::ZERO {
            return Ok(S::NEG_INFINITY);
        }
        let z = (x.ln() - self.mu) / self.sigma;
        Ok(-z * (z / S::TWO)
            - x.ln()
            - self.sigma.ln()
            - S::try_from_f64_lossy(0.5 * crate::distribution::special::log_two_pi::<f64>())?)
    }
}
impl<S: RealScalar> crate::InverseSurvival<S> for LogNormal<S> {
    fn isf(&self, p: S) -> Result<S, SciOxideError> {
        crate::law::probability(p, scioxide_core::argument!("p"))?;
        if p == S::ZERO {
            return Ok(S::INFINITY);
        }
        if p == S::ONE {
            return Ok(S::ZERO);
        }
        let exponent = crate::InverseSurvival::isf(&self.normal, p)?;
        positive_log_normal_result(exponent.exp())
    }
}
impl<S: RealScalar> crate::Entropy<S> for LogNormal<S> {
    fn entropy(&self) -> Result<S, SciOxideError> {
        Ok(self.mu
            + self.sigma.ln()
            + S::try_from_f64_lossy(
                0.5 * (1.0 + crate::distribution::special::log_two_pi::<f64>()),
            )?)
    }
}

impl<S: RealScalar> crate::LogCoordinateLaw<S> for LogNormal<S> {
    fn cdf_at_log(&self, y: S) -> Result<S, SciOxideError> {
        self.normal.cdf(y)
    }
    fn sf_at_log(&self, y: S) -> Result<S, SciOxideError> {
        crate::Survival::sf(&self.normal, y)
    }
    fn pushforward_pdf(&self, y: S) -> Result<S, SciOxideError> {
        self.normal.pdf(y)
    }
    fn log_cdf_at_log(&self, y: S) -> Result<S, SciOxideError> {
        crate::LogCdf::log_cdf(&self.normal, y)
    }
    fn log_sf_at_log(&self, y: S) -> Result<S, SciOxideError> {
        crate::LogSurvival::log_sf(&self.normal, y)
    }
    fn log_pushforward_pdf(&self, y: S) -> Result<S, SciOxideError> {
        crate::LogDensity::log_pdf(&self.normal, y)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::prng::SplitMix64;
    use crate::traits::SeedableRng;

    #[test]
    fn test_new_valid() {
        assert!(LogNormal::<f64>::new(0.0, 1.0).is_ok());
    }

    #[test]
    fn test_new_invalid_sigma() {
        let error = LogNormal::<f64>::new(0.0, 0.0).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    }

    #[test]
    fn test_new_non_finite_params() {
        let error = LogNormal::<f64>::new(f64::NAN, 1.0).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
        let error = LogNormal::<f64>::new(0.0, f64::INFINITY).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
    }

    #[test]
    fn test_sample_positive() {
        let dist = LogNormal::new(0.0_f64, 1.0).unwrap();
        let mut rng = SplitMix64::from_seed(42);
        for _ in 0..1000 {
            let x: f64 = dist.sample(&mut rng).unwrap();
            assert!(x > 0.0);
        }
    }

    #[test]
    fn test_sample_is_exponential_of_the_default_normal_stream() {
        let dist = LogNormal::new(0.25_f64, 1.5).unwrap();
        let normal = Normal::new(0.25_f64, 1.5).unwrap();
        let mut expected_rng = SplitMix64::from_seed(73);
        let expected = RealScalar::exp(normal.sample(&mut expected_rng).unwrap());

        let mut actual_rng = SplitMix64::from_seed(73);
        assert_eq!(dist.sample(&mut actual_rng).unwrap(), expected);
    }

    #[test]
    fn test_pdf_at_one_standard() {
        // LogNormal(0, 1) pdf at x=1: 1/(√(2π)) ≈ 0.3989
        let dist = LogNormal::new(0.0_f64, 1.0).unwrap();
        let expected = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        assert!((dist.pdf(1.0).unwrap() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_pdf_negative() {
        let dist = LogNormal::new(0.0_f64, 1.0).unwrap();
        assert!((dist.pdf(-1.0).unwrap()).abs() < 1e-12);
    }

    #[test]
    fn test_cdf_at_one_standard() {
        // LogNormal(0, 1) CDF at x=1: Φ(0) = 0.5
        let dist = LogNormal::new(0.0_f64, 1.0).unwrap();
        assert!((dist.cdf(1.0).unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_quantile_roundtrip() {
        let dist = LogNormal::new(0.5_f64, 0.5).unwrap();
        let p = 0.75;
        let x = dist.quantile(p).unwrap();
        let p_back = dist.cdf(x).unwrap();
        assert!((p - p_back).abs() < 1e-10);
    }

    #[test]
    fn quantile_endpoints_are_exact_support_limits() {
        let dist = LogNormal::new(0.5_f64, 0.5).unwrap();
        assert_eq!(dist.quantile(0.0).unwrap(), 0.0);
        assert_eq!(dist.quantile(1.0).unwrap(), f64::INFINITY);
    }

    #[test]
    fn test_quantile_domain_error() {
        let dist = LogNormal::new(0.0_f64, 1.0).unwrap();
        let error = dist.quantile(-0.1).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    }

    #[test]
    fn test_quantile_non_finite_probability() {
        let dist = LogNormal::new(0.0_f64, 1.0).unwrap();
        let error = dist.quantile(f64::NAN).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
    }

    #[test]
    fn test_mean() {
        // mean = exp(μ + σ²/2) = exp(0 + 0.5) = exp(0.5)
        let dist = LogNormal::new(0.0_f64, 1.0).unwrap();
        let expected = (0.5_f64).exp();
        assert!((dist.mean().unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn test_variance() {
        // Var = (e^σ² - 1) · e^(2μ+σ²) = (e-1)·e for μ=0, σ=1
        let dist = LogNormal::new(0.0_f64, 1.0).unwrap();
        let e = std::f64::consts::E;
        let expected = (e - 1.0) * e;
        assert!((dist.variance().unwrap() - expected).abs() < 1e-10);
    }
}
