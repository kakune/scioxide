//! Chi-squared distribution.

use scioxide_core::checked_float_product;
use scioxide_core::{RealScalar, SciOxideError};

use crate::distribution::gamma::{Gamma, GammaKernel};
use crate::traits::{Cdf, Density, Distribution, Mean, Quantile, Rng, Variance};

/// Chi-squared distribution with `k` degrees of freedom.
///
/// ```text
/// f(x) = x^(k/2−1) e^(−x/2) / (2^(k/2) Γ(k/2)),   x ≥ 0, k > 0.
/// ```
///
/// `χ²(k)` is [`Gamma`](crate::Gamma) with shape `k/2` and scale 2, and every
/// operation uses the gamma formulas. The density at zero is `+∞`, `1/2` or `0`
/// for `k < 2`, `k = 2` or `k > 2`. The quantile at `p = 1` is `+∞`.
///
/// # Examples
///
/// ```
/// use scioxide_random::{Cdf, ChiSquared, Distribution, Mean, Quantile, SeedableRng};
/// use scioxide_random::{SplitMix64, Support, Variance};
/// let law = ChiSquared::new(4.0_f64)?;
/// assert_eq!((law.mean()?, law.variance()?), (4.0, 8.0));
/// // For k = 4: F(x) = 1 − (1 + x/2) e^(−x/2), so F(2) = 1 − 2/e.
/// assert!((law.cdf(2.0)? - (1.0 - 2.0 / core::f64::consts::E)).abs() < 1e-15);
/// assert!((law.cdf(law.quantile(0.3)?)? - 0.3).abs() < 1e-14);
/// let mut rng = SplitMix64::from_seed(21);
/// assert!(law.contains(law.sample(&mut rng)?)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
pub struct ChiSquared<S: RealScalar> {
    k: S,
    gamma: Gamma<S>,
    draw: ChiSquareKernel<S>,
}

impl<S: RealScalar> ChiSquared<S> {
    /// Creates a chi-squared distribution with `k` degrees of freedom.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `k` is NaN or infinite.
    /// - `DomainError` if `k ≤ 0`.
    pub fn new(k: S) -> Result<Self, SciOxideError> {
        Self::with_options(k, crate::LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside `[64ε, 0.1)`.
    pub fn with_options(k: S, options: crate::LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        crate::law::positive(k, scioxide_core::argument!("k"))?;
        let half = S::HALF;
        let two = S::TWO;
        // chi^2(k) is Gamma(k/2, scale 2); every capability delegates to that
        // law so both share one set of numerically careful kernels.
        let gamma = Gamma::with_options(k * half, two, options)?;
        Ok(Self {
            k,
            gamma,
            draw: ChiSquareKernel::new(k)?,
        })
    }
}

/// Prepared chi-square variate: twice a standard gamma of half the degrees,
/// without evaluating a normalizer. Used by laws, test statistics and
/// Wishart/Bartlett factors.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ChiSquareKernel<S>(GammaKernel<S>);
impl<S: RealScalar> ChiSquareKernel<S> {
    /// Prepare `chi^2(degrees)` for positive finite degrees.
    pub(crate) fn new(degrees: S) -> Result<Self, SciOxideError> {
        GammaKernel::new(degrees * S::HALF).map(Self)
    }
    /// One chi-square variate.
    #[inline]
    pub(crate) fn sample<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        checked_float_product(S::TWO, self.0.sample(rng)?)
    }
    /// Logarithm of one variate, finite even where the variate underflows.
    pub(crate) fn sample_log<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        Ok(self.0.sample_log(rng)? + S::LN_2)
    }
    /// Logarithm of the ratio of two independent variates.
    pub(crate) fn sample_log_ratio<R: Rng<S>>(
        &self,
        other: &Self,
        rng: &mut R,
    ) -> Result<f64, SciOxideError> {
        self.0.sample_log_ratio(&other.0, rng)
    }
}

impl<S: RealScalar> Distribution<S> for ChiSquared<S> {
    fn batch_byte_limit(&self, token: crate::traits::private::Token) -> usize {
        self.gamma.batch_byte_limit(token)
    }
    fn sample<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        self.draw.sample(rng)
    }
}

impl<S: RealScalar> Density<S> for ChiSquared<S> {
    fn pdf(&self, x: S) -> Result<S, SciOxideError> {
        self.gamma.pdf(x)
    }
}

impl<S: RealScalar> Cdf<S> for ChiSquared<S> {
    fn cdf(&self, x: S) -> Result<S, SciOxideError> {
        self.gamma.cdf(x)
    }
}

impl<S: RealScalar> Quantile<S> for ChiSquared<S> {
    fn quantile(&self, p: S) -> Result<S, SciOxideError> {
        crate::law::probability(p, scioxide_core::argument!("p"))?;
        self.gamma.quantile(p)
    }

    fn quantile_scoped(
        &self,
        p: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::Quantile::quantile_scoped(&self.gamma, p, context)
    }
}

impl<S: RealScalar> Mean<S> for ChiSquared<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        Ok(self.k)
    }
}

impl<S: RealScalar> Variance<S> for ChiSquared<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        let two = S::TWO;
        checked_float_product(two, self.k)
    }
}

impl<S: RealScalar> crate::Support<S> for ChiSquared<S> {
    fn support(&self) -> Result<scioxide_core::ExtendedInterval<S>, SciOxideError> {
        scioxide_core::ExtendedInterval::new(Some(S::ZERO), None)
    }
}

impl<S: RealScalar> crate::LogDensity<S> for ChiSquared<S> {
    fn log_pdf(&self, x: S) -> Result<S, SciOxideError> {
        crate::LogDensity::log_pdf(&self.gamma, x)
    }
}
impl<S: RealScalar> crate::Survival<S> for ChiSquared<S> {
    fn sf(&self, x: S) -> Result<S, SciOxideError> {
        crate::Survival::sf(&self.gamma, x)
    }
}
impl<S: RealScalar> crate::LogCdf<S> for ChiSquared<S> {
    fn log_cdf(&self, x: S) -> Result<S, SciOxideError> {
        crate::LogCdf::log_cdf(&self.gamma, x)
    }

    fn log_cdf_scoped(
        &self,
        x: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::LogCdf::log_cdf_scoped(&self.gamma, x, context)
    }
}
impl<S: RealScalar> crate::LogSurvival<S> for ChiSquared<S> {
    fn log_sf(&self, x: S) -> Result<S, SciOxideError> {
        crate::LogSurvival::log_sf(&self.gamma, x)
    }

    fn log_sf_scoped(
        &self,
        x: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::LogSurvival::log_sf_scoped(&self.gamma, x, context)
    }
}
impl<S: RealScalar> crate::InverseSurvival<S> for ChiSquared<S> {
    fn isf(&self, p: S) -> Result<S, SciOxideError> {
        crate::InverseSurvival::isf(&self.gamma, p)
    }

    fn isf_scoped(
        &self,
        p: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::InverseSurvival::isf_scoped(&self.gamma, p, context)
    }
}
impl<S: RealScalar> crate::Entropy<S> for ChiSquared<S> {
    fn entropy(&self) -> Result<S, SciOxideError> {
        crate::Entropy::entropy(&self.gamma)
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
        assert!(ChiSquared::<f64>::new(3.0).is_ok());
    }

    #[test]
    fn test_new_invalid() {
        let error = ChiSquared::<f64>::new(0.0).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    }

    #[test]
    fn test_new_non_finite() {
        let error = ChiSquared::<f64>::new(f64::NAN).unwrap_err();
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
        let dist = ChiSquared::new(5.0_f64).unwrap();
        let mut rng = SplitMix64::from_seed(42);
        for _ in 0..1000 {
            let x: f64 = dist.sample(&mut rng).unwrap();
            assert!(x > 0.0);
        }
    }

    #[test]
    fn test_sample_mean_approx() {
        let k = 4.0_f64;
        let dist = ChiSquared::new(k).unwrap();
        let mut rng = SplitMix64::from_seed(42);
        let n = 10_000;
        let sum: f64 = (0..n).map(|_| dist.sample(&mut rng).unwrap()).sum();
        let sample_mean = sum / f64::from(n);
        assert!((sample_mean - k).abs() < 0.5, "sample_mean={sample_mean}");
    }

    #[test]
    fn test_cdf_positive() {
        let dist = ChiSquared::new(2.0_f64).unwrap();
        // χ²(2) CDF = 1 - e^(-x/2); at x=2: 1 - e^(-1)
        let expected = 1.0 - (-1.0_f64).exp();
        assert!((dist.cdf(2.0).unwrap() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_quantile_roundtrip() {
        let dist = ChiSquared::new(3.0_f64).unwrap();
        let p = 0.7;
        let x = dist.quantile(p).unwrap();
        let p_back = dist.cdf(x).unwrap();
        assert!((p - p_back).abs() < 1e-8);
    }

    #[test]
    fn endpoint_limits_are_inherited_from_gamma() {
        assert_eq!(
            ChiSquared::new(1.0_f64).unwrap().pdf(0.0).unwrap(),
            f64::INFINITY
        );
        assert_eq!(ChiSquared::new(2.0_f64).unwrap().pdf(0.0).unwrap(), 0.5);
        assert_eq!(ChiSquared::new(3.0_f64).unwrap().pdf(0.0).unwrap(), 0.0);

        let dist = ChiSquared::new(3.0_f64).unwrap();
        assert_eq!(dist.quantile(0.0).unwrap(), 0.0);
        assert_eq!(dist.quantile(1.0).unwrap(), f64::INFINITY);
    }

    #[test]
    fn test_quantile_invalid_probability() {
        let dist = ChiSquared::new(3.0_f64).unwrap();
        let error = dist.quantile(-0.1).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
        let error = dist.quantile(f64::INFINITY).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
    }

    #[test]
    fn test_mean_and_variance() {
        let dist = ChiSquared::new(5.0_f64).unwrap();
        assert!((dist.mean().unwrap() - 5.0).abs() < 1e-12);
        assert!((dist.variance().unwrap() - 10.0).abs() < 1e-12);
    }
}
