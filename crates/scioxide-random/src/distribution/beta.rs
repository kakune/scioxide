//! Beta and generalized beta-related laws.

use crate::{InverseSurvival, LogCdf, LogSurvival};
#[cfg(test)]
use alloc::vec::Vec;

use scioxide_core::backend::checked_final_rounding;
use scioxide_core::{RealScalar, SciOxideError};

#[cfg(test)]
use crate::distribution::gamma::Gamma;
use crate::distribution::gamma::GammaKernel;
use crate::traits::{Cdf, Density, Distribution, Mean, Quantile, Rng, Variance};

use crate::distribution::formula::Formula;
use crate::distribution::formula::formula_law;
use crate::distribution::formula::positive_support;
use crate::distribution::moment_math::variance_from_moments;
use crate::law::constant as number;
use crate::law::positive;
use crate::law::{finite, log1mexp};

/// Beta distribution with shape parameters `α` and `β`.
///
/// ```text
/// f(x) = x^(α−1) (1 − x)^(β−1) / B(α, β),   0 ≤ x ≤ 1, α > 0, β > 0,
/// ```
///
/// where `B` is the beta function. The CDF is the regularized incomplete beta
/// function `I_x(α, β)` (DLMF 8.17.2). At the endpoints the density takes its
/// limit: at `x = 0` it is `+∞`, `β` or `0` for `α < 1`, `α = 1` or `α > 1`, and
/// at `x = 1` it is `+∞`, `α` or `0` for `β < 1`, `β = 1` or `β > 1`. If
/// `ln B(α, β)` is not representable, only the operations that need it fail.
///
/// Sampling draws independent `X₁ ~ Gamma(α, 1)` and `X₂ ~ Gamma(β, 1)` and
/// returns `X₁ / (X₁ + X₂)`. For shapes below one it works with `ln X₁` and
/// `ln X₂`, so no pair is rejected for underflow. A sample may still round to
/// 0 or 1.
/// For shapes at least 32, densities cancel the leading Stirling terms before
/// evaluation; the first omitted normalization correction is below `3e-17`.
/// Density evaluation costs O(1) and allocates nothing.
///
/// # References
///
/// NIST DLMF, [§5.11](https://dlmf.nist.gov/5.11),
/// [§5.12](https://dlmf.nist.gov/5.12) and [§8.17](https://dlmf.nist.gov/8.17).
///
/// # Examples
///
/// ```
/// use scioxide_random::{Beta, Cdf, Density, Distribution, Mean, Quantile, SeedableRng};
/// use scioxide_random::{SplitMix64, Support, Variance};
/// let law = Beta::new(2.0_f64, 3.0)?;
/// assert!((law.mean()? - 0.4).abs() < 1e-15 && (law.variance()? - 0.04).abs() < 1e-15);
/// // f(x) = 12 x (1 − x)², so f(1/2) = 3/2 and F(1/2) = I_{1/2}(2, 3) = 11/16.
/// assert!((law.pdf(0.5)? - 1.5).abs() < 1e-14 && (law.cdf(0.5)? - 11.0 / 16.0).abs() < 1e-15);
/// assert!((law.cdf(law.quantile(0.3)?)? - 0.3).abs() < 1e-14);
/// let mut rng = SplitMix64::from_seed(17);
/// assert!(law.contains(law.sample(&mut rng)?)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
#[must_use]
#[allow(clippy::struct_field_names)]
pub struct Beta<S: RealScalar> {
    options: crate::LawOptions<S>,
    alpha: S,
    beta: S,
    log_beta: Result<S, SciOxideError>,
    kernel: BetaKernel<S>,
}

/// Prepared Beta variate from a pair of standard gamma kernels.
/// Construction evaluates no special functions, so mixtures prepare it once.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BetaKernel<S> {
    first: GammaKernel<S>,
    second: GammaKernel<S>,
    logarithmic: bool,
}

impl<S: RealScalar> BetaKernel<S> {
    /// Prepare for positive finite shapes; errors name the shape argument.
    pub(crate) fn new(alpha: S, beta: S) -> Result<Self, SciOxideError> {
        crate::law::positive(alpha, scioxide_core::argument!("alpha"))?;
        crate::law::positive(beta, scioxide_core::argument!("beta"))?;
        Ok(Self {
            first: GammaKernel::new(alpha)?,
            second: GammaKernel::new(beta)?,
            logarithmic: alpha < S::ONE || beta < S::ONE,
        })
    }

    #[inline]
    pub(crate) fn sample<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        if self.logarithmic {
            let odds = self.sample_log_odds(rng)?;
            let small = (-odds.abs()).exp();
            return S::try_from_f64_lossy(if odds > 0.0 {
                1.0 / (1.0 + small)
            } else {
                small / (1.0 + small)
            });
        }
        normalize_gamma_pair(self.first.sample(rng)?, self.second.sample(rng)?)
    }

    /// Log(p / (1-p)), without rounding either probability to an endpoint.
    pub(crate) fn sample_log_odds<R: Rng<S>>(&self, rng: &mut R) -> Result<f64, SciOxideError> {
        self.first.sample_log_ratio(&self.second, rng)
    }
}

impl<S: RealScalar> Beta<S> {
    /// Creates a beta distribution with shapes `alpha` and `beta`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `alpha` or `beta` is NaN or infinite.
    /// - `DomainError` if `alpha ≤ 0` or `beta ≤ 0`.
    pub fn new(alpha: S, beta: S) -> Result<Self, SciOxideError> {
        Self::with_options(alpha, beta, crate::LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside `[64ε, 0.1)`.
    pub fn with_options(
        alpha: S,
        beta: S,
        options: crate::LawOptions<S>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        let kernel = BetaKernel::new(alpha, beta)?;
        Ok(Self {
            options,
            alpha,
            beta,
            log_beta: scioxide_special::ln_beta(alpha, beta),
            kernel,
        })
    }

    fn log_cdf_work(&self, x: S, left: &mut usize) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        if x <= S::ZERO {
            return Ok(S::NEG_INFINITY);
        }
        if x >= S::ONE {
            return Ok(S::ZERO);
        }
        crate::law::special(left, |budget| {
            scioxide_special::beta_inc_log_with_options(
                self.alpha,
                self.beta,
                x,
                scioxide_special::BetaOptions::new(budget),
            )
        })
    }

    fn log_sf_work(&self, x: S, left: &mut usize) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        if x <= S::ZERO {
            return Ok(S::ZERO);
        }
        if x >= S::ONE {
            return Ok(S::NEG_INFINITY);
        }
        crate::law::special(left, |budget| {
            scioxide_special::beta_inc_log_q_with_options(
                self.alpha,
                self.beta,
                x,
                scioxide_special::BetaOptions::new(budget),
            )
        })
    }

    fn quantile_work(&self, p: S, left: &mut usize) -> Result<S, SciOxideError> {
        crate::law::probability(p, scioxide_core::argument!("p"))?;
        if p == S::ZERO {
            return Ok(S::ZERO);
        }
        if p == S::ONE {
            return Ok(S::ONE);
        }
        let quantile = crate::law::special(left, |budget| {
            scioxide_special::beta_inc_inv_with_options(
                self.alpha,
                self.beta,
                p,
                scioxide_special::BetaOptions::new(budget),
            )
        })?;
        if quantile.is_finite() && quantile >= S::ZERO && quantile <= S::ONE {
            Ok(quantile)
        } else {
            Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Breakdown,
                "beta quantile left the represented support",
            ))
        }
    }

    fn isf_work(&self, p: S, left: &mut usize) -> Result<S, SciOxideError> {
        crate::law::probability(p, scioxide_core::argument!("p"))?;
        let value = crate::law::special(left, |budget| {
            scioxide_special::beta_inc_q_inv_with_options(
                self.alpha,
                self.beta,
                p,
                scioxide_special::BetaOptions::new(budget),
            )
        })?;
        if p > S::ZERO && p < S::ONE {
            crate::law::interior_result(value, Some(S::ZERO), Some(S::ONE))
        } else {
            Ok(value)
        }
    }
}

fn normalize_gamma_pair<S: RealScalar>(x1: S, x2: S) -> Result<S, SciOxideError> {
    if !x1.is_finite()
        || !x2.is_finite()
        || x1 < S::ZERO
        || x2 < S::ZERO
        || (x1 == S::ZERO && x2 == S::ZERO)
    {
        return Err(SciOxideError::numerical_instability(
            scioxide_core::NumericalIssue::Breakdown,
            "gamma pair cannot be normalized",
        ));
    }
    // X1 / (X1 + X2) with independent Gamma(alpha), Gamma(beta) variates is
    // Beta(alpha, beta). The fallback rescales by the larger variate when the
    // sum overflows. Invalid pairs are numerical failures, never redraws.
    let direct_sum = x1 + x2;
    if direct_sum.is_finite() && direct_sum > S::ZERO {
        return Ok(x1 / direct_sum);
    }
    let scale = x1.max(x2);
    let first = x1 / scale;
    let second = x2 / scale;
    Ok(first / (first + second))
}

/// Stirling's log-beta formula (DLMF 5.11.1, 5.12.1), with the two leading
/// powers centered at their mean. The exact delta retains sub-ULP distances
/// from a/(a+b), even when a+b overflows. At shapes >=32 the omitted Stirling
/// correction is below 3e-17; the 24-term log1p series has ratio <=1/8.
fn centered_beta_log_density(a: f64, b: f64, x: f64) -> Result<f64, SciOxideError> {
    let mut delta = scioxide_core::RangeAccumulator::new();
    delta.add_product(a, x)?;
    delta.add_product(b, x)?;
    delta.add(-a)?;
    let delta = delta.finish()?;
    let log_total = crate::law::logadd(a.ln(), b.ln())?;
    let log_x = x.ln();
    let log_q = (-x).ln_1p();
    let centered_power = |shape: f64, difference: f64, log_mean: f64| {
        let u = difference / shape;
        if u.abs() <= 0.125 {
            let series = (2..=24)
                .rev()
                .fold(0.0, |sum, k| 1.0 / f64::from(k) - u * sum);
            -difference * u * series
        } else {
            shape * (log_mean - shape.ln()) - difference
        }
    };
    let remainder = super::discrete_math::factorial_stirling_correction(a)
        + super::discrete_math::factorial_stirling_correction(b)
        - super::discrete_math::factorial_stirling_correction(a + b);
    crate::law::checked_log_density(
        centered_power(a, delta, log_total + log_x)
            + centered_power(b, -delta, log_total + log_q)
            + 0.5 * (a.ln() + b.ln() - log_total)
            - log_x
            - log_q
            - 0.5 * core::f64::consts::TAU.ln()
            - remainder,
    )
}

impl<S: RealScalar> Distribution<S> for Beta<S> {
    fn batch_byte_limit(&self, _: crate::traits::private::Token) -> usize {
        self.options.max_bytes()
    }
    #[inline]
    fn sample<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        self.kernel.sample(rng)
    }
}

impl<S: RealScalar> Density<S> for Beta<S> {
    fn pdf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        if x < S::ZERO || x > S::ONE {
            return Ok(S::ZERO);
        }
        if x == S::ZERO {
            if self.alpha < S::ONE {
                return Ok(S::INFINITY);
            }
            if self.alpha > S::ONE {
                return Ok(S::ZERO);
            }
            return Ok(self.beta);
        }
        if x == S::ONE {
            if self.beta < S::ONE {
                return Ok(S::INFINITY);
            }
            if self.beta > S::ONE {
                return Ok(S::ZERO);
            }
            return Ok(self.alpha);
        }
        // log_pdf = (α-1)*ln(x) + (β-1)*ln(1-x) - ln_beta(α, β)
        let log_pdf = crate::LogDensity::log_pdf(self, x)?;
        checked_final_rounding(
            log_pdf.exp(),
            "beta density exceeds the finite scalar range",
        )
    }
}

impl<S: RealScalar> Cdf<S> for Beta<S> {
    fn cdf(&self, x: S) -> Result<S, SciOxideError> {
        Ok(crate::LogCdf::log_cdf(self, x)?.exp())
    }
}

impl<S: RealScalar> Quantile<S> for Beta<S> {
    fn quantile(&self, p: S) -> Result<S, SciOxideError> {
        self.quantile_work(p, &mut self.options.work_budget().get())
    }

    fn quantile_scoped(
        &self,
        p: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            self.quantile_work(p, left)
        })
    }
}

impl<S: RealScalar> Mean<S> for Beta<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        // alpha / (alpha + beta) rewritten in terms of the smaller-over-larger
        // ratio so that neither the sum nor the ratio can overflow for extreme
        // parameters.
        let mean = if self.alpha >= self.beta {
            S::ONE / (S::ONE + self.beta / self.alpha)
        } else {
            let ratio = self.alpha / self.beta;
            ratio / (S::ONE + ratio)
        };
        if mean > S::ZERO && mean < S::ONE && mean.is_finite() {
            Ok(mean)
        } else {
            Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Underflow,
                "positive beta parameter ratio rounded to a mean endpoint",
            ))
        }
    }
}

impl<S: RealScalar> Variance<S> for Beta<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        let a = self.alpha.into_f64().ln();
        let b = self.beta.into_f64().ln();
        let total = crate::law::logadd(a, b)?;
        super::moment_math::variance(a + b - 2.0 * total - scioxide_core::softplus(total)?)
    }
}

impl<S: RealScalar> crate::Support<S> for Beta<S> {
    fn support(&self) -> Result<scioxide_core::ExtendedInterval<S>, SciOxideError> {
        scioxide_core::ExtendedInterval::new(Some(S::ZERO), Some(S::ONE))
    }
}

impl<S: RealScalar> crate::LogCdf<S> for Beta<S> {
    fn log_cdf(&self, x: S) -> Result<S, SciOxideError> {
        self.log_cdf_work(x, &mut self.options.work_budget().get())
    }

    fn log_cdf_scoped(
        &self,
        x: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            self.log_cdf_work(x, left)
        })
    }
}
impl<S: RealScalar> crate::LogSurvival<S> for Beta<S> {
    fn log_sf(&self, x: S) -> Result<S, SciOxideError> {
        self.log_sf_work(x, &mut self.options.work_budget().get())
    }

    fn log_sf_scoped(
        &self,
        x: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            self.log_sf_work(x, left)
        })
    }
}
impl<S: RealScalar> crate::Survival<S> for Beta<S> {
    fn sf(&self, x: S) -> Result<S, SciOxideError> {
        Ok(crate::LogSurvival::log_sf(self, x)?.exp())
    }
}
impl<S: RealScalar> crate::InverseSurvival<S> for Beta<S> {
    fn isf(&self, p: S) -> Result<S, SciOxideError> {
        self.isf_work(p, &mut self.options.work_budget().get())
    }

    fn isf_scoped(
        &self,
        p: S,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            self.isf_work(p, left)
        })
    }
}
impl<S: RealScalar> crate::LogDensity<S> for Beta<S> {
    fn log_pdf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        if x <= S::ZERO || x >= S::ONE {
            return Ok(self.pdf(x)?.ln());
        }
        if self.alpha.into_f64() >= 32.0 && self.beta.into_f64() >= 32.0 {
            return number(centered_beta_log_density(
                self.alpha.into_f64(),
                self.beta.into_f64(),
                x.into_f64(),
            )?);
        }
        crate::law::checked_log_density(
            (self.alpha - S::ONE) * x.ln() + (self.beta - S::ONE) * (-x).ln_1p()
                - self.log_beta.clone()?,
        )
    }
}
impl<S: RealScalar> crate::Entropy<S> for Beta<S> {
    fn entropy(&self) -> Result<S, SciOxideError> {
        let a = self.alpha.into_f64();
        let b = self.beta.into_f64();
        let sum = crate::law::finite(a + b)?;
        S::try_from_f64_lossy(crate::law::finite(
            super::moment_math::gamma_entropy(a)? + super::moment_math::gamma_entropy(b)?
                - super::moment_math::gamma_entropy(sum)?
                - scioxide_special::digamma(sum)?,
        )?)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::prng::SplitMix64;
    use crate::traits::{BitGenerator, SeedableRng};

    struct ScriptedRng<S> {
        values: std::vec::IntoIter<S>,
        fallback: S,
        draws: usize,
    }

    impl<S> ScriptedRng<S> {
        fn from_values(values: Vec<S>, fallback: S) -> Self {
            Self {
                values: values.into_iter(),
                fallback,
                draws: 0,
            }
        }

        fn constant(value: S) -> Self {
            Self::from_values(Vec::new(), value)
        }
    }

    impl<S> BitGenerator for ScriptedRng<S> {
        #[inline]
        fn next_uniform_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
            self.next_u32()
        }
        #[inline]
        fn next_uniform_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
            self.next_u64()
        }
        fn next_u32(&mut self) -> Result<u32, scioxide_core::SciOxideError> {
            Ok(0)
        }

        fn next_u64(&mut self) -> Result<u64, scioxide_core::SciOxideError> {
            Ok(0)
        }

        fn fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), scioxide_core::SciOxideError> {
            dest.fill(0);
            Ok(())
        }
    }

    impl<S: RealScalar> Rng<S> for ScriptedRng<S> {
        fn next_float(&mut self) -> Result<S, scioxide_core::SciOxideError> {
            self.draws += 1;
            Ok(self.values.next().unwrap_or(self.fallback))
        }
    }

    fn assert_tiny_pair<S: RealScalar>(shape: S, half: S) {
        let dist = Beta::new(shape, shape).unwrap();
        let mut rng = ScriptedRng::constant(half);

        assert_eq!(dist.sample(&mut rng).unwrap(), half);
        assert_eq!(rng.draws, 4);
    }

    #[test]
    fn test_new_valid() {
        assert!(Beta::<f64>::new(2.0, 3.0).is_ok());
    }

    #[test]
    fn test_new_invalid_alpha() {
        let error = Beta::<f64>::new(0.0, 1.0).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    }

    #[test]
    fn test_new_invalid_beta() {
        let error = Beta::<f64>::new(1.0, -1.0).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    }

    #[test]
    fn test_new_non_finite_params() {
        let error = Beta::<f64>::new(f64::NAN, 1.0).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
        let error = Beta::<f64>::new(1.0, f64::INFINITY).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
    }

    #[test]
    fn test_sample_in_01() {
        let dist = Beta::new(2.0_f64, 5.0).unwrap();
        let mut rng = SplitMix64::from_seed(42);
        for _ in 0..1000 {
            let x: f64 = dist.sample(&mut rng).unwrap();
            assert!(x > 0.0 && x < 1.0, "x={x}");
        }
    }

    #[test]
    fn test_sample_mean_approx() {
        let alpha = 2.0_f64;
        let beta = 5.0_f64;
        let dist = Beta::new(alpha, beta).unwrap();
        let mut rng = SplitMix64::from_seed(42);
        let n = 10_000;
        let sum: f64 = (0..n).map(|_| dist.sample(&mut rng).unwrap()).sum();
        let sample_mean = sum / f64::from(n);
        let expected = alpha / (alpha + beta);
        assert!(
            (sample_mean - expected).abs() < 0.05,
            "sample_mean={sample_mean}, expected={expected}"
        );
    }

    #[test]
    fn test_gamma_pair_normalization_distinguishes_values_from_failures() {
        assert_eq!(normalize_gamma_pair(1.0_f64, 1.0).unwrap(), 0.5);
        assert_eq!(normalize_gamma_pair(0.0_f64, 1.0).unwrap(), 0.0);
        assert_eq!(normalize_gamma_pair(1.0_f64, 0.0).unwrap(), 1.0);
        for (a, b) in [
            (0.0, 0.0),
            (f64::INFINITY, 1.0),
            (f64::NAN, 1.0),
            (-1.0, 2.0),
        ] {
            assert!(matches!(
                normalize_gamma_pair(a, b).unwrap_err().view(),
                scioxide_core::SciOxideErrorView::NumericalInstability {
                    issue: scioxide_core::NumericalIssue::Breakdown,
                    ..
                }
            ));
        }
    }

    #[test]
    fn underflowing_gamma_pair_is_normalized_without_retry() {
        let dist = Beta::new(0.0005_f64, 0.0005).unwrap();
        let mut rng = ScriptedRng::from_values(
            vec![
                0.5, 0.5, // First Gamma underflows as a value.
                0.5, 0.5001, // Second Gamma also underflows as a value.
            ],
            0.5,
        );

        let sample = dist.sample(&mut rng).unwrap();

        assert!(sample > 0.0 && sample < 1.0);
        assert_ne!(sample, 0.5);
        assert_eq!(rng.draws, 4);
    }

    #[test]
    fn logarithmic_pairs_retain_the_midpoint_for_both_widths() {
        assert_tiny_pair(f32::MIN_POSITIVE, 0.5_f32);
        assert_tiny_pair(f64::MIN_POSITIVE, 0.5_f64);
        assert_tiny_pair(f32::from_bits(1), 0.5_f32);
        assert_tiny_pair(f64::from_bits(1), 0.5_f64);
    }

    #[test]
    fn gamma_ratio_cancels_before_materializing_either_variate() {
        let law = BetaPrime::new(f64::from_bits(1), f64::from_bits(1)).unwrap();
        let mut rng = ScriptedRng::constant(0.5);
        assert_eq!(law.sample(&mut rng).unwrap(), 1.0);
        assert_eq!(rng.draws, 4);
    }

    #[test]
    fn test_regular_sampling_matches_single_pass_gamma_stream() {
        let dist = Beta::new(2.0_f64, 5.0).unwrap();
        let gamma_a = Gamma::new(2.0_f64, 1.0).unwrap();
        let gamma_b = Gamma::new(5.0_f64, 1.0).unwrap();
        let mut actual_rng = SplitMix64::from_seed(42);
        let mut expected_rng = SplitMix64::from_seed(42);

        for _ in 0..100 {
            let x1 = gamma_a.sample(&mut expected_rng).unwrap();
            let x2 = gamma_b.sample(&mut expected_rng).unwrap();
            let expected = x1 / (x1 + x2);

            assert_eq!(dist.sample(&mut actual_rng).unwrap(), expected);
        }
        assert_eq!(
            actual_rng.next_u64().unwrap(),
            expected_rng.next_u64().unwrap()
        );
    }

    #[test]
    fn test_cdf_at_half_symmetric() {
        // Beta(a, a) is symmetric, so CDF(0.5) = 0.5
        let dist = Beta::new(3.0_f64, 3.0).unwrap();
        assert!((dist.cdf(0.5).unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_quantile_roundtrip() {
        let dist = Beta::new(2.0_f64, 3.0).unwrap();
        let p = 0.6;
        let x = dist.quantile(p).unwrap();
        let p_back = dist.cdf(x).unwrap();
        assert!((p - p_back).abs() < 1e-6, "p_back={p_back}");
    }

    #[test]
    fn test_quantile_domain_error() {
        let dist = Beta::new(1.0_f64, 1.0).unwrap();
        let error = dist.quantile(1.5).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    }

    #[test]
    fn test_quantile_non_finite_probability() {
        let dist = Beta::new(1.0_f64, 1.0).unwrap();
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
        let dist = Beta::new(2.0_f64, 3.0).unwrap();
        assert!((dist.mean().unwrap() - 0.4).abs() < 1e-12);
    }

    #[test]
    fn test_variance() {
        let dist = Beta::new(2.0_f64, 3.0).unwrap();
        // Var = αβ / ((α+β)²(α+β+1)) = 6 / (25*6) = 0.04
        assert!((dist.variance().unwrap() - 0.04).abs() < 1e-12);
    }

    #[test]
    fn test_pdf_uniform_case() {
        // Beta(1,1) = Uniform(0,1), pdf = 1 everywhere in (0,1)
        let dist = Beta::new(1.0_f64, 1.0).unwrap();
        assert!((dist.pdf(0.3).unwrap() - 1.0).abs() < 1e-10);
        assert!((dist.pdf(0.7).unwrap() - 1.0).abs() < 1e-10);
        assert_eq!(dist.pdf(0.0).unwrap(), 1.0);
        assert_eq!(dist.pdf(1.0).unwrap(), 1.0);
    }

    #[test]
    fn endpoint_limits_and_extreme_moments_are_explicit() {
        assert_eq!(Beta::new(1.0_f64, 3.0).unwrap().pdf(0.0).unwrap(), 3.0);
        assert_eq!(Beta::new(3.0_f64, 1.0).unwrap().pdf(1.0).unwrap(), 3.0);
        assert_eq!(Beta::new(2.0_f64, 3.0).unwrap().pdf(0.0).unwrap(), 0.0);
        assert_eq!(Beta::new(3.0_f64, 2.0).unwrap().pdf(1.0).unwrap(), 0.0);
        assert_eq!(
            Beta::new(0.5_f64, 2.0).unwrap().pdf(0.0).unwrap(),
            f64::INFINITY
        );
        assert_eq!(
            Beta::new(2.0_f64, 0.5).unwrap().pdf(1.0).unwrap(),
            f64::INFINITY
        );
        let distribution = Beta::new(2.0_f64, 3.0).unwrap();
        assert_eq!(distribution.quantile(0.0).unwrap(), 0.0);
        assert_eq!(distribution.quantile(1.0).unwrap(), 1.0);

        let extreme = Beta::new(f64::MAX, f64::MAX).unwrap();
        assert_eq!(extreme.mean().unwrap(), 0.5);
        let variance = extreme.variance().unwrap();
        assert!(variance.is_finite() && variance > 0.0);
    }

    #[test]
    fn scaled_gamma_normalization_handles_overflowing_sum() {
        assert_eq!(normalize_gamma_pair(f64::MAX, f64::MAX).unwrap(), 0.5);
    }
}

/// Beta prime distribution with shape parameters `a` and `b`.
///
/// ```text
/// f(x) = x^(a−1) (1 + x)^(−a−b) / B(a, b),   x > 0, a > 0, b > 0.
/// ```
///
/// Equivalently `X/(1 + X) ~ Beta(a, b)`. The law has unit scale. Scale it with
/// [`Affine`](crate::Affine). The mean exists for `b > 1` and the variance for
/// `b > 2`.
///
/// Sampling returns `exp(ln G_a − ln G_b)` for independent `G_a ~ Gamma(a, 1)`
/// and `G_b ~ Gamma(b, 1)`. A sample that underflows to zero or overflows is a
/// `NumericalInstability` error.
///
/// # Examples
///
/// ```
/// use scioxide_random::{BetaPrime, Cdf, Distribution, Mean, Quantile, SeedableRng};
/// use scioxide_random::{SplitMix64, Support, Variance};
/// let law = BetaPrime::new(2.0_f64, 3.0)?;
/// assert!((law.mean()? - 1.0).abs() < 1e-14 && (law.variance()? - 2.0).abs() < 1e-14);
/// assert!((law.cdf(1.0)? - 11.0 / 16.0).abs() < 1e-15); // I_{1/2}(2, 3)
/// assert!((law.cdf(law.quantile(0.3)?)? - 0.3).abs() < 1e-14);
/// let mut rng = SplitMix64::from_seed(18);
/// assert!(law.contains(law.sample(&mut rng)?)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone)]
pub struct BetaPrime<S: RealScalar> {
    options: crate::LawOptions<S>,
    a: S,
    b: S,
    left: super::Beta<S>,
    right: super::Beta<S>,
    kernel: BetaKernel<S>,
}
impl<S: RealScalar> BetaPrime<S> {
    /// Creates a beta prime distribution with shapes `a` and `b`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `a` or `b` is NaN or infinite.
    /// - `DomainError` if `a ≤ 0` or `b ≤ 0`.
    pub fn new(a: S, b: S) -> Result<Self, SciOxideError> {
        Self::with_options(a, b, crate::LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside `[64ε, 0.1)`.
    pub fn with_options(a: S, b: S, options: crate::LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        positive(a, scioxide_core::argument!("a"))?;
        positive(b, scioxide_core::argument!("b"))?;
        Ok(Self {
            options,
            a,
            b,
            left: super::Beta::with_options(a, b, options)?,
            right: super::Beta::with_options(b, a, options)?,
            kernel: BetaKernel::new(a, b)?,
        })
    }
}
impl<S: RealScalar> Formula<S> for BetaPrime<S> {
    fn tolerance(&self) -> scioxide_core::Tolerance<S> {
        self.options.tolerance()
    }

    fn work_budget(&self) -> scioxide_core::Budget {
        self.options.work_budget()
    }
    fn byte_limit(&self) -> usize {
        self.options.max_bytes()
    }

    fn bounds(&self) -> (Option<S>, Option<S>) {
        positive_support()
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        if x < S::ZERO {
            return Ok(S::NEG_INFINITY);
        }
        if x == S::ZERO {
            return Ok(if self.a < S::ONE {
                S::INFINITY
            } else if self.a == S::ONE {
                -(self.left.log_beta.clone()?)
            } else {
                S::NEG_INFINITY
            });
        }
        Ok((self.a - S::ONE) * x.ln()
            - (self.a + self.b) * x.ln_1p()
            - self.left.log_beta.clone()?)
    }
    fn tail_with_work(&self, x: S, upper: bool, left: &mut usize) -> Result<S, SciOxideError> {
        if upper {
            if x <= S::ZERO {
                return Ok(S::ZERO);
            }
            if x > S::ONE {
                let y = (S::ONE / x) / (S::ONE + S::ONE / x);
                crate::law::scoped(left, |context| self.right.log_cdf_scoped(y, context))
            } else {
                crate::law::scoped(left, |context| {
                    self.left.log_sf_scoped(x / (S::ONE + x), context)
                })
            }
        } else {
            if x <= S::ZERO {
                return Ok(S::NEG_INFINITY);
            }
            // X / (1 + X) ~ Beta(a, b) and 1 / (1 + X) ~ Beta(b, a). For x > 1 the
            // mapped value x / (1 + x) crowds against 1 and loses resolution, so the
            // reflected Beta(b, a) is evaluated at 1 / (1 + x) instead, formed from
            // 1 / x so it cannot overflow.
            if x > S::ONE {
                let y = (S::ONE / x) / (S::ONE + S::ONE / x);
                log1mexp(crate::law::scoped(left, |context| {
                    self.right.log_cdf_scoped(y, context)
                })?)
            } else {
                crate::law::scoped(left, |context| {
                    self.left.log_cdf_scoped(x / (S::ONE + x), context)
                })
            }
        }
    }

    fn inverse_with_remaining(
        &self,
        p: S,
        upper: bool,
        left: &mut usize,
    ) -> Result<S, SciOxideError> {
        if upper {
            if p == S::ZERO {
                return Ok(S::INFINITY);
            }
            if p == S::ONE {
                return Ok(S::ZERO);
            }
            if p < S::HALF {
                let y = crate::law::scoped(left, |context| self.right.quantile_scoped(p, context))?;
                finite(S::ONE / y - S::ONE)
            } else {
                let y = crate::law::scoped(left, |context| self.left.isf_scoped(p, context))?;
                finite(y / (S::ONE - y))
            }
        } else {
            if p == S::ZERO {
                return Ok(S::ZERO);
            }
            if p == S::ONE {
                return Ok(S::INFINITY);
            }
            // Upper quantiles come from the reflected Beta(b, a) so large x is
            // recovered as 1/y - 1 from a small, well-resolved y.
            if p > S::HALF {
                let y = crate::law::scoped(left, |context| self.right.isf_scoped(p, context))?;
                finite(S::ONE / y - S::ONE)
            } else {
                let y = crate::law::scoped(left, |context| self.left.quantile_scoped(p, context))?;
                finite(y / (S::ONE - y))
            }
        }
    }

    fn draw_with_remaining<R: Rng<S>>(
        &self,
        rng: &mut R,
        _left: &mut usize,
    ) -> Result<S, SciOxideError> {
        crate::law::positive_draw(self.kernel.sample_log_odds(rng)?)
    }
}
formula_law!(BetaPrime);
impl<S: RealScalar> Mean<S> for BetaPrime<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        if self.b <= S::ONE {
            Err(crate::law::undefined_moment(
                "beta-prime mean does not exist",
            ))
        } else {
            finite(self.a / (self.b - S::ONE))
        }
    }
}
impl<S: RealScalar> Variance<S> for BetaPrime<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        if self.b <= number::<S>(2.)? {
            Err(crate::law::undefined_moment(
                "beta-prime variance does not exist",
            ))
        } else {
            let a = self.a.into_f64().ln();
            let b = self.b.into_f64();
            let bm1 = (b - 1.0).ln();
            super::moment_math::variance(
                a + crate::law::logadd(a, bm1)? - 2.0 * bm1 - (b - 2.0).ln(),
            )
        }
    }
}

/// Mielke beta-kappa distribution with shape parameters `k` and `s`.
///
/// ```text
/// F(x) = (x^s / (1 + x^s))^(k/s),   x > 0, k > 0, s > 0.
/// ```
///
/// `k = s` gives the log-logistic distribution. The law has unit scale. Scale
/// it with [`Affine`](crate::Affine). The moment `E[X^r]` exists for `r < s`.
///
/// # Examples
///
/// ```
/// use scioxide_random::{Cdf, Distribution, Mean, Mielke, Quantile, SeedableRng};
/// use scioxide_random::{SplitMix64, Support};
/// let law = Mielke::new(2.0_f64, 2.0)?;
/// assert!((law.cdf(1.0)? - 0.5).abs() < 1e-15);
/// assert!((law.mean()? - core::f64::consts::FRAC_PI_2).abs() < 1e-14);
/// assert!((law.cdf(law.quantile(0.3)?)? - 0.3).abs() < 1e-14);
/// let mut rng = SplitMix64::from_seed(19);
/// assert!(law.contains(law.sample(&mut rng)?)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Mielke<S: RealScalar> {
    k: S,
    s: S,
}
impl<S: RealScalar> Mielke<S> {
    /// Creates a Mielke beta-kappa distribution with shapes `k` and `s`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `k` or `s` is NaN or infinite.
    /// - `DomainError` if `k ≤ 0` or `s ≤ 0`.
    pub fn new(k: S, s: S) -> Result<Self, SciOxideError> {
        positive(k, scioxide_core::argument!("k"))?;
        positive(s, scioxide_core::argument!("s"))?;
        Ok(Self { k, s })
    }
    fn inverse_log_cdf(&self, log_p: S) -> Result<S, SciOxideError> {
        let log_power = self.s.ln() - self.k.ln() + (-log_p).ln();
        let power = log_power.exp();
        let log_x = if power > S::ONE {
            // ln(expm1(power))/s = -ln(F)/k + ln(1-exp(-power))/s.
            log_p / self.k - log1mexp(-power)? / self.s
        } else if power < S::MIN_POSITIVE {
            -log_power / self.s
        } else {
            -crate::law::log_expm1(power)? / self.s
        };
        finite(log_x.exp())
    }
    fn raw_moment(&self, r: S) -> Result<S, SciOxideError> {
        if r >= self.s {
            return Err(crate::law::undefined_moment("Mielke moment does not exist"));
        }
        let base = self.k.into_f64() / self.s.into_f64();
        let shift = r.into_f64() / self.s.into_f64();
        finite(S::try_from_f64_lossy(
            (super::special::log_gamma_ratio(base, shift)?
                + scioxide_special::ln_gamma(1.0 - shift)?)
            .exp(),
        )?)
    }
}
impl<S: RealScalar> Formula<S> for Mielke<S> {
    fn bounds(&self) -> (Option<S>, Option<S>) {
        positive_support()
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        if x < S::ZERO {
            return Ok(S::NEG_INFINITY);
        }
        if x == S::ZERO {
            return Ok(if self.k < S::ONE {
                S::INFINITY
            } else if self.k == S::ONE {
                S::ZERO
            } else {
                S::NEG_INFINITY
            });
        }
        let log_x = x.ln();
        let y = self.s * log_x;
        let coefficient = scioxide_core::softplus(self.k.ln() - self.s.ln())?;
        let remainder = (coefficient + crate::law::log_softplus(-y.abs())?).exp();
        let lead = if y < S::ZERO {
            (self.k - S::ONE) * log_x
        } else {
            -(self.s + S::ONE) * log_x
        };
        Ok(self.k.ln() + lead - remainder)
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        if x <= S::ZERO {
            return Ok(if upper { S::ZERO } else { S::NEG_INFINITY });
        }
        let log_x = x.ln();
        let y = self.s * log_x;
        if y < S::ZERO {
            // log_sigmoid(y)=y-softplus(y). Cancel s symbolically so an
            // overflowing s*ln(x) still gives the finite leading k*ln(x).
            let correction = (self.k.ln() - self.s.ln() + crate::law::log_softplus(y)?).exp();
            let lower = self.k * log_x - correction;
            return if upper { log1mexp(lower) } else { Ok(lower) };
        }
        let log_rate = self.k.ln() - self.s.ln() + crate::law::log_softplus(-y)?;
        let rate = log_rate.exp();
        if !upper {
            Ok(-rate)
        } else if rate == S::ZERO {
            Ok(log_rate)
        } else {
            log1mexp(-rate)
        }
    }
    fn inverse(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ZERO {
            return Ok(S::ZERO);
        }
        if p == S::ONE {
            return Ok(S::INFINITY);
        }
        self.inverse_log_cdf(p.ln())
    }
    fn inverse_upper(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ZERO {
            return Ok(S::INFINITY);
        }
        if p == S::ONE {
            return Ok(S::ZERO);
        }
        // F=1-p=(1+x^-s)^(-k/s). Form ln(1-p) directly so an upper
        // probability below machine epsilon is not rounded away.
        self.inverse_log_cdf((-p).ln_1p())
    }
}
formula_law!(Mielke);
impl<S: RealScalar> Mean<S> for Mielke<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        self.raw_moment(S::ONE)
    }
}
impl<S: RealScalar> Variance<S> for Mielke<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        let m = self.mean()?;
        variance_from_moments(self.raw_moment(number::<S>(2.)?)?, m)
    }
}

/// Burr type XII distribution with shape parameters `c` and `d`.
///
/// ```text
/// F(x) = 1 − (1 + x^c)^(−d),   x ≥ 0, c > 0, d > 0.
/// ```
///
/// The law has unit scale. Scale it with [`Affine`](crate::Affine). The moment
/// `E[X^r]` exists for `r < cd`.
///
/// # Examples
///
/// ```
/// use scioxide_random::{Burr12, Cdf, Distribution, Mean, Quantile, SeedableRng};
/// use scioxide_random::{SplitMix64, Support};
/// let law = Burr12::new(2.0_f64, 3.0)?;
/// assert!((law.cdf(1.0)? - 0.875).abs() < 1e-15);
/// assert!((law.mean()? - 3.0 * core::f64::consts::PI / 16.0).abs() < 1e-14);
/// assert!((law.cdf(law.quantile(0.3)?)? - 0.3).abs() < 1e-14);
/// let mut rng = SplitMix64::from_seed(20);
/// assert!(law.contains(law.sample(&mut rng)?)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Burr12<S: RealScalar> {
    c: S,
    d: S,
}
impl<S: RealScalar> Burr12<S> {
    /// Creates a Burr type XII distribution with shapes `c` and `d`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `c` or `d` is NaN or infinite.
    /// - `DomainError` if `c ≤ 0` or `d ≤ 0`.
    pub fn new(c: S, d: S) -> Result<Self, SciOxideError> {
        positive(c, scioxide_core::argument!("c"))?;
        positive(d, scioxide_core::argument!("d"))?;
        Ok(Self { c, d })
    }
    fn raw_moment(&self, r: S) -> Result<S, SciOxideError> {
        if r / self.c >= self.d {
            return Err(crate::law::undefined_moment(
                "requested moment does not exist",
            ));
        }
        let shift = (r / self.c).into_f64();
        let log = scioxide_special::ln_gamma(1.0 + shift)?
            + super::special::log_gamma_ratio(self.d.into_f64(), -shift)?;
        finite(number::<S>(log.exp())?)
    }
}
impl<S: RealScalar> Formula<S> for Burr12<S> {
    fn bounds(&self) -> (Option<S>, Option<S>) {
        (Some(S::ZERO), None)
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        if x < S::ZERO {
            return Ok(S::NEG_INFINITY);
        }
        if x == S::ZERO {
            return Ok(if self.c < S::ONE {
                S::INFINITY
            } else if self.c == S::ONE {
                self.d.ln()
            } else {
                S::NEG_INFINITY
            });
        }
        Ok(self.c.ln() + self.d.ln() + (self.c - S::ONE) * x.ln()
            - (self.d + S::ONE) * scioxide_core::softplus(self.c * x.ln())?)
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        if upper {
            if x <= S::ZERO {
                Ok(S::ZERO)
            } else {
                Ok(-self.d * scioxide_core::softplus(self.c * x.ln())?)
            }
        } else {
            if x <= S::ZERO {
                Ok(S::NEG_INFINITY)
            } else {
                log1mexp(self.tail_with_work(x, true, _left)?)
            }
        }
    }
    fn inverse(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ONE {
            return Ok(S::INFINITY);
        }
        if p == S::ZERO {
            return Ok(S::ZERO);
        }
        finite((crate::law::log_expm1(-(-p).ln_1p() / self.d)? / self.c).exp())
    }
    fn inverse_upper(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ZERO {
            return Ok(S::INFINITY);
        }
        if p == S::ONE {
            return Ok(S::ZERO);
        }
        finite((crate::law::log_expm1(-p.ln() / self.d)? / self.c).exp())
    }
}
formula_law!(Burr12);
impl<S: RealScalar> Mean<S> for Burr12<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        self.raw_moment(S::ONE)
    }
}
impl<S: RealScalar> Variance<S> for Burr12<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        let m = self.mean()?;
        variance_from_moments(self.raw_moment(number(2.)?)?, m)
    }
}
