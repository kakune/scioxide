//! Exponential and generalized exponential laws.
use crate::distribution::LawOptions;

use scioxide_core::backend::checked_final_rounding;
use scioxide_core::checked_float_product;
use scioxide_core::{RealScalar, SciOxideError};

use crate::traits::{Cdf, Density, Distribution, Mean, Quantile, Rng, Variance};

use crate::distribution::formula::formula_law;
use crate::distribution::formula::positive_support;
use crate::distribution::formula::{Formula, Inverse};
use crate::distribution::moment_math::quantile_mean;
use crate::distribution::moment_math::quantile_variance;
use crate::law::positive;
use crate::law::{finite, log1mexp};

mod ziggurat;
mod ziggurat_tables;

/// One standard exponential draw from the width-specific Ziggurat kernel.
/// The value is finite and nonnegative.
#[inline(always)]
pub(crate) fn standard_exponential<S: RealScalar, R: Rng<S>>(
    rng: &mut R,
) -> Result<S, SciOxideError> {
    match super::normal::sampling_kernel::<S>()? {
        super::normal::SamplingKernel::F32Ziggurat => S::from_value(
            scioxide_core::backend::ScalarValue::F32(ziggurat::sample_f32(rng)?),
        ),
        super::normal::SamplingKernel::F64Ziggurat => S::from_value(
            scioxide_core::backend::ScalarValue::F64(ziggurat::sample_f64(rng)?),
        ),
    }
}

/// Exponential distribution with rate `λ`.
///
/// ```text
/// f(x) = λ e^(−λx),   x ≥ 0, λ > 0.
/// ```
///
/// Sampling divides a standard exponential draw by `λ`. The draw comes from a
/// Ziggurat with 256 layers. About 98.9% of draws take one uniform word and no
/// logarithm. The tail beyond the last layer is `R − ln U` for the cutoff `R`
/// and a uniform `U`, which is exact because the law is memoryless. The
/// quantile at `p = 1` is `+∞`. Every other quantile is finite.
///
/// # Examples
///
/// ```
/// use scioxide_random::{Cdf, Distribution, Exponential, Mean, Quantile, SeedableRng};
/// use scioxide_random::{SplitMix64, Survival, Support, Variance};
/// let law = Exponential::new(2.0_f64)?;
/// assert_eq!((law.mean()?, law.variance()?), (0.5, 0.25));
/// assert!((law.sf(1.0)? - (-2.0_f64).exp()).abs() < 1e-16);
/// assert!((law.cdf(law.quantile(0.3)?)? - 0.3).abs() < 1e-15);
/// let mut rng = SplitMix64::from_seed(10);
/// assert!(law.contains(law.sample(&mut rng)?)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
///
/// # References
///
/// G. Marsaglia and W. W. Tsang, "The Ziggurat method for generating random
/// variables", *Journal of Statistical Software* 5(8), 2000.
#[derive(Debug, Clone)]
#[must_use]
pub struct Exponential<S> {
    lambda: S,
}

impl<S: RealScalar> Exponential<S> {
    /// Creates an exponential distribution with rate `lambda`.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `lambda` is NaN or infinite.
    /// - `DomainError` if `lambda ≤ 0`.
    pub fn new(lambda: S) -> Result<Self, SciOxideError> {
        crate::law::positive(lambda, scioxide_core::argument!("lambda"))?;
        Ok(Self { lambda })
    }
}

impl<S: RealScalar> Distribution<S> for Exponential<S> {
    #[inline]
    fn sample<R: Rng<S>>(&self, rng: &mut R) -> Result<S, SciOxideError> {
        checked_final_rounding(
            standard_exponential(rng)? / self.lambda,
            "exponential sample exceeds the finite scalar range",
        )
    }
}

impl<S: RealScalar> Density<S> for Exponential<S> {
    fn pdf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        if x < S::ZERO {
            Ok(S::ZERO)
        } else {
            checked_final_rounding(
                self.lambda * (-self.lambda * x).exp(),
                "exponential density exceeds the finite scalar range",
            )
        }
    }
}

impl<S: RealScalar> Cdf<S> for Exponential<S> {
    fn cdf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        if x < S::ZERO {
            Ok(S::ZERO)
        } else {
            // 1 - e^{-lambda x} via `expm1` keeps full relative accuracy for
            // small lambda x, where the direct form would cancel to zero.
            let probability = -(-self.lambda * x).exp_m1();
            if probability.is_finite() && probability >= S::ZERO && probability <= S::ONE {
                Ok(probability)
            } else {
                Err(SciOxideError::numerical_instability(
                    scioxide_core::NumericalIssue::Breakdown,
                    "exponential CDF left the represented probability interval",
                ))
            }
        }
    }
}

impl<S: RealScalar> Quantile<S> for Exponential<S> {
    fn quantile(&self, p: S) -> Result<S, SciOxideError> {
        crate::law::probability(p, scioxide_core::argument!("p"))?;
        if p == S::ONE {
            return Ok(S::INFINITY);
        }
        // -ln(1 - p) via `ln_1p` is accurate for small p (the common case for
        // lower quantiles), where `ln(1 - p)` would lose most digits.
        checked_final_rounding(
            -(-p).ln_1p() / self.lambda,
            "exponential quantile exceeds the finite scalar range",
        )
    }
}

impl<S: RealScalar> Mean<S> for Exponential<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        checked_final_rounding(
            S::ONE / self.lambda,
            "exponential mean exceeds the finite scalar range",
        )
    }
}

impl<S: RealScalar> Variance<S> for Exponential<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        let reciprocal = checked_final_rounding(
            S::ONE / self.lambda,
            "exponential standard deviation exceeds the finite scalar range",
        )?;
        let variance = checked_float_product(reciprocal, reciprocal)?;
        if variance == S::ZERO {
            return Err(SciOxideError::numerical_instability(
                scioxide_core::NumericalIssue::Underflow,
                "positive exponential variance rounded to zero",
            ));
        }
        Ok(variance)
    }
}

impl<S: RealScalar> crate::Support<S> for Exponential<S> {
    fn support(&self) -> Result<scioxide_core::ExtendedInterval<S>, SciOxideError> {
        scioxide_core::ExtendedInterval::new(Some(S::ZERO), None)
    }
}

impl<S: RealScalar> crate::LogSurvival<S> for Exponential<S> {
    fn log_sf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        Ok(if x <= S::ZERO {
            S::ZERO
        } else {
            -self.lambda * x
        })
    }
}
impl<S: RealScalar> crate::Survival<S> for Exponential<S> {
    fn sf(&self, x: S) -> Result<S, SciOxideError> {
        Ok(crate::LogSurvival::log_sf(self, x)?.exp())
    }
}
impl<S: RealScalar> crate::LogCdf<S> for Exponential<S> {
    fn log_cdf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::log1mexp(crate::LogSurvival::log_sf(self, x)?)
    }
}
impl<S: RealScalar> crate::LogDensity<S> for Exponential<S> {
    fn log_pdf(&self, x: S) -> Result<S, SciOxideError> {
        crate::law::argument(x, "x")?;
        Ok(if x < S::ZERO {
            S::NEG_INFINITY
        } else {
            self.lambda.ln() - self.lambda * x
        })
    }
}
impl<S: RealScalar> crate::InverseSurvival<S> for Exponential<S> {
    fn isf(&self, p: S) -> Result<S, SciOxideError> {
        crate::law::probability(p, scioxide_core::argument!("p"))?;
        if p == S::ZERO {
            return Ok(S::INFINITY);
        }
        crate::law::finite(-p.ln() / self.lambda)
    }
}
impl<S: RealScalar> crate::Entropy<S> for Exponential<S> {
    fn entropy(&self) -> Result<S, SciOxideError> {
        Ok(S::ONE - self.lambda.ln())
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
        assert!(Exponential::<f64>::new(1.0).is_ok());
    }

    #[test]
    fn test_new_zero_lambda() {
        let error = Exponential::<f64>::new(0.0).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    }

    #[test]
    fn test_new_non_finite_lambda() {
        let error = Exponential::<f64>::new(f64::NAN).unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NonFinite {
                origin: scioxide_core::NonFiniteOrigin::Input,
                ..
            }
        ));
    }

    #[test]
    fn exponential_power_density_at_zero_is_continuous_for_unit_shape() {
        use crate::{Density, LogDensity};
        let law = ExponentialPower::new(1.0_f64).unwrap();
        assert_eq!(law.log_pdf(0.0).unwrap(), 0.0);
        assert!((law.pdf(1e-12).unwrap() - 1.0).abs() < 1e-11);
        let below = ExponentialPower::new(0.5_f64).unwrap();
        assert_eq!(below.pdf(0.0).unwrap(), f64::INFINITY);
        let above = ExponentialPower::new(2.0_f64).unwrap();
        assert_eq!(above.pdf(0.0).unwrap(), 0.0);
    }

    #[test]
    fn test_sample_positive() {
        let dist = Exponential::new(2.0_f64).unwrap();
        let mut rng = SplitMix64::from_seed(42);
        for _ in 0..1000 {
            let x: f64 = dist.sample(&mut rng).unwrap();
            assert!(x >= 0.0);
        }
    }

    #[test]
    fn test_sample_mean_approx() {
        let lambda = 3.0_f64;
        let dist = Exponential::new(lambda).unwrap();
        let mut rng = SplitMix64::from_seed(42);
        let n = 10_000;
        let sum: f64 = (0..n).map(|_| dist.sample(&mut rng).unwrap()).sum();
        let sample_mean = sum / f64::from(n);
        assert!((sample_mean - 1.0 / lambda).abs() < 0.05);
    }

    #[test]
    fn test_pdf_at_zero() {
        let dist = Exponential::new(2.0_f64).unwrap();
        assert!((dist.pdf(0.0).unwrap() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_pdf_negative() {
        let dist = Exponential::new(1.0_f64).unwrap();
        assert!((dist.pdf(-1.0).unwrap()).abs() < 1e-12);
    }

    #[test]
    fn test_cdf_at_zero() {
        let dist = Exponential::new(1.0_f64).unwrap();
        assert!((dist.cdf(0.0).unwrap()).abs() < 1e-12);
    }

    #[test]
    fn test_cdf_at_one() {
        let dist = Exponential::new(1.0_f64).unwrap();
        let expected = 1.0 - (-1.0_f64).exp();
        assert!((dist.cdf(1.0).unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn test_quantile_roundtrip() {
        let dist = Exponential::new(1.5_f64).unwrap();
        let p = 0.75;
        let x = dist.quantile(p).unwrap();
        let p_back = dist.cdf(x).unwrap();
        assert!((p - p_back).abs() < 1e-12);
    }

    #[test]
    fn test_quantile_domain_error() {
        let dist = Exponential::new(1.0_f64).unwrap();
        let error = dist.quantile(1.5).unwrap_err();
        assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::DomainError);
    }

    #[test]
    fn test_quantile_non_finite_probability() {
        let dist = Exponential::new(1.0_f64).unwrap();
        let error = dist.quantile(f64::NEG_INFINITY).unwrap_err();
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
        let dist = Exponential::new(4.0_f64).unwrap();
        assert!((dist.mean().unwrap() - 0.25).abs() < 1e-12);
        assert!((dist.variance().unwrap() - 0.0625).abs() < 1e-12);
    }

    #[test]
    fn stable_small_probability_and_argument_tails_do_not_cancel() {
        let dist = Exponential::new(1.0_f64).unwrap();
        let minimum = f64::from_bits(1);
        assert_eq!(dist.cdf(minimum).unwrap(), minimum);
        assert_eq!(dist.quantile(minimum).unwrap(), minimum);

        let probability = 1.0e-20_f64;
        assert_eq!(dist.cdf(probability).unwrap(), -(-probability).exp_m1());
        assert_eq!(dist.quantile(probability).unwrap(), -(-probability).ln_1p());
    }

    #[test]
    fn quantile_endpoints_are_exact_support_limits() {
        let dist = Exponential::new(3.0_f64).unwrap();
        assert_eq!(dist.quantile(0.0).unwrap(), 0.0);
        assert_eq!(dist.quantile(1.0).unwrap(), f64::INFINITY);
    }

    #[test]
    fn non_finite_queries_and_unrepresentable_results_are_typed() {
        let dist = Exponential::new(1.0_f64).unwrap();
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let error = dist.pdf(value).unwrap_err();
            assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::NonFinite);
            let error = dist.cdf(value).unwrap_err();
            assert_eq!(error.kind(), scioxide_core::SciOxideErrorKind::NonFinite);
        }
        let tiny_rate = Exponential::new(f64::from_bits(1)).unwrap();
        let error = tiny_rate.mean().unwrap_err();
        assert!(matches!(
            error.view(),
            scioxide_core::SciOxideErrorView::NumericalInstability {
                issue: scioxide_core::NumericalIssue::Overflow,
                ..
            }
        ));
    }
}

/// Exponential power lifetime distribution with shape `b`.
///
/// ```text
/// S(x) = exp(1 − exp(x^b)),   x ≥ 0, b > 0.
/// ```
///
/// This is a lifetime law, not the generalized normal distribution
/// ([`GeneralizedNormal`](crate::GeneralizedNormal)). It has unit scale. Scale
/// it with [`Affine`](crate::Affine). Moments are computed by quadrature to the
/// tolerance in [`LawOptions`].
///
/// # Examples
///
/// ```
/// use scioxide_random::{Density, Distribution, ExponentialPower, Mean, Quantile};
/// use scioxide_random::{SeedableRng, SplitMix64, Support, Survival};
/// let law = ExponentialPower::new(1.0_f64)?;
/// assert_eq!(law.pdf(0.0)?, 1.0);
/// assert!((law.sf(1.0)? - (1.0 - core::f64::consts::E).exp()).abs() < 1e-15);
/// assert!((law.sf(law.quantile(0.3)?)? - 0.7).abs() < 1e-14);
/// assert!((law.mean()? - 0.596_347_362_323_194).abs() < 1e-7); // e E₁(1)
/// let mut rng = SplitMix64::from_seed(11);
/// assert!(law.contains(law.sample(&mut rng)?)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ExponentialPower<S: RealScalar> {
    options: LawOptions<S>,
    shape: S,
}
impl<S: RealScalar> ExponentialPower<S> {
    /// Creates an exponential power distribution with shape `shape`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `shape` is NaN or infinite.
    /// - `DomainError` if `shape ≤ 0`.
    pub fn new(shape: S) -> Result<Self, SciOxideError> {
        Self::with_options(shape, LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside `[64ε, 0.1)`.
    pub fn with_options(shape: S, options: LawOptions<S>) -> Result<Self, SciOxideError> {
        options.validate()?;
        positive(shape, scioxide_core::argument!("shape"))?;
        Ok(Self { options, shape })
    }
}
impl<S: RealScalar> Formula<S> for ExponentialPower<S> {
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
            // At b = 1 the density is exp(x + 1 - e^x), equal to one at zero.
            return Ok(if self.shape < S::ONE {
                S::INFINITY
            } else if self.shape == S::ONE {
                S::ZERO
            } else {
                S::NEG_INFINITY
            });
        }
        let y = x.powf(self.shape);
        Ok(S::ONE + self.shape.ln() + (self.shape - S::ONE) * x.ln() + y - y.exp())
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        if upper {
            Ok(if x <= S::ZERO {
                S::ZERO
            } else {
                -x.powf(self.shape).exp_m1()
            })
        } else {
            Ok(if x <= S::ZERO {
                S::NEG_INFINITY
            } else {
                // S(x) = exp(1 - e^{x^b}) = exp(-expm1(x^b)); `expm1` keeps the
                // exponent exact for small x where e^{x^b} is close to 1.
                log1mexp(-x.powf(self.shape).exp_m1())?
            })
        }
    }
    fn inverse(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ONE {
            return Ok(S::INFINITY);
        }
        finite((-(-p).ln_1p()).ln_1p().powf(S::ONE / self.shape))
    }
    fn inverse_upper(&self, p: S) -> Result<S, SciOxideError> {
        if p == S::ZERO {
            return Ok(S::INFINITY);
        }
        finite((-p.ln()).ln_1p().powf(S::ONE / self.shape))
    }
}
formula_law!(ExponentialPower);
impl<S: RealScalar> Mean<S> for ExponentialPower<S> {
    fn mean(&self) -> Result<S, SciOxideError> {
        self.mean_work(&mut self.options.work_budget().get())
    }

    fn mean_scoped(
        &self,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            self.mean_work(left)
        })
    }
}
impl<S: RealScalar> ExponentialPower<S> {
    fn mean_work(&self, left: &mut usize) -> Result<S, SciOxideError> {
        let options = self.options;
        quantile_mean(self, options, left)
    }
}

impl<S: RealScalar> Variance<S> for ExponentialPower<S> {
    fn variance(&self) -> Result<S, SciOxideError> {
        self.variance_work(&mut self.options.work_budget().get())
    }

    fn variance_scoped(
        &self,
        context: &mut scioxide_core::SolveContext<'_, '_>,
    ) -> Result<S, SciOxideError> {
        crate::law::with_work(context, self.options.work_budget(), |left| {
            self.variance_work(left)
        })
    }
}
impl<S: RealScalar> ExponentialPower<S> {
    fn variance_work(&self, left: &mut usize) -> Result<S, SciOxideError> {
        let options = self.options;
        quantile_variance(self, options, left)
    }
}

/// Generalized exponential distribution with hazard rate
/// `h(x) = a + b(1 − e^(−cx))`.
///
/// ```text
/// S(x) = exp(−(a + b)x + b(1 − e^(−cx))/c),   x ≥ 0, a, b, c > 0.
/// ```
///
/// Quantiles are computed by numerical inversion of the CDF.
///
/// # Examples
///
/// ```
/// use scioxide_random::{Cdf, Distribution, GeneralizedExponential, Quantile};
/// use scioxide_random::{SeedableRng, SplitMix64, Support, Survival};
/// let law = GeneralizedExponential::new(1.0_f64, 2.0, 3.0)?;
/// let exact = (-3.0 + 2.0 * (1.0 - (-3.0_f64).exp()) / 3.0).exp();
/// assert!((law.sf(1.0)? - exact).abs() < 1e-15);
/// assert!((law.cdf(law.quantile(0.3)?)? - 0.3).abs() < 1e-10);
/// let mut rng = SplitMix64::from_seed(12);
/// assert!(law.contains(law.sample(&mut rng)?)?);
/// # Ok::<(), scioxide_core::SciOxideError>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GeneralizedExponential<S: RealScalar> {
    options: crate::LawOptions<S>,
    a: S,
    b: S,
    c: S,
}
impl<S: RealScalar> GeneralizedExponential<S> {
    /// Creates a generalized exponential distribution with parameters `a`,
    /// `b` and `c`.
    ///
    /// It uses the default [`LawOptions`](crate::LawOptions); [`Self::with_options`] sets the
    /// tolerance and work limits.
    ///
    /// # Errors
    ///
    /// - `NonFinite` if `a`, `b` or `c` is NaN or infinite.
    /// - `DomainError` if `a ≤ 0`, `b ≤ 0` or `c ≤ 0`.
    pub fn new(a: S, b: S, c: S) -> Result<Self, SciOxideError> {
        Self::with_options(a, b, c, crate::LawOptions::default())
    }

    /// Creates the law of [`Self::new`] with the tolerance and work limits of `options`.
    ///
    /// # Errors
    ///
    /// - The errors of [`Self::new`].
    /// - `InvalidInput` if the tolerance in `options` is outside `[64ε, 0.1)`.
    pub fn with_options(
        a: S,
        b: S,
        c: S,
        options: crate::LawOptions<S>,
    ) -> Result<Self, SciOxideError> {
        options.validate()?;
        positive(a, scioxide_core::argument!("a"))?;
        positive(b, scioxide_core::argument!("b"))?;
        positive(c, scioxide_core::argument!("c"))?;
        Ok(Self { options, a, b, c })
    }
    fn hazard_integral(&self, x: S) -> Result<S, SciOxideError> {
        if x == S::ZERO {
            return Ok(S::ZERO);
        }
        let x = x.into_f64();
        let c = self.c.into_f64();
        let y = c * x;
        // Integral of 1-exp(-ct): x * (exp(-y)-1+y)/y. The small-y
        // branch retains its logarithm even when c*x itself underflows.
        let log_ratio = if y < f64::EPSILON {
            c.ln() + x.ln() - core::f64::consts::LN_2
        } else if y <= 0.01 {
            super::special::expm1_minus(-y)?.ln() - y.ln()
        } else {
            (1.0 + (-y).exp_m1() / y).ln()
        };
        let log = crate::law::logadd(
            self.a.into_f64().ln() + x.ln(),
            self.b.into_f64().ln() + x.ln() + log_ratio,
        )?;
        Ok(crate::law::constant::<S>(log)?.exp())
    }
}
impl<S: RealScalar> Formula<S> for GeneralizedExponential<S> {
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
        (Some(S::ZERO), None)
    }
    fn density_log(&self, x: S) -> Result<S, SciOxideError> {
        if x < S::ZERO {
            return Ok(S::NEG_INFINITY);
        }
        let log_hazard = if x == S::ZERO {
            self.a.into_f64().ln()
        } else {
            let y = self.c.into_f64() * x.into_f64();
            let log_increment = if y < f64::EPSILON {
                self.c.into_f64().ln() + x.into_f64().ln()
            } else {
                log1mexp(-y)?
            };
            crate::law::logadd(
                self.a.into_f64().ln(),
                self.b.into_f64().ln() + log_increment,
            )?
        };
        Ok(crate::law::constant::<S>(log_hazard)? - self.hazard_integral(x)?)
    }
    fn tail_with_work(&self, x: S, upper: bool, _left: &mut usize) -> Result<S, SciOxideError> {
        if upper {
            Ok(if x <= S::ZERO {
                S::ZERO
            } else {
                -self.hazard_integral(x)?
            })
        } else {
            Ok(if x <= S::ZERO {
                S::NEG_INFINITY
            } else {
                log1mexp(-self.hazard_integral(x)?)?
            })
        }
    }

    const INVERSE: Inverse = Inverse::Smooth;
}
formula_law!(GeneralizedExponential);

impl<S: RealScalar> crate::LogCoordinateLaw<S> for Exponential<S> {
    fn cdf_at_log(&self, y: S) -> Result<S, SciOxideError> {
        crate::law::argument(y, "y")?;
        Ok(-(-(y + self.lambda.ln()).exp()).exp_m1())
    }
    fn sf_at_log(&self, y: S) -> Result<S, SciOxideError> {
        crate::law::argument(y, "y")?;
        Ok((-(y + self.lambda.ln()).exp()).exp())
    }
    fn pushforward_pdf(&self, y: S) -> Result<S, SciOxideError> {
        crate::law::argument(y, "y")?;
        let log_rate = y + self.lambda.ln();
        if log_rate.exp() == S::INFINITY {
            return Ok(S::ZERO);
        }
        finite((log_rate - log_rate.exp()).exp())
    }
    fn log_cdf_at_log(&self, y: S) -> Result<S, SciOxideError> {
        crate::law::argument(y, "y")?;
        let log_rate = y + self.lambda.ln();
        if log_rate < S::EPSILON.ln() {
            return crate::law::finite(log_rate);
        }
        log1mexp(-log_rate.exp())
    }
    fn log_sf_at_log(&self, y: S) -> Result<S, SciOxideError> {
        crate::law::argument(y, "y")?;
        finite(-(y + self.lambda.ln()).exp())
    }
    fn log_pushforward_pdf(&self, y: S) -> Result<S, SciOxideError> {
        crate::law::argument(y, "y")?;
        let log_rate = y + self.lambda.ln();
        finite(log_rate - log_rate.exp())
    }
}
